import numpy as np
import torch
from itertools import count

from transformers.generation.utils import _crop_past_key_values

from model.shotgun.lru_cache import ShotgunCache


def get_draft_tokens(input_ids: np.ndarray, shotgun_cache: ShotgunCache):
    key = input_ids[-shotgun_cache.max_prefix_len:]
    drafts = shotgun_cache.get_draft_tokens(key)

    if not drafts:
        return np.array([[]], dtype=input_ids.dtype)

    num_drafts = len(drafts)
    draft_len = max(len(draft) for draft in drafts)

    draft_matrix = np.zeros((num_drafts, draft_len), dtype=input_ids.dtype)
    for i, draft in enumerate(drafts):
        draft_matrix[i, :len(draft)] = draft
    return draft_matrix


def make_4d_attention_mask(
        cached_prefix_len: int,
        uncached_prefix_len: int,
        draft_lens: list[int],
        dtype: np.dtype = np.float16,
):
    """
    Build a (1, 1, query_len, total_len) causal mask where:
    - query_len = uncached_prefix_len + sum(draft_lens)
    - total_len = cached_prefix_len + query_len
    - rows 0..uncached_prefix_len-1 are standard causal attention
    - rows belonging to draft tokens can attend to all prefix positions AND
      to earlier positions within their own draft, but not to other drafts.

    Args:
    - `cached_prefix_len`: `int`, length of the KV-cached prefix
    - `uncached_prefix_len`: `int`, length of the uncached prefix
    - `draft_lens`: `list[int]`, lengths of each draft token sequence
    - `dtype`: `np.dtype`, desired dtype of the output mask

    Returns:
    - `mask`: (1, 1, query_len, total_len) NumPy array of 1.0 and 0.0

    Example:

    For cached_prefix_len=2, uncached_prefix_len=3, draft_lens=[2,3], the resulting mask will have
    shape (1, 1, 8, 10) with values:
    ```
    [[[[ 1.    1.    1.    .     .     .     .     .     .     .]     # uncached prefix row 0 
       [ 1.    1.    1.    1.    .     .     .     .     .     .]     # uncached prefix row 1
       [ 1.    1.    1.    1.    1.    .     .     .     .     .]     # uncached prefix row 2
       [ 1.    1.    1.    1.    1.    1.    .     .     .     .]     # draft 1 row 0
       [ 1.    1.    1.    1.    1.    1.    1.    .     .     .]     # draft 1 row 1  
       [ 1.    1.    1.    1.    1.    .     .     1.    .     .]     # draft 2 row 0
       [ 1.    1.    1.    1.    1.    .     .     1.    1.    .]     # draft 2 row 1
       [ 1.    1.    1.    1.    1.    .     .     1.    1.    1.]]]] # draft 2 row 2
       | cached   |    uncached     |  draft 1   |    draft 2     |
       | prefix   |     prefix      |            |                |
    ```

    Reference:
    https://github.com/huggingface/transformers/blob/e94a480/src/transformers/models/llama/modeling_llama.py#L664
    """
    sum_draft_len = sum(draft_lens)
    query_len = uncached_prefix_len + sum_draft_len
    total_len = cached_prefix_len + query_len

    # Make row and col index grids of shape (query_len, query_len).
    seqs = np.arange(query_len)  # shape (query_len,)
    rows = seqs[:, None] # shape (query_len, 1)
    cols = seqs[None, :] # shape (1, query_len)

    # Compute the *end* boundaries of each segment: [P, P+D1, P+D1+D2, ...].
    query_lens = [uncached_prefix_len] + draft_lens
    ends = np.cumsum(query_lens)

    # segment_id[i] = which chunk row i belongs to:
    # prefix rows [0..P-1] -> 0
    # draft k rows [ ends[k-1] .. ends[k]-1 ] -> k
    segment_id = np.repeat(np.arange(len(ends)), query_lens)
    segment_id = segment_id[:, None]  # shape (query_len, 1)

    # Compute the *start* of each segment similarly:
    # starts = [0, P, P+D1, ...]
    starts = np.concatenate(([0], ends[:-1]))
    seg_start = starts[segment_id]    # shape (query_len, 1)

    # Draft-region mask:
    #   allow all cols < prefix_len,
    #   plus cols in [ seg_start .. i ] (i.e. its own draft-causal)
    draft_mask = (cols <= rows) & ((cols < uncached_prefix_len) | (cols >= seg_start))

    # Full attention mask:
    mask = np.empty((query_len, total_len), dtype=dtype)
    mask[:, :cached_prefix_len] = 1.0
    mask[:, cached_prefix_len:] = draft_mask.astype(dtype)

    return mask.astype(dtype).reshape((1, 1, query_len, total_len))


@torch.no_grad()
def shotgun(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_length: int,
    eos_token_id: int,
    shotgun_cache: ShotgunCache,
    **model_kwargs,
):
    device = input_ids.device
    accept_length_list = []
    model_kwargs["past_key_values"] = None
    model_kwargs["use_cache"] = True

    prefix_ids = input_ids.detach().cpu().numpy().squeeze(0)
    prefix_len = prefix_ids.shape[0]
    uncached_prefix_len = prefix_len
    prefix_poss = np.arange(prefix_len)
    shotgun_cache.update_cache(prefix_ids)

    for step in count():
        # Query the cache table to get the draft tokens.
        drafts = get_draft_tokens(prefix_ids, shotgun_cache)
        num_drafts = drafts.shape[0]
        draft_len = drafts.shape[1]
        sum_draft_len = num_drafts * draft_len

        # Store the prefix and draft tokens into a single sequence.
        combined_ids = np.concatenate((prefix_ids, drafts.ravel()))
        combined_ids = torch.from_numpy(combined_ids).unsqueeze(0).to(device)
        model_kwargs["input_ids"] = combined_ids

        # Create the position ids.
        draft_poss = np.arange(prefix_len, prefix_len + draft_len)
        combined_poss = np.concatenate([prefix_poss] + [draft_poss] * num_drafts)
        combined_poss = torch.from_numpy(combined_poss).to(device).unsqueeze(0)
        model_kwargs["position_ids"] = combined_poss

        # Create the attention mask where each draft attends to the prefix and
        # to earlier positions within its own draft but not to other drafts.
        mask = make_4d_attention_mask(prefix_len - uncached_prefix_len, uncached_prefix_len, [draft_len] * num_drafts)
        mask = torch.from_numpy(mask).to(device)
        model_kwargs["attention_mask"] = mask

        # Invoke the model to get the logits.
        model_inputs = model.prepare_inputs_for_generation(**model_kwargs)
        model_outputs = model(**model_inputs, return_dict=True)
        logits = model_outputs.logits[:, -sum_draft_len-1:]

        # Sample the next token.
        # next_tok_id is the ground truth token to be added to the prefix.
        # sampled_tok_ids is reshaped to be a 2D array of shape (num_drafts, draft_len)
        # for further verification.
        sampled_tok_ids = logits.argmax(dim=-1).detach().cpu().numpy().squeeze(0)
        next_tok_id = sampled_tok_ids[0]
        sampled_tok_ids = sampled_tok_ids[1:].reshape((num_drafts, draft_len))
        
        # Calculate the number of matching tokens between the sampled tokens and the speculation drafts.
        mismatch = np.empty_like(drafts, dtype=bool)
        mismatch[:, :1] = drafts[:, :1] != next_tok_id
        mismatch[:, 1:] = drafts[:, 1:] != sampled_tok_ids[:, :-1]
        accept_nums = (mismatch.cumsum(axis=1) < 1).sum(axis=1)

        # Take the speculation draft with the most matching tokens.
        max_accept_idx = accept_nums.argmax()
        max_accept_num = accept_nums[max_accept_idx]
        accepted_ids = sampled_tok_ids[max_accept_idx, :max_accept_num]

        # Form the new prefix by concatenating the ground truth token and the accepted speculation draft.
        prefix_ids = np.concatenate((prefix_ids, [next_tok_id], accepted_ids))

        # Crop the KV cache. Keep only the tokens that were in the prefix.
        model_kwargs["past_key_values"] = _crop_past_key_values(
            model,
            model_outputs.past_key_values,
            prefix_len,
        )

        # Update the length of the accepted sequence.
        uncached_prefix_len = prefix_ids.shape[0] - prefix_len
        accept_length_list.append(uncached_prefix_len)
        prefix_len = prefix_ids.shape[0]
        prefix_poss = np.arange(prefix_len - uncached_prefix_len, prefix_len)

        # Update the cache.
        cache_update_offset = uncached_prefix_len - 1
        shotgun_cache.update_cache(prefix_ids[-shotgun_cache.max_prefix_followup_len-cache_update_offset:])

        # Check termination conditions.
        if (accepted_ids == eos_token_id).any() or (next_tok_id == eos_token_id):
            break
        if prefix_ids.shape[0] >= max_length:
            break

    final_ids = torch.from_numpy(prefix_ids[:max_length]).unsqueeze(0).to(device)
    return final_ids, step, accept_length_list
