import numpy as np
import torch
from itertools import count

from transformers.generation.utils import _crop_past_key_values

from model.shotgun.lru_cache import ShotgunCache


def get_draft_tokens(
        prefix_ids: list[int],
        uncached_prefix_len: int,
        shotgun_cache: ShotgunCache
) -> tuple[np.ndarray, list[int]]:
    drafts, drafts_lens = shotgun_cache.get_draft_tokens(prefix_ids)

    sum_drafts_len = sum(drafts_lens)
    drafts_ids = np.empty((uncached_prefix_len+sum_drafts_len,), dtype=np.int64)

    offset = uncached_prefix_len
    for draft, draft_len in zip(drafts, drafts_lens):
        drafts_ids[offset:offset+draft_len] = draft
        offset += draft_len

    return drafts_ids, drafts_lens, sum_drafts_len


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
    model_kwargs["return_dict"] = True

    uncached_prefix_ids = input_ids.detach().cpu().numpy().squeeze(0)
    uncached_prefix_len = uncached_prefix_ids.shape[0]
    uncached_prefix_poss = np.arange(uncached_prefix_len)

    all_tok_ids = uncached_prefix_ids.tolist()
    prefix_len = len(all_tok_ids)
    shotgun_cache.update_cache(all_tok_ids)

    for step in count():
        # Query the cache table to get the draft tokens.
        # The first `uncached_prefix_len` tokens in `drafts_ids` will later
        # be filled by the uncached prefix tokens. Drafts tokens follow them.
        drafts_ids, drafts_lens, sum_draft_len = get_draft_tokens(all_tok_ids, uncached_prefix_len, shotgun_cache)

        # Store the prefix and draft tokens into a single sequence.
        combined_ids = drafts_ids
        combined_ids[:uncached_prefix_len] = uncached_prefix_ids
        combined_ids = torch.from_numpy(combined_ids).to(device).unsqueeze(0)
        model_kwargs["input_ids"] = combined_ids

        # Create the position ids.
        combined_poss = uncached_prefix_poss
        for draft_len in drafts_lens:
            combined_poss = np.concatenate([combined_poss, np.arange(prefix_len, prefix_len + draft_len)])
        combined_poss = torch.from_numpy(combined_poss).to(device).unsqueeze(0)
        model_kwargs["position_ids"] = combined_poss

        # Create the attention mask where each draft attends to the prefix and
        # to earlier positions within its own draft but not to other drafts.
        mask = make_4d_attention_mask(prefix_len - uncached_prefix_len, uncached_prefix_len, drafts_lens)
        mask = torch.from_numpy(mask).to(device)
        model_kwargs["attention_mask"] = mask

        # Invoke the model to get the logits.
        model_outputs = model(**model_kwargs)
        logits = model_outputs.logits[:, -sum_draft_len-1:]

        # Sample the next token.
        # next_tok_id is the ground truth token to be added to the prefix.
        # sampled_ids is reshaped to be a 2D array of shape (num_drafts, draft_len)
        # for further verification.
        sampled_ids = logits.argmax(dim=-1).detach().cpu().numpy().squeeze(0)
        next_id = sampled_ids[0]
        sampled_ids = sampled_ids[1:]

        # Calculate the number of matching tokens between the sampled tokens and the speculation drafts.
        # Find the speculation draft with the most matching tokens.
        max_accept_num = 0
        max_accepted_ids = np.array([], dtype=np.int64)
        draft_offset = uncached_prefix_len
        sample_offset = 0
        for draft_len in drafts_lens:
            draft = drafts_ids[draft_offset:draft_offset+draft_len]
            sampled = sampled_ids[sample_offset:sample_offset+draft_len]
            draft_offset += draft_len
            sample_offset += draft_len

            # Verify the first token in the draft.
            if draft[0] != next_id:
                continue
            accept_num = 1

            # Verify the rest of the draft.
            mismatch = draft[1:] != sampled[:-1]
            accept_num += (mismatch.cumsum(axis=0) < 1).sum(axis=0)

            # Update the longest matching draft.
            if accept_num > max_accept_num:
                max_accept_num = accept_num
                max_accepted_ids = sampled[:max_accept_num]

        # Form the new prefix by concatenating the ground truth token and the accepted speculation draft.
        uncached_prefix_ids = np.concatenate(([next_id], max_accepted_ids))
        all_tok_ids.extend(uncached_prefix_ids)

        # Crop the KV cache. Keep only the tokens that were in the prefix.
        model_kwargs["past_key_values"] = _crop_past_key_values(
            model,
            model_outputs.past_key_values,
            prefix_len,
        )

        # Update the length of the accepted sequence.
        uncached_prefix_len = uncached_prefix_ids.shape[0]
        prefix_len = len(all_tok_ids)
        accept_length_list.append(uncached_prefix_len)
        uncached_prefix_poss = np.arange(prefix_len - uncached_prefix_len, prefix_len)

        # Update the cache.
        cache_update_offset = uncached_prefix_len - 1
        shotgun_cache.update_cache(all_tok_ids[-shotgun_cache.max_prefix_followup_len-cache_update_offset:])

        # Check termination conditions.
        if (max_accepted_ids == eos_token_id).any() or (next_id == eos_token_id):
            break
        if len(all_tok_ids) >= max_length:
            break

    final_ids = torch.tensor(all_tok_ids[:max_length], dtype=torch.long, device=device).unsqueeze(0)
    return final_ids, step, accept_length_list
