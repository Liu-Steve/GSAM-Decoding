import numpy as np
import torch
from itertools import count

from transformers.generation.utils import _crop_past_key_values

from model.shotgun.lru_cache import ShotgunCache, Tokens

type DraftNode = tuple[Tokens, list[DraftNode]]


def recursive_get_draft_tokens(
        prefix_ids: list[int],
        draft_node: DraftNode,
        step_until_leaf: int,
        shotgun_cache: ShotgunCache,
        max_total_drafts_len: int,
) -> tuple[int, bool, bool]:
    draft, children = draft_node

    # Extend the prefix with the draft tokens in the current node.
    prefix_ids.extend(draft)

    # Keep track of the remaining number of draft tokens that can be added.
    remain_drafts_len = max_total_drafts_len

    # If the remaining number of draft tokens has been exhausted.
    full = False

    # If any child node has been grown.
    grown = False

    # If the current node is a leaf node, get the draft tokens for the children.
    if step_until_leaf == 0:
        child_drafts, child_drafts_lens = shotgun_cache.get_draft_tokens(
            prefix_ids)

        # As long as the remaining number of draft tokens allows, add the child
        # draft tokens to the tree.
        for child_draft, child_draft_len in zip(child_drafts, child_drafts_lens):
            if remain_drafts_len >= child_draft_len:
                draft_node[1].append(child_draft)
                remain_drafts_len -= child_draft_len
                grown = True
            else:
                full = True
                break

    # If the current node is not a leaf node, step down to the children.
    else:
        for child in children:
            remain_drafts_len, full, child_grown = recursive_get_draft_tokens(
                prefix_ids,
                child,
                step_until_leaf - 1,
                shotgun_cache,
                remain_drafts_len
            )

            grown = grown or child_grown

            if full:
                break

    # Restore the prefix.
    for _ in range(len(draft)):
        prefix_ids.pop()

    return remain_drafts_len, full, grown


def get_chained_draft_tokens(
        prefix_ids: list[int],
        shotgun_cache: ShotgunCache,
        max_total_drafts_len: int,
        chaining: bool,
        chaining_reserve_len: int,
) -> tuple[DraftNode, int]:
    drafts_root = ((), [])
    remain_drafts_len = max_total_drafts_len - chaining_reserve_len
    
    for step_until_leaf in count():
        remain_drafts_len, full, grown = recursive_get_draft_tokens(
            prefix_ids,
            drafts_root,
            step_until_leaf,
            shotgun_cache,
            remain_drafts_len
        )

        # If chaining is disabled, run only the first iteration.
        if not chaining:
            break

        # If we reserved some tokens for chaining, add them back since
        # starting from the second iteration we will grow the tree using
        # chaining.
        if chaining_reserve_len > 0:
            remain_drafts_len += chaining_reserve_len
            chaining_reserve_len = 0
            continue

        if full or not grown:
            break

    sum_drafts_len = max_total_drafts_len - remain_drafts_len
    return drafts_root, sum_drafts_len


def make_chained_4d_attention_mask(
        cached_prefix_len: int,
        uncached_prefix_len: int,
        drafts_root,
        sum_drafts_len: int,
        dtype: np.dtype = np.float16,
):
    num_rows = uncached_prefix_len + sum_drafts_len
    num_cols = num_rows + cached_prefix_len
    prefix_len = cached_prefix_len + uncached_prefix_len

    mask = np.empty((num_rows, num_cols), dtype=dtype)
    mask[:, :prefix_len] = 1.0
    mask[:, prefix_len:] = 0.0

    for i in range(uncached_prefix_len):
        mask[i, cached_prefix_len+i+1:prefix_len] = 0.0

    row_offset = uncached_prefix_len
    col_offset = prefix_len

    working_row = np.empty((num_cols,), dtype=dtype)
    working_row[:prefix_len] = 1.0
    working_row[prefix_len:] = 0.0

    def recursive_mask_fill(prefix_tree_node):
        nonlocal row_offset
        nonlocal col_offset

        draft, children = prefix_tree_node

        prev_col_offset = col_offset

        for _ in range(len(draft)):
            working_row[col_offset] = 1.0
            mask[row_offset, :] = working_row
            row_offset += 1
            col_offset += 1

        for child in children:
            recursive_mask_fill(child)

        working_row[prev_col_offset:prev_col_offset+len(draft)] = 0.0

    recursive_mask_fill(drafts_root)

    return mask


def make_chained_input_ids(
        uncached_prefix_ids: np.ndarray,
        drafts_root,
        sum_drafts_len: int,
):
    uncached_prefix_len = uncached_prefix_ids.shape[0]
    input_ids = np.empty((uncached_prefix_len+sum_drafts_len,), dtype=np.int64)
    input_ids[:uncached_prefix_len] = uncached_prefix_ids

    offset = uncached_prefix_len

    def recursive_input_ids_fill(prefix_tree_node):
        nonlocal offset

        draft, children = prefix_tree_node
        input_ids[offset:offset+len(draft)] = draft
        offset += len(draft)

        for child in children:
            recursive_input_ids_fill(child)

    recursive_input_ids_fill(drafts_root)

    return input_ids


def make_chained_position_ids(
        cached_prefix_len: int,
        uncached_prefix_len: int,
        drafts_root,
        sum_drafts_len: int,
):
    prefix_len = cached_prefix_len + uncached_prefix_len

    position_ids = np.empty(
        (uncached_prefix_len+sum_drafts_len,), dtype=np.int64)
    position_ids[:uncached_prefix_len] = np.arange(
        cached_prefix_len, prefix_len)

    offset = uncached_prefix_len
    position_offset = prefix_len

    def recursive_position_fill(prefix_tree_node):
        nonlocal offset
        nonlocal position_offset

        draft, children = prefix_tree_node

        position_ids[offset:offset+len(draft)] = np.arange(
            position_offset, position_offset+len(draft))
        offset += len(draft)
        position_offset += len(draft)

        for child in children:
            recursive_position_fill(child)

        position_offset -= len(draft)

    recursive_position_fill(drafts_root)

    return position_ids


def verify_chained_drafts(
        drafts_root,
        sampled_ids,
        next_tok,
):
    max_accept_num = 1
    max_accepted_ids = [next_tok]
    working_accepted_ids = [next_tok]
    sample_offset = 0

    def recursive_verify(prefix_tree_node, next_tok, mismatched):
        nonlocal max_accept_num
        nonlocal max_accepted_ids
        nonlocal working_accepted_ids
        nonlocal sample_offset

        draft, children = prefix_tree_node
        sample = sampled_ids[sample_offset:sample_offset+len(draft)]
        sample_offset += len(draft)
        matching_num = 0

        if not mismatched:
            for draft_tok, sample_tok in zip(draft, sample):
                if draft_tok == next_tok:
                    next_tok = sample_tok
                    working_accepted_ids.append(sample_tok)
                    matching_num += 1
                else:
                    mismatched = True
                    break

        if children:
            for child in children:
                recursive_verify(child, next_tok, mismatched)
        else:
            if len(working_accepted_ids) > max_accept_num:
                max_accept_num = len(working_accepted_ids)
                max_accepted_ids = working_accepted_ids.copy()

        for _ in range(matching_num):
            working_accepted_ids.pop()

    recursive_verify(drafts_root, next_tok, False)

    return max_accepted_ids


@torch.inference_mode()
def shotgun(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_length: int,
    eos_token_id: int,
    shotgun_cache: ShotgunCache,
    max_query_len: int,
    chaining: bool,
    chaining_reserve_len: int,
    **model_kwargs,
):
    device = input_ids.device
    accept_length_list = []
    model_kwargs["past_key_values"] = None
    model_kwargs["use_cache"] = True
    model_kwargs["return_dict"] = True

    uncached_prefix_ids = input_ids.detach().cpu().numpy().squeeze(0)
    uncached_prefix_len = uncached_prefix_ids.shape[0]

    all_tok_ids = uncached_prefix_ids.tolist()
    prefix_len = len(all_tok_ids)
    shotgun_cache.update_cache(all_tok_ids)

    for step in count():
        # Query the cache table to get the draft tokens.
        drafts_root, sum_drafts_len = get_chained_draft_tokens(
            prefix_ids=all_tok_ids,
            shotgun_cache=shotgun_cache,
            max_total_drafts_len=max(0, max_query_len-uncached_prefix_len),
            chaining=chaining,
            chaining_reserve_len=chaining_reserve_len,
        )

        # Store the prefix and draft tokens into a single sequence.
        combined_ids = make_chained_input_ids(
            uncached_prefix_ids=uncached_prefix_ids,
            drafts_root=drafts_root,
            sum_drafts_len=sum_drafts_len,
        )
        combined_ids = torch.from_numpy(combined_ids).unsqueeze(0).to(device)
        model_kwargs["input_ids"] = combined_ids

        # Create the position ids.
        combined_poss = make_chained_position_ids(
            cached_prefix_len=prefix_len-uncached_prefix_len,
            uncached_prefix_len=uncached_prefix_len,
            drafts_root=drafts_root,
            sum_drafts_len=sum_drafts_len,
        )
        combined_poss = torch.from_numpy(combined_poss).unsqueeze(0).to(device)
        model_kwargs["position_ids"] = combined_poss

        # Create the attention mask where each draft attends to the prefix and
        # to earlier positions within its own draft but not to other drafts.
        mask = make_chained_4d_attention_mask(
            cached_prefix_len=prefix_len-uncached_prefix_len,
            uncached_prefix_len=uncached_prefix_len,
            drafts_root=drafts_root,
            sum_drafts_len=sum_drafts_len,
        )
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        model_kwargs["attention_mask"] = mask

        # Invoke the model to get the logits.
        model_outputs = model(**model_kwargs)
        logits = model_outputs.logits[:, -sum_drafts_len-1:]

        # Sample the next token.
        # `next_id` is the ground truth token following the previously uncached prefix.
        # `sampled_ids` stores the sampled tokens from the speculation drafts.
        sampled_ids = logits.argmax(dim=-1).detach().cpu().numpy().squeeze(0)
        next_id = sampled_ids[0]
        sampled_ids = sampled_ids[1:]

        # Find the speculation draft with the most matching tokens.
        accepted_ids = verify_chained_drafts(
            drafts_root=drafts_root,
            sampled_ids=sampled_ids,
            next_tok=next_id,
        )

        # Store the accepted tokens
        all_tok_ids.extend(accepted_ids)

        # Crop the KV cache. Keep only the tokens that were in the prefix.
        model_kwargs["past_key_values"] = _crop_past_key_values(
            model,
            model_outputs.past_key_values,
            prefix_len,
        )

        # Update the length of the accepted sequence.
        uncached_prefix_len = len(accepted_ids)
        prefix_len = len(all_tok_ids)
        accept_length_list.append(uncached_prefix_len)

        # Prepare for the next iteration.
        uncached_prefix_ids = np.array(accepted_ids)

        # Update shotgun cache.
        cache_update_offset = uncached_prefix_len - 1
        shotgun_cache.update_cache(
            all_tok_ids[-shotgun_cache.max_leader_follower_len-cache_update_offset:])

        # Check termination conditions.
        if (uncached_prefix_ids == eos_token_id).any() or (next_id == eos_token_id):
            break
        if len(all_tok_ids) >= max_length:
            break

    final_ids = torch.tensor(
        all_tok_ids[:max_length], dtype=torch.long, device=device).unsqueeze(0)
    return final_ids, step, accept_length_list
