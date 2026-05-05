import torch

from model.samd.samd_config import ForwardType
from model.samd.samd_model import SamdModel as BaseSamdModel
from model.samd.utils import CandidateType, OptionalTensor, eval_posterior, gen_candidates


class SamdModel(BaseSamdModel):
    def decode(self, sample_p: torch.Tensor, length: int):
        candidates = gen_candidates(
            sample_p,
            self.base_tree_retrieve_indices,
            self.draft,
            self.samd_config,
            self.gen_config,
            self.device,
        )
        self.update_buffers(candidates.buffers_kwargs)
        if candidates.type == CandidateType.sequence:
            self.forward_state.forward_type = ForwardType.seq_decode
            position_ids = self.seq_position_ids + length
            attention_mask = None
        else:
            self.forward_state.forward_type = ForwardType.tree_decode
            position_ids = self.tree_position_ids + length
            attention_mask = self._tree_decode_attention_mask(candidates.tokens, length)

        input_ids = candidates.tokens
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
        )
        tree_logits = outputs.logits
        if self.samd_config.use_last_hidden_states:
            tree_last_hidden_states = OptionalTensor(outputs.last_hidden_states)
        else:
            tree_last_hidden_states = OptionalTensor(None)
        if candidates.type == CandidateType.sequence:
            candidate_logits = tree_logits
            candidate_last_hidden_states = tree_last_hidden_states
            candidate_indices = OptionalTensor(None)
        else:
            candidate_logits = tree_logits.squeeze(0)[self.tree_retrieve_indices]
            candidate_last_hidden_states = tree_last_hidden_states.apply(
                lambda x: x.squeeze(0)[self.tree_retrieve_indices]
            )
            candidate_indices = OptionalTensor(self.tree_retrieve_indices)

        best_candidate, accept_length, sample_p = eval_posterior(
            candidate_logits,
            candidates.candidate_tokens,
            self.gen_config,
        )
        new_tokens = self.update_state(
            input_ids.squeeze(0),
            tree_logits.squeeze(0),
            best_candidate,
            accept_length,
            candidates.candidate_tokens,
            candidate_indices,
            candidate_last_hidden_states,
        )
        return sample_p, new_tokens

    def _tree_decode_attention_mask(self, input_ids: torch.Tensor, length: int):
        batch_size = input_ids.shape[0]
        query_length = input_ids.shape[-1]
        key_value_length = length + query_length
        attention_mask = torch.zeros(
            (batch_size, 1, query_length, key_value_length),
            dtype=torch.float32,
            device=input_ids.device,
        )
        for i in range(query_length):
            attention_mask[:, :, i, : length + i + 1] = 1.0
        return attention_mask
