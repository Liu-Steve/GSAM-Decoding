import torch
from typing import Optional

from model.samd.draft import CandidateType
from model.samd.tree_model import TreeModel, tree_model_cls
from transformers import LlamaForCausalLM

from .sam import DynSAM, StaticSAM, NullStaticSAM
from .samd_config import SamdConfig


class DraftModel(torch.nn.Module):
    def __init__(
        self,
        config: SamdConfig,
        sam_dyn: DynSAM = None,
        sam_static: StaticSAM = None,
        tree_model: TreeModel = None,
        lm: LlamaForCausalLM = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        static_use_gsam = getattr(sam_static, "use_gsam", config.use_gsam)
        static_use_small_dict = getattr(sam_static, "use_small_dict", config.use_small_dict)
        print(
            "GSAMD config: use_gsam={}, use_small_dict={}".format(
                static_use_gsam,
                static_use_small_dict,
            )
        )
        self.config = config
        self.sam_dyn = sam_dyn if sam_dyn is not None else DynSAM(
            config.n_predicts,
            device=device,
            use_gsam=static_use_gsam,
            use_small_dict=static_use_small_dict,
        )
        self.sam_static = sam_static if sam_static is not None else NullStaticSAM(
            config.n_predicts,
            device=device,
            use_gsam=static_use_gsam,
            use_small_dict=static_use_small_dict,
        )
        if config.tree_method is not None:
            tree_cls = tree_model_cls[config.tree_method]
            self.tree_model = tree_model if tree_model is not None else tree_cls(config, lm, dtype, device)
        else:
            self.tree_model = None

        self.sam_dyn.n_predicts = config.n_predicts
        self.sam_static.n_predicts = config.n_predicts
        self.len_bias = config.len_bias
        self.len_threshold = config.len_threshold

    def reset(self):
        self.sam_dyn.reset()
        self.sam_static.reset()
        if self.tree_model is not None:
            self.tree_model.reset()

    def lookup(self, start_token: int):
        if self.tree_model is not None:
            index_dyn, match_dyn = self.sam_dyn.lookup(start_token)
            index_static, match_static = self.sam_static.lookup(start_token)
            match_static -= self.len_bias
            if max(match_dyn, match_static) >= self.len_threshold or self.tree_model is None:
                if match_dyn >= match_static:
                    seq = self.sam_dyn.gen_draft(index_dyn, start_token)
                else:
                    seq = self.sam_static.gen_draft(index_static, start_token)
                return (CandidateType.sequence, seq, {})
            return (CandidateType.tree,) + self.tree_model.gen_draft(start_token)

        index_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        index_static, match_static = self.sam_static.lookup(start_token)
        if match_dyn >= match_static:
            seq, buffers_kwargs = self.sam_dyn.gen_dyn_draft(index_dyn, match_dyn, start_token)
            return (CandidateType.sequence, seq, buffers_kwargs)
        tree, buffers_kwargs = self.sam_static.gen_dyn_draft(index_static, match_static, start_token)
        return (CandidateType.tree, tree, buffers_kwargs)

    def update(
        self,
        tokens: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
        tree_tokens: Optional[torch.Tensor] = None,
        tree_logits: Optional[torch.Tensor] = None,
    ):
        tokens_list = tokens.tolist()
        self.sam_dyn.add_tokens(tokens_list)
        self.sam_static.transfer_tokens(tokens_list)
        if self.tree_model is not None:
            self.tree_model.update(
                tokens=tokens,
                last_hidden_states=last_hidden_states,
                tree_tokens=tree_tokens,
                tree_logits=tree_logits,
            )
