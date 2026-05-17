import os
import time
from typing import List

import torch

try:
    from . import _gsamd_core
except ImportError as exc:
    raise ImportError(
        "GSAMD C++ extension is not built. From the repo root, run "
        "`python model/gsamd/setup.py build_ext --inplace`."
    ) from exc


def pad_path(path, length, pad_value=-1):
    return path + [pad_value] * (length - len(path))


class DynSAM:
    def __init__(
        self,
        n_predicts: int = 40,
        alpha: float = 4.0,
        device: str = "cuda",
        use_gsam: bool = True,
        use_small_dict: bool = True,
        map_type: str = "",
        lazy_threshold: int = 1,
    ):
        self.device = device
        self.core = _gsamd_core.DynSAMCore(
            n_predicts,
            alpha,
            use_gsam,
            use_small_dict,
            map_type,
            lazy_threshold,
        )

    @property
    def n_predicts(self):
        return self.core.n_predicts

    @n_predicts.setter
    def n_predicts(self, value):
        self.core.n_predicts = value

    @property
    def use_gsam(self):
        return self.core.use_gsam

    @property
    def use_small_dict(self):
        return self.core.use_small_dict

    @property
    def map_type(self):
        return self.core.map_type

    @property
    def lazy_threshold(self):
        return self.core.lazy_threshold

    def reset(self):
        self.core.reset()

    def add_tokens(self, tokens: List[int]):
        self.core.add_tokens(tokens)

    def transfer_tokens(self, tokens: List[int]):
        self.core.transfer_tokens(tokens)

    def lookup(self, token: int):
        return self.core.lookup(token)

    def gen_draft(self, index: int, start_token: int):
        return self.core.gen_draft(index, start_token)

    def gen_dyn_draft(self, index: int, match_length: int, start_token: int):
        result = self.core.gen_dyn_draft(index, match_length, start_token)
        seq = result.tokens
        seq_position_ids = torch.arange(0, len(seq), dtype=torch.long, device=self.device).unsqueeze(0)
        return seq, {"seq_position_ids": seq_position_ids}

    def state_count(self):
        return self.core.state_count()

    def edge_stats(self):
        return self.core.edge_stats()

    def transition_memory_usage(self):
        return self.core.transition_memory_usage()


class StaticSAM:
    def __init__(
        self,
        n_predicts: int = 40,
        alpha: float = 4.0,
        K: int = 8,
        device: str = "cuda",
        use_gsam: bool = True,
        use_small_dict: bool = True,
        map_type: str = "",
        lazy_threshold: int = 1,
        core=None,
    ):
        self.device = device
        self.K = K
        self.core = core if core is not None else _gsamd_core.StaticSAMCore(
            n_predicts,
            alpha,
            K,
            use_gsam,
            use_small_dict,
            map_type,
            lazy_threshold,
        )

    @staticmethod
    def build(
        batch_tokens: List[List[int]],
        eos_token: int,
        verbose: bool = True,
        n_predicts: int = 40,
        alpha: float = 4.0,
        K: int = 8,
        device: str = "cuda",
        use_gsam: bool = True,
        use_small_dict: bool = True,
        map_type: str = "",
        lazy_threshold: int = 1,
    ):
        sam = StaticSAM(
            n_predicts,
            alpha,
            K,
            device,
            use_gsam,
            use_small_dict,
            map_type,
            lazy_threshold,
        )
        sam.core.add_batch_tokens(batch_tokens, eos_token, True)
        sam.core.init_topk_next()
        stats = sam.core.edge_stats()
        print(
            "single-next states: {} ({:.6f}), total states: {}, map_type: {}, lazy_threshold: {}, transition bytes: {}".format(
                stats.single_next_states,
                stats.ratio,
                stats.total_states,
                sam.map_type,
                sam.lazy_threshold,
                sam.transition_memory_usage(),
            )
        )
        return sam

    @property
    def n_predicts(self):
        return self.core.n_predicts

    @n_predicts.setter
    def n_predicts(self, value):
        self.core.n_predicts = value

    @property
    def use_gsam(self):
        return self.core.use_gsam

    @property
    def use_small_dict(self):
        return self.core.use_small_dict

    @property
    def map_type(self):
        return self.core.map_type

    @property
    def lazy_threshold(self):
        return self.core.lazy_threshold

    def transfer_tokens(self, tokens: List[int]):
        self.core.transfer_tokens(tokens)

    def lookup(self, token: int):
        return self.core.lookup(token)

    def reset(self):
        self.core.reset()

    def gen_draft(self, index: int, start_token: int):
        return self.core.gen_draft(index, start_token)

    def gen_buffers(self, anc_tree: List[int]):
        n = len(anc_tree)
        is_leaf = [True] * n
        tree_position_ids = [0] * n
        for i in range(1, n):
            is_leaf[anc_tree[i]] = False
            tree_position_ids[i] = tree_position_ids[anc_tree[i]] + 1
        tree_position_ids = torch.tensor([tree_position_ids], dtype=torch.long, device=self.device)

        tree_attn_mask = torch.zeros((n, n), dtype=torch.bool)
        for i in range(n):
            j = i
            while j != -1:
                tree_attn_mask[i, j] = True
                j = anc_tree[j]
        tree_attn_mask = tree_attn_mask.view(1, 1, n, n).to(self.device)

        retrieve_indices_nest = []
        for i in range(n):
            if not is_leaf[i]:
                continue
            retrieve_indices = [i]
            while retrieve_indices[-1] != 0:
                retrieve_indices.append(anc_tree[retrieve_indices[-1]])
            retrieve_indices_nest.append(list(reversed(retrieve_indices)))
        max_depth = max(len(x) for x in retrieve_indices_nest)
        retrieve_indices_nest = [pad_path(x, max_depth) for x in retrieve_indices_nest]
        tree_retrieve_indices = torch.tensor(retrieve_indices_nest, dtype=torch.long, device=self.device)
        return {
            "tree_attn_mask": tree_attn_mask,
            "tree_position_ids": tree_position_ids,
            "tree_retrieve_indices": tree_retrieve_indices,
        }

    def gen_dyn_draft(self, index: int, match_length: int, start_token: int):
        result = self.core.gen_dyn_draft(index, match_length, start_token)
        return result.tokens, self.gen_buffers(result.ancestors)

    def save(self, path: str):
        self.core.save(os.path.expanduser(path))

    def state_count(self):
        return self.core.state_count()

    def edge_stats(self):
        return self.core.edge_stats()

    def transition_memory_usage(self):
        return self.core.transition_memory_usage()


class NullStaticSAM(StaticSAM):
    def __init__(self, n_predicts=40, *args, **kwargs):
        super().__init__(n_predicts)

    def transfer_tokens(self, tokens):
        pass

    def gen_draft(self, index, start_token):
        raise NotImplementedError

    def gen_dyn_draft(self, index, match_length: int, start_token):
        raise NotImplementedError


def build_sam(
    batch_tokens: List[List[int]],
    eos_token: int,
    n_predicts: int = 40,
    alpha: float = 4.0,
    K: int = 8,
    use_gsam: bool = True,
    use_small_dict: bool = True,
    map_type: str = "",
    lazy_threshold: int = 1,
):
    return StaticSAM.build(
        batch_tokens,
        eos_token,
        n_predicts=n_predicts,
        alpha=alpha,
        K=K,
        use_gsam=use_gsam,
        use_small_dict=use_small_dict,
        map_type=map_type,
        lazy_threshold=lazy_threshold,
    )


def dump_sam(path: str, sam: StaticSAM):
    sam.save(path)


def load_sam(path: str, map_type: str = "", lazy_threshold: int = 1):
    print("load gsam...")
    start = time.perf_counter()
    core = _gsamd_core.StaticSAMCore.load(
        os.path.expanduser(path),
        map_type,
        lazy_threshold,
    )
    sam = StaticSAM(core=core)
    end = time.perf_counter()
    print("loading ended in {} seconds.".format(end - start))
    return sam
