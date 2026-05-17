import pytest
import torch

from model.gsamd import DraftModel, SamdConfig, SamdGenerationConfig, SamdModel
from model.gsamd.sam import StaticSAM, build_sam, dump_sam, load_sam


def test_gsam_draft_does_not_cross_source_boundary():
    sam = StaticSAM.build(
        [[1, 2], [3, 4]],
        eos_token=0,
        n_predicts=4,
        device="cpu",
        use_gsam=True,
        use_small_dict=True,
    )

    sam.transfer_tokens([1])
    index, match = sam.lookup(2)

    assert match == 2
    assert sam.gen_draft(index, 2) == [2, 0, 0, 0]


@pytest.mark.parametrize("use_small_dict", [True, False])
def test_static_sam_lookup_and_tree_draft(use_small_dict):
    sam = StaticSAM.build(
        [[1, 2, 3], [1, 2, 4], [1, 2, 3]],
        eos_token=0,
        n_predicts=5,
        device="cpu",
        use_gsam=True,
        use_small_dict=use_small_dict,
    )

    sam.transfer_tokens([1])
    index, match = sam.lookup(2)
    tree, buffers = sam.gen_dyn_draft(index, match, 2)

    assert match == 2
    assert tree[0] == 2
    assert 3 in tree
    assert buffers["tree_attn_mask"].device.type == "cpu"


@pytest.mark.parametrize(
    ("map_type", "lazy_threshold"),
    [
        ("unordered", 1),
        ("lazy", 1),
        ("lazy", 2),
        ("lazy", 3),
        ("lazy", 4),
        ("lazy", 5),
        ("int32", 1),
        ("lazy_int32", 1),
        ("lazy_int32", 3),
        ("lazy_int32", 5),
    ],
)
def test_static_sam_map_backends_build_same_language(map_type, lazy_threshold):
    sam = StaticSAM.build(
        [[1, 2, 3], [1, 2, 4], [5, 6]],
        eos_token=0,
        n_predicts=5,
        device="cpu",
        use_gsam=True,
        use_small_dict=(map_type == "lazy"),
        map_type=map_type,
        lazy_threshold=lazy_threshold,
    )

    sam.transfer_tokens([1])
    index, match = sam.lookup(2)

    assert sam.map_type == map_type
    assert sam.lazy_threshold == lazy_threshold
    assert sam.transition_memory_usage() > 0
    assert match == 2
    assert sam.gen_draft(index, 2) in ([2, 3, 0, 0, 0], [2, 4, 0, 0, 0])


def test_static_sam_protobuf_round_trip_preserves_flags_and_lookup(tmp_path):
    path = tmp_path / "static_gsam.pb"
    sam = build_sam(
        [[10, 11, 12], [10, 11, 13]],
        eos_token=0,
        n_predicts=4,
        use_gsam=True,
        use_small_dict=True,
        map_type="lazy",
        lazy_threshold=3,
    )
    dump_sam(str(path), sam)

    loaded = load_sam(str(path))
    loaded.device = "cpu"
    loaded.transfer_tokens([10])
    index, match = loaded.lookup(11)

    assert loaded.use_gsam is True
    assert loaded.use_small_dict is True
    assert loaded.map_type == "lazy"
    assert loaded.lazy_threshold == 3
    assert match == 2
    assert loaded.gen_draft(index, 11) in ([11, 12, 0, 0], [11, 13, 0, 0])


def test_static_sam_protobuf_round_trip_preserves_int32_map(tmp_path):
    path = tmp_path / "static_gsam_int32.pb"
    sam = build_sam(
        [[10, 11, 12], [10, 11, 13]],
        eos_token=0,
        n_predicts=4,
        use_gsam=True,
        use_small_dict=False,
        map_type="int32",
    )
    dump_sam(str(path), sam)

    loaded = load_sam(str(path))

    assert loaded.map_type == "int32"
    assert loaded.use_small_dict is False
    loaded.transfer_tokens([10])
    assert loaded.lookup(11)[1] == 2


def test_static_sam_load_can_override_map_type_from_same_file(tmp_path):
    path = tmp_path / "static_gsam_prototype.pb"
    sam = build_sam(
        [[1, 2, 3], [1, 2, 4], [5, 6]],
        eos_token=0,
        n_predicts=5,
        use_gsam=True,
        use_small_dict=True,
        map_type="lazy",
        lazy_threshold=1,
    )
    dump_sam(str(path), sam)

    for map_type, lazy_threshold in [
        ("unordered", 1),
        ("lazy", 4),
        ("int32", 1),
        ("lazy_int32", 3),
    ]:
        loaded = load_sam(str(path), map_type=map_type, lazy_threshold=lazy_threshold)
        loaded.transfer_tokens([1])
        index, match = loaded.lookup(2)

        assert loaded.use_gsam is True
        assert loaded.map_type == map_type
        assert loaded.lazy_threshold == lazy_threshold
        assert match == 2
        assert loaded.gen_draft(index, 2) in ([2, 3, 0, 0, 0], [2, 4, 0, 0, 0])


def test_small_dict_and_hash_map_build_same_language_size():
    small = build_sam([[1, 2, 1, 3], [1, 2, 4]], 0, use_gsam=True, use_small_dict=True)
    hashed = build_sam([[1, 2, 1, 3], [1, 2, 4]], 0, use_gsam=True, use_small_dict=False)

    assert small.state_count() == hashed.state_count()
    assert small.edge_stats().total_states == hashed.edge_stats().total_states


def test_tree_decode_passes_attention_mask_to_lm():
    class DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type(
                "Config",
                (),
                {
                    "max_position_embeddings": 32,
                    "num_hidden_layers": 1,
                },
            )()
            self.hf_device_map = {}
            self.last_attention_mask = None

        def named_modules(self, memo=None, prefix="", remove_duplicate=True):
            yield "", self

        def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None):
            self.last_attention_mask = attention_mask
            logits = torch.zeros((1, input_ids.shape[-1], 8), dtype=torch.float32)
            logits[:, :, 1] = 1.0
            return type("Outputs", (), {"logits": logits, "last_hidden_states": None})()

    config = SamdConfig(n_predicts=3)
    static_sam = StaticSAM.build([[1, 2, 3]], eos_token=0, n_predicts=3, device="cpu")
    draft = DraftModel(config, sam_static=static_sam, device="cpu")
    model = SamdModel(config, DummyLM(), draft, eos_token_id=0, dtype=torch.float32, device="cpu")
    model.cache = type(
        "Cache",
        (),
        {
            "select_indices": lambda self, indices, accept_length: None,
        },
    )()
    draft.update(tokens=torch.tensor([1, 2]), tree_tokens=torch.tensor([1, 2]), tree_logits=torch.zeros(2, 8))
    model.gen_config = SamdGenerationConfig(max_new_tokens=4, max_cache_len=32)

    model.decode(torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]]), length=2)

    assert model.lm.last_attention_mask is not None
    assert model.lm.last_attention_mask.shape == (
        1,
        1,
        model.tree_position_ids.shape[-1],
        2 + model.tree_position_ids.shape[-1],
    )
    assert model.lm.last_attention_mask[0, 0, 0, :3].tolist() == [1.0, 1.0, 1.0]
    assert model.lm.last_attention_mask[0, 0, 0, 3:].sum().item() == 0.0
