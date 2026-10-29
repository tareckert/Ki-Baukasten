"""Tests: jedes Modell baut, rechnet vorwärts/rückwärts; Varianten und Kombinationen."""

import pytest
import torch

from ki_baukasten import build_model, registry
from ki_baukasten.models.composite import _call, _main_output


# Import-Modelle ('extern') brauchen eine Datei – sie werden in test_import.py geprüft.
_BUILDABLE = [n for n in registry("model").names() if "import" not in registry("model").get(n).tags]


@pytest.mark.parametrize("name", _BUILDABLE)
def test_every_model_forward_backward(name):
    m = build_model(name)
    out = _call(m, m.example_input(2))
    out = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else _main_output(out))
    loss = out.float().pow(2).mean()
    aux = m.collect_aux_losses()
    (loss + (aux if aux is not None else 0)).backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("block", [
    {"mixer": "ssm", "ffn": "none"}, {"mixer": "conv"}, {"mixer": "rnn"},
    {"parallel": True, "norm_position": "sandwich"}, {"norm_position": "post", "ffn": {"name": "moe", "num_experts": 2}},
])
def test_transformer_blocks_causal(block):
    m = build_model({"name": "transformer", "args": {"vocab_size": 20, "d_model": 32, "n_layers": 2, "block": block}}).eval()
    a = torch.randint(0, 20, (1, 10))
    b = a.clone()
    b[0, 6:] = (b[0, 6:] + 1) % 20
    assert torch.allclose(m(a)[0, :6], m(b)[0, :6], atol=1e-5)


def test_layer_overrides_change_individual_layers():
    m = build_model({"name": "transformer", "args": {
        "d_model": 32, "n_layers": 4, "block": {"mixer": {"name": "attention", "args": {"num_heads": 4}}},
        "layer_overrides": {"0": {"mixer.num_heads": 2}, "1::2": {"mixer": "ssm"}, "-1": {"ffn": "mlp"}}}})
    assert m.blocks[0].mixer.attn.num_heads == 2
    assert type(m.blocks[1].mixer).__name__ == "SelectiveSSM"
    assert type(m.blocks[3].ffn).__name__ == "MLPFeedForward"


def test_architectures_and_generation():
    for arch, extra in [("encoder_only", {"pooling": "cls", "output_dim": 3}), ("encoder_decoder", {})]:
        m = build_model({"name": "transformer", "args": {"architecture": arch, "d_model": 32, "n_layers": 1, **extra}})
        x = m.example_input(2)
        assert _call(m, x).shape[0] == 2
    lm = build_model({"name": "transformer", "args": {"vocab_size": 30, "d_model": 32, "n_layers": 1}})
    out = lm.generate(torch.randint(0, 30, (2, 3)), max_new_tokens=5, temperature=0.7, top_k=5, top_p=0.9)
    assert out.shape == (2, 8)


@pytest.mark.parametrize("block_type", ["plain", "residual", "bottleneck", "inverted_residual", "convnext",
                                        "depthwise_separable"])
def test_cnn_block_types(block_type):
    stages = [{"channels": 16, "num_blocks": 1, "stride": 1, "block_type": block_type, "se_ratio": 4.0},
              {"channels": 32, "num_blocks": 1, "stride": 2, "block_type": block_type}]
    m = build_model({"name": "cnn", "args": {"input_size": [16, 16], "stages": stages}})
    assert m(m.example_input(2)).shape == (2, 10)


def test_graph_model_branches_and_weight_sharing():
    m = build_model({"name": "graph", "args": {
        "inputs": [{"name": "a", "shape": [4]}, {"name": "b", "shape": [4]}],
        "nodes": [{"name": "l1", "layer": {"name": "linear", "args": {"out_features": 8}}, "inputs": ["a"]},
                  {"name": "l2", "layer": {"name": "linear", "args": {"out_features": 8}}, "inputs": ["b"],
                   "share_weights_with": "l1"},
                  {"name": "out", "layer": {"name": "linear", "args": {"out_features": 2}}, "inputs": ["l1", "l2"],
                   "merge": "add"}]}})
    assert m.layers["l1"] is m.layers["l2"]
    assert m(a=torch.randn(3, 4), b=torch.randn(3, 4)).shape == (3, 2)


def test_nested_combinations():
    ens = build_model({"name": "ensemble", "args": {"combine": "stacking", "members": [
        {"name": "rnn", "args": {"input_dim": 4, "output_dim": 3}},
        {"name": "pipeline", "args": {"stages": [{"name": "rnn", "args": {"input_dim": 4, "output_dim": 8}},
                                                 {"name": "mlp", "args": {"output_dim": 3}}]}}]}})
    assert ens(torch.randn(2, 12, 4)).shape == (2, 3)


def test_autoencoder_variants_and_sampling():
    for v in ["ae", "vae", "denoising", "sparse", "vq"]:
        m = build_model({"name": "autoencoder", "args": {"variant": v}})
        out = m(m.example_input(3))
        assert out["reconstruction"].shape == (3, 32)
        assert m.sample(2).shape == (2, 32)
