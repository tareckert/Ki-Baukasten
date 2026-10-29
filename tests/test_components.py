"""Tests aller Komponenten-Registries (jede Komponente muss mit Standardwerten funktionieren)."""

import pytest
import torch

from ki_baukasten import registry
from ki_baukasten.components import AttentionConfig, MultiHeadAttention


@pytest.mark.parametrize("name", registry("activation").names())
def test_activations(name):
    x = torch.randn(3, 7)
    assert registry("activation").build(name)(x).shape == x.shape


@pytest.mark.parametrize("name", registry("normalization").names())
@pytest.mark.parametrize("dims,shape", [("feature", (2, 5, 16)), (2, (2, 16, 4, 4))])
def test_norms(name, dims, shape):
    m = registry("normalization").build(name, 16, dims=dims)
    assert m(torch.randn(*shape)).shape == shape


@pytest.mark.parametrize("name", registry("initializer").names())
def test_initializers(name):
    for shape in [(6, 4), (6,), (6, 4, 3, 3)]:
        t = torch.empty(*shape)
        registry("initializer").build(name, t, num_layers=4)
        assert torch.isfinite(t).all()


@pytest.mark.parametrize("kernel", registry("attention_kernel").names())
@pytest.mark.parametrize("pos", ["none", "rope", "alibi", "relative_bias", "learned"])
def test_attention_variants_are_causal(kernel, pos):
    att = MultiHeadAttention(32, AttentionConfig(num_heads=4, num_kv_heads=2, kernel=kernel)).eval()
    pe = registry("positional").build(pos, 32, 8, 4, 64)
    x = torch.randn(1, 10, 32)
    x2 = x.clone()
    x2[:, 7:] += 1.0
    y1 = att(pe.embed(x, torch.arange(10)), positional=pe, causal=True)
    y2 = att(pe.embed(x2, torch.arange(10)), positional=pe, causal=True)
    assert torch.allclose(y1[:, :7], y2[:, :7], atol=1e-5)


def test_sdpa_matches_manual():
    torch.manual_seed(0)
    a = MultiHeadAttention(32, AttentionConfig(num_heads=4)).eval()
    cfg = AttentionConfig(num_heads=4)
    cfg.set("kernel.backend", "manual")
    b = MultiHeadAttention(32, cfg).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 9, 32)
    pad = torch.ones(2, 9, dtype=torch.bool)
    pad[1, -2:] = False
    assert torch.allclose(a(x, key_padding_mask=pad, causal=True), b(x, key_padding_mask=pad, causal=True), atol=1e-5)


@pytest.mark.parametrize("name", registry("feedforward").names())
def test_feedforward(name):
    x = torch.randn(2, 5, 16)
    m = registry("feedforward").build(name, 16)
    assert m(x).shape == x.shape


@pytest.mark.parametrize("name", registry("sequence_mixer").names())
def test_mixers(name):
    m = registry("sequence_mixer").build(name, 16)
    causal = name != "fourier"
    assert m(torch.randn(2, 6, 16), causal=causal).shape == (2, 6, 16)


def test_moe_aux_loss():
    moe = registry("feedforward").build({"name": "moe", "num_experts": 4, "top_k": 1, "capacity_factor": 1.0}, 16)
    moe(torch.randn(2, 8, 16)).sum().backward()
    assert moe.aux_loss is not None and moe.last_load.sum().item() == pytest.approx(1.0)
