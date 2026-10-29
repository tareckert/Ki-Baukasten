"""Tests: schneller Scan (Mamba/minGRU) und KV-/Zustands-Cache für die Generierung."""

import pytest
import torch

from ki_baukasten import build_model
from ki_baukasten.components.scan import linear_scan, scan_raw, selective_scan

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 3, 17, 70])
@pytest.mark.parametrize("mode", ["parallel", "chunked", "sequential", "auto"])
@pytest.mark.parametrize("memory_efficient", [True, False])
def test_linear_scan_matches_reference(T, mode, memory_efficient):
    a = torch.rand(2, T, 5, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, T, 5, 3, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    ref = linear_scan(a, b, h0, mode="sequential", memory_efficient=False)
    out = linear_scan(a, b, h0, mode=mode, chunk_size=8, memory_efficient=memory_efficient)
    g = torch.randn_like(ref)
    for x, y in zip(torch.autograd.grad(ref, [a, b, h0], g), torch.autograd.grad(out, [a, b, h0], g)):
        assert torch.allclose(x, y)
    assert torch.allclose(out, ref)


@pytest.mark.parametrize("mode", ["parallel", "chunked", "sequential"])
def test_reverse_scan(mode):
    a, b = torch.rand(2, 13, 4), torch.randn(2, 13, 4)
    with torch.no_grad():
        ref = scan_raw(a.flip(1), b.flip(1), "sequential").flip(1)
        assert torch.allclose(scan_raw(a, b, mode, 4, reverse=True), ref, atol=1e-6)


@pytest.mark.parametrize("backward", ["fused", "scan"])
def test_selective_scan_gradients(backward):
    T, E, N = 9, 4, 3
    f64 = dict(dtype=torch.float64, requires_grad=True)
    ins = [torch.randn(2, T, E, **f64), torch.rand(2, T, E, **f64), (-torch.rand(E, N, dtype=torch.float64) - 0.5),
           torch.randn(2, T, N, **f64), torch.randn(2, T, N, **f64), torch.randn(E, **f64), torch.randn(2, E, N, **f64)]
    ins[2].requires_grad_()
    y, last = selective_scan(*ins, mode="sequential", backward="autograd")
    y2, last2 = selective_scan(*ins, mode="auto", chunk_size=4, backward=backward)
    gy, gl = torch.randn_like(y), torch.randn_like(last)
    ref = torch.autograd.grad((y * gy).sum() + (last * gl).sum(), ins)
    got = torch.autograd.grad((y2 * gy).sum() + (last2 * gl).sum(), ins)
    assert torch.allclose(y, y2) and torch.allclose(last, last2)
    for r, g in zip(ref, got):
        assert torch.allclose(r, g)


def test_ssm_mixer_modes_agree():
    m = build_model({"name": "ssm", "vocab_size": 30, "d_model": 32, "n_layers": 2})
    x = torch.randint(0, 30, (2, 20))
    outs, grads = [], []
    for mode, bw in [("sequential", "autograd"), ("auto", "fused"), ("chunked", "scan"), ("parallel", "fused")]:
        for b in m.blocks:
            b.mixer.cfg.scan_mode, b.mixer.cfg.scan_backward = mode, bw
        m.zero_grad()
        out = m(x)
        out.square().mean().backward()
        outs.append(out.detach())
        grads.append(m.blocks[0].mixer.A_log.grad.clone())
    for o, g in zip(outs[1:], grads[1:]):
        assert torch.allclose(o, outs[0], atol=1e-5)
        assert torch.allclose(g, grads[0], atol=1e-5)


def test_mingru_parallel_matches_loop():
    from ki_baukasten.components.mixers import MinGRU
    cell = MinGRU(4, 6)
    x = torch.randn(3, 11, 4)
    h, last = cell(x)
    z, cand = torch.sigmoid(cell.gate(x)), cell.cand(x)
    ref, hs = torch.zeros(3, 6), []
    for t in range(11):
        ref = (1 - z[:, t]) * ref + z[:, t] * cand[:, t]
        hs.append(ref)
    assert torch.allclose(h, torch.stack(hs, 1), atol=1e-6) and torch.allclose(last, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

BASE = {"name": "transformer", "vocab_size": 40, "d_model": 32, "n_layers": 2, "max_seq_len": 64,
        "block": {"mixer": {"name": "attention", "num_heads": 4}}}
ATT = {"name": "attention", "num_heads": 4}

CACHE_MODELS = {
    "rope": BASE,
    "alibi": {**BASE, "positional": "alibi"},
    "learned": {**BASE, "positional": "learned"},
    "t5_bias": {**BASE, "positional": "relative_bias"},
    "gqa": {**BASE, "block": {"mixer": {**ATT, "num_kv_heads": 2}}},
    "mla_lite": {**BASE, "block": {"mixer": {**ATT, "kv_compression_dim": 8}}},
    "window": {**BASE, "block": {"mixer": {**ATT, "window_size": 3}}},
    "linear_kernel": {**BASE, "block": {"mixer": {**ATT, "kernel": "linear"}}},
    "register_tokens": {**BASE, "num_register_tokens": 2},
    "ssm": {"name": "ssm", "vocab_size": 40, "d_model": 32, "n_layers": 2},
    "hybrid": {"name": "hybrid", "vocab_size": 40, "d_model": 32, "n_layers": 4},
    "gru": {**BASE, "block": {"mixer": {"name": "rnn"}}},
    "lstm": {**BASE, "block": {"mixer": {"name": "rnn", "cell": "lstm"}}},
    "mingru": {**BASE, "block": {"mixer": {"name": "rnn", "cell": "mingru"}}},
    "conv": {**BASE, "block": {"mixer": {"name": "conv", "kernel_size": 3}}},
}


@pytest.mark.parametrize("name", list(CACHE_MODELS))
def test_cache_matches_full_recompute(name):
    m = build_model(CACHE_MODELS[name]).eval()
    assert m.supports_cache()
    x = torch.randint(0, 40, (2, 5))
    with_cache = m.generate(x, max_new_tokens=8, temperature=0, use_cache=True)
    without = m.generate(x, max_new_tokens=8, temperature=0, use_cache=False)
    assert torch.equal(with_cache, without)
    with torch.no_grad():
        full = m(with_cache)
        cache = m.new_cache()
        parts = [m(with_cache[:, :6], cache=cache)] + [m(with_cache[:, t:t + 1], cache=cache)
                                                        for t in range(6, with_cache.shape[1])]
    assert torch.allclose(torch.cat(parts, 1), full, atol=1e-5)


def test_cache_dtypes_and_limits():
    m = build_model(BASE).eval()
    x = torch.randint(0, 40, (2, 30))
    with torch.no_grad():
        ref = m(x)
        sizes = {}
        for cfg in [{}, {"dtype": "float16"}, {"dtype": "int8"}, {"max_length": 8, "sink_tokens": 2}]:
            cache = m.new_cache(cfg)
            out = torch.cat([m(x[:, :20], cache=cache)] + [m(x[:, t:t + 1], cache=cache) for t in range(20, 30)], 1)
            sizes[str(cfg)] = cache.nbytes()
            if "max_length" not in cfg:
                assert torch.allclose(out, ref, atol=5e-2)
    assert sizes["{'dtype': 'float16'}"] * 2 == sizes["{}"]
    assert sizes["{'dtype': 'int8'}"] < sizes["{'dtype': 'float16'}"]
    assert sizes["{'max_length': 8, 'sink_tokens': 2}"] < sizes["{}"]


def test_mla_cache_is_smaller():
    full = build_model(BASE).eval()
    mla = build_model(CACHE_MODELS["mla_lite"]).eval()
    x = torch.randint(0, 40, (2, 20))
    c1, c2 = full.new_cache(), mla.new_cache()
    with torch.no_grad():
        full(x, cache=c1)
        mla(x, cache=c2)
    assert c2.nbytes() < c1.nbytes() / 2


def test_generation_beyond_context_and_padding():
    m = build_model({**BASE, "max_seq_len": 10}).eval()
    x = torch.randint(0, 40, (2, 5))
    exact = m.generate(x, max_new_tokens=15, temperature=0, cache={"overflow_keep": 1.0})
    assert torch.equal(exact, m.generate(x, max_new_tokens=15, temperature=0, use_cache=False))
    assert m.generate(x, max_new_tokens=15, temperature=0).shape == (2, 20)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, :2] = False
    m2 = build_model(BASE).eval()
    a = m2.generate(x, max_new_tokens=6, temperature=0, attention_mask=mask)
    b = m2.generate(x, max_new_tokens=6, temperature=0, attention_mask=mask, use_cache=False)
    assert torch.equal(a, b)


def test_uncachable_mixer_falls_back():
    m = build_model({**BASE, "architecture": "encoder_only"})
    assert not m.supports_cache()
    with pytest.raises(ValueError):
        m.new_cache()
