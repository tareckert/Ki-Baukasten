"""Tests: eigene Gewichts-Quantisierung, LoRA/DoRA/QLoRA, 8-Bit-AdamW."""

import copy
import warnings

import pytest
import torch
from torch import nn

from ki_baukasten import MemoryConfig, Project, Trainer, TrainingConfig, build_model
from ki_baukasten.memory import (LoRAConfig, QuantLinear, apply_lora, lora_parameter_count, merge_lora,
                                 quantize_model, quantized_bytes)
from ki_baukasten.training.optimizers import AdamW8bit

SPEC = {"name": "transformer", "args": {"vocab_size": 40, "d_model": 64, "n_layers": 2}}


@pytest.mark.parametrize("fmt,tol,ratio", [("int8", 0.01, 0.30), ("int4", 0.2, 0.17), ("nf4", 0.2, 0.17)])
@pytest.mark.parametrize("group_size", [None, 32])
def test_quant_linear_accuracy_and_size(fmt, tol, ratio, group_size):
    torch.manual_seed(0)
    lin = nn.Linear(128, 64)
    nn.init.normal_(lin.weight, std=0.05)
    q = QuantLinear.from_linear(lin, fmt, group_size)
    x = torch.randn(4, 128)
    rel = (q(x) - lin(x)).abs().mean() / lin(x).abs().mean()
    assert rel < tol
    assert q.nbytes() < ratio * lin.weight.numel() * 4
    clone = QuantLinear(128, 64, True, fmt, group_size)
    clone.load_state_dict(q.state_dict())
    assert torch.equal(clone(x), q(x))


def test_nf4_beats_int4_on_normal_weights():
    torch.manual_seed(0)
    lin = nn.Linear(256, 256)
    nn.init.normal_(lin.weight)
    err = {f: (QuantLinear.from_linear(lin, f, 64).dequantize() - lin.weight).pow(2).mean() for f in ("int4", "nf4")}
    assert err["nf4"] < err["int4"]


def test_quantize_model_skips_head_and_saves_memory():
    m = build_model(SPEC)
    names = quantize_model(m, "int8")
    assert names and all("lm_head" not in n for n in names)
    q, full = quantized_bytes(m)
    assert q < full / 3
    assert m(torch.randint(0, 40, (2, 8))).shape == (2, 8, 40)


@pytest.mark.parametrize("init", ["kaiming", "gaussian", "pissa"])
@pytest.mark.parametrize("dora", [False, True])
def test_lora_starts_at_base_and_merges(init, dora):
    torch.manual_seed(0)
    m = build_model(SPEC)
    keys = list(m.state_dict())
    x = torch.randint(0, 40, (2, 8))
    ref = m(x)
    apply_lora(m, LoRAConfig(enabled=True, init=init, use_dora=dora))
    assert torch.allclose(m(x), ref, atol=1e-4)
    lora, total = lora_parameter_count(m)
    assert 0 < lora < total * 0.3
    assert all(".lora_" in n for n, p in m.named_parameters() if p.requires_grad)
    m(x).square().mean().backward()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.1)
        before = m(x)
        merge_lora(m)
        assert list(m.state_dict()) == keys
        assert torch.allclose(m(x), before, atol=1e-4)


def test_qlora_trains_and_reloads(tmp_path):
    warnings.simplefilter("ignore")
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.max_steps", 5)
    p.set("training.verbose", False)
    p.set("training.optimizer", "adamw8bit")
    p.set("memory.quantization", "nf4")
    p.set("memory.lora.enabled", True)
    p.set("memory.lora.train_also", "norm")
    p.train()
    trainable = [n for n, t in p.model.named_parameters() if t.requires_grad]
    assert trainable and all(".lora_" in n or "norm" in n for n in trainable)
    assert any(isinstance(mod, QuantLinear) for mod in p.model.modules())
    p.save(tmp_path)
    q = Project.load(tmp_path)
    x = torch.randint(0, 40, (1, 8))
    p.model.cpu().eval()
    q.model.eval()
    with torch.no_grad():
        assert torch.allclose(p.model(x), q.model(x))


def test_adamw8bit_converges_like_adamw():
    torch.manual_seed(0)
    X = torch.randn(256, 16)
    Y = torch.tanh(X @ torch.randn(16, 4))
    net = nn.Sequential(nn.Linear(16, 128), nn.GELU(), nn.Linear(128, 4))
    losses = {}
    for name, make in [("adamw", lambda ps: torch.optim.AdamW(ps, lr=3e-3)),
                       ("8bit", lambda ps: AdamW8bit(ps, lr=3e-3, min_8bit_size=0))]:
        n = copy.deepcopy(net)
        opt = make(n.parameters())
        for _ in range(200):
            loss = (n(X) - Y).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses[name] = loss.item()
        if name == "8bit":
            assert opt.state_bytes() < sum(t.numel() for t in n.parameters()) * 3
    assert losses["8bit"] < 2 * losses["adamw"] + 1e-3


def test_trainer_with_quantization_options():
    for mem in [{"quantization": "int8", "lora": {"enabled": True, "targets": "q_proj|v_proj"}},
                {"quantization": "int4", "quant_group_size": 32, "lora": {"enabled": True, "layers": "-1"}}]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr = Trainer(build_model(SPEC), TrainingConfig(paradigm="causal_lm", epochs=1, verbose=False),
                         MemoryConfig(**mem))
            tr.fit({"inputs": torch.randint(0, 40, (16, 10))})
