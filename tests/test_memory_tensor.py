"""Tests: Speicheroptimierung und Tensor-Chirurgie."""

import copy

import pytest
import torch

from ki_baukasten import MemoryConfig, TensorRuleConfig, TensorSurgeon, Trainer, TrainingConfig, build_model
from ki_baukasten.memory import MemoryManager, enable_gradient_checkpointing, estimate_memory, quantize_dynamic_int8

SPEC = {"name": "transformer", "args": {"vocab_size": 40, "d_model": 32, "n_layers": 3}}


def test_gradient_checkpointing_is_exact_and_keeps_names():
    m1 = build_model(SPEC)
    m2 = copy.deepcopy(m1)
    keys = list(m1.state_dict())
    assert enable_gradient_checkpointing(m2) == 3
    assert list(m2.state_dict()) == keys
    x = torch.randint(0, 40, (2, 12))
    m1(x).sum().backward()
    m2(x).sum().backward()
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


@pytest.mark.parametrize("mode", ["cpu", "disk"])
def test_offload_inference_exact(mode, tmp_path):
    m = build_model(SPEC).eval()
    x = torch.randint(0, 40, (2, 12))
    ref = m(x)
    mm = MemoryManager(MemoryConfig(offload={"mode": mode, "directory": str(tmp_path)})).prepare_model(m, training=False)
    with torch.no_grad():
        assert torch.allclose(mm(x), ref, atol=1e-6)


def test_offload_training_is_rejected():
    with pytest.raises(Exception, match="stream"):
        Trainer(build_model(SPEC), TrainingConfig(verbose=False), MemoryConfig(offload={"mode": "cpu"}))


@pytest.mark.parametrize("spec", [
    {"name": "transformer", "args": {"vocab_size": 40, "d_model": 32, "n_layers": 3,
                                     "block": {"dropout": 0.2, "mixer": {"name": "attention", "dropout": 0.1}}}},
    "cnn", "ssm",
])
def test_block_streaming_gradients_exact(spec):
    from ki_baukasten.memory import enable_block_streaming
    torch.manual_seed(0)
    m1 = build_model(spec)
    m2 = copy.deepcopy(m1)
    keys = list(m1.state_dict())
    states = enable_block_streaming(m2, m2.checkpointable_blocks(), torch.device("cpu"))
    assert list(m2.state_dict()) == keys
    x = m1.example_input(3)
    torch.manual_seed(1)
    m1(x).float().square().mean().backward()
    torch.manual_seed(1)
    m2(x).float().square().mean().backward()
    for (n, a), b in zip(m1.named_parameters(), m2.parameters()):
        if a.grad is None:  # ungenutzter Parameter
            assert b.grad is None or not b.grad.any(), n
        else:
            assert torch.allclose(a.grad, b.grad, atol=1e-5), n
    for a, b in zip(m1.buffers(), m2.buffers()):  # BatchNorm-Statistiken genau einmal aktualisiert
        assert torch.allclose(a.float(), b.float())
    assert all(s.transfers == 2 for s in states)  # Forward + Backward


def test_block_streaming_training_matches_normal():
    data = {"inputs": torch.randint(0, 40, (32, 10))}
    results = []
    for mem in [MemoryConfig(), MemoryConfig(offload={"mode": "stream", "layers": "0:2"}, gradient_checkpointing=True)]:
        torch.manual_seed(0)
        model = build_model(SPEC)
        tr = Trainer(model, TrainingConfig(paradigm="causal_lm", epochs=1, verbose=False, seed=0), mem)
        tr.fit(data)
        results.append([p.detach().clone() for p in tr.model.parameters()])
    for a, b in zip(*results):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("mem", [
    {"quantization": "fake_quant", "quant_bits": 4}, {"mixed_precision": "bf16", "activation_offload": True},
    {"param_dtype": "bfloat16"}, {"gradient_checkpointing": True, "checkpoint_layers": "::2"},
])
def test_memory_options_train(mem):
    tr = Trainer(build_model(SPEC), TrainingConfig(paradigm="causal_lm", epochs=1, verbose=False), MemoryConfig(**mem))
    tr.fit({"inputs": torch.randint(0, 40, (32, 10))})


def test_estimator_and_int8():
    rep = estimate_memory(build_model(SPEC), batch_size=16, memory=MemoryConfig(memory_budget_gb=1e-4))
    assert rep.activations_bytes and not rep.fits() and rep.recommendations
    q = quantize_dynamic_int8(build_model("mlp"))
    assert q(torch.randn(2, 16)).shape == (2, 1)


def test_tensor_surgery_end_to_end():
    m = build_model(SPEC)
    rules = [TensorRuleConfig(target="embed", op="freeze"),
             TensorRuleConfig(target=r"blocks\.0\..*q_proj\.weight", index="0:4, :", op="freeze"),
             TensorRuleConfig(target=r"blocks\.1\.", module_type="Linear", param_kind="weight",
                              op={"name": "prune", "args": {"amount": 0.5}}),
             TensorRuleConfig(target="o_proj", op={"name": "constraint", "args": {"kind": "max_norm", "value": 0.3}}),
             TensorRuleConfig(target="lm_head|final_norm", op={"name": "lr_scale", "args": {"factor": 0.1}}),
             TensorRuleConfig(target=r"blocks\.2\..*k_proj\.weight", op={"name": "low_rank", "args": {"rank": 2}}),
             TensorRuleConfig(target=r"blocks\.2\..*v_proj\.weight",
                              op={"name": "tie", "args": {"source": "blocks.1.mixer.attn.v_proj.weight"}})]
    s = TensorSurgeon(m, rules)
    assert "Regel" in s.preview()
    s.apply_build(num_layers=3)
    q0 = m.blocks[0].mixer.attn.q_proj.weight.detach().clone()
    tr = Trainer(m, TrainingConfig(paradigm="causal_lm", epochs=2, verbose=False,
                                   optimizer={"name": "adamw", "args": {"lr": 1e-2}}), surgeon=s)
    assert len({g["lr"] for g in tr.optimizers["main"].param_groups}) == 2
    tr.fit({"inputs": torch.randint(0, 40, (64, 10))})
    q1 = m.blocks[0].mixer.attn.q_proj.weight.detach()
    assert torch.equal(q0[:4], q1[:4]) and not torch.equal(q0[4:], q1[4:])
    assert not m.embed.weight.requires_grad
    assert (m.blocks[1].mixer.attn.q_proj.weight == 0).float().mean() >= 0.5
    assert m.blocks[0].mixer.attn.o_proj.weight.norm(dim=1).max() <= 0.3 + 1e-5
    assert m.blocks[2].mixer.attn.v_proj.weight is m.blocks[1].mixer.attn.v_proj.weight
