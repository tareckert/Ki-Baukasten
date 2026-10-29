"""Tests: Kompression (Pruning, Low-Rank, Sparsity, Palette), Lehrer-Schüler, Weiterlernen, Zielsysteme."""

import copy
import warnings

import pytest
import torch

from ki_baukasten import Project, build_model
from ki_baukasten.analysis.targets import TARGETS, recommend
from ki_baukasten.compress import (apply_rules, drop_layers, low_rank, prune_ffn, prune_heads, prune_mlp, sparsify,
                                   sparsity_of)
from ki_baukasten.training.continual import ewc_penalty, fisher_information, mix_replay, shrink_config

warnings.simplefilter("ignore")
SPEC = {"name": "transformer", "vocab_size": 40, "d_model": 64, "n_layers": 4,
        "block": {"mixer": {"name": "attention", "num_heads": 8, "num_kv_heads": 4}}}
CALIB = [{"inputs": torch.randint(0, 40, (4, 16), generator=torch.Generator().manual_seed(i))} for i in range(3)]


def _params(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("importance", ["magnitude", "activation", "taylor", "random"])
def test_structured_pruning_rebuilds_exactly(importance):
    torch.manual_seed(0)
    m = build_model(SPEC).eval()
    n0 = _params(m)
    rules = prune_heads(m, 0.5, importance, CALIB) + prune_ffn(m, 0.5, importance, CALIB)
    rules += drop_layers(m, 1, importance="block_influence", calibration=CALIB) + low_rank(m, energy=0.8)
    assert _params(m) < 0.5 * n0
    x = CALIB[0]["inputs"]
    out = m(x)
    rebuilt = build_model(SPEC).eval()
    apply_rules(rebuilt, rules)
    rebuilt.load_state_dict(m.state_dict())
    assert torch.allclose(rebuilt(x), out)
    assert len(m.blocks) == 3 and all(b.mixer.attn.num_heads == 4 for b in m.blocks)
    assert torch.equal(m.generate(x[:, :4], max_new_tokens=4, temperature=0),
                       m.generate(x[:, :4], max_new_tokens=4, temperature=0, use_cache=False))


def test_head_pruning_query_granularity_keeps_kv():
    m = build_model(SPEC)
    prune_heads(m, 0.5, "magnitude", granularity="query")
    a = m.blocks[0].mixer.attn
    assert (a.num_heads, a.num_kv_heads) == (4, 4)


def test_prune_mlp_with_norms_and_rebuild():
    spec = {"name": "mlp", "input_dim": 10, "output_dim": 3, "hidden_dims": [64, 64, 32], "layer": {"norm": "layernorm"}}
    m = build_model(spec)
    rules = prune_mlp(m, 0.5, "activation", [{"inputs": torch.randn(32, 10)}])
    x = torch.randn(5, 10)
    assert [layer.ops["linear"].out_features for layer in m.layers] == [32, 32, 16]
    r = build_model(spec)
    apply_rules(r, rules)
    r.load_state_dict(m.state_dict())
    assert torch.allclose(r(x), m(x))


def test_sparsify_and_palette_quantization():
    m = build_model(SPEC)
    masks = sparsify(m, 0.6)
    assert masks and abs(sparsity_of(m) - 0.6) < 0.05
    from ki_baukasten.memory.quant import QuantLinear
    lin = torch.nn.Linear(128, 64)
    q = QuantLinear.from_linear(lin, "pal4")
    assert (q.dequantize() - lin.weight).abs().mean() < 0.2 * lin.weight.abs().mean()
    assert q.nbytes() < lin.weight.numel()  # < 1 Byte pro Gewicht


def test_project_compress_pipeline_save_load(tmp_path):
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.set("training.max_steps", 10)
    p.train()
    report = p.compress([{"name": "prune_heads", "args": {"amount": 0.5}}, {"name": "prune_ffn", "args": {"amount": 0.5}},
                         {"name": "drop_layers", "args": {"count": 1}}, {"name": "heal", "args": {"steps": 5}},
                         {"name": "quantize", "args": {"format": "int8"}}])
    assert [s.stage for s in report.stages] == ["Original", "prune_heads", "prune_ffn", "drop_layers", "heal",
                                                "quantize"]
    assert report.ratio("size_bytes") > 4 and report.after.loss is not None
    assert "Größe ÷" in str(report)
    p.save(tmp_path)
    q = Project.load(tmp_path)
    x = torch.randint(0, 40, (1, 12))
    p.model.cpu().eval()
    q.model.eval()
    with torch.no_grad():
        assert torch.allclose(p.model(x), q.model(x))


def test_distill_student_and_feature_loss():
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.set("training.max_steps", 10)
    p.train()
    s = p.distill(steps=8, feature_weight=0.5, kd_loss="reverse_kl")
    assert _params(s.model) < _params(p.model) / 3
    keys = {k for h in s.history for k in h}
    assert {"val_distill", "val_features", "val_agreement"} <= keys


@pytest.mark.parametrize("name", ["transformer", "mlp", "rnn", "cnn", "ssm"])
def test_shrink_config_builds_smaller(name):
    m = build_model(name)
    small = build_model(shrink_config({"name": name, "args": m.config.to_dict()}))
    assert _params(small) < _params(m)


def test_continual_methods_run():
    c = Project.from_preset("tabellen_klassifikation")
    c.set("training.verbose", False)
    c.set("training.epochs", 1)
    c.train()
    new = {"inputs": torch.randn(64, c.config.model.args.input_dim),
           "targets": torch.randint(0, c.config.model.args.output_dim, (64,))}
    for method in ["lwf", "ewc", "replay", "none"]:
        q = Project(c.config.copy())
        q._bundle, q.model = c._bundle, copy.deepcopy(c.model)
        assert q.continue_training(new, method=method, steps=3)
    with pytest.raises(Exception):
        c.continue_training(new, method="unbekannt")


def test_ewc_penalty_and_replay():
    m = build_model("mlp")
    batches = [{"inputs": torch.randn(8, 16), "targets": torch.randn(8, 1)}]
    from ki_baukasten.compress import calibration_loss
    fisher = fisher_information(m, batches, calibration_loss)
    pen = ewc_penalty(m, fisher, 1.0)
    assert pen().item() == 0.0
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1)
    assert pen().item() > 0.0
    mixed = mix_replay({"inputs": torch.zeros(10, 4)}, {"inputs": torch.ones(20, 4)}, 0.5)
    assert mixed["inputs"].shape[0] == 15


def test_target_recommendations():
    m = build_model({"name": "transformer", "vocab_size": 1000, "d_model": 512, "n_layers": 8})
    from ki_baukasten.analysis.benchmark import count_macs
    macs = count_macs(m, m.example_input(1))
    big = recommend(m, "rechenzentrum_gpu", macs)
    tiny = recommend(m, "mikrocontroller", macs)
    assert big.fits_now and not tiny.fits_now
    assert tiny.student is not None and tiny.size_after_mb < tiny.size_now_mb / 20
    assert any(s["name"] == "quantize" for s in recommend(m, "smartphone", macs).steps)
    assert set(TARGETS) >= {"smartphone", "raspberry_pi", "browser", "apple_silicon"}


def test_reduce_for_and_overview():
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.set("training.max_steps", 5)
    p.set("deployment.heal_steps", 3)
    p.train()
    assert "Smartphone" in p.deployment_overview(["smartphone", "browser"])
    reduced = p.reduce_for("smartphone")
    assert reduced.last_compression is not None
    assert reduced.last_compression.ratio("size_bytes") > 3
