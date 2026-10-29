"""Tests: Aufbau-Darstellung, Ratgeber, Export, Rückgängig/Vergleich, neue CLI-Befehle, KAN, YaRN."""

import json
import warnings

import pytest
import torch

from ki_baukasten import Project, build_model, registry
from ki_baukasten.analysis import architecture_report, inspect_model
from ki_baukasten.ui.cli import main as cli

HYBRID = {"name": "hybrid", "vocab_size": 30, "d_model": 32, "n_layers": 4, "block": {"dropout": 0.1},
          "layer_overrides": {"1::2": {"mixer": "attention"}, "-1": {"mixer.num_kv_heads": 2, "dropout": 0.0}}}


@pytest.mark.parametrize("name", ["transformer", "cnn", "mlp", "rnn", "gnn", "vae", "graph"])
def test_architecture_formats_for_models(name):
    if name not in registry("model"):
        pytest.skip(f"Modell {name} nicht registriert")
    m = build_model(name)
    for fmt in ("text", "markdown", "mermaid", "json"):
        assert architecture_report(m, fmt)
    data = json.loads(architecture_report(m, "json"))
    assert data["parameters"] == sum(p.numel() for p in m.parameters())


def test_architecture_layer_rows_and_shapes():
    m = build_model(HYBRID)
    info = inspect_model(m)
    assert info.output_shapes == [[1, 16, 30]]
    assert len(info.layer_rows) == 4
    assert info.layer_rows[0]["abweichungen"] == {}
    assert info.layer_rows[1]["abweichungen"] == {"mixer": "attention"}
    assert info.layer_rows[3]["abweichungen"] == {"mixer": "attention", "mixer.num_kv_heads": 2, "dropout": 0.0}
    assert info.modules["blocks.0"].out_shapes == [[1, 16, 32]]
    assert "Cache" in info.modules["blocks.0.mixer"].markers
    md = info.to_markdown()
    assert "```mermaid" in md and "model.layer_overrides.3" in md and "blocks.0.mixer.in_proj.weight" in md


def test_advisor_and_apply_with_undo():
    warnings.simplefilter("ignore")
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.set("model.block.mixer.num_heads", 8)
    p.set("training.gradient_clipping.mode", "none")
    p.set("memory.memory_budget_gb", 0.02)
    p.set("training.batch_size", 4)
    tips = {t.key: t for t in p.advise()}
    assert {"adamw8bit", "gqa", "clipping", "checkpointing"} <= set(tips)
    assert all(t.reason for t in tips.values())
    chosen = p.apply_advice(["adamw8bit", "gqa"])
    assert {c.key for c in chosen} == {"adamw8bit", "gqa"}
    assert p.get("training.optimizer").name == "adamw8bit" and p.get("model.block.mixer.num_kv_heads") == 2
    assert p.undo()  # ein Schritt für alle übernommenen Vorschläge
    assert p.get("training.optimizer").name == "adamw" and p.get("model.block.mixer.num_kv_heads") is None
    assert p.redo() and p.get("training.optimizer").name == "adamw8bit"


def test_undo_redo_and_diff():
    p = Project.from_preset("sprachmodell_mini")
    p.set("model.d_model", 96)
    p.set_all("dropout", 0.2)
    p.update({"training.batch_size": 8})
    assert p.undo(2) and p.get("model.d_model") == 96 and p.get("model.block.dropout") != 0.2
    assert p.redo() and p.get("model.block.dropout") == 0.2
    with pytest.raises(Exception):
        p.set("model.d_model", -5)
    assert p.undo() and p.get("model.block.dropout") != 0.2  # fehlgeschlagenes set landet nicht im Verlauf
    paths = {d[0] for d in p.diff("sprachmodell_mini")}
    assert "model.d_model" in paths and "name" not in paths
    other = {d[0] for d in p.diff("sprachmodell_mamba")}
    assert "model" in other and "model.block.mixer" in other


@pytest.mark.parametrize("fmt", ["export", "torchscript", "state_dict"])
def test_export_formats(tmp_path, fmt):
    warnings.simplefilter("ignore")
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.set("memory.lora.enabled", True)
    p.build()
    target = p.export(tmp_path / "modell", fmt)
    assert target.exists() and target.stat().st_size > 0
    if fmt == "export":
        loaded = torch.export.load(str(target)).module()
        x = torch.randint(0, 40, (1, 65))
        p.model.eval()
        with torch.no_grad():
            assert torch.allclose(loaded(x), p.model(x), atol=1e-5)


def test_search_finds_settings_and_components():
    p = Project.from_preset("sprachmodell_mini")
    hits = dict(p.search("8 bit"))
    assert "optimizer:adamw8bit" in hits
    assert any(path.startswith("memory.quant") for path in hits)


def test_cli_new_commands(capsys, tmp_path):
    cli(["aufbau", "sprachmodell_mini", "--tiefe", "1"])
    assert "Schichten" in capsys.readouterr().out
    cli(["aufbau", "sprachmodell_mini", "--format", "markdown", "-o", str(tmp_path / "a.md")])
    assert (tmp_path / "a.md").read_text(encoding="utf-8").startswith("# Aufbau")
    cli(["optimieren", "sprachmodell_mini", "--setze", "training.gradient_clipping.mode=none",
         "--anwenden", "clipping", "-o", str(tmp_path / "opt.json")])
    out = capsys.readouterr().out
    assert "clipping" in out and (tmp_path / "opt.json").exists()
    cli(["suche", "kv cache", "--quelle", "sprachmodell_mini"])
    assert "num_kv_heads" in capsys.readouterr().out
    cli(["vergleiche", "sprachmodell_mini", "sprachmodell_mamba"])
    assert "model.block.mixer" in capsys.readouterr().out
    cli(["zeige", "sprachmodell_mini", "--setze", "dropout=0.3", "--geaendert"])
    assert "0.3" in capsys.readouterr().out
    cli(["exportieren", "tabellen_klassifikation", "--format", "state_dict", "-o", str(tmp_path / "m")])
    assert (tmp_path / "m.pt").exists()


def test_kan_ffn_and_yarn():
    m = build_model({"name": "transformer", "vocab_size": 30, "d_model": 32, "n_layers": 2, "block": {"ffn": "kan"},
                     "positional": {"name": "rope", "scaling": "yarn", "factor": 4.0,
                                                                   "original_context": 4096}})
    x = torch.randint(0, 30, (2, 12))
    out = m(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    rope = m.positional
    assert rope.mscale > 1.0
    inv = rope._inv_freq(4, torch.device("cpu"))[1]
    base = 1.0 / (10000 ** (torch.arange(0, 4, 2).float() / 4))
    assert inv[0] == pytest.approx(base[0])          # hohe Frequenz bleibt
    assert inv[-1] < base[-1]                         # niedrige Frequenz wird interpoliert
