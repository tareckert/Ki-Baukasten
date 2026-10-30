"""Schalter An / Aus / Eigen für jede Einstellung."""

import pytest

from ki_baukasten import Project
from ki_baukasten.core import UNSET, ConfigError, off_value, on_value, state_of, switch_table
from ki_baukasten.core.switches import summary
from ki_baukasten.ui.cli import main as cli_main


def _param(cfg, path):
    return cfg.param_at(path).param


def test_derived_off_and_on_values():
    p = Project.from_preset("sprachmodell_mini")
    cfg = p.config
    assert off_value(_param(cfg, "training.regularization.mixup_alpha")) == 0.0
    assert on_value(_param(cfg, "training.regularization.mixup_alpha")) == 0.2
    assert off_value(_param(cfg, "training.ema.enabled")) is False
    assert off_value(_param(cfg, "training.curriculum.strategy")) == "none"
    assert off_value(_param(cfg, "training.scheduler")) == "constant"          # neutrale Komponente
    assert off_value(_param(cfg, "training.callbacks")) == []
    assert off_value(_param(cfg, "model.d_model")) is UNSET                    # Pflicht
    assert off_value(_param(cfg, "training.optimizer.lr")) is UNSET


def test_states_and_summary():
    p = Project.from_preset("sprachmodell_mini")
    ref = p.config.param_at("training.regularization.sam_rho")
    assert state_of(ref.param, 0.0) == "aus"
    assert state_of(ref.param, 0.05) == "an"
    assert state_of(ref.param, 0.11) == "eigen"
    counts = summary(p.config)
    assert counts["schaltbar"] > 50 and counts["pflicht"] > 20
    assert counts["an"] + counts["aus"] + counts["eigen"] == counts["schaltbar"] + counts["pflicht"]


def test_switch_roundtrip_remembers_custom_value_and_undo():
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.regularization.sam_rho", 0.11)
    p.switch("training.regularization.sam_rho", "aus")
    assert p.get("training.regularization.sam_rho") == 0.0
    p.switch("training.regularization.sam_rho", "eigen")          # gemerkter eigener Wert kommt zurück
    assert p.get("training.regularization.sam_rho") == 0.11
    p.switch("training.regularization.sam_rho", "standard")
    assert p.get("training.regularization.sam_rho") == 0.0
    assert p.undo()
    assert p.get("training.regularization.sam_rho") == 0.11


def test_set_understands_switch_words_and_patterns():
    p = Project.from_preset("sprachmodell_mini")
    p.set("**.dropout", "aus")
    assert all(p.get(path) == 0.0 for path in p.find("dropout"))
    p.set("training.ema", "an")                                    # Gruppe mit enabled-Schalter
    assert p.get("training.ema.enabled") is True
    p.set("training.scheduler", "aus")
    assert p.get("training.scheduler").name == "constant"
    p.set("training.scheduler", "an")
    assert p.get("training.scheduler").name == "warmup_cosine"
    with pytest.raises(ConfigError, match="Pflicht"):
        p.set("model.d_model", "aus")


def test_group_off_and_on_restores_exactly():
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.regularization.sam_rho", 0.07)
    p.set("training.regularization.l1", 1e-6)
    p.switch("training.regularization", "aus")
    assert p.get("training.regularization.sam_rho") == 0.0 and p.get("training.regularization.l1") == 0.0
    p.switch("training.regularization", "an")
    assert p.get("training.regularization.sam_rho") == 0.07 and p.get("training.regularization.l1") == 1e-6
    assert p.get("training.regularization.mixup_alpha") == 0.0    # war vorher aus → bleibt aus


def test_switch_table_filter_and_overview():
    p = Project.from_preset("tabellen_klassifikation")
    off = switch_table(p.config, "training.regularization", state="aus")
    assert off and all(i.state == "aus" for i in off)
    text = p.switch_overview()
    assert "Eingeschaltet" in text and "Ausgeschaltet" in text


def test_switched_project_still_trains():
    p = Project.from_preset("tabellen_klassifikation", **{"training.max_steps": 3, "training.verbose": False})
    p.set("**.dropout", "aus")
    p.switch("training.callbacks", "aus")
    p.switch("training.gradient_clipping.mode", "aus")
    hist = p.train()
    assert hist is not None


def test_cli_schalter(capsys, tmp_path):
    out = tmp_path / "s.json"
    cli_main(["schalter", "tabellen_klassifikation", "--aus", "dropout", "--an", "training.ema",
              "--speichern", str(out)])
    text = capsys.readouterr().out
    assert "aus: **.dropout" in text and out.exists()
    p = Project.from_file(out)
    assert p.get("training.ema.enabled") is True
