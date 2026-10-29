"""Tests für Anforderungs-Katalog, Perfektions-Check, neue Techniken, Ablauf-Prüfungen und CLI-Befehle."""

import pytest
import torch

from ki_baukasten import Project
from ki_baukasten.requirements import (BY_KEY, CATALOG, DOMAINS, IMPORTANCE, PHASES, coverage, perfection_check,
                                       project_domains, requirements_markdown)
from ki_baukasten.techniques import BY_KEY as TECH
from ki_baukasten.ui.cli import main as cli_main


def test_catalog_is_consistent():
    assert len(CATALOG) == len(BY_KEY) >= 45
    for r in CATALOG:
        assert r.phase in PHASES and r.importance in IMPORTANCE, r.key
        assert set(r.domains) <= set(DOMAINS), r.key
        assert all(k in TECH for k in r.techniques), r.key        # jede genannte Technik existiert
        assert r.why and r.how
    cov = coverage()
    gaps = {r.key for r, ok in cov["zeilen"] if not ok}
    assert gaps == {"objekterkennung", "empfehlungssystem"}        # ehrlich benannte Lücken
    assert all(BY_KEY[k].gap for k in gaps)
    text = requirements_markdown()
    assert "Ehrliche Lücken" in text and all(t in text for t in PHASES.values())


@pytest.mark.parametrize("preset,expect", [
    ("tabellen_klassifikation", {"tabellen", "klassifikation", "datensatz"}),
    ("zeitreihen_lstm", {"zeitreihen", "regression"}),
    ("sprachmodell_sft", {"sprache", "generierung"}),
    ("segmentierung_unet", {"bild", "segmentierung"}),
    ("rl_cartpole_ppo", {"rl"}),
])
def test_project_domains(preset, expect):
    assert expect <= set(project_domains(Project.from_preset(preset)))


def test_perfection_check_and_apply_raise_score():
    p = Project.from_preset("tabellen_klassifikation")
    before = p.perfection_check()
    assert 0 < before.score < 100 and "Perfektions-Check" in str(before)
    keys = {c.requirement.key for c in before.checks}
    assert "aufteilung_zeit" not in keys and "objekterkennung" not in keys      # nur passende Anforderungen
    switch = before.switchable()
    assert "testmenge" in switch and "kalibrierung" in switch
    done = p.apply_perfection()
    assert set(done) == set(switch)
    after = p.perfection_check()
    assert after.score > before.score
    assert p.config.training.test_split > 0 and p.config.inference.auto_calibrate
    for _ in done:
        p.undo()
    assert p.perfection_check().score == pytest.approx(before.score)


def test_steps_become_active_after_running():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.max_steps", 40)
    p.set("training.verbose", False)
    p.apply_perfection()
    p.set("training.callbacks", [])                    # ohne Checkpoint-Dateien im Arbeitsordner
    p.set("training.logging.backends", ["console"])
    p.train()
    p.test()
    p.baseline()
    report = perfection_check(p)
    status = {c.requirement.key: c.status for c in report.checks}
    assert status["testbewertung"] == "aktiv" and status["baseline"] == "aktiv"
    assert status["kalibrierung"] == "aktiv"            # auto_calibrate nach train()
    assert p.last_calibration and "temperatur" in p.last_calibration


def test_gaps_shown_for_detection_goals():
    p = Project.from_preset("bildklassifikation_cnn")
    p.set("goals.description", "Objekterkennung von Bauteilen mit Bounding Boxes")
    report = p.perfection_check()
    missing = [c.requirement.key for c in report.missing()]
    assert missing == ["objekterkennung"]


def test_new_techniques_round_trip():
    p = Project.from_preset("tabellen_klassifikation")
    before = p.config.to_dict()
    for key in ("testmenge", "kalibrierung", "conformal", "enthaltung", "klassengewichte", "adversarial_training",
                "robustheitspruefung", "fortsetzen", "stratifiziert"):
        p.enable(key)
    assert p.config.training.adversarial.method == "pgd" and p.config.training.resume_from == "auto"
    for key in ("testmenge", "kalibrierung", "conformal", "enthaltung", "klassengewichte", "adversarial_training",
                "robustheitspruefung", "fortsetzen", "stratifiziert"):
        p.disable(key)
    assert p.config.to_dict() == before
    lm = Project.from_preset("sprachmodell_mini")
    lm.enable("beam_search", "keine_wiederholungen")
    assert lm.config.generation.num_beams == 4 and lm.config.generation.no_repeat_ngram_size == 3
    states = {r.technique.key: r.status for r in Project.from_preset("tabellen_klassifikation").techniques()}
    assert states["beam_search"] == "nicht_anwendbar" and states["tta"] == "nicht_anwendbar"


def test_workflow_runs_quality_steps():
    p = Project.from_preset("tabellen_klassifikation")
    p.update({"training.max_steps": 30, "training.verbose": False, "training.test_split": 0.15,
              "workflow.robustness_check": True})
    report = p.run()
    assert {"datenpruefung", "training", "test", "basis", "robustheit"} <= set(report)
    assert report["test"]["test_loss"] > 0


def test_generation_defaults_from_project_config():
    p = Project.from_preset("sprachmodell_mini")
    p.set("model.n_layers", 1)
    p.set("model.d_model", 32)
    p.set("generation.temperature", 0.0)
    a = p.generate("Der", max_new_tokens=5)
    b = p.generate("Der", max_new_tokens=5)
    assert a == b                                       # gierig laut Projekt-Einstellung
    model = p.model
    p.set("generation.num_beams", 2)
    assert p.model is model                             # Generierungs-Einstellungen behalten das Modell


def test_baseline_and_error_analysis():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.max_steps", 60)
    p.set("training.verbose", False)
    p.train()
    base = p.baseline()
    assert base["basis_auroc"] == pytest.approx(0.5) and "modell_f1" in base
    ea = p.error_analysis(top=3)
    cm = torch.tensor(ea["konfusionsmatrix"])
    assert cm.sum() == len(p.trainer.val_data) and len(ea["sicherste_fehler"]) <= 3


def test_cli_commands(capsys, tmp_path):
    cli_main(["perfektion", "tabellen_klassifikation", "--stufe", "empfohlen"])
    assert "Perfektions-Check" in capsys.readouterr().out
    out = tmp_path / "p.json"
    cli_main(["perfektion", "tabellen_klassifikation", "--anwenden", "--speichern", str(out)])
    assert "Eingeschaltet:" in capsys.readouterr().out
    assert Project.from_file(out).config.training.test_split > 0
    cat = tmp_path / "A.md"
    cli_main(["perfektion", "--katalog", "-o", str(cat)])
    assert "Anforderungen" in cat.read_text(encoding="utf-8")
    cli_main(["datenpruefung", "tabellen_klassifikation"])
    assert "Datenprüfung" in capsys.readouterr().out
    cli_main(["bewerten", "tabellen_klassifikation", "--setze", "training.max_steps=20",
              "--setze", "training.verbose=false"])
    text = capsys.readouterr().out
    assert "Validierung:" in text and "Basis:" in text and "Konfusionsmatrix" in text


def test_cli_saved_project_card_and_drift(capsys, tmp_path):
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.max_steps", 10)
    p.set("training.verbose", False)
    p.train()
    folder = p.save(tmp_path / "ki")
    torch.save({"inputs": p.data().train["inputs"][:200] + 2.0}, tmp_path / "neu.pt")
    cli_main(["modellkarte", str(folder)])
    assert "# Modellkarte" in capsys.readouterr().out
    cli_main(["drift", str(folder), str(tmp_path / "neu.pt")])
    assert "Drift-Bericht" in capsys.readouterr().out
    cli_main(["bewerten", str(folder)])
    assert "Validierung:" in capsys.readouterr().out
