"""Techniken-Katalog (an/aus mit Status) und Ziel-Assistent (Empfehlen/Abraten)."""

import pytest

from ki_baukasten import Project
from ki_baukasten.analysis.goals import HEAVY, PROFILES, evaluate_goals
from ki_baukasten.core import registry
from ki_baukasten.techniques import BY_KEY, CATALOG, CATEGORIES
from ki_baukasten.ui.cli import main as cli_main

FAST_PRESETS = ["tabellen_klassifikation", "sprachmodell_mini", "bildklassifikation_cnn", "gan_2d"]


def test_catalog_is_consistent():
    assert len(BY_KEY) == len(CATALOG) >= 80
    for t in CATALOG:
        assert t.category in CATEGORIES and all(c in CATEGORIES for c in t.also), t.key
        assert t.tooltip.what and t.title
    used = {t.category for t in CATALOG}
    assert used == set(CATEGORIES)                      # jede Kategorie hat Techniken
    applicable = {t.key for pre in FAST_PRESETS for r in Project.from_preset(pre).techniques()
                  if r.status != "nicht_anwendbar" for t in [r.technique]}
    assert len(applicable) >= len(CATALOG) - 6          # fast alles passt zu mindestens einer Vorlage


@pytest.mark.parametrize("preset", FAST_PRESETS)
def test_enable_disable_restores_previous_state(preset):
    for t in CATALOG:
        p = Project.from_preset(preset)
        before = p.config.to_dict()
        state = next(r.status for r in p.techniques() if r.technique.key == t.key)
        if state == "nicht_anwendbar":
            continue
        p.enable(t.key)
        assert next(r.status for r in p.techniques() if r.technique.key == t.key) == "an", t.key
        p.disable(t.key)
        assert p.config.to_dict() == before, t.key       # exakt der vorherige Zustand


def test_disable_without_memory_switches_off_and_undo():
    p = Project.from_preset("tabellen_klassifikation")
    p.enable("mixup", "early_stopping", "spike_schutz")
    q = Project(p.config.copy())                         # z. B. nach dem Laden: kein Merkzettel
    q.disable("mixup", "early_stopping")
    assert q.get("training.regularization.mixup_alpha") == 0.0
    assert "early_stopping" not in [c.name for c in q.get("training.callbacks")]
    assert q.undo() and q.undo()
    assert q.get("training.regularization.mixup_alpha") == 0.2


def test_not_applicable_techniques_are_refused():
    p = Project.from_preset("tabellen_klassifikation")
    states = {r.technique.key: r.status for r in p.techniques()}
    assert states["cutmix"] == "nicht_anwendbar" and states["stream_ram"] == "nicht_anwendbar"
    with pytest.raises(Exception, match="passt nicht"):
        p.enable("cutmix")
    with pytest.raises(Exception, match="Unbekannte Technik"):
        p.enable("gibt_es_nicht")


def test_overview_and_cli(capsys, tmp_path):
    p = Project.from_preset("sprachmodell_mini")
    text = p.technique_overview()
    assert all(title in text for title, _ in CATEGORIES.values())
    out = tmp_path / "t.json"
    cli_main(["techniken", "tabellen_klassifikation", "--an", "mixup", "--an", "ema", "--speichern", str(out)])
    assert "an:  mixup" in capsys.readouterr().out
    q = Project.from_file(out)
    assert q.get("training.ema.enabled") and q.get("training.regularization.mixup_alpha") == 0.2


def test_enabled_techniques_train():
    p = Project.from_preset("tabellen_klassifikation", **{"training.max_steps": 4, "training.verbose": False,
                                                           "deployment.show_after_training": False})
    p.enable("gradient_clipping", "spike_schutz", "ema", "mixup", "weight_decay", "lookahead",
             "gradient_centralization", "swa", "oom_schutz", "klassen_ausgleich", "label_smoothing")
    assert p.train()


# --------------------------------------------------------------------------- Ziele

def test_profiles_cover_all_presets():
    assert set(PROFILES) == set(registry("preset").names())


@pytest.mark.parametrize("goals, expected", [
    ({"description": "Bilder auf dem Handy in Echtzeit erkennen", "data_size": "klein"}, "bildklassifikation_mobil"),
    ({"task": "text_erzeugen", "sequence_length": "sehr_lang", "priorities": ["lange_kontexte"]},
     ("sprachmodell_mamba", "sprachmodell_hybrid")),
    ({"task": "tabelle_klassifizieren", "priorities": ["einfachheit", "schnelles_training"]},
     "tabellen_klassifikation"),
    ({"description": "Betrug in Transaktionen finden", "labels": "keine"}, "anomalie_autoencoder"),
    ({"task": "steuerung_rl", "priorities": ["robustheit"]}, ("rl_cartpole_ppo", "rl_pendel_kontinuierlich")),
    ({"task": "bild_erzeugen", "priorities": ["genauigkeit", "robustheit"]}, "diffusion_2d"),
    ({"task": "text_erzeugen", "pretrained": True, "priorities": ["wenig_speicher"]}, "sprachmodell_qlora"),
])
def test_goal_recommendations(goals, expected):
    report = evaluate_goals(goals)
    best = report.recommended[0].preset
    assert best in (expected if isinstance(expected, tuple) else (expected,)), str(report)


def test_discouraged_with_reasons():
    report = evaluate_goals({"task": "bild_klassifizieren", "data_size": "winzig", "target": "smartphone"})
    vit = next(s for s in report.discouraged if s.preset == "bildklassifikation_vit")
    assert any("mehr Daten" in r for r in vit.reasons)
    moe = evaluate_goals({"task": "text_erzeugen", "target": "smartphone", "priorities": ["wenig_speicher"]})
    assert "sprachmodell_moe" in [s.preset for s in moe.discouraged]


def test_from_goals_set_goals_and_persistence(tmp_path):
    p = Project.from_goals(task="text_klassifizieren", priorities=["datenschutz", "robustheit"], data_size="klein",
                           description="Spam-Filter fürs Handy")
    assert p.config.goals.target == "smartphone" and p.config.deployment.main_target == "smartphone"
    enabled = {r.technique.key for r in p.techniques(status="an")}
    assert {"gradient_clipping", "spike_schutz", "early_stopping"} <= enabled
    assert not (enabled & HEAVY)                          # DP-SGD/föderiert nicht automatisch
    assert "dp_sgd" in [k for k, _ in p.goal_report.techniques]
    check = p.goal_check()
    assert check.current is not None and "✓" in check.current_verdict or "empfohlen" in check.current_verdict
    changed = p.set_goals(task="bild_klassifizieren")
    assert "passt nicht" in changed.current_verdict
    d = p.save(tmp_path / "ziel")
    assert Project.load(d).config.goals.task == "bild_klassifizieren"


def test_cli_ziele(capsys, tmp_path):
    out = tmp_path / "neu.json"
    cli_main(["ziele", "--text", "Zeitreihe von Sensoren in Echtzeit vorhersagen", "--erstellen", str(out)])
    text = capsys.readouterr().out
    assert "Empfohlen" in text and out.exists()
    assert Project.from_file(out).config.goals.task == "zeitreihe_prognose"
    cli_main(["ziele", "--liste"])
    assert "text_erzeugen" in capsys.readouterr().out
