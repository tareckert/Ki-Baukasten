"""Runde 6: Einstellungen, die verworfen/überschrieben wurden oder nichts bewirkten, und der Weg zur besten KI
(Diagnose nach dem Training, automatisches Verbessern)."""

import csv
import random
import warnings

import pytest
import torch

from ki_baukasten import Project, registry
from ki_baukasten.core.errors import ConfigError

QUIET = {"training.verbose": False, "deployment.show_after_training": False}


def _tab(steps=40, **over):
    return Project.from_preset("tabellen_klassifikation", **{**QUIET, "training.max_steps": steps, **over})


def _csv(path, n=200):
    rng = random.Random(0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "klasse"])
        for _ in range(n):
            x, y = rng.random(), rng.random()
            w.writerow([x, y, "a" if x + y > 1 else "b"])


# ---------------------------------------------------------------- überschriebene / verworfene Änderungen
def test_project_auto_keeps_explicit_wishes(tmp_path):
    _csv(tmp_path / "k.csv")
    p = Project.auto(tmp_path / "k.csv", train=False, **{"training.test_split": 0.0, "training.verbose": False})
    assert p.config.training.test_split == 0.0          # vorher von der Perfektions-Automatik auf 0.15 gesetzt
    assert "testmenge" not in {r.technique.key for r in p.techniques(status="an")}


def test_direct_config_change_rebuilds_model():
    warnings.simplefilter("ignore")
    p = _tab(20)
    p.train()
    before = sum(t.numel() for t in p.model.parameters())
    p.config.model.args.hidden_dims = [32]               # direkt statt p.set → früher stillschweigend ignoriert
    p.train()
    assert sum(t.numel() for t in p.model.parameters()) < before
    assert any("neu gebaut" in n for n in p.notes)
    q = _tab(10)
    q.train()
    q.train()
    assert not any("neu gebaut" in n for n in q.notes)   # ohne Änderung bleibt das Modell


def test_save_load_keeps_results_and_exact_config(tmp_path):
    warnings.simplefilter("ignore")
    p = _tab(60)
    p.enable("kalibrierung", "testmenge")
    p.train()
    p.test()
    p.baseline()
    p.save(tmp_path / "a")
    q = Project.load(tmp_path / "a")
    assert q.last_test == p.last_test and q.last_baseline == p.last_baseline and q.last_calibration
    assert q.config.to_dict() == Project.from_file(tmp_path / "a" / "projekt.json").config.to_dict()
    assert q.model_card() == p.model_card()
    q.save(tmp_path / "b")
    assert Project.load(tmp_path / "b").model_card() == p.model_card()


def test_pbt_keeps_trained_model_and_its_settings():
    warnings.simplefilter("ignore")
    p = _tab(60)
    p.enable("pbt")
    p.update({"workflow.search.trials": 4, "workflow.search.steps_per_trial": 15})
    result = p.tune()
    assert p.model is not None and p.trainer is not None            # vorher durch update() verworfen
    for path, value in result.pbt_best["params"].items():
        assert p.config.get(path) == value
    assert "val_loss" in p.evaluate()


@pytest.mark.parametrize("tech", ["foederiert", "model_soup"])
def test_run_tests_after_federated_and_soup(tech):
    warnings.simplefilter("ignore")
    p = _tab(30)
    p.enable(tech, "testmenge")
    if tech == "model_soup":
        p.set("workflow.soup_runs", 2)
    report = p.run()
    assert p.test_data is not None and "test" in report and "basis" in report


def test_scheduler_techniques_keep_warmup_and_floor():
    p = Project.from_preset("sprachmodell_mini")
    p.enable("plateau_lr")
    s = p.config.training.scheduler
    assert s.name == "plateau" and s.args.warmup_ratio > 0 and s.args.min_lr_ratio == 0.1
    assert {r.technique.key: r.status for r in p.techniques()}["warmup"] == "an"
    p.disable("plateau_lr")
    assert p.config.training.scheduler.name == "warmup_cosine"


def test_mc_dropout_and_tta_combine():
    p = Project.from_preset("bildklassifikation_cnn")
    status = lambda: {r.technique.key: r.status for r in p.techniques()}   # noqa: E731
    p.enable("mc_dropout")
    p.enable("tta")
    assert p.config.inference.uncertainty == "beides" and status()["mc_dropout"] == status()["tta"] == "an"
    p.disable("mc_dropout")
    assert p.config.inference.uncertainty == "tta"
    p.disable("tta")
    assert p.config.inference.uncertainty == "keine"


def test_technique_conflict_is_reported():
    p = Project.from_preset("sprachmodell_mini")
    p.enable("lion")
    p.enable("muon")
    assert any("lion" in n and "ausgeschaltet" in n for n in p.notes)


def test_auto_lr_never_overwrites_an_explicit_learning_rate():
    warnings.simplefilter("ignore")
    p = _tab(20)
    p.enable("auto_lernrate")
    p.train()
    found = p.config.training.optimizer.args.lr
    p.train()                                            # gleiches Modell → LR-Finder läuft nicht erneut
    assert sum("LR-Finder" in n for n in p.notes) == 1 and p.config.training.optimizer.args.lr == found
    p.set("training.optimizer.lr", 0.0123)               # ausdrücklicher Wunsch …
    p.train()
    assert p.config.training.optimizer.args.lr == 0.0123  # … gewinnt gegen die Automatik
    assert any("ausdrücklich gesetzter Wert" in n for n in p.notes)


# ---------------------------------------------------------------- wirkungslose Einstellungen
def test_inherited_settings_without_effect_are_gone():
    from ki_baukasten.training.optimizers import AdamW8bitConfig, NAdamConfig
    from ki_baukasten.training.schedulers import OneCycleConfig

    for cls in (NAdamConfig, AdamW8bitConfig):
        assert "amsgrad" not in cls.params() and "fused" not in cls.params()
    assert not {"warmup_type", "warmup_start_factor", "min_lr_ratio"} & set(OneCycleConfig.params())
    p = Project.from_preset("tabellen_klassifikation")
    with pytest.warns(UserWarning, match="amsgrad"):            # ältere Datei lädt weiter
        p.config.update({"training": {"optimizer": {"name": "nadam", "args": {"amsgrad": False}}}})
    with pytest.raises(ConfigError):
        p.set("training.optimizer.amsgrad", True)


def test_inactive_settings_are_reported():
    from ki_baukasten.analysis.effects import inactive_settings

    for name in registry("preset").names():                 # Vorlagen selbst enthalten keinen Ballast
        assert inactive_settings(Project.from_preset(name).config) == [], name
    p = _tab(20)
    assert p.inactive_settings() == []                      # max_steps neben epochs der Vorlage: kein Hinweis
    p.set("memory.lora.rank", 16)
    p.set("training.ema.decay", 0.99)
    p.set("training.epochs", 5)
    found = {s.path: s for s in p.inactive_settings()}
    assert set(found) == {"memory.lora.rank", "training.ema.decay", "training.epochs"}
    assert "memory.lora.enabled" in found["memory.lora.rank"].fix
    p.enable("lora")
    assert "memory.lora.rank" not in {s.path for s in p.inactive_settings()}
    warnings.simplefilter("ignore")
    p.train()
    assert any("wirkt nicht" in n for n in p.notes)


# ---------------------------------------------------------------- Korrekturen in Komponenten
def test_sequence_metrics_accept_token_ids():
    t = torch.tensor([[1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 6, 0]])
    part = torch.tensor([[1, 2, 3, 9, 9, 0], [2, 3, 4, 5, 6, 0]])
    for name in ("bleu", "token_accuracy", "exact_match"):
        f = registry("metric").build(name)
        assert f(t.clone(), t) == pytest.approx(1.0)
        assert f(torch.nn.functional.one_hot(t, 8).float(), t) == pytest.approx(1.0)
        assert f(part, t) < 1.0


def test_plateau_scheduler_follows_monitor_direction():
    def scale(monitor, values):
        p = torch.nn.Parameter(torch.zeros(1))
        s = registry("scheduler").build({"name": "plateau", "args": {"monitor": monitor, "patience": 1}},
                                        torch.optim.SGD([p], lr=1.0), 100)
        for v in values:
            s.observe({"val_loss": 1.0, monitor: v})
        return s.scale

    assert scale("val_accuracy", [0.5, 0.6, 0.7, 0.8, 0.9]) == 1.0      # Genauigkeit steigt → nicht senken
    assert scale("val_accuracy", [0.9, 0.8, 0.7, 0.6, 0.5]) < 1.0
    assert scale("val_loss", [1.0, 0.9, 0.8, 0.7, 0.6]) == 1.0
    assert scale("val_loss", [0.6, 0.7, 0.8, 0.9, 1.0]) < 1.0


def test_random_erasing_batched_rectangles():
    from types import SimpleNamespace

    from ki_baukasten.training.trainer import Trainer

    aug = SimpleNamespace(hflip=False, vflip=False, crop_padding=0, random_erasing=1.0)
    fake = SimpleNamespace(cfg=SimpleNamespace(augmentation=aug))
    x = torch.rand(6, 3, 12, 12) + 0.5
    y, _ = Trainer._augment_images(fake, x, None)
    erased = (y == 0).all(1)
    for i in range(6):
        rows, cols = erased[i].any(1).sum(), erased[i].any(0).sum()
        assert 0 < erased[i].sum() == rows * cols                      # genau ein Rechteck je Bild
    assert torch.equal(y[~erased[:, None].expand_as(y)], x[~erased[:, None].expand_as(x)])


def test_eval_data_falls_back_to_test_set():
    p = _tab(5)
    p.test_data = {"inputs": torch.randn(4, 3), "targets": torch.zeros(4, dtype=torch.long)}
    p.trainer = None
    p._bundle = type("B", (), {"val": None, "train": None, "test": None, "info": {}})()
    assert p._eval_data() is p.test_data


# ---------------------------------------------------------------- Diagnose und automatisches Verbessern
def test_diagnosis_finds_too_small_learning_rate_and_improve_fixes_it():
    warnings.simplefilter("ignore")
    p = _tab(150, **{"training.optimizer.lr": 1e-6})
    p.train()
    report = p.diagnose()
    assert "lernt_kaum" in {f.key for f in report.findings}
    restore = [a for a in report.actions() if a.changes and "training.optimizer.lr" in a.changes]
    assert restore and restore[0].changes["training.optimizer.lr"] == 0.003       # bewährter Wert der Vorlage
    result = p.improve(rounds=1)
    assert result["runden"] and result["runden"][0]["behalten"]
    assert result["ende"] > result["start"] + 0.2
    assert p.config.training.optimizer.args.lr == 0.003
    assert p.test_data is None                          # Verbessern ändert die Datenaufteilung nicht
    q = Project(p.config.copy())                        # ohne Vorlage als Bezug → LR-Finder vorschlagen
    q.update({"training.optimizer.lr": 1e-6, "training.max_steps": 150, "training.callbacks": []})
    q.train()
    assert "auto_lernrate" in {a.technique for a in q.diagnose().actions()}


def test_diagnosis_finds_real_overfitting():
    warnings.simplefilter("ignore")
    p = _tab(1500, **{"data.n_samples": 120, "data.noise": 0.3, "model.hidden_dims": [256, 256],
                      "model.layer.dropout": 0.0, "training.callbacks": [], "training.eval_every_steps": 100,
                      "training.optimizer.weight_decay": 0.0})
    p.train()
    found = {f.key: f for f in p.diagnose().findings}
    assert "ueberanpassung" in found
    assert {"early_stopping", "dropout"} & {a.technique for a in found["ueberanpassung"].actions}


def test_improve_rolls_back_when_worse(monkeypatch):
    warnings.simplefilter("ignore")
    p = _tab(300)
    p.train()
    lr = p.config.training.optimizer.args.lr
    model = p.model

    def harmful(self, report=None, max_actions=4, include_hints=True):
        self.update({"training.optimizer.lr": 5.0, "training.max_steps": 30})
        return ["absichtlich schlecht"]

    monkeypatch.setattr(Project, "apply_diagnosis", harmful)
    result = p.improve(rounds=1)
    assert result["runden"][0]["behalten"] is False
    assert p.config.training.optimizer.args.lr == lr and p.model is model     # zurück zum besseren Stand


def test_diagnosis_healthy_run_has_no_problems():
    warnings.simplefilter("ignore")
    p = _tab(600)
    p.enable("testmenge")
    p.train()
    assert p.diagnose().ok


def test_diagnosis_works_for_every_kind_of_model():
    warnings.simplefilter("ignore")
    for name in ("sprachmodell_mini", "gan_2d", "anomalie_autoencoder", "rl_cartpole_ppo", "diffusion_2d"):
        p = Project.from_preset(name, **{**QUIET, "training.max_steps": 6, "training.eval_every_steps": 3})
        p.train()
        assert isinstance(str(p.diagnose()), str)


def test_cli_diagnose(capsys):
    from ki_baukasten.ui.cli import main

    main(["diagnose", "tabellen_klassifikation", "--setze", "training.max_steps=20", "--setze",
          "training.verbose=false"])
    assert "Diagnose" in capsys.readouterr().out
