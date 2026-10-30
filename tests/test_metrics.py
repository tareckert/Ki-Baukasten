"""Tests der Metriken, der Testmenge, der Klassengewichte und der automatischen Richtung (min/max)."""

import math

import pytest
import torch

from ki_baukasten import Project, Trainer, TrainingConfig, build_model
from ki_baukasten.core import ComponentSpec
from ki_baukasten.training.metrics import (AUTO, Collector, METRICS, SeqPred, compute_metrics, direction, task_of,
                                           target_inverse)
from ki_baukasten.training.paradigms import auto_loss, is_segmentation


def m(name, **args):
    return ComponentSpec.from_value("metric", {"name": name, **args}).build()


def onehot_logits(classes, c):
    return torch.nn.functional.one_hot(torch.tensor(classes), c).float() * 5


def test_classification_metrics_known_values():
    target = torch.tensor([0, 0, 0, 1, 1, 2])
    pred = onehot_logits([0, 0, 1, 1, 1, 0], 3)
    assert m("accuracy")(pred, target) == pytest.approx(4 / 6)
    # Klasse 0: P=2/3 R=2/3; Klasse 1: P=2/3 R=1; Klasse 2: P=0 R=0
    assert m("recall")(pred, target) == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert m("precision")(pred, target) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m("balanced_accuracy")(pred, target) == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert m("precision", average="micro")(pred, target) == pytest.approx(4 / 6)
    assert -1 <= m("mcc")(pred, target) <= 1
    assert m("top_k_accuracy", k=3)(pred, target) == 1.0
    perfect = onehot_logits(target.tolist(), 3)
    assert m("f1")(perfect, target) == pytest.approx(1.0)
    assert m("mcc")(perfect, target) == pytest.approx(1.0)


def test_auroc_ranks_and_ties():
    y = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[2.0, -2], [1, -1], [-1, 1], [-2, 2]])
    assert m("auroc")(good, y) == pytest.approx(1.0)
    assert m("auroc")(good.flip(1), y) == pytest.approx(0.0)
    assert m("auroc")(torch.zeros(4, 2), y) == pytest.approx(0.5)


def test_calibration_metrics():
    y = torch.randint(0, 2, (2000,))
    confident_wrong = onehot_logits((1 - y).tolist(), 2) * 4
    assert m("ece")(confident_wrong, y) > 0.9
    assert m("brier")(confident_wrong, y) > 1.5
    assert m("log_loss")(confident_wrong, y) > 5


def test_regression_metrics():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = torch.tensor([1.0, 2.0, 3.0, 6.0])
    assert m("mae")(p, t) == pytest.approx(0.5)
    assert m("mse")(p, t) == pytest.approx(1.0)
    assert m("rmse")(p, t) == pytest.approx(1.0)
    assert m("max_error")(p, t) == pytest.approx(2.0)
    assert m("r2")(t, t) == pytest.approx(1.0)
    assert m("mape")(p, t) == pytest.approx(12.5)
    assert m("smape")(t, t) == pytest.approx(0.0)


def test_sequence_metrics_and_compact_predictions():
    v = 7
    target = torch.randint(1, v, (4, 5))
    target[0, 3:] = -100
    uniform = torch.zeros(4, 5, v)
    assert m("perplexity")(uniform, target) == pytest.approx(v, rel=1e-4)
    exact = torch.nn.functional.one_hot(target.clamp_min(0), v).float() * 10
    assert m("token_accuracy")(exact, target) == 1.0
    assert m("exact_match")(exact, target) == 1.0
    assert m("bleu")(exact, target) == pytest.approx(1.0)
    compact = SeqPred.from_logits(exact, target)                 # gleiche Werte ohne volle Logits
    assert m("token_accuracy")(compact, target) == 1.0
    assert m("bleu")(compact, target) == pytest.approx(1.0)


def test_segmentation_and_multilabel():
    mask = torch.randint(0, 3, (2, 8, 8))
    logits = torch.nn.functional.one_hot(mask, 3).permute(0, 3, 1, 2).float()
    assert is_segmentation(logits, mask)
    assert m("iou")(logits, mask) == pytest.approx(1.0)
    assert m("dice")(logits, mask) == pytest.approx(1.0)
    assert float(auto_loss(logits * 10, mask)) < 1e-3              # Kreuzentropie über die Kanal-Achse
    labels = torch.tensor([[1.0, 0, 1], [0, 1, 0]])
    assert m("multilabel_f1")(labels * 4 - 2, labels) == pytest.approx(1.0)
    assert m("subset_accuracy")(-(labels * 4 - 2), labels) == 0.0


def test_task_detection_and_auto_lists():
    assert task_of(torch.randn(4, 3), torch.tensor([0, 1, 2, 0])) == "klassifikation"
    assert task_of(torch.randn(4, 1), torch.randn(4, 1)) == "regression"
    assert task_of(torch.randn(2, 3, 8, 8), torch.randint(0, 3, (2, 8, 8))) == "segmentierung"
    assert task_of(torch.randn(2, 5, 9), torch.randint(0, 9, (2, 5)), "sprache") == "sprache"
    for names in AUTO.values():
        for n in names:
            assert n in METRICS


def test_direction():
    assert direction("val_loss") == "min"
    assert direction("val_accuracy") == "max"
    assert direction("test_f1") == "max"
    assert direction("val_rmse") == "min"
    assert direction("val_genauigkeit") == "max"        # Alias


def test_collector_pads_sequences_and_limits_samples():
    col = Collector(max_items=5)
    col.add(torch.randn(3, 4, 6), torch.randint(0, 6, (3, 4)), "sprache")
    col.add(torch.randn(3, 7, 6), torch.randint(0, 6, (3, 7)), "sprache")
    pred, target = col.result()
    assert isinstance(pred, SeqPred) and pred.ids.shape == (5, 7) and target.shape == (5, 7)
    assert (target[:3, 4:] == -100).all()
    out = col.metrics([], auto=True)
    assert set(out) == {"val_perplexity", "val_token_accuracy"}


def test_compute_metrics_skips_unfitting():
    specs = [ComponentSpec.from_value("metric", n) for n in ("accuracy", "iou")]
    out = compute_metrics(specs, torch.randn(4, 1), torch.randn(4, 1))
    assert "val_iou" not in out


def test_target_inverse():
    inv = target_inverse({"art": "standard", "mittel": 10.0, "std": 2.0})
    p, t = inv(torch.tensor([0.0, 1.0]), torch.tensor([1.0, -1.0]))
    assert p.tolist() == [10.0, 12.0] and t.tolist() == [12.0, 8.0]
    assert target_inverse(None) is None


def _xy(n=300):
    x = torch.randn(n, 4)
    y = (x[:, 0] > 0.8).long()          # unausgewogen: ~20 % Klasse 1
    return x, y


def test_trainer_reports_configured_and_auto_metrics():
    x, y = _xy()
    model = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden_dims": [16]}})
    tr = Trainer(model, TrainingConfig(epochs=3, verbose=False, metrics=["f1", "mcc"]))
    tr.fit((x, y))
    out = tr.evaluate((x, y))
    assert {"val_loss", "val_f1", "val_mcc"} <= set(out) and "val_auroc" not in out
    tr2 = Trainer(model, TrainingConfig(epochs=1, verbose=False, auto_metrics=False))
    assert set(tr2.evaluate((x, y))) == {"val_loss", "val_accuracy"}
    out3 = Trainer(model, TrainingConfig(epochs=1, verbose=False)).evaluate((x, y), prefix="test_")
    assert "test_balanced_accuracy" in out3 and "test_auroc" in out3


def test_class_weights_change_loss():
    x, y = _xy()
    model = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden_dims": [8]}})
    tr = Trainer(model, TrainingConfig(epochs=1, verbose=False, class_weights="ausgeglichen", validation_split=0.0))
    tr.fit((x, y))
    w = tr.class_weight
    counts = torch.bincount(y).float()
    assert w is not None and w[1] > w[0]
    assert torch.allclose(w, len(y) / (2 * counts))
    tr_w = Trainer(model, TrainingConfig(epochs=1, verbose=False, class_weights="wurzel", validation_split=0.0))
    tr_w.fit((x, y))
    assert torch.allclose(tr_w.class_weight, (len(y) / (2 * counts)).sqrt())


def test_early_stopping_auto_direction_for_accuracy():
    x, y = _xy()
    model = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden_dims": [16]}})
    cfg = TrainingConfig(epochs=4, verbose=False, callbacks=[{"name": "early_stopping", "monitor": "val_f1",
                                                              "patience": 10}])
    tr = Trainer(model, cfg)
    tr.fit((x, y))
    es = tr.callbacks[0]
    best = max(h["val_f1"] for h in tr.history if "val_f1" in h)
    assert es.best == pytest.approx(best)


def test_project_test_split_holds_out_and_stays_fixed():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.max_steps", 10)
    p.set("training.verbose", False)
    with pytest.raises(Exception):
        p.test()
    p.turn_on("training.test_split")
    assert p.config.training.test_split == pytest.approx(0.15)
    p.train()
    n_total = len(p.data().train["inputs"])
    held = p.test_data
    assert len(held) == round(n_total * 0.15)
    out = p.test()
    assert "test_loss" in out and "test_f1" in out
    p.train()                                         # Neutrainieren: gleiche Testmenge, kein Leck
    assert p.test_data is held
    assert len(p.trainer.history) > 0 and not math.isnan(out["test_loss"])
