"""Tests für Kalibrierung, Schwellen, Enthaltung, Conformal Prediction, Unsicherheit, Erklärbarkeit und Fairness."""

import pytest
import torch
import torch.nn.functional as F

from ki_baukasten import Project
from ki_baukasten.analysis.explain import gradient_x_input, integrated_gradients, occlusion, permutation_importance
from ki_baukasten.analysis.fairness import fairness_report
from ki_baukasten.analysis.inference import (best_threshold, conformal_quantile, conformal_scores, fit_temperature,
                                             prediction_sets, tta_variants)
from ki_baukasten.core import ComponentSpec


def test_temperature_scaling_fixes_overconfidence():
    y = torch.randint(0, 3, (2000,))
    logits = (F.one_hot(y, 3).float() * 2 + torch.randn(2000, 3) * 2) * 4
    t = fit_temperature(logits, y)
    ece = ComponentSpec.from_value("metric", "ece").build()
    assert t > 2 and ece(logits / t, y) < ece(logits, y) / 3


def test_threshold_and_conformal_coverage():
    y = (torch.rand(3000) < 0.1).long()
    p = (0.6 * y.float() + 0.4 * torch.rand(3000)).clamp(0, 1)
    assert best_threshold(p, y, "f1") > 0.3
    logits = torch.randn(3000, 4) + F.one_hot(torch.randint(0, 4, (3000,)), 4) * 2
    target = logits.argmax(-1)
    flip = torch.rand(3000) < 0.3
    target[flip] = torch.randint(0, 4, (int(flip.sum()),))
    cal, test = slice(0, 1500), slice(1500, 3000)
    q = conformal_quantile(conformal_scores(logits[cal], target[cal], True), alpha=0.1)
    sets = prediction_sets(logits[test].softmax(-1), q)
    coverage = sum(int(t) in s for t, s in zip(target[test].tolist(), sets)) / len(sets)
    assert coverage >= 0.87


def test_tta_variants():
    img = torch.randn(2, 3, 4, 4)
    out = tta_variants(img, ["hflip", "vflip"], 0.0)
    assert len(out) == 3 and torch.equal(out[1], img.flip(-1))
    assert len(tta_variants(torch.randn(5, 3), ["rauschen", "hflip"], 0.1)) == 2


def test_attribution_methods():
    w = torch.tensor([[2.0, -1.0, 0.0]])
    forward = (lambda x: x.float() @ w.T)
    x = torch.tensor([[1.0, 1.0, 1.0]])
    assert torch.allclose(gradient_x_input(forward, x), torch.tensor([[2.0, -1.0, 0.0]]))
    ig = integrated_gradients(forward, x, steps=8)
    assert torch.allclose(ig.sum(), forward(x).sum(), atol=1e-4)        # Vollständigkeit
    occ = occlusion(forward, x)
    assert torch.allclose(occ, torch.tensor([[2.0, -1.0, 0.0]]))
    with pytest.raises(ValueError):
        gradient_x_input(forward, torch.tensor([[1, 2, 3]]))
    imp = permutation_importance(lambda xx: -float((forward(xx) - forward(x2)).abs().mean()),
                                 x2 := torch.randn(200, 3), True, ["a", "b", "c"])
    assert [r[0] for r in imp.rows][0] == "a" and imp.rows[-1][0] == "c"


def test_fairness_report():
    target = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    pred = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0])
    groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    rep = fairness_report(pred, target, groups)
    assert rep.groups["a"]["positive_rate"] == 0.5 and rep.groups["b"]["positive_rate"] == 0.0
    assert rep.disparate_impact == 0.0 and not rep.fair
    assert "Fairness-Bericht" in str(rep)


@pytest.fixture(scope="module")
def trained():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.max_steps", 120)
    p.set("training.verbose", False)
    p.set("inference.conformal_alpha", 0.1)
    p.set("inference.uncertainty", "mc_dropout")
    p.set("inference.samples", 4)
    p.train()
    return p


def test_settings_change_keeps_trained_model(trained):
    model = trained.model
    trained.set("inference.abstain_below", 0.99)
    assert trained.model is model and trained.trainer is not None
    trained.set("inference.abstain_below", None)
    trained.undo()
    assert trained.model is model


def test_project_calibrate_and_predict(trained):
    res = trained.calibrate()
    assert {"temperatur", "ece_vorher", "ece_nachher", "conformal_q"} <= set(res)
    assert trained.config.inference.temperature == pytest.approx(res["temperatur"])
    x = trained.data().train["inputs"][:6]
    out = trained.predict_with_uncertainty(x)
    assert len(out["vorhersage"]) == 6 and out["wahrscheinlichkeiten"].shape == (6, 3)
    assert "streuung" in out and all(len(m) >= 1 for m in out["menge"])
    assert torch.allclose(trained.predict_proba(x).sum(-1), torch.ones(6), atol=1e-5)
    trained.set("inference.abstain_below", 0.999)
    assert set(trained.predict_labels(x)) == {"unsicher"}
    trained.set("inference.abstain_below", None)


def test_project_explain_and_importance(trained):
    x = trained.data().train["inputs"][:2]
    e = trained.explain_prediction(x)
    assert e.method == "integrated_gradients" and len(e.top()) == 2
    imp = trained.feature_importance(repeats=2)
    assert len(imp.rows) == 2 and "Merkmals-Wichtigkeit" in str(imp)
    groups = (x.new_zeros(len(trained.trainer.val_data))).long().tolist()
    assert "Fairness" in str(trained.fairness_report(groups))


def test_regression_conformal_interval():
    p = Project.from_preset("tabellen_regression")
    p.set("training.max_steps", 60)
    p.set("training.verbose", False)
    p.set("inference.conformal_alpha", 0.2)
    p.train()
    res = p.calibrate()
    lo, hi = p.predict_with_uncertainty(p.data().train["inputs"][:4])["intervall"]
    assert (hi - lo).allclose(torch.full_like(hi, 2 * res["intervall_halbbreite"]))


def test_token_explanation_finds_key_token():
    p = Project.from_preset("text_klassifikation")
    p.set("training.max_steps", 150)
    p.set("training.verbose", False)
    p.train()
    data = p.data().train
    idx = int((data["targets"] == 1).nonzero()[0])
    e = p.explain_prediction(data["inputs"][idx:idx + 1])
    top_name = e.top(1)[0][0]
    position = int(top_name.split(":")[0])
    assert int(data["inputs"][idx, position]) == 7                 # das gesuchte Token (vocab_size − 1)
