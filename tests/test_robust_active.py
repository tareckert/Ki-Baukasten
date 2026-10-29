"""Tests für Angriffe/adversariales Training, Robustheitsbericht, Active Learning, Pseudo-Labeling und Fortsetzen."""

import torch

from ki_baukasten import Project, Trainer, TrainingConfig, build_model
from ki_baukasten.analysis.robustness import attack
from ki_baukasten.workflow.active import informativeness, k_center, pseudo_labels, select


def test_attacks_stay_in_epsilon_ball_and_increase_loss():
    w = torch.randn(3, 4)
    forward = (lambda x: x @ w.T)
    loss = (lambda out, y: torch.nn.functional.cross_entropy(out, y))
    x, y = torch.randn(32, 4), torch.randint(0, 3, (32,))
    for method in ("fgsm", "pgd"):
        xa = attack(forward, loss, x, y, method, epsilon=0.1, steps=4)
        assert (xa - x).abs().max() <= 0.1 + 1e-6
        assert loss(forward(xa), y) > loss(forward(x), y)


def test_adversarial_training_runs():
    x = torch.randn(200, 4)
    y = (x[:, 0] > 0).long()
    m = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden_dims": [16]}})
    tr = Trainer(m, TrainingConfig(epochs=15, verbose=False, optimizer={"name": "adamw", "lr": 1e-2},
                                   adversarial={"method": "pgd", "epsilon": 0.05, "steps": 2}))
    tr.fit((x, y))
    assert tr.evaluate((x, y))["val_accuracy"] > 0.8


def test_selection_strategies():
    probs = torch.tensor([[0.98, 0.01, 0.01], [0.4, 0.35, 0.25], [0.5, 0.5, 0.0], [0.7, 0.2, 0.1]])
    assert int(informativeness(probs, "unsicherheit").argmax()) == 1
    assert int(informativeness(probs, "marge").argmax()) == 2
    idx, _ = select(probs, 2, "entropie")
    assert idx[0] == 1
    pts = torch.tensor([[0.0, 0], [0.1, 0], [10, 10], [0, 0.1]])
    assert 2 in k_center(pts, 2)
    mask, labels = pseudo_labels(probs, 0.9)
    assert mask.tolist() == [True, False, False, False] and labels[0] == 0


def test_project_robustness_active_and_pseudo_labeling():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.max_steps", 60)
    p.set("training.verbose", False)
    p.train()
    rep = p.robustness_report(epsilons=[0.05], methods=("rauschen", "fgsm"))
    assert len(rep.rows) == 2 and "Robustheit" in str(rep)
    pool = torch.randn(100, 2)
    sel = p.select_for_labeling(pool, 7, "unsicherheit")
    assert len(sel.indices) == 7 and len(set(sel.indices)) == 7
    out = p.pseudo_label(pool, threshold=0.0, rounds=1)
    assert out["runden"][0]["uebernommen"] == 100


def test_resume_from_checkpoint(tmp_path):
    def project(steps):
        p = Project.from_preset("tabellen_klassifikation")
        p.set("training.max_steps", steps)
        p.set("training.verbose", False)
        p.set("training.callbacks", [{"name": "checkpoint", "directory": str(tmp_path), "every_n_steps": 10}])
        return p

    first = project(20)
    first.train()
    second = project(30)
    second.set("training.resume_from", "auto")
    second.train()
    steps = [h["step"] for h in second.history if "step" in h]
    assert second.trainer.global_step == 30 and min(steps) <= 20      # Historie des ersten Laufs übernommen
    assert (tmp_path / "schritt_30.pt").exists()
