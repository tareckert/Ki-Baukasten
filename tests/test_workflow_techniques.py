"""Ablauf (Suche, föderiert, Model Soup, ES) und neue Trainings-Techniken."""

import copy
import math

import pytest
import torch

from ki_baukasten import MemoryConfig, Project, Trainer, TrainingConfig, build_model
from ki_baukasten.workflow import average_state_dicts, parse_space, split_clients


def _tab(**over):
    base = {"training.callbacks": [], "training.verbose": False, "deployment.show_after_training": False}
    base.update(over)
    return Project.from_preset("tabellen_klassifikation", **base)


@pytest.mark.parametrize("method", ["random", "grid", "evolution", "pso", "pbt"])
def test_search_methods(method):
    p = _tab()
    space = {"training.optimizer.lr": {"min": 1e-3, "max": 3e-2, "log": True},
             "training.optimizer.weight_decay": [0.0, 0.1]}
    res = p.tune(method, space=space, trials=3, steps_per_trial=8, population=3, generations=2)
    assert res.trials and math.isfinite(res.best.score)
    # übernommen – bei PBT genau die Einstellungen des behaltenen (fertig trainierten) Modells
    applied = res.pbt_best["params"] if method == "pbt" else res.best_params
    assert p.get("training.optimizer.lr") == applied["training.optimizer.lr"]
    if method == "pbt":
        assert p.model is not None
    if method in ("evolution", "pso", "pbt"):
        assert {t.generation for t in res.trials} == {0, 1}


def test_space_parsing_and_units():
    dims = parse_space({"a": [1, 2, 3], "b": {"min": 1e-4, "max": 1e-1, "log": True}, "c": {"min": 1, "max": 9,
                                                                                         "type": "int"}})
    assert dims[0].value(0.99) == 3 and dims[0].value(0.0) == 1
    assert abs(dims[1].value(dims[1].unit(1e-3)) - 1e-3) < 1e-6
    assert isinstance(dims[2].value(0.5), int)


def test_average_and_split_clients():
    a = {"w": torch.ones(2), "n": torch.tensor(3)}
    b = {"w": torch.zeros(2), "n": torch.tensor(7)}
    avg = average_state_dicts([a, b], [3.0, 1.0])
    assert torch.allclose(avg["w"], torch.full((2,), 0.75)) and int(avg["n"]) == 3
    data = {"inputs": torch.randn(100, 3), "targets": torch.randint(0, 4, (100,))}
    iid = split_clients(data, 4, "iid")
    assert sum(c["inputs"].shape[0] for c in iid) == 100
    skew = split_clients(data, 4, "nicht_iid", alpha=0.1)
    assert sum(c["inputs"].shape[0] for c in skew) >= 100 - 4


@pytest.mark.parametrize("algorithm", ["fedavg", "fedprox"])
def test_federated_training(algorithm):
    p = _tab()
    hist = p.federated_train(clients=3, rounds=2, local_steps=5, algorithm=algorithm, split="nicht_iid")
    assert len(hist) == 2 and "val_loss" in hist[-1] and p.model is not None


def test_model_soup_and_run_workflow(tmp_path):
    p = _tab(**{"training.max_steps": 10})
    out = p.model_soup(2, "uniform")
    assert out["zutaten"] == 2 and "val_loss" in out["ergebnis"]
    q = _tab(**{"training.max_steps": 10, "workflow.cross_validation": 2, "workflow.compress_after_training": True,
                "workflow.export_format": "state_dict", "workflow.export_path": str(tmp_path / "m"),
                "compression.steps": [{"name": "quantize", "args": {"format": "int8"}}]})
    report = q.run()
    assert {"training", "kreuzvalidierung", "kompression", "export"} <= set(report)
    assert list(tmp_path.glob("m*"))


def test_evolution_strategies_learns_without_backward():
    torch.manual_seed(0)
    x = torch.randn(256, 4)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    model = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 1, "hidden_dims": []}})
    tr = Trainer(model, TrainingConfig(paradigm={"name": "evolution_strategies", "args": {"pairs": 8, "sigma": 0.05}},
                                       max_steps=150, batch_size=64, verbose=False, validation_split=0.0,
                                       optimizer={"name": "adam", "args": {"lr": 0.05}}, scheduler="constant"),
                 MemoryConfig())
    before = tr.evaluate({"inputs": x, "targets": y})["val_loss"]
    tr.fit({"inputs": x, "targets": y})
    after = tr.evaluate({"inputs": x, "targets": y})["val_loss"]
    assert after < 0.3 * before


def _mlp_trainer(**training):
    torch.manual_seed(0)
    model = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 3, "hidden_dims": [16]}})
    t = TrainingConfig(**{"max_steps": 12, "batch_size": 16, "verbose": False, "validation_split": 0.0, **training})
    return Trainer(model, t, MemoryConfig())


DATA = {"inputs": torch.randn(64, 4), "targets": torch.randint(0, 3, (64,))}


def test_spike_guard_skips_and_rolls_back():
    tr = _mlp_trainer(stability={"spike_guard": True, "spike_warmup": 2, "spike_action": "rollback",
                                 "rollback_every": 1})
    original = tr.paradigm.compute
    calls = {"n": 0}

    def spiky(trainer, batch, train):
        loss, m = original(trainer, batch, train)
        calls["n"] += 1
        if train and calls["n"] == 8:
            loss = loss * 100.0               # künstliche Verlust-Spitze
        return loss, m

    tr.paradigm.compute = spiky
    lr_before = tr.optimizers["main"].param_groups[0]["lr"]
    tr.fit(DATA)
    sched = tr.schedulers["main"]
    assert tr._good_state is not None
    assert all(abs(b - lr_before * 0.5) < 1e-12 for b in sched.base_lrs)   # Rücksprung: Lernrate halbiert


@pytest.mark.parametrize("extras", [{"lookahead_k": 3}, {"gradient_centralization": True},
                                    {"grokfast_alpha": 0.9, "grokfast_lambda": 1.0}])
def test_optimizer_extras_train(extras):
    tr = _mlp_trainer(optimizer_extras=extras)
    before = tr.evaluate(DATA)["val_loss"]
    tr.fit(DATA)
    assert tr.evaluate(DATA)["val_loss"] < before + 0.05


def test_lookahead_pulls_to_slow_weights():
    tr = _mlp_trainer(optimizer_extras={"lookahead_k": 1, "lookahead_alpha": 0.0}, max_steps=3)
    start = [p.detach().clone() for p in tr.model.parameters()]
    tr.fit(DATA)
    for a, b in zip(start, tr.model.parameters()):               # alpha=0: Gewichte bleiben auf dem Start
        assert torch.allclose(a, b)


def test_swa_callback_averages():
    tr = _mlp_trainer(callbacks=[{"name": "swa", "args": {"start_fraction": 0.5}}])
    tr.fit(DATA)
    assert tr.callbacks[0].n >= 5


def test_image_augmentation_and_cutmix():
    p = Project.from_preset("bildklassifikation_cnn", **{
        "training.max_steps": 3, "training.verbose": False, "deployment.show_after_training": False,
        "training.augmentation": {"hflip": True, "crop_padding": 2, "random_erasing": 0.5, "cutmix_alpha": 1.0}})
    p.train()
    tr = p.trainer
    x = torch.randn(4, 3, 8, 8)
    y = torch.tensor([0, 1, 0, 1])
    xa, ya = tr._cutmix(x, y, 1.0)
    assert xa.shape == x.shape and ya.shape == (4, tr._num_classes(y)) and torch.allclose(ya.sum(1), torch.ones(4))


def test_balance_classes_and_label_smoothing():
    targets = torch.cat([torch.zeros(90), torch.ones(10)]).long()
    data = {"inputs": torch.randn(100, 4), "targets": targets}
    tr = _mlp_trainer(balance_classes=True, paradigm={"name": "supervised", "args": {"label_smoothing": 0.1}})
    loader, _ = tr._loaders(data, None)
    seen = torch.cat([b["targets"] for b in loader])
    assert 0.3 < seen.float().mean() < 0.7              # seltene Klasse wird deutlich häufiger gezogen
    tr.fit(data)


def test_dp_sgd_clips_per_sample_and_trains():
    tr = _mlp_trainer(privacy={"enabled": True, "clip_norm": 0.5, "noise_multiplier": 0.0}, max_steps=2)
    assert tr._split == 16
    ref = copy.deepcopy(tr.model)
    tr.fit(DATA)
    assert any(not torch.equal(a, b) for a, b in zip(ref.parameters(), tr.model.parameters()))


def test_lr_finder_and_cross_validation():
    p = _tab()
    res = p.find_lr(steps=40)
    assert res.suggestion is not None and 1e-6 < res.suggestion < 1.0
    cv = p.cross_validate(k=2, steps=5)
    assert len(cv["folds"]) == 2 and "val_loss" in cv["mittel"]
