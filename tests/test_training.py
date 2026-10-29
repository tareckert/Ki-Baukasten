"""Tests des Trainers und aller Paradigmen (klein gehalten, damit die Suite schnell bleibt)."""

import pytest
import torch

from ki_baukasten import Trainer, TrainingConfig, build_model, registry


def cfg(**kw):
    base = dict(epochs=2, batch_size=32, verbose=False, validation_split=0.2)
    base.update(kw)
    return TrainingConfig(**base)


def test_classification_learns():
    x = torch.randn(400, 2)
    y = ((x[:, 0] * x[:, 1]) > 0).long()
    m = build_model({"name": "mlp", "args": {"input_dim": 2, "output_dim": 2, "hidden_dims": [32, 32]}})
    tr = Trainer(m, cfg(epochs=25, optimizer={"name": "adamw", "args": {"lr": 1e-2}}))
    tr.fit((x, y))
    assert tr.evaluate((x, y))["val_accuracy"] > 0.85


def test_causal_lm_learns_and_generates():
    seq = (torch.arange(12).unsqueeze(0) + torch.randint(0, 16, (128, 1))) % 16
    m = build_model({"name": "transformer", "args": {"vocab_size": 16, "d_model": 32, "n_layers": 2,
                                                       "block": {"mixer.num_heads": 4}}})
    tr = Trainer(m, cfg(paradigm="causal_lm", epochs=12, optimizer={"name": "adamw", "args": {"lr": 3e-3}}))
    tr.fit({"inputs": seq})
    assert tr.evaluate({"inputs": seq})["val_accuracy"] > 0.7
    out = m.generate(seq[:1, :3].to(tr.device), max_new_tokens=4, temperature=0)
    assert out.shape == (1, 7)


@pytest.mark.parametrize("paradigm,data,model", [
    ("masked_lm", lambda: {"inputs": torch.randint(1, 15, (64, 10))},
     {"name": "transformer", "args": {"architecture": "encoder_only", "vocab_size": 16, "d_model": 16, "n_layers": 1}}),
    ("seq2seq", lambda: (torch.randint(1, 10, (64, 6)), torch.randint(1, 10, (64, 6))),
     {"name": "transformer", "args": {"architecture": "encoder_decoder", "vocab_size": 10, "d_model": 16, "n_layers": 1}}),
    ("autoencoder", lambda: torch.rand(64, 32), {"name": "autoencoder", "args": {"variant": "vae"}}),
    ("contrastive", lambda: torch.randn(64, 16), {"name": "mlp", "args": {"output_dim": 8}}),
    ("gan", lambda: torch.randn(64, 2), {"name": "gan"}),
    ("diffusion", lambda: torch.randn(64, 2), {"name": "diffusion", "args": {"timesteps": 50}}),
    ("preference", lambda: {"chosen": torch.randn(64, 16), "rejected": torch.randn(64, 16)}, {"name": "mlp"}),
    ("dpo", lambda: {"chosen": torch.randint(0, 8, (32, 6)), "rejected": torch.randint(0, 8, (32, 6))},
     {"name": "transformer", "args": {"vocab_size": 8, "d_model": 16, "n_layers": 1}}),
])
def test_paradigms_run(paradigm, data, model):
    tr = Trainer(build_model(model), cfg(paradigm=paradigm, epochs=1))
    tr.fit(data())
    assert tr.global_step > 0 and tr.history


def test_distillation(tmp_path):
    teacher = build_model({"name": "mlp", "args": {"input_dim": 2, "output_dim": 2}})
    torch.save({"model": teacher.state_dict()}, tmp_path / "t.pt")
    x, y = torch.randn(64, 2), torch.randint(0, 2, (64,))
    tr = Trainer(build_model({"name": "mlp", "args": {"input_dim": 2, "output_dim": 2, "hidden_dims": [8]}}),
                 cfg(paradigm={"name": "distillation", "args": {
                     "teacher": {"name": "mlp", "args": {"input_dim": 2, "output_dim": 2}},
                     "teacher_checkpoint": str(tmp_path / "t.pt")}}, epochs=1))
    tr.fit((x, y))


@pytest.mark.parametrize("opt", registry("optimizer").names())
def test_optimizers(opt):
    tr = Trainer(build_model("mlp"), cfg(optimizer=opt, epochs=1))
    tr.fit((torch.randn(64, 16), torch.randn(64, 1)))


@pytest.mark.parametrize("sched", registry("scheduler").names())
def test_schedulers(sched):
    tr = Trainer(build_model("mlp"), cfg(scheduler=sched, epochs=1))
    tr.fit((torch.randn(64, 16), torch.randn(64, 1)))


def test_regularization_stability_and_callbacks(tmp_path):
    c = cfg(epochs=3, gradient_accumulation_steps=2, ema={"enabled": True, "decay": 0.9},
            regularization={"l1": 1e-6, "l2": 1e-6, "orthogonality": 1e-6, "mixup_alpha": 0.2,
                            "input_noise_std": 0.01, "gradient_noise_std": 1e-4, "flooding_level": 0.01},
            gradient_clipping={"mode": "adaptive"}, layerwise_lr_decay=0.8,
            callbacks=[{"name": "early_stopping", "args": {"patience": 1}},
                       {"name": "checkpoint", "args": {"directory": str(tmp_path / "ck")}},
                       {"name": "csv_logger", "args": {"path": str(tmp_path / "log.csv")}},
                       {"name": "jsonl_logger", "args": {"path": str(tmp_path / "log.jsonl")}},
                       {"name": "gradient_monitor"}, {"name": "time_limit"}],
            losses=[{"name": "mse"}, {"name": "huber", "args": {"weight": 0.5}}])
    tr = Trainer(build_model("mlp"), c)
    tr.fit((torch.randn(128, 16), torch.randn(128, 1)))
    assert (tmp_path / "ck" / "beste.pt").exists() and (tmp_path / "log.csv").exists()
    state = tr.load_checkpoint(tmp_path / "ck" / "beste.pt")
    assert "optimizers" in state


def test_sam_and_curriculum():
    seqs = torch.randint(1, 10, (64, 8))
    seqs[:32, 4:] = 0
    tr = Trainer(build_model({"name": "mlp", "args": {"input_dim": 8, "output_dim": 1}}),
                 cfg(epochs=1, regularization={"sam_rho": 0.05}, curriculum={"strategy": "length"}))
    tr.fit((seqs.float(), torch.randn(64, 1)))


def test_nan_policy_skip():
    tr = Trainer(build_model("mlp"), cfg(epochs=1, nan_policy="skip"))
    bad = torch.randn(64, 16)
    bad[0, 0] = float("nan")
    tr.fit((bad, torch.randn(64, 1)))  # darf nicht abstürzen
