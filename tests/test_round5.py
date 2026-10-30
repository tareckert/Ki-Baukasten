"""Regressionstests Runde 5: Fehlerkorrekturen, schlanker Export, Leistung, Hardware-Automatik."""

import json
import warnings

import pytest
import torch
from torch.utils.data import DataLoader

from ki_baukasten import Project, Trainer, TrainingConfig, build_model
from ki_baukasten.analysis.export import fold_batchnorm, prepare_for_export
from ki_baukasten.components.feedforward import MixtureOfExperts, MoEConfig
from ki_baukasten.core.errors import ConfigError
from ki_baukasten.training.data_utils import DictDataset, TensorBatchLoader, collate, make_loader, split_dataset
from ki_baukasten.training.paradigms import main_output
from ki_baukasten.training.trainer import EMA, LazyMetrics, mean_metrics

QUIET = {"training.verbose": False, "deployment.show_after_training": False}


def _project(name, steps=3, **over):
    p = Project.from_preset(name)
    p.update({"training.max_steps": steps, **QUIET, **over})
    return p


def _export_matches(p, tmp_path, fmt="export"):
    p.train()
    ex = p._example_input()
    model = p.model.eval()
    with torch.no_grad():
        ref = main_output(model(ex)).cpu()
    path = p.export(tmp_path / "m", fmt)
    if fmt == "export":
        out = main_output(torch.export.load(str(path)).module()(ex.cpu()))
    else:
        out = main_output(torch.jit.load(str(path))(ex.cpu()))
    assert torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-3)
    return p.last_export


@pytest.mark.parametrize("preset", ["text_klassifikation", "bildklassifikation_vit", "zeitreihen_transformer",
                                    "sprachmodell_moe"])
def test_previously_unexportable_models_export(preset, tmp_path):
    warnings.simplefilter("ignore")
    _export_matches(_project(preset), tmp_path)


def test_cnn_export_folds_batchnorm_and_matches(tmp_path):
    report = _export_matches(_project("bildklassifikation_cnn"), tmp_path)
    assert report.batchnorm_folded > 0 and report.max_diff < 1e-4


def test_fold_batchnorm_exact():
    seq = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, bias=False), torch.nn.BatchNorm2d(8), torch.nn.ReLU())
    seq.train()
    seq(torch.randn(16, 3, 8, 8))                      # Statistiken füllen
    seq.eval()
    x = torch.randn(2, 3, 8, 8)
    ref = seq(x)
    assert fold_batchnorm(seq) == 1 and isinstance(seq[1], torch.nn.Identity)
    assert torch.allclose(seq(x), ref, atol=1e-5)


def test_gan_export_strips_discriminator(tmp_path):
    report = _export_matches(_project("gan_2d"), tmp_path)
    assert report.removed and all(n.startswith("discriminator.") or n.endswith("num_batches_tracked")
                                  for n in report.removed)
    assert report.bytes_after < report.bytes_before


def test_qlora_export_is_exact(tmp_path):
    warnings.simplefilter("ignore")
    p = _project("sprachmodell_qlora", steps=5)
    p.set("model.n_layers", 2)
    report = _export_matches(p, tmp_path)
    assert report.lora_merged == 0                    # Adapter auf nf4-Basis bleiben (exakt)


def test_streaming_wrappers_not_in_export(tmp_path):
    warnings.simplefilter("ignore")
    p = _project("sprachmodell_ram_streaming")
    p.set("model.n_layers", 2)
    report = _export_matches(p, tmp_path, "torchscript")
    assert report.wrappers_removed > 0


def test_state_dict_export_without_unused(tmp_path):
    p = _project("gan_2d")
    p.train()
    path = p.export(tmp_path / "g", "state_dict")
    state = torch.load(path, weights_only=True)
    assert not any(k.startswith("discriminator.") for k in state)


def test_export_settings_switchable(tmp_path):
    p = _project("bildklassifikation_cnn")
    p.train()
    p.export(tmp_path / "a", "export", fold_batchnorm=False, strip_unused=False)
    assert p.last_export.batchnorm_folded == 0 and not p.last_export.removed


def test_ema_warmup_apply_at_end_and_early_stopping():
    lin = torch.nn.Linear(2, 2)
    ema = EMA(lin, 0.999)
    assert ema.current_decay() == pytest.approx(0.1)
    x, y = torch.randn(200, 4), torch.randint(0, 2, (200,))
    m = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden_dims": [8]}})
    tr = Trainer(m, TrainingConfig(epochs=3, verbose=False, ema={"enabled": True, "decay": 0.9}))
    tr.fit((x, y))
    assert all(torch.allclose(p.float(), tr.ema.shadow[n]) for n, p in tr.model.named_parameters())
    m2 = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2, "hidden_dims": [8]}})
    tr2 = Trainer(m2, TrainingConfig(epochs=3, verbose=False, ema={"enabled": True, "decay": 0.9},
                                     callbacks=[{"name": "early_stopping", "patience": 10}]))
    tr2.fit((x, y))
    es = tr2.callbacks[0]
    assert all(torch.equal(v, es.best_state[k]) for k, v in tr2.model.state_dict().items())


def test_tensor_batch_loader_identical_to_dataloader():
    data = {"inputs": torch.randn(103, 4), "targets": torch.randint(0, 3, (103,)), "m": {"a": torch.arange(103)}}
    ds, _ = split_dataset(DictDataset(data), 0.2, 0)
    rest, _ = split_dataset(ds, 0.1, 1)                  # verschachtelte Teilmengen
    loader = make_loader(rest, 16, True)
    assert isinstance(loader, TensorBatchLoader) and len(loader) == len(DataLoader(rest, batch_size=16))
    torch.manual_seed(3)
    a = list(loader)
    after_a = torch.rand(1)
    torch.manual_seed(3)
    b = list(DataLoader(rest, batch_size=16, shuffle=True, collate_fn=collate))
    after_b = torch.rand(1)
    assert len(a) == len(b) and all(torch.equal(x["inputs"], y["inputs"]) and torch.equal(x["m"]["a"], y["m"]["a"])
                                    for x, y in zip(a, b))
    assert torch.equal(after_a, after_b)


def test_lazy_and_mean_metrics():
    lm = LazyMetrics({"loss": torch.tensor(2.0), "lr": 0.1})
    assert lm["loss"] == 2.0 and dict(lm) == {"loss": 2.0, "lr": 0.1} and {**lm}["loss"] == 2.0
    avg = mean_metrics({"a": [torch.tensor(1.0), torch.tensor(float("nan")), torch.tensor(3.0)], "b": [1.0, 2.0],
                        "c": [torch.tensor(float("inf"))]})
    assert avg == {"a": 2.0, "b": 1.5}


def test_fused_optimizer_chosen_automatically():
    m = build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2}})
    tr = Trainer(m, TrainingConfig(verbose=False))
    assert tr.optimizers["main"].defaults.get("fused") is True
    tr2 = Trainer(build_model({"name": "mlp", "args": {"input_dim": 4, "output_dim": 2}}),
                  TrainingConfig(verbose=False, optimizer={"name": "adamw", "fused": False}))
    assert not tr2.optimizers["main"].defaults.get("fused")


def test_moe_dispatch_and_export():
    torch.manual_seed(0)
    m = MixtureOfExperts(16, MoEConfig(num_experts=4, top_k=2)).eval()
    x = torch.randn(3, 5, 16)
    out = m(x)
    flat = x.reshape(-1, 16)
    probs = m.router(flat).float().softmax(-1)
    w, i = probs.topk(2, -1)
    w = w / w.sum(-1, keepdim=True)
    ref = sum((i == e).any(-1, keepdim=True) * ex(flat) * (w * (i == e)).sum(-1, keepdim=True)
              for e, ex in enumerate(m.experts))
    assert torch.allclose(out.reshape(-1, 16), ref, atol=1e-5)
    assert torch.allclose(torch.export.export(m, (x,)).module()(x), out, atol=1e-6)
    m.cfg.capacity_factor = 0.1                          # Kapazität wirkt nur im Training
    assert torch.allclose(m(x), out, atol=1e-6)


def test_positional_length_checked_in_python():
    m = build_model({"name": "transformer", "args": {"vocab_size": 20, "d_model": 16, "n_layers": 1, "max_seq_len": 8,
                                                       "positional": "learned"}})
    with pytest.raises(ValueError, match="max_seq_len"):
        m(torch.randint(0, 20, (1, 9)))


def test_hardware_resolution(monkeypatch):
    from ki_baukasten import MemoryConfig
    from ki_baukasten.memory import hardware

    cnn = build_model("cnn")
    cases = {(8, 0): ("bf16", "high", True, True), (7, 5): ("fp16", "highest", True, True)}
    for cap, (mp, mm, bench, cl) in cases.items():
        monkeypatch.setattr(hardware, "detect", lambda d, cap=cap: hardware.HardwareInfo(
            "cuda", "GPU", cap, bf16=cap >= (8, 0), tf32=cap >= (8, 0)))
        r = hardware.resolve(MemoryConfig(), torch.device("cpu"), cnn)
        assert (r["mixed_precision"][0], r["matmul_precision"][0], r["cudnn_benchmark"][0], r["channels_last"][0]) \
            == (mp, mm, bench, cl)
    monkeypatch.setattr(hardware, "detect", lambda d: hardware.HardwareInfo("cpu", "CPU"))
    r = hardware.resolve(MemoryConfig(mixed_precision="bf16"), torch.device("cpu"), cnn)
    assert r["mixed_precision"] == ("bf16", "fest eingestellt") and r["channels_last"][0] is False
    assert "Hardware" in Project.from_preset("tabellen_klassifikation").hardware_report()


def test_second_train_keeps_disk_streamed_weights(tmp_path, monkeypatch):
    warnings.simplefilter("ignore")
    monkeypatch.chdir(tmp_path)
    p = _project("sprachmodell_mini", **{"memory.offload.mode": "stream", "memory.offload.storage": "disk",
                                         "memory.lora.enabled": True})
    p.set("model.n_layers", 2)
    p.train()
    p.train()
    with torch.no_grad():
        assert torch.isfinite(p.model(p.data().train["inputs"][:2])).all()


def test_technique_fixes():
    warnings.simplefilter("ignore")
    p = _project("bildklassifikation_cnn")
    p.enable("dp_sgd")
    assert p.get("model.stem.norm").name == "groupnorm"
    p.train()
    states = {r.technique.key: r.status for r in Project.from_preset("segmentierung_unet").techniques()}
    assert states["lora"] == "nicht_anwendbar" and states["qlora"] == "nicht_anwendbar"
    assert {r.technique.key: r.status for r in Project.from_preset("rl_cartpole_ppo").techniques()}["pbt"] \
        == "nicht_anwendbar"
    q = _project("text_klassifikation")
    q.enable("layerweise")
    q.train()                                            # Training läuft als SSD-Streaming statt abzubrechen


def test_compact_config_roundtrip(tmp_path):
    p = _project("tabellen_klassifikation")
    p.train()
    d = p.save(tmp_path / "x")
    saved = json.loads((d / "projekt.json").read_text(encoding="utf-8"))
    assert "deployment" not in saved or "export" not in saved.get("deployment", {})
    assert Project.load(d).config.to_dict() == p.config.to_dict()        # exakt wie gespeichert
    p.set("deployment.export.compact_config", False)
    full = json.loads((p.save(tmp_path / "y") / "projekt.json").read_text(encoding="utf-8"))
    assert len(json.dumps(full)) > len(json.dumps(saved))


def test_auto_lr_and_batch_size_on_cpu():
    warnings.simplefilter("ignore")
    p = _project("tabellen_klassifikation", steps=5, **{"training.auto_lr": True, "training.auto_batch_size": True})
    bs = p.config.training.batch_size
    p.train()
    assert p.config.training.batch_size == bs            # CPU: Batchgröße bleibt
    assert p.config.training.optimizer.args.lr > 0


def test_segmentation_logits_accuracy_metric():
    from ki_baukasten.core import ComponentSpec

    mask = torch.randint(0, 3, (2, 4, 4))
    logits = torch.nn.functional.one_hot(mask, 3).permute(0, 3, 1, 2).float()
    acc = ComponentSpec.from_value("metric", "accuracy").build()
    assert acc(logits, mask) == 1.0 and acc(mask, mask) == 1.0


def test_prepare_for_export_keeps_original_untouched():
    p = _project("bildklassifikation_cnn")
    p.train()
    before = {k: v.clone() for k, v in p.model.state_dict().items()}
    prepare_for_export(p.model, p._example_input())
    assert all(torch.equal(before[k], v) for k, v in p.model.state_dict().items())
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in p.model.modules())


def _csv(path, rows, header):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_detect_data_kinds(tmp_path):
    import random

    from ki_baukasten.project import detect_data

    rng = random.Random(0)
    _csv(tmp_path / "k.csv", [[rng.random(), rng.choice("ab"), rng.choice(["ja", "nein"])] for _ in range(50)],
         ["x", "farbe", "kauft"])
    assert detect_data(tmp_path / "k.csv", "kauft")[0] == "tabellen_klassifikation"
    _csv(tmp_path / "r.csv", [[rng.random(), rng.random() * 1000] for _ in range(80)], ["x", "preis"])
    assert detect_data(tmp_path / "r.csv")[0] == "tabellen_regression"
    _csv(tmp_path / "t.csv", [["Das ist ein recht langer Beispieltext Nummer %d" % i, "a"] for i in range(30)],
         ["text", "label"])
    preset, data = detect_data(tmp_path / "t.csv")
    assert preset == "text_klassifikation" and data["text_column"] == "text"
    (tmp_path / "q.jsonl").write_text('{"prompt": "a", "response": "b"}\n', encoding="utf-8")
    assert detect_data(tmp_path / "q.jsonl")[0] == "sprachmodell_sft"
    (tmp_path / "bilder" / "rot").mkdir(parents=True)
    (tmp_path / "bilder" / "rot" / "1.png").write_bytes(b"")
    assert detect_data(tmp_path / "bilder")[0] == "bildklassifikation_cnn"
    with pytest.raises(ConfigError, match="nicht gefunden"):
        detect_data(tmp_path / "fehlt.csv")


def test_detect_data_excel_csv_and_consistent_task(tmp_path):
    import random

    from ki_baukasten.data.files import CSVConfig, load_csv
    from ki_baukasten.project import detect_data

    rng = random.Random(1)
    lines = ["flaeche;preis"] + [f"{rng.uniform(20, 99):.1f}".replace(".", ",") + f";{rng.randint(1, 9)},{i}"
                                 for i in range(60)]
    (tmp_path / "de.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    preset, data = detect_data(tmp_path / "de.csv")
    assert preset == "tabellen_regression" and data["delimiter"] == ";"
    assert load_csv(CSVConfig(**{k: v for k, v in data.items() if k != "name"})).info["task"] == "regression"
    # ganzzahliges Ziel mit vielen Werten: Vorlage und Lader müssen dieselbe Aufgabe wählen
    _csv(tmp_path / "n.csv", [[rng.random(), rng.randint(0, 45)] for _ in range(100)], ["x", "anzahl"])
    preset, data = detect_data(tmp_path / "n.csv")
    task = load_csv(CSVConfig(**{k: v for k, v in data.items() if k != "name"})).info["task"]
    assert task == ("regression" if preset == "tabellen_regression" else "classification")


def test_project_auto_regression_in_real_units(tmp_path):
    import random

    warnings.simplefilter("ignore")
    rng = random.Random(0)
    rows = []
    for _ in range(400):
        a = rng.uniform(20, 200)
        rows.append([round(a, 1), rng.randint(1, 6), round(a * 3000 + rng.gauss(0, 5000))])
    _csv(tmp_path / "preise.csv", rows, ["flaeche", "zimmer", "preis"])
    p = Project.auto(tmp_path / "preise.csv", steps=200, **QUIET)
    assert p.config.name == "tabellen_regression" and p.last_test["test_r2"] > 0.9
    assert p.last_baseline["modell_mae"] < p.last_baseline["basis_mae"] / 3
    value = p.predict_raw([{"flaeche": 100, "zimmer": 3}])["vorhersage"][0]
    assert 240_000 < value < 360_000                     # echte Einheiten (Ziel-Skalierung zurückgerechnet)
    assert {"checkpoints", "log_datei"}.isdisjoint({r.technique.key for r in p.techniques(status="an")})


def test_cli_auto_and_hardware(tmp_path, capsys):
    import random

    from ki_baukasten.ui.cli import main as cli

    rng = random.Random(1)
    _csv(tmp_path / "k.csv", [[a := rng.random(), rng.random(), "ja" if a > 0.5 else "nein"] for _ in range(200)],
         ["a", "b", "y"])
    cli(["auto", str(tmp_path / "k.csv"), "--schritte", "40", "--setze", "training.verbose=false",
         "--setze", "deployment.show_after_training=false", "--speichern", str(tmp_path / "ki")])
    out = capsys.readouterr().out
    assert "Test:" in out and (tmp_path / "ki" / "MODELLKARTE.md").exists()
    cli(["hardware"])
    assert "Automatisch gewählt" in capsys.readouterr().out
