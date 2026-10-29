"""Tests: Probelauf (Wege, Statistiken, Gradienten, Warnungen), Gewichte, Attention, Testlauf, neue CLI-Befehle."""

import json
import warnings

import torch

from ki_baukasten import Project, build_model
from ki_baukasten.analysis.probe import (attention_maps, get_weight, heatmap, histogram, probe, tensor_stats,
                                         evaluate_model, evaluation_text, weight_stats)
from ki_baukasten.ui.cli import main as cli

warnings.simplefilter("ignore")


def test_probe_records_paths_and_gradients():
    m = build_model({"name": "transformer", "vocab_size": 30, "d_model": 32, "n_layers": 2})
    r = probe(m, torch.randint(0, 30, (2, 10)))
    names = [rec.name for rec in r.records]
    assert names.index("embed") < names.index("blocks.0") < names.index("blocks.1") < names.index("lm_head")
    assert r.loss is not None and r.records[0].grad_stats is not None
    assert all(rec.out_shapes for rec in r.records)
    assert "Probelauf" in r.to_text()
    assert "blocks.0.mixer" in r.activations


def test_probe_warns_on_dead_and_nan():
    m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    with torch.no_grad():
        m[0].bias.fill_(-100.0)  # alle ReLU-Ausgaben 0 → tote Neuronen
    r = probe(m, torch.randn(16, 4))
    assert any("null" in w for _, w in r.warnings())
    with torch.no_grad():
        m[2].weight.fill_(float("nan"))
    assert any("NaN" in w for _, w in probe(m, torch.randn(4, 4), backward=False).warnings())


def test_weight_views_and_attention():
    m = build_model({"name": "transformer", "vocab_size": 30, "d_model": 32, "n_layers": 2})
    rows = weight_stats(m)
    assert rows and {"name", "shape", "std", "zeros"} <= set(rows[0])
    w = get_weight(m, rows[1]["name"])
    assert len(heatmap(w, 16, 16)) <= 16
    edges, counts = histogram(w, 10)
    assert len(edges) == 11 and sum(counts) == w.numel()
    maps = attention_maps(m, torch.randint(0, 30, (1, 7)))
    assert len(maps) == 2 and next(iter(maps.values())).shape[-2:] == (7, 7)
    assert not any(a.attn.kernel.cfg.store_weights for a in (b.mixer for b in m.blocks))
    assert tensor_stats(torch.tensor([0.0, float("inf")]))["inf"] == 1


def test_test_run_classification():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.verbose", False)
    p.set("training.epochs", 1)
    p.train()
    res = evaluate_model(p.model, p.data().train, max_batches=2)
    assert "accuracy" in res and "confusion" in res and res["beispiele"]
    assert "Konfusionsmatrix" in evaluation_text(res)


def test_cli_compression_targets_probe_catalog(capsys, tmp_path):
    cli(["zielsysteme"])
    assert "smartphone" in capsys.readouterr().out
    cli(["probelauf", "sprachmodell_mini", "--tiefe", "1"])
    assert "Probelauf" in capsys.readouterr().out
    steps = json.dumps([{"name": "prune_ffn", "args": {"amount": 0.5, "importance": "magnitude"}},
                        {"name": "quantize"}])
    cli(["komprimieren", "sprachmodell_mini", "--schritte", steps, "--speichern", str(tmp_path / "k")])
    out = capsys.readouterr().out
    assert "Kompression" in out and (tmp_path / "k" / "projekt.json").exists()
    cli(["reduzieren", str(tmp_path / "k"), "--ziel", "browser"])
    assert "Browser" in capsys.readouterr().out
    cli(["moeglichkeiten", "-o", str(tmp_path / "m.md")])
    assert "Modelle eindampfen" in (tmp_path / "m.md").read_text(encoding="utf-8")
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.build()
    pt2 = p.export(tmp_path / "lm", "export")
    cli(["importieren", str(pt2), "--form", "1,16", "--speichern", str(tmp_path / "imp")])
    assert "ExternalModel" in capsys.readouterr().out and (tmp_path / "imp" / "modell.pt").exists()
