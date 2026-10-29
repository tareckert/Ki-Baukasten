"""Tests der Daten-Bausteine: Tokenizer, CSV-Vorverarbeitung, Texte mit Labels, Bildordner, Aufteilungen."""

import csv

import pytest
import torch

from ki_baukasten import Project
from ki_baukasten.data.files import TabularPreprocessor, split_indices
from ki_baukasten.data.tokenizers import BPETokenizer, CharTokenizer, WordTokenizer, make_tokenizer
from ki_baukasten.training.data_utils import DictDataset, class_labels, split_dataset

TEXT = "Der Hund läuft. Der Hund bellt laut!\nDie Katze schläft. Die Katze schnurrt leise."


def test_tokenizers_roundtrip():
    for tok in (CharTokenizer(TEXT), BPETokenizer(TEXT, vocab_size=80)):
        assert tok.decode(tok.encode(TEXT)) == TEXT
    bpe = BPETokenizer(TEXT, vocab_size=80)
    assert len(bpe.encode(TEXT)) < len(CharTokenizer(TEXT).encode(TEXT))      # BPE verdichtet
    word = WordTokenizer(TEXT, vocab_size=50)
    assert word.decode(word.encode("der hund schläft.")) == "der hund schläft."
    assert "<unk>" in word.decode(word.encode("der elefant"))
    with pytest.raises(ValueError):
        make_tokenizer("gibtsnicht", TEXT)


def test_split_strategies():
    labels = [0] * 80 + [1] * 20
    train, val = split_indices(100, 0.2, "stratifiziert", labels, 0)
    assert sum(labels[i] for i in val) == 4 and len(val) == 20
    train, val = split_indices(100, 0.2, "zeitlich", None, 0)
    assert val == list(range(80, 100)) and max(train) < min(val)
    ds = DictDataset({"inputs": torch.randn(100, 2), "targets": torch.tensor(labels)})
    rest, held = split_dataset(ds, 0.2, 0, "stratifiziert")
    assert int(class_labels(held).sum()) == 4
    rest2, val2 = split_dataset(rest, 0.25, 0, "stratifiziert")          # verschachtelt: Labels bleiben auffindbar
    assert class_labels(val2) is not None and len(val2) == 20


def test_tabular_preprocessor_fits_on_train_only():
    header = ["alter", "stadt", "einkommen"]
    rows = [["30", "Berlin", "3000"], ["40", "Hamburg", ""], ["", "Berlin", "5000"], ["50", "München", "4000"]]
    prep = TabularPreprocessor("standard", "median", "onehot", 10, True, []).fit(header, rows)
    x = prep.transform(rows)
    assert x.shape[0] == 4 and not torch.isnan(x).any()
    names = prep.feature_names
    assert any("stadt" in n for n in names) and any("fehlt" in n or "missing" in n for n in names)
    unseen = prep.transform([["35", "Köln", "3500"]])                     # unbekannte Kategorie → keine Spalte an
    assert not torch.isnan(unseen).any()


def _write_csv(path, n=200):
    g = torch.Generator().manual_seed(0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["groesse", "farbe", "gewicht", "klasse"])
        for i in range(n):
            size = float(torch.rand(1, generator=g)) * 10
            color = ["rot", "gruen", "blau"][i % 3]
            weight = "" if i % 17 == 0 else f"{size * 2 + float(torch.randn(1, generator=g)):.2f}"
            w.writerow([f"{size:.2f}", color, weight, "gross" if size > 5 else "klein"])


def test_csv_dataset_trains_with_categories_and_missing(tmp_path):
    path = tmp_path / "daten.csv"
    _write_csv(path)
    p = Project.from_preset("tabellen_klassifikation")
    p.set("data", {"name": "csv", "path": str(path), "target": "klasse", "split": "stratifiziert"})
    p.set("training.max_steps", 150)
    p.set("training.verbose", False)
    info = p.data().info
    assert info["label_names"] == ["gross", "klein"] and info["input_dim"] >= 5
    p.train()
    assert p.evaluate()["val_accuracy"] > 0.8


def test_text_classification_demo():
    p = Project.from_preset("text_klassifikation")
    p.set("data", {"name": "text_classification", "tokenizer": "bpe", "vocab_size": 200, "seq_len": 24,
                   "n_samples": 300})
    p.set("training.max_steps", 5)
    p.set("training.verbose", False)
    b = p.data()
    assert b.train["inputs"].shape[1] == 24 and b.info["label_names"] == ["negativ", "positiv"]
    p.train()
    assert "val_f1" in p.evaluate()


def test_image_folder(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    for cls, color in (("rot", (255, 0, 0)), ("blau", (0, 0, 255))):
        (tmp_path / cls).mkdir()
        for i in range(6):
            PIL.new("RGB", (20, 16), tuple(max(0, c - 10 * i) for c in color)).save(tmp_path / cls / f"{i}.png")
    p = Project.from_preset("bildklassifikation_cnn")
    p.set("data", {"name": "image_folder", "path": str(tmp_path), "size": [16, 16], "val_fraction": 0.25})
    b = p.data()
    assert b.train["inputs"].shape[1:] == (3, 16, 16)
    assert sorted(b.info["label_names"]) == ["blau", "rot"]


def test_data_check_finds_problems_and_fixes_them():
    from ki_baukasten.analysis.datacheck import check_data

    x = torch.randn(400, 3)
    x[:, 2] = 7.0                                            # konstant
    x[:5, 0] = float("nan")
    y = (torch.rand(400) < 0.08).long()                     # stark unausgewogen
    x[:, 1] = y.float() * 3 + 0.001 * torch.randn(400)      # Datenleck
    val = {"inputs": x[:40].clone(), "targets": y[:40]}     # Überlappung mit dem Training
    rep = check_data({"inputs": x, "targets": y}, val)
    checks = {f.check for f in rep.problems("hinweis")}
    assert {"nan", "konstant", "klassen", "leck", "ueberlappung"} <= checks
    assert not rep.ok and rep.settings()["training.class_weights"] in ("wurzel", "ausgeglichen")
    assert "Datenprüfung" in str(rep)


def test_project_data_check_and_apply():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.verbose", False)
    rep = p.data_check()
    assert rep.ok
    changed = p.apply_data_fixes()
    assert any("test_split" in c for c in changed)
    assert p.config.training.test_split > 0


def test_drift_report_detects_shift():
    from ki_baukasten.analysis.drift import drift_report

    ref = torch.randn(2000, 3)
    same = drift_report(ref, torch.randn(1000, 3))
    assert same.verdict == "stabil"
    moved = torch.randn(1000, 3)
    moved[:, 1] += 1.5
    rep = drift_report(ref, moved, names=["a", "b", "c"])
    assert rep.verdict == "deutlich" and [d["merkmal"] for d in rep.drifted] == ["b"]
    tokens = drift_report(torch.randint(2, 10, (200, 16)), torch.randint(10, 20, (200, 16)))
    assert tokens.kind == "tokens" and tokens.verdict == "deutlich"


def test_project_drift_report_with_prediction_drift():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("training.max_steps", 20)
    p.set("training.verbose", False)
    p.train()
    x = p.data().train["inputs"]
    rep = p.drift_report({"inputs": x[:300] + 3.0})
    assert rep.prediction is not None and rep.verdict == "deutlich"
    assert "Drift-Bericht" in str(rep)
