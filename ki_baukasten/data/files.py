"""
ki_baukasten.data.files
=======================

**Eigene Daten** aus Dateien – so vorbereitet, wie es für gute Ergebnisse nötig ist.

=====================  =======================================================================================
Datensatz              Was passiert
=====================  =======================================================================================
``csv``                Tabellen: Zahlen- **und** Kategorie-Spalten (One-Hot/Index), fehlende Werte ergänzen
                       (+ Indikator), Skalierung (Standard/Min-Max/robust) – Statistiken nur aus den
                       Trainingszeilen (keine Datenleckage), Text-Labels → Klassen, Ziel-Transformation
``text_classification``  Texte mit Labels (CSV/TSV) oder eingebautes Beispiel; Tokenizer zeichen/wort/bpe,
                       Auffüllen/Kürzen auf ``seq_len`` mit Attention-Maske
``image_folder``       Bilder aus ``ordner/<klasse>/*.png|jpg`` (Pillow), Größe anpassen, normieren
=====================  =======================================================================================

Aufteilung im Lader (``val_fraction``) mit ``split = zufall | stratifiziert | zeitlich``: dann werden
Statistiken (Mittelwerte, Skalen, Vokabular) nur auf dem Trainingsteil berechnet.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ..core import BaseConfig, Level, Param, T
from .tokenizers import KINDS as TOKENIZERS, PAD, make_tokenizer

MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-", "#n/a"}
SPLITS = {"zufall": "zufällige Auswahl", "stratifiziert": "Klassenanteile bleiben in beiden Teilen gleich",
          "zeitlich": "die letzten Zeilen werden Validierung (Zeitreihen, keine Zukunftsinformation im Training)"}


def split_indices(n: int, fraction: float, strategy: str = "zufall", labels: list | None = None,
                  seed: int = 0) -> tuple[list[int], list[int]]:
    """Indizes für (Training, Validierung) nach Strategie."""
    n_val = int(round(n * fraction))
    if fraction <= 0 or n_val == 0 or n_val >= n:
        return list(range(n)), []
    if strategy == "zeitlich":
        return list(range(n - n_val)), list(range(n - n_val, n))
    rng = random.Random(seed)
    if strategy == "stratifiziert" and labels is not None:
        by: dict[Any, list[int]] = {}
        for i, y in enumerate(labels):
            by.setdefault(y, []).append(i)
        train, val = [], []
        for idx in by.values():
            rng.shuffle(idx)
            k = int(round(len(idx) * fraction))
            if len(idx) > 1:
                k = min(max(k, 1), len(idx) - 1)
            val += idx[:k]
            train += idx[k:]
        return sorted(train), sorted(val)
    perm = list(range(n))
    rng.shuffle(perm)
    return sorted(perm[n_val:]), sorted(perm[:n_val])


# ===========================================================================
# Tabellen (CSV)
# ===========================================================================

def _num(v: str) -> float | None:
    v = v.strip()
    if v.lower() in MISSING:
        return None
    try:
        f = float(v.replace(",", ".")) if v.count(",") == 1 and "." not in v else float(v)
        return f if math.isfinite(f) else None
    except ValueError:
        return None


class TabularPreprocessor:
    """Lernt Spaltentypen, Kategorien, Ergänzungswerte und Skalen auf Trainingszeilen; wendet sie auf neue an."""

    def __init__(self, scaling: str = "standard", missing: str = "mittelwert", encoding: str = "onehot",
                 max_categories: int = 30, indicator: bool = True, categorical: list[str] | None = None):
        self.scaling, self.missing, self.encoding = scaling, missing, encoding
        self.max_categories, self.indicator, self.forced = max_categories, indicator, set(categorical or [])
        self.columns: list[dict[str, Any]] = []

    def fit(self, header: list[str], rows: list[list[str]]) -> "TabularPreprocessor":
        self.columns = []
        for j, name in enumerate(header):
            values = [r[j] if j < len(r) else "" for r in rows]
            present = [v for v in values if v.strip().lower() not in MISSING]
            nums = [_num(v) for v in present]
            numeric = name not in self.forced and present and all(x is not None for x in nums)
            col: dict[str, Any] = {"name": name, "numeric": bool(numeric),
                                   "has_missing": len(present) < len(values)}
            if numeric:
                xs = sorted(float(x) for x in nums)  # type: ignore[arg-type]
                mean = sum(xs) / len(xs)
                median = xs[len(xs) // 2]
                col["fill"] = {"mittelwert": mean, "median": median, "null": 0.0}.get(self.missing, mean)
                std = (sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
                q1, q3 = xs[len(xs) // 4], xs[(3 * len(xs)) // 4]
                col["scale"] = {"standard": (mean, std or 1.0), "minmax": (xs[0], (xs[-1] - xs[0]) or 1.0),
                                "robust": (median, (q3 - q1) or 1.0), "keine": (0.0, 1.0)}[self.scaling]
            else:
                counts = Counter(v.strip() for v in present)
                cats = [c for c, _ in counts.most_common(self.max_categories)]
                col["categories"] = cats
                col["fill"] = cats[0] if cats else ""
            self.columns.append(col)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Gelernter Zustand als JSON-fähiges dict (zum Speichern mit dem Modell)."""
        return {"scaling": self.scaling, "missing": self.missing, "encoding": self.encoding,
                "max_categories": self.max_categories, "indicator": self.indicator, "forced": sorted(self.forced),
                "columns": [{**c, "scale": list(c["scale"])} if "scale" in c else dict(c) for c in self.columns]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TabularPreprocessor":
        prep = cls(d["scaling"], d["missing"], d["encoding"], d["max_categories"], d["indicator"], d["forced"])
        prep.columns = [{**c, "scale": tuple(c["scale"])} if "scale" in c else dict(c) for c in d["columns"]]
        return prep

    @property
    def input_columns(self) -> list[str]:
        """Spalten, die neue Zeilen enthalten müssen (in dieser Reihenfolge)."""
        return [c["name"] for c in self.columns]

    @property
    def feature_names(self) -> list[str]:
        names = []
        for c in self.columns:
            if c["numeric"]:
                names.append(c["name"])
            elif self.encoding == "onehot":
                names += [f"{c['name']}={k}" for k in c["categories"]] + [f"{c['name']}=<andere>"]
            else:
                names.append(c["name"])
            if self.indicator and c["has_missing"]:
                names.append(f"{c['name']}_fehlt")
        return names

    def transform(self, rows: list[list[str]]) -> torch.Tensor:
        out = []
        for r in rows:
            feats: list[float] = []
            for j, c in enumerate(self.columns):
                raw = r[j] if j < len(r) else ""
                missing = raw.strip().lower() in MISSING
                if c["numeric"]:
                    v = c["fill"] if missing or _num(raw) is None else float(_num(raw))  # type: ignore[arg-type]
                    loc, scale = c["scale"]
                    feats.append((v - loc) / scale)
                else:
                    v = c["fill"] if missing else raw.strip()
                    cats = c["categories"]
                    if self.encoding == "onehot":
                        feats += [1.0 if v == k else 0.0 for k in cats] + [0.0 if v in cats else 1.0]
                    else:
                        feats.append(float(cats.index(v) + 1 if v in cats else 0))
                if self.indicator and c["has_missing"]:
                    feats.append(1.0 if missing else 0.0)
            out.append(feats)
        return torch.tensor(out, dtype=torch.float32)


class CSVConfig(BaseConfig):
    __title__ = "CSV-Tabelle"
    path = Param("daten.csv", tooltip=T(
        "CSV-Datei mit Kopfzeile.", "Eigene Tabellendaten – Zahlen- und Kategorie-Spalten.", "Pfad zur Datei."))
    target = Param(None, type=str, tooltip=T(
        "Zielspalte (None = letzte Spalte).", "Was vorhergesagt werden soll.", "Spaltenname."))
    task = Param("auto", choices={"auto": "Text oder wenige ganzzahlige Werte → Klassifikation, sonst Regression",
                                  "classification": "Klassifikation", "regression": "Regression"},
                 tooltip=T("Aufgabentyp.", "Bestimmt Verlust und Metriken.", "auto."))
    delimiter = Param(",", tooltip=T("Trennzeichen.", "CSV-Dialekt.", "',' oder ';'."), level=Level.ADVANCED)
    drop_columns = Param([], item_type=str, tooltip=T(
        "Spalten, die nicht verwendet werden (z. B. IDs).", "IDs/Namen führen zu Auswendiglernen statt Lernen.",
        "Kundennummer, Zeitstempel als Text …"))
    categorical = Param([], item_type=str, tooltip=T(
        "Spalten, die immer als Kategorie behandelt werden (auch wenn sie Zahlen enthalten).",
        "Postleitzahlen oder Codes sind Kategorien, keine Mengen.", "z. B. ['plz', 'produkt_id']."),
        level=Level.ADVANCED)
    encoding = Param("onehot", choices={"onehot": "eine 0/1-Spalte je Kategorie", "ordinal": "eine Zahl je Kategorie"},
                     tooltip=T("Kodierung von Kategorie-Spalten.", "One-Hot ist für neuronale Netze meist besser.",
                               "onehot."), level=Level.ADVANCED)
    max_categories = Param(30, min=1, tooltip=T(
        "Höchstens so viele Kategorien je Spalte, seltene → '<andere>'.", "Begrenzt die Merkmalszahl.", "10–100."),
        level=Level.ADVANCED)
    missing = Param("mittelwert", choices={"mittelwert": "Mittelwert der Trainingszeilen", "median": "Median (robust)",
                                           "null": "0", "zeile_entfernen": "Zeilen mit Lücken weglassen"},
                    tooltip=T("Umgang mit fehlenden Werten.", "Lücken sind in echten Daten die Regel.", "median bei "
                              "Ausreißern."))
    missing_indicator = Param(True, tooltip=T(
        "Zusätzliche 0/1-Spalte 'fehlt' für Spalten mit Lücken.", "Oft ist schon das Fehlen eine Information.", "an."),
        level=Level.ADVANCED)
    scaling = Param("standard", choices={"standard": "Mittel 0, Std 1", "minmax": "auf 0…1", "robust":
                                         "Median/Quartilsabstand (robust gegen Ausreißer)", "keine": "unverändert"},
                    tooltip=T("Skalierung der Zahlen-Spalten.", "Neuronale Netze lernen mit ähnlich skalierten "
                              "Merkmalen viel besser.", "standard; robust bei Ausreißern."))
    standardize = Param(True, tooltip=T(
        "Kompatibilität: aus = keine Skalierung (überschreibt 'scaling').", "Ältere Konfigurationen.", "an."),
        level=Level.EXPERT)
    target_scaling = Param("auto", choices={"auto": "Regression: standardisieren (Mittel 0, Std 1)",
                                            "keine": "unverändert", "standard": "Mittel 0, Std 1",
                                            "log": "log(1+y) für schiefe, positive Ziele"}, tooltip=T(
        "Transformation des Regressionsziels (Metriken und Vorhersagen bleiben in echten Einheiten).",
        "Ziele in großen Einheiten (Preise ~ 100 000) lernt ein Netz sonst kaum; schiefe Ziele besser logarithmisch.",
        "auto; log bei Preisen/Umsätzen mit großer Spannweite."), level=Level.ADVANCED, off="keine", on="auto")
    val_fraction = Param(0.0, min=0.0, max=0.9, tooltip=T(
        "Validierungsanteil direkt beim Laden (0 = später im Training abteilen).",
        "Beim Laden abgeteilt werden Mittelwerte/Skalen nur aus Trainingszeilen berechnet – keine Datenleckage.",
        "0.1–0.2."))
    split = Param("stratifiziert", choices=SPLITS, tooltip=T(
        "Art der Aufteilung (bei val_fraction > 0).", "Zeitreihen immer 'zeitlich' – sonst sieht das Modell die "
        "Zukunft.", "Klassifikation: stratifiziert."))
    seed = Param(0, min=0, tooltip=T("Seed der Aufteilung.", "Reproduzierbar.", "0."), level=Level.ADVANCED)


def load_csv(a: CSVConfig) -> Any:
    from .datasets import DatasetBundle

    with open(a.path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter=a.delimiter))
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    target = a.target or header[-1]
    if target not in header:
        raise ValueError(f"Zielspalte '{target}' nicht in {header}.")
    ti = header.index(target)
    keep = [j for j, h in enumerate(header) if j != ti and h not in a.drop_columns]
    if a.missing == "zeile_entfernen":
        body = [r for r in body if all(r[j].strip().lower() not in MISSING for j in keep + [ti])]
    raw_y = [r[ti].strip() for r in body]
    numeric_y = [_num(v) for v in raw_y]
    as_numbers = all(v is not None for v in numeric_y)
    is_cls = a.task == "classification" or (a.task == "auto" and (not as_numbers or (
        all(float(v).is_integer() for v in numeric_y) and len(set(numeric_y)) <= 50)))  # type: ignore[arg-type]
    if is_cls:
        labels = sorted(set(raw_y), key=lambda v: (_num(v) is None, _num(v) or 0.0, v))
        y = torch.tensor([labels.index(v) for v in raw_y])
    else:
        y = torch.tensor([float(v) for v in numeric_y], dtype=torch.float32).unsqueeze(1)  # type: ignore[arg-type]
    train_idx, val_idx = split_indices(len(body), a.val_fraction, a.split, y.tolist() if is_cls else None, a.seed)
    sub_header = [header[j] for j in keep]
    project = (lambda idx: [[body[i][j] if j < len(body[i]) else "" for j in keep] for i in idx])
    prep = TabularPreprocessor("keine" if not a.standardize else a.scaling, a.missing, a.encoding, a.max_categories,
                               a.missing_indicator, a.categorical).fit(sub_header, project(train_idx))
    info: dict[str, Any] = {"task": "classification" if is_cls else "regression", "features": prep.feature_names,
                            "preprocessor": prep}
    if not is_cls and a.target_scaling != "keine":         # 'auto' = standard
        tr = y[train_idx]
        if a.target_scaling == "log":
            y = torch.log1p(y.clamp_min(0))
            info["target_transform"] = {"art": "log"}
        else:
            m, s = float(tr.mean()), float(tr.std()) or 1.0
            y = (y - m) / s
            info["target_transform"] = {"art": "standard", "mittel": m, "std": s}
    x_train = prep.transform(project(train_idx))
    info.update({"input_dim": x_train.shape[1], "output_dim": len(labels) if is_cls else 1})
    if is_cls:
        info["label_names"] = labels
    train = {"inputs": x_train, "targets": y[train_idx]}
    val = {"inputs": prep.transform(project(val_idx)), "targets": y[val_idx]} if val_idx else None
    return DatasetBundle(train, val, info)


# ===========================================================================
# Texte mit Labels
# ===========================================================================

_POS = ["großartig", "toll", "hervorragend", "super", "schön", "gelungen", "begeistert", "zufrieden", "prima", "klasse"]
_NEG = ["schlecht", "furchtbar", "enttäuschend", "kaputt", "langweilig", "ärgerlich", "mies", "teuer", "laut", "falsch"]
_THINGS = ["das Produkt", "der Service", "die Lieferung", "das Essen", "der Film", "die App", "das Hotel", "das Buch"]


def _demo_texts(n: int, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n):
        pos = rng.random() < 0.5
        word = rng.choice(_POS if pos else _NEG)
        thing = rng.choice(_THINGS)
        neg = rng.random() < 0.2                       # Verneinung kehrt die Stimmung um
        tmpl = rng.choice(["{t} war {n}{w}.", "Ich finde {t} {n}{w}!", "Insgesamt: {t} ist {n}{w}.",
                           "{t} – {n}{w}, wirklich."])
        texts.append(tmpl.format(t=thing, w=word, n="nicht " if neg else "").capitalize())
        labels.append("positiv" if pos != neg else "negativ")
    return texts, labels


class TextClassificationConfig(BaseConfig):
    __title__ = "Texte mit Labels"
    path = Param(None, type=str, nullable=True, tooltip=T(
        "CSV/TSV-Datei mit Text- und Label-Spalte (None = eingebautes Stimmungs-Beispiel).",
        "Eigene Textklassifikation (Spam, Stimmung, Themen …).", "Pfad zur Datei."))
    text_column = Param("text", tooltip=T("Spalte mit dem Text.", "Eingabe.", "text."))
    label_column = Param("label", tooltip=T("Spalte mit dem Label.", "Ziel.", "label."))
    delimiter = Param("auto", tooltip=T("Trennzeichen ('auto' = Tab bei .tsv, sonst Komma).", "Dateiformat.", "auto."),
                      level=Level.ADVANCED)
    tokenizer = Param("bpe", choices=TOKENIZERS, tooltip=T(
        "Tokenizer.", "BPE ist der heutige Standard: kompakte Folgen, keine unbekannten Wörter.", "bpe."))
    vocab_size = Param(1000, min=8, tooltip=T(
        "Vokabulargröße (wort/bpe).", "Größer = kürzere Folgen, mehr Parameter.", "500–30.000."))
    seq_len = Param(64, min=4, tooltip=T(
        "Länge nach Auffüllen/Kürzen.", "Längere Texte werden abgeschnitten.", "95 % der Texte sollten passen."))
    n_samples = Param(800, min=10, tooltip=T("Anzahl Beispiele im eingebauten Beispiel.", "Nur ohne path.", "800."))
    val_fraction = Param(0.15, min=0.0, max=0.9, tooltip=T(
        "Validierungsanteil (Tokenizer wird nur auf dem Trainingsteil gelernt).", "Ehrliche Bewertung.", "0.1–0.2."))
    split = Param("stratifiziert", choices=SPLITS, tooltip=T("Art der Aufteilung.", "Klassen fair verteilt.", ""))
    seed = Param(0, min=0, tooltip=T("Seed.", "Reproduzierbar.", "0."), level=Level.ADVANCED)


def load_text_classification(a: TextClassificationConfig) -> Any:
    from .datasets import DatasetBundle

    if a.path:
        delim = ("\t" if str(a.path).endswith((".tsv", ".tab")) else ",") if a.delimiter == "auto" else a.delimiter
        with open(a.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            pairs = [(r[a.text_column], r[a.label_column]) for r in reader if r.get(a.text_column)]
        texts, raw = [p[0] for p in pairs], [p[1].strip() for p in pairs]
    else:
        texts, raw = _demo_texts(a.n_samples, a.seed)
    labels = sorted(set(raw))
    y = torch.tensor([labels.index(v) for v in raw])
    train_idx, val_idx = split_indices(len(texts), a.val_fraction, a.split, y.tolist(), a.seed)
    tok = make_tokenizer(a.tokenizer, "\n".join(texts[i] for i in train_idx), a.vocab_size)

    def encode(idx: list[int]) -> dict[str, torch.Tensor]:
        ids = torch.full((len(idx), a.seq_len), PAD, dtype=torch.long)
        for row, i in enumerate(idx):
            enc = tok.encode(texts[i])[: a.seq_len]
            ids[row, : len(enc)] = torch.tensor(enc, dtype=torch.long)
        return {"inputs": ids, "attention_mask": ids != PAD, "targets": y[idx]}

    lengths = [len(tok.encode(texts[i])) for i in train_idx[:500]]
    info = {"vocab_size": tok.vocab_size, "tokenizer": tok, "seq_len": a.seq_len, "output_dim": len(labels),
            "label_names": labels, "task": "classification",
            "truncated_fraction": sum(n > a.seq_len for n in lengths) / max(1, len(lengths))}
    return DatasetBundle(encode(train_idx), encode(val_idx) if val_idx else None, info)


# ===========================================================================
# Bilder aus Ordnern
# ===========================================================================

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")


class ImageFolderConfig(BaseConfig):
    __title__ = "Bilder aus Ordnern"
    path = Param("bilder", tooltip=T(
        "Ordner mit einem Unterordner je Klasse (bilder/katze/*.jpg, bilder/hund/*.jpg).",
        "Der übliche Aufbau von Bilddatensätzen (wie ImageFolder).", "Pfad."))
    val_path = Param(None, type=str, nullable=True, tooltip=T(
        "Eigener Validierungsordner gleichen Aufbaus (sonst val_fraction).", "Feste Testbilder.", "Optional."))
    size = Param([64, 64], item_type=int, length=2, min=4, tooltip=T(
        "Zielgröße [Höhe, Breite].", "Alle Bilder brauchen dieselbe Größe; größer = genauer, langsamer.",
        "32–224."))
    channels = Param(3, choices=[1, 3], tooltip=T("1 = Graustufen, 3 = Farbe.", "Eingabekanäle.", "3."))
    normalize = Param("datensatz", choices={"datensatz": "Mittel/Std je Kanal aus den Trainingsbildern",
                                            "imagenet": "ImageNet-Werte (für vortrainierte Modelle)",
                                            "halb": "auf −1…1", "keine": "0…1"}, tooltip=T(
        "Normierung der Pixel.", "Stabileres, schnelleres Training.", "datensatz; vortrainiert: imagenet."))
    max_per_class = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Höchstens so viele Bilder je Klasse (None = alle).", "Schnelle Versuche / Ausgleich.", "None."),
        level=Level.ADVANCED)
    val_fraction = Param(0.15, min=0.0, max=0.9, tooltip=T(
        "Validierungsanteil ohne val_path.", "Ehrliche Bewertung.", "0.1–0.2."))
    seed = Param(0, min=0, tooltip=T("Seed.", "Reproduzierbar.", "0."), level=Level.ADVANCED)


def _load_image(path: Path, size: list[int], channels: int) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB" if channels == 3 else "L").resize((size[1], size[0]))
        data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return data.view(size[0], size[1], channels).permute(2, 0, 1).float() / 255.0


def _scan(folder: Path, classes: list[str] | None, limit: int | None) -> tuple[list[Path], list[str], list[str]]:
    classes = classes or sorted(d.name for d in folder.iterdir() if d.is_dir())
    files, labels = [], []
    for c in classes:
        found = sorted(p for p in (folder / c).rglob("*") if p.suffix.lower() in _IMG_EXT) if (folder / c).exists() \
            else []
        found = found[:limit] if limit else found
        files += found
        labels += [c] * len(found)
    return files, labels, classes


def load_image_folder(a: ImageFolderConfig) -> Any:
    from .datasets import DatasetBundle

    root = Path(a.path)
    if not root.is_dir():
        raise ValueError(f"Bildordner '{root}' nicht gefunden (Aufbau: ordner/<klasse>/<bild>).")
    files, labels, classes = _scan(root, None, a.max_per_class)
    if not files:
        raise ValueError(f"Keine Bilder in '{root}' gefunden ({', '.join(_IMG_EXT)}).")
    y = torch.tensor([classes.index(c) for c in labels])
    if a.val_path:
        vfiles, vlabels, _ = _scan(Path(a.val_path), classes, a.max_per_class)
        train_idx, val_idx = list(range(len(files))), []
    else:
        train_idx, val_idx = split_indices(len(files), a.val_fraction, "stratifiziert", y.tolist(), a.seed)
        vfiles, vlabels = [], []
    x = torch.stack([_load_image(f, a.size, a.channels) for f in files])
    xt = x[train_idx]
    if a.normalize == "datensatz":
        mean, std = xt.mean((0, 2, 3)), xt.std((0, 2, 3)).clamp_min(1e-3)
    elif a.normalize == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406][: a.channels])
        std = torch.tensor([0.229, 0.224, 0.225][: a.channels])
    elif a.normalize == "halb":
        mean, std = torch.full((a.channels,), 0.5), torch.full((a.channels,), 0.5)
    else:
        mean, std = torch.zeros(a.channels), torch.ones(a.channels)
    norm = (lambda t: (t - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1))
    train = {"inputs": norm(xt), "targets": y[train_idx]}
    if vfiles:
        val = {"inputs": norm(torch.stack([_load_image(f, a.size, a.channels) for f in vfiles])),
               "targets": torch.tensor([classes.index(c) for c in vlabels])}
    elif val_idx:
        val = {"inputs": norm(x[val_idx]), "targets": y[val_idx]}
    else:
        val = None
    info = {"in_channels": a.channels, "input_size": list(a.size), "output_dim": len(classes), "label_names": classes,
            "task": "classification", "normalize": {"mean": mean.tolist(), "std": std.tolist()}}
    return DatasetBundle(train, val, info)
