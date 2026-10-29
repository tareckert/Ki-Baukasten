"""
ki_baukasten.data.datasets
==========================

Datensätze: synthetische Beispieldaten für jede Modellart **und** Lader für
eigene Daten (CSV, Textdateien, Tensor-Dateien).

Warum synthetische Daten?
-------------------------
Jede Vorlage (Preset) soll sofort lauffähig sein – ohne Downloads. Die
synthetischen Aufgaben sind klein, aber so gebaut, dass man am Lernerfolg
sieht, ob eine Konfiguration funktioniert (z. B. Spiralen trennen,
Sinuswellen vorhersagen, Zeichenfolgen fortsetzen).

Registry ``dataset``
--------------------
Fabrik-Signatur: ``factory(args) -> DatasetBundle`` mit ``train``, optional
``val`` und ``info`` (z. B. Tokenizer, Vokabulargröße, Eingabeform).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..core import BaseConfig, Level, Param, T, create_registry
from .files import (CSVConfig, ImageFolderConfig, TextClassificationConfig, load_csv, load_image_folder,
                    load_text_classification)
from .tokenizers import KINDS as TOKENIZERS, CharTokenizer, make_tokenizer  # noqa: F401 - CharTokenizer: Export

DATASETS = create_registry(
    "dataset",
    "Datensätze",
    T(
        "Datenquellen: synthetische Beispielaufgaben und Lader für eigene Dateien.",
        "Sofort lauffähige Experimente und einfacher Weg zu eigenen Daten.",
        "Jedes Training mit Daten.",
    ),
    context_doc="factory(args) -> DatasetBundle(train, val, info)",
)


@dataclass
class DatasetBundle:
    """Ergebnis eines Datensatzes: Trainings-/Validierungsdaten + Zusatzinfos."""

    train: Any
    val: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    test: Any = None            # optionale, nie fürs Training benutzte Testmenge (Project.test)


BEISPIELTEXT = (
    "Der Baukasten erlaubt es, künstliche Intelligenz Stück für Stück zusammenzusetzen. "
    "Jede Schicht, jede Funktion und jede Belohnung lässt sich einstellen. "
    "Ein Modell lernt aus Daten, indem es seine Gewichte Schritt für Schritt anpasst. "
    "Die Lernrate bestimmt, wie groß diese Schritte sind. "
    "Aufmerksamkeit verbindet Wörter miteinander, Faltungen erkennen Muster in Bildern, "
    "und rekurrente Netze erinnern sich an das, was zuvor geschah. "
    "Wer experimentiert, entdeckt neue Wege. Wer misst, versteht, was wirklich hilft. "
)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ===========================================================================
# Konfigurationen
# ===========================================================================

class SizeConfig(BaseConfig):
    n_samples = Param(1000, min=2, tooltip=T("Anzahl Beispiele.", "Mehr Daten = bessere Generalisierung.", "500–10000."))
    seed = Param(0, tooltip=T("Seed der Datenerzeugung.", "Reproduzierbare Daten.", "0."), level=Level.ADVANCED)


class ClassificationDataConfig(SizeConfig):
    kind = Param("spirals", choices={"blobs": "Gauß-Wolken", "xor": "XOR-Quadranten", "spirals": "verschlungene Spiralen",
                                     "moons": "zwei Halbmonde", "rings": "konzentrische Ringe"}, tooltip=T(
        "Form der Klassen.", "Steuert die Schwierigkeit (blobs leicht, spirals schwer).", "spirals für Nichtlinearität."))
    n_classes = Param(3, min=2, tooltip=T("Anzahl Klassen.", "Ausgabegröße.", "2–10."))
    n_features = Param(2, min=2, tooltip=T(
        "Merkmale (2 = sichtbar; weitere sind Rauschdimensionen).", "Zusätzliche irrelevante Merkmale testen Robustheit.",
        "2–50."))
    noise = Param(0.05, min=0.0, tooltip=T(
        "Rauschen.", "Überlappung der Klassen (ab ~0.1 überlappen die Spiralarme im Zentrum stark).", "0.02–0.2."))


class RegressionDataConfig(SizeConfig):
    function = Param("sin", choices={"linear": "y = w·x", "sin": "y = Σ sin(x)", "poly": "Polynom 3. Grades",
                                     "friedman": "Friedman-1 (klassischer Benchmark)"}, tooltip=T(
        "Zielfunktion.", "Schwierigkeit der Regression.", "sin oder friedman."))
    n_features = Param(5, min=1, tooltip=T("Merkmale.", "Eingabebreite.", "5–20."))
    noise = Param(0.05, min=0.0, tooltip=T("Rauschen auf dem Ziel.", "Messfehler.", "0–0.2."))


class SequenceDataConfig(SizeConfig):
    task = Param("parity", choices={"parity": "Parität der Einsen (Gedächtnis über die ganze Folge)",
                                    "contains": "enthält die Folge ein bestimmtes Token? (Klassen ausgewogen)",
                                    "majority": "häufigstes Token"}, tooltip=T(
        "Sequenzklassifikationsaufgabe.", "Testet Langzeitgedächtnis / globale Aggregation.", "parity ist schwer."))
    seq_len = Param(16, min=2, tooltip=T("Folgenlänge.", "Länger = schwieriger.", "8–64."))
    vocab_size = Param(8, min=2, tooltip=T("Vokabular.", "Anzahl Symbole.", "2–20."))


class TextDataConfig(BaseConfig):
    text = Param(None, type=str, tooltip=T(
        "Eigener Text (None = eingebauter Beispieltext).", "Trainingsmaterial des Sprachmodells.", "Beliebiger Text."))
    path = Param(None, type=str, tooltip=T(
        "Textdatei (UTF-8) statt 'text'.", "Eigene Korpora.", "z. B. buch.txt."))
    seq_len = Param(64, min=2, tooltip=T("Länge der Trainingsausschnitte.", "Kontextlänge beim Training.", "32–512."))
    stride = Param(8, min=1, tooltip=T("Schrittweite der Ausschnitte.", "Kleiner = mehr (überlappende) Beispiele.", "seq_len/4."))
    repeat = Param(4, min=1, tooltip=T("Text so oft wiederholen (für sehr kurze Texte).", "Mehr Trainingsbeispiele.", "1–20."),
                   level=Level.ADVANCED)
    tokenizer = Param("zeichen", choices=TOKENIZERS, tooltip=T(
        "Tokenizer für den Text.", "Zeichen: winziges Vokabular; BPE: kürzere Folgen, mehr Inhalt pro Kontext (wie "
        "GPT/LLaMA).", "Kleine Texte: zeichen; größere Korpora: bpe."))
    vocab_size = Param(500, min=8, tooltip=T(
        "Vokabulargröße für wort/bpe.", "Größer = kürzere Folgen, größere Einbettungen.", "500–32.000."),
        level=Level.ADVANCED)


class Seq2SeqDataConfig(SizeConfig):
    task = Param("reverse", choices={"reverse": "Folge umkehren", "copy": "Folge kopieren", "sort": "Folge sortieren",
                                     "shift": "jedes Token +1"}, tooltip=T(
        "Sequenz-zu-Sequenz-Aufgabe.", "Testet Encoder-Decoder-Fähigkeiten.", "reverse."))
    seq_len = Param(8, min=2, tooltip=T("Folgenlänge.", "Schwierigkeit.", "6–20."))
    vocab_size = Param(12, min=4, tooltip=T("Vokabular (0 = Start-Token, 1.. = Symbole).", "Anzahl Symbole.", "10–50."))


class TimeSeriesDataConfig(SizeConfig):
    window = Param(32, min=2, tooltip=T("Eingabefenster (vergangene Schritte).", "Wie weit zurückgeschaut wird.", "16–128."))
    horizon = Param(4, min=1, tooltip=T("Vorhersagehorizont (zukünftige Schritte).", "Prognosereichweite.", "1–24."))
    n_series = Param(1, min=1, tooltip=T("Variablen je Zeitschritt.", "Multivariate Reihen.", "1–8."))
    noise = Param(0.05, min=0.0, tooltip=T("Rauschen.", "Schwierigkeit.", "0–0.2."))
    output = Param("vector", choices={"vector": "Ziel [N, horizon·n_series] (für pooling-Modelle)",
                                      "sequence": "Ziel [N, horizon, n_series] (für seq2seq)"}, tooltip=T(
        "Form des Ziels.", "Passend zum Modell.", "vector."), level=Level.ADVANCED)


class ImageDataConfig(SizeConfig):
    size = Param(16, min=8, tooltip=T("Bildgröße (quadratisch).", "Auflösung.", "16–32."))
    channels = Param(1, min=1, tooltip=T("Kanäle.", "1 = grau, 3 = farbig.", "1 oder 3."))
    noise = Param(0.1, min=0.0, tooltip=T("Pixelrauschen.", "Schwierigkeit.", "0.05–0.3."))


class GraphDataConfig(SizeConfig):
    n_samples = Param(400, min=2, tooltip=T("Anzahl Graphen.", "Datenmenge.", "200–2000."))
    min_nodes = Param(6, min=3, tooltip=T("Minimale Knotenzahl.", "Graphgröße.", "5–10."))
    max_nodes = Param(14, min=3, tooltip=T("Maximale Knotenzahl.", "Graphgröße.", "10–30."))
    feature_dim = Param(8, min=1, tooltip=T("Merkmale je Knoten.", "Eingabebreite.", "4–16."))


class DistributionConfig(SizeConfig):
    kind = Param("moons", choices={"moons": "zwei Halbmonde", "gaussians": "Ring aus 8 Gauß-Glocken",
                                   "ring": "Kreisring", "spiral": "Spirale"}, tooltip=T(
        "2D-Zielverteilung für generative Modelle.", "Sichtbarer Generierungserfolg.", "moons, gaussians."))
    noise = Param(0.05, min=0.0, tooltip=T("Rauschen.", "Breite der Verteilung.", "0.02–0.1."))


class PreferenceDataConfig(SizeConfig):
    dim = Param(8, min=1, tooltip=T("Merkmale je Option.", "Eingabebreite des Reward-Modells.", "4–32."))
    noise = Param(0.1, min=0.0, tooltip=T("Rauschen in den Präferenzen.", "Unzuverlässige Bewerter.", "0–0.5."))


class AnomalyDataConfig(SizeConfig):
    n_features = Param(16, min=2, tooltip=T("Merkmale.", "Eingabebreite.", "8–64."))
    anomaly_fraction = Param(0.05, min=0.0, max=0.5, tooltip=T(
        "Anteil Anomalien in der Testmenge.", "Seltenheit der Ausreißer.", "0.01–0.1."))


class TensorFileConfig(BaseConfig):
    path = Param("daten.pt", tooltip=T(
        "torch.save-Datei mit dict von Tensoren (z. B. inputs/targets).", "Beliebige vorbereitete Daten.", "Eigene Daten."))
    val_path = Param(None, type=str, tooltip=T("Optionale Validierungsdatei.", "Eigene Validierung.", "Optional."))


# ===========================================================================
# Erzeuger
# ===========================================================================

def _classification(a: ClassificationDataConfig) -> DatasetBundle:
    g = _gen(a.seed)
    n, k = a.n_samples, a.n_classes
    y = torch.randint(0, k, (n,), generator=g)
    if a.kind == "blobs":
        centers = torch.randn(k, 2, generator=g) * 3
        x = centers[y] + torch.randn(n, 2, generator=g)
    elif a.kind == "xor":
        x = torch.rand(n, 2, generator=g) * 2 - 1
        y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).long() % k
    elif a.kind == "moons":
        t = torch.rand(n, generator=g) * math.pi
        y = torch.randint(0, 2, (n,), generator=g) % k
        x = torch.stack([torch.cos(t) + y * 1.0, torch.sin(t) * (1 - 2 * y) + y * 0.5], 1)
    elif a.kind == "rings":
        r = y.float() + 1 + torch.rand(n, generator=g) * 0.5
        t = torch.rand(n, generator=g) * 2 * math.pi
        x = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    else:  # spirals
        t = torch.rand(n, generator=g) * 3 * math.pi + 0.5
        phase = y.float() * 2 * math.pi / k
        x = torch.stack([t * torch.cos(t + phase), t * torch.sin(t + phase)], 1) / (3 * math.pi)
    x = x + torch.randn(n, 2, generator=g) * a.noise
    if a.n_features > 2:
        x = torch.cat([x, torch.randn(n, a.n_features - 2, generator=g)], 1)
    return DatasetBundle({"inputs": x.float(), "targets": y}, info={"input_dim": a.n_features, "output_dim": k,
                                                                   "task": "classification"})


def _regression(a: RegressionDataConfig) -> DatasetBundle:
    g = _gen(a.seed)
    x = torch.rand(a.n_samples, a.n_features, generator=g) * 2 - 1
    if a.function == "linear":
        y = x @ torch.randn(a.n_features, generator=g)
    elif a.function == "poly":
        y = (x ** 3).sum(1) - (x ** 2).sum(1) + x.sum(1)
    elif a.function == "friedman" and a.n_features >= 5:
        u = (x + 1) / 2
        y = 10 * torch.sin(math.pi * u[:, 0] * u[:, 1]) + 20 * (u[:, 2] - 0.5) ** 2 + 10 * u[:, 3] + 5 * u[:, 4]
        y = (y - 14) / 5
    else:
        y = torch.sin(3 * x).sum(1)
    y = y + torch.randn(a.n_samples, generator=g) * a.noise
    return DatasetBundle({"inputs": x, "targets": y.unsqueeze(1)}, info={"input_dim": a.n_features, "output_dim": 1,
                                                                        "task": "regression"})


def _sequence(a: SequenceDataConfig) -> DatasetBundle:
    g = _gen(a.seed)
    x = torch.randint(1, a.vocab_size, (a.n_samples, a.seq_len), generator=g)
    if a.task == "parity":
        y = (x == 1).sum(1) % 2
    elif a.task == "contains":                     # ausgewogen: die Hälfte enthält das Token, die andere nicht
        token = a.vocab_size - 1
        x = torch.randint(1, max(2, token), (a.n_samples, a.seq_len), generator=g)
        y = (torch.rand(a.n_samples, generator=g) < 0.5).long()
        pos = torch.randint(0, a.seq_len, (a.n_samples,), generator=g)
        rows = y.bool().nonzero().squeeze(1)
        x[rows, pos[rows]] = token
    else:
        y = torch.stack([torch.bincount(r, minlength=a.vocab_size) for r in x]).argmax(1)
    return DatasetBundle({"inputs": x, "targets": y}, info={"vocab_size": a.vocab_size, "seq_len": a.seq_len,
                                                           "output_dim": int(y.max()) + 1, "task": "classification"})


def _text(a: TextDataConfig) -> DatasetBundle:
    text = Path(a.path).read_text(encoding="utf-8") if a.path else (a.text or BEISPIELTEXT)
    text = text * a.repeat
    tok = make_tokenizer(a.tokenizer, text, a.vocab_size)
    ids = torch.tensor(tok.encode(text))
    L = a.seq_len + 1
    if len(ids) < L:
        raise ValueError(f"Text zu kurz für seq_len={a.seq_len}.")
    windows = ids.unfold(0, L, a.stride)
    return DatasetBundle({"inputs": windows}, info={"tokenizer": tok, "vocab_size": tok.vocab_size,
                                                    "seq_len": L, "task": "language_modeling"})


def _seq2seq(a: Seq2SeqDataConfig) -> DatasetBundle:
    g = _gen(a.seed)
    x = torch.randint(1, a.vocab_size, (a.n_samples, a.seq_len), generator=g)
    y = {"reverse": x.flip(1), "copy": x, "sort": x.sort(1).values,
         "shift": (x % (a.vocab_size - 1)) + 1}[a.task]
    return DatasetBundle({"inputs": x, "targets": y}, info={"vocab_size": a.vocab_size, "seq_len": a.seq_len,
                                                           "bos_token_id": 0, "task": "seq2seq"})


def _timeseries(a: TimeSeriesDataConfig) -> DatasetBundle:
    g = _gen(a.seed)
    n, W, H, S = a.n_samples, a.window, a.horizon, a.n_series
    t = torch.arange(W + H).float()
    freq = torch.rand(n, 1, S, generator=g) * 0.4 + 0.1
    phase = torch.rand(n, 1, S, generator=g) * 2 * math.pi
    amp = torch.rand(n, 1, S, generator=g) + 0.5
    series = amp * torch.sin(freq * t.view(1, -1, 1) + phase) + 0.3 * torch.sin(0.05 * t.view(1, -1, 1))
    series = series + torch.randn(series.shape, generator=g) * a.noise
    x, y = series[:, :W], series[:, W:]
    if a.output == "vector":
        y = y.reshape(n, -1)
    return DatasetBundle({"inputs": x, "targets": y}, info={"input_dim": S, "output_dim": H * S, "seq_len": W,
                                                           "task": "regression"})


def _images(a: ImageDataConfig) -> DatasetBundle:
    g = _gen(a.seed)
    n, s = a.n_samples, a.size
    y = torch.randint(0, 3, (n,), generator=g)
    imgs = torch.zeros(n, a.channels, s, s)
    yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
    for i in range(n):
        cx, cy = (torch.randint(s // 4, 3 * s // 4, (2,), generator=g)).tolist()
        r = int(torch.randint(s // 6, s // 3, (1,), generator=g))
        if y[i] == 0:      # Quadrat
            mask = ((xx - cx).abs() <= r) & ((yy - cy).abs() <= r)
        elif y[i] == 1:    # Kreis
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        else:              # Kreuz
            mask = (((xx - cx).abs() <= r // 3) & ((yy - cy).abs() <= r)) | (((yy - cy).abs() <= r // 3) & ((xx - cx).abs() <= r))
        imgs[i, :, mask] = 1.0
    imgs = imgs + torch.randn(imgs.shape, generator=g) * a.noise
    return DatasetBundle({"inputs": imgs, "targets": y}, info={"in_channels": a.channels, "input_size": [s, s],
                                                              "output_dim": 3, "task": "classification",
                                                              "classes": ["Quadrat", "Kreis", "Kreuz"],
                                                              "label_names": ["Quadrat", "Kreis", "Kreuz"]})


class GraphDataset(torch.utils.data.Dataset):
    """Liste kleiner Graphen; ``collate_fn`` baut einen Batch-Graphen (x, edge_index, batch)."""

    def __init__(self, graphs: list[dict]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> dict:
        return self.graphs[i]

    @staticmethod
    def collate_fn(items: list[dict]) -> dict:
        xs, eis, batch, ys, offset = [], [], [], [], 0
        for gi, it in enumerate(items):
            n = it["x"].shape[0]
            xs.append(it["x"])
            eis.append(it["edge_index"] + offset)
            batch.append(torch.full((n,), gi, dtype=torch.long))
            ys.append(it["y"])
            offset += n
        return {"inputs": {"x": torch.cat(xs), "edge_index": torch.cat(eis, 1), "batch": torch.cat(batch)},
                "targets": torch.stack(ys)}


def _graphs(a: GraphDataConfig) -> DatasetBundle:
    """Graph-Klassifikation: Ring (Klasse 1) vs. Baum/Kette ohne Kreis (Klasse 0)."""
    g = _gen(a.seed)
    graphs = []
    for _ in range(a.n_samples):
        n = int(torch.randint(a.min_nodes, a.max_nodes + 1, (1,), generator=g))
        label = int(torch.randint(0, 2, (1,), generator=g))
        if label == 1:
            src = torch.arange(n)
            dst = (src + 1) % n
        else:
            dst = torch.arange(1, n)
            src = torch.tensor([int(torch.randint(0, i, (1,), generator=g)) for i in range(1, n)])
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        deg = torch.bincount(ei[0], minlength=n).float().unsqueeze(1)
        x = torch.cat([deg / 3, torch.randn(n, a.feature_dim - 1, generator=g) * 0.1], 1) if a.feature_dim > 1 else deg
        graphs.append({"x": x, "edge_index": ei, "y": torch.tensor(label)})
    return DatasetBundle(GraphDataset(graphs), info={"input_dim": a.feature_dim, "output_dim": 2, "task": "classification"})


def sample_distribution(kind: str, n: int, noise: float, g: torch.Generator) -> torch.Tensor:
    if kind == "gaussians":
        k = torch.randint(0, 8, (n,), generator=g).float()
        centers = torch.stack([torch.cos(k * math.pi / 4), torch.sin(k * math.pi / 4)], 1) * 2
        return centers + torch.randn(n, 2, generator=g) * max(noise, 0.02) * 2
    if kind == "ring":
        t = torch.rand(n, generator=g) * 2 * math.pi
        return torch.stack([torch.cos(t), torch.sin(t)], 1) * 1.5 + torch.randn(n, 2, generator=g) * noise
    if kind == "spiral":
        t = torch.rand(n, generator=g) * 3 * math.pi
        return torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 4 + torch.randn(n, 2, generator=g) * noise
    t = torch.rand(n, generator=g) * math.pi
    lab = torch.randint(0, 2, (n,), generator=g).float()
    x = torch.stack([torch.cos(t) + lab - 0.5, torch.sin(t) * (1 - 2 * lab) + lab * 0.5 - 0.25], 1)
    return x + torch.randn(n, 2, generator=g) * noise


def _distribution(a: DistributionConfig) -> DatasetBundle:
    x = sample_distribution(a.kind, a.n_samples, a.noise, _gen(a.seed))
    return DatasetBundle({"inputs": x}, info={"data_dim": 2, "task": "generation"})


def _preferences(a: PreferenceDataConfig) -> DatasetBundle:
    """Verstecktes 'menschliches' Urteil: w·x (mit Rauschen) entscheidet, welche Option bevorzugt wird."""
    g = _gen(a.seed)
    w = torch.randn(a.dim, generator=g)
    xa, xb = torch.randn(a.n_samples, a.dim, generator=g), torch.randn(a.n_samples, a.dim, generator=g)
    sa = xa @ w + torch.randn(a.n_samples, generator=g) * a.noise
    sb = xb @ w + torch.randn(a.n_samples, generator=g) * a.noise
    better = (sa > sb).unsqueeze(1)
    return DatasetBundle({"chosen": torch.where(better, xa, xb), "rejected": torch.where(better, xb, xa)},
                         info={"input_dim": a.dim, "true_weights": w, "task": "preference"})


def _anomaly(a: AnomalyDataConfig) -> DatasetBundle:
    g = _gen(a.seed)
    basis = torch.randn(3, a.n_features, generator=g)
    normal = torch.randn(a.n_samples, 3, generator=g) @ basis + torch.randn(a.n_samples, a.n_features, generator=g) * 0.05
    n_test = max(10, a.n_samples // 4)
    n_anom = max(1, int(n_test * a.anomaly_fraction))
    test_normal = torch.randn(n_test - n_anom, 3, generator=g) @ basis
    anomalies = torch.randn(n_anom, a.n_features, generator=g) * 2
    test = torch.cat([test_normal, anomalies])
    labels = torch.cat([torch.zeros(n_test - n_anom), torch.ones(n_anom)])
    return DatasetBundle({"inputs": normal}, val={"inputs": test, "targets": labels},
                         info={"input_dim": a.n_features, "task": "anomaly"})


def _tensor_file(a: TensorFileConfig) -> DatasetBundle:
    train = torch.load(a.path, map_location="cpu", weights_only=False)
    val = torch.load(a.val_path, map_location="cpu", weights_only=False) if a.val_path else None
    return DatasetBundle(train, val, info={"task": "custom"})


class MultimodalConfig(ImageDataConfig):
    meta_dim = Param(4, min=1, tooltip=T("Länge des Zusatzvektors (Metadaten).", "Zweite Modalität.", "2–16."))


def _multimodal(a: MultimodalConfig) -> DatasetBundle:
    """Bild + Metadaten; das Label hängt von BEIDEN ab (Form, verschoben um das Vorzeichen von meta[0])."""
    img = _images(a)
    g = _gen(a.seed + 1)
    meta = torch.randn(a.n_samples, a.meta_dim, generator=g)
    shape = img.train["targets"]
    label = torch.where(meta[:, 0] > 0, shape, (shape + 1) % 3)
    return DatasetBundle({"inputs": {"inputs": {"bild": img.train["inputs"], "meta": meta}}, "targets": label},
                         info={"output_dim": 3, "task": "classification"})


R = DATASETS.register
R("multimodal", _multimodal, config=MultimodalConfig, tags=("multimodal",), tooltip=T(
    "Bild + Metadaten, Label hängt von beiden ab.", "Testet echte Fusion zweier Modalitäten.", "Fusion-Presets."))
R("classification", _classification, config=ClassificationDataConfig, aliases=("klassifikation",), tags=("tabular",),
  tooltip=T("Synthetische 2D-Klassifikation (Spiralen, Monde, XOR …).", "Sichtbar nichtlineare Entscheidungsgrenzen.",
            "MLP-Tests, Tabellen-Presets."))
R("regression", _regression, config=RegressionDataConfig, tags=("tabular",), tooltip=T(
    "Synthetische Regression (sin, Polynom, Friedman).", "Kontinuierliche Zielwerte.", "Regressions-Presets."))
R("sequence_classification", _sequence, config=SequenceDataConfig, aliases=("sequenz",), tooltip=T(
    "Sequenzklassifikation (Parität, Enthaltensein, Mehrheit).", "Testet Gedächtnis von RNNs/Transformern.",
    "Sequenzmodelle."))
R("text", _text, config=TextDataConfig, aliases=("char_lm", "zeichen"), tags=("nlp",), tooltip=T(
    "Zeichenbasierter Text für Sprachmodelle (eigener Text oder Beispieltext).", "Sofort lauffähiges Mini-LLM.",
    "causal_lm-Presets."))
R("seq2seq_toy", _seq2seq, config=Seq2SeqDataConfig, aliases=("seq2seq",), tooltip=T(
    "Sequenz-Transformationen (umkehren, sortieren …).", "Encoder-Decoder-Tests.", "seq2seq-Presets."))
R("time_series", _timeseries, config=TimeSeriesDataConfig, aliases=("zeitreihe",), tags=("zeitreihen",), tooltip=T(
    "Sinus-Mischungen für Prognosen.", "Fenster → Zukunft.", "Zeitreihen-Presets."))
R("shapes", _images, config=ImageDataConfig, aliases=("bilder", "images"), tags=("vision",), tooltip=T(
    "Synthetische Bilder: Quadrat, Kreis, Kreuz.", "Bildklassifikation ohne Download.", "CNN/ViT-Presets."))
R("graphs", _graphs, config=GraphDataConfig, aliases=("graphen",), tags=("graph",), tooltip=T(
    "Kleine Graphen: Ring vs. Baum.", "Graph-Klassifikation.", "GNN-Presets."))
R("distribution_2d", _distribution, config=DistributionConfig, aliases=("verteilung",), tags=("generativ",), tooltip=T(
    "2D-Verteilungen für generative Modelle.", "Sichtbarer Generierungserfolg.", "GAN/Diffusion-Presets."))
R("preferences", _preferences, config=PreferenceDataConfig, aliases=("praeferenzen",), tooltip=T(
    "Präferenzpaare (bevorzugt/abgelehnt) aus verstecktem Urteil.", "Reward-Modell-Training.", "Präferenz-Presets."))
R("anomaly", _anomaly, config=AnomalyDataConfig, aliases=("anomalien",), tooltip=T(
    "Normaldaten zum Training + Testmenge mit Anomalien.", "Anomalieerkennung per Autoencoder.", "AE-Presets."))
R("csv", load_csv, config=CSVConfig, tags=("eigene",), tooltip=T(
    "Eigene CSV-Tabelle: Zahlen- und Kategorie-Spalten, fehlende Werte, Skalierung ohne Datenleckage.",
    "Eigene Tabellendaten ohne Code – vorbereitet wie in der Praxis üblich.", "Echte Projekte."))
R("text_classification", load_text_classification, config=TextClassificationConfig,
  aliases=("textklassifikation", "texte_mit_labels"), tags=("nlp", "eigene"), tooltip=T(
      "Texte mit Labels aus CSV/TSV (oder eingebautes Stimmungs-Beispiel) mit Tokenizer zeichen/wort/bpe.",
      "Spam, Stimmung, Themen, Absichten – mit Attention-Maske für Auffüllung.", "Textklassifikation."))
R("image_folder", load_image_folder, config=ImageFolderConfig, aliases=("bildordner",), tags=("vision", "eigene"),
  tooltip=T("Bilder aus ordner/<klasse>/*.png|jpg (Pillow), Größe angepasst und normiert.",
            "Eigene Bilddaten ohne Code.", "Bildklassifikation mit echten Fotos."))
R("tensor_file", _tensor_file, config=TensorFileConfig, aliases=("pt",), tags=("eigene",), tooltip=T(
    "Eigene torch.save-Datei mit Tensoren.", "Beliebige vorbereitete Daten.", "Echte Projekte."))
R("none", lambda a: DatasetBundle(None), aliases=("keine",), tooltip=T(
    "Keine Daten (z. B. Reinforcement Learning).", "RL erzeugt Daten selbst.", "RL-Presets."))
