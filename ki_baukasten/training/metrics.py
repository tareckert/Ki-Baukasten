"""
ki_baukasten.training.metrics
=============================

**Metriken** – woran man misst, ob eine KI wirklich gut ist (Registry ``metric``).

Der Verlust allein sagt wenig: Bei unausgewogenen Klassen ist eine hohe Genauigkeit wertlos, bei
Prognosen interessiert der Fehler in echten Einheiten, bei Risiken die Kalibrierung. Jede Metrik
wird in der Validierung (``val_…``) und im Test (``Project.test()`` → ``test_…``) über **alle**
Beispiele berechnet.

======================  ==============================================================================
Aufgabe                 Metriken
======================  ==============================================================================
Klassifikation          accuracy, top_k_accuracy, precision, recall, f1, balanced_accuracy, mcc, auroc,
                        log_loss, ece (Kalibrierungsfehler), brier
Regression/Prognose     mae, mse, rmse, r2, mape, smape, max_error
Sprachmodelle           perplexity, token_accuracy
Übersetzung/Seq2Seq     token_accuracy, exact_match, bleu
Segmentierung           iou (mittlere IoU), dice
======================  ==============================================================================

``training.metrics`` (Liste) wählt sie; leer = ``auto`` (sinnvolle Auswahl je Aufgabe, ``training.auto_metrics``).
Eigene: ``@register("metric", "name", config=...)`` mit ``factory(args) -> fn(pred, target) -> float``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..core import BaseConfig, Level, Param, T, create_registry

METRICS = create_registry(
    "metric", "Metriken",
    T("Kennzahlen zur Bewertung (Klassifikation, Regression, Sprache, Übersetzung, Segmentierung).",
      "Zeigen, wie gut die KI wirklich ist – der Verlust allein reicht nicht.",
      "Validierung, Test, Suche, Early Stopping (monitor='val_f1' …)."),
    context_doc="factory(args) -> fn(pred, target) -> float",
)


def _cls(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Logits [N, C] und Klassen [N] (ignore -100 entfernt); Segmentierung [N, C, H, W] → je Pixel."""
    if pred.dim() == 4 and target.dim() == 3 and pred.shape[2:] == target.shape[1:]:
        pred = pred.movedim(1, -1)
    if target.is_floating_point() or pred.dim() != target.dim() + 1:
        raise ValueError("Klassifikationsmetrik braucht Logits [N, C] und ganzzahlige Klassen [N].")
    logits = pred.reshape(-1, pred.shape[-1]).float()
    y = target.reshape(-1).long()
    keep = y != -100
    return logits[keep], y[keep], logits.shape[-1]


def _confusion(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits, y, c = _cls(pred, target)
    p = logits.argmax(-1)
    return torch.bincount(y * c + p, minlength=c * c).reshape(c, c).float()


def _prf(pred: torch.Tensor, target: torch.Tensor, average: str) -> tuple[float, float, float]:
    cm = _confusion(pred, target)
    tp = cm.diag()
    support = cm.sum(1)
    predicted = cm.sum(0)
    if average == "micro":
        p = r = float(tp.sum() / cm.sum().clamp_min(1))
        return p, r, p
    prec = tp / predicted.clamp_min(1)
    rec = tp / support.clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    present = support > 0
    if cm.shape[0] == 2 and average == "binary":
        return float(prec[1]), float(rec[1]), float(f1[1])
    if average == "weighted":
        w = support / support.sum().clamp_min(1)
        return float((prec * w).sum()), float((rec * w).sum()), float((f1 * w).sum())
    return float(prec[present].mean()), float(rec[present].mean()), float(f1[present].mean())


def _auc_binary(scores: torch.Tensor, pos: torch.Tensor) -> float:
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    sorted_s = scores[order]                       # Bindungen: mittlerer Rang
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _reg(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(pred, SeqPred) or (not target.is_floating_point() and pred.dim() == target.dim() + 1):
        raise ValueError("Regressionsmetrik braucht Zahlen-Vorhersagen und -Ziele (keine Klassen).")
    p, t = pred.float().reshape(-1), target.float().reshape(-1)
    n = min(len(p), len(t))
    return p[:n], t[:n]


def _ngrams(seq: list[int], n: int) -> Counter:
    return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def _strip(seq: list[int]) -> list[int]:
    return [t for t in seq if t not in (-100, 0)]


class SeqPred:
    """Kompakte Vorhersage für Folgen: Token-IDs ``[B, T]`` und Verlust je Token ``[B, T]``.

    Sprachmodelle haben riesige Logits (Vokabular × Länge) – gesammelt wird deshalb nur, was die
    Metriken brauchen.
    """

    def __init__(self, ids: torch.Tensor, nll: torch.Tensor):
        self.ids, self.nll = ids, nll

    @classmethod
    def from_logits(cls, logits: torch.Tensor, target: torch.Tensor) -> "SeqPred":
        lp = logits.float().log_softmax(-1)
        nll = -lp.gather(-1, target.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return cls(logits.argmax(-1), nll)

    def __len__(self) -> int:
        return self.ids.shape[0]


def _ids(pred: Any) -> torch.Tensor:
    if isinstance(pred, SeqPred):
        return pred.ids
    return pred if not pred.is_floating_point() else pred.argmax(-1)     # schon Token-IDs (z. B. generiert)


def _token_nll(pred: Any, target: torch.Tensor) -> torch.Tensor:
    nll = pred.nll if isinstance(pred, SeqPred) else SeqPred.from_logits(pred, target).nll
    return nll[target != -100]


# ===========================================================================
# Konfigurationen
# ===========================================================================

class AverageConfig(BaseConfig):
    average = Param("macro", choices={"macro": "Mittel über Klassen (jede Klasse gleich wichtig)",
                                      "weighted": "nach Häufigkeit gewichtet", "micro": "über alle Beispiele",
                                      "binary": "nur positive Klasse (2 Klassen)"}, tooltip=T(
        "Mittelung über Klassen.", "macro deckt Schwächen bei seltenen Klassen auf.", "macro."))


class TopKConfig(BaseConfig):
    k = Param(5, min=1, tooltip=T("Treffer, wenn die richtige Klasse unter den k besten ist.", "Viele Klassen.", "5."))


class ECEConfig(BaseConfig):
    bins = Param(15, min=2, tooltip=T("Anzahl Konfidenz-Intervalle.", "Feinheit der Kalibrierungsmessung.", "10–20."))


# ===========================================================================
# Metriken
# ===========================================================================

R = METRICS.register


def _accuracy(a):
    def fn(pred, target):
        if not pred.is_floating_point() and pred.shape == target.shape:     # schon Klassen (z. B. Masken)
            valid = target != -100
            return float((pred == target)[valid].float().mean()) if valid.any() else float("nan")
        logits, y, _ = _cls(pred, target)
        return float((logits.argmax(-1) == y).float().mean()) if len(y) else float("nan")
    return fn


def _topk(a):
    def fn(pred, target):
        logits, y, c = _cls(pred, target)
        k = min(a.k, c)
        return float((logits.topk(k, -1).indices == y[:, None]).any(-1).float().mean())
    return fn


def _metric_prf(index: int) -> Callable:
    def factory(a):
        return lambda pred, target: _prf(pred, target, a.average)[index]
    return factory


def _balanced(a):
    def fn(pred, target):
        cm = _confusion(pred, target)
        rec = cm.diag() / cm.sum(1).clamp_min(1)
        return float(rec[cm.sum(1) > 0].mean())
    return fn


def _mcc(a):
    def fn(pred, target):
        cm = _confusion(pred, target)
        t, p, c, s = cm.sum(1), cm.sum(0), cm.diag().sum(), cm.sum()
        num = c * s - (t * p).sum()
        den = torch.sqrt((s ** 2 - (p * p).sum()) * (s ** 2 - (t * t).sum()))
        return float(num / den) if den > 0 else 0.0
    return fn


def _auroc(a):
    def fn(pred, target):
        logits, y, c = _cls(pred, target)
        if c == 1:
            return _auc_binary(logits[:, 0], y.bool())
        prob = logits.softmax(-1)
        if c == 2:
            return _auc_binary(prob[:, 1], y == 1)
        aucs = [_auc_binary(prob[:, k], y == k) for k in range(c)]
        aucs = [v for v in aucs if not math.isnan(v)]
        return sum(aucs) / len(aucs) if aucs else float("nan")
    return fn


def _log_loss(a):
    def fn(pred, target):
        logits, y, _ = _cls(pred, target)
        return float(F.cross_entropy(logits, y))
    return fn


def _ece(a):
    def fn(pred, target):
        logits, y, _ = _cls(pred, target)
        prob = logits.softmax(-1)
        conf, guess = prob.max(-1)
        correct = (guess == y).float()
        edges = torch.linspace(0, 1, a.bins + 1)
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                total += float(m.float().mean()) * abs(float(conf[m].mean()) - float(correct[m].mean()))
        return total
    return fn


def _brier(a):
    def fn(pred, target):
        logits, y, c = _cls(pred, target)
        return float((logits.softmax(-1) - F.one_hot(y, c).float()).square().sum(-1).mean())
    return fn


def _simple(fn: Callable[[torch.Tensor, torch.Tensor], float]) -> Callable:
    return lambda a: (lambda pred, target: fn(*_reg(pred, target)))


def _r2(p, t):
    ss_res = (t - p).square().sum()
    ss_tot = (t - t.mean()).square().sum()
    return float(1 - ss_res / ss_tot.clamp_min(1e-12))


def _perplexity(a):
    def fn(pred, target):
        nll = _token_nll(pred, target)
        return float(math.exp(min(30.0, float(nll.mean())))) if len(nll) else float("nan")
    return fn


def _token_accuracy(a):
    def fn(pred, target):
        valid = target != -100
        return float((_ids(pred) == target)[valid].float().mean()) if valid.any() else float("nan")
    return fn


def _exact(a):
    def fn(pred, target):
        p, t = _ids(pred), target
        valid = t != -100
        same = ((p == t) | ~valid).all(-1)
        return float(same.float().mean())
    return fn


def _bleu(a):
    def fn(pred, target):
        p_all, t_all = _ids(pred).tolist(), target.tolist()
        matches, totals = [0] * 4, [0] * 4
        hyp_len = ref_len = 0
        for p, t in zip(p_all, t_all):
            p, t = _strip(p), _strip(t)
            hyp_len, ref_len = hyp_len + len(p), ref_len + len(t)
            for n in range(1, 5):
                hp, rf = _ngrams(p, n), _ngrams(t, n)
                matches[n - 1] += sum(min(c, rf[g]) for g, c in hp.items())
                totals[n - 1] += max(0, len(p) - n + 1)
        if hyp_len == 0:
            return 0.0
        logs = [math.log((m + 1) / (tt + 1)) for m, tt in zip(matches, totals)]     # Glättung (+1)
        bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
        return bp * math.exp(sum(logs) / 4)
    return fn


def _seg(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Segmentierung: Logits [N, C, H, W] (oder [N, 1, H, W]) bzw. Klassenkarte [N, H, W] gegen Masken."""
    if pred.dim() != (4 if pred.is_floating_point() else 3) or target.is_floating_point():
        raise ValueError("Segmentierungsmetrik braucht Logits [N, C, H, W] und Masken [N, H, W].")
    if not pred.is_floating_point():                 # schon verdichtet (Collector)
        return pred, target.long().reshape(pred.shape), int(max(pred.max(), target.max())) + 1
    if pred.shape[1] == 1:
        p = (pred[:, 0] > 0).long()
        return p, target.long().reshape(p.shape), 2
    return pred.argmax(1), target.long().reshape(pred.shape[0], *pred.shape[2:]), pred.shape[1]


def _iou(a):
    def fn(pred, target):
        p, t, c = _seg(pred, target)
        ious = []
        for k in range(c):
            inter = ((p == k) & (t == k)).sum()
            union = ((p == k) | (t == k)).sum()
            if union > 0:
                ious.append(float(inter) / float(union))
        return sum(ious) / len(ious) if ious else float("nan")
    return fn


def _dice_metric(a):
    def fn(pred, target):
        p, t, c = _seg(pred, target)
        ds = []
        for k in range(1 if c == 2 else 0, c):
            inter = ((p == k) & (t == k)).sum()
            denom = (p == k).sum() + (t == k).sum()
            if denom > 0:
                ds.append(2 * float(inter) / float(denom))
        return sum(ds) / len(ds) if ds else float("nan")
    return fn


_C = ("klassifikation",)
R("accuracy", _accuracy, tags=_C, aliases=("genauigkeit",), tooltip=T(
    "Anteil richtiger Vorhersagen.", "Einfach verständlich.", "Ausgewogene Klassen.",
    "Bei unausgewogenen Klassen irreführend – dann balanced_accuracy/f1."))
R("top_k_accuracy", _topk, config=TopKConfig, tags=_C, tooltip=T(
    "Richtige Klasse unter den k wahrscheinlichsten.", "Viele, ähnliche Klassen.", "ImageNet-Stil (Top-5)."))
R("precision", _metric_prf(0), config=AverageConfig, tags=_C, aliases=("praezision",), tooltip=T(
    "Anteil der als positiv Vorhergesagten, die wirklich positiv sind.", "Wichtig, wenn Fehlalarme teuer sind.",
    "Spam-Filter, Betrugsalarm."))
R("recall", _metric_prf(1), config=AverageConfig, tags=_C, aliases=("trefferquote", "sensitivitaet"), tooltip=T(
    "Anteil der wirklich Positiven, die gefunden werden.", "Wichtig, wenn Übersehen teuer ist.", "Diagnosen, Sicherheit."))
R("f1", _metric_prf(2), config=AverageConfig, tags=_C, aliases=("f1_score",), tooltip=T(
    "Harmonisches Mittel aus Präzision und Trefferquote.", "Ein Wert für unausgewogene Klassen.", "Standard bei "
    "seltenen Klassen (macro)."))
R("balanced_accuracy", _balanced, tags=_C, aliases=("ausgeglichene_genauigkeit",), tooltip=T(
    "Mittlere Trefferquote je Klasse.", "Unbeeindruckt von Klassenhäufigkeiten.", "Unausgewogene Daten."))
R("mcc", _mcc, tags=_C, level=Level.ADVANCED, tooltip=T(
    "Matthews-Korrelationskoeffizient (−1 … 1).", "Robusteste Einzelzahl für unausgewogene Daten.", "Medizin, Biologie."))
R("auroc", _auroc, tags=_C, aliases=("roc_auc",), tooltip=T(
    "Fläche unter der ROC-Kurve (Trennschärfe über alle Schwellen).", "Unabhängig von der Entscheidungsschwelle.",
    "Risiko-Scores, Ranking.", "Mehrklassig: Mittel über 'eine gegen alle'."))
R("log_loss", _log_loss, tags=_C, level=Level.ADVANCED, tooltip=T(
    "Kreuzentropie der vorhergesagten Wahrscheinlichkeiten.", "Bewertet die Wahrscheinlichkeiten selbst.",
    "Wahrscheinlichkeiten werden weiterverwendet."))
R("ece", _ece, config=ECEConfig, tags=_C, aliases=("kalibrierungsfehler",), tooltip=T(
    "Expected Calibration Error: Abstand zwischen Konfidenz und tatsächlicher Trefferquote.",
    "Ein Modell, das '90 %' sagt, soll in 90 % der Fälle recht haben.", "Entscheidungen mit Risiko.",
    "Verbessern: Project.calibrate() (Temperature Scaling)."))
R("brier", _brier, tags=_C, level=Level.ADVANCED, tooltip=T(
    "Brier-Score: mittlerer quadratischer Fehler der Wahrscheinlichkeiten.", "Genauigkeit + Kalibrierung.",
    "Wettervorhersage, Risiken."))
_RG = ("regression",)
R("mae", _simple(lambda p, t: float((p - t).abs().mean())), tags=_RG, aliases=("mittlerer_absoluter_fehler",),
  tooltip=T("Mittlerer absoluter Fehler.", "In echten Einheiten, robust gegen Ausreißer.", "Prognosen."))
R("mse", _simple(lambda p, t: float((p - t).square().mean())), tags=_RG, tooltip=T(
    "Mittlerer quadratischer Fehler.", "Bestraft große Fehler stark.", "Regression."))
R("rmse", _simple(lambda p, t: float((p - t).square().mean().sqrt())), tags=_RG, tooltip=T(
    "Wurzel des MSE.", "Einheiten wie das Ziel.", "Regression, Prognose."))
R("r2", _simple(_r2), tags=_RG, aliases=("bestimmtheitsmass",), tooltip=T(
    "Bestimmtheitsmaß R² (1 = perfekt, 0 = wie Mittelwert).", "Anteil erklärter Streuung.", "Regression."))
R("mape", _simple(lambda p, t: float(((p - t).abs() / t.abs().clamp_min(1e-8)).mean() * 100)), tags=_RG, tooltip=T(
    "Mittlerer prozentualer Fehler.", "Verständlich für Fachleute.", "Absatz-/Umsatzprognosen.",
    "Instabil bei Zielen nahe 0 – dann smape."))
R("smape", _simple(lambda p, t: float((2 * (p - t).abs() / (p.abs() + t.abs()).clamp_min(1e-8)).mean() * 100)),
  tags=_RG, tooltip=T("Symmetrischer prozentualer Fehler.", "Stabil bei kleinen Werten.", "Zeitreihen-Wettbewerbe (M4)."))
R("max_error", _simple(lambda p, t: float((p - t).abs().max())), tags=_RG, level=Level.ADVANCED, tooltip=T(
    "Größter Einzelfehler.", "Schlimmster Fall.", "Sicherheitskritische Prognosen."))
R("perplexity", _perplexity, tags=("sprache",), aliases=("perplexitaet",), tooltip=T(
    "exp(Kreuzentropie) je Token.", "Wie 'überrascht' das Sprachmodell ist – kleiner ist besser.", "Sprachmodelle."))
R("token_accuracy", _token_accuracy, tags=("sprache", "seq2seq"), tooltip=T(
    "Anteil richtig vorhergesagter Tokens.", "Anschaulich.", "Sprachmodelle, Seq2Seq."))
R("exact_match", _exact, tags=("seq2seq",), tooltip=T(
    "Anteil komplett richtiger Ausgaben.", "Streng: alles oder nichts.", "Übersetzung kurzer Folgen, Code, Rechnen."))
R("bleu", _bleu, tags=("seq2seq",), tooltip=T(
    "BLEU (n-Gramm-Übereinstimmung 1–4, mit Kürzestrafe; geglättet).", "Klassische Übersetzungsmetrik.",
    "Übersetzung, Zusammenfassung.", "Auf Token-IDs (mit Teacher Forcing) – Näherung an Satz-BLEU."))
R("iou", _iou, tags=("segmentierung",), aliases=("miou", "jaccard"), tooltip=T(
    "Mittlere Intersection-over-Union je Klasse.", "Standard für Segmentierung.", "Segmentierung."))
R("dice", _dice_metric, tags=("segmentierung",), tooltip=T(
    "Dice-Koeffizient (2·Schnitt / Summe).", "Robust bei kleinen Objekten.", "Medizinische Segmentierung."))



def _multilabel(kind: str) -> Callable:
    def factory(a):
        def fn(pred, target):
            p = (pred.float() > a.threshold_logit).reshape(target.shape[0], -1)
            t = target.float().reshape(p.shape) > 0.5
            if kind == "accuracy":
                return float((p == t).float().mean())
            if kind == "subset":
                return float((p == t).all(-1).float().mean())
            tp, fp, fn_ = float((p & t).sum()), float((p & ~t).sum()), float((~p & t).sum())
            return 2 * tp / max(1e-12, 2 * tp + fp + fn_)
        return fn
    return factory


class MultilabelConfig(BaseConfig):
    threshold_logit = Param(0.0, tooltip=T("Schwelle auf den Logits (0 = Wahrscheinlichkeit 0.5).",
                                           "Höher = vorsichtiger (weniger Labels).", "0."))


_ML = ("multilabel",)
R("multilabel_accuracy", _multilabel("accuracy"), config=MultilabelConfig, tags=_ML, aliases=("hamming_genauigkeit",),
  tooltip=T("Anteil richtiger Einzel-Labels (1 − Hamming-Verlust).", "Mehrere Labels je Beispiel.", "Tagging."))
R("subset_accuracy", _multilabel("subset"), config=MultilabelConfig, tags=_ML, tooltip=T(
    "Anteil Beispiele, bei denen **alle** Labels stimmen.", "Streng.", "Wenige Labels je Beispiel."))
R("multilabel_f1", _multilabel("f1"), config=MultilabelConfig, tags=_ML, tooltip=T(
    "F1 über alle Labels (micro).", "Robust bei seltenen Labels.", "Tagging, Mehrfach-Diagnosen."))

#: Automatische Auswahl je Aufgabe.
AUTO = {"klassifikation": ["accuracy", "balanced_accuracy", "f1", "auroc", "ece"],
        "regression": ["mae", "rmse", "r2"],
        "sprache": ["perplexity", "token_accuracy"],
        "seq2seq": ["token_accuracy", "exact_match", "bleu"],
        "segmentierung": ["iou", "dice", "accuracy"],
        "multilabel": ["multilabel_accuracy", "subset_accuracy", "multilabel_f1"]}

#: Metriken, bei denen **größer besser** ist (für Early Stopping/Checkpoints/Suche mit ``mode='auto'``).
HIGHER_IS_BETTER = {"accuracy", "top_k_accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "auroc",
                    "r2", "token_accuracy", "exact_match", "bleu", "iou", "dice", "multilabel_accuracy",
                    "subset_accuracy", "multilabel_f1", "reward", "return"}


def direction(name: str) -> str:
    """``max`` oder ``min`` für einen Metriknamen (Präfixe ``val_``/``test_``/``train_`` werden ignoriert)."""
    base = name
    for prefix in ("val_", "test_", "train_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if base in HIGHER_IS_BETTER:
        return "max"
    try:
        comp = METRICS.get(base)
        return "max" if comp.name in HIGHER_IS_BETTER else "min"
    except Exception:  # noqa: BLE001 - unbekannt → kleiner ist besser (Verlust, Fehler)
        return "min"


def compute_metrics(specs: list, pred: torch.Tensor, target: torch.Tensor, prefix: str = "val_",
                    skipped: list | None = None) -> dict[str, float]:
    """Berechnet alle ``specs`` (ComponentSpecs) auf den gesammelten Vorhersagen.

    Metriken, die nicht zur Aufgabe passen, werden weggelassen (Name in ``skipped``).
    """
    out = {}
    for spec in specs:
        try:
            value = spec.build()(pred, target)
        except Exception:  # noqa: BLE001 - Metrik passt nicht zur Aufgabe → weglassen
            if skipped is not None:
                skipped.append(spec.name)
            continue
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            out[f"{prefix}{spec.name}"] = float(value)
    return out


def task_of(pred: torch.Tensor, target: torch.Tensor, kind: str = "auto") -> str:
    """Aufgabe aus Paradigma-Art und Formen (für ``auto``)."""
    if kind in AUTO:
        return kind
    if pred.dim() == 4 and target.dim() in (3, 4) and target.shape[-2:] == pred.shape[-2:]:
        return "segmentierung"
    if not target.is_floating_point() and pred.dim() == target.dim() + 1:
        return "klassifikation"
    return "regression"


def auto_specs(task: str) -> list:
    from ..core import ComponentSpec

    return [ComponentSpec.from_value("metric", n) for n in AUTO.get(task, [])]


def target_inverse(transform: dict | None) -> Callable | None:
    """Rückrechnung skalierter Regressionsziele (``info['target_transform']`` der CSV-Daten) für Metriken."""
    if not transform:
        return None
    if transform.get("art") == "log":
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.expm1(x.float())
    elif transform.get("art") == "standard":
        m, sd = float(transform["mittel"]), float(transform["std"])

        def f(x: torch.Tensor) -> torch.Tensor:
            return x.float() * sd + m
    else:
        return None
    return lambda pred, target: (f(pred), f(target))


class Collector:
    """Sammelt Vorhersagen und Ziele während einer Evaluierung (kompakt, höchstens ``max_items`` Beispiele).

    Paradigmen rufen ``trainer.collect(pred, target, kind)`` auf; danach berechnet ``metrics()`` alle
    gewählten Metriken über die **gesamte** Menge (nicht als Mittel über Batches – wichtig für F1, AUROC …).
    """

    def __init__(self, max_items: int = 5000):
        self.max_items = max_items
        self.task: str | None = None
        self.preds: list[Any] = []
        self.targets: list[torch.Tensor] = []
        self.n = 0

    def add(self, pred: Any, target: Any, kind: str = "auto") -> None:
        if self.n >= self.max_items or not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            return
        if pred.dim() == 0 or pred.shape[0] != target.shape[0]:
            return
        if self.task is None:
            self.task = task_of(pred, target, kind)
        take = min(pred.shape[0], self.max_items - self.n)
        pred, target = pred[:take].detach(), target[:take].detach()
        if self.task in ("sprache", "seq2seq"):
            sp = SeqPred.from_logits(pred, target)
            pred = SeqPred(sp.ids.cpu(), sp.nll.cpu())
        elif self.task == "segmentierung":
            pred = ((pred[:, 0] > 0).long() if pred.shape[1] == 1 else pred.argmax(1)).cpu()
        else:
            pred = pred.float().cpu()
        self.preds.append(pred)
        self.targets.append(target.cpu())
        self.n += take

    def result(self) -> tuple[Any, torch.Tensor] | None:
        """Alle gesammelten Vorhersagen und Ziele (Folgen auf gleiche Länge aufgefüllt)."""
        if not self.preds:
            return None
        if isinstance(self.preds[0], SeqPred):
            t_max = max(p.ids.shape[1] for p in self.preds)

            def pad(x: torch.Tensor, value: float) -> torch.Tensor:
                return F.pad(x, (0, t_max - x.shape[1]), value=value)

            pred = SeqPred(torch.cat([pad(p.ids, 0) for p in self.preds]), torch.cat([pad(p.nll, 0) for p in self.preds]))
            return pred, torch.cat([pad(t, -100) for t in self.targets])
        try:
            return torch.cat(self.preds), torch.cat(self.targets)
        except RuntimeError:                        # unterschiedliche Formen → flach (nur Regression sinnvoll)
            return torch.cat([p.reshape(-1) for p in self.preds]), torch.cat([t.reshape(-1) for t in self.targets])

    def metrics(self, specs: list, auto: bool = True, prefix: str = "val_",
                skipped: list | None = None) -> dict[str, float]:
        res = self.result()
        if res is None:
            return {}
        chosen = list(specs) or (auto_specs(self.task or "regression") if auto else [])
        return compute_metrics(chosen, *res, prefix=prefix, skipped=skipped if specs else None)
