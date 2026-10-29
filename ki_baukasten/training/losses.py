"""
ki_baukasten.training.losses
============================

Verlustfunktionen – *was* das Modell optimieren soll.

Kombinierbare Verluste
----------------------
Jede Verlustfunktion hat neben ihren eigenen Einstellungen drei gemeinsame:

* ``weight``      – Gewicht im Gesamtverlust (Summe aller Terme)
* ``output_key``  – welche Modellausgabe (bei dict-Ausgaben, z. B. Multi-Task)
* ``target_key``  – welches Ziel (bei dict-Zielen)

Damit lassen sich beliebige Kombinationen bilden, z. B. Klassifikation +
Regression in einem Multi-Task-Modell oder Cross-Entropy + Focal-Loss::

    "losses": [
        {"name": "cross_entropy", "args": {"output_key": "klasse", "target_key": "klasse"}},
        {"name": "huber", "args": {"weight": 0.5, "output_key": "wert", "target_key": "wert"}}
    ]

Registry ``loss``
-----------------
Fabrik-Signatur: ``factory(args) -> nn.Module`` mit ``forward(pred, target)``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, Level, Param, T, create_registry

LOSSES = create_registry(
    "loss",
    "Verlustfunktionen",
    T(
        "Funktionen, die messen, wie falsch eine Vorhersage ist.",
        "Das Training minimiert genau diese Größe – sie definiert, was 'gut' bedeutet.",
        "Jedes überwachte Training; frei kombinierbar.",
    ),
    context_doc="factory(args) -> Modul mit forward(pred, target)",
)


class LossConfig(BaseConfig):
    """Gemeinsame Einstellungen aller Verluste."""

    __title__ = "Verlust"
    weight = Param(1.0, tooltip=T(
        "Gewicht dieses Terms im Gesamtverlust.", "Balanciert mehrere Ziele gegeneinander.",
        "Kombinierte Verluste, Multi-Task.", "Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht."))
    output_key = Param(None, type=str, tooltip=T(
        "Schlüssel der Modellausgabe (bei dict-Ausgaben).", "Wählt die passende Ausgabe.", "Multi-Task, Autoencoder."),
        level=Level.ADVANCED)
    target_key = Param(None, type=str, tooltip=T(
        "Schlüssel des Ziels (bei dict-Zielen).", "Wählt das passende Ziel.", "Multi-Task."), level=Level.ADVANCED)


def _flatten_logits(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, T, C] + [B, T] → [B·T, C] + [B·T] (für Token-Klassifikation/Sprachmodelle)."""
    if pred.dim() > 2 and target.dim() == pred.dim() - 1 and not target.is_floating_point():
        return pred.reshape(-1, pred.shape[-1]), target.reshape(-1)
    if pred.dim() > 2 and target.shape == pred.shape:
        return pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
    return pred, target


# ===========================================================================
# Konfigurationen
# ===========================================================================

class CrossEntropyConfig(LossConfig):
    label_smoothing = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Label Smoothing: Zielwahrscheinlichkeit wird leicht über alle Klassen verteilt.",
        "Verhindert Überkonfidenz, verbessert Kalibrierung und Generalisierung.",
        "Bildklassifikation (0.1), Übersetzung (0.1).",
    ), group="Regularisierung")
    ignore_index = Param(-100, tooltip=T(
        "Zielwert, der ignoriert wird (z. B. Padding).", "Diese Positionen zählen nicht zum Verlust.",
        "Sprachmodelle, Tagging mit Padding."), level=Level.ADVANCED)
    class_weights = Param([], item_type=float, min=0.0, tooltip=T(
        "Gewicht pro Klasse (leer = alle gleich).", "Gleicht unbalancierte Datensätze aus.",
        "Seltene Klassen stärker gewichten (z. B. ∝ 1/Häufigkeit)."), level=Level.ADVANCED)


class BCEConfig(LossConfig):
    pos_weight = Param(None, type=float, min=0.0, tooltip=T(
        "Gewicht positiver Beispiele.", "Gleicht seltene Positive aus (> 1).", "Unbalancierte Binärprobleme."))
    label_smoothing = Param(0.0, min=0.0, max=0.5, tooltip=T(
        "Ziele 0/1 werden zu ε/2 bzw. 1-ε/2.", "Weniger Überkonfidenz.", "GAN-Diskriminatoren, Multi-Label."),
        group="Regularisierung")


class HuberConfig(LossConfig):
    delta = Param(1.0, min=1e-9, tooltip=T(
        "Übergang quadratisch → linear.", "Robust gegen Ausreißer (linear für große Fehler).", "Skala der Fehler."))


class QuantileConfig(LossConfig):
    quantiles = Param([0.1, 0.5, 0.9], item_type=float, min=0.0, max=1.0, tooltip=T(
        "Vorherzusagende Quantile (letzte Ausgabeachse = Anzahl Quantile).",
        "Liefert Vorhersageintervalle statt Punktschätzungen.",
        "Prognosen mit Unsicherheit (Nachfrage, Energie).",
    ))


class FocalConfig(LossConfig):
    gamma = Param(2.0, min=0.0, tooltip=T(
        "Fokussierungsstärke: leichte Beispiele werden heruntergewichtet.",
        "Konzentriert das Training auf schwierige/seltene Fälle.", "Objekterkennung, starke Unbalance: 2."))
    alpha = Param(None, type=float, min=0.0, max=1.0, tooltip=T(
        "Zusätzliche Klassengewichtung (binär) bzw. Faktor.", "Balanciert Klassen.", "0.25 (RetinaNet)."))
    ignore_index = Param(-100, tooltip=T("Zu ignorierender Zielwert.", "Padding.", "Selten."), level=Level.EXPERT)


class DiceConfig(LossConfig):
    smooth = Param(1.0, min=0.0, tooltip=T(
        "Glättungsterm gegen Division durch 0.", "Stabilität bei leeren Masken.", "1.0."), level=Level.ADVANCED)


class TemperatureConfig(LossConfig):
    temperature = Param(1.0, min=1e-6, tooltip=T(
        "Temperatur zum Weichzeichnen der Verteilungen.", "Höher = mehr Information über Nebenklassen.",
        "Distillation: 2–5."))


class InfoNCEConfig(LossConfig):
    temperature = Param(0.1, min=1e-6, tooltip=T(
        "Temperatur der Kontrast-Softmax.", "Kleiner = schärfere Trennung, schwerer zu trainieren.",
        "SimCLR: 0.1–0.5; CLIP: lernbar ~0.07."))
    symmetric = Param(True, tooltip=T(
        "Verlust in beide Richtungen (a→b und b→a).", "Symmetrisches Lernsignal.", "CLIP-artig."))


class MarginConfig(LossConfig):
    margin = Param(1.0, min=0.0, tooltip=T("Abstandsmarge.", "Mindestabstand korrekt vs. falsch.", "1.0."))


class PoissonConfig(LossConfig):
    log_input = Param(True, tooltip=T(
        "Vorhersage ist log(Rate) (empfohlen).", "Numerisch stabil, Rate immer positiv.", "Zähldaten."))


# ===========================================================================
# Module
# ===========================================================================

class CrossEntropy(nn.Module):
    def __init__(self, a: CrossEntropyConfig):
        super().__init__()
        self.a = a
        self.register_buffer("w", torch.tensor(a.class_weights) if a.class_weights else None, persistent=False)

    def forward(self, pred, target):
        pred, target = _flatten_logits(pred, target)
        if target.is_floating_point() and target.shape == pred.shape:  # weiche Ziele (Wahrscheinlichkeiten)
            return F.cross_entropy(pred.float(), target.float(), weight=self.w, label_smoothing=self.a.label_smoothing)
        return F.cross_entropy(pred.float(), target.long(), weight=self.w, ignore_index=self.a.ignore_index,
                               label_smoothing=self.a.label_smoothing)


class BCE(nn.Module):
    def __init__(self, a: BCEConfig):
        super().__init__()
        self.a = a

    def forward(self, pred, target):
        target = target.float().reshape(pred.shape)
        if self.a.label_smoothing:
            target = target * (1 - self.a.label_smoothing) + 0.5 * self.a.label_smoothing
        pw = torch.tensor(self.a.pos_weight, device=pred.device) if self.a.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(pred.float(), target, pos_weight=pw)


def _match(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Formt Regressionsziele passend zur Vorhersage (z. B. [B] → [B, 1])."""
    target = target.to(pred.dtype)
    if target.shape != pred.shape and target.numel() == pred.numel():
        target = target.reshape(pred.shape)
    return target


class Elementwise(nn.Module):
    """Wickelt eine elementweise Verlustfunktion ``fn(pred, target)``."""

    def __init__(self, fn, a: LossConfig):
        super().__init__()
        self.fn, self.a = fn, a

    def forward(self, pred, target):
        return self.fn(pred.float(), _match(pred.float(), target), self.a)


class Quantile(nn.Module):
    def __init__(self, a: QuantileConfig):
        super().__init__()
        self.register_buffer("q", torch.tensor(a.quantiles), persistent=False)

    def forward(self, pred, target):
        target = target.float()
        if target.dim() == pred.dim() - 1 or target.shape[-1] != pred.shape[-1]:
            target = target.reshape(*pred.shape[:-1], -1)[..., :1]
        diff = target - pred.float()
        return torch.maximum(self.q * diff, (self.q - 1) * diff).mean()


class Focal(nn.Module):
    def __init__(self, a: FocalConfig):
        super().__init__()
        self.a = a

    def forward(self, pred, target):
        pred, target = _flatten_logits(pred, target)
        if pred.shape[-1] == 1 or target.is_floating_point():  # binär
            t = target.float().reshape(pred.shape)
            p = torch.sigmoid(pred.float())
            ce = F.binary_cross_entropy_with_logits(pred.float(), t, reduction="none")
            pt = p * t + (1 - p) * (1 - t)
            loss = ce * (1 - pt) ** self.a.gamma
            if self.a.alpha is not None:
                loss = loss * (self.a.alpha * t + (1 - self.a.alpha) * (1 - t))
            return loss.mean()
        valid = target != self.a.ignore_index
        logp = F.log_softmax(pred.float()[valid], -1)
        t = target[valid].long()
        lp = logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        loss = -((1 - lp.exp()) ** self.a.gamma) * lp
        if self.a.alpha is not None:
            loss = loss * self.a.alpha
        return loss.mean() if loss.numel() else pred.sum() * 0


class Dice(nn.Module):
    """Dice-Verlust für Segmentierung: pred ``[B, C, *]`` (Logits), target ``[B, *]`` (Klassen) oder Masken."""

    def __init__(self, a: DiceConfig):
        super().__init__()
        self.a = a

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            probs = torch.sigmoid(pred.float())
            onehot = target.float().reshape(probs.shape)
        else:
            probs = pred.float().softmax(1)
            onehot = F.one_hot(target.long(), pred.shape[1]).movedim(-1, 1).float() if target.dim() == pred.dim() - 1 \
                else target.float()
        dims = tuple(range(2, probs.dim())) + (0,)
        inter = (probs * onehot).sum(dims)
        union = probs.sum(dims) + onehot.sum(dims)
        return 1 - ((2 * inter + self.a.smooth) / (union + self.a.smooth)).mean()


class KLDiv(nn.Module):
    """KL(target ‖ pred) mit Temperatur; pred = Logits, target = Logits oder Wahrscheinlichkeiten."""

    def __init__(self, a: TemperatureConfig):
        super().__init__()
        self.t = a.temperature

    def forward(self, pred, target):
        pred, target = _flatten_logits(pred, target)
        logp = F.log_softmax(pred.float() / self.t, -1)
        tgt = target.float()
        q = tgt if (tgt.min() >= 0 and torch.allclose(tgt.sum(-1), torch.ones_like(tgt.sum(-1)), atol=1e-3)) \
            else F.softmax(tgt / self.t, -1)
        return F.kl_div(logp, q, reduction="batchmean") * self.t ** 2


class InfoNCE(nn.Module):
    """Kontrastiver InfoNCE-Verlust: Zeile i von ``pred`` gehört zu Zeile i von ``target``."""

    def __init__(self, a: InfoNCEConfig):
        super().__init__()
        self.a = a

    def forward(self, pred, target):
        a = F.normalize(pred.float().flatten(1), dim=-1)
        b = F.normalize(target.float().flatten(1), dim=-1)
        logits = a @ b.T / self.a.temperature
        labels = torch.arange(a.shape[0], device=a.device)
        loss = F.cross_entropy(logits, labels)
        if self.a.symmetric:
            loss = 0.5 * (loss + F.cross_entropy(logits.T, labels))
        return loss


class Hinge(nn.Module):
    def __init__(self, a: MarginConfig):
        super().__init__()
        self.m = a.margin

    def forward(self, pred, target):
        t = _match(pred.float(), target)
        t = torch.where(t <= 0, -torch.ones_like(t), torch.ones_like(t))  # {0,1} oder {-1,1} → {-1,1}
        return F.relu(self.m - t * pred.float()).mean()


class GaussianNLL(nn.Module):
    """Negative Log-Likelihood einer Normalverteilung; pred ``[..., 2]`` = (Mittelwert, log-Varianz)."""

    def forward(self, pred, target):
        mean, log_var = pred.float()[..., 0], pred.float()[..., 1].clamp(-15, 15)
        t = target.float().reshape(mean.shape)
        return 0.5 * (log_var + (t - mean).square() / log_var.exp()).mean()


# ===========================================================================
# Registrierung
# ===========================================================================

R = LOSSES.register
R("cross_entropy", lambda a: CrossEntropy(a), config=CrossEntropyConfig, aliases=("ce",), tags=("klassifikation",),
  tooltip=T("Kreuzentropie für Klassifikation (Logits + Klassenindizes oder Wahrscheinlichkeiten).",
            "Standardverlust für Klassifikation und Sprachmodelle.", "Klassifikation, LMs, Tagging."))
R("bce", lambda a: BCE(a), config=BCEConfig, aliases=("binary_cross_entropy",), tags=("klassifikation",), tooltip=T(
    "Binäre Kreuzentropie (mit Logits).", "Binär- und Multi-Label-Klassifikation.", "Ja/Nein-Fragen, Multi-Label."))
R("mse", lambda a: Elementwise(lambda p, t, a: F.mse_loss(p, t), a), config=LossConfig, aliases=("l2",),
  tags=("regression",), tooltip=T("Mittlerer quadratischer Fehler.", "Bestraft große Fehler stark.", "Regression."))
R("l1", lambda a: Elementwise(lambda p, t, a: F.l1_loss(p, t), a), config=LossConfig, aliases=("mae",),
  tags=("regression",), tooltip=T("Mittlerer absoluter Fehler.", "Robust gegen Ausreißer (Median-Schätzer).",
                                  "Regression mit Ausreißern."))
R("huber", lambda a: Elementwise(lambda p, t, a: F.huber_loss(p, t, delta=a.delta), a), config=HuberConfig,
  tags=("regression",), tooltip=T("Huber-Verlust: quadratisch nah, linear fern.", "Kompromiss aus MSE und L1.",
                                  "Robuste Regression, DQN."))
R("smooth_l1", lambda a: Elementwise(lambda p, t, a: F.smooth_l1_loss(p, t, beta=a.delta), a), config=HuberConfig,
  level=Level.ADVANCED, tooltip=T("Smooth-L1.", "Wie Huber (skaliert).", "Bounding-Box-Regression."))
R("log_cosh", lambda a: Elementwise(lambda p, t, a: (torch.logaddexp(p - t, t - p) - torch.log(torch.tensor(2.0))).mean(),
                                    a), config=LossConfig, level=Level.ADVANCED, tooltip=T(
    "log(cosh(Fehler)).", "Glatt wie MSE, robust wie L1.", "Regression."))
R("quantile", lambda a: Quantile(a), config=QuantileConfig, aliases=("pinball",), tags=("regression",), tooltip=T(
    "Quantil-(Pinball-)Verlust.", "Vorhersageintervalle.", "Prognosen mit Unsicherheit."))
R("gaussian_nll", lambda a: GaussianNLL(), config=LossConfig, level=Level.ADVANCED, tooltip=T(
    "Gauß-Likelihood mit gelernter Varianz.", "Modell schätzt seine eigene Unsicherheit.", "Probabilistische Regression."))
R("poisson_nll", lambda a: Elementwise(lambda p, t, a: F.poisson_nll_loss(p, t, log_input=a.log_input), a),
  config=PoissonConfig, level=Level.ADVANCED, tooltip=T("Poisson-Likelihood.", "Für Zähldaten.", "Ereignisanzahlen."))
R("focal", lambda a: Focal(a), config=FocalConfig, tags=("klassifikation",), tooltip=T(
    "Focal Loss.", "Fokus auf schwierige Beispiele.", "Starke Klassen-Unbalance, Detektion."))
R("dice", lambda a: Dice(a), config=DiceConfig, tags=("segmentierung",), tooltip=T(
    "Dice-Verlust.", "Optimiert direkt die Überlappung von Masken.", "Segmentierung (Medizin)."))
R("kl_div", lambda a: KLDiv(a), config=TemperatureConfig, aliases=("kl",), level=Level.ADVANCED, tooltip=T(
    "KL-Divergenz zwischen Verteilungen.", "Angleichen an eine Zielverteilung.", "Distillation."))
R("cosine", lambda a: Elementwise(lambda p, t, a: (1 - F.cosine_similarity(p, t, dim=-1)).mean(), a), config=LossConfig,
  level=Level.ADVANCED, tooltip=T("1 - Kosinus-Ähnlichkeit.", "Richtungsgleichheit von Vektoren.", "Embeddings."))
R("info_nce", lambda a: InfoNCE(a), config=InfoNCEConfig, aliases=("nt_xent", "contrastive"), level=Level.ADVANCED,
  tooltip=T("InfoNCE-Kontrastverlust.", "Zieht zusammengehörige Paare zusammen, stößt andere ab.",
            "Selbstüberwachtes Lernen, CLIP."))
R("hinge", lambda a: Hinge(a), config=MarginConfig, level=Level.ADVANCED, tooltip=T(
    "Hinge-Verlust (SVM-artig).", "Maximiert die Trennmarge.", "Binäre Klassifikation."))
R("nll", lambda a: Elementwise(lambda p, t, a: F.nll_loss(*_flatten_logits(p, t.long())), a), config=LossConfig,
  level=Level.ADVANCED, tooltip=T("Negative Log-Likelihood (Eingabe: Log-Wahrscheinlichkeiten).",
                                  "Wenn das Modell bereits log_softmax ausgibt.", "Spezialfälle."))


def build_losses(specs: list[Any]) -> list[tuple[nn.Module, LossConfig]]:
    """Baut eine Liste (Modul, Einstellungen) aus Komponenten-Specs."""
    return [(spec.build(), spec.args) for spec in specs]
