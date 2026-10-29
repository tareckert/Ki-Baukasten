"""
ki_baukasten.analysis.inference
===============================

**Verlässliche Vorhersagen** – was nach dem Training aus einer guten eine *vertrauenswürdige* KI macht.
Alles ist einzeln an/abwählbar (``project.inference``) und wird von ``Project.calibrate()`` auf der
Validierungsmenge eingestellt.

======================  ============================================================================
Baustein                Wirkung
======================  ============================================================================
Temperature Scaling     Wahrscheinlichkeiten ehrlich machen ("90 %" heißt dann wirklich 90 % Treffer)
Entscheidungsschwelle   2 Klassen: Schwelle optimal für F1 / ausgeglichene Genauigkeit / Youden wählen
Enthaltung              unter einer Mindest-Konfidenz "unsicher" melden statt raten (Reject Option)
Conformal Prediction    Vorhersage-*Mengen* (Klassifikation) bzw. -*Intervalle* (Regression) mit
                        garantierter Abdeckung, z. B. 90 % – ohne Annahmen über das Modell
MC-Dropout              Unsicherheit aus mehreren Durchläufen mit aktivem Dropout
TTA                     Test-Time Augmentation: Mittel über gespiegelte/verrauschte Eingaben
======================  ============================================================================
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..core import BaseConfig, Level, Param, T


class InferenceConfig(BaseConfig):
    __title__ = "Vorhersage & Vertrauen"
    __tooltip__ = T(
        "Kalibrierung, Schwellen, Enthaltung, Conformal Prediction, Unsicherheit und TTA für Vorhersagen.",
        "Im Einsatz zählt nicht nur *was* die KI sagt, sondern *wie sicher* – und dass diese Sicherheit stimmt.",
        "Medizin, Finanzen, Qualitätskontrolle, alles mit Risiko. Einstellen mit Project.calibrate().",
    )
    temperature = Param(1.0, min=1e-3, tooltip=T(
        "Temperatur für die Wahrscheinlichkeiten (Logits / T). 1 = unverändert.",
        "Moderne Netze sind oft überkonfident; T > 1 macht sie ehrlich (Guo et al. 2017).",
        "Wird von Project.calibrate() bestimmt."), off=1.0)
    auto_calibrate = Param(False, tooltip=T(
        "Nach train() automatisch kalibrieren (Temperatur, Schwelle, Conformal) auf der Validierungsmenge.",
        "Kalibrierte Wahrscheinlichkeiten ohne zusätzlichen Schritt.", "Klassifikation mit Risiko: an."))
    threshold = Param(None, type=float, min=0.0, max=1.0, nullable=True, tooltip=T(
        "Entscheidungsschwelle für die positive Klasse (nur 2 Klassen; None = 0.5 bzw. argmax).",
        "Bei unausgewogenen Daten oder ungleichen Fehlerkosten ist 0.5 selten optimal.",
        "Wird mit optimize_threshold bestimmt."))
    optimize_threshold = Param("keine", choices={
        "keine": "nicht anpassen", "f1": "F1 der positiven Klasse maximieren",
        "balanced_accuracy": "mittlere Trefferquote beider Klassen maximieren",
        "youden": "Youden-J (Sensitivität + Spezifität − 1)"}, tooltip=T(
        "Kriterium für die Schwelle in Project.calibrate().", "Passend zu den Fehlerkosten.", "f1 bei seltenen Positiven."),
        on="f1")
    abstain_below = Param(None, type=float, min=0.0, max=1.0, nullable=True, tooltip=T(
        "Enthaltung: liegt die höchste Wahrscheinlichkeit darunter, lautet die Antwort 'unsicher'.",
        "Lieber an einen Menschen weitergeben als falsch entscheiden.", "0.6–0.9 je nach Risiko."), on=0.7)
    conformal_alpha = Param(None, type=float, min=0.001, max=0.5, nullable=True, tooltip=T(
        "Conformal Prediction: erlaubte Fehlerrate α (0.1 = Mengen/Intervalle enthalten in 90 % der Fälle die "
        "richtige Antwort).", "Statistisch garantierte Unsicherheitsangabe – für jedes Modell.",
        "0.05–0.2."), on=0.1)
    conformal_q = Param(None, type=float, nullable=True, level=Level.EXPERT, tooltip=T(
        "Gelernte Conformal-Schwelle (von calibrate bestimmt).", "Intern.", "Nicht von Hand setzen."))
    uncertainty = Param("keine", choices={
        "keine": "eine Vorhersage", "mc_dropout": "mehrere Durchläufe mit aktivem Dropout (Monte-Carlo)",
        "tta": "Test-Time Augmentation (Spiegeln/Rauschen)", "beides": "MC-Dropout und TTA"}, tooltip=T(
        "Unsicherheitsschätzung in predict_with_uncertainty / predict_proba.",
        "Zeigt, wo das Modell unsicher ist (Streuung der Vorhersagen).", "mc_dropout bei Modellen mit Dropout."),
        on="mc_dropout")
    samples = Param(16, min=2, tooltip=T(
        "Durchläufe für MC-Dropout bzw. TTA.", "Mehr = genauer, langsamer.", "8–32."), level=Level.ADVANCED)
    tta_transforms = Param(["hflip"], item_type=str, tooltip=T(
        "TTA-Varianten: hflip, vflip (Bilder), rauschen (Zahlen), keine.",
        "Mittel über Varianten ist robuster als eine einzelne Vorhersage.", "Bilder: hflip."), level=Level.ADVANCED)
    tta_noise = Param(0.05, min=0.0, tooltip=T(
        "Stärke des Rauschens für TTA 'rauschen' (in Standardabweichungen).", "Robustheit.", "0.01–0.1."),
        level=Level.ADVANCED)

    def check(self) -> list[str]:
        bad = [t for t in self.tta_transforms if t not in TTA]
        return [f"Unbekannte TTA-Varianten {bad}. Erlaubt: {', '.join(TTA)}."] if bad else []


TTA = {"keine": "unverändert", "hflip": "horizontal spiegeln", "vflip": "vertikal spiegeln",
       "rauschen": "leichtes Rauschen"}


# ===========================================================================
# Kalibrierung
# ===========================================================================

def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, steps: int = 100) -> float:
    """Temperatur, die die Kreuzentropie auf (Validierungs-)Logits minimiert (Temperature Scaling)."""
    logits, targets = logits.detach().float().reshape(-1, logits.shape[-1]), targets.detach().reshape(-1).long()
    keep = targets != -100
    logits, targets = logits[keep], targets[keep]
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    return t if math.isfinite(t) and 0.01 < t < 100 else 1.0


def best_threshold(p_pos: torch.Tensor, targets: torch.Tensor, criterion: str = "f1") -> float:
    """Schwelle für die positive Klasse, die ``criterion`` maximiert."""
    y = targets.reshape(-1).bool()
    best, best_t = -1.0, 0.5
    for t in torch.linspace(0.02, 0.98, 49).tolist():
        pred = p_pos >= t
        tp, fp = float((pred & y).sum()), float((pred & ~y).sum())
        fn, tn = float((~pred & y).sum()), float((~pred & ~y).sum())
        tpr, tnr = tp / max(1.0, tp + fn), tn / max(1.0, tn + fp)
        score = {"f1": 2 * tp / max(1e-12, 2 * tp + fp + fn), "balanced_accuracy": (tpr + tnr) / 2,
                 "youden": tpr + tnr - 1}[criterion]
        if score > best:
            best, best_t = score, t
    return round(best_t, 3)


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    """Split-Conformal: Quantil der Nichtkonformitäts-Werte mit endlicher Korrektur."""
    n = scores.numel()
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / max(1, n))
    return float(torch.quantile(scores.float(), level))


def conformal_scores(pred: torch.Tensor, targets: torch.Tensor, classification: bool) -> torch.Tensor:
    if classification:
        probs = pred.softmax(-1)
        return 1 - probs.gather(1, targets.reshape(-1, 1).long()).squeeze(1)
    return (pred.reshape(-1) - targets.float().reshape(-1)).abs()


def prediction_sets(probs: torch.Tensor, q: float) -> list[list[int]]:
    """Conformal-Mengen: alle Klassen mit Wahrscheinlichkeit ≥ 1 − q (mindestens die beste)."""
    out = []
    for row in probs:
        members = (row >= 1 - q).nonzero().flatten().tolist()
        out.append(members or [int(row.argmax())])
    return out


# ===========================================================================
# Unsicherheit
# ===========================================================================

def _dropout_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, nn.modules.dropout._DropoutNd) or
            type(m).__name__ in ("DropPath", "StochasticDepth")]


def tta_variants(x: torch.Tensor, transforms: list[str], noise: float) -> list[torch.Tensor]:
    out = [x]
    for t in transforms:
        if t == "hflip" and x.dim() == 4:
            out.append(x.flip(-1))
        elif t == "vflip" and x.dim() == 4:
            out.append(x.flip(-2))
        elif t == "rauschen" and x.is_floating_point():
            g = torch.Generator(device="cpu").manual_seed(len(out))
            out.append(x + noise * x.std() * torch.randn(x.shape, generator=g).to(x.device, x.dtype))
    return out


@torch.no_grad()
def sample_outputs(model: nn.Module, forward: Callable[[Any], torch.Tensor], x: Any, mode: str, samples: int,
                   transforms: list[str], noise: float) -> torch.Tensor:
    """Gestapelte Ausgaben ``[S, …]`` für MC-Dropout und/oder TTA."""
    inputs = tta_variants(x, transforms, noise) if mode in ("tta", "beides") and isinstance(x, torch.Tensor) else [x]
    drop = _dropout_modules(model) if mode in ("mc_dropout", "beides") else []
    runs = max(1, samples // len(inputs)) if drop else 1
    was = [m.training for m in drop]
    outs = []
    try:
        for m in drop:
            m.train()
        for inp in inputs:
            for _ in range(runs):
                outs.append(forward(inp).float())
    finally:
        for m, w in zip(drop, was):
            m.train(w)
    return torch.stack(outs)


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-12).log() * probs).sum(-1)
