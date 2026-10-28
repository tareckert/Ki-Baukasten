"""
ki_baukasten.training.schedulers
================================

Lernraten-Zeitpläne – wie sich die Lernrate im Laufe des Trainings ändert.

Warum?
------
Zu Beginn sind große Schritte gefährlich (zufällige Gewichte, instabile
Statistiken von Adam) → **Warmup**. Gegen Ende helfen kleine Schritte, um
fein in ein Minimum zu konvergieren → **Abklingen** (Decay).

Alle Zeitpläne arbeiten **pro Optimierungsschritt** und haben gemeinsame
Einstellungen (``warmup_steps``/``warmup_ratio``, ``warmup_type``,
``min_lr_ratio``). Die Lernrate ergibt sich als ``lr · faktor(schritt)``.

Registry ``scheduler``
----------------------
Fabrik-Signatur: ``factory(args, optimizer, total_steps) -> Scheduler`` mit
``step()``, ``state_dict()``, ``load_state_dict()``. Metrikbasierte Zeitpläne
(``plateau``) haben zusätzlich ``step_metric(wert)``.
"""

from __future__ import annotations

import bisect
import math
from typing import Callable

import torch
from torch.optim.lr_scheduler import LambdaLR

from ..core import BaseConfig, Level, Param, T, create_registry

SCHEDULERS = create_registry(
    "scheduler",
    "Lernraten-Zeitpläne",
    T(
        "Verlauf der Lernrate über das Training.",
        "Warmup verhindert frühe Instabilität, Abklingen verbessert die Endqualität.",
        "Praktisch jedes Training größerer Modelle.",
    ),
    context_doc="factory(args, optimizer, total_steps)",
)


class SchedulerConfig(BaseConfig):
    """Gemeinsame Einstellungen aller Zeitpläne."""

    __title__ = "Zeitplan"
    warmup_steps = Param(None, type=int, min=0, tooltip=T(
        "Anzahl Aufwärmschritte (überschreibt warmup_ratio).",
        "Lernrate steigt langsam an – schützt vor Divergenz am Anfang.",
        "Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte.",
    ))
    warmup_ratio = Param(0.05, min=0.0, max=1.0, tooltip=T(
        "Aufwärmphase als Anteil aller Schritte.", "Skaliert automatisch mit der Trainingslänge.", "0.01–0.1."))
    warmup_type = Param("linear", choices={"linear": "linearer Anstieg", "cosine": "weicher Kosinus-Anstieg",
                                           "exponential": "exponentieller Anstieg"}, tooltip=T(
        "Form des Anstiegs.", "Feinheit; linear ist Standard.", "Selten zu ändern."), level=Level.EXPERT)
    warmup_start_factor = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).", "Sanfter oder härterer Start.", "0–0.1."),
        level=Level.EXPERT)
    min_lr_ratio = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Untergrenze der Lernrate relativ zur Start-Lernrate.",
        "Verhindert, dass das Training am Ende 'einfriert'.", "LLMs: 0.1."), level=Level.ADVANCED)

    def warmup(self, total: int) -> int:
        return self.warmup_steps if self.warmup_steps is not None else int(round(self.warmup_ratio * total))


class ConstantConfig(SchedulerConfig):
    warmup_ratio = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Aufwärmphase als Anteil aller Schritte.", "Optional auch bei konstanter Lernrate.", "0 oder 0.05."))


class CosineRestartsConfig(SchedulerConfig):
    cycles = Param(3, min=1, tooltip=T(
        "Anzahl Kosinus-Zyklen (Neustarts).", "Neustarts können aus flachen Minima herausspringen (SGDR).",
        "2–5."))
    cycle_decay = Param(1.0, min=0.0, max=1.0, tooltip=T(
        "Faktor, mit dem jeder neue Zyklus kleiner startet.", "Allmählich vorsichtiger werden.", "0.5–1."),
        level=Level.ADVANCED)


class PolynomialConfig(SchedulerConfig):
    power = Param(1.0, min=0.0, tooltip=T(
        "Exponent des Abklingens (1 = linear, 2 = quadratisch).", "Form der Kurve.", "1–2."))


class ExponentialConfig(SchedulerConfig):
    decay_rate = Param(0.96, min=0.0, max=1.0, tooltip=T(
        "Faktor pro decay_steps Schritte.", "Geometrisches Abklingen.", "0.9–0.99."))
    decay_steps = Param(100, min=1, tooltip=T("Schritte pro Abklingstufe.", "Granularität.", "Je nach Länge."))


class StepConfig(SchedulerConfig):
    warmup_ratio = Param(0.0, min=0.0, max=1.0, tooltip=T("Aufwärmphase.", "Optional.", "0."))
    milestones = Param([0.5, 0.75], item_type=float, min=0.0, tooltip=T(
        "Zeitpunkte der Reduktion (Werte < 1 = Anteil der Schritte, sonst Schrittzahl).",
        "Klassisch für ResNet-Training (bei 50 % und 75 %).", "[0.5, 0.75]."))
    gamma = Param(0.1, min=0.0, max=1.0, tooltip=T("Faktor je Meilenstein.", "Stärke der Stufe.", "0.1."))


class InverseSqrtConfig(SchedulerConfig):
    warmup_ratio = Param(0.1, min=0.0, max=1.0, tooltip=T("Aufwärmphase.", "Danach ∝ 1/√schritt.", "Original-Transformer."))


class OneCycleConfig(SchedulerConfig):
    warmup_ratio = Param(0.3, min=0.0, max=1.0, tooltip=T(
        "Anteil des Anstiegs zur Maximal-Lernrate (pct_start).", "Langer Anstieg, dann Abfall.", "0.25–0.3."))
    div_factor = Param(25.0, min=1.0, tooltip=T("Start-LR = max_lr / div_factor.", "Startniveau.", "25."),
                       level=Level.ADVANCED)
    final_div_factor = Param(1e4, min=1.0, tooltip=T("End-LR = Start-LR / final_div_factor.", "Endniveau.", "1e4."),
                             level=Level.ADVANCED)


class WSDConfig(SchedulerConfig):
    decay_ratio = Param(0.2, min=0.0, max=1.0, tooltip=T(
        "Anteil der Schritte für die finale Abklingphase.",
        "Warmup-Stable-Decay: lange konstante Phase, kurzes Abklingen am Ende – erlaubt Weitertrainieren.",
        "LLM-Vortraining (MiniCPM, DeepSeek): 0.1–0.2.",
    ))
    decay_shape = Param("linear", choices=["linear", "cosine", "sqrt"], tooltip=T(
        "Form des Abklingens.", "sqrt fällt am Ende steil ab.", "linear oder sqrt."), level=Level.ADVANCED)


class CyclicConfig(SchedulerConfig):
    warmup_ratio = Param(0.0, min=0.0, max=1.0, tooltip=T("Aufwärmphase.", "Optional.", "0."))
    cycle_steps = Param(200, min=2, tooltip=T("Länge eines Zyklus.", "Periode der Schwingung.", "2–10 Epochen."))
    low_ratio = Param(0.1, min=0.0, max=1.0, tooltip=T("Tiefpunkt relativ zur LR.", "Amplitude.", "0.1."))


class PlateauConfig(SchedulerConfig):
    warmup_ratio = Param(0.0, min=0.0, max=1.0, tooltip=T("Aufwärmphase.", "Optional.", "0."))
    factor = Param(0.5, min=0.0, max=1.0, tooltip=T(
        "Reduktionsfaktor bei Stillstand.", "Wie stark die LR fällt.", "0.1–0.5."))
    patience = Param(2, min=0, tooltip=T(
        "Evaluierungen ohne Verbesserung bis zur Reduktion.", "Geduld.", "2–10."))
    mode = Param("min", choices=["min", "max"], tooltip=T(
        "Ob die Metrik kleiner (Verlust) oder größer (Genauigkeit) besser ist.", "Richtung.", "Verlust: min."))
    threshold = Param(1e-4, min=0.0, tooltip=T("Mindestverbesserung (relativ).", "Rauschen ignorieren.", "1e-4."),
                      level=Level.EXPERT)


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================

def _with_warmup(a: SchedulerConfig, total: int, body: Callable[[float], float]) -> Callable[[int], float]:
    """Kombiniert Warmup mit einem Hauptverlauf ``body(fortschritt∈[0,1]) -> faktor``."""
    w = a.warmup(total)
    floor = a.min_lr_ratio

    def fn(step: int) -> float:
        if w > 0 and step < w:
            p = (step + 1) / w
            if a.warmup_type == "cosine":
                p = 0.5 * (1 - math.cos(math.pi * p))
            elif a.warmup_type == "exponential":
                p = p ** 3
            return a.warmup_start_factor + (1 - a.warmup_start_factor) * p
        progress = min(1.0, (step - w) / max(1, total - w))
        return floor + (1 - floor) * body(progress)

    return fn


class PlateauScheduler:
    """Reduziert die Lernrate, wenn sich eine Metrik nicht mehr verbessert (mit optionalem Warmup).

    Intern ein ``LambdaLR`` mit Faktor ``warmup(schritt) · scale``; ``step_metric``
    verkleinert ``scale`` nach ``patience`` Evaluierungen ohne Verbesserung.
    """

    def __init__(self, a: PlateauConfig, optimizer: torch.optim.Optimizer, total: int):
        self.a = a
        self.scale = 1.0
        self.best: float | None = None
        self.bad = 0
        warm = _with_warmup(a, total, lambda p: 1.0)
        self.inner = LambdaLR(optimizer, lambda step: max(warm(step) * self.scale, a.min_lr_ratio))

    def step(self) -> None:
        self.inner.step()

    def step_metric(self, value: float) -> None:
        a = self.a
        if self.best is None:
            better = True
        elif a.mode == "min":
            better = value < self.best - abs(self.best) * a.threshold
        else:
            better = value > self.best + abs(self.best) * a.threshold
        if better:
            self.best, self.bad = value, 0
            return
        self.bad += 1
        if self.bad > a.patience:
            self.scale *= a.factor
            self.bad = 0

    def get_last_lr(self) -> list[float]:
        return self.inner.get_last_lr()

    def state_dict(self) -> dict:
        return {"inner": self.inner.state_dict(), "scale": self.scale, "best": self.best, "bad": self.bad}

    def load_state_dict(self, state: dict) -> None:
        self.inner.load_state_dict(state["inner"])
        self.scale, self.best, self.bad = state["scale"], state["best"], state["bad"]


# ===========================================================================
# Registrierung
# ===========================================================================

R = SCHEDULERS.register


def _lambda(body_factory: Callable[[BaseConfig, int], Callable[[float], float]]):
    def factory(a: SchedulerConfig, optimizer, total_steps: int):
        return LambdaLR(optimizer, _with_warmup(a, max(1, total_steps), body_factory(a, max(1, total_steps))))
    return factory


R("constant", _lambda(lambda a, n: (lambda p: 1.0)), config=ConstantConfig, aliases=("none", "konstant"), tooltip=T(
    "Konstante Lernrate (optional mit Warmup).", "Einfach, gut zum Debuggen und für kurze Trainings.", "Prototypen."))
R("linear", _lambda(lambda a, n: (lambda p: 1.0 - p)), config=SchedulerConfig, aliases=("warmup_linear",), tooltip=T(
    "Warmup, dann linear auf min_lr fallen.", "Einfach und wirksam.", "BERT-Feintuning."))
R("warmup_cosine", _lambda(lambda a, n: (lambda p: 0.5 * (1 + math.cos(math.pi * p)))), config=SchedulerConfig,
  aliases=("cosine",), tags=("transformer",), tooltip=T(
      "Warmup, dann Kosinus-Abfall.", "Weiches Abklingen, Standard moderner Trainings.", "Transformer, ViT, LLMs."))


def _restarts(a: CosineRestartsConfig, n: int) -> Callable[[float], float]:
    def body(p: float) -> float:
        c = min(int(p * a.cycles), a.cycles - 1)
        local = p * a.cycles - c
        return (a.cycle_decay ** c) * 0.5 * (1 + math.cos(math.pi * local))
    return body


R("cosine_restarts", _lambda(_restarts), config=CosineRestartsConfig, aliases=("sgdr",), level=Level.ADVANCED,
  tooltip=T("Kosinus mit Neustarts (SGDR).", "Periodische Neustarts erkunden neue Minima.", "Lange Trainings."))
R("polynomial", _lambda(lambda a, n: (lambda p: (1 - p) ** a.power)), config=PolynomialConfig, level=Level.ADVANCED,
  tooltip=T("Polynomielles Abklingen.", "Flexible Kurvenform.", "Segmentierung (DeepLab: power 0.9)."))
R("exponential", _lambda(lambda a, n: (lambda p: a.decay_rate ** (p * n / a.decay_steps))), config=ExponentialConfig,
  level=Level.ADVANCED, tooltip=T("Exponentielles Abklingen.", "Gleichmäßig geometrisch.", "Klassische Netze, RL."))


def _step_body(a: StepConfig, n: int) -> Callable[[float], float]:
    marks = sorted(m / n if m >= 1 else m for m in a.milestones)
    return lambda p: a.gamma ** bisect.bisect_right(marks, p)


R("step", _lambda(_step_body), config=StepConfig, aliases=("multistep",), tooltip=T(
    "Stufenweise Reduktion an Meilensteinen.", "Klassisch und interpretierbar.", "ResNet-Training mit SGD."))
R("inverse_sqrt", _lambda(lambda a, n: (lambda p: 1.0 / math.sqrt(1 + p * (n / max(1, a.warmup(n)))))),
  config=InverseSqrtConfig, level=Level.ADVANCED, tooltip=T(
      "Warmup, dann ∝ 1/√Schritt.", "Kein festes Trainingsende nötig.", "Original-Transformer, fairseq."))


def _one_cycle(a: OneCycleConfig, optimizer, total_steps: int):
    n = max(1, total_steps)
    w = max(1, a.warmup(n))
    start = 1.0 / a.div_factor
    end = start / a.final_div_factor

    def fn(step: int) -> float:
        if step < w:
            return start + (1 - start) * 0.5 * (1 - math.cos(math.pi * (step + 1) / w))
        p = min(1.0, (step - w) / max(1, n - w))
        return end + (1 - end) * 0.5 * (1 + math.cos(math.pi * p))

    return LambdaLR(optimizer, fn)


R("one_cycle", _one_cycle, config=OneCycleConfig, aliases=("onecycle",), tooltip=T(
    "1cycle-Politik (Smith): steigen auf Maximum, dann tief abfallen.",
    "Sehr schnelle Konvergenz ('Super-Convergence').", "CNNs, kurze Trainingsbudgets."))


def _wsd(a: WSDConfig, n: int) -> Callable[[float], float]:
    w = a.warmup(n)
    stable_end = 1.0 - a.decay_ratio * n / max(1, n - w) if n > w else 1.0

    def body(p: float) -> float:
        if p <= stable_end:
            return 1.0
        q = (p - stable_end) / max(1e-9, 1 - stable_end)
        if a.decay_shape == "cosine":
            return 0.5 * (1 + math.cos(math.pi * q))
        if a.decay_shape == "sqrt":
            return 1 - math.sqrt(q)
        return 1 - q
    return body


R("wsd", _lambda(_wsd), config=WSDConfig, aliases=("warmup_stable_decay",), level=Level.ADVANCED, tooltip=T(
    "Warmup-Stable-Decay.", "Konstante Hauptphase, kurzes Abklingen; Checkpoints vor dem Decay weitertrainierbar.",
    "LLM-Vortraining."))


def _cyclic(a: CyclicConfig, n: int) -> Callable[[float], float]:
    def body(p: float) -> float:
        phase = (p * n / a.cycle_steps) % 1.0
        tri = 1 - abs(2 * phase - 1)
        return a.low_ratio + (1 - a.low_ratio) * tri
    return body


R("cyclic", _lambda(_cyclic), config=CyclicConfig, level=Level.EXPERT, tooltip=T(
    "Dreieckige zyklische Lernrate.", "Pendelt zwischen tiefer und hoher LR.", "LR-Bereichssuche, Experimente."))
R("plateau", lambda a, opt, n: PlateauScheduler(a, opt, max(1, n)), config=PlateauConfig, aliases=("reduce_on_plateau",),
  tooltip=T("Reduktion bei Stillstand der Validierungsmetrik.", "Adaptiv, ohne festen Plan.",
            "Wenn die Trainingslänge unklar ist."))
