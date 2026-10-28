"""
ki_baukasten.tensor.surgery
===========================

**Tensor-Chirurgie** – gezielte Eingriffe in einzelne Gewichtstensoren oder
sogar einzelne Ausschnitte davon.

Eine Regel (``TensorRuleConfig``) besteht aus
------------------------------------------
* **Auswahl**: Regex auf Parameternamen (``target``/``exclude``), Modultyp
  (``module_type``, z. B. ``"Linear"``), Art (weight/bias), Dimensionalität
* **Ausschnitt** (optional): ``index`` wie in Python, z. B. ``"0:16, :"``
  (erste 16 Zeilen), ``"..., -1"`` (letzte Spalte), ``"3"`` (Zeile 3)
* **Operation** (Registry ``tensor_op``) mit eigenen Argumenten

Beispiele::

    "tensor_rules": [
      {"target": "embed",            "op": "freeze"},
      {"target": "blocks\\.0\\..*q_proj.weight", "index": "0:8, :", "op": {"name": "scale", "factor": 0.5}},
      {"module_type": "Linear", "param_kind": "weight", "op": {"name": "prune", "amount": 0.3}},
      {"target": "lm_head",          "op": {"name": "lr_scale", "factor": 0.1}},
      {"target": "mixer.*o_proj",    "op": {"name": "constraint", "kind": "max_norm", "value": 2.0}},
      {"target": "head.weight",      "op": {"name": "grad_noise", "std": 1e-3}}
    ]

Wann wirkt welche Operation?
----------------------------
==============  ==============================================================
Phase           Operationen
==============  ==============================================================
Aufbau          init, scale, add, set, noise, clamp, prune, low_rank,
                orthogonalize, load, custom, tie, freeze, unfreeze
Optimierer      lr_scale, weight_decay (eigene Parametergruppen)
Gradient        grad_scale, grad_clip, grad_noise, grad_mask (+ prune, freeze-Ausschnitt)
nach jedem      constraint (max_norm, unit_norm, non_negative, clip, orthogonal),
Schritt         prune (Maske halten), freeze mit Ausschnitt (Werte festhalten)
==============  ==============================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..components.initializers import apply_initializer
from ..core import (BaseConfig, ComponentParam, ConfigError, Level, Param, T, create_registry)

TENSOR_OPS = create_registry(
    "tensor_op",
    "Tensor-Operationen",
    T(
        "Eingriffe in einzelne Gewichtstensoren (oder Ausschnitte).",
        "Feinste Kontrolle: Einfrieren, Initialisieren, Beschneiden, Beschränken, eigene Lernraten ...",
        "Transfer-Learning, Kompression, Forschung, Reparatur instabiler Schichten.",
    ),
    context_doc="factory(args) -> TensorOp",
)
TENSOR_FUNCTIONS = create_registry(
    "tensor_function",
    "Tensor-Funktionen (Chirurgie)",
    T(
        "Eigene Funktionen f(tensor) → tensor für die Operation 'custom'.",
        "Beliebige, selbst definierte Gewichtsmanipulationen.",
        "Forschung, Spezialfälle.",
    ),
    context_doc="factory(args) -> callable(tensor) -> tensor",
)


# ===========================================================================
# Index-Ausdrücke
# ===========================================================================

def parse_index(expr: str | None) -> tuple | None:
    """Wandelt ``"0:16, :, 3"`` sicher (ohne eval) in ein Index-Tupel um."""
    if expr is None or not str(expr).strip():
        return None
    parts: list[Any] = []
    for raw in str(expr).split(","):
        p = raw.strip()
        if p in ("...", "…"):
            parts.append(Ellipsis)
        elif ":" in p:
            nums = p.split(":")
            if len(nums) > 3:
                raise ConfigError(f"ungültiger Slice '{p}'.")
            try:
                parts.append(slice(*[int(n) if n.strip() else None for n in nums]))
            except ValueError:
                raise ConfigError(f"ungültiger Slice '{p}'.") from None
        else:
            try:
                parts.append(int(p))
            except ValueError:
                raise ConfigError(f"ungültiger Index '{p}' (erlaubt: Zahlen, a:b:s, ...).") from None
    return tuple(parts)


# ===========================================================================
# Operationen
# ===========================================================================

class TensorOp:
    """Basisklasse. Unterklassen überschreiben die benötigten Methoden."""

    #: True/False: setzt requires_grad des ganzen Parameters (nur ohne Index).
    requires_grad: bool | None = None

    def build(self, t: torch.Tensor, ctx: dict) -> None:
        """Einmalig beim Aufbau, ``t`` ist der (Ausschnitt des) Tensor(s) – in-place ändern."""

    def grad(self, g: torch.Tensor) -> torch.Tensor | None:
        """Gradienten-Transformation (oder ``None`` = keine)."""
        return None

    def post_step(self, t: torch.Tensor) -> None:
        """Nach jedem Optimierungsschritt (in-place)."""

    def group_override(self) -> dict[str, float] | None:
        """Optimierer-Einstellungen für den Parameter (lr_scale, weight_decay)."""
        return None

    has_grad = False
    has_post = False


class FreezeOp(TensorOp):
    requires_grad = False
    has_grad = True
    has_post = True

    def __init__(self):
        self.snapshot: dict[int, torch.Tensor] = {}

    # Bei Ausschnitten: Gradient maskieren + Werte nach jedem Schritt wiederherstellen.
    def build(self, t, ctx):
        self.snapshot[ctx["key"]] = t.detach().clone()

    def grad(self, g):
        return torch.zeros_like(g)

    def restore(self, t: torch.Tensor, key: int) -> None:
        t.copy_(self.snapshot[key].to(t.device, t.dtype))


class UnfreezeOp(TensorOp):
    requires_grad = True


class InitConfig(BaseConfig):
    initializer = ComponentParam("initializer", "normal", tooltip=T(
        "Initialisierer für diesen Tensor(-Ausschnitt).", "Gezielte Startwerte.", "z. B. zeros für Zero-Init."))


class InitOp(TensorOp):
    def __init__(self, a: InitConfig):
        self.a = a

    def build(self, t, ctx):
        tmp = torch.empty_like(t)
        apply_initializer(self.a.initializer, tmp, num_layers=ctx.get("num_layers"))
        t.copy_(tmp)


class FactorConfig(BaseConfig):
    factor = Param(1.0, tooltip=T("Faktor.", "Skaliert Werte bzw. Lernrate/Gradienten.", "0.1–10."))


class ValueConfig(BaseConfig):
    value = Param(0.0, tooltip=T("Wert.", "Wird gesetzt bzw. addiert.", "Beliebig."))


class NoiseConfig(BaseConfig):
    std = Param(0.01, min=0.0, tooltip=T("Rauschstärke.", "Störung der Werte.", "1e-3–1e-1."))
    distribution = Param("normal", choices=["normal", "uniform"], tooltip=T("Verteilung.", "Form des Rauschens.", "normal."))
    relative = Param(False, tooltip=T(
        "Relativ zur Streuung des Tensors.", "Skaleninvariante Störung.", "Robustheitstests."))


class ClampConfig(BaseConfig):
    min = Param(None, type=float, tooltip=T("Untergrenze.", "Begrenzt Werte nach unten.", "Beliebig."))
    max = Param(None, type=float, tooltip=T("Obergrenze.", "Begrenzt Werte nach oben.", "Beliebig."))


class PruneConfig(BaseConfig):
    amount = Param(0.5, min=0.0, max=1.0, tooltip=T(
        "Anteil zu entfernender Gewichte.", "Mehr = kleineres/schnelleres, aber schwächeres Modell.", "0.3–0.9."))
    method = Param("magnitude", choices={"magnitude": "kleinste Beträge entfernen", "random": "zufällig"},
                   tooltip=T("Auswahlkriterium.", "Magnitude ist fast immer besser.", "magnitude."))
    structure = Param("none", choices={"none": "einzelne Gewichte", "rows": "ganze Zeilen (Ausgabe-Neuronen)",
                                       "cols": "ganze Spalten (Eingänge)"}, tooltip=T(
        "Struktur des Prunings.", "Strukturiertes Pruning beschleunigt echte Hardware.", "Kompression."),
        level=Level.ADVANCED)


class LowRankConfig(BaseConfig):
    rank = Param(8, min=1, tooltip=T(
        "Ziel-Rang (SVD-Abschneiden).", "Entfernt 'feine' Richtungen, glättet die Abbildung.", "Kompressionsstudien."))


class LoadConfig(BaseConfig):
    path = Param("gewichte.pt", tooltip=T(
        "Datei mit Werten (.pt: Tensor oder dict; .npy: NumPy-Array).", "Übernimmt externe Gewichte.", "Transfer."))
    key = Param(None, type=str, tooltip=T("Schlüssel im dict (.pt).", "Welcher Tensor.", "Bei dict-Dateien."))


class TieConfig(BaseConfig):
    source = Param("", tooltip=T(
        "Exakter Name des Quellparameters, dessen Gewichte geteilt werden.",
        "Weight Tying: beide Stellen nutzen denselben Tensor (weniger Parameter, Regularisierung).",
        "Encoder/Decoder-Kopplung, Siamesische Teile."))


class GradClipConfig(BaseConfig):
    max_norm = Param(1.0, min=0.0, tooltip=T(
        "Maximale Norm des Gradienten dieses Tensors.", "Lokale Stabilisierung einzelner Schichten.", "0.1–10."))


class GradNoiseConfig(BaseConfig):
    std = Param(1e-3, min=0.0, tooltip=T("Rauschstärke auf den Gradienten.", "Erkundung, Regularisierung.", "1e-4–1e-2."))


class WeightDecayConfig(BaseConfig):
    value = Param(0.0, min=0.0, tooltip=T(
        "Weight-Decay nur für diese Parameter.", "Individuelle Regularisierung.", "0 (aus) bis 0.1."))


class ConstraintConfig(BaseConfig):
    kind = Param("max_norm", choices={
        "max_norm": "Zeilennormen auf ≤ value begrenzen (Srivastava et al.)",
        "unit_norm": "Zeilen auf Länge 1 normieren",
        "non_negative": "negative Werte auf 0 (z. B. monotone Netze)",
        "clip": "auf [-value, value] begrenzen",
        "orthogonal": "auf die nächste orthogonale Matrix projizieren (polar)",
    }, tooltip=T("Art der Beschränkung.", "Wird nach jedem Optimierungsschritt erzwungen.",
                 "Stabilität, Interpretierbarkeit, physikalische Randbedingungen."))
    value = Param(1.0, min=0.0, tooltip=T("Grenzwert.", "Stärke der Beschränkung.", "1–4."))
    every_n_steps = Param(1, min=1, tooltip=T("Nur alle n Schritte erzwingen.", "Spart Zeit (orthogonal ist teuer).",
                                              "1; orthogonal: 10."), level=Level.ADVANCED)


class CustomConfig(BaseConfig):
    function = ComponentParam("tensor_function", "identity", tooltip=T(
        "Eigene Funktion aus der Registry 'tensor_function'.", "Beliebige Manipulation.", "Eigene Registrierung."))


class _Simple(TensorOp):
    def __init__(self, fn):
        self.fn = fn

    def build(self, t, ctx):
        self.fn(t, ctx)


class PruneOp(TensorOp):
    has_grad = True
    has_post = True

    def __init__(self, a: PruneConfig):
        self.a = a
        self.masks: dict[int, torch.Tensor] = {}

    def build(self, t, ctx):
        a = self.a
        if a.structure == "none":
            score = t.abs() if a.method == "magnitude" else torch.rand_like(t)
            k = int(round(a.amount * t.numel()))
            mask = torch.ones_like(t, dtype=torch.bool)
            if k > 0:
                idx = score.flatten().argsort()[:k]
                mask.view(-1)[idx] = False
        else:
            dim = 0 if a.structure == "rows" else 1
            if t.dim() < 2:
                raise ConfigError("Strukturiertes Pruning braucht ≥ 2D-Tensoren.")
            other = [d for d in range(t.dim()) if d != dim]
            score = t.abs().pow(2).sum(other).sqrt() if a.method == "magnitude" else torch.rand(t.shape[dim])
            k = int(round(a.amount * t.shape[dim]))
            keep = torch.ones(t.shape[dim], dtype=torch.bool, device=t.device)
            if k > 0:
                keep[score.argsort()[:k]] = False
            shape = [1] * t.dim()
            shape[dim] = -1
            mask = keep.view(shape).expand_as(t)
        self.masks[ctx["key"]] = mask.clone()
        t.mul_(mask)

    def mask_for(self, key: int) -> torch.Tensor:
        return self.masks[key]


class ConstraintOp(TensorOp):
    has_post = True

    def __init__(self, a: ConstraintConfig):
        self.a = a
        self.calls = 0

    def apply(self, t: torch.Tensor) -> None:
        a = self.a
        if a.kind == "non_negative":
            t.clamp_(min=0)
        elif a.kind == "clip":
            t.clamp_(-a.value, a.value)
        elif t.dim() >= 2:
            flat = t.reshape(t.shape[0], -1)
            if a.kind == "max_norm":
                norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                flat.mul_(torch.clamp(a.value / norms, max=1.0))
            elif a.kind == "unit_norm":
                flat.div_(flat.norm(dim=1, keepdim=True).clamp_min(1e-12))
            else:  # orthogonal (polar-Zerlegung)
                u, _, vh = torch.linalg.svd(flat.float(), full_matrices=False)
                flat.copy_((u @ vh).to(flat.dtype))
        else:
            if a.kind == "max_norm":
                n = t.norm().clamp_min(1e-12)
                t.mul_(torch.clamp(a.value / n, max=1.0))
            elif a.kind == "unit_norm":
                t.div_(t.norm().clamp_min(1e-12))

    def build(self, t, ctx):
        self.apply(t)

    def post_step(self, t):
        self.calls += 1
        if self.calls % self.a.every_n_steps == 0:
            self.apply(t)


def _noise(a: NoiseConfig):
    def fn(t, ctx):
        scale = a.std * (float(t.float().std()) if a.relative and t.numel() > 1 else 1.0)
        if a.distribution == "uniform":
            t.add_((torch.rand_like(t) * 2 - 1) * scale)
        else:
            t.add_(torch.randn_like(t) * scale)
    return fn


def _low_rank(a: LowRankConfig):
    def fn(t, ctx):
        if t.dim() < 2:
            raise ConfigError("low_rank braucht ≥ 2D-Tensoren.")
        flat = t.reshape(t.shape[0], -1).float()
        u, s, vh = torch.linalg.svd(flat, full_matrices=False)
        r = min(a.rank, s.numel())
        t.copy_(((u[:, :r] * s[:r]) @ vh[:r]).reshape(t.shape).to(t.dtype))
    return fn


def _orthogonalize(t, ctx):
    if t.dim() < 2:
        raise ConfigError("orthogonalize braucht ≥ 2D-Tensoren.")
    flat = t.reshape(t.shape[0], -1).float()
    u, _, vh = torch.linalg.svd(flat, full_matrices=False)
    t.copy_((u @ vh).reshape(t.shape).to(t.dtype))


def _load(a: LoadConfig):
    def fn(t, ctx):
        path = Path(a.path)
        if path.suffix == ".npy":
            import numpy as np

            value = torch.from_numpy(np.load(path))
        else:
            value = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(value, dict):
                if a.key is None:
                    raise ConfigError(f"{path} enthält ein dict – bitte 'key' angeben.")
                value = value[a.key]
        if tuple(value.shape) != tuple(t.shape):
            raise ConfigError(f"Form {tuple(value.shape)} aus {path} passt nicht zu {tuple(t.shape)}.")
        t.copy_(value.to(t.dtype))
    return fn


class _GradOp(TensorOp):
    has_grad = True

    def __init__(self, fn):
        self.fn = fn

    def grad(self, g):
        return self.fn(g)


class _GroupOp(TensorOp):
    def __init__(self, override: dict[str, float]):
        self.override = override

    def group_override(self):
        return dict(self.override)


class TieOp(TensorOp):
    def __init__(self, a: TieConfig):
        self.a = a


R = TENSOR_OPS.register
R("freeze", lambda a: FreezeOp(), tooltip=T(
    "Einfrieren: nicht mehr trainieren (mit index: nur der Ausschnitt).",
    "Schützt vortrainiertes Wissen, spart Rechenzeit und Speicher.", "Transfer-Learning, Adapter-Training."))
R("unfreeze", lambda a: UnfreezeOp(), tooltip=T(
    "Auftauen: wieder trainierbar machen.", "Hebt ein früheres freeze auf (Regeln wirken in Reihenfolge).",
    "'Alles einfrieren außer ...'."))
R("init", lambda a: InitOp(a), config=InitConfig, tooltip=T(
    "Neu initialisieren mit beliebigem Initialisierer.", "Gezielte Startwerte einzelner Tensoren.",
    "Zero-Init von Ausgangsschichten, Re-Init von Köpfen."))
R("scale", lambda a: _Simple(lambda t, c: t.mul_(a.factor)), config=FactorConfig, tooltip=T(
    "Werte mit Faktor multiplizieren.", "Abschwächen/Verstärken einzelner Verbindungen.", "Experimente."))
R("add", lambda a: _Simple(lambda t, c: t.add_(a.value)), config=ValueConfig, tooltip=T(
    "Konstante addieren.", "Verschieben von Werten (z. B. Gate-Bias).", "Forget-Gate-Bias +1."))
R("set", lambda a: _Simple(lambda t, c: t.fill_(a.value)), config=ValueConfig, tooltip=T(
    "Auf festen Wert setzen.", "Exakte Kontrolle.", "Ablationen (z. B. einen Kopf auf 0)."))
R("noise", lambda a: _Simple(_noise(a)), config=NoiseConfig, tooltip=T(
    "Rauschen hinzufügen.", "Symmetriebruch, Robustheitstests, 'Shrink & Perturb'.", "Warm-Starts."))
R("clamp", lambda a: _Simple(lambda t, c: t.clamp_(a.min, a.max) if (a.min is not None or a.max is not None) else t),
  config=ClampConfig, tooltip=T("Werte einmalig begrenzen.", "Entfernt Ausreißer.", "Reparatur."))
R("prune", lambda a: PruneOp(a), config=PruneConfig, tooltip=T(
    "Pruning: Gewichte entfernen (Maske bleibt im Training erhalten).", "Kleinere, spärliche Modelle.",
    "Kompression, Lottery-Ticket-Experimente."))
R("low_rank", lambda a: _Simple(_low_rank(a)), config=LowRankConfig, level=Level.ADVANCED, tooltip=T(
    "Auf niedrigen Rang projizieren (SVD).", "Begrenzt die Komplexität einer Abbildung.", "Kompressionsstudien."))
R("orthogonalize", lambda a: _Simple(_orthogonalize), level=Level.ADVANCED, tooltip=T(
    "Einmalig orthogonalisieren.", "Normerhaltende Abbildung.", "RNNs, tiefe Netze."))
R("load", lambda a: _Simple(_load(a)), config=LoadConfig, tooltip=T(
    "Werte aus Datei laden.", "Externe/vortrainierte Gewichte für einzelne Tensoren.", "Transfer, Experimente."))
R("custom", lambda a: _Simple(lambda t, c: t.copy_(a.function.build()(t.clone()))), config=CustomConfig,
  level=Level.EXPERT, tooltip=T("Eigene Funktion anwenden.", "Unbegrenzte Freiheit.", "Forschung."))
R("tie", lambda a: TieOp(a), config=TieConfig, level=Level.ADVANCED, tooltip=T(
    "Gewichte mit anderem Parameter teilen (gleiches Tensor-Objekt).", "Weight Tying.", "Kopplung von Schichten."))
R("lr_scale", lambda a: _GroupOp({"lr_scale": a.factor}), config=FactorConfig, tooltip=T(
    "Eigene Lernrate (Faktor auf die Basis-Lernrate).", "Langsameres/schnelleres Lernen einzelner Teile.",
    "Kleine LR für Embeddings, große für neue Köpfe."))
R("weight_decay", lambda a: _GroupOp({"weight_decay": a.value}), config=WeightDecayConfig, tooltip=T(
    "Eigener Weight-Decay.", "Individuelle Regularisierung.", "Feintuning."))
R("grad_scale", lambda a: _GradOp(lambda g: g * a.factor), config=FactorConfig, tooltip=T(
    "Gradienten skalieren.", "Wie lr_scale, aber unabhängig vom Optimierer-Zustand.", "Experimente."))
R("grad_clip", lambda a: _GradOp(lambda g: g * torch.clamp(a.max_norm / g.norm().clamp_min(1e-12), max=1.0)),
  config=GradClipConfig, tooltip=T("Gradientennorm dieses Tensors begrenzen.", "Lokale Stabilität.",
                                   "Instabile Einzelschichten."))
R("grad_noise", lambda a: _GradOp(lambda g: g + torch.randn_like(g) * a.std), config=GradNoiseConfig,
  level=Level.ADVANCED, tooltip=T("Rauschen auf Gradienten.", "Regularisierung/Erkundung.", "Experimente."))
R("grad_mask", lambda a: _GradOp(lambda g: torch.zeros_like(g)), level=Level.ADVANCED, tooltip=T(
    "Gradienten nullen (Werte können sich durch Weight-Decay/Momentum noch ändern; freeze hält sie fest).",
    "Weiches Einfrieren.", "Experimente."))
R("constraint", lambda a: ConstraintOp(a), config=ConstraintConfig, tooltip=T(
    "Beschränkung nach jedem Schritt erzwingen.", "Stabilität, physikalische/strukturelle Randbedingungen.",
    "Max-Norm-Regularisierung, nicht-negative Gewichte, Orthogonalität."))

FN = TENSOR_FUNCTIONS.register
FN("identity", lambda a: (lambda t: t), tooltip=T("Unverändert.", "Platzhalter.", "Test."))
FN("abs", lambda a: torch.abs, tooltip=T("Beträge.", "Nur positive Gewichte.", "Experimente."))
FN("sign", lambda a: torch.sign, tooltip=T("Vorzeichen (-1/0/1).", "Binäre Gewichte.", "Binarisierungs-Experimente."))
FN("normalize_rows", lambda a: (lambda t: t / t.reshape(t.shape[0], -1).norm(dim=1).clamp_min(1e-12)
                                .view(-1, *([1] * (t.dim() - 1))) if t.dim() > 1 else t / t.norm().clamp_min(1e-12)),
   tooltip=T("Zeilen auf Länge 1.", "Normierte Gewichtsvektoren.", "Kosinus-Klassifikatoren."))
FN("flip", lambda a: (lambda t: t.flip(0)), tooltip=T("Reihenfolge umkehren (Achse 0).", "Permutation.", "Ablationen."))
FN("shuffle", lambda a: (lambda t: t.flatten()[torch.randperm(t.numel())].reshape(t.shape)), tooltip=T(
    "Werte zufällig permutieren (Verteilung bleibt, Struktur geht verloren).", "Kontrollexperiment.", "Ablationen."))


# ===========================================================================
# Regeln & Chirurg
# ===========================================================================

class TensorRuleConfig(BaseConfig):
    __title__ = "Tensor-Regel"
    __tooltip__ = T(
        "Eine Regel der Tensor-Chirurgie: Auswahl + optionaler Ausschnitt + Operation.",
        "Erlaubt Eingriffe bis auf einzelne Gewichte.",
        "Einfrieren, Initialisieren, Pruning, eigene Lernraten, Beschränkungen ...",
    )
    label = Param("", tooltip=T("Beschreibung (frei).", "Für die Übersicht.", "z. B. 'Embeddings einfrieren'."))
    target = Param(".*", tooltip=T(
        "Regulärer Ausdruck auf den Parameternamen (re.search).",
        "Wählt Parameter, z. B. 'blocks\\.0\\.', 'q_proj', 'bias$'.",
        "Namen zeigt 'python -m ki_baukasten parameter' bzw. model.parameter_table().",
    ))
    exclude = Param(None, type=str, tooltip=T(
        "Regex für Ausnahmen.", "Schließt Parameter wieder aus.", "z. B. 'norm'."), level=Level.ADVANCED)
    module_type = Param(None, type=str, tooltip=T(
        "Nur Parameter in Modulen dieses Typs (Klassenname, z. B. 'Linear', 'Conv2d', 'Embedding').",
        "Auswahl nach Schichtart.", "Pruning aller linearen Schichten."), level=Level.ADVANCED)
    param_kind = Param("any", choices={"any": "alle", "weight": "nur Gewichte", "bias": "nur Biases"}, tooltip=T(
        "Art des Parameters.", "Feinauswahl.", "Pruning nur auf Gewichten."), level=Level.ADVANCED)
    ndim = Param(None, type=int, min=0, tooltip=T(
        "Nur Tensoren mit genau so vielen Dimensionen.", "z. B. 2 = Matrizen.", "Orthogonalität nur für Matrizen."),
        level=Level.EXPERT)
    index = Param(None, type=str, tooltip=T(
        "Ausschnitt des Tensors (Python-Syntax ohne Klammern), z. B. '0:16, :' oder '..., -1'.",
        "Die Operation wirkt nur auf diesen Teil – bis hin zu einzelnen Gewichten.",
        "Einzelne Köpfe/Neuronen/Kanäle verändern oder einfrieren.",
    ), level=Level.ADVANCED)
    op = ComponentParam("tensor_op", "freeze", tooltip=T(
        "Operation (Registry 'tensor_op').", "Was mit den ausgewählten Tensoren passiert.", "Siehe Optionen."))
    enabled = Param(True, tooltip=T("Regel aktiv.", "Schnell an/aus schalten.", "Experimente."))


@dataclass
class _Binding:
    rule: TensorRuleConfig
    op: TensorOp
    name: str
    param: nn.Parameter
    index: tuple | None
    key: int = 0
    extra: dict = field(default_factory=dict)


class TensorSurgeon:
    """Wendet eine Liste von ``TensorRuleConfig`` auf ein Modell an."""

    def __init__(self, model: nn.Module, rules: list[TensorRuleConfig]):
        self.model = model
        self.rules = [r for r in rules if r.enabled]
        self.bindings: list[_Binding] = []
        self._hook_handles: list = []
        self._step = 0

    # ------------------------------------------------------------ Auswahl
    def _owner_types(self) -> dict[str, str]:
        out = {}
        for mname, module in self.model.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                out[f"{mname}.{pname}" if mname else pname] = type(module).__name__
        return out

    def matches(self, rule: TensorRuleConfig) -> list[tuple[str, nn.Parameter]]:
        owners = self._owner_types()
        rx = re.compile(rule.target)
        ex = re.compile(rule.exclude) if rule.exclude else None
        found = []
        for name, p in self.model.named_parameters(remove_duplicate=True):
            if not rx.search(name) or (ex and ex.search(name)):
                continue
            if rule.module_type and owners.get(name) != rule.module_type:
                continue
            last = name.split(".")[-1]
            if rule.param_kind == "weight" and "weight" not in last and last != "original":
                continue
            if rule.param_kind == "bias" and "bias" not in last:
                continue
            if rule.ndim is not None and p.dim() != rule.ndim:
                continue
            found.append((name, p))
        return found

    def preview(self) -> str:
        """Welche Parameter würde jede Regel treffen? (ohne etwas zu ändern)"""
        lines = []
        for i, rule in enumerate(self.rules):
            hits = self.matches(rule)
            title = rule.label or f"Regel {i}"
            lines.append(f"{title}: op={rule.op.name} target='{rule.target}' index={rule.index} → {len(hits)} Parameter")
            for name, p in hits[:12]:
                lines.append(f"    {name}  {tuple(p.shape)}")
            if len(hits) > 12:
                lines.append(f"    … und {len(hits) - 12} weitere")
        return "\n".join(lines) or "(keine Regeln)"

    # ------------------------------------------------------------ Aufbau
    def apply_build(self, num_layers: int | None = None) -> "TensorSurgeon":
        """Führt alle Aufbau-Operationen aus und bereitet Hooks/Gruppen vor."""
        for rule in self.rules:
            op = rule.op.build()
            hits = self.matches(rule)
            if not hits:
                raise ConfigError(f"Tensor-Regel '{rule.label or rule.target}' trifft keinen Parameter.")
            index = parse_index(rule.index)
            for name, p in hits:
                b = _Binding(rule, op, name, p, index, key=len(self.bindings))
                if isinstance(op, TieOp):
                    self._tie(name, op.a.source)
                    continue
                if op.requires_grad is not None and index is None:
                    p.requires_grad_(op.requires_grad)
                with torch.no_grad():
                    view = p.data[index] if index is not None else p.data
                    if isinstance(op, FreezeOp) and index is None:
                        pass  # ganzer Parameter: requires_grad=False genügt
                    else:
                        op.build(view, {"key": b.key, "name": name, "num_layers": num_layers})
                self.bindings.append(b)
        return self

    def _tie(self, target_name: str, source_name: str) -> None:
        params = dict(self.model.named_parameters(remove_duplicate=False))
        if source_name not in params:
            raise ConfigError(f"tie: Quellparameter '{source_name}' existiert nicht.")
        src = params[source_name]
        if tuple(src.shape) != tuple(params[target_name].shape):
            raise ConfigError(f"tie: Formen passen nicht ({source_name} {tuple(src.shape)} vs. "
                              f"{target_name} {tuple(params[target_name].shape)}).")
        owner_name, _, attr = target_name.rpartition(".")
        owner = self.model.get_submodule(owner_name) if owner_name else self.model
        setattr(owner, attr, src)

    # ------------------------------------------------------------ Training
    def install_hooks(self) -> "TensorSurgeon":
        """Registriert Gradienten-Hooks (einmal, vor dem Training)."""
        per_param: dict[int, list[_Binding]] = {}
        for b in self.bindings:
            needs = b.op.has_grad and not (isinstance(b.op, FreezeOp) and b.index is None)
            if needs:
                per_param.setdefault(id(b.param), []).append(b)
        for bs in per_param.values():
            param = bs[0].param

            def hook(g: torch.Tensor, bs=bs) -> torch.Tensor:
                g = g.clone()
                for b in bs:
                    if isinstance(b.op, PruneOp):
                        m = b.op.mask_for(b.key).to(g.device)
                        if b.index is None:
                            g.mul_(m)
                        else:
                            g[b.index] = g[b.index] * m
                        continue
                    sub = g[b.index] if b.index is not None else g
                    new = b.op.grad(sub)
                    if new is not None:
                        if b.index is None:
                            g = new
                        else:
                            g[b.index] = new
                return g

            if param.requires_grad:
                self._hook_handles.append(param.register_hook(hook))
        return self

    def overrides_for(self, param: nn.Parameter) -> dict[str, float]:
        """Optimierer-Overrides (lr_scale, weight_decay) für einen Parameter (spätere Regeln gewinnen)."""
        out: dict[str, float] = {}
        for b in self.bindings:
            if b.param is param:
                o = b.op.group_override()
                if o:
                    out.update(o)
        return out

    @torch.no_grad()
    def post_step(self) -> None:
        """Nach jedem Optimierungsschritt: Masken, Beschränkungen, festgehaltene Ausschnitte."""
        self._step += 1
        for b in self.bindings:
            if not b.op.has_post or b.param.numel() == 0:
                continue
            view = b.param.data[b.index] if b.index is not None else b.param.data
            if isinstance(b.op, PruneOp):
                view.mul_(b.op.mask_for(b.key).to(view.device, view.dtype))
            elif isinstance(b.op, FreezeOp):
                if b.index is not None:
                    b.op.restore(view, b.key)
            else:
                b.op.post_step(view)

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def report(self) -> str:
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        lines = [f"Tensor-Chirurgie: {len(self.rules)} Regeln, {len(self.bindings)} Bindungen; eingefroren "
                 + f"{frozen:,}/{total:,}".replace(",", ".") + " Parameter"]
        for b in self.bindings[:30]:
            lines.append(f"  {b.rule.op.name:<13} {b.name}" + (f"[{b.rule.index}]" if b.index is not None else ""))
        if len(self.bindings) > 30:
            lines.append(f"  … {len(self.bindings) - 30} weitere")
        return "\n".join(lines)
