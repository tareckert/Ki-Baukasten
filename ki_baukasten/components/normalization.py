"""
ki_baukasten.components.normalization
=====================================

Normalisierungsschichten – halten Aktivierungen in einem gesunden Wertebereich.

Warum normalisieren?
--------------------
In tiefen Netzen können Aktivierungen von Schicht zu Schicht immer größer
oder kleiner werden. Normalisierung stabilisiert das, erlaubt höhere
Lernraten und macht das Training robuster gegenüber Initialisierung.
Unterschiede liegen vor allem darin, **über welche Achsen** normalisiert wird:

==============  ==============================  ==============================
Verfahren       normalisiert über               typisch für
==============  ==============================  ==============================
LayerNorm       Features eines Elements         Transformer, RNNs
RMSNorm         Features (nur Skala, kein Mean)  moderne LLMs (LLaMA, Mistral)
BatchNorm       Batch (pro Kanal)               CNNs mit großen Batches
GroupNorm       Kanalgruppen eines Elements     CNNs mit kleinen Batches
InstanceNorm    Raum pro Kanal & Element        Stil-Transfer, Generatoren
==============  ==============================  ==============================

Registry ``normalization``
--------------------------
Fabrik-Signatur: ``factory(args, num_features, dims="feature")``

* ``num_features``: Größe der normalisierten Feature-/Kanal-Dimension.
* ``dims``: Datenlayout

  - ``"feature"`` → Features in der **letzten** Dimension (``[B, C]`` oder
    ``[B, T, C]``) – Transformer, MLPs, RNNs.
  - ``1``/``2``/``3`` → Faltungslayout mit Kanälen in Dimension 1
    (``[B, C, L]``, ``[B, C, H, W]``, ``[B, C, D, H, W]``).

Jede Normalisierung funktioniert in jedem Layout – nötige Achsen-
Vertauschungen übernimmt der Baukasten automatisch.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..core import BaseConfig, Level, Param, T, create_registry

NORMALIZATIONS = create_registry(
    "normalization",
    "Normalisierungen",
    T(
        "Schichten, die Aktivierungen auf eine stabile Skala bringen.",
        "Ermöglichen tiefere Netze, höhere Lernraten und stabileres Training.",
        "Praktisch in jeder modernen Architektur.",
    ),
    context_doc="factory(args, num_features, dims='feature'|1|2|3)",
)


# ===========================================================================
# Layout-Adapter
# ===========================================================================

class ChannelFirstToLast(nn.Module):
    """Wendet ein Modul, das auf der *letzten* Achse arbeitet, auf Kanal-zuerst-Daten an."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 2:
            return self.module(x)
        return self.module(x.movedim(1, -1)).movedim(-1, 1)


class LastToChannelFirst(nn.Module):
    """Wendet ein Modul, das Kanäle in Achse 1 erwartet, auf ``[..., C]``-Daten an."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 2:
            return self.module(x)
        return self.module(x.movedim(-1, 1)).movedim(1, -1)


def _for_last_dim(module: nn.Module, dims: object) -> nn.Module:
    return module if dims == "feature" else ChannelFirstToLast(module)


def _for_channel_first(module: nn.Module, dims: object) -> nn.Module:
    return LastToChannelFirst(module) if dims == "feature" else module


def _conv_dims(dims: object) -> int:
    return 1 if dims == "feature" else int(dims)  # type: ignore[arg-type]


# ===========================================================================
# Eigene Normalisierungen
# ===========================================================================

class RMSNorm(nn.Module):
    """Root-Mean-Square-Norm: x / rms(x) · g (ohne Mittelwertabzug).

    ``scale_offset=True`` speichert ``g - 1`` (Start bei 0) – so machen es
    Gemma-Modelle; Weight-Decay zieht dann Richtung g = 1 statt g = 0.
    """

    def __init__(self, dim: int, eps: float, learnable: bool, scale_offset: bool, compute_in_fp32: bool):
        super().__init__()
        self.eps = eps
        self.scale_offset = scale_offset
        self.compute_in_fp32 = compute_in_fp32
        init = torch.zeros(dim) if scale_offset else torch.ones(dim)
        self.weight = nn.Parameter(init) if learnable else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float() if self.compute_in_fp32 else x
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = h.to(dtype)
        if self.weight is None:
            return h
        w = self.weight + 1.0 if self.scale_offset else self.weight
        return h * w.to(dtype)


class ScaleNorm(nn.Module):
    """g · x / ||x|| mit einem einzigen lernbaren Skalar g (Nguyen & Salazar 2019)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.tensor(math.sqrt(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g * x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)


class L2Norm(nn.Module):
    """Projektion auf die Einheitskugel (ohne Parameter) – vgl. nGPT."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)


class DynamicTanh(nn.Module):
    """DyT: γ · tanh(α · x) + β (Zhu et al. 2025, "Transformers without Normalization")."""

    def __init__(self, dim: int, alpha_init: float, affine: bool):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.alpha * x)
        if self.weight is not None:
            h = h * self.weight + self.bias
        return h


class FlattenBatchNorm(nn.Module):
    """BatchNorm für ``[B, T, C]``: behandelt alle Positionen als Stichproben."""

    def __init__(self, bn: nn.Module):
        super().__init__()
        self.bn = bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.bn(x)
        shape = x.shape
        return self.bn(x.reshape(-1, shape[-1])).reshape(shape)


# ===========================================================================
# Argument-Konfigurationen
# ===========================================================================

_EPS_TIP = T(
    "Kleine Zahl gegen Division durch 0.",
    "Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.",
    "fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8).",
)


class LayerNormConfig(BaseConfig):
    eps = Param(1e-5, min=0.0, level=Level.EXPERT, tooltip=_EPS_TIP)
    elementwise_affine = Param(True, level=Level.ADVANCED, tooltip=T(
        "Lernbare Skala (γ) und Verschiebung (β) pro Feature.",
        "Gibt dem Netz die Möglichkeit, die Normalisierung teilweise rückgängig zu machen.",
        "Fast immer an; aus für parameterfreie Normierung.",
    ))
    bias = Param(True, level=Level.ADVANCED, tooltip=T(
        "Lernbare Verschiebung β verwenden.",
        "Ohne Bias etwas weniger Parameter; manche LLMs verzichten darauf.",
        "Moderne Transformer (oft ohne Bias).",
    ))


class RMSNormConfig(BaseConfig):
    eps = Param(1e-6, min=0.0, level=Level.EXPERT, tooltip=_EPS_TIP)
    learnable = Param(True, level=Level.ADVANCED, tooltip=T(
        "Lernbare Skala pro Feature.", "Wie γ bei LayerNorm.", "Standard: an."))
    scale_offset = Param(False, level=Level.EXPERT, tooltip=T(
        "Skala als (1 + w) parametrisieren, w startet bei 0.",
        "Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.",
        "Gemma-Stil.",
    ))
    compute_in_fp32 = Param(True, level=Level.EXPERT, tooltip=T(
        "Normalisierung intern in 32-Bit rechnen.",
        "Verhindert Genauigkeitsverlust bei fp16/bf16-Training.",
        "Mixed-Precision-Training.",
    ))


class BatchNormConfig(BaseConfig):
    eps = Param(1e-5, min=0.0, level=Level.EXPERT, tooltip=_EPS_TIP)
    momentum = Param(0.1, min=0.0, max=1.0, nullable=True, level=Level.ADVANCED, tooltip=T(
        "Gewicht neuer Batches im laufenden Mittelwert/Varianz.",
        "Klein = glattere, trägere Statistik; None = kumulativer Durchschnitt.",
        "Kleine/rauschige Batches → kleineres Momentum.",
    ))
    affine = Param(True, level=Level.ADVANCED, tooltip=T("Lernbare Skala/Verschiebung.", "Wie γ/β.", "Standard: an."))
    track_running_stats = Param(True, level=Level.EXPERT, tooltip=T(
        "Laufende Statistik für die Inferenz mitführen.",
        "Aus → auch bei Inferenz Batch-Statistik (Ergebnis hängt dann vom Batch ab!).",
        "Spezialfälle (z. B. Test-Time-Adaption).",
    ))


class GroupNormConfig(BaseConfig):
    num_groups = Param(8, min=1, tooltip=T(
        "Anzahl Kanalgruppen.",
        "1 Gruppe = LayerNorm über Kanäle+Raum; Gruppen = Kanäle → InstanceNorm.",
        "CNNs mit kleinen Batches (Detektion, Segmentierung, Diffusions-U-Nets).",
        "Kanalanzahl muss durch num_groups teilbar sein – sonst wird automatisch der größte Teiler genommen.",
    ))
    eps = Param(1e-5, min=0.0, level=Level.EXPERT, tooltip=_EPS_TIP)
    affine = Param(True, level=Level.ADVANCED, tooltip=T("Lernbare Skala/Verschiebung.", "Wie γ/β.", "Standard."))


class InstanceNormConfig(BaseConfig):
    eps = Param(1e-5, min=0.0, level=Level.EXPERT, tooltip=_EPS_TIP)
    affine = Param(False, tooltip=T("Lernbare Skala/Verschiebung.", "Wie γ/β.", "Stil-Transfer: oft an."))
    track_running_stats = Param(False, level=Level.EXPERT, tooltip=T(
        "Laufende Statistik mitführen.", "Normalerweise aus bei InstanceNorm.", "Selten."))


class ScaleNormConfig(BaseConfig):
    eps = Param(1e-5, min=0.0, level=Level.EXPERT, tooltip=_EPS_TIP)


class DyTConfig(BaseConfig):
    alpha_init = Param(0.5, tooltip=T(
        "Startwert von α in tanh(α·x).",
        "Bestimmt, ab welcher Größe Werte gesättigt werden.",
        "Normalisierungsfreie Transformer.",
        "Paper empfiehlt 0.5 (größere Modelle evtl. kleiner).",
    ))
    affine = Param(True, tooltip=T("Lernbare Skala/Verschiebung.", "Wie γ/β.", "Standard."))


def _groups(num_features: int, wanted: int) -> int:
    """Größter Teiler von ``num_features``, der ≤ ``wanted`` ist (robuste GroupNorm)."""
    g = max(1, min(wanted, num_features))
    while num_features % g:
        g -= 1
    return g


# ===========================================================================
# Registrierung
# ===========================================================================

R = NORMALIZATIONS.register


@R("layernorm", config=LayerNormConfig, aliases=("ln", "layer_norm"), tooltip=T(
    "Layer Normalization: Mittelwert 0, Varianz 1 über die Features jedes Elements.",
    "Unabhängig von der Batchgröße, stabil für Sequenzen.",
    "Standard in Transformern (BERT, GPT-2) und RNNs.",
))
def _layernorm(a: LayerNormConfig, num_features: int, dims: object = "feature") -> nn.Module:
    return _for_last_dim(nn.LayerNorm(num_features, eps=a.eps, elementwise_affine=a.elementwise_affine,
                                      bias=a.bias), dims)


@R("rmsnorm", config=RMSNormConfig, aliases=("rms",), tooltip=T(
    "Root-Mean-Square-Normalisierung (nur Skalierung, kein Zentrieren).",
    "Billiger als LayerNorm bei praktisch gleicher Qualität.",
    "Moderne LLMs (LLaMA, Mistral, Gemma).",
))
def _rmsnorm(a: RMSNormConfig, num_features: int, dims: object = "feature") -> nn.Module:
    return _for_last_dim(RMSNorm(num_features, a.eps, a.learnable, a.scale_offset, a.compute_in_fp32), dims)


@R("batchnorm", config=BatchNormConfig, aliases=("bn", "batch_norm"), tooltip=T(
    "Batch Normalization: normalisiert jeden Kanal über den Batch.",
    "Beschleunigt CNN-Training stark, wirkt leicht regularisierend.",
    "CNNs mit Batchgröße ≥ 16.",
    "Bei kleinen Batches oder Sequenzmodellen besser GroupNorm/LayerNorm.",
))
def _batchnorm(a: BatchNormConfig, num_features: int, dims: object = "feature") -> nn.Module:
    kw = dict(eps=a.eps, momentum=a.momentum, affine=a.affine, track_running_stats=a.track_running_stats)
    if dims == "feature":
        return FlattenBatchNorm(nn.BatchNorm1d(num_features, **kw))
    cls = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[_conv_dims(dims)]
    return cls(num_features, **kw)


@R("groupnorm", config=GroupNormConfig, aliases=("gn", "group_norm"), tooltip=T(
    "Group Normalization: normalisiert Kanalgruppen pro Element.",
    "Batchgrößen-unabhängig, gut für CNNs mit kleinen Batches.",
    "Detektion, Segmentierung, Diffusions-U-Nets.",
))
def _groupnorm(a: GroupNormConfig, num_features: int, dims: object = "feature") -> nn.Module:
    gn = nn.GroupNorm(_groups(num_features, a.num_groups), num_features, eps=a.eps, affine=a.affine)
    return _for_channel_first(gn, dims)


@R("instancenorm", config=InstanceNormConfig, aliases=("in", "instance_norm"), level=Level.ADVANCED, tooltip=T(
    "Instance Normalization: normalisiert jeden Kanal jedes Elements über den Raum.",
    "Entfernt Kontrast-/Stil-Informationen pro Bild.",
    "Stil-Transfer, Bild-Generatoren.",
))
def _instancenorm(a: InstanceNormConfig, num_features: int, dims: object = "feature") -> nn.Module:
    cls = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[_conv_dims(dims)]
    mod = cls(num_features, eps=a.eps, affine=a.affine, track_running_stats=a.track_running_stats)
    return _for_channel_first(mod, dims)


@R("scalenorm", config=ScaleNormConfig, level=Level.EXPERT, tooltip=T(
    "ScaleNorm: g · x / ||x|| mit einem einzigen lernbaren Skalar.",
    "Extrem wenige Parameter, schnelleres Konvergieren in kleinen Transformern.",
    "Maschinelle Übersetzung mit wenig Daten.",
))
def _scalenorm(a: ScaleNormConfig, num_features: int, dims: object = "feature") -> nn.Module:
    return _for_last_dim(ScaleNorm(num_features, a.eps), dims)


@R("l2norm", config=ScaleNormConfig, level=Level.EXPERIMENTAL, tooltip=T(
    "Projektion auf die Einheitskugel (Länge 1), ohne Parameter.",
    "Alle Repräsentationen haben dieselbe Norm – Basis von 'normalisierten' Transformern (nGPT).",
    "Forschung, Embedding-Räume mit Kosinus-Ähnlichkeit.",
))
def _l2norm(a: ScaleNormConfig, num_features: int, dims: object = "feature") -> nn.Module:
    return _for_last_dim(L2Norm(a.eps), dims)


@R("dyt", config=DyTConfig, aliases=("dynamic_tanh",), level=Level.EXPERIMENTAL, tooltip=T(
    "Dynamic Tanh: γ·tanh(α·x)+β als Ersatz für Normalisierung.",
    "Keine Statistik-Berechnung nötig; laut Paper gleichwertig zu LayerNorm in Transformern.",
    "Experimente mit normalisierungsfreien Transformern.",
))
def _dyt(a: DyTConfig, num_features: int, dims: object = "feature") -> nn.Module:
    return _for_last_dim(DynamicTanh(num_features, a.alpha_init, a.affine), dims)


@R("identity", aliases=("none", "keine"), tooltip=T(
    "Keine Normalisierung.",
    "Spart Rechenzeit; nötig für Ablationen oder selbstnormalisierende Netze (SELU).",
    "Flache Netze, SELU-Netze, Experimente.",
))
def _identity(a: BaseConfig, num_features: int = 0, dims: object = "feature") -> nn.Module:
    return nn.Identity()


def build_norm(spec: object, num_features: int, dims: object = "feature") -> nn.Module:
    """Baut eine Normalisierung (``None`` → Identität)."""
    if spec is None:
        return nn.Identity()
    return NORMALIZATIONS.build(spec, num_features, dims=dims)
