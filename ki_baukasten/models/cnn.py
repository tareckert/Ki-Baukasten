"""
ki_baukasten.models.cnn
=======================

Faltungsnetze (CNN) für 1D-, 2D- und 3D-Daten.

Aufbau
------
::

    Eingabe ─► Stem ─► Stufe 1 ─► Stufe 2 ─► … ─► Kopf
                       (n Blöcke)  (n Blöcke)      (Pooling + MLP  oder  dichte Ausgabe)

* **Stem**: erste Faltung(en), oft mit großem Kernel und Downsampling.
* **Stufen** (``stages``): jede Stufe hat eigene Kanalzahl, Blockanzahl,
  Blocktyp, Kernelgröße, Normalisierung, Aktivierung, SE-Attention ...
* **Blocktypen**: ``plain`` (VGG), ``residual`` (ResNet-Basic),
  ``bottleneck`` (ResNet-50+), ``inverted_residual`` (MobileNetV2/EfficientNet),
  ``convnext`` (ConvNeXt), ``depthwise_separable`` (MobileNetV1).
* **Kopf**: ``classification`` (Pooling + MLP → Vektor) oder ``dense``
  (1×1-Faltung pro Pixel – Segmentierung, Heatmaps).
* **Abweichungen einzelner Blöcke** (global durchnummeriert) über
  ``block_overrides``.
"""

from __future__ import annotations

import torch
from torch import nn

from ..components.layers import DropPath, GlobalPool, SqueezeExcitation, _conv_cls
from ..components.normalization import build_norm
from ..core import (BaseConfig, ComponentParam, ConfigListParam, Level, NestedParam, Param, T,
                    resolve_layer_config, validate_overrides)
from .base import MODELS, BaseModel, ModelConfig

_BLOCK_TYPES = {
    "plain": "Faltung → Norm → Aktivierung (VGG-Stil)",
    "residual": "zwei 3×3-Faltungen + Skip (ResNet-18/34)",
    "bottleneck": "1×1 → k×k → 1×1 + Skip (ResNet-50+)",
    "inverted_residual": "1×1-Expansion → Tiefenfaltung → 1×1-Projektion (MobileNetV2/EfficientNet)",
    "convnext": "Tiefenfaltung 7×7 → Norm → 1×1-Expansion → GELU → 1×1 + LayerScale (ConvNeXt)",
    "depthwise_separable": "Tiefen- + Punktfaltung (MobileNetV1)",
}


class ConvBlockConfig(BaseConfig):
    """Gemeinsame Block-Einstellungen (pro Stufe und pro Block überschreibbar)."""

    __title__ = "Faltungsblock"
    block_type = Param("residual", choices=_BLOCK_TYPES, tooltip=T(
        "Bauart der Blöcke.",
        "Bestimmt Tiefe-vs-Effizienz-Abwägung und Gradientenfluss.",
        "residual: robuster Standard; inverted_residual: effizient; convnext: modern & stark.",
    ))
    kernel_size = Param(3, min=1, tooltip=T(
        "Kernelgröße der (Haupt-)Faltung.", "Größer = mehr Kontext pro Schicht.", "3 Standard; ConvNeXt: 7."))
    expansion = Param(4.0, min=0.1, tooltip=T(
        "Expansionsfaktor (bottleneck: Reduktion; inverted_residual/convnext: Aufweitung).",
        "Steuert Breite der inneren Schicht.",
        "bottleneck 4, MobileNetV2 6, ConvNeXt 4.",
    ), level=Level.ADVANCED)
    groups = Param(1, min=1, tooltip=T(
        "Gruppen der Hauptfaltung (ResNeXt-Kardinalität).", "Mehr Gruppen = effizienter.", "ResNeXt: 32."),
        level=Level.EXPERT)
    dilation = Param(1, min=1, tooltip=T(
        "Dilatation der Hauptfaltung.", "Größeres rezeptives Feld ohne Downsampling.", "Segmentierung."),
        level=Level.EXPERT)
    norm = ComponentParam("normalization", "batchnorm", tooltip=T(
        "Normalisierung in den Blöcken.", "BatchNorm Standard; GroupNorm bei kleinen Batches.", "CNNs."))
    activation = ComponentParam("activation", "relu", tooltip=T(
        "Aktivierung in den Blöcken.", "relu klassisch; silu (EfficientNet); gelu (ConvNeXt).", "CNNs."))
    se_ratio = Param(None, type=float, min=1.0, tooltip=T(
        "Squeeze-Excitation-Reduktion (None = aus, z. B. 4 oder 16).",
        "Kanal-Attention verbessert Genauigkeit bei geringen Kosten.",
        "EfficientNet, SE-ResNet.",
    ), level=Level.ADVANCED)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Kanal-Dropout im Block.", "Regularisierung.", "Kleine Datensätze."), group="Regularisierung")
    drop_path = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Stochastic Depth: Wahrscheinlichkeit, den Residual-Zweig auszulassen.",
        "Starke Regularisierung tiefer Residual-Netze.",
        "Tiefe ResNets/ConvNeXt (0.1–0.4).",
    ), level=Level.ADVANCED, group="Regularisierung")
    layer_scale_init = Param(1e-6, type=float, min=0.0, nullable=True, tooltip=T(
        "Startwert der LayerScale (nur convnext; None = aus).",
        "Kleiner Start macht tiefe Netze stabil.", "ConvNeXt: 1e-6."), level=Level.EXPERT)
    bias = Param(False, tooltip=T(
        "Bias in den Faltungen.", "Vor Normalisierung überflüssig.", "Ohne Normalisierung: an."), level=Level.EXPERT)
    padding_mode = Param("zeros", choices=["zeros", "reflect", "replicate", "circular"], tooltip=T(
        "Randauffüllung.", "Beeinflusst Randartefakte.", "circular für periodische Daten."), level=Level.EXPERT)


class StageConfig(ConvBlockConfig):
    """Eine Stufe: n Blöcke gleicher Kanalzahl (erbt alle Block-Einstellungen)."""

    __title__ = "CNN-Stufe"
    channels = Param(64, min=1, tooltip=T(
        "Ausgabekanäle der Stufe.", "Mehr Kanäle = mehr Merkmale.", "Typisch verdoppelnd: 64, 128, 256, 512."))
    num_blocks = Param(2, min=1, tooltip=T(
        "Anzahl Blöcke der Stufe.", "Tiefe der Stufe.", "ResNet-18: 2,2,2,2; ResNet-50: 3,4,6,3."))
    stride = Param(2, min=1, tooltip=T(
        "Schrittweite des ersten Blocks (Downsampling).", "2 halbiert die Auflösung.", "Erste Stufe oft 1."))
    pool_after = Param("none", choices=["none", "max", "avg"], tooltip=T(
        "Zusätzliches Pooling nach der Stufe.", "Extra-Downsampling.", "Plain-Netze (VGG)."), level=Level.ADVANCED)


class StemConfig(BaseConfig):
    __title__ = "Stem"
    channels = Param(32, min=1, tooltip=T("Kanäle des Stems.", "Breite der ersten Merkmalsschicht.", "32–64."))
    kernel_size = Param(3, min=1, tooltip=T("Kernelgröße.", "Groß (7) für hochauflösende Bilder.", "3 klein, 7 ImageNet."))
    stride = Param(1, min=1, tooltip=T("Schrittweite.", "Frühes Downsampling spart Rechenzeit.", "1 kleine, 2 große Bilder."))
    pool = Param("none", choices=["none", "max"], tooltip=T(
        "Max-Pooling nach dem Stem.", "Weiteres Downsampling.", "ResNet auf ImageNet: max."))
    norm = ComponentParam("normalization", "batchnorm", tooltip=T("Normalisierung im Stem.", "Stabilität.", "CNNs."))
    activation = ComponentParam("activation", "relu", tooltip=T("Aktivierung im Stem.", "Nichtlinearität.", "CNNs."))
    patchify = Param(False, tooltip=T(
        "Stem als nicht-überlappende Patch-Faltung (kernel = stride).",
        "ConvNeXt/ViT-artig: grobe Patches statt Pixel.", "ConvNeXt (4×4-Patches)."), level=Level.ADVANCED)


def _default_stages() -> list[StageConfig]:
    return [StageConfig(channels=32, num_blocks=1, stride=1), StageConfig(channels=64, num_blocks=2, stride=2),
            StageConfig(channels=128, num_blocks=2, stride=2)]


class CNNConfig(ModelConfig):
    __title__ = "CNN"
    __tooltip__ = T(
        "Faltungsnetz aus Stem, Stufen und Kopf.",
        "Nutzt lokale Strukturen und Verschiebungsinvarianz – sehr dateneffizient bei Bildern/Signalen.",
        "Bildklassifikation, Segmentierung, Audio, Sensorsignale, 3D-Volumen.",
    )

    dims = Param(2, choices={1: "1D (Signale)", 2: "2D (Bilder)", 3: "3D (Volumen/Video)"}, tooltip=T(
        "Räumliche Dimensionalität.", "Bestimmt Faltungstyp und Eingabeform [B, C, *raum].", "Bilder: 2."))
    in_channels = Param(3, min=1, tooltip=T(
        "Eingabekanäle.", "RGB = 3, Graustufen = 1, Mehrkanal-Sensoren = n.", "Muss zu den Daten passen."))
    input_size = Param([32, 32], item_type=int, min=1, tooltip=T(
        "Räumliche Eingabegröße (für example_input und 'flatten'-Köpfe).",
        "Wird für Dummy-Eingaben/Speicherschätzung und den Flatten-Kopf benötigt.",
        "Länge = dims, z. B. [32, 32] oder [16000] für 1D.",
    ))
    stem = NestedParam(StemConfig)
    stages = ConfigListParam(StageConfig, default_factory=_default_stages, min_items=1, tooltip=T(
        "Liste der Stufen (jede mit eigenen Einstellungen).",
        "Frei kombinierbar: z. B. erst residual, dann inverted_residual mit SE.",
        "Tiefe und Breite des Netzes.",
    ))
    block_overrides = Param({}, tooltip=T(
        "Abweichungen einzelner Blöcke (global durchnummeriert über alle Stufen).",
        "Jeder einzelne Block kann anders sein.",
        "Feinstes Tuning, z. B. {'-1': {'se_ratio': 4}}.",
    ), level=Level.ADVANCED)
    drop_path_schedule = Param("linear", choices={
        "constant": "alle Blöcke gleich", "linear": "linear von 0 bis zum eingestellten Wert ansteigend"},
        tooltip=T("Verteilung der Stochastic-Depth-Raten.", "Tiefe Blöcke häufiger auslassen ist üblich.",
                  "Bei drop_path > 0."), level=Level.EXPERT)
    task = Param("classification", choices={
        "classification": "globales Pooling + MLP → Vektor [B, output_dim]",
        "dense": "1×1-Faltung pro Position → [B, output_dim, *raum] (Segmentierung)",
    }, tooltip=T("Art der Ausgabe.", "Ein Wert pro Bild vs. ein Wert pro Pixel.", "Klassifikation vs. Segmentierung."))
    global_pool = Param("avg", choices=["avg", "max", "avgmax", "flatten"], tooltip=T(
        "Verdichtung der Feature-Maps vor dem Kopf.",
        "avg ist robust; flatten behält Ortsinformation (braucht feste input_size).",
        "Klassifikation.",
    ))
    head_hidden_dims = Param([], item_type=int, min=1, tooltip=T(
        "Versteckte Schichten des Klassifikationskopfs.", "Mehr Kapazität im Kopf.", "Meist leer (linearer Kopf)."))
    head_dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout vor dem Kopf.", "Regularisierung.", "EfficientNet: 0.2–0.5."), group="Regularisierung")
    head_activation = ComponentParam("activation", "relu", tooltip=T(
        "Aktivierung im Kopf-MLP.", "Nichtlinearität.", "Nur mit head_hidden_dims."), level=Level.ADVANCED)
    output_dim = Param(10, min=1, tooltip=T("Anzahl Ausgaben/Klassen.", "Form der Vorhersage.", "Klassenzahl."))
    upsample_dense = Param(True, tooltip=T(
        "Dichte Ausgabe auf Eingabeauflösung hochskalieren.", "Pixelgenaue Masken.", "Segmentierung."),
        level=Level.ADVANCED)

    def num_blocks(self) -> int:
        return sum(s.num_blocks for s in self.stages)

    def check(self) -> list[str]:
        problems = []
        if len(self.input_size) != self.dims:
            problems.append(f"input_size hat {len(self.input_size)} Werte, dims ist {self.dims}.")
        base = self.stages[0] if self.stages else StageConfig()
        problems += validate_overrides(base, self.block_overrides, max(1, self.num_blocks()))
        return problems


# ===========================================================================
# Blöcke
# ===========================================================================

class ConvBlock(nn.Module):
    """Ein Block gemäß ``block_type``."""

    def __init__(self, dims: int, cin: int, cout: int, stride: int, cfg: ConvBlockConfig):
        super().__init__()
        Conv = _conv_cls(dims)
        k, t = cfg.kernel_size, cfg.block_type
        pad = (k // 2) * cfg.dilation
        common = dict(bias=cfg.bias, padding_mode=cfg.padding_mode)

        def norm(c: int) -> nn.Module:
            return build_norm(cfg.norm, c, dims)

        def act() -> nn.Module:
            return cfg.activation.build()

        self.residual = t in ("residual", "bottleneck", "inverted_residual", "convnext")
        layers: list[nn.Module] = []
        if t == "plain":
            layers = [Conv(cin, cout, k, stride, pad, dilation=cfg.dilation, groups=cfg.groups, **common), norm(cout), act()]
        elif t == "residual":
            layers = [Conv(cin, cout, k, stride, pad, dilation=cfg.dilation, groups=cfg.groups, **common), norm(cout), act(),
                      Conv(cout, cout, k, 1, pad, dilation=cfg.dilation, **common), norm(cout)]
        elif t == "bottleneck":
            mid = max(1, int(cout / cfg.expansion))
            mid = max(cfg.groups, mid // cfg.groups * cfg.groups)
            layers = [Conv(cin, mid, 1, **common), norm(mid), act(),
                      Conv(mid, mid, k, stride, pad, dilation=cfg.dilation, groups=cfg.groups, **common), norm(mid), act(),
                      Conv(mid, cout, 1, **common), norm(cout)]
        elif t == "inverted_residual":
            mid = max(1, int(cin * cfg.expansion))
            layers = []
            if mid != cin:
                layers += [Conv(cin, mid, 1, **common), norm(mid), act()]
            layers += [Conv(mid, mid, k, stride, pad, dilation=cfg.dilation, groups=mid, **common), norm(mid), act()]
            if cfg.se_ratio:
                layers.append(SqueezeExcitation(mid, cfg.se_ratio, cfg.activation.build(), nn.Sigmoid()))
            layers += [Conv(mid, cout, 1, **common), norm(cout)]
        elif t == "convnext":
            mid = max(1, int(cout * cfg.expansion))
            layers = [Conv(cin, cout, k, stride, pad, groups=cin if cin == cout else 1, **common), norm(cout),
                      Conv(cout, mid, 1, bias=True), act(), Conv(mid, cout, 1, bias=True)]
        elif t == "depthwise_separable":
            layers = [Conv(cin, cin, k, stride, pad, groups=cin, **common), norm(cin), act(),
                      Conv(cin, cout, 1, **common), norm(cout), act()]
        if cfg.se_ratio and t not in ("inverted_residual",):
            layers.append(SqueezeExcitation(cout, cfg.se_ratio, cfg.activation.build(), nn.Sigmoid()))
        if cfg.dropout > 0:
            layers.append({1: nn.Dropout1d, 2: nn.Dropout2d, 3: nn.Dropout3d}[dims](cfg.dropout))
        self.body = nn.Sequential(*layers)

        self.gamma = None
        if t == "convnext" and cfg.layer_scale_init is not None:
            self.gamma = nn.Parameter(torch.full((cout,), cfg.layer_scale_init))
        self.drop_path = DropPath(cfg.drop_path)
        self.shortcut: nn.Module = nn.Identity()
        if self.residual and (cin != cout or stride != 1):
            self.shortcut = nn.Sequential(Conv(cin, cout, 1, stride, bias=False), norm(cout))
        # Nach residual/bottleneck folgt klassisch eine Aktivierung auf der Summe.
        self.post_act = act() if t in ("residual", "bottleneck") else nn.Identity()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if self.gamma is not None:
            h = h * self.gamma.view(1, -1, *([1] * self.dims))
        if not self.residual:
            return h
        return self.post_act(self.shortcut(x) + self.drop_path(h))


class CNN(BaseModel):
    """CNN gemäß ``CNNConfig``. Eingabe ``[B, C, *raum]``."""

    def __init__(self, config: CNNConfig):
        super().__init__(config)
        config.validate()
        d = config.dims
        Conv = _conv_cls(d)
        st = config.stem
        k = st.stride if st.patchify else st.kernel_size
        stem: list[nn.Module] = [
            Conv(config.in_channels, st.channels, k, st.stride, 0 if st.patchify else k // 2, bias=False),
            build_norm(st.norm, st.channels, d), st.activation.build(),
        ]
        if st.pool == "max":
            stem.append({1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}[d](3, 2, 1))
        self.stem = nn.Sequential(*stem)

        total = config.num_blocks()
        blocks: list[nn.Module] = []
        cin, idx = st.channels, 0
        for stage in config.stages:
            for b in range(stage.num_blocks):
                cfg = resolve_layer_config(stage, config.block_overrides, idx, total)
                if config.drop_path_schedule == "linear" and total > 1:
                    cfg.drop_path = cfg.drop_path * idx / (total - 1)
                blocks.append(ConvBlock(d, cin, stage.channels, stage.stride if b == 0 else 1, cfg))
                cin, idx = stage.channels, idx + 1
            if stage.pool_after != "none":
                pool = {"max": (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d),
                        "avg": (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)}[stage.pool_after][d - 1]
                blocks.append(pool(2))
        self.blocks = nn.ModuleList(blocks)

        if config.task == "dense":
            self.head = Conv(cin, config.output_dim, 1)
            self.pool = None
        else:
            if config.global_pool == "flatten":
                with torch.no_grad():
                    feats = self._features(torch.zeros(1, config.in_channels, *config.input_size)).flatten(1).shape[1]
                self.pool = nn.Flatten(1)
            else:
                feats = cin * (2 if config.global_pool == "avgmax" else 1)
                self.pool = GlobalPool(config.global_pool)
            dims_ = [feats] + list(config.head_hidden_dims)
            head: list[nn.Module] = [nn.Dropout(config.head_dropout)]
            for i in range(len(config.head_hidden_dims)):
                head += [nn.Linear(dims_[i], dims_[i + 1]), config.head_activation.build()]
            head.append(nn.Linear(dims_[-1], config.output_dim))
            self.head = nn.Sequential(*head)
        self.finish()

    def depth(self) -> int:
        return len(self.blocks)

    def checkpointable_blocks(self) -> list[nn.Module]:
        return list(self.blocks)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._features(x)
        if self.pool is None:
            out = self.head(h)
            if self.config.upsample_dense and out.shape[2:] != x.shape[2:]:
                mode = {1: "linear", 2: "bilinear", 3: "trilinear"}[self.config.dims]
                out = nn.functional.interpolate(out, size=x.shape[2:], mode=mode, align_corners=False)
            return out
        return self.head(self.pool(h))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Gepoolte Merkmale ohne Kopf (für Fusion/Transfer-Learning)."""
        h = self._features(x)
        return self.pool(h) if self.pool is not None else h

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        return torch.randn(batch_size, self.config.in_channels, *self.config.input_size)


MODELS.register("cnn", CNN, config=CNNConfig, aliases=("convnet", "faltungsnetz"), tags=("vision", "audio"), tooltip=T(
    "Faltungsnetz (1D/2D/3D) mit frei konfigurierbaren Stufen und Blocktypen.",
    "Effizient und dateneffizient für räumlich strukturierte Daten.",
    "Bilder, Audio, Sensorsignale, Volumendaten.",
))
