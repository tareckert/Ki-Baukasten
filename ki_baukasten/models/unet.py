"""
ki_baukasten.models.unet
========================

**U-Net** (Ronneberger et al. 2015) – der Standard für **Segmentierung** (eine Klasse je Pixel):
Medizin (Tumore, Organe), Satellitenbilder, Qualitätskontrolle, autonomes Fahren; auch das
Rückgrat von Diffusionsmodellen.

Aufbau: Encoder (Faltungen + Verkleinern) → Engpass → Decoder (Vergrößern + Faltungen), wobei
jede Decoder-Stufe über **Skip-Verbindungen** die feinen Details der passenden Encoder-Stufe erhält.

Eingabe ``[B, in_channels, H, W]`` → Logits ``[B, output_dim, H, W]`` (Verlust: Kreuzentropie über die
Kanal-Achse bzw. ``dice``/``focal``; Metriken: ``iou``, ``dice``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..components.normalization import build_norm
from ..core import ComponentParam, Level, Param, T
from .base import MODELS, BaseModel, ModelConfig


class UNetConfig(ModelConfig):
    __title__ = "U-Net (Segmentierung)"
    __tooltip__ = T(
        "Encoder-Decoder-Faltungsnetz mit Skip-Verbindungen; eine Vorhersage je Pixel.",
        "Kombiniert Kontext (tiefe Stufen) mit Details (Skip-Verbindungen) – sehr dateneffizient.",
        "Segmentierung: Medizin, Satellit, Industrie; Bild-zu-Bild-Aufgaben.",
    )
    in_channels = Param(3, min=1, tooltip=T("Eingabekanäle.", "RGB = 3, Graustufen = 1.", "Muss zu den Daten passen."))
    output_dim = Param(2, min=1, tooltip=T(
        "Anzahl Klassen je Pixel (inkl. Hintergrund).", "Form der Ausgabe [B, Klassen, H, W].", "Binär: 2 (oder 1)."))
    input_size = Param([64, 64], item_type=int, length=2, min=4, tooltip=T(
        "Bildgröße (für Beispiel-Eingaben und Speicherschätzung).", "Muss durch 2^depth teilbar sein – sonst wird "
        "intern aufgefüllt.", "64–512."))
    base_channels = Param(16, min=1, tooltip=T(
        "Kanäle der ersten Stufe (verdoppeln sich je Stufe).", "Kapazität.", "Klein: 16; Original: 64."))
    depth = Param(3, min=1, max=6, tooltip=T(
        "Anzahl Verkleinerungs-Stufen.", "Tiefer = mehr Kontext, größere Objekte.", "3–5."))
    norm = ComponentParam("normalization", "batchnorm", tooltip=T(
        "Normalisierung in den Blöcken.", "BatchNorm Standard; GroupNorm bei kleinen Batches.", "batchnorm."))
    activation = ComponentParam("activation", "relu", tooltip=T(
        "Aktivierung.", "Nichtlinearität.", "relu; modern: silu/gelu."))
    upsampling = Param("transposed", choices={
        "transposed": "transponierte Faltung (lernbar)", "bilinear": "bilinear + Faltung (weniger Schachbrett-Artefakte)"},
        tooltip=T("Art des Vergrößerns.", "bilinear vermeidet Artefakte.", "transposed."), level=Level.ADVANCED)
    dropout = Param(0.0, min=0.0, max=1.0, group="Regularisierung", tooltip=T(
        "Dropout im Engpass.", "Regularisierung.", "Kleine Datensätze: 0.1–0.3."))


class _Double(nn.Module):
    def __init__(self, cin: int, cout: int, cfg: UNetConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), build_norm(cfg.norm, cout, dims=2), cfg.activation.build(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), build_norm(cfg.norm, cout, dims=2), cfg.activation.build())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(BaseModel):
    def __init__(self, config: UNetConfig):
        super().__init__(config)
        config.validate()
        c = config
        ch = [c.base_channels * 2 ** i for i in range(c.depth + 1)]
        self.down = nn.ModuleList([_Double(c.in_channels if i == 0 else ch[i - 1], ch[i], c) for i in range(c.depth)])
        self.bottleneck = _Double(ch[c.depth - 1], ch[c.depth], c)
        self.drop = nn.Dropout2d(c.dropout)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(c.depth)):
            if c.upsampling == "transposed":
                self.up.append(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2))
            else:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                             nn.Conv2d(ch[i + 1], ch[i], 1)))
            self.dec.append(_Double(ch[i] * 2, ch[i], c))
        self.head = nn.Conv2d(ch[0], c.output_dim, 1)
        self.finish()

    def depth(self) -> int:
        return self.config.depth

    def checkpointable_blocks(self) -> list[nn.Module]:
        return list(self.down) + [self.bottleneck] + list(self.dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = 2 ** self.config.depth
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.drop(self.bottleneck(x))
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)[..., :h, :w]

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        return torch.randn(batch_size, self.config.in_channels, *self.config.input_size)


MODELS.register("unet", UNet, config=UNetConfig, aliases=("u_net", "segmentierungsnetz"), tags=("bild", "segmentierung"),
                tooltip=UNetConfig.__tooltip__)
