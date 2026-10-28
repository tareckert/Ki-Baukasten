"""
ki_baukasten.components.mixers
==============================

Sequenz-Mixer – der austauschbare "Kommunikationsteil" eines Blocks.

Idee
----
Ein Transformer-Block besteht aus zwei Teilen: einem **Mixer**, der
Information *zwischen* Positionen austauscht, und einem **FFN**, das jede
Position einzeln verarbeitet. Klassisch ist der Mixer eine Attention. Dieser
Baukasten macht ihn austauschbar – und zwar **pro Schicht**. Damit sind
Hybride wie Jamba (Attention + Mamba) oder Experimente wie "jede vierte
Schicht Attention, sonst Faltung" mit einer einzigen Zeile möglich::

    "layer_overrides": {"*": {"mixer": "ssm"}, "3::4": {"mixer": "attention"}}

Mixer
-----
==============  ==========================================================
``attention``   Multi-Head-Attention (alle Optionen von ``AttentionConfig``)
``ssm``         Selektives Zustandsraummodell (Mamba-artig), lineare Kosten
``rnn``         GRU/LSTM als Mixer (kausal oder bidirektional)
``conv``        Tiefenfaltung über die Zeit (lokale Muster, kausal möglich)
``fourier``     FNet: Fourier-Transformation statt Attention (nicht kausal)
``none``        kein Mixer (reines FFN-Netz, Ablationen)
==============  ==========================================================

Registry ``sequence_mixer``
---------------------------
Fabrik-Signatur: ``factory(args, d_model) -> nn.Module`` mit
``forward(x, key_padding_mask=None, positional=None, positions=None, causal=False)``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, Level, Param, T, create_registry
from .attention import AttentionConfig, MultiHeadAttention

MIXERS = create_registry(
    "sequence_mixer",
    "Sequenz-Mixer",
    T(
        "Teil eines Blocks, der Informationen zwischen Sequenzpositionen austauscht.",
        "Bestimmt Kosten (quadratisch vs. linear), Reichweite und induktive Annahmen.",
        "Transformer, State-Space-Modelle, Hybride.",
    ),
    context_doc="factory(args, d_model) -> forward(x, key_padding_mask, positional, positions, causal)",
)


def _mask_padding(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    if key_padding_mask is None:
        return x
    return x * key_padding_mask.to(x.dtype).unsqueeze(-1)


class AttentionMixer(nn.Module):
    def __init__(self, d_model: int, cfg: AttentionConfig):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, cfg)

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False):
        return self.attn(x, key_padding_mask=key_padding_mask, positional=positional, positions=positions, causal=causal)


# ===========================================================================
# Selektives State-Space-Modell (Mamba-artig)
# ===========================================================================

class SSMConfig(BaseConfig):
    __title__ = "Selektives SSM"
    __tooltip__ = T(
        "Selektives Zustandsraummodell nach dem Vorbild von Mamba (Gu & Dao 2023).",
        "Verarbeitet Sequenzen mit linearen Kosten und konstantem Speicher pro Schritt.",
        "Lange Sequenzen (Audio, Genomik, lange Texte), schnelle Generierung.",
    )
    d_state = Param(16, min=1, tooltip=T(
        "Größe des inneren Zustands pro Kanal (N).",
        "Mehr Zustand = mehr Gedächtniskapazität, mehr Rechenaufwand.",
        "Mamba: 16.",
    ))
    expand = Param(2.0, min=0.25, tooltip=T(
        "Aufweitung der inneren Breite (E = expand · d_model).", "Kapazität des Blocks.", "Mamba: 2."))
    d_conv = Param(4, min=1, tooltip=T(
        "Kernelgröße der kurzen kausalen Faltung vor dem Scan.",
        "Erfasst lokale Muster, bevor der Zustand aktualisiert wird.", "Mamba: 4."), level=Level.ADVANCED)
    dt_rank = Param("auto", type=(int, str), tooltip=T(
        "Rang der Δ-Projektion ('auto' = ceil(d_model/16)).",
        "Steuert, wie fein die Schrittweite Δ von der Eingabe abhängen kann.",
        "Selten zu ändern.",
    ), level=Level.EXPERT)
    dt_min = Param(0.001, min=1e-6, tooltip=T(
        "Kleinste initiale Schrittweite Δ.", "Kleine Δ = langes Gedächtnis.", "Mamba: 0.001."), level=Level.EXPERT)
    dt_max = Param(0.1, min=1e-6, tooltip=T(
        "Größte initiale Schrittweite Δ.", "Große Δ = Fokus auf aktuelle Eingabe.", "Mamba: 0.1."), level=Level.EXPERT)
    a_init = Param("s4d_real", choices={
        "s4d_real": "A_n = -(n+1) (S4D-Real, Mamba-Standard)",
        "s4d_lin": "A_n = -1/2 (alle gleich)",
        "random": "zufällig in [-N, -1]",
    }, tooltip=T("Initialisierung der Zustandsmatrix A.", "Bestimmt Start-Gedächtnis-Zeitskalen.", "Forschung."),
        level=Level.EXPERT)
    bidirectional = Param(False, tooltip=T(
        "Zusätzlicher Rückwärts-Scan (wird bei kausalen Modellen ignoriert).",
        "Kontext von beiden Seiten für Encoder-Aufgaben.", "Klassifikation, Tagging."), level=Level.ADVANCED)
    conv_bias = Param(True, tooltip=T("Bias der kurzen Faltung.", "Verschiebung.", "Standard."), level=Level.EXPERT)
    bias = Param(False, tooltip=T("Bias der Ein-/Ausgangsprojektion.", "Verschiebung.", "Mamba: aus."), level=Level.EXPERT)


class SelectiveSSM(nn.Module):
    """Vereinfachter Mamba-Block (sequenzieller Scan in reinem PyTorch).

    Ablauf: in_proj → kausale Tiefenfaltung → SiLU → (Δ, B, C) aus der Eingabe →
    diskretisierter Scan  h_t = exp(Δ·A)·h_{t-1} + Δ·B_t·x_t,  y_t = C_t·h_t + D·x_t →
    Gating mit SiLU(z) → out_proj.

    Hinweis: Der Scan ist eine Python-Schleife über die Zeit – korrekt und
    gut lesbar, aber langsamer als die CUDA-Kernel des Originals.
    """

    def __init__(self, d_model: int, cfg: SSMConfig):
        super().__init__()
        self.cfg = cfg
        e = int(cfg.expand * d_model)
        n = cfg.d_state
        r = math.ceil(d_model / 16) if cfg.dt_rank == "auto" else int(cfg.dt_rank)
        self.e, self.n, self.r = e, n, r
        self.in_proj = nn.Linear(d_model, 2 * e, bias=cfg.bias)
        self.conv = nn.Conv1d(e, e, cfg.d_conv, groups=e, padding=cfg.d_conv - 1, bias=cfg.conv_bias)
        self.x_proj = nn.Linear(e, r + 2 * n, bias=False)
        self.dt_proj = nn.Linear(r, e, bias=True)
        # Δ-Bias so initialisieren, dass softplus(bias) in [dt_min, dt_max] log-gleichverteilt ist.
        dt = torch.exp(torch.rand(e) * (math.log(cfg.dt_max) - math.log(cfg.dt_min)) + math.log(cfg.dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        if cfg.a_init == "s4d_real":
            a = torch.arange(1, n + 1).float().repeat(e, 1)
        elif cfg.a_init == "s4d_lin":
            a = torch.full((e, n), 0.5)
        else:
            a = torch.rand(e, n) * (n - 1) + 1
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, d_model, bias=cfg.bias)

    def _scan(self, u: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.A_log.float())                      # [E, N]
        dA = torch.exp(delta.unsqueeze(-1) * A)                 # [B, T, E, N]
        dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        h = torch.zeros(u.shape[0], self.e, self.n, device=u.device, dtype=dA.dtype)
        ys = []
        for t in range(u.shape[1]):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1) + u * self.D

    def _direction(self, x: torch.Tensor) -> torch.Tensor:
        T_ = x.shape[1]
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :T_].transpose(1, 2)
        u = F.silu(u)
        dbc = self.x_proj(u)
        dt, B, C = dbc.split([self.r, self.n, self.n], dim=-1)
        delta = F.softplus(self.dt_proj(dt)).float()
        y = self._scan(u.float(), delta, B.float(), C.float()).to(x.dtype)
        return self.out_proj(y * F.silu(z))

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False):
        x = _mask_padding(x, key_padding_mask)
        out = self._direction(x)
        if self.cfg.bidirectional and not causal:
            out = out + self._direction(x.flip(1)).flip(1)
        return out


# ===========================================================================
# Weitere Mixer
# ===========================================================================

class RNNMixerConfig(BaseConfig):
    cell = Param("gru", choices=["gru", "lstm", "rnn"], tooltip=T(
        "Rekurrente Zelle.", "GRU: schlank; LSTM: mehr Gedächtnissteuerung.", "Hybride, Streaming."))
    bidirectional = Param(True, tooltip=T(
        "Beide Richtungen (nur ohne Kausalität).", "Kontext von beiden Seiten.", "Encoder."))


class RNNMixer(nn.Module):
    def __init__(self, d_model: int, cfg: RNNMixerConfig):
        super().__init__()
        cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[cfg.cell]
        self.fwd = cls(d_model, d_model, batch_first=True)
        self.bwd = cls(d_model, d_model, batch_first=True) if cfg.bidirectional else None

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False):
        x = _mask_padding(x, key_padding_mask)
        out, _ = self.fwd(x)
        if self.bwd is not None and not causal:
            back, _ = self.bwd(x.flip(1))
            out = out + back.flip(1)
        return out


class ConvMixerConfig(BaseConfig):
    kernel_size = Param(7, min=1, tooltip=T(
        "Fenstergröße der Zeitfaltung.", "Reichweite pro Schicht.", "Lokale Muster: 3–15."))
    activation = Param(True, tooltip=T("SiLU nach der Faltung.", "Nichtlinearität im Mixer.", "Standard."))


class ConvMixer(nn.Module):
    """Tiefenfaltung über die Zeitachse (kausal: nur Vergangenheit)."""

    def __init__(self, d_model: int, cfg: ConvMixerConfig):
        super().__init__()
        self.k = cfg.kernel_size
        self.conv = nn.Conv1d(d_model, d_model, self.k, groups=d_model)
        self.act = nn.SiLU() if cfg.activation else nn.Identity()

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False):
        h = _mask_padding(x, key_padding_mask).transpose(1, 2)
        pad = (self.k - 1, 0) if causal else ((self.k - 1) // 2, self.k // 2)
        return self.act(self.conv(F.pad(h, pad))).transpose(1, 2)


class FourierMixer(nn.Module):
    """FNet: Realteil der 2D-FFT über (Sequenz, Features) – parameterfrei, nicht kausal."""

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False):
        if causal:
            raise ValueError("Der Fourier-Mixer ist nicht kausal und kann nicht in Decodern verwendet werden.")
        return torch.fft.fft2(_mask_padding(x, key_padding_mask).float()).real.to(x.dtype)


class NoMixer(nn.Module):
    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False):
        return torch.zeros_like(x)


R = MIXERS.register
R("attention", lambda a, d_model: AttentionMixer(d_model, a), config=AttentionConfig, aliases=("self_attention",),
  tooltip=T("Multi-Head-Self-Attention.", "Globale, inhaltsabhängige Interaktion aller Positionen.", "Standard."))
R("ssm", lambda a, d_model: SelectiveSSM(d_model, a), config=SSMConfig, aliases=("mamba",), level=Level.ADVANCED,
  tooltip=T("Selektives Zustandsraummodell (Mamba-artig).", "Lineare Kosten in der Sequenzlänge.",
            "Lange Sequenzen, Hybride (Jamba)."))
R("rnn", lambda a, d_model: RNNMixer(d_model, a), config=RNNMixerConfig, level=Level.ADVANCED, tooltip=T(
    "Rekurrente Schicht als Mixer.", "Starke Reihenfolge-Annahme, Streaming-fähig.", "Hybride, kleine Daten."))
R("conv", lambda a, d_model: ConvMixer(d_model, a), config=ConvMixerConfig, level=Level.ADVANCED, tooltip=T(
    "Tiefenfaltung über die Zeit.", "Billig, lokal, kausal möglich.", "Audio, lokale Muster, Hybride."))
R("fourier", lambda a, d_model: FourierMixer(), aliases=("fnet",), level=Level.EXPERIMENTAL, tooltip=T(
    "Fourier-Transformation als Mixer (FNet).", "Parameterfrei, sehr schnell; etwas schwächer als Attention.",
    "Schnelle Encoder, Experimente."))
R("none", lambda a, d_model: NoMixer(), aliases=("keiner",), level=Level.EXPERIMENTAL, tooltip=T(
    "Kein Mixer – Positionen bleiben getrennt.", "Ablationen; reine positionsweise Netze.", "Forschung."))
