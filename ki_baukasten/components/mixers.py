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

Cache für die Generierung
-------------------------
Mixer mit ``supports_cache = True`` nehmen zusätzlich ``cache=`` (ein
``LayerCache``, siehe ``components.cache``) entgegen und verarbeiten dann nur
die *neuen* Positionen. Alle eingebauten kausalen Mixer unterstützen das
(Attention: KV-Cache, SSM: Faltungsfenster + Zustand, RNN: versteckter
Zustand, Faltung: letzte Eingaben).
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, Level, Param, T, create_registry
from .attention import AttentionConfig, MultiHeadAttention
from .scan import SCAN_MODES, linear_scan, selective_scan  # noqa: F401

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
    supports_cache = True

    def __init__(self, d_model: int, cfg: AttentionConfig):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, cfg)

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False, cache=None):
        return self.attn(x, key_padding_mask=key_padding_mask, positional=positional, positions=positions, causal=causal,
                         cache=cache)


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

    # --- Geschwindigkeit & Speicher des Scans -------------------------------
    scan_mode = Param("auto", choices=dict(SCAN_MODES, mamba_ssm="offizieller CUDA-Kernel (Paket mamba-ssm), "
                                                                 "sonst automatisch auto"), tooltip=T(
        "Algorithmus für den Zustands-Scan h_t = exp(Δ·A)·h_{t-1} + Δ·B·x.",
        "Der parallele/chunked Scan braucht nur log₂(T) Runden statt T Schleifenschritte – "
        "Training oft 10–40× schneller als die Python-Schleife.",
        "auto ist fast immer richtig (CPU: fusionierte Schleife, GPU: chunked); mamba_ssm auf NVIDIA-GPUs mit "
        "installiertem Paket.",
    ), group="Speicher & Geschwindigkeit")
    scan_chunk_size = Param(32, min=1, tooltip=T(
        "Blocklänge des chunked-Scans.",
        "Größer = weniger sequenzielle Überträge, aber mehr Zwischenspeicher pro Runde.",
        "CPU: 16–32; GPU: 64–256.",
    ), level=Level.ADVANCED, group="Speicher & Geschwindigkeit")
    scan_backward = Param("fused", choices={
        "fused": "nur Eingaben speichern, große Tensoren im Backward neu berechnen (minimaler Speicher)",
        "scan": "Zerfall und Zustände speichern, Backward per Rückwärts-Scan (etwas schneller, mehr Speicher)",
        "autograd": "alles von Autograd aufzeichnen lassen (langsam, nur zum Vergleichen)",
    }, tooltip=T(
        "Wie der Gradient des Scans berechnet wird.",
        "'fused' senkt den Aktivierungsspeicher des Scans etwa um den Faktor d_state – wie der offizielle "
        "Mamba-Kernel, aber in reinem PyTorch (CPU, GPU, MPS).",
        "Lange Sequenzen / große Batches: fused. Kurze Sequenzen mit viel Speicher: scan.",
    ), level=Level.EXPERT, group="Speicher & Geschwindigkeit")


def _mamba_kernel():
    """Optionaler offizieller Mamba-CUDA-Kernel (``pip install mamba-ssm``) oder ``None``."""
    try:  # pragma: no cover - nur mit installiertem Paket + CUDA
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore
        return selective_scan_fn
    except Exception:  # noqa: BLE001
        return None


class SelectiveSSM(nn.Module):
    """Vereinfachter Mamba-Block in reinem PyTorch (optional mit offiziellem CUDA-Kernel).

    Ablauf: in_proj → kausale Tiefenfaltung → SiLU → (Δ, B, C) aus der Eingabe →
    diskretisierter Scan  h_t = exp(Δ·A)·h_{t-1} + Δ·B_t·x_t,  y_t = C_t·h_t + D·x_t →
    Gating mit SiLU(z) → out_proj.

    Der Scan läuft standardmäßig **parallel in Blöcken** (``components.scan``)
    mit speichersparendem Backward. Für die Generierung hält ein ``LayerCache``
    das Faltungsfenster und den Zustand h – jedes neue Token kostet dann O(1).
    """

    supports_cache = True

    def __init__(self, d_model: int, cfg: SSMConfig):
        super().__init__()
        self.cfg = cfg
        e = int(cfg.expand * d_model)
        n = cfg.d_state
        r = math.ceil(d_model / 16) if cfg.dt_rank == "auto" else int(cfg.dt_rank)
        self.e, self.n, self.r = e, n, r
        self.in_proj = nn.Linear(d_model, 2 * e, bias=cfg.bias)
        self.conv = nn.Conv1d(e, e, cfg.d_conv, groups=e, padding=0, bias=cfg.conv_bias)
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
        self._warned_kernel = False

    # ------------------------------------------------------------ Scan
    def _scan(self, u: torch.Tensor, delta: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
              h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Liefert ``(y [B,T,E], letzter Zustand [B,E,N])``."""
        c = self.cfg
        A = -torch.exp(self.A_log.float())                      # [E, N]
        mode = c.scan_mode
        if mode == "mamba_ssm":
            out = self._scan_cuda(u, delta, A, B, C, h0)
            if out is not None:
                return out
            mode = "auto"
        return selective_scan(u, delta, A, B, C, self.D.float(), h0, mode=mode, chunk_size=c.scan_chunk_size,
                              backward=c.scan_backward)

    def _scan_cuda(self, u, delta, A, B, C, h0):
        kernel = _mamba_kernel()
        if kernel is None or not u.is_cuda or h0 is not None:
            if not self._warned_kernel and (kernel is None or not u.is_cuda):
                warnings.warn("scan_mode='mamba_ssm': Paket mamba-ssm bzw. CUDA nicht verfügbar – nutze 'auto'.")
                self._warned_kernel = True
            return None
        y, last = kernel(u.transpose(1, 2).contiguous(), delta.transpose(1, 2).contiguous(), A,  # pragma: no cover
                         B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(), self.D.float(),
                         return_last_state=True)
        return y.transpose(1, 2), last  # pragma: no cover

    # ------------------------------------------------------------ Vorwärts
    def _direction(self, x: torch.Tensor, cache=None) -> torch.Tensor:
        Bsz, T_, _ = x.shape
        k = self.cfg.d_conv
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        seq = u.transpose(1, 2)                                 # [B, E, T]
        prev = cache.get("conv") if cache is not None else None
        if prev is None:
            prev = seq.new_zeros(Bsz, self.e, k - 1)            # kausal: links mit Nullen auffüllen
        full = torch.cat([prev.to(seq.dtype), seq], -1)
        if cache is not None:
            cache["conv"] = full[..., full.shape[-1] - (k - 1):].detach()
        u = F.silu(F.conv1d(full, self.conv.weight, self.conv.bias, groups=self.e).transpose(1, 2))
        dbc = self.x_proj(u)
        dt, B, C = dbc.split([self.r, self.n, self.n], dim=-1)
        delta = F.softplus(self.dt_proj(dt)).float()
        h0 = cache.get("h") if cache is not None else None
        y, last = self._scan(u.float(), delta, B.float(), C.float(), h0)
        if cache is not None:
            cache["h"] = last.detach()
        return self.out_proj(y.to(x.dtype) * F.silu(z))

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False, cache=None):
        x = _mask_padding(x, key_padding_mask)
        out = self._direction(x, cache if causal else None)
        if self.cfg.bidirectional and not causal:
            out = out + self._direction(x.flip(1)).flip(1)
        return out


# ===========================================================================
# Weitere Mixer
# ===========================================================================

class MinGRU(nn.Module):
    """minGRU (Feng et al. 2024, "Were RNNs All We Needed?").

    Gate und Kandidat hängen nur von der Eingabe ab (nicht vom Zustand):
    ``h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t``. Das ist eine lineare Rekurrenz
    und läuft deshalb über den **parallelen Scan** (``components.scan``) –
    log₂(T) Runden statt T Schleifenschritte.
    """

    def __init__(self, input_size: int, hidden_size: int, scan_mode: str = "auto"):
        super().__init__()
        self.gate = nn.Linear(input_size, hidden_size)
        self.cand = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.scan_mode = scan_mode

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self.gate(x))
        h = linear_scan(1 - z, z * self.cand(x), h0, mode=self.scan_mode)
        return h, h[:, -1]



class RNNMixerConfig(BaseConfig):
    cell = Param("gru", choices={
        "gru": "GRU (cuDNN, sequenziell)", "lstm": "LSTM (cuDNN, sequenziell)", "rnn": "Elman-RNN",
        "mingru": "minGRU – paralleler Scan, sehr schnell im Training",
    }, tooltip=T(
        "Rekurrente Zelle.", "GRU: schlank; LSTM: mehr Gedächtnissteuerung; minGRU: parallel trainierbar.",
        "Hybride, Streaming."))
    bidirectional = Param(True, tooltip=T(
        "Beide Richtungen (nur ohne Kausalität).", "Kontext von beiden Seiten.", "Encoder."))


class RNNMixer(nn.Module):
    supports_cache = True

    def __init__(self, d_model: int, cfg: RNNMixerConfig):
        super().__init__()
        if cfg.cell == "mingru":
            self.fwd = MinGRU(d_model, d_model)
            self.bwd = MinGRU(d_model, d_model) if cfg.bidirectional else None
        else:
            cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[cfg.cell]
            self.fwd = cls(d_model, d_model, batch_first=True)
            self.bwd = cls(d_model, d_model, batch_first=True) if cfg.bidirectional else None

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False, cache=None):
        x = _mask_padding(x, key_padding_mask)
        if cache is not None and causal:
            out, cache["h"] = self.fwd(x, cache.get("h"))
            return out
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

    supports_cache = True

    def __init__(self, d_model: int, cfg: ConvMixerConfig):
        super().__init__()
        self.k = cfg.kernel_size
        self.conv = nn.Conv1d(d_model, d_model, self.k, groups=d_model)
        self.act = nn.SiLU() if cfg.activation else nn.Identity()

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False, cache=None):
        h = _mask_padding(x, key_padding_mask).transpose(1, 2)
        if causal:
            prev = cache.get("x") if cache is not None else None
            if prev is None:
                prev = h.new_zeros(h.shape[0], h.shape[1], self.k - 1)
            full = torch.cat([prev.to(h.dtype), h], -1)
            if cache is not None:
                cache["x"] = full[..., full.shape[-1] - (self.k - 1):].detach()
            return self.act(self.conv(full)).transpose(1, 2)
        pad = ((self.k - 1) // 2, self.k // 2)
        return self.act(self.conv(F.pad(h, pad))).transpose(1, 2)


class FourierMixer(nn.Module):
    """FNet: Realteil der 2D-FFT über (Sequenz, Features) – parameterfrei, nicht kausal."""

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False):
        if causal:
            raise ValueError("Der Fourier-Mixer ist nicht kausal und kann nicht in Decodern verwendet werden.")
        return torch.fft.fft2(_mask_padding(x, key_padding_mask).float()).real.to(x.dtype)


class NoMixer(nn.Module):
    supports_cache = True

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False, cache=None):
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
