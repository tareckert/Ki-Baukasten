"""
ki_baukasten.components.attention
=================================

Multi-Head-Attention mit (fast) jedem denkbaren Stellrad.

Aufbau einer Attention-Schicht
------------------------------
::

    x ──► Q-, K-, V-Projektion ──► (QK-Norm) ──► (RoPE) ──► Kernel ──► (Gate) ──► Ausgangsprojektion
                                                         ▲
                       Maske (kausal / Fenster / Padding) + Positions-Bias (ALiBi, T5)

Stellräder (``AttentionConfig``)
--------------------------------
* **Köpfe**: ``num_heads``, ``num_kv_heads`` (GQA/MQA), ``head_dim``
* **Kernel** (Registry ``attention_kernel``): wie aus Q und K Gewichte werden –
  ``softmax`` (Standard), ``sigmoid``, ``relu``, ``cosine``, ``linear`` ...
* **Masken**: ``causal``, ``window_size`` (Sliding Window)
* **Stabilität**: ``qk_norm``, ``scale``, Kernel-Optionen wie ``logit_softcap``
* **Regularisierung**: ``dropout`` (auf Gewichte), ``head_dropout`` (ganze Köpfe)
* **Exotik**: ``output_gate`` (gated attention), Bias-Schalter

Registry ``attention_kernel``
-----------------------------
Fabrik-Signatur: ``factory(args) -> nn.Module`` mit
``forward(q, k, v, masks: MaskInfo, bias, scale, dropout_p) -> out``,
alle Tensoren im Format ``[B, H, T, hd]``. Eigene Kernel (z. B. neue
Forschungsideen) lassen sich registrieren und sofort in jedem Transformer
verwenden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, ComponentParam, Level, Param, T, create_registry
from .normalization import build_norm

KERNELS = create_registry(
    "attention_kernel",
    "Attention-Kernel",
    T(
        "Rechenvorschrift, die aus Query/Key-Ähnlichkeiten Gewichte über die Values macht.",
        "Bestimmt Ausdrucksstärke, Kosten (quadratisch vs. linear) und Stabilität.",
        "Transformer jeder Art.",
    ),
    context_doc="factory(args) -> Modul mit forward(q, k, v, masks, bias, scale, dropout_p)",
)


# ===========================================================================
# Masken
# ===========================================================================

@dataclass
class MaskInfo:
    """Alle Masken-Informationen einer Attention-Berechnung.

    Die kombinierte boolesche Maske wird erst bei Bedarf gebaut (``combined``),
    damit schnelle Pfade (z. B. ``is_causal`` von PyTorch-SDPA) sie gar nicht
    erst materialisieren müssen.

    Attribute:
        key_padding_mask: ``[B, S]``, ``True`` = gültiges Element, ``False`` = Padding.
    """

    q_len: int
    k_len: int
    causal: bool
    window: int | None
    key_padding_mask: torch.Tensor | None
    device: torch.device
    _cache: torch.Tensor | None = field(default=None, repr=False)

    @property
    def is_empty(self) -> bool:
        return not self.causal and self.window is None and self.key_padding_mask is None

    @property
    def is_simple_causal(self) -> bool:
        return self.causal and self.window is None and self.key_padding_mask is None and self.q_len == self.k_len

    @property
    def is_decode_step(self) -> bool:
        """Generierung mit Cache: ein Query hinter allen Keys – kausal erlaubt es alle (ohne Fenster/Padding)."""
        return self.causal and self.window is None and self.key_padding_mask is None and self.q_len == 1

    def combined(self) -> torch.Tensor | None:
        """Boolesche Maske ``[B|1, 1, T, S]`` (``True`` = erlaubt) oder ``None``."""
        if self.is_empty:
            return None
        if self._cache is not None:
            return self._cache
        i = torch.arange(self.q_len, device=self.device).unsqueeze(1) + (self.k_len - self.q_len)
        j = torch.arange(self.k_len, device=self.device).unsqueeze(0)
        allowed = torch.ones(self.q_len, self.k_len, dtype=torch.bool, device=self.device)
        if self.causal:
            allowed &= j <= i
        if self.window is not None:
            allowed &= ((i - j) < self.window) if self.causal else ((i - j).abs() < self.window)
        mask = allowed.view(1, 1, self.q_len, self.k_len)
        if self.key_padding_mask is not None:
            mask = mask & self.key_padding_mask.bool().view(-1, 1, 1, self.k_len)
        self._cache = mask
        return mask


_GQA_OK: bool | None = None


def sdpa_gqa_supported() -> bool:
    """Kann ``scaled_dot_product_attention`` Gruppen-KV direkt (``enable_gqa``, PyTorch ≥ 2.5)?"""
    global _GQA_OK
    if _GQA_OK is None:
        try:
            t = torch.zeros(1, 2, 1, 4)
            F.scaled_dot_product_attention(t, t[:, :1], t[:, :1], enable_gqa=True)
            _GQA_OK = True
        except (TypeError, RuntimeError):
            _GQA_OK = False
    return _GQA_OK


def _neg(dtype: torch.dtype) -> float:
    """Sehr negative, aber endliche Zahl (verhindert NaN bei komplett maskierten Zeilen)."""
    return torch.finfo(dtype).min


# ===========================================================================
# Kernel
# ===========================================================================

class SoftmaxKernelConfig(BaseConfig):
    backend = Param("auto", choices={
        "auto": "PyTorch-SDPA (Flash/Memory-Efficient) wenn möglich, sonst manuell",
        "sdpa": "immer PyTorch scaled_dot_product_attention",
        "manual": "explizite Berechnung (langsamer, aber alle Optionen + Gewichte speicherbar)",
    }, tooltip=T(
        "Implementierung der Softmax-Attention.",
        "SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.",
        "Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging.",
    ), group="Speicher & Geschwindigkeit")
    temperature = Param(1.0, min=1e-6, tooltip=T(
        "Softmax-Temperatur (Logits werden durch sie geteilt).",
        "< 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.",
        "Experimente, Kalibrierung.",
    ), level=Level.EXPERT)
    logit_softcap = Param(None, type=float, min=1e-6, tooltip=T(
        "Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).",
        "Verhindert extreme Logits und damit Instabilität in großen Modellen.",
        "Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).",
        "Erzwingt den manuellen (langsameren) Pfad.",
    ), level=Level.EXPERT)
    softmax_fp32 = Param(True, tooltip=T(
        "Softmax im manuellen Pfad in 32-Bit rechnen.",
        "Numerisch stabil bei fp16/bf16.",
        "Mixed Precision.",
    ), level=Level.EXPERT)
    store_weights = Param(False, tooltip=T(
        "Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).",
        "Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.",
        "Interpretierbarkeit, Debugging.",
    ), level=Level.ADVANCED)


class SoftmaxKernel(nn.Module):
    """Klassische Scaled-Dot-Product-Attention."""

    def __init__(self, cfg: SoftmaxKernelConfig):
        super().__init__()
        self.cfg = cfg
        self.last_weights: torch.Tensor | None = None

    @property
    def supports_gqa(self) -> bool:
        """Weniger KV- als Query-Köpfe direkt verarbeiten (ohne KV-Kopien) – im SDPA-Pfad."""
        c = self.cfg
        return c.backend != "manual" and c.logit_softcap is None and not c.store_weights and sdpa_gqa_supported()

    def forward(self, q, k, v, masks: MaskInfo, bias, scale, dropout_p):
        c = self.cfg
        scale = scale / c.temperature
        use_sdpa = c.backend != "manual" and c.logit_softcap is None and not c.store_weights
        if use_sdpa:
            gqa = {"enable_gqa": True} if k.shape[1] != q.shape[1] else {}
            if bias is None and masks.is_simple_causal:
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True, scale=scale,
                                                      **gqa)
            if bias is None and masks.is_decode_step:      # ein neues Token sieht alle bisherigen → keine Maske
                return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale, **gqa)
            mask = masks.combined()
            attn_mask = None
            if bias is not None:
                attn_mask = bias.to(q.dtype).unsqueeze(0)
            if mask is not None:
                base = attn_mask if attn_mask is not None else torch.zeros((), dtype=q.dtype, device=q.device)
                neg = torch.tensor(_neg(q.dtype), dtype=q.dtype, device=q.device)
                attn_mask = torch.where(mask, base, neg)  # broadcastet auf [B, H, T, S]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale,
                                                  **gqa)

        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        if c.logit_softcap is not None:
            logits = c.logit_softcap * torch.tanh(logits / c.logit_softcap)
        mask = masks.combined()
        if mask is not None:
            logits = logits.masked_fill(~mask, _neg(logits.dtype))
        weights = logits.float().softmax(-1).to(q.dtype) if c.softmax_fp32 else logits.softmax(-1)
        if c.store_weights:
            self.last_weights = weights.detach()
        weights = F.dropout(weights, dropout_p, self.training)
        return torch.matmul(weights, v)


class SigmoidKernelConfig(BaseConfig):
    bias = Param(None, type=float, tooltip=T(
        "Konstanter Bias b in sigmoid(qk + b). None = -log(Sequenzlänge).",
        "Hält die Summe der Gewichte anfangs nahe 1 – wichtig für stabiles Training.",
        "Sigmoid-Attention (Ramapuram et al. 2024).",
    ), level=Level.EXPERT)


class SigmoidKernel(nn.Module):
    """Sigmoid statt Softmax: Gewichte unabhängig voneinander (keine Normierung)."""

    def __init__(self, cfg: SigmoidKernelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, q, k, v, masks, bias, scale, dropout_p):
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        b = self.cfg.bias if self.cfg.bias is not None else -math.log(max(1, masks.k_len))
        weights = torch.sigmoid(logits + b)
        mask = masks.combined()
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
        weights = F.dropout(weights, dropout_p, self.training)
        return torch.matmul(weights, v)


class ReLUKernel(nn.Module):
    """ReLU-Attention, normiert durch die Anzahl sichtbarer Keys (experimentell)."""

    def forward(self, q, k, v, masks, bias, scale, dropout_p):
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        weights = F.relu(logits)
        mask = masks.combined()
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
            count = mask.sum(-1, keepdim=True).clamp_min(1).to(weights.dtype)
        else:
            count = torch.tensor(float(masks.k_len), dtype=weights.dtype, device=weights.device)
        weights = F.dropout(weights / count, dropout_p, self.training)
        return torch.matmul(weights, v)


class CosineKernelConfig(BaseConfig):
    init_scale = Param(10.0, min=1e-3, tooltip=T(
        "Startwert der lernbaren Temperatur τ (Logits = τ · cos(q, k)).",
        "Kosinus-Ähnlichkeit liegt in [-1, 1]; τ macht die Softmax ausreichend scharf.",
        "Swin-Transformer-V2-artige Stabilisierung großer Modelle.",
    ))
    learnable = Param(True, tooltip=T("τ lernbar machen.", "Netz wählt die Schärfe selbst.", "Standard."))


class CosineKernel(nn.Module):
    """Kosinus-Attention mit lernbarer Temperatur."""

    def __init__(self, cfg: CosineKernelConfig):
        super().__init__()
        t = torch.tensor(math.log(cfg.init_scale))
        self.log_tau = nn.Parameter(t) if cfg.learnable else nn.Parameter(t, requires_grad=False)

    def forward(self, q, k, v, masks, bias, scale, dropout_p):
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        logits = torch.matmul(qn, kn.transpose(-2, -1)) * self.log_tau.exp().clamp_max(100.0)
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        mask = masks.combined()
        if mask is not None:
            logits = logits.masked_fill(~mask, _neg(logits.dtype))
        weights = F.dropout(logits.softmax(-1), dropout_p, self.training)
        return torch.matmul(weights, v)


class LinearKernelConfig(BaseConfig):
    feature_map = Param("elu", choices={
        "elu": "φ(x) = elu(x) + 1 (Katharopoulos et al. 2020)",
        "relu": "φ(x) = relu(x)",
        "exp": "φ(x) = exp(x - max) – nähert Softmax grob an",
    }, tooltip=T(
        "Merkmalsabbildung φ für die Kernel-Näherung.",
        "Bestimmt, wie gut die Softmax-Attention angenähert wird.",
        "Lange Sequenzen mit linearen Kosten.",
    ))
    eps = Param(1e-6, min=0.0, level=Level.EXPERT, tooltip=T(
        "Stabilisator im Nenner.", "Verhindert Division durch 0.", "Selten zu ändern."))


class LinearKernel(nn.Module):
    """Lineare Attention: O(T·d²) statt O(T²·d).

    Unterstützt kausale Masken und Padding, aber **keine** Sliding-Window- oder
    Positions-Bias-Terme (die Gewichtsmatrix wird nie gebildet).
    """

    def __init__(self, cfg: LinearKernelConfig):
        super().__init__()
        self.cfg = cfg

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.cfg.feature_map
        if fm == "elu":
            return F.elu(x) + 1.0
        if fm == "relu":
            return F.relu(x)
        return torch.exp(x - x.amax(-1, keepdim=True))

    def forward(self, q, k, v, masks, bias, scale, dropout_p):
        qf, kf = self._phi(q * math.sqrt(scale)), self._phi(k * math.sqrt(scale))
        if masks.key_padding_mask is not None:
            kf = kf * masks.key_padding_mask.to(kf.dtype).view(kf.shape[0], 1, -1, 1)
        eps = self.cfg.eps
        if masks.causal:
            # Bei Cache-Nutzung gibt es weniger Queries als Keys: Queries = letzte q_len Positionen.
            tq = qf.shape[2]
            kv = torch.einsum("bhsd,bhse->bhsde", kf, v).cumsum(2)[:, :, -tq:]
            num = torch.einsum("bhtd,bhtde->bhte", qf, kv)
            den = torch.einsum("bhtd,bhtd->bht", qf, kf.cumsum(2)[:, :, -tq:]) + eps
        else:
            kv = torch.einsum("bhsd,bhse->bhde", kf, v)
            num = torch.einsum("bhtd,bhde->bhte", qf, kv)
            den = torch.einsum("bhtd,bhd->bht", qf, kf.sum(2)) + eps
        return num / den.unsqueeze(-1)


R = KERNELS.register
R("softmax", lambda a: SoftmaxKernel(a), config=SoftmaxKernelConfig, aliases=("standard", "dot_product"), tooltip=T(
    "Standard-Softmax-Attention (scaled dot product).",
    "Bewährt, ausdrucksstark; Kosten quadratisch in der Sequenzlänge.",
    "Fast alle Transformer.",
))
R("sigmoid", lambda a: SigmoidKernel(a), config=SigmoidKernelConfig, level=Level.EXPERIMENTAL, tooltip=T(
    "Sigmoid-Attention: jedes Gewicht unabhängig in (0, 1).",
    "Keine Konkurrenz zwischen Keys – kann mehrere Stellen gleichzeitig voll beachten.",
    "Forschung; laut Apple-Paper 2024 gleichwertig zu Softmax mit passendem Bias.",
))
R("relu", lambda a: ReLUKernel(), level=Level.EXPERIMENTAL, tooltip=T(
    "ReLU-Attention, normiert durch die Sequenzlänge.",
    "Erzeugt spärliche Gewichte, keine Exponentialfunktion.",
    "Forschung, hardware-effiziente Varianten.",
))
R("cosine", lambda a: CosineKernel(a), config=CosineKernelConfig, level=Level.ADVANCED, tooltip=T(
    "Kosinus-Ähnlichkeit mit lernbarer Temperatur.",
    "Begrenzte Logits → stabil auch in sehr großen Modellen.",
    "Vision-Transformer (Swin V2), instabile Trainings.",
))
R("linear", lambda a: LinearKernel(a), config=LinearKernelConfig, level=Level.ADVANCED, tooltip=T(
    "Lineare Attention (Kernel-Näherung).",
    "Kosten wachsen nur linear mit der Sequenzlänge.",
    "Sehr lange Sequenzen, Streaming.",
    "Unterstützt kein Sliding Window und keinen Positions-Bias (RoPE geht).",
))


# ===========================================================================
# Multi-Head-Attention
# ===========================================================================

class AttentionConfig(BaseConfig):
    __title__ = "Attention"
    __tooltip__ = T(
        "Einstellungen einer Multi-Head-Attention-Schicht.",
        "Attention lässt jedes Element gezielt Informationen von anderen Elementen sammeln.",
        "Transformer-Blöcke (Self- und Cross-Attention).",
    )

    num_heads = Param(8, min=1, tooltip=T(
        "Anzahl paralleler Attention-Köpfe.",
        "Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).",
        "Alle Attention-Modelle.",
        "Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128.",
    ))
    num_kv_heads = Param(None, type=int, min=1, tooltip=T(
        "Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.",
        "Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.",
        "Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.",
        "num_heads muss durch num_kv_heads teilbar sein.",
    ), level=Level.ADVANCED)
    head_dim = Param(None, type=int, min=1, tooltip=T(
        "Dimension pro Kopf. None = d_model / num_heads.",
        "Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).",
        "Feintuning der Kapazität.",
    ), level=Level.ADVANCED)
    kernel = ComponentParam("attention_kernel", "softmax", tooltip=T(
        "Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).",
        "Bestimmt Kosten (quadratisch/linear) und Verhalten.",
        "Softmax ist Standard; andere für lange Sequenzen oder Forschung.",
    ))
    causal = Param(None, type=bool, tooltip=T(
        "Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.",
        "Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.",
        "None = automatisch durch die Architektur (Decoder kausal, Encoder nicht).",
    ), level=Level.ADVANCED)
    window_size = Param(None, type=int, min=1, tooltip=T(
        "Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.",
        "Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.",
        "Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).",
        "Per layer_overrides z. B. nur jede zweite Schicht lokal machen.",
    ), level=Level.ADVANCED)
    scale = Param(None, type=float, min=0.0, tooltip=T(
        "Skalierungsfaktor der Logits. None = 1/√head_dim.",
        "Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.",
        "µP-Skalierung nutzt 1/head_dim; Experimente.",
    ), level=Level.EXPERT)
    qk_norm = ComponentParam("normalization", None, tooltip=T(
        "Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).",
        "Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.",
        "Große Modelle (OLMo-2, Gemma 3, ViT-22B).",
    ), level=Level.ADVANCED)
    qkv_bias = Param(False, tooltip=T(
        "Bias in den Q/K/V-Projektionen.",
        "Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.",
        "Kompatibilität; Qwen nutzt nur QKV-Bias.",
    ), level=Level.ADVANCED)
    out_bias = Param(False, tooltip=T(
        "Bias in der Ausgangsprojektion.", "Siehe qkv_bias.", "Kompatibilität."), level=Level.ADVANCED)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout auf die Attention-Gewichte.",
        "Regularisiert, indem zufällig Verbindungen gekappt werden.",
        "Kleine Datensätze. Große LLMs: 0.",
    ), group="Regularisierung")
    head_dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Ganze Köpfe im Training zufällig abschalten (DropHead).",
        "Verhindert, dass sich das Modell auf wenige Köpfe verlässt.",
        "Regularisierung, Robustheit, Pruning-Vorbereitung.",
    ), level=Level.EXPERIMENTAL, group="Regularisierung")
    output_gate = Param(False, tooltip=T(
        "Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').",
        "Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.",
        "Neuere LLM-Forschung (Qwen3-Next).",
    ), level=Level.EXPERIMENTAL)
    kv_compression_dim = Param(None, type=int, min=1, tooltip=T(
        "MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.",
        "Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – "
        "oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.",
        "Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).",
        "Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe).",
    ), level=Level.EXPERT, group="Speicher & Geschwindigkeit")
    kv_latent_norm = ComponentParam("normalization", "rmsnorm", tooltip=T(
        "Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).",
        "Stabilisiert die Rekonstruktion von Keys/Values.", "DeepSeek: RMSNorm."),
        level=Level.EXPERT, group="Speicher & Geschwindigkeit")

    def check(self) -> list[str]:
        problems = []
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads:
            problems.append(f"num_heads ({self.num_heads}) muss durch num_kv_heads ({self.num_kv_heads}) teilbar sein.")
        if self.kernel.name == "linear" and self.window_size is not None:
            problems.append("Der lineare Kernel unterstützt kein window_size.")
        return problems


class MultiHeadAttention(nn.Module):
    """Konfigurierbare Multi-Head-Attention (Self- oder Cross-Attention).

    Args:
        d_model: Modellbreite (Eingabe und Ausgabe).
        cfg:     ``AttentionConfig``.
        kv_dim:  Breite der Key/Value-Quelle bei Cross-Attention (Standard: d_model).
    """

    def __init__(self, d_model: int, cfg: AttentionConfig, kv_dim: int | None = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads or cfg.num_heads
        if cfg.head_dim is None and d_model % cfg.num_heads:
            raise ValueError(f"d_model ({d_model}) ist nicht durch num_heads ({cfg.num_heads}) teilbar – head_dim setzen.")
        self.head_dim = cfg.head_dim or d_model // cfg.num_heads
        kv_dim = kv_dim or d_model
        inner, kv_inner = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, inner, bias=cfg.qkv_bias)
        r = cfg.kv_compression_dim
        if r is None:
            self.k_proj = nn.Linear(kv_dim, kv_inner, bias=cfg.qkv_bias)
            self.v_proj = nn.Linear(kv_dim, kv_inner, bias=cfg.qkv_bias)
        else:
            # MLA-lite: x → Latent c (r) → Keys/Values. Gecacht wird nur c.
            self.kv_down = nn.Linear(kv_dim, r, bias=False)
            self.kv_norm = build_norm(cfg.kv_latent_norm, r) if cfg.kv_latent_norm is not None else nn.Identity()
            self.k_up = nn.Linear(r, kv_inner, bias=cfg.qkv_bias)
            self.v_up = nn.Linear(r, kv_inner, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(inner, d_model, bias=cfg.out_bias)
        self.gate_proj = nn.Linear(d_model, inner, bias=True) if cfg.output_gate else None
        self.q_norm = build_norm(cfg.qk_norm, self.head_dim) if cfg.qk_norm is not None else None
        self.k_norm = build_norm(cfg.qk_norm, self.head_dim) if cfg.qk_norm is not None else None
        self.kernel = cfg.kernel.build()
        self.scale = cfg.scale if cfg.scale is not None else 1.0 / math.sqrt(self.head_dim)

    supports_cache = True

    def _keys_values(self, src: torch.Tensor, B: int, S: int) -> tuple[torch.Tensor, torch.Tensor]:
        Hkv, hd = self.num_kv_heads, self.head_dim
        if self.cfg.kv_compression_dim is None:
            k, v = self.k_proj(src), self.v_proj(src)
        else:
            c = self.kv_norm(self.kv_down(src))
            k, v = self.k_up(c), self.v_up(c)
        return k.view(B, S, Hkv, hd).transpose(1, 2), v.view(B, S, Hkv, hd).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positional=None,
        positions: torch.Tensor | None = None,
        causal: bool = False,
        cache=None,
    ) -> torch.Tensor:
        """
        Args:
            x:   Queries-Quelle ``[B, T, D]``.
            kv:  Key/Value-Quelle ``[B, S, D_kv]`` (Cross-Attention); ``None`` = Self-Attention.
            key_padding_mask: ``[B, S]``, ``True`` = gültig (mit Cache: nur für die neuen Positionen).
            positional: ``PositionalEncoding`` (nur bei Self-Attention angewendet).
            positions:  Positionsindizes ``[T]`` (mit Cache: absolute Positionen der neuen Tokens).
            causal: kausale Maske (wird durch ``cfg.causal`` überschrieben, falls gesetzt).
            cache:  ``LayerCache`` dieser Schicht (nur Self-Attention) – speichert Keys/Values bzw.
                    bei MLA-lite den Latent-Vektor; ``x`` enthält dann nur die **neuen** Positionen.
        """
        B, Tq, _ = x.shape
        src = x if kv is None else kv
        S = src.shape[1]
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        is_self = kv is None
        if cache is not None and not is_self:
            cache = None  # Cross-Attention: Quelle ist ohnehin vollständig vorhanden
        if positions is None:
            positions = torch.arange(Tq, device=x.device)

        q = self.q_proj(x).view(B, Tq, H, hd).transpose(1, 2)
        mla = self.cfg.kv_compression_dim is not None
        if mla and cache is not None:
            # Latent cachen, Keys/Values für alle (alten + neuen) Positionen daraus rekonstruieren.
            c = cache.append("latent", self.kv_norm(self.kv_down(src)), dim=1)
            old = cache.get("positions")
            k_positions = positions if old is None else torch.cat([old, positions])
            cache["positions"] = cache.truncate(k_positions, 0)  # exakte Positionen (nie quantisiert)
            S = c.shape[1]
            k = self.k_up(c).view(B, S, Hkv, hd).transpose(1, 2)
            v = self.v_up(c).view(B, S, Hkv, hd).transpose(1, 2)
        else:
            k, v = self._keys_values(src, B, S)
            k_positions = positions
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        if positional is not None and is_self:
            if mla and cache is not None:
                # Queries (neue Positionen) und rekonstruierte Keys (alle Positionen) getrennt drehen.
                q = positional.rotate_keys(q, positions)
                k = positional.rotate_keys(k, k_positions)
            else:
                q, k = positional.rotate(q, k, positions)
        if cache is not None and not mla:
            k = cache.append("k", k, dim=2)
            v = cache.append("v", v, dim=2)
            S = k.shape[2]
        if cache is not None:
            key_padding_mask = cache.append_mask(key_padding_mask, B, Tq, x.device)
        if Hkv != H and not getattr(self.kernel, "supports_gqa", False):
            # GQA: KV-Köpfe für Gruppen von Query-Köpfen wiederverwenden (SDPA kann das ohne Kopie)
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)

        use_causal = (self.cfg.causal if self.cfg.causal is not None else causal) and is_self
        masks = MaskInfo(Tq, S, use_causal, self.cfg.window_size if is_self else None, key_padding_mask, x.device)
        bias = positional.bias(Tq, S, H, x.device, use_causal) if (positional is not None and is_self) else None
        dropout_p = self.cfg.dropout if self.training else 0.0

        out = self.kernel(q, k, v, masks=masks, bias=bias, scale=self.scale, dropout_p=dropout_p)  # [B,H,T,hd]

        if self.training and self.cfg.head_dropout > 0:
            keep = 1.0 - self.cfg.head_dropout
            head_mask = torch.bernoulli(torch.full((B, H, 1, 1), keep, device=x.device, dtype=out.dtype))
            out = out * head_mask / keep
        out = out.transpose(1, 2).reshape(B, Tq, H * hd)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out)
