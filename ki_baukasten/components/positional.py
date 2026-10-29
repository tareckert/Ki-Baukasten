"""
ki_baukasten.components.positional
==================================

Positionskodierungen – wie ein Modell die *Reihenfolge* von Elementen erfährt.

Warum braucht man das?
----------------------
Attention selbst ist reihenfolgeblind: "Hund beißt Mann" und "Mann beißt
Hund" sähen ohne Positionsinformation gleich aus. Es gibt drei grundlegend
verschiedene Eingriffspunkte, die dieses Modul alle abdeckt:

1. **Absolut, auf die Embeddings addiert** – ``embed(x, positions)``
   (sinusförmig, gelernt)
2. **Rotation von Query/Key** – ``rotate(q, k, positions)`` (RoPE): relative
   Positionen ergeben sich aus dem Winkel zwischen q und k.
3. **Bias auf die Attention-Logits** – ``bias(...)`` (ALiBi, T5-Relativ-Bias)

Über ``combined`` lassen sich mehrere Verfahren **gleichzeitig** verwenden
(z. B. gelernt + RoPE) – eher experimentell, aber erlaubt.

Registry ``positional``
-----------------------
Fabrik-Signatur: ``factory(args, d_model, head_dim, num_heads, max_seq_len)``
→ ``PositionalEncoding``. ``num_heads`` ist die *größte* Kopfzahl im Modell
(Schichten dürfen per Override weniger Köpfe haben).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..core import BaseConfig, ComponentListParam, Level, Param, T, create_registry

POSITIONALS = create_registry(
    "positional",
    "Positionskodierungen",
    T(
        "Verfahren, mit denen ein Modell die Position von Elementen in einer Sequenz kennt.",
        "Ohne sie ist Attention reihenfolgeblind.",
        "Transformer, Attention-basierte Modelle.",
    ),
    context_doc="factory(args, d_model, head_dim, num_heads, max_seq_len)",
)


class PositionalEncoding(nn.Module):
    """Basisklasse mit drei (standardmäßig wirkungslosen) Eingriffspunkten."""

    #: Größte unterstützte Position + 1 (``None`` = unbegrenzt) – wird in Python geprüft (kein GPU-Sync, exportierbar).
    max_positions: int | None = None

    def check_length(self, length: int) -> None:
        """Länge der Positionen prüfen (reine Python-Zahlen → keine Synchronisation, kein Export-Hindernis)."""
        if self.max_positions is not None and length > self.max_positions:
            raise ValueError(f"Sequenz länger als max_seq_len={self.max_positions} der Positionskodierung.")

    def embed(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Absolute Kodierung auf Embeddings ``x`` ``[B, T, D]``."""
        return x

    def rotate(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotation von Query/Key ``[B, H, T, hd]``."""
        return q, k

    def bias(self, q_len: int, k_len: int, num_heads: int, device: torch.device, causal: bool) -> torch.Tensor | None:
        """Additiver Bias ``[H, Tq, Tk]`` auf die Attention-Logits (oder ``None``)."""
        return None

    def rotate_keys(self, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Nur Keys rotieren (z. B. bei MLA-lite, wo Keys aus dem Cache neu entstehen)."""
        return self.rotate(k, k, positions)[1]


class NoPositional(PositionalEncoding):
    pass


class SinusoidalPositional(PositionalEncoding):
    """Feste Sinus/Kosinus-Kodierung aus "Attention Is All You Need"."""

    def __init__(self, d_model: int, max_len: int, base: float, scale: float):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)
        self.scale = scale
        self.max_positions = max_len

    def embed(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.pe[positions].to(x.dtype)


class LearnedPositional(PositionalEncoding):
    """Gelernte absolute Positions-Embeddings (GPT-2, BERT)."""

    def __init__(self, d_model: int, max_len: int, init_std: float):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, std=init_std)
        self.max_positions = max_len

    def embed(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x + self.embedding(positions).to(x.dtype)


class RotaryPositional(PositionalEncoding):
    """Rotary Position Embedding (RoPE, Su et al. 2021).

    Dreht Paare von Query/Key-Komponenten um positionsabhängige Winkel. Das
    Skalarprodukt q·k hängt dadurch nur noch von der *relativen* Position ab.

    Die Frequenzen werden passend zur tatsächlichen Kopfdimension berechnet
    (und zwischengespeichert) – Schichten mit unterschiedlich großen Köpfen
    (über ``layer_overrides``) funktionieren dadurch ohne Weiteres.
    """

    def __init__(self, theta: float, fraction: float, scaling: str, factor: float, interleaved: bool,
                 original_context: int = 2048, beta_fast: float = 32.0, beta_slow: float = 1.0):
        super().__init__()
        self.theta, self.fraction, self.scaling, self.factor = theta, fraction, scaling, factor
        self.pos_scale = 1.0 / factor if scaling == "linear" else 1.0
        self.interleaved = interleaved
        self.original_context, self.beta_fast, self.beta_slow = original_context, beta_fast, beta_slow
        # YaRN: Temperatur-Korrektur der Attention (auf q und k verteilt → Faktor auf cos/sin).
        self.mscale = (0.1 * math.log(factor) + 1.0) if scaling == "yarn" and factor > 1 else 1.0
        self._cache: dict[tuple[int, str], torch.Tensor] = {}

    def _inv_freq(self, head_dim: int, device: torch.device) -> tuple[int, torch.Tensor]:
        rot = max(2, int(head_dim * self.fraction) // 2 * 2)
        key = (rot, str(device))
        if key not in self._cache:
            theta = self.theta
            if self.scaling == "ntk" and self.factor != 1.0:
                # NTK-aware: Basis vergrößern statt Positionen zu stauchen
                theta = theta * self.factor ** (rot / max(1, rot - 2))
            inv = 1.0 / (theta ** (torch.arange(0, rot, 2, device=device).float() / rot))
            if self.scaling == "yarn" and self.factor != 1.0:
                # YaRN (Peng et al. 2023): hohe Frequenzen (kurze Wellenlängen) unverändert lassen,
                # niedrige Frequenzen interpolieren, dazwischen weich überblenden.
                ratio = self.original_context / (2 * math.pi / inv)       # Umdrehungen im Originalkontext
                keep = ((ratio - self.beta_slow) / (self.beta_fast - self.beta_slow)).clamp(0.0, 1.0)
                inv = inv / self.factor * (1 - keep) + inv * keep
            self._cache[key] = inv
        return rot, self._cache[key]

    def _rotate_one(self, x: torch.Tensor, rot: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        xr, xp = x[..., :rot], x[..., rot:]
        if self.interleaved:
            x1, x2 = xr[..., 0::2], xr[..., 1::2]
            out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
        else:
            half = rot // 2
            x1, x2 = xr[..., :half], xr[..., half:]
            out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return torch.cat((out.to(x.dtype), xp), dim=-1)

    def _cos_sin(self, head_dim: int, positions: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        rot, inv_freq = self._inv_freq(head_dim, positions.device)
        freqs = torch.outer(positions.float() * self.pos_scale, inv_freq)  # [T, rot/2]
        return rot, freqs.cos() * self.mscale, freqs.sin() * self.mscale

    def rotate(self, q, k, positions):
        rot, cos, sin = self._cos_sin(q.shape[-1], positions)
        return self._rotate_one(q, rot, cos, sin), self._rotate_one(k, rot, cos, sin)

    def rotate_keys(self, k, positions):
        rot, cos, sin = self._cos_sin(k.shape[-1], positions)
        return self._rotate_one(k, rot, cos, sin)


def alibi_slopes(n: int) -> torch.Tensor:
    """Kopf-spezifische Steigungen nach Press et al. (auch für n ≠ 2^k)."""

    def pow2(m: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(m) - 3)))
        return [start * start ** i for i in range(m)]

    if math.log2(n).is_integer():
        return torch.tensor(pow2(n))
    closest = 2 ** math.floor(math.log2(n))
    extra = pow2(2 * closest)[0::2][: n - closest]
    return torch.tensor(pow2(closest) + extra)


class ALiBiPositional(PositionalEncoding):
    """Attention with Linear Biases: Strafterm proportional zur Distanz."""

    def __init__(self, num_heads: int, learnable: bool, slope_scale: float):
        super().__init__()
        self.num_heads, self.slope_scale = num_heads, slope_scale
        slopes = alibi_slopes(num_heads) * slope_scale
        if learnable:
            self.slopes = nn.Parameter(slopes)
        else:
            self.register_buffer("slopes", slopes, persistent=False)

    def bias(self, q_len, k_len, num_heads, device, causal):
        i = torch.arange(q_len, device=device).unsqueeze(1) + (k_len - q_len)
        j = torch.arange(k_len, device=device).unsqueeze(0)
        dist = (i - j).clamp_min(0) if causal else (i - j).abs()
        # Schichten mit abweichender Kopfzahl bekommen eigene (feste) Standard-Steigungen.
        slopes = self.slopes if num_heads == self.num_heads else alibi_slopes(num_heads) * self.slope_scale
        return -slopes.to(device).view(-1, 1, 1) * dist.unsqueeze(0).float()


class RelativeBiasPositional(PositionalEncoding):
    """Gelernter, gebucketter Relativ-Bias wie in T5."""

    def __init__(self, num_heads: int, num_buckets: int, max_distance: int, bidirectional: bool):
        super().__init__()
        self.num_buckets, self.max_distance, self.bidirectional = num_buckets, max_distance, bidirectional
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _bucket(self, rel: torch.Tensor, causal: bool) -> torch.Tensor:
        n = self.num_buckets
        bidir = self.bidirectional and not causal
        out = torch.zeros_like(rel)
        if bidir:
            n //= 2
            out = out + (rel > 0).long() * n
            rel = rel.abs()
        else:
            rel = (-rel).clamp_min(0)
        exact = n // 2
        is_small = rel < exact
        large = exact + (
            torch.log(rel.float().clamp_min(1) / exact) / math.log(self.max_distance / exact) * (n - exact)
        ).long()
        large = large.clamp_max(n - 1)
        return out + torch.where(is_small, rel, large)

    def bias(self, q_len, k_len, num_heads, device, causal):
        if num_heads > self.embedding.embedding_dim:
            raise ValueError(f"relative_bias wurde für {self.embedding.embedding_dim} Köpfe gebaut, Schicht hat {num_heads}.")
        ctx = torch.arange(q_len, device=device).unsqueeze(1) + (k_len - q_len)
        mem = torch.arange(k_len, device=device).unsqueeze(0)
        buckets = self._bucket(mem - ctx, causal)
        # Bei Schichten mit weniger Köpfen werden die ersten num_heads Bias-Kanäle genutzt.
        return self.embedding(buckets).permute(2, 0, 1)[:num_heads]  # [H, Tq, Tk]


class CombinedPositional(PositionalEncoding):
    """Wendet mehrere Positionskodierungen nacheinander an (experimentell)."""

    def __init__(self, parts: list[PositionalEncoding]):
        super().__init__()
        self.parts = nn.ModuleList(parts)
        limits = [p.max_positions for p in parts if p.max_positions is not None]
        self.max_positions = min(limits) if limits else None

    def embed(self, x, positions):
        for p in self.parts:
            x = p.embed(x, positions)
        return x

    def rotate(self, q, k, positions):
        for p in self.parts:
            q, k = p.rotate(q, k, positions)
        return q, k

    def rotate_keys(self, k, positions):
        for p in self.parts:
            k = p.rotate_keys(k, positions)
        return k

    def bias(self, q_len, k_len, num_heads, device, causal):
        total = None
        for p in self.parts:
            b = p.bias(q_len, k_len, num_heads, device, causal)
            if b is not None:
                total = b if total is None else total + b
        return total


# ===========================================================================
# Argument-Konfigurationen
# ===========================================================================

class MaxLenConfig(BaseConfig):
    max_len = Param(None, type=int, min=1, tooltip=T(
        "Maximale Sequenzlänge dieser Kodierung.",
        "Bestimmt die Größe der Tabelle; längere Sequenzen sind danach unmöglich.",
        "Nur setzen, wenn abweichend von max_seq_len des Modells (None = übernehmen).",
    ), level=Level.ADVANCED)


class SinusoidalConfig(MaxLenConfig):
    base = Param(10000.0, min=1.0, tooltip=T(
        "Basis der Wellenlängen (geometrische Reihe bis base·2π).",
        "Größere Basis = langsamere Frequenzen = längere unterscheidbare Abstände.",
        "Sehr lange Sequenzen.",
    ), level=Level.EXPERT)
    scale = Param(1.0, tooltip=T(
        "Faktor, mit dem die Kodierung addiert wird.",
        "Balanciert Positions- gegen Inhaltsinformation.",
        "Experimente.",
    ), level=Level.EXPERT)


class LearnedConfig(MaxLenConfig):
    init_std = Param(0.02, min=0.0, tooltip=T(
        "Startstreuung der gelernten Positionsvektoren.", "Kleiner Start stört Token-Embeddings wenig.",
        "GPT-2/BERT: 0.02."), level=Level.EXPERT)


class RoPEConfig(BaseConfig):
    theta = Param(10000.0, min=1.0, tooltip=T(
        "Basisfrequenz θ der Rotationen.",
        "Größeres θ → längere Kontexte unterscheidbar (LLaMA-3: 500000).",
        "Lange Kontexte → θ erhöhen.",
    ))
    fraction = Param(1.0, min=0.0, max=1.0, tooltip=T(
        "Anteil der Kopf-Dimensionen, die rotiert werden ('partial rotary').",
        "Rest bleibt positionsfrei – manche Modelle (GPT-NeoX: 0.25) profitieren davon.",
        "Feintuning von Positions- vs. Inhaltsanteil.",
    ), level=Level.ADVANCED)
    scaling = Param("none", choices={
        "none": "keine Kontext-Erweiterung",
        "linear": "Positionen stauchen (Position Interpolation)",
        "ntk": "Basis vergrößern (NTK-aware) – meist bessere Qualität ohne Nachtraining",
        "yarn": "YaRN: frequenzabhängige Interpolation + Attention-Temperatur (beste Qualität, Qwen/DeepSeek)",
    }, tooltip=T("Verfahren zur Kontextverlängerung.", "Erlaubt Sequenzen länger als im Training.",
                 "Kontext-Erweiterung vortrainierter Modelle."), level=Level.ADVANCED)
    factor = Param(1.0, min=1.0, tooltip=T(
        "Erweiterungsfaktor des Kontexts (z. B. 4 = vierfache Länge).", "Stärke der Skalierung.",
        "Nur mit scaling ≠ none."), level=Level.ADVANCED)
    original_context = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Kontextlänge, mit der ursprünglich trainiert wurde (nur YaRN; None = max_seq_len / factor).",
        "YaRN entscheidet anhand dieser Länge, welche Frequenzen interpoliert werden.",
        "Vortrainiertes Modell mit 4096 Tokens auf 16k erweitern: 4096, factor 4."), level=Level.EXPERT)
    yarn_beta_fast = Param(32.0, min=0.0, tooltip=T(
        "YaRN β_fast: ab so vielen Umdrehungen im Originalkontext bleibt eine Frequenz unverändert.",
        "Schützt die lokale Positionsauflösung.", "32 (Paper)."), level=Level.EXPERT)
    yarn_beta_slow = Param(1.0, min=0.0, tooltip=T(
        "YaRN β_slow: unter so vielen Umdrehungen wird voll interpoliert.", "Niedrige Frequenzen.", "1 (Paper)."),
        level=Level.EXPERT)
    interleaved = Param(False, tooltip=T(
        "Paare benachbarter Dimensionen rotieren (GPT-J) statt Hälften (GPT-NeoX/LLaMA).",
        "Nur für Kompatibilität mit fremden Gewichten relevant.",
        "Gewichte-Import.",
    ), level=Level.EXPERT)


class ALiBiConfig(BaseConfig):
    learnable = Param(False, tooltip=T(
        "Steigungen lernbar machen.", "Netz wählt selbst, wie stark Distanz bestraft wird.", "Experimente."),
        level=Level.EXPERIMENTAL)
    slope_scale = Param(1.0, min=0.0, tooltip=T(
        "Faktor auf alle Steigungen.", "Größer = stärkerer Fokus auf nahe Elemente.", "Experimente."),
        level=Level.EXPERT)


class RelativeBiasConfig(BaseConfig):
    num_buckets = Param(32, min=4, tooltip=T(
        "Anzahl Distanz-Klassen.", "Mehr Buckets = feinere Unterscheidung von Distanzen.", "T5: 32."))
    max_distance = Param(128, min=2, tooltip=T(
        "Ab dieser Distanz teilen sich alle Abstände einen Bucket.", "Begrenzt Parameter, generalisiert auf Länge.",
        "T5: 128."))
    bidirectional = Param(True, tooltip=T(
        "Vor/zurück unterscheiden (nur ohne Kausalmaske).", "Encoder profitieren von Richtungsinfo.", "Encoder."))


class CombinedConfig(BaseConfig):
    parts = ComponentListParam("positional", default=["learned", "rope"], tooltip=T(
        "Liste der zu kombinierenden Positionskodierungen.",
        "Experimentell: verschiedene Positionssignale gleichzeitig.",
        "Forschung / Ablationen.",
    ))


# ===========================================================================
# Registrierung
# ===========================================================================

R = POSITIONALS.register


@R("none", aliases=("nope", "keine"), tooltip=T(
    "Keine Positionskodierung (NoPE).",
    "Kausale Decoder können Positionen teils implizit aus der Maske lernen; generalisiert oft gut auf Länge.",
    "Forschung, Längen-Generalisierung, Mengen-Daten (Reihenfolge egal).",
))
def _none(a, d_model, head_dim, num_heads, max_seq_len):
    return NoPositional()


@R("sinusoidal", config=SinusoidalConfig, aliases=("sin",), tooltip=T(
    "Feste Sinus/Kosinus-Kodierung (Original-Transformer).",
    "Keine Parameter, deterministisch.",
    "Klassische Encoder-Decoder, Zeitreihen.",
))
def _sin(a, d_model, head_dim, num_heads, max_seq_len):
    return SinusoidalPositional(d_model, a.max_len or max_seq_len, a.base, a.scale)


@R("learned", config=LearnedConfig, aliases=("absolute",), tooltip=T(
    "Gelernte absolute Positions-Embeddings.",
    "Flexibel, aber keine Extrapolation über max_seq_len hinaus.",
    "GPT-2, BERT, ViT.",
))
def _learned(a, d_model, head_dim, num_heads, max_seq_len):
    return LearnedPositional(d_model, a.max_len or max_seq_len, a.init_std)


@R("rope", config=RoPEConfig, aliases=("rotary",), tags=("llm",), tooltip=T(
    "Rotary Position Embedding (RoPE).",
    "Relative Positionen, gute Qualität, Kontext erweiterbar.",
    "Standard moderner LLMs (LLaMA, Mistral, Qwen).",
))
def _rope(a, d_model, head_dim, num_heads, max_seq_len):
    original = a.original_context or max(1, int(max_seq_len / max(a.factor, 1.0)))
    return RotaryPositional(a.theta, a.fraction, a.scaling, a.factor, a.interleaved, original,
                            a.yarn_beta_fast, a.yarn_beta_slow)


@R("alibi", config=ALiBiConfig, tooltip=T(
    "ALiBi: linearer Distanz-Strafterm pro Kopf.",
    "Keine Parameter, extrapoliert gut auf längere Sequenzen.",
    "Modelle, die länger generieren sollen als trainiert (BLOOM, MPT).",
))
def _alibi(a, d_model, head_dim, num_heads, max_seq_len):
    return ALiBiPositional(num_heads, a.learnable, a.slope_scale)


@R("relative_bias", config=RelativeBiasConfig, aliases=("t5",), level=Level.ADVANCED, tooltip=T(
    "Gelernter relativer Positions-Bias (T5).",
    "Lernt, wie wichtig Distanzen sind; gebucketted für lange Distanzen.",
    "Encoder-Decoder (T5), Encoder.",
))
def _relbias(a, d_model, head_dim, num_heads, max_seq_len):
    return RelativeBiasPositional(num_heads, a.num_buckets, a.max_distance, a.bidirectional)


@R("combined", config=CombinedConfig, level=Level.EXPERIMENTAL, tooltip=T(
    "Kombination mehrerer Positionskodierungen.",
    "Maximale Freiheit: z. B. absolute + rotierende Kodierung gleichzeitig.",
    "Experimente.",
))
def _combined(a, d_model, head_dim, num_heads, max_seq_len):
    return CombinedPositional([p.build(d_model, head_dim, num_heads, max_seq_len) for p in a.parts])
