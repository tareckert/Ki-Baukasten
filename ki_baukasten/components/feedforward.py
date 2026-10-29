"""
ki_baukasten.components.feedforward
===================================

Feed-Forward-Netze (FFN) der Transformer-Blöcke – inkl. Mixture-of-Experts.

Rolle im Transformer
--------------------
Attention *mischt* Informationen zwischen Positionen; das FFN *verarbeitet*
jede Position einzeln weiter. In großen Sprachmodellen steckt der Großteil
der Parameter (und vermutlich des "Wissens") in den FFNs.

Varianten
---------
* ``mlp``      – klassisch: Linear → Aktivierung → Linear
* ``glu``      – gated: (Aktivierung(xW) ⊙ xV) W₂; mit SiLU = **SwiGLU**
  (LLaMA), mit GELU = GeGLU, mit Sigmoid = GLU, mit Identität = Bilinear
* ``moe``      – Mixture-of-Experts: ein Router wählt pro Token die top-k
  von N Experten. Jeder Experte ist selbst wieder ein beliebiges FFN aus
  dieser Registry (sogar ein MoE – verschachtelt, falls gewünscht).
* ``none``     – kein FFN (reine Attention-Netze, Ablationen)

Registry ``feedforward``
------------------------
Fabrik-Signatur: ``factory(args, d_model) -> nn.Module`` (Ein-/Ausgabe ``[..., d_model]``).
Module können ein Attribut ``aux_loss`` setzen (z. B. MoE-Lastverteilung);
Modelle sammeln diese über ``BaseModel.collect_aux_losses()`` ein.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..core import BaseConfig, ComponentParam, Level, Param, T, create_registry

FEEDFORWARDS = create_registry(
    "feedforward",
    "Feed-Forward-Netze",
    T(
        "Positionsweise Teilnetze in Transformer-Blöcken.",
        "Tragen den Großteil der Kapazität eines Transformers.",
        "Transformer, Mixer-Architekturen.",
    ),
    context_doc="factory(args, d_model)",
)


def _hidden(d_model: int, mult: float, explicit: int | None, multiple_of: int) -> int:
    """Breite der versteckten Schicht, aufgerundet auf ein Vielfaches (hardwarefreundlich)."""
    h = explicit if explicit is not None else int(round(d_model * mult))
    return max(1, int(math.ceil(h / multiple_of)) * multiple_of)


# ===========================================================================
# Konfigurationen
# ===========================================================================

class MLPFFNConfig(BaseConfig):
    __title__ = "MLP-FFN"
    hidden_mult = Param(4.0, min=0.01, tooltip=T(
        "Breite der versteckten Schicht relativ zu d_model.",
        "Mehr Breite = mehr Kapazität und Parameter (FFN ≈ 2·mult·d² Parameter).",
        "Klassisch 4 (GPT/BERT).",
    ))
    hidden_dim = Param(None, type=int, min=1, tooltip=T(
        "Exakte Breite der versteckten Schicht (überschreibt hidden_mult).",
        "Für exakte Nachbauten bekannter Architekturen.",
        "Gewichte-Import, exakte Parameterzahlen.",
    ), level=Level.ADVANCED)
    multiple_of = Param(1, min=1, tooltip=T(
        "Versteckte Breite auf ein Vielfaches davon aufrunden.",
        "Vielfache von 64/256 rechnen auf GPUs schneller (Tensor Cores).",
        "Geschwindigkeitsoptimierung (LLaMA: 256).",
    ), level=Level.EXPERT)
    activation = ComponentParam("activation", "gelu", tooltip=T(
        "Aktivierung zwischen den beiden linearen Schichten.",
        "Bestimmt die Nichtlinearität des FFN.",
        "gelu (Standard), relu, squared_relu ...",
    ))
    bias = Param(True, tooltip=T(
        "Bias in den linearen Schichten.", "Klassisch an; moderne LLMs oft aus.", "Kompatibilität."),
        level=Level.ADVANCED)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout nach der Aktivierung (innerhalb des FFN).",
        "Regularisiert die versteckte Schicht.",
        "Kleine Datensätze.",
    ), group="Regularisierung")


class GLUFFNConfig(MLPFFNConfig):
    __title__ = "GLU-FFN"
    hidden_mult = Param(8 / 3, min=0.01, tooltip=T(
        "Breite relativ zu d_model (GLU hat 3 Matrizen → Standard 8/3 hält die Parameter wie MLP mit 4).",
        "Mehr Breite = mehr Kapazität.",
        "LLaMA: 8/3 (aufgerundet auf multiple_of).",
    ))
    activation = ComponentParam("activation", "silu", tooltip=T(
        "Gate-Aktivierung: silu → SwiGLU, gelu → GeGLU, sigmoid → GLU, relu → ReGLU, identity → Bilinear.",
        "Das Gate entscheidet elementweise, welche Information durchgelassen wird.",
        "SwiGLU ist Standard moderner LLMs.",
    ))
    bias = Param(False, tooltip=T(
        "Bias in den linearen Schichten.", "Moderne LLMs verzichten darauf.", "Kompatibilität."),
        level=Level.ADVANCED)


class MoEConfig(BaseConfig):
    __title__ = "Mixture-of-Experts"
    num_experts = Param(8, min=1, tooltip=T(
        "Anzahl Experten.",
        "Mehr Experten = viel mehr Parameter bei (fast) gleichen Rechenkosten pro Token.",
        "Große Modelle mit begrenztem Rechenbudget (Mixtral: 8, DeepSeek: 64+).",
    ))
    top_k = Param(2, min=1, tooltip=T(
        "Wie viele Experten pro Token aktiv sind.",
        "Mehr = bessere Qualität, aber höhere Kosten. 1 = Switch-Transformer.",
        "Mixtral: 2.",
        "Muss ≤ num_experts sein.",
    ))
    expert = ComponentParam("feedforward", "glu", tooltip=T(
        "Bauart jedes Experten (beliebiges FFN, sogar ein weiteres MoE).",
        "Experten können MLP, GLU oder exotische Varianten sein.",
        "Standard: GLU (SwiGLU).",
    ))
    num_shared_experts = Param(0, min=0, tooltip=T(
        "Zusätzliche Experten, die *jedes* Token immer durchläuft.",
        "Fangen gemeinsames Wissen ab, Routing-Experten können sich spezialisieren.",
        "DeepSeek-MoE-Stil.",
    ), level=Level.ADVANCED)
    normalize_topk = Param(True, tooltip=T(
        "Gewichte der gewählten Experten auf Summe 1 normieren.",
        "Hält die Ausgabegröße unabhängig von der Router-Sicherheit.",
        "Mixtral: an.",
    ), level=Level.ADVANCED)
    router_noise_std = Param(0.0, min=0.0, tooltip=T(
        "Rauschen auf die Router-Logits im Training (Noisy Top-k Gating).",
        "Fördert Erkundung, verhindert Kollaps auf wenige Experten.",
        "Training großer MoEs.",
    ), level=Level.EXPERT)
    router_bias = Param(False, tooltip=T(
        "Bias im Router.", "Erlaubt globale Vorlieben für Experten.", "Selten."), level=Level.EXPERT)
    capacity_factor = Param(None, type=float, min=0.1, tooltip=T(
        "Maximale Tokens pro Experte = factor · (Tokens·top_k / Experten). None = unbegrenzt.",
        "Begrenzt Lastspitzen (überzählige Tokens werden für diesen Experten verworfen).",
        "Verteiltes Training mit festen Puffergrößen.",
    ), level=Level.EXPERT)
    aux_loss_coef = Param(0.01, min=0.0, tooltip=T(
        "Gewicht des Lastverteilungs-Verlusts (Switch-Transformer).",
        "Bestraft ungleichmäßige Expertennutzung – sonst lernt oft nur ein Experte.",
        "Jedes MoE-Training. 0 = aus.",
    ), group="Verluste")
    z_loss_coef = Param(0.0, min=0.0, tooltip=T(
        "Gewicht des Router-z-Loss (bestraft große Router-Logits).",
        "Stabilisiert das Routing numerisch (ST-MoE).",
        "Große MoEs mit Instabilitäten (typisch 1e-3).",
    ), level=Level.EXPERT, group="Verluste")
    expert_dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Wahrscheinlichkeit, im Training eine Experten-Zuweisung zu verwerfen.",
        "Regularisiert, erzwingt Redundanz zwischen Experten.",
        "Experimente.",
    ), level=Level.EXPERIMENTAL, group="Regularisierung")

    def check(self) -> list[str]:
        return [f"top_k ({self.top_k}) > num_experts ({self.num_experts})."] if self.top_k > self.num_experts else []


# ===========================================================================
# Module
# ===========================================================================

class MLPFeedForward(nn.Module):
    def __init__(self, d_model: int, cfg: MLPFFNConfig):
        super().__init__()
        h = _hidden(d_model, cfg.hidden_mult, cfg.hidden_dim, cfg.multiple_of)
        self.up = nn.Linear(d_model, h, bias=cfg.bias)
        self.act = cfg.activation.build()
        self.drop = nn.Dropout(cfg.dropout)
        self.down = nn.Linear(h, d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(self.act(self.up(x))))


class GLUFeedForward(nn.Module):
    def __init__(self, d_model: int, cfg: GLUFFNConfig):
        super().__init__()
        h = _hidden(d_model, cfg.hidden_mult, cfg.hidden_dim, cfg.multiple_of)
        self.gate = nn.Linear(d_model, h, bias=cfg.bias)
        self.up = nn.Linear(d_model, h, bias=cfg.bias)
        self.act = cfg.activation.build()
        self.drop = nn.Dropout(cfg.dropout)
        self.down = nn.Linear(h, d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(self.act(self.gate(x)) * self.up(x)))


class MixtureOfExperts(nn.Module):
    """Token-weises Top-k-Routing über N Experten (+ optionale geteilte Experten)."""

    def __init__(self, d_model: int, cfg: MoEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.router = nn.Linear(d_model, cfg.num_experts, bias=cfg.router_bias)
        self.experts = nn.ModuleList([cfg.expert.build(d_model) for _ in range(cfg.num_experts)])
        self.shared = nn.ModuleList([cfg.expert.build(d_model) for _ in range(cfg.num_shared_experts)])
        self.aux_loss: torch.Tensor | None = None
        #: Anteil der Tokens pro Experte im letzten Forward (für Analyse/Logging).
        self.last_load: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        n_tokens = flat.shape[0]
        logits = self.router(flat).float()
        if self.training and c.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * c.router_noise_std
        probs = logits.softmax(-1)
        top_w, top_i = probs.topk(c.top_k, dim=-1)
        if c.normalize_topk:
            top_w = top_w / top_w.sum(-1, keepdim=True).clamp_min(1e-9)
        if self.training and c.expert_dropout > 0:
            top_w = top_w * (torch.rand_like(top_w) > c.expert_dropout)

        out = torch.zeros_like(flat)
        capacity = None
        if c.capacity_factor is not None:
            capacity = max(1, int(math.ceil(c.capacity_factor * n_tokens * c.top_k / c.num_experts)))
        for e, expert in enumerate(self.experts):
            token_idx, slot = (top_i == e).nonzero(as_tuple=True)
            if capacity is not None and token_idx.numel() > capacity:
                token_idx, slot = token_idx[:capacity], slot[:capacity]
            if token_idx.numel() == 0:
                continue
            y = expert(flat[token_idx])
            out.index_add_(0, token_idx, (y * top_w[token_idx, slot].unsqueeze(-1).to(y.dtype)).to(out.dtype))
        for expert in self.shared:
            out = out + expert(flat)

        # --- Hilfsverluste ---------------------------------------------------
        counts = torch.zeros(c.num_experts, device=x.device).index_add_(
            0, top_i.reshape(-1), torch.ones(top_i.numel(), device=x.device))
        load = counts / max(1, top_i.numel())  # Anteil der Zuweisungen pro Experte
        self.last_load = load.detach()
        aux = torch.zeros((), device=x.device)
        if c.aux_loss_coef > 0:
            aux = aux + c.aux_loss_coef * c.num_experts * (load * probs.mean(0)).sum()
        if c.z_loss_coef > 0:
            aux = aux + c.z_loss_coef * torch.logsumexp(logits, dim=-1).square().mean()
        self.aux_loss = aux
        return out.reshape(shape)


class KANConfig(BaseConfig):
    __title__ = "KAN-FFN"
    __tooltip__ = T(
        "Kolmogorov-Arnold-Netz als FFN: lernbare Funktionen auf den Kanten statt fester Aktivierungen.",
        "Jede Verbindung lernt ihre eigene eindimensionale Funktion (hier über Gauß-Basisfunktionen, FastKAN).",
        "Forschung, wissenschaftliche/symbolische Daten, Interpretierbarkeit.",
    )
    hidden_mult = Param(2.0, min=0.01, tooltip=T(
        "Breite der versteckten KAN-Schicht relativ zu d_model.",
        "KAN-Schichten haben grid_size-mal mehr Parameter als lineare – daher kleiner als bei MLP.",
        "1–2."))
    grid_size = Param(8, min=2, tooltip=T(
        "Anzahl Basisfunktionen pro Eingabe.", "Feinere Funktionen, mehr Parameter (∝ grid_size).", "5–10."))
    grid_range = Param([-2.0, 2.0], item_type=float, length=2, tooltip=T(
        "Wertebereich der Basisfunktions-Zentren (Eingaben werden vorher normalisiert).",
        "Sollte die typischen Werte abdecken.", "[-2, 2]."), level=Level.EXPERT)
    base_activation = ComponentParam("activation", "silu", tooltip=T(
        "Zusätzlicher Basis-Pfad: lineare Schicht auf act(x) (wie im KAN-Original).",
        "Stabilisiert das Training, gibt einen glatten Grundverlauf.", "silu; None = aus."), nullable=True)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T("Dropout zwischen den KAN-Schichten.", "Regularisierung.",
                                                        "0–0.1."), group="Regularisierung")


class KANLayer(nn.Module):
    """Eine KAN-Schicht (FastKAN, Li 2024): y = W_s · φ(LN(x)) + W_b · act(x), φ = Gauß-RBF-Basis."""

    def __init__(self, d_in: int, d_out: int, cfg: KANConfig):
        super().__init__()
        lo, hi = cfg.grid_range
        self.register_buffer("centers", torch.linspace(lo, hi, cfg.grid_size), persistent=False)
        self.inv_width = (cfg.grid_size - 1) / (hi - lo)
        self.norm = nn.LayerNorm(d_in)
        self.spline = nn.Linear(d_in * cfg.grid_size, d_out, bias=False)
        nn.init.trunc_normal_(self.spline.weight, std=0.1 / math.sqrt(d_in))
        self.act = cfg.base_activation.build() if cfg.base_activation is not None else None
        self.base = nn.Linear(d_in, d_out) if self.act is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (self.norm(x).unsqueeze(-1) - self.centers.to(x.dtype)) * self.inv_width
        out = self.spline(torch.exp(-z.square()).flatten(-2))
        return out + self.base(self.act(x)) if self.base is not None else out


class KANFeedForward(nn.Module):
    def __init__(self, d_model: int, cfg: KANConfig):
        super().__init__()
        h = max(1, int(round(d_model * cfg.hidden_mult)))
        self.l1, self.l2 = KANLayer(d_model, h, cfg), KANLayer(h, d_model, cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.drop(self.l1(x)))


class NoFeedForward(nn.Module):
    """Gibt Nullen zurück – zusammen mit der Residualverbindung also 'kein FFN'."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


# ===========================================================================
# Registrierung
# ===========================================================================

R = FEEDFORWARDS.register
R("mlp", lambda a, d_model: MLPFeedForward(d_model, a), config=MLPFFNConfig, tooltip=T(
    "Klassisches FFN: Linear → Aktivierung → Linear.",
    "Einfach, bewährt.",
    "GPT-2, BERT, ViT.",
))
R("glu", lambda a, d_model: GLUFeedForward(d_model, a), config=GLUFFNConfig, aliases=("swiglu", "gated"), tooltip=T(
    "Gated Linear Unit FFN (SwiGLU mit silu).",
    "Gating verbessert Qualität messbar bei gleicher Parameterzahl.",
    "LLaMA, Mistral, PaLM.",
))
R("moe", lambda a, d_model: MixtureOfExperts(d_model, a), config=MoEConfig, aliases=("mixture_of_experts",),
  level=Level.ADVANCED, tags=("llm",), tooltip=T(
      "Mixture-of-Experts: Router wählt pro Token die besten k Experten.",
      "Sehr viele Parameter bei geringen Rechenkosten pro Token.",
      "Große Sprachmodelle (Mixtral, DeepSeek, Switch).",
  ))
R("kan", lambda a, d_model: KANFeedForward(d_model, a), config=KANConfig, aliases=("kolmogorov_arnold",),
  level=Level.EXPERIMENTAL, tooltip=T(
      "Kolmogorov-Arnold-FFN (lernbare Kantenfunktionen, FastKAN-Variante).",
      "Andere induktive Annahme als MLPs: glatte, interpretierbare 1D-Funktionen pro Verbindung.",
      "Forschung (Liu et al. 2024), wissenschaftliche Regression.",
  ))
R("none", lambda a, d_model: NoFeedForward(), aliases=("keins", "identity"), level=Level.EXPERIMENTAL, tooltip=T(
    "Kein FFN (Block besteht nur aus Attention).",
    "Ablationsstudien, extrem leichte Modelle.",
    "Forschung.",
))
