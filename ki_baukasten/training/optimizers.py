"""
ki_baukasten.training.optimizers
================================

Optimierer – *wie* aus Gradienten Gewichtsänderungen werden.

Überblick
---------
=============  ==============================================================
Optimierer     Charakter
=============  ==============================================================
sgd            einfach, gut generalisierend (mit Momentum), braucht Tuning
adam / adamw   adaptiv, robust, Standard für fast alles (AdamW = korrekter WD)
nadam, radam   Adam-Varianten (Nesterov-Momentum / Varianz-Korrektur)
adamax         Adam mit ∞-Norm
adagrad        für spärliche Merkmale
adadelta       lernratenarm
rmsprop        klassisch in RL/RNNs
adafactor      speichersparend (faktorisierte zweite Momente) – große Modelle
lion           nur Vorzeichen-Updates, wenig Speicher (Chen et al. 2023)
muon           orthogonalisierte Momentum-Updates für Matrizen (2024/25, experimentell)
=============  ==============================================================

Registry ``optimizer``
----------------------
Fabrik-Signatur: ``factory(args, param_groups) -> torch.optim.Optimizer``.
``param_groups`` werden vom Trainer gebaut (Weight-Decay-Ausnahmen,
schichtweise Lernraten, Tensor-Chirurgie-Regeln) und enthalten bereits ``lr``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer

from ..core import BaseConfig, Level, Param, T, create_registry

OPTIMIZERS = create_registry(
    "optimizer",
    "Optimierer",
    T(
        "Verfahren, das Gradienten in Gewichtsänderungen übersetzt.",
        "Bestimmt Konvergenzgeschwindigkeit, Stabilität, Speicherbedarf und Generalisierung.",
        "Jedes Training.",
    ),
    context_doc="factory(args, param_groups) -> torch.optim.Optimizer",
)


# ===========================================================================
# Konfigurationen
# ===========================================================================

class OptimizerConfig(BaseConfig):
    """Gemeinsame Einstellungen aller Optimierer."""

    __title__ = "Optimierer"
    lr = Param(1e-3, min=0.0, tooltip=T(
        "Lernrate – Schrittweite der Gewichtsänderungen.",
        "Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.",
        "Jedes Training.",
        "Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5.",
    ))
    weight_decay = Param(0.0, min=0.0, tooltip=T(
        "Gewichtsverfall: zieht Gewichte in Richtung 0.",
        "Regularisiert gegen Überanpassung, hält Gewichte klein.",
        "AdamW: 0.01…0.1; SGD: 1e-4…5e-4.",
        "Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns).",
    ), group="Regularisierung")


class SGDConfig(OptimizerConfig):
    lr = Param(0.01, min=0.0, tooltip=T("Lernrate.", "Schrittweite.", "SGD: 0.01–0.1."))
    momentum = Param(0.9, min=0.0, max=1.0, tooltip=T(
        "Momentum: Anteil der letzten Bewegung, der beibehalten wird.",
        "Glättet die Richtung, beschleunigt in Tälern.", "0.9 Standard."))
    dampening = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dämpfung des Momentums.", "Reduziert den Gradientenanteil im Momentum.", "Meist 0."), level=Level.EXPERT)
    nesterov = Param(True, tooltip=T(
        "Nesterov-Momentum (schaut voraus).", "Oft etwas schnellere Konvergenz.", "Mit momentum > 0."), level=Level.ADVANCED)


class AdamConfig(OptimizerConfig):
    betas = Param([0.9, 0.999], item_type=float, min=0.0, max=1.0, length=2, tooltip=T(
        "Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).",
        "β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.",
        "LLMs: [0.9, 0.95]; Standard: [0.9, 0.999].",
    ), level=Level.ADVANCED)
    eps = Param(1e-8, min=0.0, tooltip=T(
        "Stabilisator im Nenner.", "Verhindert Division durch ~0; größer = vorsichtigere Updates.",
        "fp16: 1e-6…1e-5."), level=Level.EXPERT)
    amsgrad = Param(False, tooltip=T(
        "AMSGrad-Variante (monotones 2. Moment).", "Theoretische Konvergenzgarantie, praktisch selten besser.",
        "Experimente."), level=Level.EXPERT)
    fused = Param(None, type=bool, tooltip=T(
        "Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).", "Schneller auf GPUs.",
        "Geschwindigkeit auf CUDA."), level=Level.EXPERT, group="Geschwindigkeit")


class AdamWConfig(AdamConfig):
    weight_decay = Param(0.01, min=0.0, tooltip=T(
        "Entkoppelter Gewichtsverfall (AdamW).",
        "Korrekte Regularisierung für adaptive Optimierer – der Grund, AdamW statt Adam zu nehmen.",
        "0.01…0.1 (LLMs 0.1).",
    ), group="Regularisierung")


class NAdamConfig(AdamConfig):
    momentum_decay = Param(4e-3, min=0.0, tooltip=T(
        "Momentum-Zerfall von NAdam.", "Steuert Nesterov-Anteil über die Zeit.", "Selten."), level=Level.EXPERT)


class AdagradConfig(OptimizerConfig):
    lr = Param(0.01, min=0.0, tooltip=T("Lernrate.", "Schrittweite.", "0.01."))
    lr_decay = Param(0.0, min=0.0, tooltip=T("Lernratenzerfall.", "Zusätzliches Abklingen.", "Meist 0."), level=Level.EXPERT)
    initial_accumulator_value = Param(0.0, min=0.0, tooltip=T(
        "Startwert der Gradienten-Summe.", "Dämpft frühe Schritte.", "0–0.1."), level=Level.EXPERT)
    eps = Param(1e-10, min=0.0, tooltip=T("Stabilisator.", "Gegen Division durch 0.", "Selten."), level=Level.EXPERT)


class AdadeltaConfig(OptimizerConfig):
    lr = Param(1.0, min=0.0, tooltip=T("Skalierungsfaktor (Adadelta ist lernratenarm).", "Meist 1.", "1.0."))
    rho = Param(0.9, min=0.0, max=1.0, tooltip=T("Abklingrate.", "Fenster der Gradientenstatistik.", "0.9."))
    eps = Param(1e-6, min=0.0, tooltip=T("Stabilisator.", "Gegen Division durch 0.", "Selten."), level=Level.EXPERT)


class RMSpropConfig(OptimizerConfig):
    lr = Param(1e-3, min=0.0, tooltip=T("Lernrate.", "Schrittweite.", "1e-3; RL: 7e-4."))
    alpha = Param(0.99, min=0.0, max=1.0, tooltip=T("Glättung des 2. Moments.", "Gedächtnis der Skalierung.", "0.99."))
    eps = Param(1e-8, min=0.0, tooltip=T("Stabilisator.", "Gegen Division durch 0.", "RL (A2C): 1e-5."), level=Level.EXPERT)
    momentum = Param(0.0, min=0.0, max=1.0, tooltip=T("Momentum.", "Glättung der Richtung.", "0."), level=Level.ADVANCED)
    centered = Param(False, tooltip=T(
        "Zentrierte Variante (normiert mit Varianz statt 2. Moment).", "Manchmal stabiler.", "Experimente."),
        level=Level.EXPERT)


class AdafactorConfig(OptimizerConfig):
    lr = Param(1e-2, min=0.0, tooltip=T("Lernrate.", "Schrittweite.", "1e-2 (relativ)."))
    beta2_decay = Param(-0.8, max=0.0, tooltip=T(
        "Exponent des β₂-Zeitplans.", "Steuert das Gedächtnis des 2. Moments.", "-0.8."), level=Level.EXPERT)
    d = Param(1.0, min=0.0, tooltip=T("Clipping-Schwelle der Update-RMS.", "Begrenzt Ausreißer-Updates.", "1.0."),
              level=Level.EXPERT)


class LionConfig(OptimizerConfig):
    lr = Param(1e-4, min=0.0, tooltip=T(
        "Lernrate (Lion braucht ~3–10× kleinere Lernraten als AdamW).", "Schrittweite.", "1e-4…3e-4."))
    weight_decay = Param(0.1, min=0.0, tooltip=T(
        "Entkoppelter Gewichtsverfall (Lion braucht ~3–10× größeren WD als AdamW).", "Regularisierung.", "0.1…1."),
        group="Regularisierung")
    betas = Param([0.9, 0.99], item_type=float, min=0.0, max=1.0, length=2, tooltip=T(
        "β₁ (Interpolation für Update), β₂ (Momentum-Update).", "Glättung.", "[0.9, 0.99]."), level=Level.ADVANCED)


class MuonConfig(OptimizerConfig):
    lr = Param(0.02, min=0.0, tooltip=T(
        "Lernrate für Matrix-Parameter (Muon).", "Schrittweite der orthogonalisierten Updates.", "0.02."))
    momentum = Param(0.95, min=0.0, max=1.0, tooltip=T("Momentum.", "Glättung.", "0.95."))
    nesterov = Param(True, tooltip=T("Nesterov-Momentum.", "Vorausschauende Richtung.", "an."), level=Level.ADVANCED)
    ns_steps = Param(5, min=1, max=20, tooltip=T(
        "Newton-Schulz-Iterationen zur Orthogonalisierung.", "Genauigkeit vs. Rechenaufwand.", "5."), level=Level.EXPERT)
    adamw_lr = Param(3e-4, min=0.0, tooltip=T(
        "Lernrate für Nicht-Matrix-Parameter (Biases, Normen, Embeddings), die mit AdamW laufen.",
        "Muon ist nur für 2D-Gewichte gedacht.", "3e-4."))
    adamw_betas = Param([0.9, 0.95], item_type=float, min=0.0, max=1.0, length=2, tooltip=T(
        "AdamW-Betas für Nicht-Matrix-Parameter.", "Siehe Adam.", "[0.9, 0.95]."), level=Level.EXPERT)
    treat_embeddings_as_matrix = Param(False, tooltip=T(
        "Auch Embedding-Tabellen mit Muon optimieren.", "Empfehlung des Originals: nein (AdamW).", "Experimente."),
        level=Level.EXPERIMENTAL)


# ===========================================================================
# Eigene Optimierer
# ===========================================================================

class Lion(Optimizer):
    """Lion: EvoLved Sign Momentum (Chen et al. 2023)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                update = (m * b1 + p.grad * (1 - b1)).sign_()
                p.add_(update, alpha=-group["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Näherungsweise Orthogonalisierung (quintische Newton-Schulz-Iteration, Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float() / (g.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    return (x.T if transposed else x).to(g.dtype)


class Muon(Optimizer):
    """Muon (Jordan et al. 2024): orthogonalisierte Momentum-Updates für Matrizen.

    Parameter mit ≥ 2 Dimensionen (außer Embeddings, siehe Option) laufen mit
    Muon; alle anderen intern mit AdamW.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0,
                 adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay,
                        adamw_lr=adamw_lr, adamw_betas=tuple(adamw_betas), adamw_eps=adamw_eps, use_muon=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            scale_lr = group["lr"] / max(group.get("initial_lr", group["lr"]), 1e-30) if "initial_lr" in group else 1.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if group["use_muon"] and p.dim() >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)
                    upd = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    shape = upd.shape
                    o = newton_schulz(upd.reshape(shape[0], -1), group["ns_steps"]).reshape(shape)
                    o = o * max(1.0, shape[0] / max(1, o[0].numel())) ** 0.5
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(o, alpha=-group["lr"])
                else:
                    lr = group["adamw_lr"] * scale_lr
                    b1, b2 = group["adamw_betas"]
                    if "step" not in state:
                        state["step"], state["m"], state["v"] = 0, torch.zeros_like(g), torch.zeros_like(g)
                    state["step"] += 1
                    state["m"].lerp_(g, 1 - b1)
                    state["v"].lerp_(g.square(), 1 - b2)
                    mh = state["m"] / (1 - b1 ** state["step"])
                    vh = state["v"] / (1 - b2 ** state["step"])
                    p.mul_(1 - lr * group["weight_decay"])
                    p.addcdiv_(mh, vh.sqrt().add_(group["adamw_eps"]), value=-lr)
        return loss


# ===========================================================================
# Registrierung
# ===========================================================================

R = OPTIMIZERS.register


def _kw(a: BaseConfig, *names: str) -> dict[str, Any]:
    out = {}
    for n in names:
        v = getattr(a, n)
        if n == "betas":
            v = tuple(v)
        out[n] = v
    return out


R("sgd", lambda a, g: torch.optim.SGD(g, nesterov=a.nesterov and a.momentum > 0,
                                       **_kw(a, "lr", "momentum", "dampening", "weight_decay")),
  config=SGDConfig, tooltip=T(
      "Stochastischer Gradientenabstieg (mit Momentum/Nesterov).",
      "Einfach, wenig Speicher, generalisiert oft am besten (CNNs).",
      "Bildklassifikation (ResNet), wenn genug Tuning-Zeit vorhanden.",
  ))
R("adam", lambda a, g: torch.optim.Adam(g, **_kw(a, "lr", "betas", "eps", "weight_decay", "amsgrad", "fused")),
  config=AdamConfig, tooltip=T(
      "Adam: adaptive Schrittweite pro Parameter.", "Robust, schnell, wenig Tuning.",
      "Allzweck; mit weight_decay besser AdamW nehmen."))
R("adamw", lambda a, g: torch.optim.AdamW(g, **_kw(a, "lr", "betas", "eps", "weight_decay", "amsgrad", "fused")),
  config=AdamWConfig, tags=("transformer",), tooltip=T(
      "AdamW: Adam mit entkoppeltem Gewichtsverfall.",
      "De-facto-Standard für Transformer und die meisten modernen Netze.", "Fast alles."))
R("nadam", lambda a, g: torch.optim.NAdam(g, **_kw(a, "lr", "betas", "eps", "weight_decay", "momentum_decay")),
  config=NAdamConfig, level=Level.ADVANCED, tooltip=T("Adam mit Nesterov-Momentum.", "Teils schneller.", "Experimente."))
R("radam", lambda a, g: torch.optim.RAdam(g, **_kw(a, "lr", "betas", "eps", "weight_decay")), config=AdamConfig,
  level=Level.ADVANCED, tooltip=T("Rectified Adam (Varianzkorrektur am Anfang).", "Weniger Warmup nötig.",
                                  "Wenn Warmup stört."))
R("adamax", lambda a, g: torch.optim.Adamax(g, **_kw(a, "lr", "betas", "eps", "weight_decay")), config=AdamConfig,
  level=Level.ADVANCED, tooltip=T("Adam mit ∞-Norm.", "Robust gegen seltene große Gradienten.", "Embeddings."))
R("adagrad", lambda a, g: torch.optim.Adagrad(g, **_kw(a, "lr", "lr_decay", "weight_decay", "initial_accumulator_value",
                                                         "eps")),
  config=AdagradConfig, level=Level.ADVANCED, tooltip=T(
      "Adagrad: Lernrate sinkt mit der Gradientensumme.", "Gut für spärliche Merkmale.", "Empfehlung, NLP-Klassiker."))
R("adadelta", lambda a, g: torch.optim.Adadelta(g, **_kw(a, "lr", "rho", "eps", "weight_decay")), config=AdadeltaConfig,
  level=Level.ADVANCED, tooltip=T("Adadelta.", "Kaum Lernraten-Tuning.", "Klassiker."))
R("rmsprop", lambda a, g: torch.optim.RMSprop(g, **_kw(a, "lr", "alpha", "eps", "weight_decay", "momentum", "centered")),
  config=RMSpropConfig, tags=("rl",), tooltip=T("RMSprop.", "Adaptive Skalierung ohne Momentum-Bias-Korrektur.",
                                                "RNNs, A2C/DQN-Klassiker."))
if hasattr(torch.optim, "Adafactor"):
    R("adafactor", lambda a, g: torch.optim.Adafactor(g, **_kw(a, "lr", "beta2_decay", "d", "weight_decay")),
      config=AdafactorConfig, level=Level.ADVANCED, tooltip=T(
          "Adafactor: faktorisierte 2. Momente.", "Viel weniger Optimierer-Speicher als Adam.",
          "Sehr große Modelle (T5)."))
R("lion", lambda a, g: Lion(g, **_kw(a, "lr", "betas", "weight_decay")), config=LionConfig, level=Level.ADVANCED,
  tooltip=T("Lion: Vorzeichen-Momentum.", "Halber Optimierer-Speicher von Adam, oft gleich gut.",
            "Große Modelle, Bild-Transformer."))


@R("muon", config=MuonConfig, level=Level.EXPERIMENTAL, tooltip=T(
    "Muon: orthogonalisierte Momentum-Updates für Gewichtsmatrizen (+ AdamW für den Rest).",
    "Hat in Speedrun-Experimenten Transformer schneller trainiert als AdamW.",
    "Transformer-Vortraining, Forschung.",
))
def _muon(a: MuonConfig, groups: list[dict]) -> Optimizer:
    split = []
    for g in groups:
        params = list(g["params"])
        names = list(g.get("names", [""] * len(params)))

        def is_embedding(name: str) -> bool:
            # Embeddings und Ausgabeschicht laufen laut Muon-Empfehlung besser mit AdamW.
            return any(k in name for k in ("embed", "lm_head", "head."))

        matrix = [p for p, n in zip(params, names) if p.dim() >= 2 and (a.treat_embeddings_as_matrix or not is_embedding(n))]
        other = [p for p in params if not any(p is m for m in matrix)]
        base = {k: v for k, v in g.items() if k not in ("params", "names")}
        if matrix:
            split.append({**base, "params": matrix, "use_muon": True})
        if other:
            split.append({**base, "params": other, "use_muon": False})
    return Muon(split, lr=a.lr, momentum=a.momentum, nesterov=a.nesterov, ns_steps=a.ns_steps,
                weight_decay=a.weight_decay, adamw_lr=a.adamw_lr, adamw_betas=a.adamw_betas)

