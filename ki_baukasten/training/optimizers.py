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
adamw8bit      AdamW mit 8-Bit-Zuständen (¼ Optimierer-Speicher), eigene Implementierung
=============  ==============================================================

Registry ``optimizer``
----------------------
Fabrik-Signatur: ``factory(args, param_groups) -> torch.optim.Optimizer``.
``param_groups`` werden vom Trainer gebaut (Weight-Decay-Ausnahmen,
schichtweise Lernraten, Tensor-Chirurgie-Regeln) und enthalten bereits ``lr``.
"""

from __future__ import annotations

import math
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
        "Fusionierte Implementierung (None = automatisch: fusioniert, wenn das Gerät es kann – CPU/GPU; "
        "False = Standard).", "Ein Kernel für alle Parameter – auf CPU und GPU deutlich schneller.",
        "None lassen."), level=Level.EXPERT, group="Geschwindigkeit")


class AdamWConfig(AdamConfig):
    weight_decay = Param(0.01, min=0.0, tooltip=T(
        "Entkoppelter Gewichtsverfall (AdamW).",
        "Korrekte Regularisierung für adaptive Optimierer – der Grund, AdamW statt Adam zu nehmen.",
        "0.01…0.1 (LLMs 0.1).",
    ), group="Regularisierung")


class AdamW8bitConfig(AdamWConfig):
    block_size = Param(256, min=16, tooltip=T(
        "Blockgröße der 8-Bit-Quantisierung der Optimierer-Zustände.",
        "Jeder Block hat eine eigene Skala – kleiner = genauer, minimal mehr Speicher.", "256."),
        level=Level.EXPERT)
    min_8bit_size = Param(4096, min=0, tooltip=T(
        "Parameter mit weniger Elementen behalten 32-Bit-Zustände.",
        "Kleine Tensoren (Biases, Normen) sparen kaum, sind aber empfindlich.", "4096."), level=Level.EXPERT)
    dynamic_range = Param(14.0, min=2.0, max=40.0, tooltip=T(
        "Dynamikbereich der logarithmischen Quantisierung (natürlicher Log; 14 ≈ Faktor 10⁶ pro Block).",
        "Größer = auch sehr kleine Werte darstellbar, aber gröbere Stufen.", "14 (m), doppelt für v."),
        level=Level.EXPERIMENTAL)


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


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    flat = x.reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(-1, block)


class AdamW8bit(Optimizer):
    """AdamW mit 8-Bit-Optimierer-Zuständen (eigene Implementierung, jedes Gerät).

    Beide Momente werden blockweise **logarithmisch** quantisiert (relativer
    Fehler statt absolutem – kleine Werte gehen nicht verloren):

    * ``m`` (vorzeichenbehaftet): 1 Vorzeichen-Bit + 7 Bit Betrag, 0 = exakt null
    * ``v`` (nicht negativ): 8 Bit; Unterlauf wird auf die kleinste Stufe gesetzt
      (nie 0 → keine Division durch fast null)

    Speicher: ~2 Byte statt 8 Byte pro Parameter (Faktor 4 weniger). Kleine
    Tensoren (``min_8bit_size``) behalten 32-Bit-Zustände.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, block_size=256,
                 min_8bit_size=4096, dynamic_range=14.0):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                      block_size=block_size, min_8bit_size=min_8bit_size, dynamic_range=dynamic_range))

    # ------------------------------------------------------------ Quantisierung
    @staticmethod
    def _q_signed(x: torch.Tensor, block: int, R: float) -> tuple[torch.Tensor, torch.Tensor]:
        xb = _blocks(x.float(), block)
        hi = xb.abs().amax(1, keepdim=True).clamp_min(1e-30).log()
        mag = xb.abs().clamp_min(1e-38).log()
        k = ((mag - (hi - R)) / R * 126).round().add_(1).clamp_(1, 127)
        k = torch.where(mag < hi - R, torch.zeros_like(k), k)
        return (k * xb.sign()).to(torch.int8), hi

    @staticmethod
    def _dq_signed(q: torch.Tensor, hi: torch.Tensor, R: float, like: torch.Tensor) -> torch.Tensor:
        k = q.float()
        mag = torch.exp(hi - R + (k.abs() - 1) / 126 * R)
        out = torch.where(k == 0, torch.zeros_like(mag), mag * k.sign())
        return out.reshape(-1)[: like.numel()].view_as(like)

    @staticmethod
    def _q_unsigned(x: torch.Tensor, block: int, R: float) -> tuple[torch.Tensor, torch.Tensor]:
        xb = _blocks(x.float(), block)
        hi = xb.amax(1, keepdim=True).clamp_min(1e-38).log()
        k = ((xb.clamp_min(1e-38).log() - (hi - R)) / R * 255).round().clamp_(0, 255)
        return k.to(torch.uint8), hi

    @staticmethod
    def _dq_unsigned(q: torch.Tensor, hi: torch.Tensor, R: float, like: torch.Tensor) -> torch.Tensor:
        out = torch.exp(hi - R + q.float() / 255 * R)
        out = torch.where(hi <= math.log(1e-38) + 1, torch.zeros_like(out), out)  # Block komplett null
        return out.reshape(-1)[: like.numel()].view_as(like)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            block, R = group["block_size"], group["dynamic_range"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                small = p.numel() < group["min_8bit_size"]
                if "step" not in state:
                    state["step"] = 0
                    if small:
                        state["m"], state["v"] = torch.zeros_like(g), torch.zeros_like(g)
                if small:
                    m, v = state["m"], state["v"]
                elif "m_q" in state:
                    m = self._dq_signed(state["m_q"], state["m_hi"], R, g)
                    v = self._dq_unsigned(state["v_q"], state["v_hi"], 2 * R, g)
                else:
                    m, v = torch.zeros_like(g), torch.zeros_like(g)
                state["step"] += 1
                t = state["step"]
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v / (1 - b2 ** t)).sqrt_().add_(group["eps"])
                update = (m / (1 - b1 ** t)) / denom
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
                if not small:
                    state["m_q"], state["m_hi"] = self._q_signed(m, block, R)
                    state["v_q"], state["v_hi"] = self._q_unsigned(v, block, 2 * R)
        return loss

    def state_bytes(self) -> int:
        """Tatsächlicher Speicher aller Optimierer-Zustände in Bytes."""
        total = 0
        for st in self.state.values():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
        return total


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


_FUSED_DEVICES = ("cuda", "cpu", "mps", "xpu")


def _fast(cls: type, a: BaseConfig, groups: Any, **kw: Any) -> torch.optim.Optimizer:
    """Optimierer in der schnellsten verfügbaren Ausführung bauen.

    ``fused = None`` (Standard): fusionierter Kernel, wenn alle Parameter Gleitkomma auf einem unterstützten
    Gerät liegen (auf der CPU oft 3–5× schneller als die Einzel-Tensor-Variante), sonst ``foreach``, sonst die
    Standardausführung. ``fused = False`` erzwingt die Standardausführung.
    """
    fused = kw.pop("fused", None)
    params = [p for g in (groups if isinstance(groups, list) else [{"params": groups}])
              for p in (g["params"] if isinstance(g, dict) else [g])]
    devices = {p.device.type for p in params}
    floating = all(p.is_floating_point() for p in params)
    attempts: list[dict[str, Any]] = []
    if fused is None and floating and len(devices) == 1 and devices <= set(_FUSED_DEVICES):
        attempts += [{"fused": True}, {"foreach": True}]
    elif fused:
        attempts.append({"fused": True})
    attempts.append({})
    last: Exception | None = None
    for extra in attempts:
        try:
            return cls(groups, **kw, **extra)
        except (RuntimeError, TypeError, ValueError) as exc:  # Variante auf diesem Gerät/Typ nicht verfügbar
            last = exc
            if extra.get("fused") and fused:                   # ausdrücklich gewünscht → Fehler zeigen
                raise
    raise last  # type: ignore[misc]


def _kw(a: BaseConfig, *names: str) -> dict[str, Any]:
    out = {}
    for n in names:
        v = getattr(a, n)
        if n == "betas":
            v = tuple(v)
        out[n] = v
    return out


R("sgd", lambda a, g: _fast(torch.optim.SGD, a, g, nesterov=a.nesterov and a.momentum > 0,
                                **_kw(a, "lr", "momentum", "dampening", "weight_decay")),
  config=SGDConfig, tooltip=T(
      "Stochastischer Gradientenabstieg (mit Momentum/Nesterov).",
      "Einfach, wenig Speicher, generalisiert oft am besten (CNNs).",
      "Bildklassifikation (ResNet), wenn genug Tuning-Zeit vorhanden.",
  ))
R("adam", lambda a, g: _fast(torch.optim.Adam, a, g, **_kw(a, "lr", "betas", "eps", "weight_decay", "amsgrad",
                                                             "fused")),
  config=AdamConfig, tooltip=T(
      "Adam: adaptive Schrittweite pro Parameter.", "Robust, schnell, wenig Tuning.",
      "Allzweck; mit weight_decay besser AdamW nehmen."))
R("adamw", lambda a, g: _fast(torch.optim.AdamW, a, g, **_kw(a, "lr", "betas", "eps", "weight_decay", "amsgrad",
                                                               "fused")),
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


R("adamw8bit", lambda a, g: AdamW8bit(g, **_kw(a, "lr", "betas", "eps", "weight_decay", "block_size", "min_8bit_size",
                                                  "dynamic_range")),
  config=AdamW8bitConfig, aliases=("adam8bit",), level=Level.ADVANCED, tags=("speicher",), tooltip=T(
      "AdamW mit 8-Bit-Zuständen (blockweise logarithmisch quantisiert).",
      "Optimierer-Speicher sinkt von 8 auf ~2 Byte pro Parameter – bei großen Modellen oft der größte Posten.",
      "Große Modelle / Feintuning mit knappem Speicher; Verhalten praktisch wie AdamW.",
  ))


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

