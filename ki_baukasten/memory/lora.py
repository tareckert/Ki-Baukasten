"""
ki_baukasten.memory.lora
========================

**LoRA** (Low-Rank Adaptation, Hu et al. 2021), **DoRA** und – zusammen mit
der Quantisierung aus ``memory.quant`` – **QLoRA**.

Idee
----
Statt einer großen Gewichtsmatrix W (d_out × d_in) wird nur eine kleine
Korrektur mit Rang r gelernt::

    y = x·Wᵀ + (α / r) · dropout(x)·Aᵀ·Bᵀ        A: r × d_in,  B: d_out × r   (B startet bei 0)

* Trainierbar sind nur A und B → bei r = 8 oft < 1 % der Parameter.
* Gradienten und Optimierer-Zustände gibt es nur für A und B → riesige
  Speicherersparnis beim Feintuning.
* Die Basis W kann **quantisiert** sein (int8/nf4 → QLoRA): dann liegt das
  große Modell in ½–1 Byte pro Gewicht vor und trotzdem wird trainiert.
* ``merge_lora`` rechnet die Korrektur nach dem Training in W ein
  (keine Zusatzkosten bei der Inferenz; bei quantisierter Basis wird neu quantisiert).

DoRA (Liu et al. 2024): zerlegt W in Richtung und Betrag; LoRA passt die
Richtung an, ein lernbarer Vektor ``m`` den Betrag pro Ausgabekanal – oft
näher am vollen Feintuning.

Umsetzung
---------
Wie beim Checkpointing wird die **Klasse** der Ziel-Schicht ausgetauscht
(``nn.Linear`` → LoRA-Unterklasse). Bestehende Parameternamen bleiben
erhalten; neu hinzu kommen ``lora_A``, ``lora_B`` (und ``lora_magnitude``).
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, Level, Param, T, select_indices
from .quant import QuantLinear, quantize_weight


class LoRAConfig(BaseConfig):
    __title__ = "LoRA / QLoRA (Adapter-Feintuning)"
    __tooltip__ = T(
        "Kleine, trainierbare Niedrig-Rang-Adapter auf linearen Schichten; die Basisgewichte bleiben eingefroren.",
        "Feintuning mit einem Bruchteil des Speichers (Gradienten + Optimierer nur für die Adapter).",
        "Anpassen großer vortrainierter Modelle; mit memory.quantization = 'nf4'/'int8' → QLoRA.",
    )
    enabled = Param(False, tooltip=T(
        "LoRA-Adapter einbauen.", "Aktiviert Adapter-Feintuning.", "Feintuning großer Modelle."))
    rank = Param(8, min=1, tooltip=T(
        "Rang r der Korrektur.", "Größer = ausdrucksstärker, mehr Parameter.", "4–64; Standard 8–16."))
    alpha = Param(16.0, min=0.0, tooltip=T(
        "Skalierung α (effektiver Faktor α/r).", "Entkoppelt Lernrate vom Rang.", "Meist 2·rank."))
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout auf die Adapter-Eingabe.", "Regularisierung bei kleinen Datensätzen.", "0.05–0.1."))
    targets = Param("auto", tooltip=T(
        "Welche Schichten Adapter bekommen: 'auto' = alle linearen Schichten in den Blöcken, sonst Regex auf "
        "Modulnamen.",
        "Weniger Ziele = weniger Parameter; Attention-Projektionen sind die wichtigsten.",
        "'auto', 'q_proj|v_proj' (Original-LoRA), '.*proj' (alle Projektionen).",
    ))
    layers = Param("*", tooltip=T(
        "Nur in diesen Blöcken (Selektor wie bei layer_overrides; nur mit targets='auto').",
        "Z. B. nur die oberen Schichten anpassen.", "'*', '-4:' (letzte vier Blöcke)."), level=Level.ADVANCED)
    exclude = Param("lm_head", type=str, nullable=True, tooltip=T(
        "Regex: diese Module nie mit Adaptern versehen.", "Ausgabeschicht bleibt meist unverändert.", "lm_head."),
        level=Level.ADVANCED)
    freeze_base = Param(True, tooltip=T(
        "Alle Nicht-Adapter-Parameter einfrieren.", "Der eigentliche Speichervorteil von LoRA.", "an."))
    train_also = Param(None, type=str, nullable=True, tooltip=T(
        "Regex: diese Parameter trotz freeze_base trainieren (z. B. Normen, Ausgabekopf).",
        "Kleine Zusatzteile trainieren oft spürbar besser mit.", "'norm', 'lm_head|head'."), level=Level.ADVANCED)
    init = Param("kaiming", choices={
        "kaiming": "A ~ Kaiming-uniform, B = 0 (Original-LoRA; Start = Basismodell)",
        "gaussian": "A ~ N(0, 1/r), B = 0",
        "pissa": "A, B aus den Hauptkomponenten (SVD) von W, Basis = Rest (PiSSA; schnellere Konvergenz)",
    }, tooltip=T("Initialisierung der Adapter.", "Bestimmt den Startpunkt.", "kaiming; pissa für schnelleres Lernen."),
        level=Level.EXPERT)
    use_dora = Param(False, tooltip=T(
        "DoRA: zusätzlicher lernbarer Betrag pro Ausgabekanal.",
        "Oft näher an vollem Feintuning, etwas teurer.", "Wenn LoRA zu schwach ist."), level=Level.EXPERIMENTAL)


# ===========================================================================
# LoRA-Schicht per Klassentausch
# ===========================================================================

_LORA_CLASSES: dict[type, type] = {}


def _base_weight(module: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(module, QuantLinear):
        return module.dequantize(dtype)
    return module.weight.to(dtype)


def _lora_class(cls: type) -> type:
    if cls in _LORA_CLASSES:
        return _LORA_CLASSES[cls]

    def forward(self, x):
        if self.lora_magnitude is None:
            h = F.dropout(x, self.lora_dropout, self.training)
            return _lora_out(super(new_cls, self).forward(x), h, self.lora_A, self.lora_B, self.lora_scaling)
        # DoRA: W' = m · (W + s·B·A) / ‖W + s·B·A‖  (Zeilennorm = Betrag pro Ausgabekanal)
        A, B = self.lora_A, self.lora_B
        w = _base_weight(self, A.dtype) + self.lora_scaling * (B @ A)
        w = w * (self.lora_magnitude / w.norm(dim=1).clamp_min(1e-8)).unsqueeze(1)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        parent = super(new_cls, self).extra_repr()
        return f"{parent}, lora_rank={self.lora_A.shape[0]}, dora={self.lora_magnitude is not None}"

    new_cls = type(cls.__name__, (cls,), {"forward": forward, "extra_repr": extra_repr, "_ki_lora": True,
                                          "_ki_lora_base": cls, "__module__": cls.__module__})
    _LORA_CLASSES[cls] = new_cls
    return new_cls


def _lora_out(base_out: torch.Tensor, h: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scaling: float):
    """Basis-Ausgabe + (α/r)·h·Aᵀ·Bᵀ (Rechnung im Datentyp der Basis-Ausgabe, autocast-verträglich)."""
    dt = base_out.dtype
    return base_out + F.linear(F.linear(h.to(dt), A.to(dt)), B.to(dt)) * scaling


def _attach(module: nn.Module, cfg: LoRAConfig) -> None:
    out_f, in_f = module.out_features, module.in_features
    r = cfg.rank
    dev = module.qweight.device if isinstance(module, QuantLinear) else module.weight.device
    A = torch.empty(r, in_f, device=dev)
    B = torch.zeros(out_f, r, device=dev)
    if cfg.init == "gaussian":
        nn.init.normal_(A, std=1.0 / r)
    elif cfg.init == "pissa":
        w = _base_weight(module, torch.float32)
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        sq = S[:r].sqrt()
        scaling = cfg.alpha / r
        B.copy_(U[:, :r] * sq / math.sqrt(scaling))
        A.copy_(sq.unsqueeze(1) * Vh[:r] / math.sqrt(scaling))
        rest = w - scaling * (B @ A)
        if isinstance(module, QuantLinear):
            data = quantize_weight(rest, module.fmt, module.group_size)
            module.qweight.copy_(data["qweight"])
            module.scales.copy_(data["scales"])
        else:
            with torch.no_grad():
                module.weight.copy_(rest)
    else:
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    module.lora_A = nn.Parameter(A)
    module.lora_B = nn.Parameter(B)
    module.lora_scaling = cfg.alpha / r
    module.lora_dropout = cfg.dropout
    if cfg.use_dora:
        w = _base_weight(module, torch.float32) + module.lora_scaling * (B @ A)
        module.lora_magnitude = nn.Parameter(w.norm(dim=1).detach())
    else:
        module.lora_magnitude = None
    module.__class__ = _lora_class(type(module))


def lora_targets(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Namen der linearen Schichten, die Adapter bekommen."""
    exclude = re.compile(cfg.exclude) if cfg.exclude else None
    linear = (nn.Linear, QuantLinear)
    if cfg.targets == "auto":
        blocks = model.checkpointable_blocks() if hasattr(model, "checkpointable_blocks") else []
        if blocks:
            chosen = {id(blocks[i]) for i in select_indices(cfg.layers, len(blocks))}
            prefixes = [n for n, m in model.named_modules() if id(m) in chosen]
            names = [n for n, m in model.named_modules() if isinstance(m, linear)
                     and any(n.startswith(p + ".") for p in prefixes)]
        else:
            names = [n for n, m in model.named_modules() if isinstance(m, linear)]
    else:
        rx = re.compile(cfg.targets)
        names = [n for n, m in model.named_modules() if isinstance(m, linear) and rx.search(n)]
    return [n for n in names if not (exclude and exclude.search(n))]


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Baut Adapter ein (und friert ggf. die Basis ein). Liefert die Namen der angepassten Schichten."""
    names = lora_targets(model, cfg)
    if not names:
        raise ValueError(f"LoRA: keine lineare Schicht passt auf targets='{cfg.targets}'.")
    modules = dict(model.named_modules())
    for n in names:
        if not getattr(modules[n], "_ki_lora", False):
            _attach(modules[n], cfg)
    if cfg.freeze_base:
        also = re.compile(cfg.train_also) if cfg.train_also else None
        for pname, p in model.named_parameters():
            is_adapter = ".lora_" in pname or pname.startswith("lora_")
            p.requires_grad_(is_adapter or bool(also and also.search(pname)))
    return names


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Rechnet alle Adapter in die Basisgewichte ein und entfernt sie. Liefert die Anzahl Schichten."""
    n = 0
    for module in list(model.modules()):
        if not getattr(module, "_ki_lora", False):
            continue
        A, B, s = module.lora_A, module.lora_B, module.lora_scaling
        w = _base_weight(module, torch.float32) + s * (B.float() @ A.float())
        if module.lora_magnitude is not None:
            w = w * (module.lora_magnitude / w.norm(dim=1).clamp_min(1e-8)).unsqueeze(1)
        base_cls = type(module)._ki_lora_base
        if isinstance(module, QuantLinear):
            data = quantize_weight(w, module.fmt, module.group_size)
            module.qweight.copy_(data["qweight"])
            module.scales.copy_(data["scales"])
        else:
            module.weight.copy_(w.to(module.weight.dtype))
        del module.lora_A, module.lora_B, module.lora_magnitude
        for attr in ("lora_scaling", "lora_dropout"):
            module.__dict__.pop(attr, None)
        module.__class__ = base_cls
        n += 1
    return n


def lora_parameter_count(model: nn.Module) -> tuple[int, int]:
    """(Adapter-Parameter, alle Gewichte inkl. quantisierter Basisgewichte)."""
    total = sum(p.numel() for p in model.parameters())
    total += sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, QuantLinear))
    lora = sum(p.numel() for n, p in model.named_parameters() if ".lora_" in n)
    return lora, total
