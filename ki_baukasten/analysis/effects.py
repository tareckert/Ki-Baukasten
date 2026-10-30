"""
ki_baukasten.analysis.effects
=============================

**Wirkungsprüfung**: findet geänderte Einstellungen, die gerade **nichts bewirken**, weil der
zugehörige Schalter aus ist – z. B. ``memory.lora.rank = 16`` ohne eingeschaltetes LoRA, EMA-Feinheiten
ohne EMA oder ``training.epochs`` neben einem gesetzten ``training.max_steps``.

Ohne diese Prüfung ändert man eine Einstellung, trainiert – und wundert sich, dass sich nichts
ändert. ``Project.train()`` meldet solche Einstellungen vorab, ``Project.inactive_settings()`` listet
sie mit Grund und Abhilfe::

    for s in p.inactive_settings():
        print(s)      # memory.lora.rank = 16 wirkt nicht: LoRA ist aus → memory.lora.enabled = True
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable

from ..core.config import BaseConfig, flatten_dict

Rule = tuple[tuple[str, ...], tuple[str, ...], Callable[[BaseConfig], bool], str, str]


@dataclass(frozen=True)
class InactiveSetting:
    path: str
    value: Any
    reason: str
    fix: str

    def __str__(self) -> str:
        return f"{self.path} = {self.value!r} wirkt nicht: {self.reason} → {self.fix}"


def _get(cfg: BaseConfig, path: str, default: Any = None) -> Any:
    try:
        return cfg.get(path)
    except Exception:  # noqa: BLE001 - Einstellung gibt es in dieser Konfiguration nicht
        return default


def _images(cfg: BaseConfig) -> bool:
    from ..techniques import _images as images

    return images(cfg)


def _grids(cfg: BaseConfig) -> bool:
    """Bild-Augmentierung wirkt auf jede 4D-Eingabe [N, C, H, W]: Bilder und Audio-Spektrogramme."""
    return _images(cfg) or _get(cfg, "data.name") == "audio"


def _tokens(cfg: BaseConfig) -> bool:
    from ..techniques import _tokens as tokens

    return tokens(cfg)


_QUANT_WEIGHTS = ("int8", "int4", "nf4", "pal4")

# (Muster, Ausnahmen, aktiv?, Grund wenn nicht aktiv, Abhilfe)
RULES: list[Rule] = [
    (("training.ema.*",), ("training.ema.enabled",), lambda c: c.training.ema.enabled,
     "EMA ist aus", "training.ema.enabled = True (Technik 'ema')"),
    (("training.privacy.*",), ("training.privacy.enabled",), lambda c: c.training.privacy.enabled,
     "DP-SGD ist aus", "training.privacy.enabled = True (Technik 'dp_sgd')"),
    (("training.adversarial.*",), ("training.adversarial.method",),
     lambda c: c.training.adversarial.method != "keine",
     "adversariales Training ist aus", "training.adversarial.method = 'fgsm' oder 'pgd'"),
    (("training.stability.spike_*", "training.stability.rollback_*"), ("training.stability.spike_guard",),
     lambda c: c.training.stability.spike_guard, "Spike-Schutz ist aus", "training.stability.spike_guard = True"),
    (("training.stability.rollback_*",), (),
     lambda c: c.training.stability.spike_action == "rollback",
     "Zurückrollen ist nicht die gewählte Reaktion", "training.stability.spike_action = 'rollback'"),
    (("training.curriculum.*",), ("training.curriculum.strategy",),
     lambda c: c.training.curriculum.strategy != "none",
     "Curriculum ist aus", "training.curriculum.strategy = 'loss' oder 'key'"),
    (("training.gradient_clipping.max_norm", "training.gradient_clipping.norm_type"), (),
     lambda c: c.training.gradient_clipping.mode == "norm",
     "Clipping nicht nach Gesamtnorm", "training.gradient_clipping.mode = 'norm'"),
    (("training.gradient_clipping.clip_value",), (), lambda c: c.training.gradient_clipping.mode == "value",
     "Clipping nicht elementweise", "training.gradient_clipping.mode = 'value'"),
    (("training.gradient_clipping.agc_ratio",), (), lambda c: c.training.gradient_clipping.mode == "agc",
     "kein adaptives Clipping", "training.gradient_clipping.mode = 'agc'"),
    (("training.optimizer_extras.lookahead_alpha",), (),
     lambda c: c.training.optimizer_extras.lookahead_k is not None,
     "Lookahead ist aus", "training.optimizer_extras.lookahead_k = 5"),
    (("training.optimizer_extras.grokfast_lambda",), (),
     lambda c: c.training.optimizer_extras.grokfast_alpha is not None,
     "Grokfast ist aus", "training.optimizer_extras.grokfast_alpha = 0.98"),
    (("training.regularization.gradient_noise_decay",), (),
     lambda c: c.training.regularization.gradient_noise_std > 0,
     "Gradientenrauschen ist aus", "training.regularization.gradient_noise_std > 0"),
    (("training.epochs",), (), lambda c: c.training.max_steps is None,
     "training.max_steps ist gesetzt und hat Vorrang", "training.max_steps = None"),
    (("training.augmentation.hflip", "training.augmentation.vflip", "training.augmentation.crop_padding",
      "training.augmentation.random_erasing", "training.augmentation.cutmix_alpha"), (), _grids,
     "nur für Bilder/Spektrogramme", "Bild-Augmentierung bei anderen Daten weglassen"),
    (("training.augmentation.token_noise",), (), _tokens,
     "nur für Token-Daten", "bei anderen Daten weglassen"),
    (("training.distributed.*",), ("training.distributed.strategy",),
     lambda c: c.training.distributed.strategy != "none",
     "verteiltes Training ist aus", "training.distributed.strategy = 'ddp' (Start mit torchrun)"),
    (("training.distributed.fsdp_*",), (), lambda c: str(c.training.distributed.strategy).startswith("fsdp"),
     "gilt nur für FSDP", "training.distributed.strategy = 'fsdp'"),
    (("memory.lora.*",), ("memory.lora.enabled",), lambda c: c.memory.lora.enabled,
     "LoRA ist aus", "memory.lora.enabled = True (Technik 'lora')"),
    (("memory.offload.*",), ("memory.offload.mode",), lambda c: c.memory.offload.mode != "none",
     "Auslagern ist aus", "memory.offload.mode = 'cpu' | 'disk' | 'stream' | 'split'"),
    (("memory.offload.split_device",), (), lambda c: c.memory.offload.mode == "split",
     "gilt nur für Blöcke aufteilen", "memory.offload.mode = 'split'"),
    (("memory.quant_skip", "memory.quant_only", "memory.quant_compute_dtype"), (),
     lambda c: c.memory.quantization in _QUANT_WEIGHTS,
     "keine Gewichts-Quantisierung gewählt", "memory.quantization = 'int8' | 'int4' | 'nf4'"),
    (("memory.quant_group_size",), (),
     lambda c: c.memory.quantization in _QUANT_WEIGHTS + ("torchao_int8", "torchao_int4"),
     "keine Gewichts-Quantisierung gewählt", "memory.quantization = 'int8' | 'int4' | 'nf4'"),
    (("memory.quant_bits", "memory.quant_targets"), (), lambda c: c.memory.quantization == "fake_quant",
     "gilt nur für Quantization-Aware Training", "memory.quantization = 'fake_quant'"),
    (("memory.compile_mode",), (), lambda c: c.memory.compile,
     "torch.compile ist aus", "memory.compile = True"),
    (("memory.checkpoint_layers",), (), lambda c: c.memory.gradient_checkpointing,
     "Gradient-Checkpointing ist aus", "memory.gradient_checkpointing = True"),
    (("inference.tta_*",), (), lambda c: c.inference.uncertainty in ("tta", "beides"),
     "Test-Time Augmentation ist aus", "inference.uncertainty = 'tta' (Technik 'tta')"),
    (("inference.samples",), (), lambda c: c.inference.uncertainty != "keine",
     "keine Unsicherheitsschätzung gewählt", "inference.uncertainty = 'mc_dropout' oder 'tta'"),
]


def _changed(cfg: BaseConfig, reference: BaseConfig | None) -> dict[str, Any]:
    """Vom Standard (bzw. von ``reference``, z. B. der Vorlage) abweichende Einstellungen."""
    if reference is None:
        return cfg.changed()
    mine, base = flatten_dict(cfg.to_dict()), flatten_dict(reference.to_dict())
    return {k: v for k, v in mine.items() if base.get(k, object()) != v}


def inactive_settings(cfg: BaseConfig, reference: BaseConfig | None = None) -> list[InactiveSetting]:
    """Geänderte Einstellungen, die wegen eines ausgeschalteten Schalters nichts bewirken.

    ``reference``: nur Abweichungen hiervon melden (z. B. von der Vorlage); Standard: von den Grundwerten.
    """
    out: list[InactiveSetting] = []
    seen: set[str] = set()
    changed = _changed(cfg, reference)
    for patterns, exceptions, active, reason, fix in RULES:
        hits = [p for p in changed if p not in seen and p not in exceptions
                and any(fnmatchcase(p, pat) for pat in patterns)]
        if not hits:
            continue
        try:
            if active(cfg):
                continue
        except Exception:  # noqa: BLE001 - Regel passt nicht zu dieser Konfiguration
            continue
        for path in hits:
            seen.add(path)
            out.append(InactiveSetting(path, _get(cfg, path, changed[path]), reason, fix))
    return out
