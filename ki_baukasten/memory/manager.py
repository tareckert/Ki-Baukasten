"""
ki_baukasten.memory.manager
===========================

``MemoryManager`` setzt eine ``MemoryConfig`` am Modell und im Trainer um.

Reihenfolge in ``prepare_model``
--------------------------------
1. Datentyp der Gewichte (``param_dtype``)
2. Fake-Quantisierung (Quantization-Aware Training)
3. Gradient-Checkpointing der gewählten Blöcke
4. Platzierung: Streaming (trainierbar), Offloading/Swapping **oder** komplett auf das Gerät
5. Speicherlayout ``channels_last``
6. ``torch.compile``

Zur Laufzeit liefert der Manager den passenden ``autocast()``-Kontext
(gemischte Präzision + Aktivierungs-Offload) und den ``GradScaler``.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Iterator

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

from ..core import ConfigError, select_indices
from .config import MemoryConfig
from .offload import OffloadManager, select_offload_blocks
from .streaming import enable_block_streaming

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64}


def resolve_device(spec: str) -> torch.device:
    """'auto' → cuda > mps > cpu; sonst die angegebene Gerätebezeichnung."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError(f"Gerät '{spec}' angefordert, aber CUDA ist nicht verfügbar.", "memory.device")
    return dev


# ===========================================================================
# Gradient-Checkpointing
# ===========================================================================

_CKPT_CLASSES: dict[type, type] = {}


def _checkpointed_class(cls: type) -> type:
    """Erzeugt (einmalig) eine Unterklasse, deren forward per Checkpointing läuft.

    Wir tauschen die *Klasse* des Blocks statt ihn in eine Hülle zu stecken –
    so bleiben Parameternamen (state_dict, Tensor-Chirurgie-Regeln) unverändert
    und ``copy.deepcopy`` funktioniert korrekt.
    """
    if cls in _CKPT_CLASSES:
        return _CKPT_CLASSES[cls]

    def forward(self, *args, **kwargs):
        parent_forward = super(new_cls, self).forward
        if self.training and torch.is_grad_enabled():
            return checkpoint(parent_forward, *args, use_reentrant=False, **kwargs)
        return parent_forward(*args, **kwargs)

    new_cls = type(cls.__name__, (cls,), {"forward": forward, "_ki_checkpointed": True, "__module__": cls.__module__})
    _CKPT_CLASSES[cls] = new_cls
    return new_cls


def enable_gradient_checkpointing(model: nn.Module, selector: str = "*") -> int:
    """Aktiviert Checkpointing für die ausgewählten Blöcke; liefert deren Anzahl."""
    blocks = model.checkpointable_blocks() if hasattr(model, "checkpointable_blocks") else []
    if not blocks:
        warnings.warn(f"{type(model).__name__} meldet keine checkpointfähigen Blöcke – Gradient-Checkpointing wirkungslos.")
        return 0
    chosen = select_indices(selector, len(blocks))
    for i in chosen:
        b = blocks[i]
        if not getattr(b, "_ki_checkpointed", False):
            b.__class__ = _checkpointed_class(type(b))
    return len(chosen)


# ===========================================================================
# Fake-Quantisierung (Quantization-Aware Training)
# ===========================================================================

class FakeQuantize(nn.Module):
    """Parametrisierung: Gewichte symmetrisch pro Ausgabekanal auf n Bit runden (Straight-Through)."""

    def __init__(self, bits: int):
        super().__init__()
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        flat = w.reshape(w.shape[0], -1)
        scale = flat.abs().amax(1, keepdim=True).clamp_min(1e-8) / self.qmax
        q = (flat / scale).round().clamp(-self.qmax - 1, self.qmax) * scale
        return (flat + (q - flat).detach()).reshape(w.shape)


def apply_fake_quant(model: nn.Module, bits: int, include_conv: bool) -> int:
    types: tuple[type, ...] = (nn.Linear,) + ((nn.modules.conv._ConvNd,) if include_conv else ())
    n = 0
    for module in model.modules():
        if isinstance(module, types) and not parametrize.is_parametrized(module, "weight"):
            parametrize.register_parametrization(module, "weight", FakeQuantize(bits))
            n += 1
    return n


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Dynamische int8-Quantisierung linearer Schichten (CPU-Inferenz).

    Nutzt ``torch.ao.quantization.quantize_dynamic``. PyTorch kündigt an, diese
    API zugunsten des Pakets ``torchao`` zu entfernen – ist sie nicht mehr
    vorhanden, gibt es einen verständlichen Hinweis statt eines kryptischen Fehlers.
    """
    quant = getattr(getattr(torch, "ao", None), "quantization", None)
    if quant is None or not hasattr(quant, "quantize_dynamic"):
        raise ImportError("dynamic_int8 benötigt torch.ao.quantization (in dieser PyTorch-Version nicht vorhanden). "
                          "Alternative: 'pip install torchao' und dessen quantize_-API, oder quantization='fake_quant'.")
    model = model.cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        return quant.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


# ===========================================================================
# Manager
# ===========================================================================

class MemoryManager:
    """Setzt ``MemoryConfig`` um (siehe Modul-Dokumentation)."""

    def __init__(self, cfg: MemoryConfig | None = None):
        self.cfg = (cfg or MemoryConfig()).validate()
        self.device = resolve_device(self.cfg.device)
        self.offloader: OffloadManager | None = None
        self.streams: list = []
        c = self.cfg
        if c.cpu_threads:
            torch.set_num_threads(c.cpu_threads)
        torch.set_float32_matmul_precision(c.matmul_precision)
        if c.max_memory_fraction and self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(c.max_memory_fraction, self.device)

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.cfg.mixed_precision)

    def prepare_model(self, model: nn.Module, training: bool = True) -> nn.Module:
        c = self.cfg
        model = model.to(dtype=DTYPES[c.param_dtype]) if c.param_dtype != "float32" else model
        if c.quantization == "fake_quant":
            apply_fake_quant(model, c.quant_bits, c.quant_targets == "linear_conv")
        elif c.quantization == "dynamic_int8":
            if training:
                warnings.warn("dynamic_int8 ist nur für Inferenz – im Training ignoriert (nutze fake_quant für QAT).")
            else:
                return quantize_dynamic_int8(model)
        if c.gradient_checkpointing:
            enable_gradient_checkpointing(model, c.checkpoint_layers)
        if c.offload.mode == "stream":
            blocks = [m for _, m in select_offload_blocks(model, c.offload)]
            self.streams = enable_block_streaming(model, blocks, self.device, c.offload.pin_memory, c.offload.prefetch)
        elif c.offload.mode != "none":
            self.offloader = OffloadManager(model, c.offload, self.device)
            self.offloader.attach()
            if training and self.offloader.trainable_offloaded():
                raise ConfigError(
                    f"Offload-Modus '{c.offload.mode}' ist nur für Inferenz oder eingefrorene Blöcke, "
                    f"trainierbar sind aber: {self.offloader.trainable_offloaded()[:3]} … "
                    "→ memory.offload.mode='stream' (trainierbares Auslagern in den RAM) verwenden "
                    "oder die Blöcke einfrieren.",
                    "memory.offload.mode",
                )
        else:
            model = model.to(self.device)
        if c.channels_last:
            model = model.to(memory_format=torch.channels_last)
        if c.compile:
            model = torch.compile(model, mode=c.compile_mode)  # type: ignore[assignment]
        return model

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        """Kontext für Forward-Berechnungen (gemischte Präzision, Aktivierungs-Offload)."""
        with contextlib.ExitStack() as stack:
            dtype = self.autocast_dtype
            if dtype is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=dtype))
            if self.cfg.activation_offload:
                stack.enter_context(torch.autograd.graph.save_on_cpu(pin_memory=self.device.type == "cuda"))
            yield

    def grad_scaler(self) -> torch.amp.GradScaler | None:
        mode = self.cfg.grad_scaler
        use = mode == "on" or (mode == "auto" and self.cfg.mixed_precision == "fp16")
        if not use:
            return None
        return torch.amp.GradScaler(self.device.type)

    def after_step(self, step: int) -> None:
        n = self.cfg.empty_cache_every_n_steps
        if n and step % n == 0 and self.device.type == "cuda":
            torch.cuda.empty_cache()
