"""
ki_baukasten.memory.hardware
============================

**Hardware-Automatik** – erkennt das Rechengerät und wählt die schnellsten sinnvollen Einstellungen,
damit eine mit dem Baukasten gebaute KI auf derselben Hardware so schnell läuft wie eine von Hand
optimierte. Jede Einstellung steht auf ``auto`` und kann jederzeit fest gesetzt werden.

==========================  ========================================================================
Einstellung (``auto``)      Wahl
==========================  ========================================================================
mixed_precision             NVIDIA ab Ampere (8.x): bf16 · Volta/Turing (7.x): fp16 mit Loss-Scaling ·
                            ältere GPUs, Apple (MPS), CPU: aus (dort meist nicht schneller)
matmul_precision            NVIDIA ab Ampere: TF32 (``high``) · sonst volle Genauigkeit
cudnn_benchmark             NVIDIA + Faltungsnetz: an (schnellster Faltungsalgorithmus je Form)
channels_last               NVIDIA + Faltungsnetz + gemischte Präzision: an (Tensor Cores)
pin_memory (training)       NVIDIA: an (schnellere Übertragung zur GPU)
Optimierer                  fusionierter Kernel, wenn das Gerät ihn kann (CPU und GPU)
Attention                   ``scaled_dot_product_attention`` (FlashAttention/Memory-Efficient auf GPU)
==========================  ========================================================================

``Project.hardware_report()`` zeigt Gerät, Fähigkeiten und was gewählt wurde.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class HardwareInfo:
    device: str
    name: str
    capability: tuple[int, int] | None = None
    bf16: bool = False
    tf32: bool = False
    memory_gb: float | None = None
    cpu_cores: int = 1
    ram_gb: float | None = None
    notes: list[str] = field(default_factory=list)


def _ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def detect(device: torch.device | str | None = None) -> HardwareInfo:
    """Beschreibt das Rechengerät (ohne etwas zu verändern)."""
    from .manager import resolve_device

    dev = resolve_device("auto") if device is None else torch.device(device)
    cores = os.cpu_count() or 1
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        cap = (props.major, props.minor)
        return HardwareInfo("cuda", props.name, cap, bf16=torch.cuda.is_bf16_supported(), tf32=cap >= (8, 0),
                            memory_gb=props.total_memory / 2**30, cpu_cores=cores, ram_gb=_ram_gb())
    if dev.type == "mps":
        return HardwareInfo("mps", "Apple GPU (MPS)", cpu_cores=cores, ram_gb=_ram_gb())
    return HardwareInfo("cpu", f"CPU ({cores} Kerne)", cpu_cores=cores, ram_gb=_ram_gb())


def _has_conv(model: nn.Module | None) -> bool:
    return model is not None and any(isinstance(m, nn.modules.conv._ConvNd) for m in model.modules())


def resolve(cfg: Any, device: torch.device, model: nn.Module | None = None) -> dict[str, Any]:
    """Löst alle ``auto``-Werte von ``MemoryConfig`` für ``device`` auf → {einstellung: (wert, begründung)}."""
    hw = detect(device)
    out: dict[str, Any] = {}
    cuda = hw.device == "cuda"
    if cfg.mixed_precision == "auto":
        if cuda and hw.bf16 and hw.capability and hw.capability >= (8, 0):
            out["mixed_precision"] = ("bf16", f"{hw.name}: bf16-Tensor-Cores")
        elif cuda and hw.capability and hw.capability >= (7, 0):
            out["mixed_precision"] = ("fp16", f"{hw.name}: fp16-Tensor-Cores (mit Loss-Scaling)")
        else:
            out["mixed_precision"] = ("none", "Gerät ohne schnelle 16-Bit-Rechenwerke – volle Präzision")
    else:
        out["mixed_precision"] = (cfg.mixed_precision, "fest eingestellt")
    if cfg.matmul_precision == "auto":
        out["matmul_precision"] = (("high", "TF32 (Ampere+)") if cuda and hw.tf32
                                   else ("highest", "volle fp32-Genauigkeit"))
    else:
        out["matmul_precision"] = (cfg.matmul_precision, "fest eingestellt")
    conv = _has_conv(model)
    if cfg.cudnn_benchmark is None:
        out["cudnn_benchmark"] = ((True, "Faltungsnetz auf NVIDIA") if cuda and conv else (False, "nicht nötig"))
    else:
        out["cudnn_benchmark"] = (cfg.cudnn_benchmark, "fest eingestellt")
    if cfg.channels_last is None:
        fast = out["mixed_precision"][0] != "none"
        out["channels_last"] = ((True, "Faltungsnetz + Tensor Cores") if cuda and conv and fast
                                else (False, "nicht nötig"))
    else:
        out["channels_last"] = (cfg.channels_last, "fest eingestellt")
    return out


def report_text(hw: HardwareInfo, settings: dict[str, Any]) -> str:
    lines = [f"Hardware: {hw.name} ({hw.device})"]
    if hw.capability:
        lines[0] += f", Compute Capability {hw.capability[0]}.{hw.capability[1]}"
    extra = []
    if hw.memory_gb:
        extra.append(f"{hw.memory_gb:.1f} GB Gerätespeicher")
    if hw.ram_gb:
        extra.append(f"{hw.ram_gb:.1f} GB RAM")
    extra.append(f"{hw.cpu_cores} CPU-Kerne")
    lines.append("  " + " · ".join(extra))
    lines.append("Automatisch gewählt (memory.* = 'auto' bzw. None):")
    for key, (value, why) in settings.items():
        lines.append(f"  {key:<18} = {value!s:<8}  {why}")
    lines.append("  Optimierer: fusioniert, wenn verfügbar · Attention: scaled_dot_product_attention")
    return "\n".join(lines)
