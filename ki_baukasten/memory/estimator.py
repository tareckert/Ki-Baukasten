"""
ki_baukasten.memory.estimator
=============================

Speicherschätzung **vor** dem Training – inklusive Empfehlungen.

Vorgehen
--------
* **Parameter, Gradienten, Optimierer-Zustände, EMA**: exakt aus Parameterzahl,
  Datentyp und Optimierer berechnet.
* **Aktivierungen**: *gemessen* – ein Probe-Forward/Backward mit kleiner
  Batchgröße zählt über ``saved_tensors_hooks`` jedes Byte, das Autograd für
  den Backward aufhebt. Das Ergebnis wird linear auf die echte Batchgröße
  hochgerechnet. Das funktioniert für *jedes* Modell (auch frei gebaute).
* **Empfehlungen**: Passt das Training nicht in das Budget, schlägt die
  Schätzung konkrete Einstellungen vor (bf16, Checkpointing, Akkumulation,
  speichersparende Optimierer, Offloading ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..core import format_number
from .config import MemoryConfig

_OPT_STATES = {  # zusätzliche fp32-Kopien je trainierbarem Parameter
    "adam": 2, "adamw": 2, "nadam": 2, "radam": 2, "adamax": 2, "adadelta": 2,
    "sgd": 1, "rmsprop": 1, "adagrad": 1, "lion": 1, "muon": 1, "adafactor": 0.1,
}
_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{format_number(n, 1)} {unit}"
        n /= 1024
    return f"{n} B"


@dataclass
class MemoryReport:
    parameters: int
    trainable: int
    weights_bytes: float
    gradients_bytes: float
    optimizer_bytes: float
    ema_bytes: float
    activations_bytes: float | None
    batch_size: int
    budget_bytes: float | None
    recommendations: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> float:
        return (self.weights_bytes + self.gradients_bytes + self.optimizer_bytes + self.ema_bytes
                + (self.activations_bytes or 0.0))

    def fits(self) -> bool | None:
        return None if self.budget_bytes is None else self.total_bytes <= self.budget_bytes

    def __str__(self) -> str:
        lines = [
            "Speicherschätzung (Training)",
            f"  Parameter:            {format_number(self.parameters)} ({format_number(self.trainable)} trainierbar)",
            f"  Gewichte:             {_fmt(self.weights_bytes)}",
            f"  Gradienten:           {_fmt(self.gradients_bytes)}",
            f"  Optimierer-Zustände:  {_fmt(self.optimizer_bytes)}",
            f"  EMA-Kopie:            {_fmt(self.ema_bytes)}",
            f"  Aktivierungen:        {_fmt(self.activations_bytes) if self.activations_bytes is not None else 'nicht gemessen'}"
            f"  (Batch {self.batch_size})",
            "  ─────────────────────────────",
            f"  Gesamt (ca.):         {_fmt(self.total_bytes)}",
        ]
        if self.budget_bytes is not None:
            verdict = "passt ✓" if self.fits() else "passt NICHT ✗"
            lines.append(f"  Budget:               {_fmt(self.budget_bytes)} → {verdict}")
        if self.recommendations:
            lines.append("  Empfehlungen:")
            lines += [f"    • {r}" for r in self.recommendations]
        return "\n".join(lines)


def _measure_activations(model: nn.Module, example: Any, autocast_dtype: torch.dtype | None) -> float:
    """Summe der von Autograd gespeicherten Tensor-Bytes für einen Forward+Backward."""
    from ..training.paradigms import call_model, main_output  # spät importieren (Zyklen vermeiden)

    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        try:
            key = t.untyped_storage().data_ptr()
            seen[key] = max(seen.get(key, 0), t.untyped_storage().nbytes())
        except (RuntimeError, NotImplementedError):
            seen[id(t)] = t.numel() * t.element_size()
        return t

    was = model.training
    model.train()
    device = next(model.parameters()).device
    from ..training.data_utils import to_device

    example = to_device(example, device)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype else torch.enable_grad()
            with ctx:
                out = main_output(call_model(model, example))
            out.float().sum().backward()
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was)
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    return float(sum(v for k, v in seen.items() if k not in params))


def detect_budget(device: torch.device) -> float | None:
    if device.type == "cuda":
        return float(torch.cuda.get_device_properties(device).total_memory)
    try:
        import os

        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def estimate_memory(model: nn.Module, optimizer_name: str = "adamw", batch_size: int = 32,
                    memory: MemoryConfig | None = None, ema: bool = False, measure: bool = True,
                    probe_batch: int = 2, example: Any = None) -> MemoryReport:
    """Schätzt den Trainingsspeicher und gibt Empfehlungen (siehe Modul-Doku)."""
    memory = memory or MemoryConfig()
    n = sum(p.numel() for p in model.parameters())
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    b = _BYTES[memory.param_dtype]
    weights = n * b
    grads = nt * b
    opt = nt * 4 * _OPT_STATES.get(optimizer_name, 2)
    ema_b = n * 4 if ema else 0.0
    act = None
    if measure:
        try:
            ex = example if example is not None else model.example_input(probe_batch)  # type: ignore[operator]
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(memory.mixed_precision)
            act = _measure_activations(model, ex, dtype) / probe_batch * batch_size
            if memory.gradient_checkpointing:
                act *= 0.35  # grobe Erfahrungsgröße: nur Block-Eingänge bleiben gespeichert
        except Exception:  # Messung ist optional – nie das Training verhindern
            act = None
    device = next(model.parameters()).device if n else torch.device("cpu")
    budget = memory.memory_budget_gb * 1024 ** 3 if memory.memory_budget_gb else detect_budget(device)
    report = MemoryReport(n, nt, weights, grads, opt, ema_b, act, batch_size, budget)
    if budget is not None and report.total_bytes > budget:
        recs = report.recommendations
        if memory.mixed_precision == "none":
            recs.append("memory.mixed_precision = 'bf16' (halbiert Aktivierungen; fp16 auf älteren GPUs)")
        if act and not memory.gradient_checkpointing:
            recs.append("memory.gradient_checkpointing = true (spart oft 60–80 % Aktivierungsspeicher)")
        if act and act > 0.3 * budget:
            recs.append(f"training.batch_size verkleinern und training.gradient_accumulation_steps erhöhen "
                        f"(gleiche effektive Batchgröße {batch_size})")
        if _OPT_STATES.get(optimizer_name, 2) >= 2:
            recs.append("training.optimizer = 'lion' oder 'adafactor' (weniger Optimierer-Zustände)")
        if memory.param_dtype == "float32":
            recs.append("memory.param_dtype = 'bfloat16' (halbiert Gewichte + Gradienten; reines bf16-Training)")
        if nt < n * 0.5:
            recs.append("eingefrorene Blöcke per memory.offload.mode = 'cpu' auslagern")
        else:
            recs.append("Teile einfrieren (tensor_rules mit op 'freeze') – spart Gradienten und Optimierer-Zustände")
        if not memory.activation_offload:
            recs.append("memory.activation_offload = true (Aktivierungen im CPU-RAM)")
    return report
