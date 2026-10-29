"""
ki_baukasten.analysis.benchmark
===============================

Messen statt raten: **Größe**, **Rechenaufwand (MACs/FLOPs)** und **Latenz** eines Modells.

* ``model_size_bytes`` – tatsächlicher Speicher der Gewichte (inkl. quantisierter Puffer, echte Datentypen)
* ``count_macs`` – Multiply-Accumulate-Operationen pro Forward (per Hooks: Linear, Faltungen,
  Attention-Matrixprodukte, SSM-Scan, Embedding = 0). FLOPs ≈ 2 · MACs.
* ``measure_latency`` – Median-Laufzeit eines Forwards (mit Aufwärmen), CPU oder GPU
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn


def model_size_bytes(model: nn.Module) -> int:
    """Gewichte + Puffer (ohne nicht-persistente Hilfspuffer) in Bytes."""
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        key = t.data_ptr()
        if key in seen or t.numel() == 0:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total


def count_macs(model: nn.Module, example: Any) -> int:
    """Multiply-Accumulates eines Forwards mit ``example`` (Näherung, deckt die teuren Operationen ab)."""
    from ..components.attention import MultiHeadAttention
    from ..components.mixers import SelectiveSSM
    from ..training.paradigms import call_model

    macs = [0]

    def linear_hook(m, args, out):
        x = args[0]
        tokens = x.numel() // max(1, x.shape[-1])
        macs[0] += tokens * m.in_features * m.out_features

    def conv_hook(m, args, out):
        k = 1
        for s in m.kernel_size:
            k *= s
        macs[0] += out.numel() * (m.in_channels // m.groups) * k

    def attn_hook(m, args, out):
        x = args[0]
        kv = args[1] if len(args) > 1 and isinstance(args[1], torch.Tensor) else x
        b, t = x.shape[0], x.shape[1]
        macs[0] += 2 * b * m.num_heads * t * kv.shape[1] * m.head_dim  # QKᵀ und Gewichte·V

    def ssm_hook(m, args, out):
        x = args[0]
        macs[0] += 6 * x.shape[0] * x.shape[1] * m.e * m.n  # Diskretisierung + Scan + Ausgabe

    handles = []
    for m in model.modules():
        hook = None
        if isinstance(m, nn.Linear) or type(m).__name__ in ("QuantLinear",):
            hook = linear_hook
        elif isinstance(m, nn.modules.conv._ConvNd):
            hook = conv_hook
        elif isinstance(m, MultiHeadAttention):
            hook = attn_hook
        elif isinstance(m, SelectiveSSM):
            hook = ssm_hook
        if hook is not None:
            try:
                handles.append(m.register_forward_hook(hook))
            except RuntimeError:  # TorchScript: keine Hooks möglich
                pass
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            call_model(model, example)
    finally:
        for h in handles:
            h.remove()
        model.train(was)
    return macs[0]


def measure_latency(model: nn.Module, example: Any, runs: int = 10, warmup: int = 2) -> float:
    """Median der Forward-Zeit in Sekunden."""
    from ..training.paradigms import call_model

    dev = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    was = model.training
    model.eval()
    times = []
    try:
        with torch.no_grad():
            for i in range(warmup + runs):
                sync()
                t0 = time.perf_counter()
                call_model(model, example)
                sync()
                if i >= warmup:
                    times.append(time.perf_counter() - t0)
    finally:
        model.train(was)
    return statistics.median(times)
