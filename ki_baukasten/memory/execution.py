"""
ki_baukasten.memory.execution
=============================

Stückweise Ausführung: Batches teilen, Teilergebnisse zusammensetzen, Speicherfehler erkennen.

Diese Helfer stecken hinter ``memory.oom_strategy = "batch_teilen"`` (Training und Vorhersage)
und ``Project.predict(..., chunk_size=…)``. Sie funktionieren mit Tensoren, dicts, Listen und
Tupeln von Tensoren – geteilt wird immer entlang der ersten Achse (Batch).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

_OOM_TEXT = ("out of memory", "can't allocate memory", "cannot allocate memory", "not enough memory",
             "mps backend out of memory")


def is_oom(exc: BaseException) -> bool:
    """``True`` für Speicherfehler (CUDA, MPS, CPU-Allokator)."""
    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    text = str(exc).lower()
    return isinstance(exc, (RuntimeError, MemoryError)) and any(t in text for t in _OOM_TEXT)


def free_memory() -> None:
    """Zwischenspeicher der Geräte leeren (nach einem Speicherfehler)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache") and torch.backends.mps.is_available():
        mps.empty_cache()


def batch_len(batch: Any) -> int:
    """Größe der ersten Achse (0, wenn nichts teilbar ist)."""
    if isinstance(batch, torch.Tensor):
        return batch.shape[0] if batch.dim() else 0
    if isinstance(batch, dict):
        return max((batch_len(v) for v in batch.values()), default=0)
    if isinstance(batch, (list, tuple)):
        return max((batch_len(v) for v in batch), default=0)
    return 0


def split_batch(batch: Any, parts: int) -> list[Any]:
    """Teilt einen Batch in höchstens ``parts`` gleich große Stücke (entlang Achse 0)."""
    n = batch_len(batch)
    parts = max(1, min(parts, n)) if n else 1
    if parts == 1:
        return [batch]
    size = -(-n // parts)
    bounds = [(i, min(n, i + size)) for i in range(0, n, size)]

    def cut(obj: Any, a: int, b: int) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj[a:b] if obj.dim() and obj.shape[0] == n else obj
        if isinstance(obj, dict):
            return {k: cut(v, a, b) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(cut(v, a, b) for v in obj)
        if isinstance(obj, list):
            return [cut(v, a, b) for v in obj]
        return obj

    return [cut(batch, a, b) for a, b in bounds]


def concat_outputs(outs: list[Any]) -> Any:
    """Setzt Teilergebnisse wieder zusammen (Tensoren entlang Achse 0)."""
    first = outs[0]
    if len(outs) == 1:
        return first
    if isinstance(first, torch.Tensor):
        if first.dim() == 0:
            return torch.stack(outs).mean()
        return torch.cat(outs, 0)
    if isinstance(first, dict):
        return {k: concat_outputs([o[k] for o in outs]) for k in first}
    if isinstance(first, (list, tuple)):
        joined = [concat_outputs([o[i] for o in outs]) for i in range(len(first))]
        return type(first)(joined) if isinstance(first, tuple) else joined
    return outs


def run_chunked(fn: Callable[[Any], Any], inputs: Any, chunk_size: int | None = None, on_oom: bool = True,
                max_splits: int = 64) -> tuple[Any, int]:
    """Führt ``fn`` stückweise aus und halbiert die Stückgröße bei Speichermangel.

    Returns:
        ``(ergebnis, anzahl_stücke)``.
    """
    n = batch_len(inputs)
    parts = 1 if not chunk_size or not n else -(-n // chunk_size)
    while True:
        try:
            outs = [fn(part) for part in split_batch(inputs, parts)]
            return concat_outputs(outs), parts
        except Exception as exc:  # noqa: BLE001 - nur Speicherfehler werden abgefangen
            if not (on_oom and is_oom(exc)) or parts >= min(max_splits, max(1, n)):
                raise
            free_memory()
            parts *= 2
