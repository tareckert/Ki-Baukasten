"""
ki_baukasten.memory
===================

Speicher-, Präzisions- und Geschwindigkeitsoptimierung.

* ``MemoryConfig`` / ``OffloadConfig`` – alle Einstellungen (mit Tooltips)
* ``MemoryManager``   – setzt die Einstellungen an Modell und Trainer um
* ``OffloadManager``  – Layer-Swapping (CPU-RAM / Festplatte, Inferenz)
* ``enable_block_streaming`` – trainierbares Auslagern von Blöcken in den RAM
* ``estimate_memory`` – Speicherschätzung mit Empfehlungen
* ``enable_gradient_checkpointing``, ``apply_fake_quant``, ``quantize_dynamic_int8``
"""

from .config import MemoryConfig, OffloadConfig
from .estimator import MemoryReport, estimate_memory
from .manager import (MemoryManager, apply_fake_quant, enable_gradient_checkpointing, quantize_dynamic_int8,
                      resolve_device)
from .offload import OffloadManager, select_offload_blocks
from .streaming import StreamState, disable_block_streaming, enable_block_streaming

__all__ = [
    "MemoryConfig", "OffloadConfig", "MemoryManager", "OffloadManager", "MemoryReport", "estimate_memory",
    "enable_gradient_checkpointing", "apply_fake_quant", "quantize_dynamic_int8", "resolve_device",
    "select_offload_blocks", "StreamState", "enable_block_streaming", "disable_block_streaming",
]
