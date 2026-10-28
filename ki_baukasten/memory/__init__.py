"""
ki_baukasten.memory
===================

Speicher-, Präzisions- und Geschwindigkeitsoptimierung.

* ``MemoryConfig`` / ``OffloadConfig`` – alle Einstellungen (mit Tooltips)
* ``MemoryManager``   – setzt die Einstellungen an Modell und Trainer um
* ``OffloadManager``  – Layer-Swapping (CPU-RAM / Festplatte)
* ``estimate_memory`` – Speicherschätzung mit Empfehlungen
* ``enable_gradient_checkpointing``, ``apply_fake_quant``, ``quantize_dynamic_int8``
"""

from .config import MemoryConfig, OffloadConfig
from .estimator import MemoryReport, estimate_memory
from .manager import (MemoryManager, apply_fake_quant, enable_gradient_checkpointing, quantize_dynamic_int8,
                      resolve_device)
from .offload import OffloadManager

__all__ = [
    "MemoryConfig", "OffloadConfig", "MemoryManager", "OffloadManager", "MemoryReport", "estimate_memory",
    "enable_gradient_checkpointing", "apply_fake_quant", "quantize_dynamic_int8", "resolve_device",
]
