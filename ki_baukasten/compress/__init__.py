"""
ki_baukasten.compress
=====================

**Modelle eindampfen**: strukturiertes Pruning (Neuronen, Köpfe, Blöcke), Low-Rank-
Zerlegung, Sparsity, Quantisierung, Präzision – plus Heilen per Distillation und
Messung vorher/nachher. Siehe ``pipeline`` (Rezepte) und ``methods`` (Verfahren).
"""

from . import structure  # noqa: F401  (registriert die Struktur-Modul-Operationen)
from .methods import (apply_rules, block_influence, calibration_loss, drop_layers, low_rank, prune_ffn, prune_heads,
                      prune_mlp, sparsify, sparsity_of)
from .pipeline import STEPS, CompressionConfig, CompressionReport, run_compression
from .structure import LowRankLinear

__all__ = ["prune_ffn", "prune_heads", "drop_layers", "prune_mlp", "low_rank", "sparsify", "sparsity_of",
           "block_influence", "calibration_loss", "apply_rules", "STEPS", "CompressionConfig", "CompressionReport",
           "run_compression", "LowRankLinear"]
