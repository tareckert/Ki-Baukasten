"""
ki_baukasten.tensor
===================

Tensor-Chirurgie: Regeln für Eingriffe in einzelne Parameter bzw. Ausschnitte
(Einfrieren, Initialisieren, Pruning, eigene Lernraten, Gradienten-
manipulation, Beschränkungen, Gewichtsbindung ...). Siehe ``surgery``.
"""

from .surgery import (TENSOR_FUNCTIONS, TENSOR_OPS, TensorOp, TensorRuleConfig, TensorSurgeon, parse_index)

__all__ = ["TENSOR_OPS", "TENSOR_FUNCTIONS", "TensorOp", "TensorRuleConfig", "TensorSurgeon", "parse_index"]
