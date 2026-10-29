"""
ki_baukasten.tensor
===================

Chirurgie am gebauten Modell:

* ``surgery``   – **Tensor-Chirurgie**: Regeln für einzelne Parameter bzw. Ausschnitte
  (Einfrieren, Initialisieren, Pruning, Kopieren, eigene Lernraten, Gradienten-
  manipulation, Beschränkungen, Gewichtsbindung ...)
* ``modules``   – **Modul-Chirurgie**: Regeln für ganze Schichten jeder Art
  (ersetzen, entfernen, LoRA/Quantisierung einzeln, Präzision, Hooks, Übernahme ...)
* ``selection`` – gemeinsamer Geltungsbereich: alle / nur Auswahl / nur einer / alle außer
"""

from .modules import MODULE_OPS, ModuleRuleConfig, ModuleSurgeon
from .selection import SCOPES, apply_scope
from .surgery import (TENSOR_FUNCTIONS, TENSOR_OPS, TensorOp, TensorRuleConfig, TensorSurgeon, parse_index)

__all__ = ["TENSOR_OPS", "TENSOR_FUNCTIONS", "TensorOp", "TensorRuleConfig", "TensorSurgeon", "parse_index",
           "MODULE_OPS", "ModuleRuleConfig", "ModuleSurgeon", "SCOPES", "apply_scope"]
