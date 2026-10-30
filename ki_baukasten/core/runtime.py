"""
ki_baukasten.core.runtime
=========================

Kleine Laufzeit-Abfragen, die Modelle brauchen, um sich beim **Export** (``torch.export``,
TorchScript-Tracing) und unter ``torch.compile`` richtig zu verhalten – z. B. datenabhängige
Verzweigungen vermeiden oder eine export-taugliche Rechenweise wählen.
"""

from __future__ import annotations

import torch


def is_exporting() -> bool:
    """``True`` während ``torch.export`` oder ``torch.jit.trace`` (Formen/Werte sind dann symbolisch)."""
    exporting = getattr(torch.compiler, "is_exporting", None)
    return bool(torch.jit.is_tracing() or (exporting is not None and exporting()))


def is_compiling() -> bool:
    """``True`` während ``torch.compile``/``torch.export`` den Code verfolgt."""
    return bool(torch.compiler.is_compiling() or torch.jit.is_tracing())
