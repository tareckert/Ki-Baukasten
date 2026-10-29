"""
ki_baukasten.core.text
======================

Kleine Text-Hilfen für deutschsprachige Ausgaben.
"""

from __future__ import annotations


def format_number(value: float, decimals: int = 0) -> str:
    """Zahl im deutschen Format: Tausenderpunkt, Dezimalkomma (``1234567.8`` → ``"1.234.567,8"``)."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
