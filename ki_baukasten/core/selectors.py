"""
ki_baukasten.core.selectors
===========================

Auswahl-Ausdrücke für Schichten und schichtweise Abweichungen ("Overrides").

Warum?
------
Ein Kernanspruch des Baukastens: *jede Schicht kann anders sein*. Statt für
ein 24-schichtiges Modell 24 vollständige Block-Konfigurationen zu schreiben,
gibt es eine Basis-Konfiguration plus gezielte Abweichungen::

    layer_overrides = {
        "0":       {"attention": {"num_heads": 4}},      # nur die erste Schicht
        "-1":      {"ffn": {"name": "moe"}},             # nur die letzte
        "2:6":     {"attention.window_size": 128},       # Schichten 2,3,4,5
        "ungerade": {"norm": "layernorm"},               # 1,3,5,...
        "*":       {"drop_path": 0.05},                  # alle
    }

Mehrere passende Regeln werden **in der angegebenen Reihenfolge** angewendet
– spätere überschreiben frühere. So lassen sich Muster wie "alle Schichten
sliding window, aber jede vierte global" ausdrücken::

    {"*": {"attention.window_size": 256}, "3::4": {"attention.window_size": None}}

Syntax eines Selektors
----------------------
Kommagetrennte Liste von Teilen (Vereinigungsmenge). Jeder Teil ist:

=================  ===================================================
Teil               Bedeutung
=================  ===================================================
``5`` / ``-1``     einzelner Index (negativ = von hinten)
``2-5``            Bereich *inklusive* Ende (2,3,4,5)
``2:6`` ``::2``    Python-Slice (Ende exklusiv, Schrittweite optional)
``*`` / ``alle``   alle Schichten
``gerade``/even    0,2,4,...
``ungerade``/odd   1,3,5,...
``erste``/first    Index 0
``letzte``/last    letzter Index
``mitte``/middle   mittlerer Index (n // 2)
=================  ===================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from .config import BaseConfig
from .errors import ConfigError

C = TypeVar("C", bound=BaseConfig)

_KEYWORDS = {
    "*": "all", "all": "all", "alle": "all",
    "even": "even", "gerade": "even",
    "odd": "odd", "ungerade": "odd",
    "first": "first", "erste": "first",
    "last": "last", "letzte": "last",
    "middle": "middle", "mitte": "middle",
}


def select_indices(selector: str | int, n: int) -> list[int]:
    """Wertet einen Selektor für ``n`` Elemente aus → sortierte Indexliste.

    >>> select_indices("0,2-3,-1", 6)
    [0, 2, 3, 5]
    >>> select_indices("::2", 5)
    [0, 2, 4]
    """
    if isinstance(selector, int):
        selector = str(selector)
    chosen: set[int] = set()
    for raw in str(selector).split(","):
        part = raw.strip().lower()
        if not part:
            continue
        if part in _KEYWORDS:
            kind = _KEYWORDS[part]
            if kind == "all":
                chosen.update(range(n))
            elif kind == "even":
                chosen.update(range(0, n, 2))
            elif kind == "odd":
                chosen.update(range(1, n, 2))
            elif kind == "first" and n:
                chosen.add(0)
            elif kind == "last" and n:
                chosen.add(n - 1)
            elif kind == "middle" and n:
                chosen.add(n // 2)
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) > 3:
                raise ConfigError(f"ungültiger Slice-Selektor '{part}'.")
            try:
                nums = [int(p) if p.strip() else None for p in pieces]
            except ValueError:
                raise ConfigError(f"ungültiger Slice-Selektor '{part}'.") from None
            chosen.update(range(n)[slice(*nums)])
            continue
        if "-" in part[1:]:  # "2-5" (aber nicht "-1")
            left, right = part.split("-", 1) if not part.startswith("-") else (None, None)
            if left is None:
                raise ConfigError(f"Bereichs-Selektor mit negativem Start wird nicht unterstützt: '{part}'. Nutze Slices.")
            try:
                a, b = int(left), int(right)
            except ValueError:
                raise ConfigError(f"ungültiger Bereichs-Selektor '{part}'.") from None
            chosen.update(i for i in range(a, b + 1) if 0 <= i < n)
            continue
        try:
            idx = int(part)
        except ValueError:
            raise ConfigError(
                f"unbekannter Selektor '{part}'. Erlaubt: Zahl, a-b, a:b:s, *, gerade, ungerade, erste, letzte, mitte."
            ) from None
        if idx < 0:
            idx += n
        if 0 <= idx < n:
            chosen.add(idx)
    return sorted(chosen)


def matches(selector: str | int, index: int, n: int) -> bool:
    """``True``, wenn ``index`` vom Selektor erfasst wird."""
    return index in select_indices(selector, n)


def resolve_layer_config(base: C, overrides: Mapping[str, Any] | None, index: int, n: int) -> C:
    """Erzeugt die Konfiguration für Schicht ``index`` aus Basis + Overrides.

    Die Basis wird **kopiert** – sie selbst bleibt unverändert. Jede passende
    Override-Regel wird per ``update`` (tiefes Zusammenführen) angewendet.
    """
    cfg = base.copy()
    if not overrides:
        return cfg
    for selector, changes in overrides.items():
        if str(selector).startswith("_"):
            continue  # Kommentar-Schlüssel
        if matches(selector, index, n):
            if not isinstance(changes, Mapping):
                raise ConfigError(f"Override für Selektor '{selector}' muss ein dict sein.")
            cfg.update(changes, _path=f"layer_overrides[{selector}]")
    return cfg


def validate_overrides(base: BaseConfig, overrides: Mapping[str, Any] | None, n: int) -> list[str]:
    """Prüft alle Overrides vorab (für ``check()``) und liefert Probleme als Text."""
    problems: list[str] = []
    if not overrides:
        return problems
    for selector, changes in overrides.items():
        if str(selector).startswith("_"):
            continue
        try:
            idx = select_indices(selector, n)
        except ConfigError as exc:
            problems.append(str(exc))
            continue
        if not idx:
            problems.append(f"Selektor '{selector}' trifft bei {n} Schichten keine einzige Schicht.")
            continue
        try:
            resolve_layer_config(base, {selector: changes}, idx[0], n).validate()
        except ConfigError as exc:
            problems.append(f"Override '{selector}': {exc}")
    return problems
