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

Übernahme von anderen Schichten (``@von`` / ``@nur`` / ``@ausser``)
-------------------------------------------------------------------
Eine Schicht kann die **fertige** Konfiguration einer anderen Schicht
übernehmen – ganz, nur bestimmte Einstellungen oder alle außer bestimmten::

    "layer_overrides": {
        "0":   {"mixer.num_heads": 16, "ffn": "moe"},
        "-1":  {"@von": 0},                                    # alles von Schicht 0 übernehmen
        "5":   {"@von": 0, "@nur": ["mixer.num_heads"]},       # nur diese eine Einstellung
        "6":   {"@von": 0, "@ausser": ["ffn"], "dropout": 0.2} # alles außer ffn, danach eigene Änderung
        "7":   {"@von": "basis"},                              # zurück auf die Basis-Konfiguration
    }

``@von`` nimmt einen Index (auch negativ) oder ``basis``; Ketten (Schicht 7 von
6, 6 von 0) sind erlaubt, Kreise werden erkannt. Englische Schlüssel ``@from``,
``@only``, ``@except`` funktionieren ebenso.

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


_FROM_KEYS, _ONLY_KEYS, _EXCEPT_KEYS = ("@von", "@from"), ("@nur", "@only"), ("@ausser", "@außer", "@except")
_META_KEYS = set(_FROM_KEYS + _ONLY_KEYS + _EXCEPT_KEYS)


def _first(changes: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in changes:
            return changes[k]
    return None


def _as_paths(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


def _copy_value(src: BaseConfig, dst: BaseConfig, path: str) -> None:
    import copy

    dst.set(path, copy.deepcopy(src.get(path)))


def _inherit(base: C, overrides: Mapping[str, Any], cfg: C, changes: Mapping[str, Any], index: int, n: int,
             stack: tuple[int, ...]) -> C:
    """Wendet ``@von``/``@nur``/``@ausser`` einer Regel an und liefert die neue Konfiguration."""
    source = _first(changes, _FROM_KEYS)
    if str(source).lower() in ("basis", "base"):
        src = base.copy()
    else:
        try:
            src_idx = int(source)
        except (TypeError, ValueError):
            raise ConfigError(f"@von erwartet einen Schichtindex oder 'basis', nicht {source!r}.") from None
        if src_idx < 0:
            src_idx += n
        if not 0 <= src_idx < n:
            raise ConfigError(f"@von: Schicht {source} existiert nicht (nur {n} Schichten).")
        if src_idx in stack + (index,):
            chain = " → ".join(str(i) for i in stack + (index, src_idx))
            raise ConfigError(f"@von bildet einen Kreis: {chain}.")
        src = resolve_layer_config(base, overrides, src_idx, n, _stack=stack + (index,))
    only, excluded = _as_paths(_first(changes, _ONLY_KEYS)), _as_paths(_first(changes, _EXCEPT_KEYS))
    if only:
        for path in only:  # nur die genannten Einstellungen übernehmen
            _copy_value(src, cfg, path)
        return cfg
    new = src.copy()
    for path in excluded:  # alles übernehmen – außer diesen (die behalten ihren bisherigen Wert)
        _copy_value(cfg, new, path)
    return new


def resolve_layer_config(base: C, overrides: Mapping[str, Any] | None, index: int, n: int,
                         _stack: tuple[int, ...] = ()) -> C:
    """Erzeugt die Konfiguration für Schicht ``index`` aus Basis + Overrides.

    Die Basis wird **kopiert** – sie selbst bleibt unverändert. Jede passende
    Override-Regel wird per ``update`` (tiefes Zusammenführen) angewendet.
    Regeln mit ``@von`` übernehmen zuerst (ganz oder teilweise) die fertige
    Konfiguration einer anderen Schicht (siehe Modul-Doku).
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
            if _first(changes, _FROM_KEYS) is not None:
                cfg = _inherit(base, overrides, cfg, changes, index, n, _stack)
            elif any(k in changes for k in _META_KEYS):
                raise ConfigError(f"Override '{selector}': @nur/@ausser brauchen ein @von.")
            rest = {k: v for k, v in changes.items() if k not in _META_KEYS}
            cfg.update(rest, _path=f"layer_overrides[{selector}]")
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
            # Vollständig auflösen (alle Regeln), damit @von-Bezüge und Kreise erkannt werden.
            resolve_layer_config(base, overrides, idx[0], n).validate()
        except ConfigError as exc:
            problems.append(f"Override '{selector}': {exc}")
    return problems
