"""
ki_baukasten.tensor.selection
=============================

Gemeinsame **Auswahl- und Geltungsbereichs-Logik** für Tensor- und Modul-Regeln.

Jede Regel wählt zuerst Kandidaten (Parameter bzw. Module) über Namen, Typ und
Schicht aus und legt dann mit ``scope`` fest, **für welche** der Treffer sie gilt:

==================  =====================================================================
``scope``           Wirkung
==================  =====================================================================
``alle``            auf alle Treffer (Standard) – "Übernahme für alle anderen"
``nur_auswahl``     nur auf die Treffer an den Positionen ``pick`` (z. B. ``"0,2,-1"``)
``nur_einer``       genau ein Treffer (``pick``, Standard: der erste) – sonst Fehler
``alle_ausser``     auf alle Treffer außer denen an den Positionen ``pick``
==================  =====================================================================

``layers`` schränkt vorab auf Parameter/Module **innerhalb** bestimmter Blöcke
ein (gleiche Selektor-Syntax wie bei ``layer_overrides``: ``"0"``, ``"-1"``,
``"::2"``, ``"2-5"``, ``"gerade"`` …). Die Blöcke liefert das Modell über
``checkpointable_blocks()`` (Transformer: alle Encoder- und Decoder-Blöcke).

``match`` steuert, wie ``target`` verglichen wird: ``search`` (Regex irgendwo
im Namen), ``full`` (Regex über den ganzen Namen) oder ``exact`` (wörtlich).
"""

from __future__ import annotations

import re
from typing import TypeVar

from torch import nn

from ..core import ConfigError, Level, Param, T, select_indices

Item = TypeVar("Item")

SCOPES = {
    "alle": "alle Treffer",
    "nur_auswahl": "nur die Treffer an den Positionen 'pick'",
    "nur_einer": "genau ein Treffer ('pick', Standard: der erste)",
    "alle_ausser": "alle Treffer außer denen an den Positionen 'pick'",
}


def scope_params() -> tuple[Param, Param, Param, Param]:
    """Frische Auswahl-Parameter ``(match, layers, scope, pick)`` für Regel-Konfigurationen::

        class MeineRegel(BaseConfig):
            match, layers, scope, pick = scope_params()
    """
    return (
        Param("search", choices={
            "search": "Regex irgendwo im Namen (re.search)",
            "full": "Regex muss den ganzen Namen treffen",
            "exact": "wörtlicher Name (kein Regex)",
        }, level=Level.ADVANCED, tooltip=T(
            "Wie 'target' mit den Namen verglichen wird.",
            "'exact' trifft garantiert nur genau das eine Element mit diesem Namen.",
            "Einzelnen Tensor/Modul gezielt treffen: exact.",
        )),
        Param(None, type=str, nullable=True, tooltip=T(
            "Nur innerhalb dieser Blöcke (Selektor wie bei layer_overrides, z. B. '0', '-1', '::2', '2-5').",
            "Schichtweise Eingriffe ohne komplizierte Regex.",
            "'0' = nur erster Block, '-2:' = letzte zwei, 'gerade'.",
        )),
        Param("alle", choices=SCOPES, tooltip=T(
            "Für welche Treffer die Regel gilt: alle, nur eine Auswahl, genau einer oder alle außer einer Auswahl.",
            "Eine Einstellung wahlweise auf alle anderen übertragen, nur auf bestimmte oder nur auf den einen.",
            "Mit 'pick' kombinieren (Positionen in der Trefferliste, z. B. '0', '-1', '0,3', '::2').",
        )),
        Param(None, type=str, nullable=True, tooltip=T(
            "Positionen in der Trefferliste für scope = nur_auswahl / nur_einer / alle_ausser.",
            "Wählt aus allen passenden Elementen bestimmte aus (Reihenfolge wie im Modell).",
            "'0' = erster Treffer, '-1' = letzter, '1:3', 'ungerade'. Trefferliste zeigt preview().",
        ), level=Level.ADVANCED),
    )


def name_matcher(target: str, mode: str):
    if mode == "exact":
        return lambda name: name == target
    rx = re.compile(target)
    if mode == "full":
        return lambda name: rx.fullmatch(name) is not None
    return lambda name: rx.search(name) is not None


def block_prefixes(model: nn.Module) -> list[str]:
    """Modulnamen der Hauptblöcke in Reihenfolge (für ``layers``)."""
    blocks = model.checkpointable_blocks() if hasattr(model, "checkpointable_blocks") else []
    by_id = {id(m): n for n, m in model.named_modules()}
    return [by_id[id(b)] for b in blocks if id(b) in by_id]


def layer_filter(model: nn.Module, selector: str | None):
    """Prädikat ``name → bool`` für Namen innerhalb der gewählten Blöcke (``None`` = alles)."""
    if selector is None:
        return lambda name: True
    prefixes = block_prefixes(model)
    if not prefixes:
        raise ConfigError(f"'layers' = {selector!r}: {type(model).__name__} meldet keine Blöcke.")
    chosen = [prefixes[i] for i in select_indices(selector, len(prefixes))]
    return lambda name: any(name == p or name.startswith(p + ".") for p in chosen)


def apply_scope(items: list[Item], scope: str, pick: str | None, what: str = "Regel") -> list[Item]:
    """Wendet ``scope``/``pick`` auf die geordnete Trefferliste an (siehe Modul-Doku)."""
    if scope == "alle":
        return items
    n = len(items)
    picked = select_indices(pick if pick is not None else "0", n) if n else []
    if scope == "nur_auswahl":
        if pick is None:
            raise ConfigError(f"{what}: scope 'nur_auswahl' braucht 'pick' (z. B. '0,2').")
        return [items[i] for i in picked]
    if scope == "nur_einer":
        if len(picked) != 1:
            raise ConfigError(f"{what}: scope 'nur_einer' muss genau einen Treffer auswählen "
                              f"(pick={pick!r} wählt {len(picked)} von {n}).")
        return [items[picked[0]]]
    if scope == "alle_ausser":
        if pick is None:
            raise ConfigError(f"{what}: scope 'alle_ausser' braucht 'pick'.")
        skip = set(picked)
        return [it for i, it in enumerate(items) if i not in skip]
    raise ConfigError(f"Unbekannter scope '{scope}'.")
