"""
ki_baukasten.core.errors
========================

Alle Fehlerklassen des Baukastens an einem Ort.

Warum eigene Fehlerklassen?
---------------------------
Ein Baukasten mit tausenden Einstellmöglichkeiten lebt davon, dass Fehler
*sofort* und *verständlich* gemeldet werden. Ein Tippfehler wie
``num_haeds`` darf nicht stillschweigend ignoriert werden, sonst trainiert
man stundenlang ein Modell mit falschen Einstellungen.

Darum:

* Jede Fehlermeldung nennt den **vollständigen Pfad** der Einstellung
  (z. B. ``model.block.attention.num_heads``).
* Bei unbekannten Namen werden **Vorschläge** gemacht ("Meinten Sie ...?").
* Alle Fehler erben von ``KIBaukastenError`` – so kann man mit einem
  einzigen ``except`` alle Baukasten-Fehler abfangen.
"""

from __future__ import annotations

import difflib
from typing import Iterable


class KIBaukastenError(Exception):
    """Basisklasse aller Fehler des KI-Baukastens."""


class ConfigError(KIBaukastenError, ValueError):
    """Eine Einstellung hat einen ungültigen Wert oder Typ.

    Erbt zusätzlich von ``ValueError``, damit auch generischer Code, der
    ``ValueError`` erwartet, korrekt reagiert.
    """

    def __init__(self, message: str, path: str | None = None):
        self.path = path
        if path:
            message = f"[{path}] {message}"
        super().__init__(message)


class UnknownParameterError(ConfigError, AttributeError):
    """Es wurde eine Einstellung angesprochen, die es nicht gibt.

    Erbt von ``AttributeError``, weil der Zugriff ``cfg.tippfehler``
    sich wie ein normaler Attributfehler anfühlen soll.
    """

    def __init__(self, name: str, available: Iterable[str], path: str | None = None, owner: str = ""):
        available = list(available)
        hint = suggest(name, available)
        where = f" in '{owner}'" if owner else ""
        message = f"Unbekannte Einstellung '{name}'{where}."
        if hint:
            message += f" Meinten Sie: {hint}?"
        elif available:
            shown = ", ".join(available[:12])
            more = " ..." if len(available) > 12 else ""
            message += f" Verfügbar: {shown}{more}"
        super().__init__(message, path)


class RegistryError(KIBaukastenError, KeyError):
    """Eine Komponente (z. B. eine Aktivierungsfunktion) ist nicht registriert."""

    def __str__(self) -> str:  # KeyError setzt sonst Anführungszeichen um die Meldung
        return str(self.args[0]) if self.args else ""


class BuildError(KIBaukastenError, RuntimeError):
    """Beim Zusammenbauen eines Modells / Systems ist etwas schiefgegangen."""


class TrainingError(KIBaukastenError, RuntimeError):
    """Fehler während des Trainings (z. B. NaN-Verluste, fehlende Daten)."""


def suggest(name: str, candidates: Iterable[str], n: int = 3) -> str:
    """Liefert die ähnlichsten Kandidaten als lesbaren Text (für "Meinten Sie ...?").

    Nutzt ``difflib`` (Standardbibliothek) mit einer moderaten Ähnlichkeits-
    schwelle von 0.6 – niedrig genug für typische Tippfehler, hoch genug,
    um keinen Unsinn vorzuschlagen.
    """
    matches = difflib.get_close_matches(name, list(candidates), n=n, cutoff=0.6)
    return " / ".join(f"'{m}'" for m in matches)
