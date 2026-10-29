"""
ki_baukasten.core.tooltip
=========================

Tooltips und Erfahrungsstufen – die "Erklär-Schicht" des Baukastens.

Grundidee
---------
Jede einzelne Einstellung im Baukasten trägt ein ``Tooltip`` mit vier Teilen:

=============  ==============================================================
Feld           Bedeutung
=============  ==============================================================
``what``       **Was ist das?** – kurze, präzise Definition.
``why``        **Warum / was bringt es?** – Wirkung auf Modell und Training.
``use_for``    **Wofür wichtig?** – typische Anwendungsfälle / Situationen.
``tip``        **Tipp** (optional) – typische Werte, Fallstricke, Faustregeln.
=============  ==============================================================

Diese Tooltips werden überall wiederverwendet:

* in der Kommandozeile (``python -m ki_baukasten erklaere <pfad>``),
* in der grafischen Oberfläche (Maus über ein Feld halten),
* in der automatisch erzeugten Parameter-Referenz (``docs/``),
* in Fehlermeldungen und ``config.describe()``.

Erfahrungsstufen (``Level``)
----------------------------
Damit die enorme Anzahl an Einstellungen nicht erschlägt, hat jede Einstellung
eine Stufe. Oberflächen können nach Stufe filtern ("zeige nur Basis"), aber
**nichts ist gesperrt** – auch experimentelle Einstellungen sind jederzeit
änderbar. Die Stufe ist reine Orientierung:

* ``BASIC``         – das, was fast jeder einstellen will (Größe, Lernrate ...).
* ``ADVANCED``      – feineres Tuning (Warmup, Normalisierungs-Position ...).
* ``EXPERT``        – tiefe Eingriffe (Initialisierungs-Gains, Epsilons ...).
* ``EXPERIMENTAL``  – Forschungs-/Exotik-Optionen, deren Nutzen unsicher ist.
"""

from __future__ import annotations

import enum
import textwrap
from dataclasses import dataclass


class Level(str, enum.Enum):
    """Erfahrungsstufe einer Einstellung oder Komponente.

    Erbt von ``str``, damit Stufen direkt als Text in JSON/YAML landen
    können (``"basic"``) und Vergleiche mit Strings funktionieren.
    """

    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERT = "expert"
    EXPERIMENTAL = "experimental"

    @property
    def rank(self) -> int:
        """Numerischer Rang (0 = Basis ... 3 = Experimentell) für Filter."""
        return _LEVEL_ORDER.index(self)

    @property
    def label(self) -> str:
        """Deutscher Anzeigename für Oberflächen."""
        return _LEVEL_LABELS[self]

    @classmethod
    def parse(cls, value: "Level | str | int | None") -> "Level":
        """Wandelt flexible Eingaben in eine Stufe um.

        Akzeptiert ``Level``, englische oder deutsche Namen
        (``"basis"``, ``"fortgeschritten"``, ``"experte"``, ``"experimentell"``)
        sowie Ränge ``0..3``. ``None`` bedeutet "alles anzeigen" (= EXPERIMENTAL).
        """
        if value is None:
            return cls.EXPERIMENTAL
        if isinstance(value, Level):
            return value
        if isinstance(value, int):
            return _LEVEL_ORDER[max(0, min(value, len(_LEVEL_ORDER) - 1))]
        key = str(value).strip().lower()
        aliases = {
            "basic": cls.BASIC, "basis": cls.BASIC, "b": cls.BASIC,
            "advanced": cls.ADVANCED, "fortgeschritten": cls.ADVANCED, "f": cls.ADVANCED,
            "expert": cls.EXPERT, "experte": cls.EXPERT, "e": cls.EXPERT,
            "experimental": cls.EXPERIMENTAL, "experimentell": cls.EXPERIMENTAL, "x": cls.EXPERIMENTAL,
            "alle": cls.EXPERIMENTAL, "all": cls.EXPERIMENTAL,
        }
        if key not in aliases:
            raise ValueError(f"Unbekannte Stufe '{value}'. Erlaubt: basis, fortgeschritten, experte, experimentell")
        return aliases[key]

    def visible_at(self, max_level: "Level | str | None") -> bool:
        """``True``, wenn diese Stufe bei Filter ``max_level`` sichtbar ist."""
        return self.rank <= Level.parse(max_level).rank


_LEVEL_ORDER = [Level.BASIC, Level.ADVANCED, Level.EXPERT, Level.EXPERIMENTAL]
_ABBREVIATIONS = {"z.", "b.", "bzw.", "ca.", "evtl.", "ggf.", "inkl.", "u.", "d.", "h.", "vgl.", "nr.", "max.", "min.",
                  "z.b.", "d.h.", "u.a.", "usw.", "etc.", "bspw.", "sog.", "ggü.", "e.g.", "i.e.", "vs."}
_LEVEL_LABELS = {
    Level.BASIC: "Basis",
    Level.ADVANCED: "Fortgeschritten",
    Level.EXPERT: "Experte",
    Level.EXPERIMENTAL: "Experimentell",
}


@dataclass(frozen=True)
class Tooltip:
    """Erklärung einer Einstellung oder Komponente (siehe Modul-Doku).

    ``frozen=True``: Tooltips sind unveränderlich, damit sie gefahrlos
    zwischen Klassen geteilt werden können.
    """

    what: str
    why: str = ""
    use_for: str = ""
    tip: str = ""

    # ------------------------------------------------------------------ Ausgabe
    def short(self) -> str:
        """Erster Satz von ``what`` – für Listen und kompakte Anzeigen.

        Abkürzungen wie "z. B.", "bzw." oder "ca." beenden keinen Satz.
        """
        text = " ".join(self.what.split())
        for i, ch in enumerate(text):
            if ch in ".?!" and (i + 1 == len(text) or text[i + 1] == " "):
                word = text[: i + 1].rsplit(" ", 1)[-1].lstrip("([{\"'„")
                if ch == "." and (word.lower() in _ABBREVIATIONS or (len(word) == 2 and word[0].isalpha())):
                    continue
                return text[: i + 1]
        return text

    def format(self, indent: str = "", width: int = 92) -> str:
        """Mehrzeilige, umbrochene Textdarstellung mit deutschen Überschriften."""
        lines: list[str] = []
        for label, text in (
            ("Was", self.what),
            ("Warum / Nutzen", self.why),
            ("Wofür wichtig", self.use_for),
            ("Tipp", self.tip),
        ):
            if not text:
                continue
            prefix = f"{indent}{label}: "
            wrapped = textwrap.fill(
                " ".join(text.split()),
                width=width,
                initial_indent=prefix,
                subsequent_indent=indent + " " * (len(label) + 2),
            )
            lines.append(wrapped)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, str]:
        return {"what": self.what, "why": self.why, "use_for": self.use_for, "tip": self.tip}

    def __str__(self) -> str:
        return self.format()


def T(what: str, why: str = "", use_for: str = "", tip: str = "") -> Tooltip:
    """Kurzschreibweise für ``Tooltip(...)``.

    Wird im gesamten Baukasten verwendet, damit die (sehr vielen) Parameter-
    Definitionen gut lesbar bleiben::

        num_heads = Param(8, tooltip=T(
            "Anzahl paralleler Attention-Köpfe.",
            "Mehr Köpfe = mehr unterschiedliche Beziehungsmuster gleichzeitig.",
            "Sprachmodelle, Bild-Transformer, alles mit Attention.",
            "d_model muss durch num_heads teilbar sein (sofern head_dim nicht gesetzt).",
        ))
    """
    return Tooltip(what, why, use_for, tip)


#: Platzhalter-Tooltip für dynamisch erzeugte Einträge ohne eigene Erklärung.
NO_TOOLTIP = Tooltip("(keine Beschreibung hinterlegt)")
