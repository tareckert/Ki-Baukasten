"""
ki_baukasten.core.config
========================

Das Konfigurations-Fundament: ``Param`` (eine Einstellung) und ``BaseConfig``
(eine Gruppe von Einstellungen).

Überblick
---------
Jede Konfigurationsklasse im Baukasten sieht so aus::

    class AttentionConfig(BaseConfig):
        __title__ = "Attention"
        num_heads = Param(8, min=1, tooltip=T("Anzahl Köpfe.", "...", "..."))
        dropout   = Param(0.0, min=0.0, max=1.0, tooltip=T(...))

Daraus ergeben sich automatisch:

* **Typsicherheit**: ``cfg.num_heads = "acht"`` → verständlicher Fehler.
* **Tippfehler-Schutz**: ``cfg.num_haeds = 4`` → "Meinten Sie 'num_heads'?".
* **Pfadzugriff**: ``cfg.get("block.attention.num_heads")`` /
  ``cfg.set("block.attention.num_heads", 16)`` – auch in Listen
  (``"stages.2.channels"``) und in Komponenten-Argumenten
  (``"optimizer.lr"``).
* **Serialisierung**: ``to_dict`` / ``from_dict`` / ``save`` / ``load``
  (JSON, optional YAML) – jede KI ist damit als Datei reproduzierbar.
* **Erklärungen**: ``cfg.explain("num_heads")`` liefert den Tooltip samt
  Typ, Bereich, Standardwert und aktuellem Wert; ``cfg.describe()`` zeigt
  den ganzen Baum.
* **Validierung**: Werte werden sofort beim Setzen geprüft, Abhängigkeiten
  zwischen Werten (z. B. "d_model teilbar durch num_heads") über die
  überschreibbare Methode ``check()``.

Parameter-Arten
---------------
=====================  =====================================================
Klasse                 Inhalt
=====================  =====================================================
``Param``              Einzelwert: Zahl, Text, Ja/Nein, Liste, dict ...
``NestedParam``        Eine eingebettete Unter-Konfiguration (``BaseConfig``).
``ConfigListParam``    Liste von Unter-Konfigurationen (z. B. CNN-Stufen).
``ComponentParam``     Auswahl einer *registrierten Komponente* inkl. ihrer
                       eigenen Argumente (siehe ``core.component``).
=====================  =====================================================

Designentscheidung: Deskriptoren statt ``dataclasses``
------------------------------------------------------
``dataclasses`` können weder Tooltips noch Wertebereiche noch Stufen tragen
und prüfen Typen nicht zur Laufzeit. Ein eigener Deskriptor (``Param``) hängt
sich in jeden Lese-/Schreibzugriff ein und kann so *alles* prüfen und erklären.
"""

from __future__ import annotations

import copy
import enum
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Sequence

from .errors import ConfigError, UnknownParameterError, suggest
from .tooltip import Level, Tooltip, T

# ---------------------------------------------------------------------------
# Hilfskonstanten
# ---------------------------------------------------------------------------

#: Marker für "beliebiger Typ erlaubt".
ANY = object

#: Texteingaben, die als Wahrheitswert interpretiert werden (für CLI/GUI-Eingaben).
_TRUE_STRINGS = {"true", "1", "yes", "y", "ja", "j", "an", "on", "wahr"}
_FALSE_STRINGS = {"false", "0", "no", "n", "nein", "aus", "off", "falsch"}

#: Deutsche Typnamen für Erklärungen.
_TYPE_LABELS = {
    bool: "Ja/Nein",
    int: "Ganzzahl",
    float: "Kommazahl",
    str: "Text",
    list: "Liste",
    tuple: "Tupel",
    dict: "Zuordnung (dict)",
    object: "beliebig",
}


def _type_label(t: type) -> str:
    return _TYPE_LABELS.get(t, getattr(t, "__name__", str(t)))


# ---------------------------------------------------------------------------
# Typumwandlung
# ---------------------------------------------------------------------------

def _convert(value: Any, target: type, item_types: tuple[type, ...] | None) -> tuple[bool, Any]:
    """Versucht ``value`` in ``target`` umzuwandeln.

    Gibt ``(True, umgewandelter_wert)`` oder ``(False, None)`` zurück.

    Die Regeln sind bewusst *tolerant bei eindeutigen Fällen* und *streng bei
    mehrdeutigen*:

    * ``3.0`` → ``int`` ist ok (eindeutig), ``3.5`` → ``int`` nicht.
    * ``True`` ist in Python technisch ein ``int`` – wir lehnen ``bool`` für
      Zahlenfelder trotzdem ab, weil es fast immer ein Versehen ist.
    * Texte aus CLI/GUI (``"1e-4"``, ``"ja"``) werden passend umgewandelt.
    """
    if target is object:
        return True, value

    if target is bool:
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _TRUE_STRINGS:
                return True, True
            if key in _FALSE_STRINGS:
                return True, False
            return False, None
        if isinstance(value, int) and value in (0, 1):
            return True, bool(value)
        return False, None

    if target is int:
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            return True, value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return True, int(value)
        if isinstance(value, str):
            try:
                return True, int(value.strip().replace("_", ""))
            except ValueError:
                try:
                    f = float(value)
                    if f.is_integer():
                        return True, int(f)
                except ValueError:
                    pass
        return False, None

    if target is float:
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)):
            return True, float(value)
        if isinstance(value, str):
            try:
                return True, float(value.strip().replace("_", ""))
            except ValueError:
                return False, None
        return False, None

    if target is str:
        if isinstance(value, enum.Enum) and isinstance(value.value, str):
            return True, value.value
        if isinstance(value, str):
            return True, value
        return False, None

    if target in (list, tuple):
        if isinstance(value, str) and target is list and value.strip().startswith("["):
            # Erlaubt Eingaben wie "[64, 128]" aus CLI/GUI.
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return False, None
        if not isinstance(value, (list, tuple)):
            return False, None
        items = list(value)
        if item_types:
            converted = []
            for item in items:
                for it in item_types:
                    ok, v = _convert(item, it, None)
                    if ok:
                        converted.append(v)
                        break
                else:
                    return False, None
            items = converted
        return True, (items if target is list else tuple(items))

    if target is dict:
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return False, None
        if isinstance(value, Mapping):
            return True, copy.deepcopy(dict(value))
        return False, None

    if isinstance(value, target):
        return True, value
    return False, None


def _normalize_types(type_: Any, default: Any) -> tuple[type, ...]:
    """Bestimmt die erlaubten Typen eines ``Param``.

    * explizit angegeben: ``type=int`` oder ``type=(int, str)``
    * sonst aus dem Standardwert abgeleitet (``8`` → ``int``)
    * Standardwert ``None`` ohne Typ → beliebiger Typ
    """
    if type_ is None:
        if default is None:
            return (object,)
        return (type(default),)
    if isinstance(type_, tuple):
        return type_
    return (type_,)


# ---------------------------------------------------------------------------
# ParamRef – Verweis auf eine konkrete Einstellung in einem Konfigurationsbaum
# ---------------------------------------------------------------------------

@dataclass
class ParamRef:
    """Ergebnis von ``walk()`` und ``param_at()``.

    Attribute:
        path:   vollständiger Punkt-Pfad (z. B. ``"block.attention.num_heads"``)
        name:   letzter Pfadteil
        param:  die ``Param``-Definition (Typ, Tooltip, Bereich ...)
        owner:  das Objekt, in dem der Wert liegt (Config oder Komponente)
        value:  aktueller Wert
        level:  *effektive* Stufe (max. aus eigener Stufe und der aller Eltern)
        depth:  Verschachtelungstiefe (für eingerückte Darstellungen)
    """

    path: str
    name: str
    param: "Param"
    owner: Any
    value: Any
    level: Level
    depth: int = 0


# ---------------------------------------------------------------------------
# Param – eine einzelne Einstellung
# ---------------------------------------------------------------------------

class Param:
    """Eine einzelne, vollständig dokumentierte Einstellung (Python-Deskriptor).

    Args:
        default:   Standardwert. ``None`` bedeutet in der Regel "automatisch"
                   bzw. "nicht verwenden" (siehe Tooltip des Parameters).
        tooltip:   Erklärung (``T(was, warum, wofür, tipp)``) – Pflicht, denn
                   *jede* Einstellung soll erklärt sein.
        type:      erlaubte(r) Typ(en); sonst aus ``default`` abgeleitet.
        choices:   erlaubte Werte. Entweder Liste oder ``dict`` *Wert →
                   Erklärung* (dann bekommt jede Option einen eigenen Tooltip).
                   Darf auch eine Funktion sein, die die Liste dynamisch liefert.
        min, max:  Wertebereich (inklusive) für Zahlen bzw. Listen-Elemente.
        nullable:  ob ``None`` erlaubt ist (Standard: ja, wenn default None ist).
        level:     Erfahrungsstufe (nur Orientierung, nichts wird gesperrt).
        group:     Gruppenname für Oberflächen (z. B. "Regularisierung").
        unit:      Einheit für die Anzeige (z. B. "Schritte", "Bytes").
        validator: zusätzliche Prüffunktion ``f(wert) -> None | str | bool``.
                   Ein zurückgegebener Text ist die Fehlermeldung.
        default_factory: Funktion für veränderliche Standardwerte.
        item_type: Elementtyp(en) für Listen/Tupel (z. B. ``int``).
        length:    erlaubte Länge für Listen: Zahl oder ``(min, max)``.
    """

    kind: ClassVar[str] = "value"

    def __init__(
        self,
        default: Any = None,
        *,
        tooltip: Tooltip,
        type: Any = None,
        choices: Sequence[Any] | Mapping[Any, str] | Callable[[], Sequence[Any]] | None = None,
        min: float | None = None,
        max: float | None = None,
        nullable: bool | None = None,
        level: Level | str = Level.BASIC,
        group: str | None = None,
        unit: str | None = None,
        validator: Callable[[Any], Any] | None = None,
        default_factory: Callable[[], Any] | None = None,
        item_type: Any = None,
        length: int | tuple[int, int] | None = None,
    ):
        if not isinstance(tooltip, Tooltip):
            raise TypeError("Param braucht ein Tooltip (T(...)) – jede Einstellung muss erklärt sein.")
        self.name: str = "?"
        self.owner_name: str = "?"
        self.default = default
        self.default_factory = default_factory
        self.tooltip = tooltip
        # Für die Typ-Ableitung wird die Factory nur aufgerufen, wenn kein Typ
        # angegeben ist (verschachtelte Configs sollen beim Import NICHT
        # instanziiert werden – deren Komponenten sind evtl. noch nicht registriert).
        sample_default = default_factory() if (default_factory is not None and type is None) else default
        self.types = _normalize_types(type, sample_default)
        if item_type is not None and not isinstance(item_type, tuple):
            item_type = (item_type,)
        self.item_types: tuple[type, ...] | None = item_type
        # Choices: Liste oder dict (Wert -> Erklärung) oder Funktion
        self.choice_help: dict[Any, str] = {}
        if isinstance(choices, Mapping):
            self.choice_help = dict(choices)
            self._choices: Any = list(choices.keys())
        else:
            self._choices = choices
        self.min = min
        self.max = max
        self.nullable = (default is None and default_factory is None) if nullable is None else nullable
        self.level = Level.parse(level)
        self.group = group
        self.unit = unit
        self.validator = validator
        self.length = length

    # ------------------------------------------------------------ Deskriptor
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.owner_name = owner.__name__
        # Sofortige Selbstprüfung: ein ungültiger Standardwert ist ein
        # Programmierfehler und soll beim Import auffallen, nicht erst später.
        if self.kind == "value":
            self.coerce(self.make_default(), f"{owner.__name__}.{name} (Standardwert)")

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj._values[self.name]

    def __set__(self, obj: Any, value: Any) -> None:
        obj._values[self.name] = self.coerce(value, f"{type(obj).__name__}.{self.name}")

    # ------------------------------------------------------------ Werte
    @property
    def choices(self) -> list[Any] | None:
        """Erlaubte Werte (dynamisch ausgewertet, falls eine Funktion hinterlegt ist)."""
        if callable(self._choices):
            return list(self._choices())
        return list(self._choices) if self._choices is not None else None

    def make_default(self) -> Any:
        """Erzeugt eine *frische Kopie* des Standardwerts.

        Wichtig, damit sich zwei Konfigurationen niemals dieselbe Liste
        teilen (klassischer Python-Fallstrick bei veränderlichen Defaults).
        """
        if self.default_factory is not None:
            return self.default_factory()
        return copy.deepcopy(self.default)

    def coerce(self, value: Any, path: str | None = None) -> Any:
        """Wandelt ``value`` in den richtigen Typ um und prüft alle Regeln."""
        if value is None:
            if self.nullable:
                return None
            raise ConfigError("darf nicht None/leer sein.", path)
        for t in self.types:
            ok, converted = _convert(value, t, self.item_types)
            if ok:
                break
        else:
            names = " oder ".join(_type_label(t) for t in self.types)
            raise ConfigError(
                f"erwartet {names}, erhalten {value!r} (Typ {type(value).__name__}).", path
            )
        converted = self._match_choice(converted, path)
        self._check_range(converted, path)
        self._check_length(converted, path)
        if self.validator is not None:
            result = self.validator(converted)
            if isinstance(result, str):
                raise ConfigError(result, path)
            if result is False:
                raise ConfigError(f"ungültiger Wert {converted!r}.", path)
        return converted

    def _match_choice(self, value: Any, path: str | None) -> Any:
        choices = self.choices
        if not choices:
            return value
        if value in choices:
            return value
        if isinstance(value, str):
            # Groß-/Kleinschreibung tolerieren: "GELU" -> "gelu"
            lowered = {str(c).lower(): c for c in choices}
            if value.lower() in lowered:
                return lowered[value.lower()]
        hint = suggest(str(value), [str(c) for c in choices])
        msg = f"ungültige Auswahl {value!r}. Erlaubt: {', '.join(map(repr, choices))}."
        if hint:
            msg += f" Meinten Sie {hint}?"
        raise ConfigError(msg, path)

    def _check_range(self, value: Any, path: str | None) -> None:
        if self.min is None and self.max is None:
            return
        numbers: list[Any]
        if isinstance(value, (list, tuple)):
            numbers = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numbers = [value]
        else:
            return
        for v in numbers:
            if self.min is not None and v < self.min:
                raise ConfigError(f"Wert {v} ist kleiner als das Minimum {self.min}.", path)
            if self.max is not None and v > self.max:
                raise ConfigError(f"Wert {v} ist größer als das Maximum {self.max}.", path)

    def _check_length(self, value: Any, path: str | None) -> None:
        if self.length is None or not isinstance(value, (list, tuple)):
            return
        lo, hi = (self.length, self.length) if isinstance(self.length, int) else self.length
        if not (lo <= len(value) <= hi):
            want = str(lo) if lo == hi else f"{lo}..{hi}"
            raise ConfigError(f"Liste muss {want} Elemente haben, hat {len(value)}.", path)

    # ------------------------------------------------------------ Beschreibung
    def type_label(self) -> str:
        """Lesbarer Typname, z. B. ``"Ganzzahl oder Text"`` oder ``"Liste[Ganzzahl]"``."""
        parts = []
        for t in self.types:
            label = _type_label(t)
            if t in (list, tuple) and self.item_types:
                label += "[" + "|".join(_type_label(i) for i in self.item_types) + "]"
            parts.append(label)
        text = " oder ".join(parts)
        if self.nullable and object not in self.types:
            text += " (oder None)"
        return text

    def constraint_label(self) -> str:
        """Lesbare Einschränkungen, z. B. ``"Bereich 0 … 1"`` oder ``"Auswahl: a | b"``."""
        parts = []
        if self.min is not None and self.max is not None:
            parts.append(f"Bereich {self.min} … {self.max}")
        elif self.min is not None:
            parts.append(f"≥ {self.min}")
        elif self.max is not None:
            parts.append(f"≤ {self.max}")
        choices = self.choices
        if choices:
            shown = " | ".join(str(c) for c in choices[:16])
            if len(choices) > 16:
                shown += f" | … ({len(choices)} Optionen)"
            parts.append(f"Auswahl: {shown}")
        if self.length is not None:
            parts.append(f"Länge {self.length}")
        if self.unit:
            parts.append(f"Einheit: {self.unit}")
        return " · ".join(parts)

    def default_repr(self) -> str:
        return _short_repr(self.make_default())

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.owner_name}.{self.name}>"


# ---------------------------------------------------------------------------
# NestedParam / ConfigListParam – verschachtelte Konfigurationen
# ---------------------------------------------------------------------------

class NestedParam(Param):
    """Eine eingebettete Unter-Konfiguration.

    Beispiel: ``TransformerConfig.block`` ist eine ``TransformerBlockConfig``,
    die wiederum eine ``AttentionConfig`` enthält. So entsteht ein Baum, in dem
    man mit Pfaden wie ``"block.attention.num_heads"`` navigiert.

    Zuweisen kann man ein Objekt der Klasse *oder* ein ``dict`` (wird dann
    automatisch zu einem Objekt umgewandelt – praktisch für JSON-Dateien).
    """

    kind = "nested"

    def __init__(
        self,
        config_cls: type["BaseConfig"],
        *,
        tooltip: Tooltip | None = None,
        level: Level | str = Level.BASIC,
        group: str | None = None,
        nullable: bool = False,
        default_factory: Callable[[], Any] | None = None,
        default_none: bool = False,
    ):
        self.config_cls = config_cls
        tip = tooltip or config_cls.__tooltip__ or T(f"Unter-Konfiguration '{config_cls.title()}'.")
        factory = default_factory or ((lambda: None) if default_none else config_cls)
        # Achtung: Param.__init__ ruft die Factory einmal zur Typ-Bestimmung auf.
        super().__init__(
            None, tooltip=tip, type=config_cls, nullable=nullable or default_none,
            level=level, group=group, default_factory=factory,
        )

    def coerce(self, value: Any, path: str | None = None) -> Any:
        if value is None:
            if self.nullable:
                return None
            raise ConfigError("darf nicht None sein.", path)
        if isinstance(value, self.config_cls):
            return value
        if isinstance(value, Mapping):
            return self.config_cls.from_dict(value, path=path)
        raise ConfigError(
            f"erwartet {self.config_cls.__name__} oder dict, erhalten {type(value).__name__}.", path
        )

    def type_label(self) -> str:
        return f"Unter-Konfiguration {self.config_cls.title()}"

    def constraint_label(self) -> str:
        return ""


class ConfigListParam(Param):
    """Eine Liste von Unter-Konfigurationen derselben Klasse.

    Beispiele: die Stufen eines CNN (jede Stufe eigene Kanäle, Kernelgröße,
    Normalisierung ...), die Knoten eines frei definierten Netz-Graphen oder
    die Regeln der Tensor-Chirurgie.
    """

    kind = "config_list"

    def __init__(
        self,
        config_cls: type["BaseConfig"],
        *,
        tooltip: Tooltip,
        default_factory: Callable[[], list] | None = None,
        level: Level | str = Level.BASIC,
        group: str | None = None,
        min_items: int = 0,
        max_items: int | None = None,
    ):
        self.config_cls = config_cls
        self.min_items = min_items
        self.max_items = max_items
        super().__init__(
            None, tooltip=tooltip, type=list, nullable=False, level=level, group=group,
            default_factory=default_factory or list,
        )

    def coerce(self, value: Any, path: str | None = None) -> Any:
        if value is None:
            value = []
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"erwartet eine Liste von {self.config_cls.__name__}.", path)
        result = []
        for i, item in enumerate(value):
            item_path = f"{path}.{i}" if path else str(i)
            if isinstance(item, self.config_cls):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(self.config_cls.from_dict(item, path=item_path))
            else:
                raise ConfigError(f"Element erwartet {self.config_cls.__name__} oder dict.", item_path)
        if len(result) < self.min_items:
            raise ConfigError(f"braucht mindestens {self.min_items} Elemente.", path)
        if self.max_items is not None and len(result) > self.max_items:
            raise ConfigError(f"erlaubt höchstens {self.max_items} Elemente.", path)
        return result

    def new_item(self) -> "BaseConfig":
        """Erzeugt ein neues Element mit Standardwerten (für Oberflächen: "Hinzufügen")."""
        return self.config_cls()

    def type_label(self) -> str:
        return f"Liste von {self.config_cls.title()}"

    def constraint_label(self) -> str:
        if self.max_items is not None:
            return f"{self.min_items}..{self.max_items} Elemente"
        return f"≥ {self.min_items} Elemente" if self.min_items else ""


# ---------------------------------------------------------------------------
# BaseConfig
# ---------------------------------------------------------------------------

def _is_spec(obj: Any) -> bool:
    """Duck-Typing-Test auf ``ComponentSpec`` (vermeidet zirkuläre Importe)."""
    return getattr(obj, "_is_component_spec", False)


def _to_plain(value: Any, only_changed: bool = False) -> Any:
    """Wandelt Konfigurationswerte rekursiv in reine Python-Daten um (für JSON)."""
    if isinstance(value, BaseConfig):
        return value.to_dict(only_changed=only_changed)
    if _is_spec(value):
        return value.to_dict(only_changed=only_changed)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_to_plain(v, only_changed) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v, only_changed) for k, v in value.items()}
    return value


def _short_repr(value: Any, limit: int = 60) -> str:
    if isinstance(value, BaseConfig):
        text = f"<{value.title()}>"
    elif _is_spec(value):
        text = f"<{value.registry}:{value.name}>"
    else:
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def flatten_dict(data: Mapping, prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` → ``{"a.b": 1}`` (Listen bleiben als Ganzes erhalten)."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out.update(flatten_dict(value, path + "."))
        else:
            out[path] = value
    return out


class BaseConfig:
    """Basisklasse aller Konfigurationen (siehe Modul-Dokumentation).

    Klassenattribute für Unterklassen:
        __title__:   Anzeigename (z. B. "Transformer-Block").
        __tooltip__: Erklärung der Konfiguration als Ganzes.
    """

    __title__: ClassVar[str] = ""
    __tooltip__: ClassVar[Tooltip | None] = None
    _params: ClassVar[dict[str, Param]] = {}

    # ------------------------------------------------------------ Klassenaufbau
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Sammelt alle ``Param``-Definitionen inkl. geerbter (in Definitionsreihenfolge).

        Unterklassen können geerbte Parameter überschreiben, indem sie einen
        Parameter gleichen Namens neu definieren (z. B. anderer Standardwert).
        """
        super().__init_subclass__(**kwargs)
        params: dict[str, Param] = {}
        for base in reversed(cls.__mro__[1:]):
            params.update(base.__dict__.get("_params", {}))
        for name, value in cls.__dict__.items():
            if isinstance(value, Param):
                params[name] = value
        cls._params = params

    @classmethod
    def params(cls) -> dict[str, Param]:
        """Alle Parameter-Definitionen dieser Klasse (Name → ``Param``)."""
        return dict(cls._params)

    @classmethod
    def title(cls) -> str:
        return cls.__title__ or cls.__name__

    # ------------------------------------------------------------ Instanzen
    def __init__(self, **values: Any) -> None:
        object.__setattr__(self, "_values", {})
        for name, param in type(self)._params.items():
            self._values[name] = param.make_default()
        for key, value in values.items():
            self._set_value(key, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Hook für Unterklassen (wird nach dem Setzen aller Werte aufgerufen)."""

    def _set_value(self, name: str, value: Any, path: str | None = None) -> None:
        params = type(self)._params
        if name not in params:
            raise UnknownParameterError(name, params.keys(), path=path, owner=self.title())
        self._values[name] = params[name].coerce(value, path or f"{type(self).__name__}.{name}")

    def _get_value(self, name: str, path: str | None = None) -> Any:
        if name not in self._values:
            raise UnknownParameterError(name, type(self)._params.keys(), path=path, owner=self.title())
        return self._values[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._set_value(name, value)

    def __getattr__(self, name: str) -> Any:
        # Wird nur aufgerufen, wenn das normale Nachschlagen scheitert,
        # also bei unbekannten Namen → hilfreiche Fehlermeldung mit Vorschlag.
        if name.startswith("_"):
            raise AttributeError(name)
        raise UnknownParameterError(name, type(self)._params.keys(), owner=self.title())

    def __deepcopy__(self, memo: dict) -> "BaseConfig":
        new = type(self).__new__(type(self))
        for key, value in self.__dict__.items():
            object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    def __getstate__(self) -> dict:
        return dict(self.__dict__)

    def __setstate__(self, state: dict) -> None:
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.to_dict() == other.to_dict()  # type: ignore[union-attr]

    def __repr__(self) -> str:
        changed = self.to_dict(only_changed=True)
        inner = ", ".join(f"{k}={_short_repr(v, 40)}" for k, v in changed.items())
        return f"{type(self).__name__}({inner})"

    # ------------------------------------------------------------ einfacher Zugriff
    def values(self) -> dict[str, Any]:
        """Flache Kopie aller Werte dieser Ebene (ohne Rekursion)."""
        return dict(self._values)

    def as_kwargs(self, exclude: Sequence[str] = ()) -> dict[str, Any]:
        """Werte als ``**kwargs`` – z. B. ``nn.LeakyReLU(**cfg.as_kwargs())``."""
        return {k: v for k, v in self._values.items() if k not in exclude}

    # ------------------------------------------------------------ Serialisierung
    def to_dict(self, only_changed: bool = False) -> dict[str, Any]:
        """Konfiguration als reines ``dict`` (rekursiv, JSON-tauglich).

        Args:
            only_changed: nur Werte ausgeben, die vom Standard abweichen.
                Ideal für kompakte, gut lesbare Konfigurationsdateien.
        """
        out: dict[str, Any] = {}
        for name, param in type(self)._params.items():
            value = self._values[name]
            if not only_changed:
                out[name] = _to_plain(value)
                continue
            default = param.make_default()
            if isinstance(value, BaseConfig) and isinstance(default, BaseConfig) and type(value) is type(default):
                sub = value.to_dict(only_changed=True)
                if sub:
                    out[name] = sub
            elif _is_spec(value) and _is_spec(default) and value.name == default.name:
                sub_args = value.args.to_dict(only_changed=True)
                if sub_args:
                    out[name] = {"name": value.name, "args": sub_args}
            elif _is_spec(value):
                # Andere Komponente als Standard: Name + nur die von *ihren* Standards abweichenden Argumente
                # (beim Laden startet ein Komponentenwechsel per dict bei deren Standardwerten).
                out[name] = value.to_dict(only_changed=True)
            elif _to_plain(value) != _to_plain(default):
                out[name] = _to_plain(value)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, path: str | None = None) -> "BaseConfig":
        """Erzeugt eine Konfiguration aus einem ``dict`` (z. B. aus JSON).

        Schlüssel, die mit ``_`` beginnen, werden ignoriert – so kann man
        Kommentare in JSON-Dateien schreiben: ``{"_kommentar": "..."}``.
        Punkt-Schlüssel (``"block.attention.num_heads": 4``) sind erlaubt.
        """
        obj = cls()
        if data:
            obj.update(data, _path=path)
        return obj

    def to_json(self, path: str | Path | None = None, indent: int = 2, only_changed: bool = False) -> str:
        """Als JSON-Text; mit ``path`` zusätzlich in Datei schreiben."""

        def _fail(o: Any) -> Any:
            raise ConfigError(
                f"Wert {o!r} ist nicht JSON-serialisierbar. Tipp: eigene Funktionen/Objekte "
                f"in einer Registry registrieren und per Namen referenzieren."
            )

        text = json.dumps(self.to_dict(only_changed=only_changed), indent=indent, ensure_ascii=False, default=_fail)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, text_or_path: str | Path) -> "BaseConfig":
        """Aus JSON-Text oder einer JSON-Datei (Pfad) laden."""
        text = str(text_or_path)
        if isinstance(text_or_path, Path) or (not text.lstrip().startswith("{") and Path(text).exists()):
            text = Path(text_or_path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path, only_changed: bool = False) -> Path:
        """Speichert als JSON (``.json``) oder YAML (``.yaml``/``.yml``, benötigt PyYAML)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            import yaml  # optional

            path.write_text(
                yaml.safe_dump(self.to_dict(only_changed=only_changed), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        else:
            self.to_json(path, only_changed=only_changed)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseConfig":
        """Lädt JSON oder YAML (an der Dateiendung erkannt)."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    # ------------------------------------------------------------ Kopieren / Ändern
    def copy(self) -> "BaseConfig":
        """Tiefe, unabhängige Kopie."""
        return copy.deepcopy(self)

    def replace(self, **changes: Any) -> "BaseConfig":
        """Kopie mit Änderungen; Punkt-Pfade über ``replace(**{"a.b": 1})``."""
        new = self.copy()
        new.update(changes)
        return new

    def update(self, changes: Mapping[str, Any], *, _path: str | None = None) -> "BaseConfig":
        """Tiefes Zusammenführen ("deep merge") von Änderungen.

        * Ist der neue Wert ein ``dict`` und der alte eine Unter-Konfiguration
          oder Komponente, wird *hineingemischt* statt ersetzt.
        * Alle anderen Werte werden ersetzt (und geprüft).
        """
        prefix = f"{_path}." if _path else ""
        for key, value in changes.items():
            if not isinstance(key, str):
                key = str(key)
            if key.startswith("_"):
                continue
            first = key.split(".", 1)[0]
            if first not in self._values and not isinstance(self, FreeConfig):
                raise UnknownParameterError(first, type(self)._params.keys(), path=prefix + key, owner=self.title())
            try:
                current = self.get(key)
            except ConfigError:
                current = None  # z. B. neuer Schlüssel in einem dict-Parameter
            if isinstance(value, Mapping) and isinstance(current, BaseConfig):
                current.update(value, _path=prefix + key)
            elif isinstance(value, Mapping) and _is_spec(current):
                current.merge(value, path=prefix + key)
            else:
                self.set(key, value, _path=prefix + key)
        return self

    # ------------------------------------------------------------ Pfade
    def get(self, path: str) -> Any:
        """Liest einen Wert per Punkt-Pfad, z. B. ``"block.attention.num_heads"``.

        Pfadregeln:
            * ``name``          – Parameter einer Konfiguration
            * ``3``             – Listenindex (auch negativ: ``-1``)
            * bei Komponenten: ``name`` (gewählte Komponente), ``args`` (Argument-
              Konfiguration) oder direkt ein Argument (``optimizer.lr``)
            * bei dicts: Schlüssel
        """
        obj: Any = self
        walked: list[str] = []
        for seg in _split_path(path):
            walked.append(seg)
            obj = _step(obj, seg, ".".join(walked))
        return obj

    def set(self, path: str, value: Any, *, _path: str | None = None) -> "BaseConfig":
        """Setzt einen Wert per Punkt-Pfad (mit vollständiger Prüfung).

        Beispiele::

            cfg.set("d_model", 512)
            cfg.set("block.attention.num_heads", 16)
            cfg.set("stages.0.channels", 64)
            cfg.set("optimizer.name", "sgd")      # Komponente wechseln
            cfg.set("optimizer.lr", 0.1)          # Argument der Komponente
            cfg.set("layer_overrides.3.attention.num_heads", 4)  # dict-Pfad
        """
        segs = _split_path(path)
        if not segs:
            raise ConfigError("leerer Pfad.")
        _set_path(self, segs, value, _path or path)
        return self

    def param_at(self, path: str) -> ParamRef:
        """Liefert Definition + Wert der Einstellung unter ``path``.

        Bei Listen- oder dict-Pfaden (``"hidden_dims.1"``) ist ``param`` der
        Parameter, der die Liste hält.
        """
        segs = _split_path(path)
        if not segs:
            raise ConfigError("leerer Pfad.")
        obj: Any = self
        holder: Param | None = None
        owner: Any = None
        level = Level.BASIC
        walked: list[str] = []
        for seg in segs:
            walked.append(seg)
            here = ".".join(walked)
            if isinstance(obj, BaseConfig):
                holder, owner = obj._param_for(seg, here), obj
                level = _max_level(level, holder.level)
                obj = obj._get_value(seg, here)
            elif _is_spec(obj):
                level = _max_level(level, obj.component.level)
                if seg == "name":
                    obj = obj.name  # holder bleibt der Parameter, der die Komponente hält
                elif seg == "args":
                    obj = obj.args
                else:
                    holder, owner = obj.args._param_for(seg, here), obj.args
                    level = _max_level(level, holder.level)
                    obj = obj.args._get_value(seg, here)
            else:
                obj = _step(obj, seg, here)
        assert holder is not None
        return ParamRef(path, segs[-1], holder, owner, obj, level, len(segs) - 1)

    def _param_for(self, name: str, path: str | None = None) -> Param:
        params = type(self)._params
        if name not in params:
            raise UnknownParameterError(name, params.keys(), path=path, owner=self.title())
        return params[name]

    # ------------------------------------------------------------ Validierung
    def check(self) -> list[str]:
        """Prüft Abhängigkeiten *zwischen* Werten. Für Unterklassen.

        Gibt eine Liste von Problembeschreibungen zurück (leer = alles gut).
        Einzelwerte werden bereits beim Setzen geprüft; hier geht es um
        Kombinationen, z. B. "d_model muss durch num_heads teilbar sein".
        """
        return []

    def problems(self, prefix: str = "") -> list[str]:
        """Alle Probleme im gesamten Baum (rekursiv) mit Pfad-Präfix."""
        found = [f"[{prefix.rstrip('.') or self.title()}] {p}" for p in self.check()]
        for name, value in self._values.items():
            found.extend(_child_problems(value, f"{prefix}{name}"))
        return found

    def validate(self) -> "BaseConfig":
        """Wirft ``ConfigError`` mit *allen* gefundenen Problemen, sonst ``self``."""
        found = self.problems()
        if found:
            raise ConfigError("Konfiguration ungültig:\n  - " + "\n  - ".join(found))
        return self

    # ------------------------------------------------------------ Darstellung
    def walk(self, prefix: str = "", max_level: Level | str | None = None,
             _level: Level = Level.BASIC, _depth: int = 0) -> Iterator[ParamRef]:
        """Durchläuft alle Einstellungen des Baums (Tiefensuche, Definitionsreihenfolge).

        Args:
            max_level: nur Einstellungen bis zu dieser Stufe (inkl. Eltern-Stufe).
        """
        limit = Level.parse(max_level)
        for name, param in self._iter_params():
            level = _max_level(_level, param.level)
            if not level.visible_at(limit):
                continue
            path = f"{prefix}{name}"
            value = self._values[name]
            yield ParamRef(path, name, param, self, value, level, _depth)
            yield from _walk_value(value, path, limit, level, _depth + 1)

    def _iter_params(self) -> Iterator[tuple[str, Param]]:
        return iter(type(self)._params.items())

    def describe(self, max_level: Level | str | None = None, tooltips: bool = True,
                 only_changed: bool = False, width: int = 100) -> str:
        """Lesbarer Baum aller Einstellungen (optional mit Kurz-Tooltips)."""
        changed = None
        if only_changed:
            # Gespeicherte Pfade enthalten ".args"/".name" der Komponenten, walk()-Pfade nicht → angleichen.
            changed = {re.sub(r"\.(args|name)(?=\.|$)", "", k) for k in flatten_dict(self.to_dict(only_changed=True))}
        lines = [f"{self.title()}" + (f" – {self.__tooltip__.short()}" if self.__tooltip__ else "")]
        for ref in self.walk(max_level=max_level):
            if changed is not None and not any(c == ref.path or c.startswith(ref.path + ".")
                                               or ref.path.startswith(c + ".") for c in changed):
                continue
            indent = "  " * (ref.depth + 1)
            value = ref.value
            if isinstance(value, BaseConfig):
                shown = f"[{value.title()}]"
            elif _is_spec(value):
                shown = f"⟨{value.name}⟩"
            elif isinstance(value, list) and value and (isinstance(value[0], BaseConfig) or _is_spec(value[0])):
                shown = f"[{len(value)} Elemente]"
            else:
                shown = _short_repr(value, 50)
            badge = "" if ref.level is Level.BASIC else f"  ({ref.level.label})"
            lines.append(f"{indent}{ref.name} = {shown}{badge}")
            if tooltips and ref.param.tooltip.what:
                tip = ref.param.tooltip.short()
                lines.append(f"{indent}    ↳ {tip[: width - len(indent) - 6]}")
        return "\n".join(lines)

    def explain(self, path: str | None = None, width: int = 92) -> str:
        """Ausführliche Erklärung einer Einstellung (oder der ganzen Konfiguration).

        Enthält: Tooltip (Was / Warum / Wofür / Tipp), Typ, Bereich bzw.
        Auswahl, Standardwert, aktuellen Wert und Stufe. Bei Komponenten
        zusätzlich alle wählbaren Optionen mit Kurzbeschreibung.
        """
        if not path:
            head = f"{self.title()}"
            body = self.__tooltip__.format(width=width) if self.__tooltip__ else ""
            names = ", ".join(type(self)._params.keys())
            return f"{head}\n{body}\nEinstellungen: {names}".strip()
        ref = self.param_at(path)
        return format_param_ref(ref, width=width)

    def changed(self) -> dict[str, Any]:
        """Alle vom Standard abweichenden Werte als flaches ``{pfad: wert}``."""
        return flatten_dict(self.to_dict(only_changed=True))

    def diff(self, other: "BaseConfig") -> dict[str, tuple[Any, Any]]:
        """Unterschiede zu einer anderen Konfiguration: ``{pfad: (mein_wert, anderer_wert)}``."""
        mine = flatten_dict(self.to_dict())
        theirs = flatten_dict(other.to_dict())
        keys = list(dict.fromkeys(list(mine) + list(theirs)))
        return {k: (mine.get(k), theirs.get(k)) for k in keys if mine.get(k) != theirs.get(k)}


class EmptyConfig(BaseConfig):
    """Konfiguration ohne Einstellungen (für Komponenten ohne Argumente)."""

    __title__ = "Keine Argumente"


class FreeConfig(BaseConfig):
    """Konfiguration, die **beliebige** Schlüssel akzeptiert (ohne Prüfung).

    Gedacht für maximale Freiheit: Eigene, experimentelle Komponenten können
    ohne Schema registriert werden; ihre Argumente werden 1:1 durchgereicht.
    Nachteil: keine Typprüfung und keine Tooltips pro Argument.
    """

    __title__ = "Freie Argumente"
    __tooltip__ = T(
        "Freie Argumente ohne festes Schema.",
        "Erlaubt beliebige Schlüssel – maximale Freiheit für eigene Komponenten.",
        "Eigene/experimentelle Komponenten, die (noch) keine eigene Config-Klasse haben.",
        "Keine Typprüfung! Für Dauerlösungen besser eine eigene BaseConfig-Klasse schreiben.",
    )

    _FREE_TOOLTIP = T("Freies Argument (ohne Schema).", "Wird unverändert an die Komponente übergeben.")

    def __init__(self, **values: Any) -> None:
        object.__setattr__(self, "_values", {})
        for key, value in values.items():
            self._values[str(key)] = copy.deepcopy(value)

    def _set_value(self, name: str, value: Any, path: str | None = None) -> None:
        self._values[name] = value

    def _get_value(self, name: str, path: str | None = None) -> Any:
        if name not in self._values:
            raise ConfigError(f"freies Argument '{name}' ist nicht gesetzt.", path)
        return self._values[name]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def _param_for(self, name: str, path: str | None = None) -> Param:
        return self.free_param(name)

    def free_param(self, name: str) -> Param:
        return Param(None, tooltip=self._FREE_TOOLTIP, type=object, level=Level.EXPERIMENTAL)

    def _iter_params(self) -> Iterator[tuple[str, Param]]:
        for key in self._values:
            yield key, self.free_param(key)

    def to_dict(self, only_changed: bool = False) -> dict[str, Any]:
        return {k: _to_plain(v) for k, v in self._values.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, path: str | None = None) -> "FreeConfig":
        return cls(**{k: v for k, v in (data or {}).items() if not str(k).startswith("_")})

    def as_kwargs(self, exclude: Sequence[str] = ()) -> dict[str, Any]:
        return {k: v for k, v in self._values.items() if k not in exclude}


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen für Pfade und Bäume
# ---------------------------------------------------------------------------

def _max_level(a: Level, b: Level) -> Level:
    return a if a.rank >= b.rank else b


def _split_path(path: str) -> list[str]:
    path = (path or "").strip()
    return [s for s in path.split(".") if s != ""] if path else []


def _step(obj: Any, seg: str, here: str) -> Any:
    """Ein Navigationsschritt in einem Konfigurationsbaum."""
    if isinstance(obj, BaseConfig):
        return obj._get_value(seg, here)
    if _is_spec(obj):
        if seg == "name":
            return obj.name
        if seg == "args":
            return obj.args
        return obj.args._get_value(seg, here)
    if isinstance(obj, (list, tuple)):
        try:
            return obj[int(seg)]
        except (ValueError, IndexError):
            raise ConfigError(f"ungültiger Listenindex '{seg}' (Länge {len(obj)}).", here) from None
    if isinstance(obj, Mapping):
        if seg in obj:
            return obj[seg]
        raise ConfigError(f"Schlüssel '{seg}' nicht vorhanden. Vorhanden: {list(obj.keys())}", here)
    raise ConfigError(f"kann nicht in einen Wert vom Typ {type(obj).__name__} hineinnavigieren.", here)


def _set_path(obj: Any, segs: list[str], value: Any, full_path: str) -> Any:
    """Setzt rekursiv und gibt das (ggf. neue) Objekt zurück.

    Listen und dicts werden *kopiert* und anschließend über ihren Parameter
    neu gesetzt – dadurch wird auch eine Änderung an einem einzelnen
    Listenelement vollständig geprüft (Typ, Bereich, Länge ...).
    """
    seg, rest = segs[0], segs[1:]
    if isinstance(obj, BaseConfig):
        if not rest:
            obj._set_value(seg, value, full_path)
            return obj
        child = obj._get_value(seg, full_path)
        new_child = _set_path(child, rest, value, full_path)
        if isinstance(child, (list, tuple, Mapping)) or new_child is not child:
            obj._set_value(seg, new_child, full_path)
        return obj
    if _is_spec(obj):
        if seg == "name" and not rest:
            obj.switch(value)
            return obj
        if seg == "args":
            if not rest:
                obj.set_args(value)
            else:
                _set_path(obj.args, rest, value, full_path)
            return obj
        _set_path(obj.args, segs, value, full_path)
        return obj
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        try:
            idx = int(seg)
            items[idx]  # noqa: B018 – prüft den Index
        except (ValueError, IndexError):
            raise ConfigError(f"ungültiger Listenindex '{seg}' (Länge {len(items)}).", full_path) from None
        items[idx] = value if not rest else _set_path(items[idx], rest, value, full_path)
        return items if isinstance(obj, list) else tuple(items)
    if isinstance(obj, Mapping):
        new = dict(obj)
        if not rest:
            new[seg] = value
        else:
            new[seg] = _set_path(new.get(seg, {}), rest, value, full_path)
        return new
    raise ConfigError(f"kann in einem Wert vom Typ {type(obj).__name__} nichts setzen.", full_path)


def _child_problems(value: Any, path: str) -> list[str]:
    if isinstance(value, BaseConfig):
        return value.problems(prefix=path + ".")
    if _is_spec(value):
        return value.args.problems(prefix=path + ".")
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for i, item in enumerate(value):
            out.extend(_child_problems(item, f"{path}.{i}"))
        return out
    return []


def _walk_value(value: Any, path: str, limit: Level, level: Level, depth: int) -> Iterator[ParamRef]:
    if isinstance(value, BaseConfig):
        yield from value.walk(path + ".", limit, level, depth)
    elif _is_spec(value):
        comp_level = _max_level(level, value.component.level)
        if comp_level.visible_at(limit):
            yield from value.args.walk(path + ".", limit, comp_level, depth)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            if isinstance(item, BaseConfig) or _is_spec(item):
                item_path = f"{path}.{i}"
                if _is_spec(item):
                    from .component import component_item_param

                    yield ParamRef(item_path, str(i), component_item_param(item), value, item, level, depth)
                    yield from _walk_value(item, item_path, limit, level, depth + 1)
                else:
                    yield ParamRef(item_path, str(i), _config_item_param(item), value, item, level, depth)
                    yield from item.walk(item_path + ".", limit, level, depth + 1)


def _config_item_param(item: "BaseConfig") -> Param:
    """Pseudo-Parameter für ein Konfigurations-Listenelement (nur Anzeige)."""
    tip = item.__tooltip__ or T(f"Element vom Typ '{item.title()}'.")
    p = Param(None, tooltip=tip, type=object)
    p.name = item.title()
    p.owner_name = "liste"
    return p


def format_param_ref(ref: ParamRef, width: int = 92) -> str:
    """Formatiert eine Einstellung ausführlich (wird von ``explain`` benutzt)."""
    p = ref.param
    lines = [f"Einstellung: {ref.path}"]
    lines.append(f"  Typ:        {p.type_label()}")
    constraint = p.constraint_label()
    if constraint:
        lines.append(f"  Regeln:     {constraint}")
    if p.kind == "value":
        lines.append(f"  Standard:   {p.default_repr()}")
    lines.append(f"  Aktuell:    {_short_repr(ref.value, 80)}")
    lines.append(f"  Stufe:      {ref.level.label}" + (f"   Gruppe: {p.group}" if p.group else ""))
    lines.append("")
    lines.append(p.tooltip.format(indent="  ", width=width))
    if p.choice_help:
        lines.append("")
        lines.append("  Optionen:")
        for key, text in p.choice_help.items():
            lines.append(f"    • {key}: {text}")
    extra = getattr(p, "options_help", None)
    if callable(extra):
        text = extra()
        if text:
            lines.append("")
            lines.append(text)
    if isinstance(ref.value, BaseConfig):
        lines.append("")
        lines.append(f"  Enthält: {', '.join(type(ref.value)._params.keys())}")
    elif _is_spec(ref.value):
        lines.append("")
        lines.append(f"  Gewählte Komponente '{ref.value.name}': {ref.value.component.tooltip.short()}")
        arg_names = list(ref.value.args.values().keys())
        if arg_names:
            lines.append(f"  Argumente: {', '.join(arg_names)}  (Pfad: {ref.path}.<argument>)")
    return "\n".join(lines)
