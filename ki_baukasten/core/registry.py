"""
ki_baukasten.core.registry
==========================

Registries – die "Teile-Kataloge" des Baukastens.

Konzept
-------
Jede austauschbare Komponente (Aktivierungsfunktion, Normalisierung,
Optimierer, Verlustfunktion, Belohnung, Modell, Tensor-Operation ...) wird in
einer **Registry** unter einem Namen eingetragen – zusammen mit

* einer **Fabrikfunktion** (``factory``), die das eigentliche Objekt baut,
* einer **Argument-Konfiguration** (``config``, eine ``BaseConfig``-Klasse
  mit eigenen, erklärten Parametern),
* einem **Tooltip**, einer **Stufe** und **Tags**.

Konfigurationen referenzieren Komponenten nur über den Namen
(z. B. ``activation = "gelu"``). Dadurch

1. bleiben alle Konfigurationen als JSON/YAML speicherbar,
2. kann jede Stelle jede passende Komponente verwenden,
3. können Nutzer **eigene Komponenten** registrieren und sofort überall
   einsetzen – ohne den Baukasten selbst zu verändern::

       from ki_baukasten import registry, BaseConfig, Param, T

       class MeineAktivierungConfig(BaseConfig):
           staerke = Param(1.0, tooltip=T("Stärke der Krümmung."))

       @registry("activation").register("meine_aktivierung", config=MeineAktivierungConfig,
                                        tooltip=T("Meine eigene Aktivierung."))
       def _build(cfg):
           return MeinModul(cfg.staerke)

Aufrufkonvention der Fabriken
-----------------------------
``factory(args, *kontext, **kontext_kw)`` – das erste Argument ist immer die
Argument-Konfiguration der Komponente; danach folgt registry-spezifischer
Kontext (z. B. bei Normalisierungen die Anzahl der Features). Welcher Kontext
übergeben wird, steht in der Beschreibung der jeweiligen Registry.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .config import BaseConfig, EmptyConfig, FreeConfig
from .errors import RegistryError, suggest
from .tooltip import Level, Tooltip, NO_TOOLTIP


@dataclass
class Component:
    """Ein Eintrag in einer Registry."""

    name: str
    registry: str
    factory: Callable[..., Any]
    config_cls: type[BaseConfig] = EmptyConfig
    tooltip: Tooltip = NO_TOOLTIP
    level: Level = Level.BASIC
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    source: str = "builtin"
    extra: dict[str, Any] = field(default_factory=dict)

    def build(self, args: BaseConfig | None = None, *context: Any, **context_kw: Any) -> Any:
        """Baut die Komponente (Standard-Argumente, falls ``args`` fehlt)."""
        return self.factory(args if args is not None else self.config_cls(), *context, **context_kw)


class Registry:
    """Ein Katalog gleichartiger Komponenten (siehe Modul-Dokumentation)."""

    def __init__(self, kind: str, title: str, tooltip: Tooltip, context_doc: str = ""):
        self.kind = kind
        self.title = title
        self.tooltip = tooltip
        #: Beschreibung, welche Kontext-Argumente die Fabriken erhalten.
        self.context_doc = context_doc
        self._entries: dict[str, Component] = {}
        self._aliases: dict[str, str] = {}

    # ------------------------------------------------------------ Registrieren
    def register(
        self,
        name: str,
        factory: Callable[..., Any] | None = None,
        *,
        config: type[BaseConfig] | None = None,
        tooltip: Tooltip | None = None,
        level: Level | str = Level.BASIC,
        tags: Iterable[str] = (),
        aliases: Iterable[str] = (),
        override: bool = False,
        free_args: bool = False,
        source: str = "builtin",
        **extra: Any,
    ) -> Any:
        """Registriert eine Komponente – direkt oder als Dekorator.

        Args:
            name:      eindeutiger Name (Kleinschreibung empfohlen).
            factory:   Fabrikfunktion ``f(args, *kontext)``. Fehlt sie, wird
                       ein Dekorator zurückgegeben.
            config:    ``BaseConfig``-Klasse der Argumente (Standard: keine).
            tooltip:   Erklärung (Pflicht für gute Bedienbarkeit).
            level:     Erfahrungsstufe.
            tags:      Schlagworte zum Filtern (z. B. "vision", "rl").
            aliases:   alternative Namen (z. B. "swish" für "silu").
            override:  vorhandenen Eintrag ersetzen erlauben.
            free_args: Argumente ohne Schema erlauben (``FreeConfig``).
        """
        if factory is None:
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                self.register(
                    name, fn, config=config, tooltip=tooltip, level=level, tags=tags,
                    aliases=aliases, override=override, free_args=free_args, source=source, **extra,
                )
                return fn

            return decorator

        key = name.lower()
        if key in self._entries and not override:
            raise RegistryError(
                f"'{name}' ist in Registry '{self.kind}' bereits registriert (override=True zum Ersetzen)."
            )
        cfg_cls = FreeConfig if free_args else (config or EmptyConfig)
        entry = Component(
            name=key, registry=self.kind, factory=factory, config_cls=cfg_cls,
            tooltip=tooltip or NO_TOOLTIP, level=Level.parse(level), tags=tuple(tags),
            aliases=tuple(a.lower() for a in aliases), source=source, extra=dict(extra),
        )
        self._entries[key] = entry
        for alias in entry.aliases:
            self._aliases[alias] = key
        return factory

    def unregister(self, name: str) -> None:
        key = self._resolve_key(name)
        entry = self._entries.pop(key)
        for alias in entry.aliases:
            self._aliases.pop(alias, None)

    # ------------------------------------------------------------ Nachschlagen
    def _resolve_key(self, name: str) -> str:
        key = str(name).lower()
        key = self._aliases.get(key, key)
        if key not in self._entries:
            _autoload()
            key = self._aliases.get(key, key)
        if key not in self._entries:
            hint = suggest(key, list(self._entries) + list(self._aliases))
            msg = f"Komponente '{name}' ist in Registry '{self.kind}' ({self.title}) nicht registriert."
            if hint:
                msg += f" Meinten Sie {hint}?"
            else:
                msg += f" Verfügbar: {', '.join(sorted(self._entries))}"
            raise RegistryError(msg)
        return key

    def get(self, name: str) -> Component:
        """Liefert den Eintrag (Groß-/Kleinschreibung egal, Aliase erlaubt)."""
        return self._entries[self._resolve_key(name)]

    def __contains__(self, name: object) -> bool:
        key = str(name).lower()
        return key in self._entries or key in self._aliases

    def __iter__(self) -> Iterator[Component]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def names(self, max_level: Level | str | None = None, tag: str | None = None) -> list[str]:
        """Namen aller Einträge (optional nach Stufe/Tag gefiltert)."""
        return [c.name for c in self.components(max_level, tag)]

    def components(self, max_level: Level | str | None = None, tag: str | None = None) -> list[Component]:
        limit = Level.parse(max_level)
        return [
            c for c in self._entries.values()
            if c.level.visible_at(limit) and (tag is None or tag in c.tags)
        ]

    # ------------------------------------------------------------ Bauen
    def build(self, name_or_spec: Any, *context: Any, **context_kw: Any) -> Any:
        """Baut eine Komponente aus Name, ``dict`` oder ``ComponentSpec``."""
        from .component import ComponentSpec

        spec = ComponentSpec.from_value(self.kind, name_or_spec)
        return spec.build(*context, **context_kw)

    # ------------------------------------------------------------ Darstellung
    def describe(self, max_level: Level | str | None = None, width: int = 100) -> str:
        """Übersicht aller Einträge mit Kurz-Tooltip (für CLI ``komponenten``)."""
        lines = [f"{self.title} ({self.kind}) – {self.tooltip.short()}"]
        if self.context_doc:
            lines.append(f"  Kontext der Fabriken: {self.context_doc}")
        for c in self.components(max_level):
            badge = "" if c.level is Level.BASIC else f" [{c.level.label}]"
            alias = f" (auch: {', '.join(c.aliases)})" if c.aliases else ""
            text = f"  • {c.name}{alias}{badge}: {c.tooltip.short()}"
            lines.append(text if len(text) <= width else text[: width - 1] + "…")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Registry {self.kind}: {len(self)} Einträge>"


# ---------------------------------------------------------------------------
# Globale Registry-Verwaltung
# ---------------------------------------------------------------------------

_REGISTRIES: dict[str, Registry] = {}

#: Module, die beim ersten fehlgeschlagenen Nachschlagen automatisch geladen
#: werden. So funktionieren Konfigurationen auch, wenn der Nutzer nur ein
#: Teilmodul importiert hat.
_BUILTIN_MODULES = (
    "ki_baukasten.components",
    "ki_baukasten.models",
    "ki_baukasten.training",
    "ki_baukasten.training.rl",
    "ki_baukasten.memory",
    "ki_baukasten.tensor",
    "ki_baukasten.data",
    "ki_baukasten.presets",
)
_autoloaded = False


def _autoload() -> None:
    """Importiert einmalig alle eingebauten Module (füllt die Registries)."""
    global _autoloaded
    if _autoloaded:
        return
    _autoloaded = True
    for mod in _BUILTIN_MODULES:
        importlib.import_module(mod)


def create_registry(kind: str, title: str, tooltip: Tooltip, context_doc: str = "") -> Registry:
    """Legt eine Registry an (oder liefert die bestehende gleichen Namens)."""
    if kind not in _REGISTRIES:
        _REGISTRIES[kind] = Registry(kind, title, tooltip, context_doc)
    return _REGISTRIES[kind]


def registry(kind: str) -> Registry:
    """Liefert die Registry ``kind`` (z. B. ``registry("activation")``)."""
    if kind not in _REGISTRIES:
        _autoload()
    if kind not in _REGISTRIES:
        hint = suggest(kind, _REGISTRIES)
        raise RegistryError(
            f"Unbekannte Registry '{kind}'." + (f" Meinten Sie {hint}?" if hint else f" Vorhanden: {', '.join(sorted(_REGISTRIES))}")
        )
    return _REGISTRIES[kind]


def all_registries() -> dict[str, Registry]:
    """Alle Registries (nach dem Laden aller eingebauten Module)."""
    _autoload()
    return dict(_REGISTRIES)


def register(kind: str, name: str, factory: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Kurzform: ``register("activation", "meine", fabrik, tooltip=T(...))``.

    Eigene Komponenten werden automatisch mit ``source="user"`` markiert.
    """
    kwargs.setdefault("source", "user")
    return registry(kind).register(name, factory, **kwargs)
