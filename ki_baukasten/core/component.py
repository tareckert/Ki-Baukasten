"""
ki_baukasten.core.component
===========================

``ComponentSpec`` und ``ComponentParam`` – austauschbare Bausteine in einer
Konfiguration.

Was ist eine ComponentSpec?
---------------------------
Eine ``ComponentSpec`` ist die *Beschreibung* einer Komponente:
"nimm aus der Registry ``optimizer`` den Eintrag ``adamw`` mit diesen
Argumenten". Sie enthält **noch kein** PyTorch-Objekt, ist also leicht,
kopierbar und als JSON speicherbar::

    {"name": "adamw", "args": {"lr": 0.0003, "weight_decay": 0.1}}

Kurzformen beim Eingeben (alle gleichwertig):

* ``"adamw"``                                  – nur der Name, Standard-Argumente
* ``{"name": "adamw", "lr": 3e-4}``            – flache Argumente
* ``{"name": "adamw", "args": {"lr": 3e-4}}``  – ausführlich (so wird gespeichert)

Wechselt man die Komponente (``spec.switch("sgd")``), werden Argumente mit
gleichem Namen übernommen (``lr`` bleibt erhalten), alle anderen bekommen die
Standardwerte der neuen Komponente.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Callable, Iterable

from .config import BaseConfig, ConfigError, Param
from .registry import Component, registry
from .tooltip import Level, Tooltip


class ComponentSpec:
    """Beschreibung einer registrierten Komponente samt Argumenten."""

    _is_component_spec = True  # Duck-Typing-Marker für core.config

    def __init__(self, registry_kind: str, name: str, args: Any = None):
        self.registry = registry_kind
        comp = registry(registry_kind).get(name)
        self.name: str = comp.name
        self.args: BaseConfig = self._make_args(comp, args)

    # ------------------------------------------------------------ Aufbau
    @staticmethod
    def _make_args(comp: Component, args: Any, path: str | None = None) -> BaseConfig:
        cls = comp.config_cls
        if args is None:
            return cls()
        if isinstance(args, cls):
            return args
        if isinstance(args, BaseConfig):
            args = args.to_dict()
        if isinstance(args, Mapping):
            return cls.from_dict(args, path=path)
        raise ConfigError(f"Argumente für '{comp.name}' müssen ein dict sein, nicht {type(args).__name__}.", path)

    @classmethod
    def from_value(cls, registry_kind: str, value: Any, path: str | None = None) -> "ComponentSpec":
        """Erzeugt eine Spec aus Name, dict oder Spec (siehe Modul-Doku)."""
        if isinstance(value, ComponentSpec):
            if value.registry != registry_kind:
                raise ConfigError(
                    f"Komponente aus Registry '{value.registry}' passt nicht – erwartet '{registry_kind}'.", path
                )
            return value
        if isinstance(value, str):
            try:
                return cls(registry_kind, value)
            except KeyError as exc:  # RegistryError
                raise ConfigError(str(exc), path) from None
        if isinstance(value, Mapping):
            if "name" not in value:
                raise ConfigError(f"Komponenten-dict braucht den Schlüssel 'name' (Registry '{registry_kind}').", path)
            args = value.get("args")
            flat = {k: v for k, v in value.items() if k not in ("name", "args")}
            if args is None:
                args = flat
            elif flat:
                args = {**dict(args), **flat}
            try:
                comp = registry(registry_kind).get(value["name"])
            except KeyError as exc:
                raise ConfigError(str(exc), path) from None
            spec = cls.__new__(cls)
            spec.registry = registry_kind
            spec.name = comp.name
            spec.args = cls._make_args(comp, args, path)
            return spec
        raise ConfigError(
            f"erwartet Komponentenname (Text) oder dict für Registry '{registry_kind}', erhalten {type(value).__name__}.",
            path,
        )

    # ------------------------------------------------------------ Eigenschaften
    @property
    def component(self) -> Component:
        return registry(self.registry).get(self.name)

    @property
    def tooltip(self) -> Tooltip:
        return self.component.tooltip

    # ------------------------------------------------------------ Bauen
    def build(self, *context: Any, **context_kw: Any) -> Any:
        """Ruft die Fabrikfunktion mit den Argumenten und dem Kontext auf."""
        return self.component.factory(self.args, *context, **context_kw)

    # ------------------------------------------------------------ Ändern
    def switch(self, name: str, keep_compatible: bool = True) -> "ComponentSpec":
        """Wechselt die Komponente; kompatible Argumente bleiben erhalten."""
        comp = registry(self.registry).get(name)
        new_args = comp.config_cls()
        if keep_compatible:
            for key, value in self.args.values().items():
                if key in type(new_args)._params:
                    try:
                        new_args._set_value(key, copy.deepcopy(value))
                    except ConfigError:
                        pass  # inkompatibler Wert -> Standard der neuen Komponente
        self.name = comp.name
        self.args = new_args
        return self

    def set_args(self, value: Any) -> None:
        self.args = self._make_args(self.component, value)

    def merge(self, mapping: Mapping[str, Any], path: str | None = None) -> "ComponentSpec":
        """Mischt ein dict ein: ``{"name": ...}`` wechselt, Rest/``args`` aktualisiert.

        Beim Wechsel über ein dict (z. B. beim Laden einer Datei) starten die
        Argumente bei den Standardwerten der *neuen* Komponente – es gelten nur
        die im dict angegebenen Werte. (Interaktives Umschalten per
        ``set("x.name", ...)`` behält dagegen kompatible Werte, siehe ``switch``.)
        """
        if "name" in mapping and str(mapping["name"]).lower() != self.name:
            self.switch(mapping["name"], keep_compatible=False)
        args = mapping.get("args")
        flat = {k: v for k, v in mapping.items() if k not in ("name", "args")}
        if args:
            self.args.update(args, _path=path)
        if flat:
            self.args.update(flat, _path=path)
        return self

    def get(self, path: str) -> Any:
        return self.args.get(path)

    def set(self, path: str, value: Any) -> "ComponentSpec":
        self.args.set(path, value)
        return self

    # ------------------------------------------------------------ Serialisierung
    def to_dict(self, only_changed: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        args = self.args.to_dict(only_changed=only_changed)
        if args or (not only_changed and type(self.args)._params):
            out["args"] = args
        return out

    def copy(self) -> "ComponentSpec":
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> "ComponentSpec":
        new = ComponentSpec.__new__(ComponentSpec)
        new.registry = self.registry
        new.name = self.name
        new.args = copy.deepcopy(self.args, memo)
        return new

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ComponentSpec)
            and other.registry == self.registry
            and other.name == self.name
            and other.args.to_dict() == self.args.to_dict()
        )

    def __repr__(self) -> str:
        changed = self.args.to_dict(only_changed=True)
        inner = ", ".join(f"{k}={v!r}" for k, v in changed.items())
        return f"{self.registry}:{self.name}({inner})"


# ---------------------------------------------------------------------------
# ComponentParam / ComponentListParam
# ---------------------------------------------------------------------------

def _options_text(kind: str, tags: tuple[str, ...] | None) -> str:
    reg = registry(kind)
    lines = [f"  Wählbare Komponenten ({reg.title}):"]
    for comp in reg.components():
        if tags and not set(tags) & set(comp.tags):
            continue
        badge = "" if comp.level is Level.BASIC else f" [{comp.level.label}]"
        lines.append(f"    • {comp.name}{badge}: {comp.tooltip.short()}")
    return "\n".join(lines)


class ComponentParam(Param):
    """Einstellung, deren Wert eine Komponente aus einer Registry ist.

    Args:
        registry_kind: Name der Registry (z. B. ``"activation"``).
        default:       Name der Standard-Komponente (``None`` = keine).
        default_args:  Standard-Argumente für die Standard-Komponente.
        tags:          nur Komponenten mit mindestens einem dieser Tags als
                       *empfohlene* Auswahl anzeigen (alle bleiben erlaubt).
    """

    kind = "component"

    def __init__(
        self,
        registry_kind: str,
        default: str | None = None,
        *,
        tooltip: Tooltip,
        default_args: Mapping[str, Any] | None = None,
        level: Level | str = Level.BASIC,
        group: str | None = None,
        nullable: bool | None = None,
        tags: Iterable[str] | None = None,
    ):
        self.registry_kind = registry_kind
        self.default_name = default
        self.default_args = dict(default_args or {})
        self.tags = tuple(tags) if tags else None
        factory: Callable[[], Any] = (
            (lambda: ComponentSpec.from_value(registry_kind, {"name": default, "args": dict(self.default_args)}))
            if default is not None else (lambda: None)
        )
        super().__init__(
            None, tooltip=tooltip, type=object, level=level, group=group,
            nullable=(default is None) if nullable is None else nullable,
            default_factory=factory,
        )

    @property
    def choices(self) -> list[Any] | None:
        return registry(self.registry_kind).names()

    def coerce(self, value: Any, path: str | None = None) -> Any:
        if value is None:
            if self.nullable:
                return None
            raise ConfigError(f"eine Komponente aus '{self.registry_kind}' ist Pflicht.", path)
        return ComponentSpec.from_value(self.registry_kind, value, path)

    def type_label(self) -> str:
        return f"Komponente aus Registry '{self.registry_kind}'" + (" (oder None)" if self.nullable else "")

    def constraint_label(self) -> str:
        names = self.choices or []
        shown = " | ".join(names[:14]) + (" | …" if len(names) > 14 else "")
        return f"Auswahl: {shown}"

    def default_repr(self) -> str:
        return repr(self.default_name)

    def options_help(self) -> str:
        return _options_text(self.registry_kind, self.tags)


class ComponentListParam(Param):
    """Liste von Komponenten derselben Registry.

    Beispiele: mehrere Verlustfunktionen mit Gewichten, mehrere
    Belohnungsterme, mehrere Callbacks, die Mitglieder eines Ensembles.
    """

    kind = "component_list"

    def __init__(
        self,
        registry_kind: str,
        *,
        tooltip: Tooltip,
        default: Iterable[Any] = (),
        level: Level | str = Level.BASIC,
        group: str | None = None,
        min_items: int = 0,
    ):
        self.registry_kind = registry_kind
        self.min_items = min_items
        default_items = list(default)
        super().__init__(
            None, tooltip=tooltip, type=list, level=level, group=group, nullable=False,
            default_factory=lambda: [ComponentSpec.from_value(registry_kind, copy.deepcopy(v)) for v in default_items],
        )

    @property
    def choices(self) -> list[Any] | None:
        return None  # Auswahl gilt pro Element, nicht für die Liste

    def coerce(self, value: Any, path: str | None = None) -> Any:
        if value is None:
            value = []
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"erwartet eine Liste von Komponenten aus '{self.registry_kind}'.", path)
        out = [
            ComponentSpec.from_value(self.registry_kind, item, f"{path}.{i}" if path else str(i))
            for i, item in enumerate(value)
        ]
        if len(out) < self.min_items:
            raise ConfigError(f"braucht mindestens {self.min_items} Elemente.", path)
        return out

    def type_label(self) -> str:
        return f"Liste von Komponenten aus Registry '{self.registry_kind}'"

    def constraint_label(self) -> str:
        names = registry(self.registry_kind).names()
        return "Elemente: " + " | ".join(names[:14]) + (" | …" if len(names) > 14 else "")

    def options_help(self) -> str:
        return _options_text(self.registry_kind, None)


def component_item_param(spec: ComponentSpec) -> Param:
    """Pseudo-Parameter für ein Listenelement (nur für Anzeige/``walk``)."""
    p = Param(None, tooltip=spec.component.tooltip, type=object, level=spec.component.level)
    p.name = spec.name
    p.owner_name = spec.registry
    return p
