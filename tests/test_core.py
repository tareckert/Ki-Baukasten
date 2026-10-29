"""Tests des Konfigurationskerns: Param-Typen, Pfade, Serialisierung, Validierung, Selektoren."""

import json

import pytest

from ki_baukasten import BaseConfig, ConfigError, Level, NestedParam, Param, T, registry
from ki_baukasten.core import (ComponentListParam, ComponentParam, ConfigListParam, FreeConfig, UnknownParameterError,
                               resolve_layer_config, select_indices)


class Inner(BaseConfig):
    heads = Param(8, min=1, tooltip=T("Köpfe."))
    window = Param(None, type=int, min=1, level="expert", tooltip=T("Fenster."))

    def check(self):
        return ["heads darf nicht 3 sein"] if self.heads == 3 else []


class Item(BaseConfig):
    size = Param(4, min=1, tooltip=T("Größe."))


class Outer(BaseConfig):
    inner = NestedParam(Inner)
    act = ComponentParam("activation", "relu", tooltip=T("Aktivierung."))
    acts = ComponentListParam("activation", default=["gelu"], tooltip=T("Liste."))
    dims = Param([8, 16], item_type=int, min=1, tooltip=T("Breiten."))
    items = ConfigListParam(Item, default_factory=lambda: [Item(), Item(size=9)], tooltip=T("Elemente."))
    mode = Param("a", choices={"a": "Option A", "b": "Option B"}, tooltip=T("Modus."))
    overrides = Param({}, tooltip=T("Overrides."))
    rate = Param(0.5, min=0.0, max=1.0, tooltip=T("Rate."))


def test_types_and_coercion():
    o = Outer()
    o.set("inner.heads", "16")
    assert o.inner.heads == 16
    o.rate = 1
    assert isinstance(o.rate, float)
    o.set("mode", "B")
    assert o.mode == "b"
    with pytest.raises(ConfigError):
        o.rate = 2.0
    with pytest.raises(ConfigError):
        o.inner.heads = True  # bool ist kein int
    with pytest.raises(ConfigError):
        o.mode = "c"


def test_typo_suggestion():
    with pytest.raises(UnknownParameterError, match="heads"):
        Inner().haeds = 3


def test_paths_lists_components_dicts():
    o = Outer()
    o.set("dims.1", 32)
    assert o.dims == [8, 32]
    with pytest.raises(ConfigError):
        o.set("dims.0", 0)
    o.set("items.1.size", 2)
    assert o.items[1].size == 2
    o.set("act.name", "leaky_relu")
    o.set("act.negative_slope", 0.2)
    assert o.act.name == "leaky_relu" and o.act.args.negative_slope == 0.2
    o.set("overrides.3.heads", 4)
    assert o.overrides == {"3": {"heads": 4}}
    assert o.get("acts.0.name") == "gelu"


def test_component_switch_keeps_compatible_args_only_interactively():
    o = Outer()
    o.set("act", {"name": "elu", "alpha": 2.0})
    o.set("act.name", "celu")          # interaktiv: alpha bleibt
    assert o.act.args.alpha == 2.0
    o.update({"act": {"name": "elu"}})  # per dict: Standardwerte der neuen Komponente
    assert o.act.args.alpha == 1.0


def test_roundtrip_and_only_changed():
    o = Outer()
    o.set("inner.heads", 4)
    o.set("act", {"name": "gelu", "approximate": "tanh"})
    d = o.to_dict()
    assert Outer.from_dict(d) == o
    small = o.to_dict(only_changed=True)
    assert small == {"inner": {"heads": 4}, "act": {"name": "gelu", "args": {"approximate": "tanh"}}}
    assert Outer.from_dict(small) == o
    assert Outer.from_dict(json.loads(o.to_json())) == o


def test_validation_and_problems():
    o = Outer()
    o.inner.heads = 3
    assert any("heads" in p for p in o.problems())
    with pytest.raises(ConfigError):
        o.validate()


def test_explain_and_describe():
    o = Outer()
    text = o.explain("mode")
    assert "Option A" in text and "Auswahl" in text
    assert "Wählbare Komponenten" in o.explain("act")
    tree = o.describe(max_level="basic")
    assert "window" not in tree        # Experten-Stufe ausgeblendet
    assert "window" in o.describe(max_level=Level.EXPERIMENTAL)


def test_free_config_and_registry_errors():
    f = FreeConfig(a=1)
    f.set("b", [1, 2])
    assert f.to_dict() == {"a": 1, "b": [1, 2]}
    with pytest.raises(KeyError, match="Meinten Sie"):
        registry("activation").get("gleu")


def test_selectors_and_overrides():
    assert select_indices("0,2-3,-1", 6) == [0, 2, 3, 5]
    assert select_indices("::2", 5) == [0, 2, 4]
    assert select_indices("ungerade", 5) == [1, 3]
    assert select_indices("letzte,erste", 4) == [0, 3]
    base = Inner()
    cfg = resolve_layer_config(base, {"*": {"heads": 2}, "1::2": {"window": 16}}, 3, 6)
    assert (cfg.heads, cfg.window) == (2, 16)
    assert base.heads == 8  # Basis unverändert


def test_tooltip_short_handles_abbreviations():
    assert T("Ein Wert (z. B. 5) für alles. Rest.").short() == "Ein Wert (z. B. 5) für alles."
