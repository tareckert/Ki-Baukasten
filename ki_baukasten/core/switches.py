"""
ki_baukasten.core.switches
==========================

**Schalter für jede Einstellung**: An · Aus · Eigen (benutzerdefiniert).

Jede Einstellung hat einen *Zustand*, der sich aus ihrem Wert ergibt:

==========  ===============================================================================
Zustand     Bedeutung
==========  ===============================================================================
``an``      empfohlener Wert ist aktiv (Standardwert bzw. der hinterlegte "An"-Wert)
``aus``     die Technik ist abgeschaltet (Dropout 0, Mixup 0, Komponente ``None``, ``"none"`` …)
``eigen``   ein eigener, benutzerdefinierter Wert
==========  ===============================================================================

Woher kommen die Aus- und An-Werte?
-----------------------------------
1. ausdrücklich am Parameter: ``Param(0.1, off=0.0, on=0.1, …)``
2. sonst abgeleitet – Ja/Nein → ``False``; Auswahl mit ``none``/``off``/``aus``/``ignore``
   → diese Option; optionale Komponenten → ``None``; Komponentenlisten → ``[]``;
   Stärken (Dropout, Weight-Decay, Rauschen, Glättung, Strafterme …) → ``0``;
   optionale Werte → ``None`` (= "nicht setzen / automatisch").
3. Einstellungen ohne sinnvollen Aus-Zustand (z. B. ``d_model``, Lernrate) sind
   **Pflicht** – sie kennen nur ``an`` (Standard) und ``eigen``.

Ist der Standardwert selbst "aus" (z. B. Mixup = 0), braucht ``an`` einen
empfohlenen Wert: aus ``on=…``, der Tabelle ``ON_VALUES`` oder dem zuletzt
benutzten eigenen Wert (der beim Ausschalten gemerkt wird).

Nutzung (meist über ``Project``)::

    p.switch("training.regularization.mixup_alpha", "an")    # 0.2
    p.switch("**.dropout", "aus")                            # alle Dropouts aus
    p.set("training.ema", "an")                              # Gruppen mit 'enabled' direkt
    p.switches(state="aus")                                  # Übersicht
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import UNSET, BaseConfig, ConfigError, Param, ParamRef, _is_spec, _to_plain
from .tooltip import Level

#: Mögliche Zustände (mit Anzeigenamen).
STATES = {"an": "An", "aus": "Aus", "eigen": "Eigen"}
#: Eingaben, die als Schalter-Befehl verstanden werden (auch in ``Project.set``).
STATE_ALIASES = {
    "an": "an", "ein": "an", "on": "an", "aktiv": "an", "empfohlen": "an",
    "aus": "aus", "off": "aus", "deaktiviert": "aus",
    "eigen": "eigen", "benutzerdefiniert": "eigen", "custom": "eigen",
    "standard": "standard", "default": "standard", "zuruecksetzen": "standard", "zurücksetzen": "standard",
}

_OFF_CHOICES = ("none", "off", "aus", "keine", "kein", "never", "nie", "disabled", "ignore")

#: Namen, die eine Stärke beschreiben (0 = wirkungslos = Aus).
_STRENGTH_RX = re.compile(
    r"(dropout|drop_path|droppath|stochastic_depth|smoothing|noise|weight_decay|^l1$|^l2$|orthogonality|mixup|"
    r"cutmix|sam_rho|z_loss|penalty|entropy|aux_loss|load_balanc|jitter|warmup|feature_weight|kl_weight|"
    r"commitment|diversity|sparsity|_coef$|^weight$|_weight$|_prob$|^prob$|shift|erasing|cutout|lisa|grokfast|"
    r"spike|^swa_|lookahead|^gc_|rdrop|consistency|^ema_)"
)
#: Ausnahmen: Namen, die zwar wie eine Stärke aussehen, aber Kernwerte sind (0 wäre kein "Aus", sondern kaputt).
_NOT_STRENGTH_RX = re.compile(r"(noise_decay|start_factor|mask_prob|value_coef|commitment|_init$|^mean$)")

#: Empfohlene "An"-Werte für Einstellungen, deren Standard "aus" ist (nach Parametername).
ON_VALUES: dict[str, Any] = {
    "dropout": 0.1, "attn_dropout": 0.1, "attention_dropout": 0.1, "resid_dropout": 0.1, "ffn_dropout": 0.1,
    "embed_dropout": 0.1, "embedding_dropout": 0.1, "drop_path": 0.1, "stochastic_depth": 0.1,
    "label_smoothing": 0.1, "mixup_alpha": 0.2, "cutmix_alpha": 1.0, "sam_rho": 0.05, "z_loss_coef": 1e-4,
    "weight_decay": 0.01, "l1": 1e-5, "l2": 1e-4, "orthogonality": 1e-4, "input_noise_std": 0.05,
    "gradient_noise_std": 0.01, "flooding_level": 0.05, "layerwise_lr_decay": 0.8, "max_steps": 1000,
    "eval_every_steps": 500, "eval_max_batches": 50, "num_workers": 2, "logit_softcap": 30.0,
    "feature_weight": 0.5, "warmup_steps": 100, "warmup_ratio": 0.05, "seed": 42,
    "memory_budget_gb": 8.0, "cpu_threads": 4, "max_memory_fraction": 0.9, "empty_cache_every_n_steps": 1000,
    "keep_resident": 1, "quant_group_size": 128, "quant_skip": "lm_head", "main_target": "smartphone",
    "qk_norm": "rmsnorm", "aux_loss_coef": 0.01, "load_balance_coef": 0.01, "norm": "layernorm",
    "embedding_norm": "layernorm", "head_dropout": 0.1, "input_dropout": 0.1, "edge_dropout": 0.1,
    "member_dropout": 0.1, "modality_dropout": 0.1, "instance_noise_std": 0.1, "router_noise_std": 0.1,
    "gradient_penalty": 10.0, "r1_penalty": 1.0, "free_bits": 0.5, "min_delta": 1e-4,
}


@dataclass
class SwitchInfo:
    """Zustand einer Einstellung (Ergebnis von ``switch_table``)."""

    path: str
    state: str                 # an | aus | eigen
    value: Any
    switchable: bool
    off: Any
    on: Any                    # UNSET = kein empfohlener An-Wert bekannt
    level: Level
    group: str | None
    what: str

    @property
    def label(self) -> str:
        return STATES[self.state] if self.switchable or self.state != "an" else "An (Pflicht)"

    def __str__(self) -> str:
        mark = {"an": "●", "aus": "○", "eigen": "◆"}[self.state]
        req = "" if self.switchable else "  [Pflicht]"
        return f"{mark} {STATES[self.state]:<5} {self.path} = {_show(self.value)}{req}"


def _show(value: Any) -> str:
    if _is_spec(value):
        return f"⟨{value.name}⟩"
    if isinstance(value, list) and value and _is_spec(value[0]):
        return "[" + ", ".join(f"⟨{v.name}⟩" for v in value) + "]"
    text = repr(value)
    return text if len(text) <= 50 else text[:49] + "…"


def _plain(value: Any) -> Any:
    return _to_plain(value)


def _same(a: Any, b: Any) -> bool:
    if _is_spec(a) and isinstance(b, str) or _is_spec(b) and isinstance(a, str):
        spec, name = (a, b) if _is_spec(a) else (b, a)
        return spec.name == name
    if _is_spec(a) or _is_spec(b):
        if not (_is_spec(a) and _is_spec(b)):
            return False
        return a.name == b.name and a.args.to_dict() == b.args.to_dict()
    return _plain(a) == _plain(b)


# ---------------------------------------------------------------------------
# Aus-/An-Werte
# ---------------------------------------------------------------------------

def off_value(param: Param) -> Any:
    """Wert für **Aus** oder ``UNSET`` (= Pflicht-Einstellung)."""
    if param.switchable is False:
        return UNSET
    if param.off is not UNSET:
        return param.off
    kind = getattr(param, "kind", "value")
    if kind in ("nested",):
        return UNSET
    if kind == "component":
        if param.nullable:
            return None
        from .registry import registry

        reg = registry(param.registry_kind)
        for cand in ("none", "keep", "identity"):      # neutrale Komponente der Registry (kanonischer Name)
            if cand in reg:
                return reg.get(cand).name
        return UNSET
    if kind in ("component_list", "config_list"):
        return [] if getattr(param, "min_items", 0) == 0 else UNSET
    types = param.types
    if bool in types and len(types) == 1:
        return False
    choices = param._choices if not callable(param._choices) else None
    if choices:
        for c in _OFF_CHOICES:
            if c in choices:
                return c
        return None if param.nullable else UNSET
    default = param.make_default()
    numeric = any(t in (int, float) for t in types) and bool not in types
    if numeric and (param.min is None or param.min <= 0) and (param.max is None or param.max >= 0):
        if _STRENGTH_RX.search(param.name or "") and not _NOT_STRENGTH_RX.search(param.name or ""):
            return 0.0 if float in types else 0
    if param.nullable:
        return None
    return UNSET


def on_value(param: Param, memory: Any = UNSET) -> Any:
    """Empfohlener Wert für **An** (``UNSET``, wenn keiner bekannt ist)."""
    if param.on is not UNSET:
        return param.on
    off = off_value(param)
    default = param.make_default()
    if off is UNSET or not _same(default, off):
        return default
    if memory is not UNSET:
        return memory
    if param.name in ON_VALUES:
        return ON_VALUES[param.name]
    types = getattr(param, "types", ())
    if bool in types and len(types) == 1:
        return True
    choices = param._choices if not callable(param._choices) else None
    if choices:
        for c in choices:
            if c != off and c not in _OFF_CHOICES:
                return c
    return UNSET


def state_of(param: Param, value: Any) -> str:
    """``an`` | ``aus`` | ``eigen`` für einen Wert.

    Bei Komponenten zählt die *Auswahl* (Name); ihre Argumente sind eigene Einstellungen mit eigenem Zustand.
    """
    off = off_value(param)
    if off is not UNSET and _same(value, off):
        return "aus"
    if _is_spec(value):
        default = param.make_default()
        on = on_value(param)
        names = {getattr(v, "name", v) for v in (default, on) if v is not UNSET and v is not None}
        return "an" if value.name in names else "eigen"
    on = on_value(param)
    if on is not UNSET and _same(value, on):
        return "an"
    if _same(value, param.make_default()):
        return "an"
    return "eigen"


_NONE_MEANING = re.compile(r"None\s*=\s*([^)\.;]+)")


def off_label(param: Param) -> str:
    """Lesbare Bedeutung des Aus-Werts, z. B. ``"0 (wirkungslos)"`` oder ``"None = batch_size"``."""
    off = off_value(param)
    if off is UNSET:
        return "– (Pflicht-Einstellung)"
    if off is None:
        tip = param.tooltip
        m = _NONE_MEANING.search(" ".join(t for t in (tip.what, tip.tip) if t)) if tip else None
        return f"None = {m.group(1).strip()}" if m else "None (nicht gesetzt)"
    if off is False:
        return "nein"
    if off == [] and isinstance(off, list):
        return "leer (keine Einträge)"
    if isinstance(off, (int, float)):
        return f"{off} (wirkungslos)"
    help_text = param.choice_help.get(off) if param.choice_help else None
    return f"{off!r}" + (f" – {help_text}" if help_text else "")


# ---------------------------------------------------------------------------
# Tabelle und Umschalten
# ---------------------------------------------------------------------------

def _is_leaf(ref: ParamRef) -> bool:
    kind = getattr(ref.param, "kind", "value")
    if kind in ("nested", "config_list"):
        return False
    return not (ref.param.owner_name in ("liste",) or ref.name.isdigit())


def switch_table(root: BaseConfig, pattern: str | None = None, state: str | None = None,
                 max_level: Level | str | None = None, only_switchable: bool = False) -> list[SwitchInfo]:
    """Alle Einstellungen mit Zustand (optional gefiltert nach Muster, Zustand, Stufe)."""
    wanted = set(root.find(pattern, max_level)) if pattern else None
    out: list[SwitchInfo] = []
    want_state = STATE_ALIASES.get(state, state) if state else None
    for ref in root.walk(max_level=max_level):
        if not _is_leaf(ref):
            continue
        if wanted is not None and ref.path not in wanted and not any(ref.path.startswith(w + ".") for w in wanted):
            continue
        p = ref.param
        off = off_value(p)
        info = SwitchInfo(ref.path, state_of(p, ref.value), ref.value, off is not UNSET, off, on_value(p),
                          ref.level, p.group, p.tooltip.short() if p.tooltip else "")
        if only_switchable and not info.switchable:
            continue
        if want_state and info.state != want_state:
            continue
        out.append(info)
    return out


def summary(root: BaseConfig, max_level: Level | str | None = None) -> dict[str, int]:
    """Anzahl Einstellungen je Zustand + Anzahl Pflicht-Einstellungen."""
    counts = {"an": 0, "aus": 0, "eigen": 0, "pflicht": 0, "schaltbar": 0}
    for info in switch_table(root, max_level=max_level):
        counts[info.state] += 1
        counts["schaltbar" if info.switchable else "pflicht"] += 1
    return counts


def _enabled_child(root: BaseConfig, path: str) -> str | None:
    """Für Gruppen mit einem Ja/Nein-Schalter ``enabled`` (z. B. ``training.ema``) dessen Pfad."""
    try:
        obj = root.get(path)
    except ConfigError:
        return None
    target = obj.args if _is_spec(obj) else obj
    if isinstance(target, BaseConfig) and "enabled" in type(target)._params:
        return f"{path}.enabled"
    return None


def parse_state(text: Any) -> str | None:
    """Erkennt Schalter-Befehle (``"aus"``, ``"an"``, ``"eigen"``, ``"standard"``) – sonst ``None``."""
    if not isinstance(text, str):
        return None
    return STATE_ALIASES.get(text.strip().lower())


def is_switch_command(root: BaseConfig, path: str, value: Any) -> bool:
    """``True``, wenn ``value`` (z. B. ``"aus"``) als Schalter und nicht als Text gemeint ist."""
    if parse_state(value) is None:
        return False
    if _enabled_child(root, path):
        return True
    try:
        ref = root.param_at(path)
    except ConfigError:
        return False
    p = ref.param
    if getattr(p, "kind", "value") in ("nested",):
        return True
    choices = p.choices
    if choices:
        return value not in choices and value.lower() not in {str(c).lower() for c in choices}
    return str not in getattr(p, "types", ()) or getattr(p, "kind", "value") != "value"


def set_state(root: BaseConfig, path: str, state: str, value: Any = UNSET,
              memory: dict[str, Any] | None = None) -> list[str]:
    """Schaltet eine Einstellung (oder eine Gruppe) um.

    Args:
        state:  ``an`` | ``aus`` | ``eigen`` | ``standard`` (auch deutsche/englische Varianten).
        value:  Wert für ``eigen`` (ohne Angabe: zuletzt gemerkter eigener Wert).
        memory: Merkzettel ``{pfad: wert}`` – beim Ausschalten wird der alte Wert hier abgelegt,
                damit ``an``/``eigen`` ihn wiederherstellen kann.

    Returns:
        Die geänderten Pfade.
    """
    st = STATE_ALIASES.get(str(state).strip().lower())
    if st is None:
        raise ConfigError(f"Unbekannter Schalterzustand '{state}'. Erlaubt: an, aus, eigen, standard.", path)
    memory = memory if memory is not None else {}
    enabled = _enabled_child(root, path)
    if enabled and st in ("an", "aus"):
        root.set(enabled, st == "an")
        return [enabled]
    try:
        ref = root.param_at(path)
    except ConfigError:
        raise
    p = ref.param
    if getattr(p, "kind", "value") == "nested" or isinstance(ref.value, BaseConfig) and not _is_spec(ref.value):
        return _set_group(root, path, st, memory)
    current = ref.value
    if st == "standard":
        root.set(path, p.make_default())
        return [path]
    if st == "aus":
        off = off_value(p)
        if off is UNSET:
            raise ConfigError("ist eine Pflicht-Einstellung und kann nicht ausgeschaltet werden "
                              "(nur 'an' = Standard oder 'eigen' = eigener Wert).", path)
        if not _same(current, off):
            memory[path] = _plain(current)
        root.set(path, off)
        return [path]
    if st == "an":
        on = on_value(p, memory.get(path, UNSET))
        if on is UNSET:
            raise ConfigError("hat keinen empfohlenen An-Wert – bitte einen eigenen Wert setzen "
                              f"(Schalter 'eigen' oder Project.set('{path}', wert)).", path)
        root.set(path, on)
        return [path]
    # eigen
    if value is UNSET:
        value = memory.get(path, UNSET)
        if value is UNSET:
            return []          # nichts zu tun: aktueller Wert bleibt, Oberfläche erlaubt jetzt die Eingabe
    root.set(path, value)
    return [path]


def _set_group(root: BaseConfig, path: str, st: str, memory: dict[str, Any]) -> list[str]:
    """Alle schaltbaren Einstellungen einer Unter-Konfiguration umschalten.

    ``aus`` schaltet alles Schaltbare ab (alte Werte werden gemerkt); ``an`` stellt genau diese gemerkten
    Werte wieder her (ohne Merkzettel: Standardwerte der Gruppe); ``standard`` setzt die Gruppe zurück.
    """
    ref = root.param_at(path)
    if st == "standard":
        root.set(path, ref.param.make_default())
        return [path]
    changed = []
    infos = switch_table(root, f"{path}.**", only_switchable=True)
    if st == "aus":
        for info in infos:
            if info.state != "aus":
                changed += set_state(root, info.path, "aus", memory=memory)
        return changed
    remembered = [i for i in infos if i.path in memory and i.state == "aus"]
    if not remembered:
        root.set(path, ref.param.make_default())
        return [path]
    for info in remembered:
        root.set(info.path, memory.pop(info.path))
        changed.append(info.path)
    return changed
