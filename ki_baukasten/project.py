"""
ki_baukasten.project
====================

Die **Projekt-Fassade**: eine komplette KI (Modell + Daten + Training +
Speicher + Tensor-Chirurgie) als *eine* Konfiguration – und eine Klasse, die
damit alles erledigt.

Schnellstart
------------
::

    from ki_baukasten import Project

    p = Project.from_preset("sprachmodell_mini")       # Vorlage je Anwendungsgebiet
    p.set("model.n_layers", 6)                          # jede Einstellung per Pfad
    p.set("model.block.mixer.num_heads", 8)
    print(p.explain("model.block.norm_position"))       # Tooltip: was/warum/wofür
    p.train()                                           # bauen + trainieren
    print(p.generate("Der Baukasten", max_new_tokens=80))
    p.save("ausgaben/mein_llm")                         # Konfiguration + Gewichte

Automatische Dimensionsanpassung
--------------------------------
Mit ``auto_fit_dims=True`` (Standard) übernimmt das Projekt passende Größen
aus den Daten bzw. der RL-Umgebung (``input_dim``, ``output_dim``,
``vocab_size``, ``in_channels``, ``obs_dim``, ``action_dim`` …) und meldet
jede Änderung. So passt eine Vorlage auch zu eigenen Daten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .core import BaseConfig, ComponentParam, ConfigError, ConfigListParam, Level, NestedParam, Param, T
from .analysis.targets import DeploymentConfig
from .compress import CompressionConfig
from .memory import MemoryConfig, MemoryManager, estimate_memory
from .tensor import ModuleRuleConfig, ModuleSurgeon, TensorRuleConfig, TensorSurgeon
from .training import Trainer, TrainingConfig
from .training.data_utils import to_device
from .training.paradigms import call_model


class ProjectConfig(BaseConfig):
    __title__ = "KI-Projekt"
    __tooltip__ = T(
        "Vollständige Beschreibung einer KI: Modell, Daten, Training, Speicher, Tensor-Regeln.",
        "Eine Datei = eine reproduzierbare KI; jede Einstellung ist erklärt und änderbar.",
        "Ausgangspunkt jedes Projekts (Vorlagen über Presets).",
    )
    name = Param("meine_ki", tooltip=T("Projektname.", "Für Ausgabeordner und Übersicht.", "Kurz, ohne Leerzeichen."))
    domain = Param("allgemein", tooltip=T(
        "Anwendungsgebiet (z. B. sprache, bild, zeitreihen, tabellen, rl …).", "Ordnet Vorlagen ein.",
        "Wird von Presets gesetzt."))
    description = Param("", tooltip=T("Freie Beschreibung.", "Dokumentation des Experiments.", "Was wird getestet?"))
    model = ComponentParam("model", "mlp", group="Modell", tooltip=T(
        "Das Modell (beliebig aus der Registry 'model', auch Kombinationen).", "Architektur der KI.",
        "Siehe 'python -m ki_baukasten modelle'."))
    data = ComponentParam("dataset", "classification", group="Daten", tooltip=T(
        "Datenquelle (synthetisch oder eigene Datei).", "Woraus gelernt wird.", "Siehe 'python -m ki_baukasten komponenten dataset'."))
    training = NestedParam(TrainingConfig, group="Training")
    memory = NestedParam(MemoryConfig, group="Speicher")
    module_rules = ConfigListParam(ModuleRuleConfig, group="Modul-Chirurgie", level=Level.ADVANCED, tooltip=T(
        "Regeln für Eingriffe in ganze Schichten (ersetzen, entfernen, LoRA/Quantisierung einzeln, Hooks …).",
        "Jede Schicht jeder Art anpassen – für alle, nur bestimmte oder genau eine (scope).",
        "Ablationen, gezieltes Feintuning, Präzision pro Schicht, Analyse. Wirken vor den Tensor-Regeln."))
    tensor_rules = ConfigListParam(TensorRuleConfig, group="Tensor-Chirurgie", level=Level.ADVANCED, tooltip=T(
        "Regeln für Eingriffe in einzelne Tensoren (Einfrieren, Pruning, Kopieren, eigene LR …).",
        "Feinste Kontrolle bis auf einzelne Gewichte – für alle, nur bestimmte oder genau einen (scope).",
        "Transfer-Learning, Kompression, Experimente."))
    compression = NestedParam(CompressionConfig, group="Kompression", level=Level.ADVANCED)
    deployment = NestedParam(DeploymentConfig, group="Kompression")
    auto_fit_dims = Param(True, tooltip=T(
        "Modellgrößen (Ein-/Ausgabe, Vokabular, Beobachtungen …) automatisch an Daten/Umgebung anpassen.",
        "Vorlagen passen dadurch ohne Handarbeit zu jedem Datensatz.", "Aus, wenn exakte Werte erzwungen werden sollen."),
        level=Level.ADVANCED)
    output_dir = Param("ausgaben", tooltip=T(
        "Basisordner für Ausgaben (Checkpoints, Protokolle, gespeicherte Projekte).", "Ordnung.", "ausgaben."),
        level=Level.ADVANCED)


# (Info-Schlüssel des Datensatzes, Modellparameter, Modus)
_FIT_RULES: list[tuple[str, str, str]] = [
    ("input_dim", "input_dim", "set"), ("output_dim", "output_dim", "set"), ("vocab_size", "vocab_size", "set"),
    ("in_channels", "in_channels", "set"), ("in_channels", "image_channels", "set"),
    ("input_size", "input_size", "set"), ("input_size", "image_size", "set"), ("data_dim", "data_dim", "set"),
    ("seq_len", "max_seq_len", "max"),
]


class Project:
    """Fassade für eine komplette KI (siehe Modul-Dokumentation)."""

    def __init__(self, config: ProjectConfig | dict | None = None):
        if isinstance(config, dict):
            config = ProjectConfig.from_dict(config)
        self.config: ProjectConfig = config or ProjectConfig()
        self.model: nn.Module | None = None
        self.surgeon: TensorSurgeon | None = None
        self.module_surgeon: ModuleSurgeon | None = None
        self.trainer: Trainer | None = None
        self._bundle: Any = None
        self.history: list[dict] = []
        self.notes: list[str] = []
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._teacher: nn.Module | None = None       # Lehrer für distillation (Project.distill)
        self.last_compression: Any = None
        self._switch_memory: dict[str, Any] = {}     # beim Ausschalten gemerkte eigene Werte (für "an"/"eigen")

    # ================================================================ Erzeugen
    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> "Project":
        """Projekt aus einer Vorlage; ``overrides`` als Pfade, z. B. ``**{"model.d_model": 256}``."""
        from .presets import get_preset

        p = cls(get_preset(name))
        if overrides:
            p.config.update(overrides)
        return p

    @classmethod
    def import_model(cls, path: str | Path, format: str = "auto", data: Any = "none", name: str | None = None,
                     **model_args: Any) -> "Project":
        """Fremdes Modell importieren (torch.export, TorchScript, state_dict, safetensors, Hugging Face).

        Danach funktionieren alle Werkzeuge: ``architecture()``, Regeln, LoRA, ``compress()``, ``train()``
        (mit passenden Daten), ``distill()`` und ``export()``. Beispiel::

            p = Project.import_model("modell.pt2", input_shape=[1, 32])
            p.set("tensor_rules", [{"target": "lm_head", "op": "freeze"}])
            p.export("modell_neu", "export")
        """
        cfg = ProjectConfig.from_dict({
            "name": name or Path(str(path)).stem or "import",
            "model": {"name": "extern", "args": {"path": str(path), "format": format, **model_args}},
            "data": data, "auto_fit_dims": False,
        })
        p = cls(cfg)
        p.build()
        return p

    @classmethod
    def from_file(cls, path: str | Path) -> "Project":
        """Projekt aus JSON/YAML-Datei."""
        return cls(ProjectConfig.load(path))

    # ================================================================ Einstellungen
    def set(self, path: str, value: Any) -> "Project":
        """Setzt eine Einstellung. Enthält ``path`` ein ``*``, gilt der Wert für alle Treffer (siehe ``set_all``).

        Die Werte ``"an"``, ``"aus"``, ``"eigen"`` und ``"standard"`` schalten die Einstellung um (siehe
        ``switch``), sofern sie dort nicht ohnehin ein gültiger Text sind – z. B. ``p.set("**.dropout", "aus")``
        oder ``p.set("training.ema", "an")``.
        """
        from .core.switches import is_switch_command

        probe = self.config.find(path)[0] if "*" in path and self.config.find(path) else path
        if isinstance(value, str) and "*" not in probe and is_switch_command(self.config, probe, value):
            self.switch(path, value)
            return self
        if "*" in path:
            self.set_all(path, value)
            return self
        self._snapshot()
        try:
            self.config.set(path, value)
        except Exception:
            self._undo.pop()  # fehlgeschlagene Änderung nicht in den Verlauf
            raise
        self._invalidate()
        return self

    def find(self, pattern: str) -> list[str]:
        """Pfade zu einem Muster (``*`` = ein Segment, ``**`` = beliebig viele; ohne Punkt = überall)."""
        return self.config.find(pattern)

    def set_all(self, pattern: str, value: Any, scope: str = "alle", pick: str | None = None) -> list[str]:
        """Setzt einen Wert für **alle** passenden Einstellungen – oder nur für bestimmte bzw. genau eine.

        Args:
            pattern: Pfad-Muster wie bei ``find`` (z. B. ``"dropout"``, ``"model.**.num_heads"``).
            scope:   ``alle`` | ``nur_auswahl`` | ``nur_einer`` | ``alle_ausser`` (mit ``pick``).
            pick:    Positionen in der Trefferliste (z. B. ``"0"``, ``"-1"``, ``"0,2"``).

        Returns:
            Die geänderten Pfade. Erst wenn **alle** Treffer den Wert akzeptieren, wird geändert
            (alles oder nichts).
        """
        from .tensor.selection import apply_scope

        paths = apply_scope(self.config.find(pattern), scope, pick, f"set_all('{pattern}')")
        if not paths:
            raise ConfigError(f"Kein Pfad passt auf '{pattern}'. Tipp: Project.search('{pattern}').")
        trial = self.config.copy()
        errors = []
        for path in paths:
            try:
                trial.set(path, value)
            except ConfigError as exc:
                errors.append(f"{path}: {exc}")
        if errors:
            raise ConfigError("set_all: nicht alle Treffer akzeptieren den Wert:\n  - " + "\n  - ".join(errors))
        self._snapshot()
        for path in paths:
            self.config.set(path, value)
        self._invalidate()
        return paths

    # ------------------------------------------------------------ Schalter (An / Aus / Eigen)
    def switch(self, pattern: str, state: str, value: Any = None, scope: str = "alle",
               pick: str | None = None) -> list[str]:
        """Schaltet Einstellungen **an**, **aus**, auf **eigen** (benutzerdefiniert) oder **standard**.

        Args:
            pattern: Pfad oder Muster (``*``/``**``) – z. B. ``"**.dropout"``, ``"training.regularization"``
                     (ganze Gruppe) oder ``"training.ema"`` (Gruppe mit ``enabled``-Schalter).
            state:   ``an`` | ``aus`` | ``eigen`` | ``standard``.
            value:   eigener Wert für ``eigen`` (ohne Angabe: zuletzt gemerkter Wert).
            scope/pick: wie bei ``set_all`` (alle, nur_auswahl, nur_einer, alle_ausser).

        Pflicht-Einstellungen (ohne Aus-Zustand) werden bei Mustern übersprungen. Rückgängig mit ``undo()``.
        """
        from .core import UNSET
        from .core.switches import set_state
        from .tensor.selection import apply_scope

        if "*" in pattern:
            paths = apply_scope(self.config.find(pattern), scope, pick, f"switch('{pattern}')")
            if not paths:
                raise ConfigError(f"Kein Pfad passt auf '{pattern}'. Tipp: Project.search('{pattern}').")
        else:
            paths = [pattern]
        trial = self.config.copy()
        memory = dict(self._switch_memory)
        changed: list[str] = []
        errors: list[str] = []
        for path in paths:
            try:
                changed += set_state(trial, path, state, UNSET if value is None else value, memory)
            except ConfigError as exc:
                if len(paths) == 1:
                    raise
                errors.append(f"{path}: {exc}")
        if not changed:
            if errors and len(errors) == len(paths):
                raise ConfigError(f"switch('{pattern}', '{state}'): nichts umschaltbar:\n  - " + "\n  - ".join(errors))
            self._switch_memory = memory
            return []
        self._snapshot()
        self.config = trial
        self._switch_memory = memory
        self._invalidate()
        return changed

    def turn_on(self, pattern: str, **kw: Any) -> list[str]:
        """Kurzform für ``switch(pattern, "an")``."""
        return self.switch(pattern, "an", **kw)

    def turn_off(self, pattern: str, **kw: Any) -> list[str]:
        """Kurzform für ``switch(pattern, "aus")``."""
        return self.switch(pattern, "aus", **kw)

    def reset(self, pattern: str, **kw: Any) -> list[str]:
        """Kurzform für ``switch(pattern, "standard")`` (Standardwert wiederherstellen)."""
        return self.switch(pattern, "standard", **kw)

    def switches(self, pattern: str | None = None, state: str | None = None, max_level: str | None = None,
                 only_switchable: bool = False) -> list:
        """Zustand aller Einstellungen (``SwitchInfo``: Pfad, an/aus/eigen, Wert, Aus-/An-Wert, Pflicht?)."""
        from .core import switch_table

        return switch_table(self.config, pattern, state, max_level, only_switchable)

    def switch_overview(self, max_level: str | None = "advanced") -> str:
        """Kompakte Übersicht: was ist an, was aus, was eigen – nach Bereichen gruppiert."""
        from .core.switches import _show, summary

        counts = summary(self.config, max_level)
        lines = [f"Schalter-Übersicht ({self.config.name}): {counts['an']} an · {counts['aus']} aus · "
                 f"{counts['eigen']} eigen  ({counts['schaltbar']} schaltbar, {counts['pflicht']} Pflicht)"]
        active: dict[str, list[str]] = {}
        custom: dict[str, list[str]] = {}
        for info in self.switches(max_level=max_level):
            section, _, rest = info.path.partition(".")
            if not rest:
                continue  # Projekt-Metadaten (Name, Gebiet …)
            if info.switchable and info.state != "aus":
                text = rest if info.value is True else f"{rest}={_show(info.value)}"
                active.setdefault(section, []).append(text + ("*" if info.state == "eigen" else ""))
            elif not info.switchable and info.state == "eigen":
                custom.setdefault(section, []).append(f"{rest}={_show(info.value)}")
        lines.append("Eingeschaltet (* = eigener Wert):")
        lines.extend(f"  [{k}] " + ", ".join(v) for k, v in active.items())
        if custom:
            lines.append("Eigene Werte bei Pflicht-Einstellungen:")
            lines.extend(f"  [{k}] " + ", ".join(v) for k, v in custom.items())
        lines.append(f"Ausgeschaltet: {counts['aus']} Einstellungen – Liste: projekt.switches(state='aus')")
        return "\n".join(lines)

    def search(self, text: str, max_level: str | None = None, limit: int = 40) -> list[tuple[str, str]]:
        """Volltextsuche über Pfade **und** Tooltips → ``[(pfad, Kurzbeschreibung)]``.

        Beispiel: ``p.search("speicher")`` findet alle Einstellungen, deren Name oder
        Erklärung mit Speicher zu tun hat.
        """
        from .core import all_registries

        words = [w.lower() for w in text.split()]

        def ok(*parts: str) -> bool:
            blob = " ".join(p for p in parts if p).lower()
            return all(w in blob for w in words)

        hits = []
        for ref in self.config.walk(max_level=max_level):
            tip = ref.param.tooltip
            if ok(ref.path, *([tip.what, tip.why, tip.use_for, tip.tip] if tip else [])):
                hits.append((ref.path, tip.what if tip else ""))
        # Auch auswählbare Komponenten (z. B. Optimierer 'adamw8bit', Mixer 'ssm') finden.
        for kind, reg in all_registries().items():
            for name in reg.names():
                comp = reg.get(name)
                tip = comp.tooltip
                if ok(name, kind, *([tip.what, tip.why, tip.use_for, tip.tip] if tip else [])):
                    hits.append((f"{kind}:{name}", f"Komponente – {tip.what if tip else ''}"))
        return hits[:limit]

    def get(self, path: str) -> Any:
        return self.config.get(path)

    def update(self, changes: dict[str, Any]) -> "Project":
        self._snapshot()
        try:
            self.config.update(changes)
        except Exception:
            self.config = ProjectConfig.from_dict(self._undo.pop())  # alles oder nichts
            raise
        self._invalidate()
        return self

    # ------------------------------------------------------------ Verlauf (Rückgängig/Wiederholen)
    def _snapshot(self) -> None:
        self._undo.append(self.config.to_dict())
        del self._undo[:-50]
        self._redo.clear()

    def undo(self, steps: int = 1) -> bool:
        """Macht die letzten ``steps`` Änderungen (set/set_all/update/Profil/Ratgeber) rückgängig."""
        done = False
        for _ in range(steps):
            if not self._undo:
                break
            self._redo.append(self.config.to_dict())
            self.config = ProjectConfig.from_dict(self._undo.pop())
            done = True
        if done:
            self._invalidate()
        return done

    def redo(self, steps: int = 1) -> bool:
        """Stellt rückgängig gemachte Änderungen wieder her."""
        done = False
        for _ in range(steps):
            if not self._redo:
                break
            self._undo.append(self.config.to_dict())
            self.config = ProjectConfig.from_dict(self._redo.pop())
            done = True
        if done:
            self._invalidate()
        return done

    def diff(self, other: "Project | ProjectConfig | str | dict") -> list[tuple[str, Any, Any]]:
        """Unterschiede zu einem anderen Projekt/Preset/einer Datei → ``[(pfad, hier, dort)]``."""
        if isinstance(other, str):
            other = Project.from_file(other) if Path(other).exists() else Project.from_preset(other)
        if isinstance(other, dict):
            other = ProjectConfig.from_dict(other)
        cfg = other.config if isinstance(other, Project) else other

        def leaves(c: ProjectConfig) -> dict[str, Any]:
            # Komponenten über ihren Namen vergleichen, Unter-Konfigurationen über ihre Einträge.
            return {r.path: (r.value.name if hasattr(r.value, "component") else r.value) for r in c.walk()
                    if not isinstance(r.value, BaseConfig)}

        mine, theirs = leaves(self.config), leaves(cfg)
        missing = object()
        out = []
        for path in dict.fromkeys(list(mine) + list(theirs)):
            a, b = mine.get(path, missing), theirs.get(path, missing)
            if a is missing and b is None or b is missing and a is None:
                continue  # Einstellung existiert nur auf einer Seite und ist dort leer – kein echter Unterschied
            if a != b:
                out.append((path, None if a is missing else a, None if b is missing else b))
        return out

    def explain(self, path: str | None = None) -> str:
        return self.config.explain(path)

    def describe(self, max_level: str | None = "basic", tooltips: bool = True, only_changed: bool = False) -> str:
        return self.config.describe(max_level=max_level, tooltips=tooltips, only_changed=only_changed)

    def validate(self) -> "Project":
        self.config.validate()
        return self

    def apply_memory_profile(self, name: str) -> "Project":
        """Speicherprofil anwenden: 'standard', 'sparsam', 'extrem_sparsam', 'schnell'."""
        from .presets import apply_memory_profile

        self._snapshot()
        apply_memory_profile(self.config, name)
        self._invalidate()
        return self

    def _invalidate(self) -> None:
        self.model = None
        self.surgeon = None
        self.trainer = None

    # ================================================================ Daten
    def data(self) -> Any:
        """Lädt (einmalig) den konfigurierten Datensatz → ``DatasetBundle``."""
        if self._bundle is None:
            self._bundle = self.config.data.build()
        return self._bundle

    # ================================================================ Bauen
    def _note(self, text: str) -> None:
        self.notes.append(text)
        if self.config.training.verbose:
            print(f"  ℹ {text}")

    def _fit_value(self, path: str, value: Any, mode: str = "set") -> None:
        try:
            current = self.config.get(path)
        except ConfigError:
            return
        if mode == "max" and isinstance(current, int) and current >= value:
            return
        if current != value:
            self.config.set(path, value)
            self._note(f"automatisch angepasst: {path} {current!r} → {value!r}")

    def _fit_dims(self) -> None:
        cfg = self.config
        args = cfg.model.args
        params = type(args)._params
        paradigm = cfg.training.paradigm
        if paradigm.name == "reinforcement":
            env = paradigm.args.environment.build()
            if "obs_dim" in params:
                self._fit_value("model.obs_dim", env.obs_dim)
            if "action_dim" in params:
                self._fit_value("model.action_dim", env.action_dim)
            if "action_type" in params:
                self._fit_value("model.action_type", env.action_type)
            if "num_actions" in params:
                self._fit_value("model.num_actions", env.action_dim)
            return
        info = self.data().info
        if not info:
            return
        mlm = paradigm.name == "masked_lm"
        for key, param, mode in _FIT_RULES:
            if key in info and param in params:
                value = info[key]
                if key == "vocab_size" and mlm:
                    value = value + 1  # Platz für das [MASK]-Token
                if param == "output_dim" and cfg.model.name in ("transformer", "ssm", "hybrid") \
                        and getattr(args, "pooling", "none") == "none":
                    continue  # Sprachmodell: Ausgabe = Vokabular
                if param == "input_dim" and getattr(args, "input_type", None) == "tokens":
                    continue
                self._fit_value(f"model.{param}", value, mode)
        if "data_dim" in info and "data_shape" in params:
            self._fit_value("model.data_shape", [info["data_dim"]])
        if "input_dim" in info and cfg.model.name == "autoencoder" and getattr(args, "backbone", "") == "mlp":
            self._fit_value("model.input_dim", info["input_dim"])
        if mlm and "vocab_size" in info and paradigm.args.mask_token_id is None:
            self._fit_value("training.paradigm.mask_token_id", info["vocab_size"])
        if "bos_token_id" in info and paradigm.name == "seq2seq":
            self._fit_value("training.paradigm.bos_token_id", info["bos_token_id"])
        if cfg.model.name == "rnn" and "vocab_size" in info and info.get("task") != "regression":
            self._fit_value("model.input_type", "tokens")

    def build(self) -> nn.Module:
        """Baut das Modell (inkl. automatischer Dimensionen und Tensor-Chirurgie)."""
        cfg = self.config
        if cfg.auto_fit_dims:
            self._fit_dims()
        cfg.validate()
        torch.manual_seed(cfg.training.seed if cfg.training.seed is not None else 0)
        model = cfg.model.build()
        self.surgeon = None
        self.module_surgeon = None
        if cfg.module_rules:
            self.module_surgeon = ModuleSurgeon(model, list(cfg.module_rules)).apply()
        if cfg.tensor_rules:
            self.surgeon = TensorSurgeon(model, list(cfg.tensor_rules))
            self.surgeon.apply_build(num_layers=model.depth() if hasattr(model, "depth") else None)
        self.model = model
        return model

    # ================================================================ Training
    def train(self, train_data: Any = None, val_data: Any = None) -> list[dict]:
        """Trainiert (baut vorher, falls nötig). Ohne Daten wird der konfigurierte Datensatz genutzt."""
        if self.model is None:
            self.build()
        if train_data is None and self.config.training.paradigm.name != "reinforcement":
            bundle = self.data()
            train_data = bundle.train
            val_data = val_data if val_data is not None else bundle.val
            if self.config.data.name == "anomaly":
                val_data = None  # Anomalie-Testmenge hat Labels → separat mit anomaly_report() auswerten
        self._fit_warning()
        self.trainer = Trainer(self.model, self.config.training, self.config.memory, self.surgeon,
                               teacher=self._teacher)
        self.model = self.trainer.model
        self.history = self.trainer.fit(train_data, val_data)
        d = self.config.deployment
        if self.config.training.verbose and d.show_after_training:
            try:
                print()
                print(self.deployment_overview())
                if d.main_target:
                    print()
                    print(self.recommend_for(d.main_target))
            except Exception as exc:  # noqa: BLE001 - Übersicht darf das Training nie scheitern lassen
                print(f"  (Reduktions-Übersicht nicht verfügbar: {exc})")
        return self.history

    def _fit_warning(self) -> None:
        """Mit gesetztem ``memory.memory_budget_gb``: vor dem Training prüfen und Notfall-Optionen nennen."""
        if not (self.config.memory.memory_budget_gb and self.config.training.verbose):
            return
        try:
            report = self.memory_report()
            if report.fits() is False:
                from .analysis.fallback import options_text

                budget = self.config.memory.memory_budget_gb * 1024
                opts = [o for o in self.fallback_options(budget_mb=budget, training=True) if o.fits][:3]
                print(f"  ⚠ Training passt voraussichtlich nicht ins Budget ({report.total_bytes / 2**20:.0f} MB > "
                      f"{budget:.0f} MB).")
                if opts:
                    print("  " + options_text(opts, budget).replace("\n", "\n  "))
        except Exception:  # noqa: BLE001 - Prüfung darf das Training nie verhindern
            pass

    # ================================================================ Zielsysteme
    def _example_input(self) -> Any:
        try:
            bundle = self.data()
            if isinstance(bundle.train, dict) and "inputs" in bundle.train:
                return bundle.train["inputs"][:1].to(self._device())
        except Exception:  # noqa: BLE001
            pass
        model = self._ensure_model()
        return model.example_input(1) if hasattr(model, "example_input") else None

    def _macs(self) -> int | None:
        from .analysis.benchmark import count_macs

        ex = self._example_input()
        try:
            return count_macs(self._ensure_model(), ex) if ex is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _can_heal(self) -> bool:
        return self.config.data.name not in ("none",) and self.config.training.paradigm.name != "reinforcement"

    def recommend_for(self, target: str) -> Any:
        """Reduktions-Empfehlung für ein Zielsystem (siehe ``analysis.targets.TARGETS``)."""
        from .analysis.targets import recommend

        return recommend(self._ensure_model(), target, self._macs(), self._can_heal(),
                         activations_mb=self._activations_mb(False))

    def deployment_overview(self, targets: list[str] | None = None) -> str:
        """Tabelle: passt das Modell auf die Zielsysteme – und wie wird es passend?"""
        from .analysis.targets import overview

        return overview(self._ensure_model(), targets or list(self.config.deployment.targets), self._macs(),
                        self._can_heal())

    def _budget_mb(self, target: str | None = None, budget_mb: float | None = None) -> float:
        from .analysis.targets import TARGETS
        from .memory.estimator import detect_budget

        if budget_mb is not None:
            return float(budget_mb)
        if target is not None:
            if target not in TARGETS:
                raise ConfigError(f"Unbekanntes Zielsystem '{target}'. Verfügbar: {', '.join(TARGETS)}.")
            return TARGETS[target].memory_mb
        if self.config.memory.memory_budget_gb:
            return self.config.memory.memory_budget_gb * 1024
        detected = detect_budget(self._device())
        if detected is None:
            raise ConfigError("Kein Speicherbudget bekannt – target=…, budget_mb=… oder memory.memory_budget_gb setzen.")
        return detected / 2**20

    def _activations_mb(self, training: bool) -> float:
        """Aktivierungsspeicher: Training für den ganzen Batch, Inferenz grob pro Beispiel."""
        try:
            report = self.memory_report()
            act = (report.activations_bytes or 0.0) / 2**20
        except Exception:  # noqa: BLE001
            return 0.0
        if training:
            return act
        return act * 0.25 / max(1, report.batch_size)   # Inferenz speichert nichts für den Backward

    def fallback_options(self, target: str | None = None, budget_mb: float | None = None, training: bool = False,
                         scale: float = 1.0) -> list:
        """Notfall-Ausführung: alle Wege, das Modell trotz Speichermangel laufen zu lassen (mit Schätzung).

        Args:
            target:    Zielsystem (Budget aus dessen Profil) – oder ``budget_mb`` direkt angeben.
            training:  Optionen fürs Training statt für die Inferenz.
        """
        from .analysis.fallback import execution_options

        model = self._ensure_model()
        t = self.config.training
        budget = self._budget_mb(target, budget_mb)
        gpu = (target is not None and ("gpu" in target or target == "apple_silicon")) or \
            (target is None and self._device().type != "cpu")
        return execution_options(model, budget, training, self._activations_mb(training), scale=scale,
                                 batch_size=t.batch_size if training else 1, gpu_target=gpu, callbacks=[c.to_dict() for c in t.callbacks],
                                 gradient_accumulation_steps=t.gradient_accumulation_steps)

    def apply_fallback(self, key: str = "auto", target: str | None = None, budget_mb: float | None = None,
                       training: bool = False, allow_quality_loss: bool = False) -> Any:
        """Schaltet eine Notfall-Option ein (``auto`` = schnellste passende ohne Qualitätsverlust).

        Returns:
            Die angewandte ``ExecutionOption`` (mit Nutzungshinweis). Rückgängig mit ``undo()``.
        """
        from .analysis.fallback import best_option

        options = self.fallback_options(target, budget_mb, training)
        if key == "auto":
            chosen = best_option(options, allow_quality_loss) or best_option(options, True)
            if chosen is None:
                raise ConfigError("Keine Notfall-Option passt in das Budget – kleineres Grundmodell, Schüler-Modell "
                                  "(distill) oder mehr Geräte (cluster) wählen.")
        else:
            matches = [o for o in options if o.key == key]
            if not matches:
                raise ConfigError(f"Unbekannte Notfall-Option '{key}'. Verfügbar: {', '.join(o.key for o in options)}.")
            chosen = matches[0]
        if chosen.changes:
            self.update(chosen.changes)
        self._note(f"Notfall-Ausführung '{chosen.key}' eingeschaltet: {chosen.title}"
                   + (f" – {chosen.usage}" if chosen.usage else ""))
        return chosen

    def reduce_for(self, target: str, export: str | Path | None = None, allow_student: bool = True,
                   fallback: str | None = "auto") -> "Project":
        """Wendet die Empfehlung für ``target`` an: ggf. Schüler-Modell (Distillation), dann Kompression.

        Passt das Ergebnis trotzdem nicht, wird mit ``fallback`` (``auto`` oder eine Kennung aus
        ``fallback_options``) eine Notfall-Ausführung eingeschaltet – z. B. Schicht für Schicht von der SSD.

        Returns:
            Das reduzierte Projekt (``self`` oder – bei Schüler-Empfehlung – ein neues Schüler-Projekt).
            Mit ``export`` wird zusätzlich im passenden Format exportiert.
        """
        rec = self.recommend_for(target)
        project: Project = self
        d = self.config.deployment
        if rec.student and allow_student:
            project = self.distill(width=rec.student["width"], depth=rec.student["depth"], steps=d.student_steps,
                                   name=f"{self.config.name}_{target}")
        steps = [dict(s) for s in rec.steps]
        for s in steps:
            if s["name"] == "heal":
                if project is not self or d.heal_steps == 0:
                    s["name"] = "skip"   # Schüler wurde bereits per Distillation trainiert
                else:
                    s["args"] = {"steps": d.heal_steps}
        steps = [s for s in steps if s["name"] != "skip"]
        if steps:
            project.compress(steps)
        if fallback and not rec.fits_after:
            from .analysis.benchmark import model_size_bytes

            if model_size_bytes(project._ensure_model()) / 2**20 > rec.target.memory_mb:
                try:
                    project.apply_fallback(fallback, target=target)
                except ConfigError as exc:
                    project._note(f"Keine Notfall-Ausführung möglich: {exc}")
        if export is not None:
            fmt = "onnx" if "ONNX" in rec.target.export.split("/")[0] and _has_onnx() else "export"
            project.export(export, fmt)
        return project

    def _batches(self, data: Any, n: int, batch_size: int | None = None) -> list[dict]:
        """Erste ``n`` Batches eines dict-Datensatzes (für Kalibrierung/Messung)."""
        if data is None or n <= 0 or not isinstance(data, dict):
            return []
        bs = batch_size or self.config.training.batch_size
        size = next(iter(data.values())).shape[0]
        out = []
        for i in range(0, min(size, n * bs), bs):
            out.append({k: v[i:i + bs] for k, v in data.items() if isinstance(v, torch.Tensor)})
        return out

    def compress(self, steps: list | None = None, calibration_batches: int | None = None,
                 eval_batches: int | None = None) -> Any:
        """Modell eindampfen (Schritte aus ``compression.steps`` oder ``steps``) – mit Vorher/Nachher-Bericht.

        Formänderungen werden als ``module_rules`` ins Projekt übernommen: ``save()``/``load()``
        stellen das kleinere Modell exakt wieder her. Beispiel::

            p.compress([{"name": "prune_ffn", "args": {"amount": 0.4}}, "heal", {"name": "quantize"}])
        """
        from .compress import run_compression

        c = self.config.compression
        steps = list(steps) if steps is not None else list(c.steps)
        if not steps:
            raise ConfigError("Keine Kompressions-Schritte – compression.steps setzen oder steps=[...] übergeben. "
                              "Vorschläge für ein Zielgerät: Project.recommend_for('smartphone').")
        model = self._ensure_model()
        calib, evals = [], []
        try:
            bundle = self.data()
            calib = self._batches(bundle.train, calibration_batches if calibration_batches is not None
                                  else c.calibration_batches)
            evals = self._batches(bundle.val if bundle.val is not None else bundle.train,
                                  eval_batches if eval_batches is not None else c.eval_batches)
        except Exception:  # noqa: BLE001 - ohne Daten: nur datenfreie Schritte möglich
            pass
        model, report, rules = run_compression(model, steps, self, calib, evals, c.measure_latency)
        for r in rules:
            self.config.module_rules.append(ModuleRuleConfig.from_dict(r))
        self.model, self.trainer = model, None
        self.last_compression = report
        if self.config.training.verbose:
            print(report)
        return report

    def distill(self, student: Any = None, width: float = 0.5, depth: float = 0.5, steps: int | None = None,
                temperature: float = 2.0, alpha: float = 0.7, kd_loss: str = "kl", feature_weight: float = 0.0,
                name: str | None = None) -> "Project":
        """Lehrer-Schüler: trainiert ein (kleineres) Schüler-Projekt mit diesem Modell als Lehrer.

        Args:
            student: Modell-Spezifikation (Name/dict/Spec) oder ``Project``; ``None`` = automatisch
                     verkleinert (``width``/``depth`` skalieren Breite und Tiefe).
            steps:   Trainingsschritte (None = Einstellung des Projekts).

        Returns:
            Das trainierte Schüler-Projekt (eigene Konfiguration, speicher- und exportierbar).
        """
        from .training.continual import shrink_config

        teacher = self._ensure_model()
        teacher.eval()
        if isinstance(student, Project):
            sp = student
        else:
            cfg = self.config.copy()
            cfg.model = student if student is not None else shrink_config(self.config.model, width, depth)
            cfg.module_rules, cfg.tensor_rules = [], []
            cfg.name = name or f"{self.config.name}_schueler"
            sp = Project(cfg)
            sp._bundle = self._bundle
        args = {"temperature": temperature, "alpha": alpha, "kd_loss": kd_loss, "feature_weight": feature_weight}
        sp.config.training.paradigm = {"name": "distillation", "args": args}
        if steps is not None:
            sp.config.training.max_steps = steps
        sp._teacher = teacher
        sp.train()
        return sp

    def continue_training(self, train_data: Any = None, val_data: Any = None, method: str = "lwf",
                          strength: float = 0.5, old_data: Any = None, replay_fraction: float = 0.3,
                          steps: int | None = None, lr: float | None = None, data: Any = None) -> list[dict]:
        """Weiterlernen mit neuen Daten, ohne das Alte zu vergessen (``lwf`` | ``ewc`` | ``replay`` | ``none``).

        Args:
            train_data / data: neue Daten (dict) bzw. eine Datensatz-Spezifikation (z. B. ``{"name": "csv", ...}``).
            strength: lwf = Anteil Lehrer-Verlust (0…1); ewc = Stärke des Strafterms.
            old_data: alte Daten für ewc/replay (Standard: der konfigurierte Datensatz).
        """
        from .compress import calibration_loss
        from .core import ComponentSpec
        from .training.continual import METHODS, ewc_penalty, fisher_information, mix_replay

        if method not in METHODS:
            raise ConfigError(f"Unbekannte Methode '{method}'. Erlaubt: {', '.join(METHODS)}.")
        model = self._ensure_model()
        if data is not None:
            bundle = ComponentSpec.from_value("dataset", data).build()
            train_data, val_data = bundle.train, val_data if val_data is not None else bundle.val
        if train_data is None:
            raise ConfigError("continue_training braucht neue Daten (train_data=... oder data=...).")
        old = old_data if old_data is not None else self.data().train
        t = self.config.training.copy()
        if steps is not None:
            t.max_steps = steps
        if lr is not None:
            t.set("optimizer.lr", lr)
        teacher = None
        if method == "lwf":
            import copy as _copy
            teacher = _copy.deepcopy(getattr(model, "_orig_mod", model)).eval()
            base = t.paradigm.name
            task = "causal_lm" if base == "causal_lm" else "auto"
            t.paradigm = {"name": "distillation", "args": {"alpha": float(min(1.0, max(0.0, strength))),
                                                         "task": task, "temperature": 2.0}}
        if method == "replay":
            train_data = mix_replay(train_data, old, replay_fraction, seed=t.seed or 0)
        trainer = Trainer(model, t, self.config.memory, self.surgeon, teacher=teacher)
        if method == "ewc":
            batches = self._batches(old, 8)
            fisher = fisher_information(trainer.model, batches, calibration_loss)
            trainer.penalties.append(ewc_penalty(trainer.model, fisher, strength * 100.0))
        self.trainer, self.model = trainer, trainer.model
        hist = trainer.fit(train_data, val_data)
        self.history = self.history + hist
        return hist

    def evaluate(self, data: Any = None) -> dict[str, float]:
        if self.trainer is None:
            raise ConfigError("Erst trainieren (train) oder Trainer erzeugen.")
        if data is None:
            bundle = self.data()
            data = bundle.val if bundle.val is not None else bundle.train
        return self.trainer.evaluate(data)

    # ================================================================ Benutzen
    def _ensure_model(self) -> nn.Module:
        if self.model is None:
            self.build()
        return self.model

    def _device(self) -> torch.device:
        model = self._ensure_model()
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def predict(self, inputs: Any, chunk_size: int | None = None, **kwargs: Any) -> Any:
        """Vorhersage im Eval-Modus (mit gemischter Präzision, falls konfiguriert).

        Args:
            chunk_size: Eingaben stückweise verarbeiten (je so viele Beispiele) – spart Aktivierungsspeicher.
                        Bei ``memory.oom_strategy = "batch_teilen"`` wird bei Speichermangel automatisch
                        weiter geteilt.
        """
        from .memory.execution import run_chunked

        model = self._ensure_model()
        was = model.training
        model.eval()
        mem = self.trainer.mem if self.trainer is not None else MemoryManager(self.config.memory)
        device = self._device()
        mc = self.config.memory

        def run(part: Any) -> Any:
            with mem.autocast():
                return call_model(model, to_device(part, device), **kwargs)

        try:
            out, _ = run_chunked(run, inputs, chunk_size, on_oom=mc.oom_strategy == "batch_teilen",
                                 max_splits=mc.oom_max_splits)
        finally:
            model.train(was)
        return out

    def generate(self, prompt: str | torch.Tensor, **generation: Any) -> Any:
        """Text-/Token-Generierung (Sprachmodelle). Mit Zeichen-Tokenizer des Datensatzes auch für Text."""
        model = self._ensure_model()
        tok = self.data().info.get("tokenizer") if self.config.data.name != "none" else None
        if isinstance(prompt, str):
            if tok is None:
                raise ConfigError("Für Text-Prompts wird ein Tokenizer im Datensatz benötigt (z. B. Datensatz 'text').")
            ids = torch.tensor([tok.encode(prompt) or [1]], device=self._device())
            out = model.generate(ids, **generation)
            return tok.decode(out[0].tolist())
        return model.generate(prompt.to(self._device()), **generation)

    def sample(self, n: int = 16, **kwargs: Any) -> torch.Tensor:
        """Neue Daten erzeugen (GAN, Diffusion, VAE/VQ-VAE)."""
        model = self._ensure_model()
        if hasattr(model, "sample"):          # Diffusion, Autoencoder (VAE/VQ)
            return model.sample(n, **kwargs)
        if hasattr(model, "sample_noise"):    # GAN
            return model.generate(n, **kwargs)
        raise ConfigError(f"{type(model).__name__} kann keine Samples erzeugen (für Text: generate).")

    def anomaly_report(self) -> dict[str, float]:
        """Für Autoencoder + Datensatz 'anomaly': Trennschärfe des Rekonstruktionsfehlers (AUROC)."""
        model = self._ensure_model()
        bundle = self.data()
        if bundle.val is None or not hasattr(model, "anomaly_score"):
            raise ConfigError("anomaly_report braucht einen Autoencoder und den Datensatz 'anomaly'.")
        x, y = bundle.val["inputs"].to(self._device()), bundle.val["targets"]
        scores = model.anomaly_score(x).cpu()
        pos, neg = scores[y == 1], scores[y == 0]
        auroc = float((pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean())
        return {"auroc": auroc, "score_normal": float(neg.mean()), "score_anomalie": float(pos.mean())}

    # ================================================================ Analyse
    def summary(self, max_depth: int = 2) -> str:
        model = self._ensure_model()
        parts = [model.summary(max_depth) if hasattr(model, "summary") else str(model)]
        if self.surgeon is not None:
            parts.append(self.surgeon.report())
        return "\n\n".join(parts)

    def architecture(self, fmt: str = "text", max_depth: int = 3, path: str | Path | None = None) -> str:
        """Komplette Aufbau-Darstellung (``text`` | ``markdown`` | ``mermaid`` | ``json``), optional als Datei."""
        from .analysis import architecture_report

        model = self._ensure_model()
        example = None
        try:
            bundle = self.data()
            if isinstance(bundle.train, dict) and "inputs" in bundle.train:
                example = bundle.train["inputs"][:1]
        except Exception:  # noqa: BLE001 - ohne Daten: Beispiel-Eingabe des Modells
            example = None
        text = architecture_report(model, fmt, max_depth, example)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    def advise(self, batch_size: int | None = None) -> list:
        """Optimierungs- und Speichervorschläge (``analysis.advisor``), sortiert nach Wichtigkeit."""
        from .analysis import advise

        return advise(self, batch_size)

    def apply_advice(self, keys: list[str] | str | None = None, category: str | None = None) -> list:
        """Wendet Vorschläge an (nach Kennung(en) und/oder Kategorie). Rückgängig mit ``undo()``."""
        from .analysis import apply_advice

        tips = self.advise()
        if isinstance(keys, str):
            keys = [keys]
        self._snapshot()
        chosen = apply_advice(self, tips, keys, category)
        # apply_advice ruft update() pro Vorschlag auf – im Verlauf soll das EIN Schritt sein.
        if chosen:
            del self._undo[len(self._undo) - len(chosen):]
        else:
            self._undo.pop()
        return chosen

    def export(self, path: str | Path, fmt: str = "export", merge_lora: bool = True) -> Path:
        """Modell exportieren: ``export`` (torch.export), ``torchscript``, ``onnx`` oder ``state_dict``."""
        from .analysis import export_model

        model = self._ensure_model()
        example = None
        try:
            bundle = self.data()
            if isinstance(bundle.train, dict) and "inputs" in bundle.train:
                example = bundle.train["inputs"][:1].cpu()
        except Exception:  # noqa: BLE001
            example = None
        return export_model(model, path, fmt, example, merge_lora)

    def memory_report(self, batch_size: int | None = None) -> Any:
        model = self._ensure_model()
        t = self.config.training
        example = None
        try:
            bundle = self.data()
            if isinstance(bundle.train, dict) and "inputs" in bundle.train and t.paradigm.name in ("supervised", "causal_lm"):
                example = bundle.train["inputs"][:2]
        except Exception:
            example = None
        return estimate_memory(model, t.optimizer.name, batch_size or t.batch_size, self.config.memory,
                               ema=t.ema.enabled, example=example)

    # ================================================================ Speichern
    def save(self, directory: str | Path | None = None) -> Path:
        """Speichert Konfiguration (projekt.json), Gewichte (modell.pt) und Verlauf (verlauf.json)."""
        d = Path(directory or Path(self.config.output_dir) / self.config.name)
        d.mkdir(parents=True, exist_ok=True)
        self.config.save(d / "projekt.json")
        if self.model is not None:
            target = getattr(self.model, "_orig_mod", self.model)
            torch.save(target.state_dict(), d / "modell.pt")
        (d / "verlauf.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = False) -> "Project":
        """Lädt ein mit ``save`` gespeichertes Projekt (inkl. Gewichten).

        Args:
            mmap: Gewichte per Memory-Map einbinden statt zu kopieren – das Betriebssystem lädt nur, was
                  gebraucht wird (für Modelle nahe an der RAM-Grenze; Inferenz).
        """
        d = Path(directory)
        p = cls.from_file(d / "projekt.json")
        p.config.auto_fit_dims = False  # gespeicherte Werte sind bereits angepasst
        model = p.build()
        weights = d / "modell.pt"
        if weights.exists():
            state = torch.load(weights, map_location="cpu", weights_only=True, mmap=mmap)
            if any(k.endswith(".qweight") or ".lora_" in k for k in state):
                # Quantisierte/LoRA-Modelle: gleiche Struktur herstellen, dann laden.
                MemoryManager(p.config.memory).apply_structure(model, training=False)
            model.load_state_dict(state, assign=mmap)
            if p.config.memory.param_dtype != "float32":  # z. B. nach Kompressions-Schritt 'dtype'
                from .memory.manager import DTYPES
                model.to(DTYPES[p.config.memory.param_dtype])
        if (d / "verlauf.json").exists():
            p.history = json.loads((d / "verlauf.json").read_text(encoding="utf-8"))
        return p

    def __repr__(self) -> str:
        return f"<Project {self.config.name!r}: model={self.config.model.name}, data={self.config.data.name}, " \
               f"paradigm={self.config.training.paradigm.name}>"


def _has_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def components_overview(max_level: str | None = None) -> str:
    """Übersicht aller Registries mit Einträgen (für CLI/Doku)."""
    from .core import all_registries

    return "\n\n".join(reg.describe(max_level) for reg in all_registries().values())

