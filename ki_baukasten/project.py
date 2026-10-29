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
from .analysis.goals import GoalsConfig
from .analysis.inference import InferenceConfig
from .analysis.targets import DeploymentConfig
from .compress import CompressionConfig
from .memory import MemoryConfig, MemoryManager, estimate_memory
from .tensor import ModuleRuleConfig, ModuleSurgeon, TensorRuleConfig, TensorSurgeon
from .training import Trainer, TrainingConfig
from .training.data_utils import to_device
from .training.paradigms import call_model
from .workflow import WorkflowConfig


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
    goals = NestedParam(GoalsConfig, group="Ziele")
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
    inference = NestedParam(InferenceConfig, group="Einsatz")
    workflow = NestedParam(WorkflowConfig, group="Ablauf")
    auto_fit_dims = Param(True, tooltip=T(
        "Modellgrößen (Ein-/Ausgabe, Vokabular, Beobachtungen …) automatisch an Daten/Umgebung anpassen.",
        "Vorlagen passen dadurch ohne Handarbeit zu jedem Datensatz.", "Aus, wenn exakte Werte erzwungen werden sollen."),
        level=Level.ADVANCED)
    output_dir = Param("ausgaben", tooltip=T(
        "Basisordner für Ausgaben (Checkpoints, Protokolle, gespeicherte Projekte).", "Ordnung.", "ausgaben."),
        level=Level.ADVANCED)


#: Bereiche, deren Änderung ein trainiertes Modell nicht ungültig macht.
_KEEP_MODEL = {"inference", "deployment", "goals", "workflow", "compression", "name", "description", "domain",
               "output_dir"}


def _changed_tops(a: dict, b: dict) -> list[str]:
    return [k for k in set(a) | set(b) if a.get(k) != b.get(k)]


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
        self._loggers: list[Any] = []                # zur Laufzeit eingebundene Logger (add_logger)
        self._technique_memory: dict[str, Any] = {}  # vorherige Werte eingeschalteter Techniken (für disable)
        self.goal_report: Any = None                 # letzte Bewertung der Ziele (goal_check / from_goals)
        self.test_data: Any = None                   # zurückgehaltene Testmenge (training.test_split / data.test)
        self._test_split: tuple | None = None        # (Original-Trainingsdaten, Rest ohne Testmenge)
        self.last_data_report: Any = None            # letzte Datenprüfung (data_check)
        self.last_test: dict[str, float] = {}        # letzte Ergebnisse für die Modellkarte
        self.last_calibration: dict[str, Any] = {}
        self.last_fairness: Any = None
        self.last_robustness: Any = None
        self._artifacts: dict[str, Any] | None = None  # gespeicherte Vorverarbeitung (Project.load)

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
    def from_goals(cls, goals: Any = None, choice: int = 0, apply_techniques: bool = True,
                   **goal_fields: Any) -> "Project":
        """Neue KI aus **Zielen**: Aufgabe, Prioritäten, Daten, Zielgerät … → beste Vorlage + passende Techniken.

        Beispiel::

            p = Project.from_goals(task="text_klassifizieren", priorities=["geschwindigkeit", "datenschutz"],
                                   data_size="klein", target="smartphone",
                                   description="Spam-Erkennung direkt auf dem Handy")
            print(p.goal_report)          # Empfehlungen, Abratungen, Techniken, Hinweise

        Args:
            choice: Index der Empfehlung (0 = beste).
            apply_techniques: empfohlene Techniken direkt einschalten.
        """
        from .analysis.goals import evaluate_goals, interpret

        data = goals.to_dict() if isinstance(goals, GoalsConfig) else dict(goals or {})
        data.update(goal_fields)
        g = interpret(GoalsConfig.from_dict(data))
        report = evaluate_goals(g)
        if not report.recommended:
            raise ConfigError(f"Keine passende Vorlage für die Aufgabe '{g.task}' – Aufgabe/Beschreibung genauer "
                              f"angeben. Aufgaben: {', '.join(t for t in GoalsConfig().params()['task'].choices)}.")
        pick = report.recommended[min(choice, len(report.recommended) - 1)].preset
        p = cls.from_preset(pick)
        g.use_description = False         # erkannte Angaben sind übernommen; spätere Änderungen gelten wörtlich
        p.config.goals = g
        if g.target:
            d = p.config.deployment
            d.main_target = g.target
            if g.target not in d.targets:
                d.targets = list(d.targets) + [g.target]
        from .analysis.goals import HEAVY

        enabled = []
        if apply_techniques:
            for key, _why in report.techniques:
                if key in HEAVY:
                    continue                  # aufwendige Techniken nur auf ausdrücklichen Wunsch
                try:
                    if p.technique(key, "an"):
                        enabled.append(key)
                except ConfigError:
                    pass                      # passt nicht zu dieser Vorlage
        p._undo.clear()
        p.goal_report = report
        p._note(f"Vorlage '{pick}' nach Zielen gewählt" + (f"; Techniken eingeschaltet: {', '.join(enabled)}"
                                                          if enabled else ""))
        return p

    def set_goals(self, **changes: Any) -> Any:
        """Ziele ändern (z. B. ``set_goals(target="raspberry_pi", priorities=[...])``) und neu bewerten."""
        self._snapshot()
        try:
            self.config.goals.update(changes)
        except Exception:
            self._undo.pop()
            raise
        return self.goal_check()

    def goal_check(self, top: int = 4) -> Any:
        """Bewertet das aktuelle Projekt gegen seine Ziele: passt das Modell, was ist besser, welche Techniken?"""
        from .analysis.goals import evaluate_goals

        self.goal_report = evaluate_goals(self.config.goals, top, project=self)
        return self.goal_report

    def apply_goal_techniques(self, include_heavy: bool = False) -> list[str]:
        """Zu den Zielen passende Techniken einschalten (aufwendige wie DP-SGD/föderiert nur mit include_heavy)."""
        from .analysis.goals import HEAVY

        report = self.goal_check()
        done = []
        for key, _why in report.techniques:
            if key in HEAVY and not include_heavy:
                continue
            try:
                if self.technique(key, "an"):
                    done.append(key)
            except ConfigError:
                pass
        return done

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
        self._invalidate([path])
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
        self._invalidate(paths)
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
        self._invalidate(changed)
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

    # ------------------------------------------------------------ Techniken (Katalog, an/aus)
    def techniques(self, category: str | None = None, status: str | None = None,
                   max_level: str | None = None) -> list:
        """Katalog aller Techniken mit Status für dieses Projekt (``TechniqueState``)."""
        from .techniques import CATALOG, status as tech_status

        limit = Level.parse(max_level)
        rows = [tech_status(self.config, t) for t in CATALOG
                if (category is None or t.in_category(category)) and t.level.visible_at(limit)]
        return [r for r in rows if status is None or r.status == status]

    def technique(self, key: str, state: str = "an") -> list[str]:
        """Eine Technik ein- (``an``) oder ausschalten (``aus``). Rückgängig mit ``undo()``."""
        from .techniques import BY_KEY, disable, enable

        if key not in BY_KEY:
            from .core.errors import suggest

            hint = suggest(key, list(BY_KEY))
            raise ConfigError(f"Unbekannte Technik '{key}'." + (f" Meinten Sie {hint}?" if hint else "") +
                              " Liste: projekt.techniques()")
        trial = self.config.copy()
        memory = {k: dict(v) for k, v in self._technique_memory.items()}
        on = str(state).lower() in ("an", "ein", "on", "true", "1")
        changed = (enable if on else disable)(trial, BY_KEY[key], memory)
        trial.validate()
        if changed:
            self._snapshot()
            self.config = trial
            self._invalidate(changed)
        self._technique_memory = memory
        return changed

    def enable(self, *keys: str) -> list[str]:
        """Techniken einschalten, z. B. ``p.enable("qlora", "gradient_clipping", "early_stopping")``."""
        return [path for k in keys for path in self.technique(k, "an")]

    def disable(self, *keys: str) -> list[str]:
        """Techniken ausschalten (vorherige Werte kommen zurück)."""
        return [path for k in keys for path in self.technique(k, "aus")]

    def technique_overview(self, only_active: bool = False, max_level: str | None = None) -> str:
        """Alle Techniken nach Kategorien mit Status (● an · ○ aus · ◐ teilweise · – passt nicht)."""
        from .techniques import overview

        return overview(self.config, only_active, max_level)

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
        self._invalidate(list(changes))
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
            self._invalidate(_changed_tops(self._redo[-1], self.config.to_dict()))
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
            self._invalidate(_changed_tops(self._undo[-1], self.config.to_dict()))
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

    def _invalidate(self, paths: list[str] | None = None) -> None:
        """Nach Änderungen: Modell/Trainer verwerfen – aber nur, wenn die geänderten Einstellungen sie betreffen.

        Einstellungen zu Vorhersage, Zielsystemen, Zielen, Ablauf, Kompression und Beschreibung lassen ein
        trainiertes Modell unangetastet; Daten-Änderungen laden die Daten neu.
        """
        tops = None if paths is None else {str(p).split(".", 1)[0].split("[", 1)[0] for p in paths}
        if tops is None or "data" in tops:
            self._bundle = None
            self.test_data = None
            self._test_split = None
        if tops is not None and not (tops - _KEEP_MODEL):
            return
        self.model = None
        self.surgeon = None
        self.trainer = None

    # ================================================================ Daten
    def data(self) -> Any:
        """Lädt (einmalig) den konfigurierten Datensatz → ``DatasetBundle``."""
        if self._bundle is None:
            self._bundle = self.config.data.build()
        return self._bundle

    def _info(self) -> dict[str, Any]:
        """Zusatzinfos der Daten (Tokenizer, Klassen, Vorverarbeitung …) – nach ``load`` auch ohne die Daten."""
        if self._bundle is not None:
            return self._bundle.info or {}
        if self._artifacts is not None:
            return self._artifacts
        if self.config.data.name == "none":
            return {}
        return self.data().info or {}

    def data_check(self, train_data: Any = None, val_data: Any = None) -> Any:
        """Prüft die Daten vor dem Training (Klassen, Lecks, Duplikate, Skalen, NaN, Texte …) → ``DataReport``.

        ``str(report)`` zeigt alle Befunde mit Abhilfe; ``apply_data_fixes()`` übernimmt die Vorschläge.
        """
        from .analysis.datacheck import check_data

        info: dict = {}
        if train_data is None:
            bundle = self.data()
            train_data, val_data, info = bundle.train, val_data if val_data is not None else bundle.val, bundle.info
        n_params = None
        try:
            if self.config.data.name != "none":
                n_params = sum(p.numel() for p in self._ensure_model().parameters())
        except Exception:  # noqa: BLE001 - Prüfung soll auch ohne baubares Modell gehen
            pass
        task = {"classification": "klassifikation", "regression": "regression"}.get(str(info.get("task")), "auto")
        self.last_data_report = check_data(train_data, val_data, info, n_params, task)
        return self.last_data_report

    def apply_data_fixes(self, report: Any = None, techniques: bool = True) -> list[str]:
        """Übernimmt die Vorschläge der Datenprüfung (Einstellungen + passende Techniken)."""
        report = report or self.last_data_report or self.data_check()
        changed: list[str] = []
        self._snapshot()
        for path, value in report.settings().items():
            try:
                if self.config.get(path) != value:
                    self.config.set(path, value)
                    changed.append(f"{path} = {value!r}")
            except ConfigError:
                continue
        if techniques:
            from .techniques import BY_KEY

            for key in report.techniques():
                if key in BY_KEY:
                    try:
                        changed += self.technique(key, "an")
                    except ConfigError:
                        continue
        self._invalidate([c.split(" ", 1)[0] for c in changed])
        return changed

    def drift_report(self, new_data: Any, reference: Any = None, with_model: bool = True) -> Any:
        """Vergleicht neue Daten (z. B. aus dem Einsatz) mit den Trainingsdaten → ``DriftReport`` (PSI, KS …).

        Mit trainiertem Modell zusätzlich die Vorhersage-Drift (Verteilung der vorhergesagten Klassen).
        """
        from .analysis.drift import drift_report

        info = {}
        if reference is None:
            bundle = self.data()
            reference, info = bundle.train, bundle.info
        predict = None
        if with_model and self.model is not None:
            predict = self.predict
        return drift_report(reference, new_data, info.get("features"), predict=predict)

    # ================================================================ Bauen
    def _verbose(self, kind: str = "info") -> bool:
        """Konsolenausgabe des Projekts gemäß ``training.verbose`` und ``training.logging.level``."""
        from .training.logger import KINDS, LEVELS

        t = self.config.training
        return t.verbose and LEVELS[t.logging.level] >= KINDS.get(kind, 2)

    def _note(self, text: str) -> None:
        self.notes.append(text)
        if self._verbose():
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
    def train(self, train_data: Any = None, val_data: Any = None, hold_out_test: bool = True) -> list[dict]:
        """Trainiert (baut vorher, falls nötig). Ohne Daten wird der konfigurierte Datensatz genutzt.

        ``hold_out_test=False``: keine Testmenge abteilen (z. B. wenn ``train_data`` schon ohne sie ist).
        """
        if self.model is None:
            self.build()
        info: dict = {}
        if train_data is None and self.config.training.paradigm.name != "reinforcement":
            bundle = self.data()
            train_data = bundle.train
            info = bundle.info or {}
            val_data = val_data if val_data is not None else bundle.val
            if bundle.test is not None:
                self.test_data = bundle.test
            if self.config.data.name == "anomaly":
                val_data = None  # Anomalie-Testmenge hat Labels → separat mit anomaly_report() auswerten
        if hold_out_test:
            train_data = self._hold_out_test(train_data)
        self._fit_warning()
        self.trainer = Trainer(self.model, self.config.training, self.config.memory, self.surgeon,
                               teacher=self._teacher, log_backends=self._loggers, run_name=self.config.name,
                               run_config=self.config.to_dict())
        from .training.metrics import target_inverse

        self.trainer.target_inverse = target_inverse(info.get("target_transform"))
        self.model = self.trainer.module
        self.history = self.trainer.fit(train_data, val_data)
        if self.config.inference.auto_calibrate and self.trainer.val_data is not None:
            try:
                self.calibrate()
            except ConfigError as exc:
                self._note(f"Kalibrierung übersprungen: {exc}")
        d = self.config.deployment
        if self._verbose() and d.show_after_training:
            try:
                print()
                print(self.deployment_overview())
                if d.main_target:
                    print()
                    print(self.recommend_for(d.main_target))
            except Exception as exc:  # noqa: BLE001 - Übersicht darf das Training nie scheitern lassen
                print(f"  (Reduktions-Übersicht nicht verfügbar: {exc})")
        return self.history

    def _hold_out_test(self, train_data: Any) -> Any:
        """``training.test_split > 0``: einen Teil der Trainingsdaten als Testmenge zurückhalten (einmalig)."""
        t = self.config.training
        if t.test_split <= 0 or train_data is None or isinstance(train_data, torch.utils.data.DataLoader):
            return train_data
        cached = self._test_split
        if cached is not None and cached[0] is train_data:
            return cached[1]                      # gleiche Daten → gleiche Testmenge (kein Leck beim Neutrainieren)
        if cached is None and self.test_data is not None:
            return train_data                     # eigene Testmenge des Datensatzes (data.test)
        from .training.data_utils import as_dataset, split_dataset

        rest, test = split_dataset(as_dataset(train_data), t.test_split, t.seed, t.validation_strategy)
        if test is None:
            return train_data
        self.test_data = test
        self._test_split = (train_data, rest)
        self._note(f"Testmenge: {len(test)} Beispiele zurückgehalten – erst am Ende mit Project.test() auswerten.")
        return rest

    def test(self, data: Any = None) -> dict[str, float]:
        """Endgültige Bewertung auf Daten, die das Training **nie** gesehen hat (Präfix ``test_``).

        Ohne ``data``: die mit ``training.test_split`` zurückgehaltene Testmenge bzw. ``data.test`` des Datensatzes.
        Validierungswerte sind durch Early Stopping/Suche leicht geschönt – der Test zeigt die echte Qualität.
        """
        if self.trainer is None:
            raise ConfigError("Erst trainieren (train), dann testen.")
        data = data if data is not None else self.test_data
        if data is None:
            raise ConfigError("Keine Testmenge vorhanden – training.test_split > 0 setzen (z. B. "
                              "project.turn_on('training.test_split')) und neu trainieren, oder Daten übergeben.",
                              "training.test_split")
        result = self.trainer.evaluate(data, prefix="test_")
        if self._verbose("eval"):
            print("  Test: " + "  ".join(f"{k}={v:.4g}" for k, v in result.items()))
        self.last_test = dict(result)
        return result

    def _fit_warning(self) -> None:
        """Mit gesetztem ``memory.memory_budget_gb``: vor dem Training prüfen und Notfall-Optionen nennen."""
        if not (self.config.memory.memory_budget_gb and self._verbose("warnung")):
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
        if self._verbose():
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
        sp._loggers = list(self._loggers)
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
        trainer = Trainer(model, t, self.config.memory, self.surgeon, teacher=teacher, log_backends=self._loggers,
                          run_name=f"{self.config.name}_weiterlernen", run_config=self.config.to_dict())
        if method == "ewc":
            batches = self._batches(old, 8)
            fisher = fisher_information(trainer.model, batches, calibration_loss)
            trainer.penalties.append(ewc_penalty(trainer.model, fisher, strength * 100.0))
        self.trainer, self.model = trainer, trainer.module
        hist = trainer.fit(train_data, val_data)
        self.history = self.history + hist
        return hist

    # ================================================================ Ablauf (Suche, Schwarm, Evolution …)
    def tune(self, method: str | None = None, space: dict | None = None, trials: int | None = None,
             steps_per_trial: int | None = None, **settings: Any) -> Any:
        """Automatische Suche (``random`` | ``grid`` | ``evolution`` | ``pso`` | ``pbt``) – Einstellungen aus
        ``workflow.search``, hier überschreibbar. Übernimmt das Beste (``apply_best``)."""
        from .workflow import run_search

        s = self.config.workflow.search.copy()
        for key, value in (("method", method), ("space", space), ("trials", trials),
                           ("steps_per_trial", steps_per_trial), *settings.items()):
            if value is not None:
                s.set(key, value)
        return run_search(self, s)

    def federated_train(self, **settings: Any) -> list[dict]:
        """Föderiertes Lernen (FedAvg/FedProx) mit simulierten Teilnehmern (``workflow.federated``)."""
        from .workflow import federated_training

        f = self.config.workflow.federated.copy()
        for key, value in settings.items():
            f.set(key, value)
        return federated_training(self, f)

    def model_soup(self, runs: int | None = None, mode: str | None = None) -> dict[str, Any]:
        """Model Soup: mehrere Läufe vom selben Startpunkt trainieren und die Gewichte mitteln."""
        from .workflow import model_soup

        w = self.config.workflow
        return model_soup(self, runs or max(2, w.soup_runs), mode or w.soup_mode)

    def run(self) -> dict[str, Any]:
        """Führt den ganzen **Ablauf** aus (``workflow``): Suche → Training (normal / föderiert / Model Soup)
        → Kreuzvalidierung → Kompression → Export. Jede Stufe ist einzeln schaltbar."""
        w = self.config.workflow
        report: dict[str, Any] = {}
        if w.search.method != "none":
            report["suche"] = self.tune()
        if w.federated.enabled:
            report["foederiert"] = self.federated_train()
        elif w.soup_runs > 1:
            report["model_soup"] = self.model_soup()
        elif self.config.workflow.search.method != "pbt":       # PBT hat bereits trainiert
            report["training"] = self.train()
        if w.cross_validation >= 2:
            report["kreuzvalidierung"] = self.cross_validate(w.cross_validation)
        if w.compress_after_training:
            steps = list(self.config.compression.steps)
            if not steps:
                target = self.config.deployment.main_target or "smartphone"
                steps = [dict(s) for s in self.recommend_for(target).steps if s["name"] != "heal"]
            if steps:
                report["kompression"] = self.compress(steps)
        if w.export_format != "none":
            path = w.export_path or str(Path(self.config.output_dir) / self.config.name / "modell")
            report["export"] = self.export(path, w.export_format)
        return report

    def find_lr(self, min_lr: float = 1e-7, max_lr: float = 1.0, steps: int = 100, apply: bool = False) -> Any:
        """Lernraten-Finder (LR-Range-Test) auf einer Kopie des Modells; ``apply=True`` übernimmt die Empfehlung."""
        from .training.lr_finder import find_lr

        result = find_lr(self, min_lr, max_lr, steps)
        if apply and result.suggestion is not None:
            self.set("training.optimizer.lr", float(f"{result.suggestion:.3g}"))
        return result

    def cross_validate(self, k: int = 5, steps: int | None = None, seed: int = 0) -> dict[str, Any]:
        """k-fache Kreuzvalidierung: k Modelle auf je k−1 Teilen trainieren, auf dem übrigen Teil messen.

        Returns:
            ``{"folds": [metriken je Teil], "mittel": {...}, "std": {...}}`` – zeigt, wie stabil das Ergebnis ist
            (kleine Datensätze, ehrliche Schätzung der Generalisierung).
        """
        bundle = self.data()
        data = bundle.train
        if not isinstance(data, dict):
            raise ConfigError("cross_validate braucht Daten als dict von Tensoren (z. B. die eingebauten Datensätze).")
        n = next(iter(v for v in data.values() if isinstance(v, torch.Tensor))).shape[0]
        if k < 2 or k > n:
            raise ConfigError(f"k muss zwischen 2 und der Anzahl Beispiele ({n}) liegen.")
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        folds = [perm[i::k] for i in range(k)]
        results = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = torch.cat([f for j, f in enumerate(folds) if j != i])
            pick = (lambda idx: {key: v[idx] if isinstance(v, torch.Tensor) and v.shape[:1] == (n,) else v
                                 for key, v in data.items()})
            cfg = self.config.copy()
            cfg.training.validation_split = 0.0
            cfg.deployment.show_after_training = False
            cfg.training.verbose = False
            if steps is not None:
                cfg.training.max_steps = steps
            q = Project(cfg)
            q._bundle = bundle
            q.train(pick(train_idx), None)
            metrics = q.evaluate(pick(val_idx))
            results.append(metrics)
            if self._verbose():
                print(f"  Teil {i + 1}/{k}: " + "  ".join(f"{m}={v:.4g}" for m, v in metrics.items()))
        keys = results[0].keys()
        mean = {m: sum(r[m] for r in results) / k for m in keys}
        std = {m: (sum((r[m] - mean[m]) ** 2 for r in results) / max(1, k - 1)) ** 0.5 for m in keys}
        if self._verbose():
            print("  Kreuzvalidierung: " + "  ".join(f"{m}={mean[m]:.4g}±{std[m]:.2g}" for m in keys))
        return {"folds": results, "mittel": mean, "std": std}

    def add_logger(self, logger: Any) -> "Project":
        """Eigenen Logger einbinden: ``logging.Logger``, ein ``LogBackend`` oder ein Objekt mit (einigen der)
        Methoden ``start``, ``message(kind, text)``, ``metrics(step, werte, phase)``, ``histogram``, ``close``.

        Umfang und Ziele aus der Konfiguration: ``training.logging`` (Stufe, Backends, Intervalle, Rotation).
        """
        self._loggers.append(logger)
        return self

    def evaluate(self, data: Any = None) -> dict[str, float]:
        if self.trainer is None:
            raise ConfigError("Erst trainieren (train) oder Trainer erzeugen.")
        if data is None:
            bundle = self.data()
            data = bundle.val if bundle.val is not None else bundle.train
        return self.trainer.evaluate(data)

    # ================================================================ Vertrauen
    def _eval_data(self, data: Any = None, prefer_test: bool = False) -> Any:
        if data is not None:
            return data
        candidates = [self.test_data] if prefer_test else []
        candidates.append(getattr(self.trainer, "val_data", None))
        for c in candidates:
            if c is not None:
                return c
        try:
            val = self.data().val if self.config.data.name != "none" else None
        except (OSError, ValueError):                 # z. B. nach load() ohne die Original-Daten
            val = None
        if val is not None:
            return val
        raise ConfigError("Keine Validierungsdaten – erst trainieren (train) oder Daten übergeben.")

    @torch.no_grad()
    def _outputs(self, data: Any, limit: int = 20000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(Ausgaben, Ziele, Eingaben) über einen Datensatz im Eval-Modus."""
        from .training.data_utils import make_loader
        from .training.paradigms import main_output

        model = self._ensure_model()
        was = model.training
        model.eval()
        device = self._device()
        outs, targets, inputs, n = [], [], [], 0
        try:
            for batch in make_loader(data, self.config.training.eval_batch_size or self.config.training.batch_size,
                                     False):
                batch = to_device(batch, device)
                if "targets" not in batch:
                    raise ConfigError("Die Daten haben keine Ziele ('targets').")
                out = main_output(call_model(model, batch["inputs"], attention_mask=batch.get("attention_mask"),
                                             lengths=batch.get("lengths")))
                outs.append(out.float().cpu())
                targets.append(batch["targets"].cpu())
                inputs.append(batch["inputs"].cpu() if isinstance(batch["inputs"], torch.Tensor) else None)
                n += out.shape[0]
                if n >= limit:
                    break
        finally:
            model.train(was)
        x = torch.cat(inputs) if all(i is not None for i in inputs) else None
        return torch.cat(outs), torch.cat(targets), x

    def _is_classification(self, out: torch.Tensor, target: torch.Tensor | None = None) -> bool:
        task = str(self._info().get("task", ""))
        if task in ("classification", "regression"):
            return task == "classification"
        if target is not None:
            return not target.is_floating_point() and out.dim() == target.dim() + 1
        return out.dim() == 2 and out.shape[-1] > 1

    def calibrate(self, data: Any = None) -> dict[str, Any]:
        """Stellt Temperatur, Schwelle und Conformal-Grenze auf Validierungsdaten ein (``project.inference``).

        Ergebnis: Kalibrierungsfehler (ECE) vorher/nachher, Temperatur, Schwelle, Conformal-Abdeckung.
        """
        from .analysis.inference import (best_threshold, conformal_quantile, conformal_scores, fit_temperature)
        from .training.metrics import METRICS, target_inverse

        from .training.paradigms import is_segmentation

        ic = self.config.inference
        out, target, _ = self._outputs(self._eval_data(data))
        if is_segmentation(out, target):                 # je Pixel kalibrieren
            out, target = out.permute(0, 2, 3, 1).reshape(-1, out.shape[1]), target.reshape(-1)
        result: dict[str, Any] = {}
        self._snapshot()
        if self._is_classification(out, target) and out.dim() == 2:
            ece = METRICS.get("ece").factory(METRICS.get("ece").config_cls())
            result["ece_vorher"] = ece(out, target)
            t = fit_temperature(out, target)
            ic.temperature = t
            scaled = out / t
            result.update(temperatur=t, ece_nachher=ece(scaled, target))
            if ic.optimize_threshold != "keine" and out.shape[1] == 2:
                ic.threshold = best_threshold(scaled.softmax(-1)[:, 1], target, ic.optimize_threshold)
                result["schwelle"] = ic.threshold
            if ic.conformal_alpha:
                ic.conformal_q = conformal_quantile(conformal_scores(scaled, target, True), ic.conformal_alpha)
                result["conformal_q"] = ic.conformal_q
        else:
            inverse = target_inverse(self._info().get("target_transform"))
            pred, tgt = inverse(out, target) if inverse else (out, target)
            if ic.conformal_alpha:
                ic.conformal_q = conformal_quantile(conformal_scores(pred, tgt, False), ic.conformal_alpha)
                result["intervall_halbbreite"] = ic.conformal_q
        if self._verbose():
            print("  Kalibrierung: " + "  ".join(f"{k}={v:.4g}" for k, v in result.items()))
        self.last_calibration = dict(result)
        return result

    def _forward_fn(self) -> Any:
        from .training.paradigms import main_output

        model = self._ensure_model()
        device = self._device()
        def forward(x: Any) -> torch.Tensor:
            if isinstance(x, dict):                     # z. B. Texte: {"inputs", "attention_mask"}
                x = to_device(x, device)
                return main_output(call_model(model, x["inputs"], attention_mask=x.get("attention_mask"))).float()
            return main_output(call_model(model, x.to(device) if isinstance(x, torch.Tensor) else x)).float()

        return forward

    @torch.no_grad()
    def _samples(self, inputs: torch.Tensor) -> torch.Tensor:
        """Ausgaben ``[S, N, …]`` (S = 1 ohne Unsicherheitsschätzung)."""
        from .analysis.inference import sample_outputs

        ic = self.config.inference
        model = self._ensure_model()
        was = model.training
        model.eval()
        try:
            if ic.uncertainty == "keine":
                return self._forward_fn()(inputs).unsqueeze(0).cpu()
            return sample_outputs(model, self._forward_fn(), inputs, ic.uncertainty, ic.samples, ic.tta_transforms,
                                  ic.tta_noise).cpu()
        finally:
            model.train(was)

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        """Kalibrierte Klassenwahrscheinlichkeiten ``[N, K]`` (Temperatur, MC-Dropout/TTA laut ``inference``)."""
        s = self._samples(inputs)
        return (s / self.config.inference.temperature).softmax(-1).mean(0)

    def predict_labels(self, inputs: torch.Tensor, names: bool = True) -> list:
        """Lesbare Vorhersagen: Klassennamen (mit Schwelle und Enthaltung ``'unsicher'``) bzw. Zahlen in echten
        Einheiten (Rückrechnung der Ziel-Skalierung)."""
        return self.predict_with_uncertainty(inputs, names)["vorhersage"]

    def predict_with_uncertainty(self, inputs: torch.Tensor, names: bool = True) -> dict[str, Any]:
        """Vorhersage + Sicherheit: Wahrscheinlichkeit, Entropie, Streuung (MC-Dropout/TTA), Conformal-Menge
        bzw. -Intervall – gemäß ``project.inference``."""
        from .analysis.inference import entropy, prediction_sets
        from .training.metrics import target_inverse

        ic = self.config.inference
        s = self._samples(inputs)
        info = self._info()
        label_names = info.get("label_names") if names else None
        if info.get("task") == "segmentation" and s.dim() == 5:      # [S, N, C, H, W] → Klassenkarte je Pixel
            probs = (s / ic.temperature).softmax(2).mean(0)
            conf, idx = probs.max(1)
            out = {"vorhersage": idx, "wahrscheinlichkeit": conf, "wahrscheinlichkeiten": probs,
                   "klassen": label_names, "entropie": entropy(probs.movedim(1, -1))}
            if ic.abstain_below is not None:
                out["unsicher"] = conf < ic.abstain_below
            return out
        if self._is_classification(s[0]):
            probs = (s / ic.temperature).softmax(-1)
            mean = probs.mean(0)
            conf, idx = mean.max(-1)
            if ic.threshold is not None and mean.shape[-1] == 2:
                idx = (mean[:, 1] >= ic.threshold).long()
                conf = mean.gather(1, idx[:, None]).squeeze(1)
            labels: list = [label_names[i] if label_names and i < len(label_names) else i for i in idx.tolist()]
            if ic.abstain_below is not None:
                labels = [lab if float(c) >= ic.abstain_below else "unsicher" for lab, c in zip(labels, conf)]
            out = {"vorhersage": labels, "wahrscheinlichkeit": conf, "wahrscheinlichkeiten": mean,
                   "entropie": entropy(mean)}
            if s.shape[0] > 1:
                out["streuung"] = probs.std(0).gather(1, idx[:, None]).squeeze(1)
            if ic.conformal_q is not None:
                sets = prediction_sets(mean, ic.conformal_q)
                out["menge"] = [[label_names[i] if label_names and i < len(label_names) else i for i in m]
                                for m in sets]
            return out
        inverse = target_inverse(info.get("target_transform"))
        values = s.mean(0)
        spread = s.std(0) if s.shape[0] > 1 else None
        if inverse is not None:
            values = inverse(values, values)[0]
            if spread is not None:
                spread = (inverse(s, s)[0]).std(0)
        flat = values.reshape(values.shape[0], -1)
        out = {"vorhersage": [float(v[0]) if v.numel() == 1 else v.tolist() for v in flat], "werte": values}
        if spread is not None:
            out["streuung"] = spread
        if ic.conformal_q is not None:
            out["intervall"] = (values - ic.conformal_q, values + ic.conformal_q)
        return out

    def explain_prediction(self, inputs: torch.Tensor, target: Any = None, method: str = "auto",
                           index: int = 0) -> Any:
        """Warum diese Vorhersage? Wichtigkeit jedes Merkmals/Tokens/Bildbereichs für das Beispiel ``index``.

        ``method``: auto | gradient | integrated_gradients | occlusion. ``target``: Klasse (Index oder Name).
        """
        from .analysis.explain import METHODS, Explanation, gradient_x_input, integrated_gradients, occlusion

        if method not in METHODS:
            raise ConfigError(f"Unbekannte Methode '{method}'. Erlaubt: {', '.join(METHODS)}.")
        x = inputs[index:index + 1] if inputs.dim() > 1 else inputs.unsqueeze(0)
        info = self._info()
        label_names = info.get("label_names") or []
        if isinstance(target, str):
            if target not in label_names:
                raise ConfigError(f"Klasse '{target}' unbekannt. Klassen: {label_names}.")
            target = label_names.index(target)
        tokens = not x.is_floating_point()
        if method == "auto":
            method = "occlusion" if tokens else ("integrated_gradients" if x.dim() == 2 else "gradient")
        model = self._ensure_model()
        was = model.training
        model.eval()
        forward = self._forward_fn()
        try:
            fn = {"gradient": gradient_x_input, "integrated_gradients": integrated_gradients,
                  "occlusion": occlusion}[method]
            attr = fn(forward, x.to(self._device()), target).cpu()
            with torch.no_grad():
                out = forward(x)
        finally:
            model.train(was)
        if tokens:
            tok = info.get("tokenizer")
            ids = x[0].tolist()
            names = [f"{i}:{tok.itos[t] if tok is not None and t < len(tok.itos) else t}" for i, t in enumerate(ids)]
        else:
            names = list(info.get("features") or [])
        pred_idx = int(out[0].argmax()) if out.dim() == 2 and out.shape[-1] > 1 else None
        shown = (label_names[pred_idx] if pred_idx is not None and pred_idx < len(label_names) else pred_idx) \
            if pred_idx is not None else [round(v, 4) for v in out[0].reshape(-1).tolist()][:4]
        tgt_name = (label_names[target] if isinstance(target, int) and target < len(label_names) else target)
        return Explanation(method, attr, names, shown, tgt_name)

    def feature_importance(self, data: Any = None, metric: str | None = None, repeats: int = 3) -> Any:
        """Globale Merkmals-Wichtigkeit per Permutation auf Validierungsdaten (welche Spalten zählen wirklich?)."""
        from .analysis.explain import permutation_importance
        from .core import ComponentSpec
        from .training.metrics import direction

        out, target, x = self._outputs(self._eval_data(data))
        if x is None or not x.is_floating_point() or x.dim() != 2:
            raise ConfigError("feature_importance braucht Tabellen-Eingaben [N, Merkmale] – für Bilder/Texte "
                              "explain_prediction(…) verwenden.")
        cls = self._is_classification(out, target)
        metric = metric or ("balanced_accuracy" if cls else "mae")
        fn = ComponentSpec.from_value("metric", metric).build()
        forward = self._forward_fn()
        model = self._ensure_model()
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                imp = permutation_importance(lambda xx: fn(forward(xx).cpu(), target), x, direction(metric) == "max",
                                             self._info().get("features"), repeats, metric=metric)
        finally:
            model.train(was)
        return imp

    def fairness_report(self, groups: Any, data: Any = None, positive_class: int = 1) -> Any:
        """Fairness je Gruppe (Genauigkeit, Positiv-Rate, TPR/FPR, Disparate Impact).

        ``groups``: Gruppe je Beispiel (Liste/Tensor) oder Name einer Merkmalsspalte (CSV, z. B. ``'geschlecht'``).
        Ohne ``data``: Testmenge, sonst Validierung.
        """
        from .analysis.fairness import fairness_report, groups_from_features

        out, target, x = self._outputs(self._eval_data(data, prefer_test=True))
        if not self._is_classification(out, target):
            raise ConfigError("Der Fairness-Bericht ist für Klassifikation gedacht.")
        names = None
        if isinstance(groups, str):
            if x is None or x.dim() != 2:
                raise ConfigError("Gruppen als Spaltenname gehen nur bei Tabellen-Daten.")
            groups, names = groups_from_features(x, list(self._info().get("features") or []), groups)
        pred = out.argmax(-1)
        ic = self.config.inference
        if ic.threshold is not None and out.shape[-1] == 2:
            pred = ((out / ic.temperature).softmax(-1)[:, 1] >= ic.threshold).long()
        self.last_fairness = fairness_report(pred, target, groups, positive_class, names)
        return self.last_fairness

    def robustness_report(self, data: Any = None, epsilons: list[float] | None = None, metric: str | None = None,
                          methods: tuple[str, ...] = ("rauschen", "fgsm", "pgd"), limit: int = 512) -> Any:
        """Wie stark leidet die Qualität unter Rauschen und gezielten Angriffen (FGSM/PGD)?

        ``epsilons``: Störstärken relativ zur Standardabweichung der Eingaben (Standard 0.01, 0.05, 0.1).
        """
        from .analysis.robustness import robustness_report
        from .core import ComponentSpec
        from .training.paradigms import auto_loss

        out, target, x = self._outputs(self._eval_data(data), limit)
        if x is None or not x.is_floating_point():
            raise ConfigError("Robustheitsprüfung braucht Zahlen-Eingaben (Tabellen, Bilder, Signale).")
        cls = self._is_classification(out, target)
        metric = metric or ("accuracy" if cls else "mae")
        fn = ComponentSpec.from_value("metric", metric).build()
        forward = self._forward_fn()
        model = self._ensure_model()
        was = model.training
        model.eval()
        device = self._device()
        try:
            self.last_robustness = robustness_report(lambda xx: forward(xx).cpu(), lambda o, y: auto_loss(o, y),
                                                     lambda o, y: fn(o, y), x.to(device), target,
                                                     epsilons or [0.01, 0.05, 0.1], metric, methods)
            return self.last_robustness
        finally:
            model.train(was)

    def select_for_labeling(self, pool: Any, n: int = 20, strategy: str = "unsicherheit") -> Any:
        """Active Learning: welche unbeschrifteten Beispiele sollten Menschen als Nächstes beschriften?

        ``pool``: Eingaben (Tensor) oder dict mit ``inputs``. Strategien: unsicherheit, marge, entropie, vielfalt,
        zufall. Mit ``inference.uncertainty = 'mc_dropout'`` fließt auch die Modell-Unsicherheit ein.
        """
        from .workflow.active import STRATEGIES, Selection, select

        if strategy not in STRATEGIES:
            raise ConfigError(f"Unbekannte Strategie '{strategy}'. Erlaubt: {', '.join(STRATEGIES)}.")
        x = pool["inputs"] if isinstance(pool, dict) else pool
        probs = self.predict_proba(x)
        idx, scores = select(probs, n, strategy, features=probs, seed=self.config.training.seed or 0)
        names = self._info().get("label_names")
        preds = [names[i] if names and i < len(names) else i for i in probs[idx].argmax(-1).tolist()]
        return Selection(strategy, idx, scores, preds)

    def pseudo_label(self, unlabeled: Any, threshold: float = 0.95, rounds: int = 1) -> dict[str, Any]:
        """Semi-überwachtes Lernen (Self-Training): sichere Vorhersagen auf unbeschrifteten Daten als Labels
        übernehmen und mittrainieren – ``rounds`` Runden. Erst normal trainieren, dann aufrufen.
        """
        from torch.utils.data import ConcatDataset

        from .training.data_utils import DictDataset, as_dataset
        from .workflow.active import pseudo_labels

        if self.trainer is None:
            raise ConfigError("Erst mit den beschrifteten Daten trainieren (train), dann pseudo_label().")
        x = unlabeled["inputs"] if isinstance(unlabeled, dict) else unlabeled
        labeled = self._hold_out_test(self.data().train)
        val = self.trainer.val_data
        report: dict[str, Any] = {"runden": []}
        for r in range(rounds):
            mask, labels = pseudo_labels(self.predict_proba(x), threshold)
            n = int(mask.sum())
            if n == 0:
                self._note(f"Pseudo-Labeling Runde {r + 1}: keine Vorhersage über der Schwelle {threshold}.")
                break
            extra = DictDataset({"inputs": x[mask].cpu(), "targets": labels[mask].cpu()})
            self.train(ConcatDataset([as_dataset(labeled), extra]), val, hold_out_test=False)
            metrics = self.trainer.evaluate(val) if val is not None else {}
            report["runden"].append({"runde": r + 1, "uebernommen": n, "anteil": n / len(x), **metrics})
            self._note(f"Pseudo-Labeling Runde {r + 1}: {n} von {len(x)} Beispielen übernommen.")
        return report

    def predict_raw(self, records: Any) -> dict[str, Any]:
        """Vorhersage aus **Rohdaten** – vorverarbeitet genau wie im Training (auch nach ``load`` ohne die Daten):

        Tabellen: Zeilen als dicts ``{spalte: wert}`` (oder Listen in Spaltenreihenfolge); Texte: Strings;
        Bilder/Audio: Dateipfade oder ``bytes``; sonst Tensoren. Ergebnis wie ``predict_with_uncertainty``.
        """
        from .data.artifacts import kind_of, prepare_raw

        info = self._info()
        kind = kind_of(info)
        if kind in ("anweisung", "sprachmodell"):
            raise ConfigError("Für Sprachmodelle ask(frage) bzw. generate(prompt) verwenden.")
        try:
            x = prepare_raw(info, records)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        return self.predict_with_uncertainty(x)

    def model_card(self, path: str | Path | None = None) -> str:
        """Modellkarte (Markdown): Zweck, Modell, Daten, Training, Ergebnisse (Validierung, Test, Kalibrierung,
        Fairness, Robustheit), Grenzen, Nutzung, Umgebung. Mit ``path`` zusätzlich als Datei."""
        from .analysis.modelcard import model_card

        text = model_card(self)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def serve(self, host: str | None = None, port: int | None = None, api_key: str | None = None) -> None:
        """HTTP-Server starten (``deployment.server``): /vorhersage, /frage, /generieren, /info – siehe ui.serve."""
        from .ui.serve import serve

        self._ensure_model()
        serve(self, host=host, port=port, api_key=api_key)

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
            part = to_device(part, device)
            if mc.param_dtype != "float32":
                from torch.utils._pytree import tree_map

                from .memory.manager import DTYPES
                part = tree_map(lambda t: t.to(DTYPES[mc.param_dtype]) if isinstance(t, torch.Tensor)
                                and t.is_floating_point() else t, part)
            with mem.autocast():
                return call_model(model, part, **kwargs)

        try:
            out, _ = run_chunked(run, inputs, chunk_size, on_oom=mc.oom_strategy == "batch_teilen",
                                 max_splits=mc.oom_max_splits)
        finally:
            model.train(was)
        return out

    def generate(self, prompt: str | torch.Tensor, **generation: Any) -> Any:
        """Text-/Token-Generierung (Sprachmodelle). Mit Zeichen-Tokenizer des Datensatzes auch für Text."""
        model = self._ensure_model()
        info = self._info()
        tok = info.get("tokenizer")
        if info.get("eos_token_id") is not None:
            generation.setdefault("eos_token_id", info["eos_token_id"])
        if isinstance(prompt, str):
            if tok is None:
                raise ConfigError("Für Text-Prompts wird ein Tokenizer im Datensatz benötigt (z. B. Datensatz 'text').")
            ids = torch.tensor([tok.encode(prompt) or [1]], device=self._device())
            out = model.generate(ids, **generation)
            return tok.decode(out[0].tolist())
        return model.generate(prompt.to(self._device()), **generation)

    def ask(self, question: str, **generation: Any) -> str:
        """Frage an ein Instruction-Modell (Datensatz ``instructions``): gleiche Vorlage wie im Training, Antwort
        bis zum Ende-Zeichen. Standard: gierige Auswahl (``temperature=0``)."""
        info = self._info()
        template, tok = info.get("prompt_template"), info.get("tokenizer")
        if template is None or tok is None:
            raise ConfigError("ask() braucht einen Datensatz mit Vorlage (data = 'instructions').")
        prompt = template.format(prompt=question)
        generation.setdefault("temperature", 0.0)
        generation.setdefault("max_new_tokens", 48)
        n = len(tok.encode(prompt))
        model = self._ensure_model()
        if info.get("eos_token_id") is not None:
            generation.setdefault("eos_token_id", info["eos_token_id"])
        ids = torch.tensor([tok.encode(prompt)], device=self._device())
        out = model.generate(ids, **generation)[0, n:].tolist()
        text = tok.decode(out)
        return text.split(info.get("eos_text", "\x00"), 1)[0].strip()

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
        """Speichert Konfiguration (projekt.json), Gewichte (modell.pt) und Verlauf (verlauf.json).

        Im verteilten Training rufen alle Prozesse ``save`` auf (FSDP sammelt die Gewichte ein);
        geschrieben wird nur von Prozess 0.
        """
        from .training.distributed import is_main_process

        d = Path(directory or Path(self.config.output_dir) / self.config.name)
        state = None
        if self.model is not None:
            target = getattr(self.model, "_orig_mod", self.model)
            state = self.trainer.dist.model_state(target) if self.trainer is not None else target.state_dict()
        if not is_main_process():
            return d
        d.mkdir(parents=True, exist_ok=True)
        self.config.save(d / "projekt.json")
        if state is not None:
            torch.save(state, d / "modell.pt")
        (d / "verlauf.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
        from .analysis.modelcard import environment_info
        from .data.artifacts import export_artifacts

        try:
            artifacts = export_artifacts(self._info())
        except Exception:  # noqa: BLE001 - ohne ladbare Daten gibt es eben keine Vorverarbeitung
            artifacts = {}
        if artifacts:
            (d / "vorverarbeitung.json").write_text(json.dumps(artifacts, ensure_ascii=False, default=str),
                                                    encoding="utf-8")
        if self.config.deployment.model_card:
            (d / "umgebung.json").write_text(json.dumps(environment_info(), ensure_ascii=False, indent=1),
                                             encoding="utf-8")
            (d / "MODELLKARTE.md").write_text(self.model_card(), encoding="utf-8")
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
        if (d / "vorverarbeitung.json").exists():
            from .data.artifacts import import_artifacts

            p._artifacts = import_artifacts(json.loads((d / "vorverarbeitung.json").read_text(encoding="utf-8")))
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

