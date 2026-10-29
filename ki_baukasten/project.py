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
from .memory import MemoryConfig, MemoryManager, estimate_memory
from .tensor import TensorRuleConfig, TensorSurgeon
from .training import Trainer, TrainingConfig
from .training.data_utils import to_device
from .training.paradigms import call_model, main_output


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
    tensor_rules = ConfigListParam(TensorRuleConfig, group="Tensor-Chirurgie", level=Level.ADVANCED, tooltip=T(
        "Regeln für Eingriffe in einzelne Tensoren (Einfrieren, Pruning, eigene LR …).",
        "Feinste Kontrolle bis auf einzelne Gewichte.", "Transfer-Learning, Kompression, Experimente."))
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
        self.trainer: Trainer | None = None
        self._bundle: Any = None
        self.history: list[dict] = []
        self.notes: list[str] = []

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
    def from_file(cls, path: str | Path) -> "Project":
        """Projekt aus JSON/YAML-Datei."""
        return cls(ProjectConfig.load(path))

    # ================================================================ Einstellungen
    def set(self, path: str, value: Any) -> "Project":
        self.config.set(path, value)
        self._invalidate()
        return self

    def get(self, path: str) -> Any:
        return self.config.get(path)

    def update(self, changes: dict[str, Any]) -> "Project":
        self.config.update(changes)
        self._invalidate()
        return self

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
        self.trainer = Trainer(self.model, self.config.training, self.config.memory, self.surgeon)
        self.model = self.trainer.model
        self.history = self.trainer.fit(train_data, val_data)
        return self.history

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
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """Vorhersage im Eval-Modus (mit gemischter Präzision, falls konfiguriert)."""
        model = self._ensure_model()
        was = model.training
        model.eval()
        mem = self.trainer.mem if self.trainer is not None else MemoryManager(self.config.memory)
        with mem.autocast():
            out = call_model(model, to_device(inputs, self._device()), **kwargs)
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
    def load(cls, directory: str | Path) -> "Project":
        """Lädt ein mit ``save`` gespeichertes Projekt (inkl. Gewichten)."""
        d = Path(directory)
        p = cls.from_file(d / "projekt.json")
        p.config.auto_fit_dims = False  # gespeicherte Werte sind bereits angepasst
        model = p.build()
        weights = d / "modell.pt"
        if weights.exists():
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        if (d / "verlauf.json").exists():
            p.history = json.loads((d / "verlauf.json").read_text(encoding="utf-8"))
        return p

    def __repr__(self) -> str:
        return f"<Project {self.config.name!r}: model={self.config.model.name}, data={self.config.data.name}, " \
               f"paradigm={self.config.training.paradigm.name}>"


def components_overview(max_level: str | None = None) -> str:
    """Übersicht aller Registries mit Einträgen (für CLI/Doku)."""
    from .core import all_registries

    return "\n\n".join(reg.describe(max_level) for reg in all_registries().values())

