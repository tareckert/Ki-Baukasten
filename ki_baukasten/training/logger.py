"""
ki_baukasten.training.logger
============================

**Logger**: wohin, wie viel und was protokolliert wird.

Umfang (``training.logging.level``)
-----------------------------------
================  =================================================================================
Stufe             Ausgabe
================  =================================================================================
``still``         nichts (auch keine Konsole)
``minimal``       Start, Ende, Warnungen, Validierungen
``normal``        + Fortschritt alle ``every_n_steps`` Schritte (Standard)
``ausfuehrlich``  + Lernrate, Gradientennorm, Durchsatz, Speicher; Meldungen der Callbacks
``debug``         + Metriken jedes Schritts, Gewichts-/Gradientenstatistiken je Tensor
================  =================================================================================

Ziele (``training.logging.backends`` – Registry ``log_backend``, frei kombinierbar)
------------------------------------------------------------------------------------
``console`` · ``datei`` (Textprotokoll mit Rotation) · ``jsonl`` · ``csv`` (Langformat) ·
``tensorboard`` · ``wandb`` · ``mlflow`` · ``python`` (Standard-``logging`` → eigene Handler,
z. B. Firmen-Logging, Syslog, Graylog). Jedes Ziel kann eine eigene Stufe haben – z. B.
Konsole ``minimal``, Datei ``debug``.

Eigene Logger einbinden
-----------------------
* als Komponente: ``@register("log_backend", "mein_ziel", config=...)`` mit einer Fabrik
  ``factory(args) -> LogBackend``,
* zur Laufzeit: ``projekt.add_logger(objekt)`` – jedes Objekt mit (einigen der) Methoden
  ``start``, ``message``, ``metrics``, ``histogram``, ``close`` oder ein ``logging.Logger``.

Fehlt ein optionales Paket (tensorboard, wandb, mlflow), wird das Ziel mit einer Warnung
übersprungen – das Training läuft weiter.
"""

from __future__ import annotations

import csv
import json
import logging as _logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ..core import BaseConfig, ComponentListParam, Level, Param, T, create_registry
from ..core.config import flatten_dict

LEVELS = {"still": 0, "minimal": 1, "normal": 2, "ausfuehrlich": 3, "debug": 4}
#: Mindest-Stufe je Meldungsart.
KINDS = {"warnung": 1, "info": 1, "eval": 1, "fortschritt": 2, "detail": 3, "debug": 4}
_PY_LEVEL = {"warnung": _logging.WARNING, "info": _logging.INFO, "eval": _logging.INFO,
             "fortschritt": _logging.INFO, "detail": _logging.DEBUG, "debug": _logging.DEBUG}

LOG_BACKENDS = create_registry(
    "log_backend", "Logger-Ziele",
    T("Wohin Trainingsprotokolle geschrieben werden (Konsole, Dateien, Dashboards, eigene Logger).",
      "Nachvollziehbare Experimente, Vergleich von Läufen, Überwachung im Team.",
      "Jedes längere Training; mehrere Ziele gleichzeitig möglich."),
    context_doc="factory(args) -> LogBackend",
)


class LogBackend:
    """Basisklasse eines Logger-Ziels (alle Methoden optional)."""

    level: str | None = None          # eigene Stufe (None = Stufe des Loggers)

    def start(self, run: "RunInfo", config: dict) -> None: ...
    def message(self, kind: str, text: str) -> None: ...
    def metrics(self, step: int, values: dict[str, float], phase: str) -> None: ...
    def histogram(self, step: int, name: str, tensor: torch.Tensor) -> None: ...
    def close(self) -> None: ...


class RunInfo:
    def __init__(self, name: str, directory: Path):
        self.name, self.directory = name, directory


# ===========================================================================
# Konfiguration
# ===========================================================================

class LoggingConfig(BaseConfig):
    __title__ = "Logging (Protokolle)"
    __tooltip__ = T(
        "Wohin, wie ausführlich und was beim Training protokolliert wird.",
        "Konsole, Dateien und Dashboards (TensorBoard, W&B, MLflow) – oder ein eigener Logger.",
        "Jedes Training; für Vergleiche und Fehlersuche unverzichtbar.",
    )
    level = Param("normal", choices={
        "still": "nichts ausgeben", "minimal": "Start, Ende, Warnungen, Validierung",
        "normal": "+ Fortschritt (Standard)", "ausfuehrlich": "+ Lernrate, Gradienten, Durchsatz, Speicher",
        "debug": "+ jeder Schritt, Statistiken je Tensor",
    }, tooltip=T(
        "Umfang der Protokolle.", "Mehr Details helfen bei der Fehlersuche, kosten aber Übersicht und etwas Zeit.",
        "Normal; debug bei Problemen, minimal für lange Läufe."))
    backends = ComponentListParam("log_backend", default=["console"], tooltip=T(
        "Logger-Ziele (beliebig kombinierbar, jedes mit eigenen Einstellungen).",
        "Gleichzeitig in Konsole, Datei und Dashboard protokollieren.",
        "z. B. ['console', 'jsonl', 'tensorboard'].",
    ))
    directory = Param("logs", tooltip=T(
        "Basisordner für Protokolldateien (je Lauf ein Unterordner).", "Ordnung bei vielen Läufen.", "logs."))
    run_name = Param(None, type=str, nullable=True, tooltip=T(
        "Name des Laufs (None = Zeitstempel).", "Wiedererkennen im Dashboard und im Ordner.", "z. B. 'lr_3e-4'."))
    every_n_steps = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Protokoll-Intervall in Schritten (None = training.log_every_steps).", "Häufiger = feiner, mehr Daten.",
        "10–100."))
    log_lr = Param(True, tooltip=T("Lernrate protokollieren.", "Zeitplan überprüfen.", "an."), level=Level.ADVANCED)
    log_grad_norm = Param(True, tooltip=T(
        "Gradientennorm protokollieren.", "Explodierende/verschwindende Gradienten früh erkennen.", "an."),
        level=Level.ADVANCED)
    log_throughput = Param(True, tooltip=T(
        "Durchsatz (Beispiele/s, Tokens/s) protokollieren.", "Tempo-Optimierungen messbar machen.", "an."),
        level=Level.ADVANCED)
    log_memory = Param(True, tooltip=T(
        "Speicherverbrauch (GPU-Spitze bzw. RAM) protokollieren.", "Engpässe erkennen.", "an."),
        level=Level.ADVANCED)
    log_config = Param(True, tooltip=T(
        "Alle Einstellungen zu Beginn protokollieren.", "Jeder Lauf bleibt reproduzierbar.", "an."),
        level=Level.ADVANCED)
    weights_every_n_steps = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Alle n Schritte Statistiken (Norm, Mittel, Std) jedes Gewichtstensors protokollieren (None = aus).",
        "Zeigt, welche Schichten lernen, stillstehen oder explodieren; TensorBoard zusätzlich Histogramme.",
        "Analyse, Debugging; 100–1000."), level=Level.ADVANCED)
    gradients_every_n_steps = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Alle n Schritte Gradientennormen je Tensor protokollieren (None = aus).",
        "Findet Schichten mit verschwindenden/explodierenden Gradienten.", "Debugging; 50–500."),
        level=Level.ADVANCED)
    max_file_mb = Param(50.0, min=0.0, tooltip=T(
        "Maximale Größe einer Protokolldatei (MB), danach wird rotiert (0 = unbegrenzt).",
        "Begrenzt den Plattenverbrauch langer Läufe.", "10–100."), level=Level.ADVANCED)
    keep_files = Param(3, min=0, tooltip=T(
        "Wie viele rotierte Protokolldateien aufbewahrt werden.", "Platz vs. Verlauf.", "3–10."),
        level=Level.ADVANCED)
    flush_every = Param(1, min=1, tooltip=T(
        "Dateien alle n Einträge auf die Platte schreiben.", "Höher = weniger Plattenzugriffe, im Absturzfall "
        "gehen die letzten Einträge verloren.", "1 (sicher) bis 50."), level=Level.EXPERT)


# ===========================================================================
# Ziele
# ===========================================================================

class _Rotating:
    """Einfache Datei mit Größenrotation (``datei.1``, ``datei.2`` …)."""

    def __init__(self, path: Path, max_mb: float, keep: int, flush_every: int):
        self.path, self.max_bytes, self.keep, self.flush_every = path, int(max_mb * 2**20), keep, flush_every
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8", newline="")
        self.n = 0

    def write(self, text: str) -> None:
        self.f.write(text)
        self.n += 1
        if self.n % self.flush_every == 0:
            self.f.flush()
        if self.max_bytes and self.f.tell() > self.max_bytes:
            self.rotate()

    def rotate(self) -> None:
        self.f.close()
        if self.keep == 0:
            self.path.unlink(missing_ok=True)
        else:
            self.path.with_name(f"{self.path.name}.{self.keep}").unlink(missing_ok=True)
            for i in range(self.keep - 1, 0, -1):
                src = self.path.with_name(f"{self.path.name}.{i}")
                if src.exists():
                    os.replace(src, self.path.with_name(f"{self.path.name}.{i + 1}"))
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        self.f = self.path.open("a", encoding="utf-8", newline="")

    def close(self) -> None:
        self.f.close()


class BackendConfig(BaseConfig):
    level = Param(None, type=str, nullable=True, choices=list(LEVELS), tooltip=T(
        "Eigene Stufe für dieses Ziel (None = Stufe des Loggers).",
        "z. B. Konsole knapp, Datei ausführlich.", "None."))


class FileBackendConfig(BackendConfig):
    filename = Param(None, type=str, nullable=True, tooltip=T(
        "Dateiname im Laufordner (None = Standardname).", "Eigene Namenskonventionen.", "None."))


class ConsoleBackend(LogBackend):
    def __init__(self, a: BackendConfig):
        self.level = a.level

    def message(self, kind, text):
        print(text, flush=True)


class _FileBackend(LogBackend):
    default_name = "log.txt"

    def __init__(self, a: FileBackendConfig):
        self.a, self.level, self.file = a, a.level, None

    def start(self, run, config):
        lc = config.get("_logging", {})
        self.file = _Rotating(run.directory / (self.a.filename or self.default_name), lc.get("max_file_mb", 50.0),
                              lc.get("keep_files", 3), lc.get("flush_every", 1))
        self._start(run, config)

    def _start(self, run, config): ...

    def close(self):
        if self.file is not None:
            self.file.close()


class TextFileBackend(_FileBackend):
    default_name = "training.log"

    def message(self, kind, text):
        if self.file is not None:
            self.file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{kind}] {text.strip()}\n")

    def metrics(self, step, values, phase):
        if self.file is not None:
            body = "  ".join(f"{k}={v:.6g}" for k, v in values.items())
            self.file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{phase}] schritt={step}  {body}\n")


class JSONLBackend(_FileBackend):
    default_name = "protokoll.jsonl"

    def _w(self, row: dict) -> None:
        if self.file is not None:
            self.file.write(json.dumps({"zeit": time.time(), **row}, ensure_ascii=False, default=str) + "\n")

    def _start(self, run, config):
        self._w({"typ": "start", "lauf": run.name, "config": {k: v for k, v in config.items() if k != "_logging"}})

    def message(self, kind, text):
        self._w({"typ": "meldung", "art": kind, "text": text.strip()})

    def metrics(self, step, values, phase):
        self._w({"typ": "metriken", "phase": phase, "schritt": step, **values})


class CSVBackend(_FileBackend):
    """Langformat ``schritt, phase, name, wert`` – robust, wenn neue Metriken hinzukommen."""

    default_name = "metriken.csv"

    def _start(self, run, config):
        if self.file is not None and self.file.f.tell() == 0:
            self.file.write("schritt,phase,name,wert\n")

    def metrics(self, step, values, phase):
        if self.file is None:
            return
        import io

        buf = io.StringIO()
        w = csv.writer(buf)
        for k, v in values.items():
            w.writerow([step, phase, k, f"{v:.8g}"])
        self.file.write(buf.getvalue())


class PythonLoggingConfig(BackendConfig):
    logger_name = Param("ki_baukasten", tooltip=T(
        "Name des Python-Loggers (logging.getLogger).", "Eigene Handler (Datei, Syslog, Graylog, Firmen-Logging) "
        "lassen sich daran hängen.", "ki_baukasten."))


class PythonLoggingBackend(LogBackend):
    def __init__(self, a: PythonLoggingConfig | None = None, logger: _logging.Logger | None = None):
        self.level = a.level if a is not None else None
        self.logger = logger or _logging.getLogger(a.logger_name if a is not None else "ki_baukasten")

    def message(self, kind, text):
        self.logger.log(_PY_LEVEL.get(kind, _logging.INFO), text.strip())

    def metrics(self, step, values, phase):
        self.logger.info("%s schritt=%d %s", phase, step, " ".join(f"{k}={v:.6g}" for k, v in values.items()))


class TensorBoardBackend(LogBackend):
    def __init__(self, a: BackendConfig):
        self.level, self.writer = a.level, None

    def start(self, run, config):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # noqa: BLE001 - optionales Paket
            warnings.warn(f"TensorBoard nicht verfügbar ({exc}) – 'pip install tensorboard'. Ziel übersprungen.")
            return
        self.writer = SummaryWriter(str(run.directory / "tensorboard"))
        self.writer.add_text("config", "```\n" + json.dumps(config, ensure_ascii=False, indent=1, default=str)[:60000]
                             + "\n```")

    def metrics(self, step, values, phase):
        if self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{phase}/{k}", v, step)

    def histogram(self, step, name, tensor):
        if self.writer is not None:
            self.writer.add_histogram(name, tensor.detach().float().cpu(), step)

    def message(self, kind, text):
        if self.writer is not None and kind in ("warnung", "info"):
            self.writer.add_text(kind, text)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class WandbConfig(BackendConfig):
    project = Param("ki-baukasten", tooltip=T("W&B-Projekt.", "Läufe gruppieren.", "Projektname."))
    entity = Param(None, type=str, nullable=True, tooltip=T("W&B-Team/Benutzer (None = Standard).", "", ""))
    mode = Param("online", choices=["online", "offline", "disabled"], tooltip=T(
        "online, offline (später synchronisieren) oder aus.", "Ohne Internet: offline.", "online."))


class WandbBackend(LogBackend):
    def __init__(self, a: WandbConfig):
        self.a, self.level, self.run = a, a.level, None

    def start(self, run, config):
        try:
            import wandb
        except ImportError:
            warnings.warn("wandb nicht installiert ('pip install wandb') – Ziel übersprungen.")
            return
        self.run = wandb.init(project=self.a.project, entity=self.a.entity, name=run.name, config=config,
                              dir=str(run.directory), mode=self.a.mode, reinit=True)

    def metrics(self, step, values, phase):
        if self.run is not None:
            self.run.log({f"{phase}/{k}": v for k, v in values.items()}, step=step)

    def close(self):
        if self.run is not None:
            self.run.finish()


class MLflowConfig(BackendConfig):
    experiment = Param("ki-baukasten", tooltip=T("MLflow-Experiment.", "Läufe gruppieren.", "Projektname."))
    tracking_uri = Param(None, type=str, nullable=True, tooltip=T(
        "MLflow-Server bzw. Ordner (None = ./mlruns oder MLFLOW_TRACKING_URI).", "Zentrale Ablage im Team.", ""))


class MLflowBackend(LogBackend):
    def __init__(self, a: MLflowConfig):
        self.a, self.level, self.mlflow = a, a.level, None

    def start(self, run, config):
        try:
            import mlflow
        except ImportError:
            warnings.warn("mlflow nicht installiert ('pip install mlflow') – Ziel übersprungen.")
            return
        if self.a.tracking_uri:
            mlflow.set_tracking_uri(self.a.tracking_uri)
        mlflow.set_experiment(self.a.experiment)
        mlflow.start_run(run_name=run.name)
        flat = {k: str(v)[:250] for k, v in flatten_dict(config).items()}
        for i in range(0, len(flat), 100):                 # MLflow: höchstens 100 Parameter pro Aufruf
            mlflow.log_params(dict(list(flat.items())[i:i + 100]))
        self.mlflow = mlflow

    def metrics(self, step, values, phase):
        if self.mlflow is not None:
            self.mlflow.log_metrics({f"{phase}_{k}": v for k, v in values.items()}, step=step)

    def close(self):
        if self.mlflow is not None:
            self.mlflow.end_run()


R = LOG_BACKENDS.register
R("console", lambda a: ConsoleBackend(a), config=BackendConfig, aliases=("konsole",), tooltip=T(
    "Ausgabe im Terminal.", "Fortschritt direkt sehen.", "Standard."))
R("datei", lambda a: TextFileBackend(a), config=FileBackendConfig, aliases=("file", "text"), tooltip=T(
    "Textprotokoll mit Zeitstempel und Größenrotation (training.log).", "Nachlesen nach dem Lauf, Server ohne Terminal.",
    "Lange Läufe, Cluster."))
R("jsonl", lambda a: JSONLBackend(a), config=FileBackendConfig, tooltip=T(
    "Alles als JSON-Zeilen (Konfiguration, Meldungen, Metriken).", "Maschinenlesbar, robust, einfach auszuwerten.",
    "Eigene Auswertungen, Dashboards."))
R("csv", lambda a: CSVBackend(a), config=FileBackendConfig, tooltip=T(
    "Metriken als CSV im Langformat (schritt, phase, name, wert).", "Direkt in Excel/Pandas.", "Tabellen-Auswertung."))
R("python", lambda a: PythonLoggingBackend(a), config=PythonLoggingConfig, aliases=("logging",), tooltip=T(
    "An das Python-'logging' weitergeben.", "Integration in bestehende Logging-Infrastruktur (Handler, Syslog, "
    "Graylog, Firmen-Logger).", "Produktionsumgebungen."))
R("tensorboard", lambda a: TensorBoardBackend(a), config=BackendConfig, level=Level.ADVANCED, tooltip=T(
    "TensorBoard: Kurven, Histogramme, Konfiguration (benötigt 'tensorboard').", "Interaktive Auswertung im Browser.",
    "Experimente vergleichen ('tensorboard --logdir logs')."))
R("wandb", lambda a: WandbBackend(a), config=WandbConfig, level=Level.ADVANCED, tooltip=T(
    "Weights & Biases (benötigt 'wandb').", "Team-Dashboards, Vergleich vieler Läufe.", "Teams, Hyperparameter-Suche."))
R("mlflow", lambda a: MLflowBackend(a), config=MLflowConfig, level=Level.ADVANCED, tooltip=T(
    "MLflow-Tracking (benötigt 'mlflow').", "Experimente und Modelle zentral verwalten.", "Unternehmen, MLOps."))


# ===========================================================================
# Laufzeit
# ===========================================================================

class _Adapter(LogBackend):
    """Macht beliebige Objekte (z. B. ``logging.Logger``) zu einem Logger-Ziel."""

    def __init__(self, obj: Any):
        self.obj, self.level = obj, getattr(obj, "level", None) if not isinstance(obj, _logging.Logger) else None

    def __getattribute__(self, name: str) -> Any:
        if name in ("start", "message", "metrics", "histogram", "close"):
            fn = getattr(object.__getattribute__(self, "obj"), name, None)
            return fn if callable(fn) else (lambda *a, **k: None)
        return object.__getattribute__(self, name)


def as_backend(obj: Any) -> LogBackend:
    if isinstance(obj, LogBackend):
        return obj
    if isinstance(obj, _logging.Logger):
        return PythonLoggingBackend(logger=obj)
    return _Adapter(obj)


class RunLogger:
    """Verteilt Meldungen und Metriken an alle Ziele – mit Stufen je Ziel."""

    def __init__(self, cfg: LoggingConfig | None = None, verbose: bool = True, main: bool = True,
                 extra: list | None = None, default_every: int = 10):
        self.cfg = cfg or LoggingConfig()
        self.level = LEVELS[self.cfg.level] if verbose else 0
        self.main = main
        self.every = self.cfg.every_n_steps or default_every
        self.backends: list[LogBackend] = []
        if main:
            self.backends = [spec.build() for spec in self.cfg.backends] + [as_backend(o) for o in (extra or [])]
        self.run: RunInfo | None = None

    def _lvl(self, backend: LogBackend) -> int:
        own = getattr(backend, "level", None)
        if own in LEVELS:
            return LEVELS[own]
        if isinstance(backend, ConsoleBackend):
            return self.level              # Konsole folgt zusätzlich 'training.verbose'
        return LEVELS[self.cfg.level]

    def wants(self, kind: str) -> bool:
        need = KINDS.get(kind, 2)
        return any(self._lvl(b) >= need for b in self.backends)

    def start(self, name: str, config: dict) -> None:
        if not self.backends:
            return
        run_name = self.cfg.run_name or f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        self.run = RunInfo(run_name, Path(self.cfg.directory) / run_name)
        payload = dict(config) if self.cfg.log_config else {}
        payload["_logging"] = {"max_file_mb": self.cfg.max_file_mb, "keep_files": self.cfg.keep_files,
                               "flush_every": self.cfg.flush_every}
        for b in self.backends:
            try:
                b.start(self.run, payload)
            except Exception as exc:  # noqa: BLE001 - ein defektes Ziel darf das Training nicht stoppen
                warnings.warn(f"Logger-Ziel {type(b).__name__} konnte nicht starten: {exc}")

    def message(self, kind: str, text: str) -> None:
        need = KINDS.get(kind, 2)
        for b in self.backends:
            if self._lvl(b) >= need:
                b.message(kind, text)

    def metrics(self, step: int, values: dict[str, Any], phase: str = "train", kind: str = "fortschritt") -> None:
        clean = {k: float(v) for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        need = KINDS.get(kind, 2)
        for b in self.backends:
            if self._lvl(b) >= need and not isinstance(b, ConsoleBackend):
                b.metrics(step, clean, phase)

    def histogram(self, step: int, name: str, tensor: torch.Tensor) -> None:
        for b in self.backends:
            b.histogram(step, name, tensor)

    def close(self) -> None:
        for b in self.backends:
            try:
                b.close()
            except Exception:  # noqa: BLE001
                pass


def memory_now() -> dict[str, float]:
    """Speicher in MB: GPU-Spitze (CUDA) bzw. Prozess-RAM."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return {"gpu_mb_spitze": torch.cuda.max_memory_allocated() / 2**20}
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"ram_mb_spitze": kb / 1024}
    except Exception:  # noqa: BLE001 - z. B. Windows
        return {}
