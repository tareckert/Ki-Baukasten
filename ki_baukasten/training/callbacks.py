"""
ki_baukasten.training.callbacks
===============================

Callbacks – Einhängepunkte in den Trainingsablauf.

Ein Callback kann an folgenden Stellen eingreifen (alle optional):

=============================  ================================================
Methode                        Zeitpunkt
=============================  ================================================
``on_train_start(trainer)``    vor dem ersten Schritt
``on_before_step(trainer, k)`` nach Backward + Clipping, vor ``optimizer.step``
``on_step_end(trainer, m)``    nach jedem Optimierungsschritt (m = Metriken)
``on_eval_end(trainer, m)``    nach jeder Evaluierung
``on_epoch_end(trainer, e, m)`` am Ende jeder Epoche
``on_train_end(trainer)``      nach dem Training
=============================  ================================================

``trainer.should_stop = True`` beendet das Training sauber.

Registry ``callback``
---------------------
Fabrik-Signatur: ``factory(args) -> Callback``. Eigene Callbacks einfach von
``Callback`` ableiten und registrieren.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from ..core import BaseConfig, Level, Param, T, create_registry

CALLBACKS = create_registry(
    "callback",
    "Callbacks",
    T(
        "Erweiterungen, die sich in den Trainingsablauf einklinken.",
        "Automatisieren Abbruch, Speichern, Protokollieren, Überwachen.",
        "Jedes längere Training.",
    ),
    context_doc="factory(args) -> Callback",
)


class Callback:
    """Basisklasse (alle Methoden sind optionale No-ops)."""

    def on_train_start(self, trainer: Any) -> None: ...
    def on_before_step(self, trainer: Any, key: str) -> None: ...
    def on_step_end(self, trainer: Any, metrics: dict) -> None: ...
    def on_eval_end(self, trainer: Any, metrics: dict) -> None: ...
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None: ...
    def on_train_end(self, trainer: Any) -> None: ...


def _better(value: float, best: float | None, mode: str, delta: float) -> bool:
    if best is None or math.isnan(best):
        return True
    return value < best - delta if mode == "min" else value > best + delta


# ===========================================================================
# Konfigurationen
# ===========================================================================

class MonitorConfig(BaseConfig):
    monitor = Param("val_loss", tooltip=T(
        "Beobachtete Metrik (z. B. val_loss, val_accuracy).", "Kriterium für 'besser'.", "val_loss."))
    mode = Param("min", choices=["min", "max"], tooltip=T(
        "min: kleiner ist besser (Verluste); max: größer ist besser (Genauigkeit).", "Richtung.", "Passend zur Metrik."))


class EarlyStoppingConfig(MonitorConfig):
    patience = Param(5, min=0, tooltip=T(
        "Evaluierungen ohne Verbesserung bis zum Abbruch.", "Verhindert Überanpassung und spart Zeit.", "3–10."))
    min_delta = Param(0.0, min=0.0, tooltip=T(
        "Mindestverbesserung, die als Verbesserung zählt.", "Ignoriert Rauschen.", "1e-4."), level=Level.ADVANCED)
    restore_best = Param(True, tooltip=T(
        "Am Ende die besten Gewichte wiederherstellen.", "Man behält das beste, nicht das letzte Modell.", "an."))


class CheckpointConfig(MonitorConfig):
    directory = Param("checkpoints", tooltip=T("Zielordner.", "Hier landen die Checkpoints.", "Projektordner."))
    every_n_steps = Param(None, type=int, min=1, tooltip=T(
        "Zusätzlich alle n Schritte speichern (None = nur bei Evaluierung).", "Absicherung langer Läufe.", "1000."))
    save_best_only = Param(True, tooltip=T(
        "Nur bei Verbesserung speichern (beste.pt).", "Spart Speicherplatz.", "an."))
    keep_last = Param(2, min=1, tooltip=T(
        "Wie viele periodische Checkpoints behalten.", "Speicherplatz vs. Rückfalloptionen.", "2–5."),
        level=Level.ADVANCED)


class LoggerConfig(BaseConfig):
    path = Param("training_log.csv", tooltip=T("Dateipfad des Protokolls.", "Hier werden Metriken angehängt.",
                                               "Projektordner."))


class GradientMonitorConfig(BaseConfig):
    every_n_steps = Param(50, min=1, tooltip=T("Messintervall.", "Wie oft Gradienten-Normen gemessen werden.", "50."))
    top_k = Param(5, min=1, tooltip=T(
        "Anzahl Parameter mit den größten Gradienten, die ausgegeben werden.", "Findet explodierende Schichten.", "5."))


class TimeLimitConfig(BaseConfig):
    max_minutes = Param(60.0, min=0.0, tooltip=T(
        "Maximale Trainingsdauer in Minuten.", "Feste Zeitbudgets.", "Cluster-Zeitslots."))


class GradualUnfreezeConfig(BaseConfig):
    patterns = Param(["head", "blocks\\.3", "blocks\\.2", "blocks\\.1", "blocks\\.0", "embed"], item_type=str, tooltip=T(
        "Regex-Gruppen in Auftau-Reihenfolge (zuerst trainierbar → zuletzt).",
        "Schrittweises Auftauen schützt vortrainiertes Wissen (ULMFiT).", "Transfer-Learning."))
    every_epochs = Param(1, min=1, tooltip=T("Alle n Epochen eine Gruppe mehr auftauen.", "Tempo.", "1."))


# ===========================================================================
# Implementierungen
# ===========================================================================

class EarlyStopping(Callback):
    def __init__(self, a: EarlyStoppingConfig):
        self.a, self.best, self.bad, self.best_state = a, None, 0, None

    def on_eval_end(self, trainer, metrics):
        if self.a.monitor not in metrics:
            return
        value = float(metrics[self.a.monitor])
        if _better(value, self.best, self.a.mode, self.a.min_delta):
            self.best, self.bad = value, 0
            if self.a.restore_best:
                self.best_state = copy.deepcopy(trainer.model.state_dict())
        else:
            self.bad += 1
            if self.bad > self.a.patience:
                trainer.log(f"Early Stopping: {self.a.monitor} seit {self.bad} Evaluierungen nicht verbessert.")
                trainer.should_stop = True

    def on_train_end(self, trainer):
        if self.a.restore_best and self.best_state is not None:
            trainer.model.load_state_dict(self.best_state)
            trainer.log(f"Beste Gewichte wiederhergestellt ({self.a.monitor} = {self.best:.5g}).")


class Checkpoint(Callback):
    def __init__(self, a: CheckpointConfig):
        self.a, self.best, self.saved = a, None, []

    def on_eval_end(self, trainer, metrics):
        value = metrics.get(self.a.monitor)
        if value is None:
            return
        improved = _better(float(value), self.best, self.a.mode, 0.0)
        if improved:
            self.best = float(value)
            trainer.save_checkpoint(Path(self.a.directory) / "beste.pt")
        elif not self.a.save_best_only:
            trainer.save_checkpoint(Path(self.a.directory) / "letzte.pt")

    def on_step_end(self, trainer, metrics):
        n = self.a.every_n_steps
        if n and trainer.global_step % n == 0:
            path = Path(self.a.directory) / f"schritt_{trainer.global_step}.pt"
            trainer.save_checkpoint(path)
            self.saved.append(path)
            while len(self.saved) > self.a.keep_last:
                old = self.saved.pop(0)
                old.unlink(missing_ok=True)


class CSVLogger(Callback):
    def __init__(self, a: LoggerConfig):
        self.path = Path(a.path)
        self.fields: list[str] | None = None

    def _write(self, row: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.fields is None:
            self.fields = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, self.fields, extrasaction="ignore").writeheader()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, self.fields, extrasaction="ignore").writerow(row)

    def on_step_end(self, trainer, metrics):
        if trainer.global_step % trainer.cfg.log_every_steps == 0:
            self._write({"step": trainer.global_step, **metrics})

    def on_eval_end(self, trainer, metrics):
        self._write({"step": trainer.global_step, **metrics})


class JSONLLogger(Callback):
    def __init__(self, a: LoggerConfig):
        self.path = Path(a.path if a.path != "training_log.csv" else "training_log.jsonl")

    def _write(self, row: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_step_end(self, trainer, metrics):
        if trainer.global_step % trainer.cfg.log_every_steps == 0:
            self._write({"step": trainer.global_step, "phase": "train", **metrics})

    def on_eval_end(self, trainer, metrics):
        self._write({"step": trainer.global_step, "phase": "eval", **metrics})


class GradientMonitor(Callback):
    def __init__(self, a: GradientMonitorConfig):
        self.a = a
        self.history: list[dict[str, float]] = []

    def on_before_step(self, trainer, key):
        if trainer.global_step % self.a.every_n_steps:
            return
        norms = {n: float(p.grad.norm()) for n, p in trainer.model.named_parameters() if p.grad is not None}
        self.history.append(norms)
        top = sorted(norms.items(), key=lambda kv: -kv[1])[: self.a.top_k]
        trainer.log("Größte Gradienten: " + ", ".join(f"{n}={v:.3g}" for n, v in top))


class TimeLimit(Callback):
    def __init__(self, a: TimeLimitConfig):
        self.a, self.start = a, 0.0

    def on_train_start(self, trainer):
        self.start = time.time()

    def on_step_end(self, trainer, metrics):
        if (time.time() - self.start) / 60 > self.a.max_minutes:
            trainer.log("Zeitlimit erreicht – Training wird beendet.")
            trainer.should_stop = True


class GradualUnfreeze(Callback):
    def __init__(self, a: GradualUnfreezeConfig):
        self.a = a

    def _apply(self, trainer, n_open: int):
        open_patterns = [re.compile(p) for p in self.a.patterns[:n_open]]
        all_patterns = [re.compile(p) for p in self.a.patterns]
        for name, p in trainer.model.named_parameters():
            if any(r.search(name) for r in all_patterns):
                p.requires_grad_(any(r.search(name) for r in open_patterns))

    def on_train_start(self, trainer):
        self._apply(trainer, 1)

    def on_epoch_end(self, trainer, epoch, metrics):
        n = 1 + (epoch + 1) // self.a.every_epochs
        self._apply(trainer, n)


R = CALLBACKS.register
R("early_stopping", lambda a: EarlyStopping(a), config=EarlyStoppingConfig, tooltip=T(
    "Training beenden, wenn sich die Validierung nicht mehr verbessert.", "Gegen Überanpassung, spart Zeit.",
    "Fast jedes überwachte Training."))
R("checkpoint", lambda a: Checkpoint(a), config=CheckpointConfig, tooltip=T(
    "Modell regelmäßig bzw. bei Verbesserung speichern.", "Fortsetzen nach Abbrüchen, bestes Modell behalten.",
    "Längere Trainings."))
R("csv_logger", lambda a: CSVLogger(a), config=LoggerConfig, tooltip=T(
    "Metriken als CSV protokollieren.", "Auswertung in Tabellenkalkulation/Pandas.", "Analyse."))
R("jsonl_logger", lambda a: JSONLLogger(a), config=LoggerConfig, tooltip=T(
    "Metriken als JSON-Zeilen protokollieren.", "Maschinenlesbar, robust.", "Analyse, Dashboards."))
R("gradient_monitor", lambda a: GradientMonitor(a), config=GradientMonitorConfig, level=Level.ADVANCED, tooltip=T(
    "Gradienten-Normen je Parameter überwachen.", "Findet explodierende/verschwindende Gradienten.", "Debugging."))
R("time_limit", lambda a: TimeLimit(a), config=TimeLimitConfig, tooltip=T(
    "Training nach fester Zeit beenden.", "Zeitbudgets einhalten.", "Cluster, Experimente."))
R("gradual_unfreeze", lambda a: GradualUnfreeze(a), config=GradualUnfreezeConfig, level=Level.EXPERIMENTAL, tooltip=T(
    "Schichten schrittweise auftauen.", "Schonendes Feintuning.", "Transfer-Learning (ULMFiT)."))
