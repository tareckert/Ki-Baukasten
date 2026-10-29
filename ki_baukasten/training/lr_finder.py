"""
ki_baukasten.training.lr_finder
===============================

**Lernraten-Finder** (LR-Range-Test, Smith 2017 / fast.ai): Die Lernrate wird über wenige
Schritte exponentiell von sehr klein nach sehr groß erhöht; aus der Verlustkurve ergibt sich
eine gute Start-Lernrate (dort, wo der Verlust am stärksten fällt).

Der Test läuft auf einer **Kopie** des Modells – das Projekt bleibt unverändert.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LRFinderResult:
    lrs: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    suggestion: float | None = None
    min_loss_lr: float | None = None

    def __str__(self) -> str:
        if self.suggestion is None:
            return "LR-Finder: keine Empfehlung (Verlust fiel nicht)."
        return (f"LR-Finder: empfohlene Lernrate ≈ {self.suggestion:.3g} (steilster Abfall), "
                f"kleinster Verlust bei {self.min_loss_lr:.3g} – übernehmen mit "
                f"projekt.set('training.optimizer.lr', {self.suggestion:.3g})")


def find_lr(project: Any, min_lr: float = 1e-7, max_lr: float = 1.0, steps: int = 100,
            smooth: float = 0.9, diverge: float = 4.0) -> LRFinderResult:
    """LR-Range-Test für ein ``Project`` (Daten aus dem Projekt)."""
    from .trainer import Trainer

    model = copy.deepcopy(project._ensure_model())
    t = project.config.training.copy()
    t.max_steps = steps
    t.verbose = False
    t.callbacks = []
    t.scheduler = "constant"
    t.set("optimizer.lr", min_lr)
    t.gradient_accumulation_steps = 1
    t.logging.backends = []
    trainer = Trainer(model, t, project.config.memory)
    factor = (max_lr / min_lr) ** (1.0 / max(1, steps - 1))
    result = LRFinderResult()
    avg, best = 0.0, math.inf
    data = project.data().train
    loader = trainer._loaders(data, None)[0]
    it = iter(loader)
    from .data_utils import to_device

    for i in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        lr = min_lr * factor ** i
        for opt in trainer.optimizers.values():
            for g in opt.param_groups:
                g["lr"] = lr
        metrics = trainer.paradigm.training_step(trainer, to_device(batch, trainer.device))
        loss = float(metrics.get("loss", float("nan")))
        if not math.isfinite(loss):
            break
        avg = smooth * avg + (1 - smooth) * loss          # gleitender Mittelwert mit Bias-Korrektur
        corrected = avg / (1 - smooth ** (i + 1))
        result.lrs.append(lr)
        result.losses.append(corrected)
        best = min(best, corrected)
        if corrected > diverge * best and i > 5:
            break
    if len(result.losses) >= 5:
        losses, lrs = result.losses, result.lrs
        start = max(1, len(losses) // 10)                 # verrauschten Anfang überspringen
        k = min(range(start, len(losses)), key=losses.__getitem__)
        result.min_loss_lr = lrs[k]
        diverged = next((j for j in range(k, len(losses)) if losses[j] > 1.5 * losses[k]), None)
        if losses[start] - losses[k] < 0.02 * abs(losses[start]):
            # kein klares Tal (flache, verrauschte Kurve): Faustregel "Lernrate, bei der es kippt, ÷ 10"
            result.suggestion = lrs[diverged] / 10 if diverged is not None else lrs[-1] / 10
        else:
            logs = [math.log10(x) for x in lrs]
            grads = {j: (losses[j + 1] - losses[j - 1]) / (logs[j + 1] - logs[j - 1]) for j in range(start, k)}
            result.suggestion = lrs[min(grads, key=grads.__getitem__)] if grads else result.min_loss_lr / 10
    return result
