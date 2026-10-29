"""
ki_baukasten.training.trainer
=============================

Der allgemeine ``Trainer`` – führt jedes Paradigma mit allen Einstellungen aus.

Was der Trainer übernimmt
-------------------------
* Geräte, Präzision, Checkpointing, Offloading (über ``MemoryManager``)
* Parametergruppen: Weight-Decay-Ausnahmen, schichtweise Lernraten,
  Tensor-Chirurgie-Overrides; **mehrere Optimierer** (z. B. GAN)
* Gradientenakkumulation, Loss-Scaling, Clipping (Norm/Wert/AGC),
  Gradientenrauschen, SAM, NaN-Behandlung
* Regularisierung im Verlust: L1, L2, Orthogonalität, Flooding;
  Datenaugmentation: Mixup, Eingaberauschen
* Lernraten-Zeitpläne (pro Schritt), EMA der Gewichte
* Evaluierung, Logging, Callbacks, Curriculum, Checkpoints (speichern/fortsetzen)
* Speichermangel (``memory.oom_strategy = "batch_teilen"``): Batch automatisch teilen und
  per Akkumulation weiterrechnen – das Training bricht nicht ab, das Ergebnis bleibt gleich

Ablauf eines Optimierungsschritts (``optimize``)
------------------------------------------------
::

    Verlust (+ Regularisierung) → /Akkumulation → backward (ggf. skaliert)
      └─ alle n Mikro-Batches: unscale → NaN-Prüfung → Gradientenrauschen → Clipping
         → Callbacks → (SAM) → optimizer.step → zero_grad → Zeitplan → Tensor-Constraints → EMA
"""

from __future__ import annotations

import contextlib
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import nn

from ..core import ConfigError, TrainingError, format_number
from ..memory import MemoryConfig, MemoryManager
from ..memory.execution import batch_len, free_memory, is_oom, split_batch
from .config import TrainingConfig
from .data_utils import (CurriculumSampler, as_dataset, difficulty_of, make_loader, split_dataset, to_device)
from .losses import build_losses


def set_seed(seed: int | None, deterministic: bool = False) -> None:
    """Setzt alle Zufallsquellen (Python, NumPy, PyTorch)."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


class EMA:
    """Exponentiell gleitender Mittelwert der Gewichte (fp32-Schattenkopie)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().float().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            s = self.shadow[n]
            s.mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    @contextlib.contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Kontext: Modell nutzt vorübergehend die EMA-Gewichte."""
        backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(self.shadow[n].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(backup[n])

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(self.shadow[n].to(p.dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay, self.shadow = state["decay"], state["shadow"]


_LAYER_RX = re.compile(r"(?:^|\.)(?:blocks|layers|encoder_blocks|stages)\.(\d+)\.")


class Trainer:
    """Allgemeiner Trainer (siehe Modul-Dokumentation).

    Args:
        model:   das zu trainierende Modell.
        config:  ``TrainingConfig``.
        memory:  ``MemoryConfig`` (Gerät, Präzision, Checkpointing ...).
        surgeon: optionaler ``TensorSurgeon`` (Gradienten-Hooks, Overrides, Constraints).
        teacher: optionales Lehrer-Modell für das Paradigma ``distillation`` (statt aus der Konfiguration).
        callbacks: zusätzliche Callback-Objekte (zur Laufzeit, z. B. Pruning-Masken halten).

    ``trainer.penalties`` ist eine Liste von Funktionen ``() → Tensor``, deren Werte zum
    Verlust addiert werden (z. B. EWC beim Weiterlernen).
    """

    def __init__(self, model: nn.Module, config: TrainingConfig | None = None, memory: MemoryConfig | None = None,
                 surgeon: Any = None, teacher: nn.Module | None = None, callbacks: list | None = None):
        self.cfg = (config or TrainingConfig()).validate()
        set_seed(self.cfg.seed, self.cfg.deterministic)
        self.mem = MemoryManager(memory)
        self.device = self.mem.device
        self.model = self.mem.prepare_model(model, training=True)
        self.raw_model = model
        self.surgeon = surgeon
        if surgeon is not None:
            surgeon.install_hooks()
        self.teacher_model = teacher
        self.penalties: list[Callable[[], torch.Tensor]] = []
        self.paradigm = self.cfg.paradigm.build()
        self.losses = build_losses(self.cfg.losses)
        self.scaler = self.mem.grad_scaler()
        if self.cfg.regularization.sam_rho > 0 and self.scaler is not None:
            raise ConfigError("SAM ist nicht mit fp16-Loss-Scaling kombinierbar – nutze bf16 oder fp32.")
        self.paradigm.setup(self)
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.group_params: dict[str, list[nn.Parameter]] = {}
        self._build_optimizers()
        self.schedulers: dict[str, Any] = {}
        self.ema = EMA(self.model, self.cfg.ema.decay) if self.cfg.ema.enabled else None
        self.callbacks = [spec.build() for spec in self.cfg.callbacks] + list(callbacks or [])
        self.global_step = 0
        self.epoch = 0
        self.total_steps = 0
        self.history: list[dict[str, Any]] = []
        self.should_stop = False
        self._micro: dict[str, int] = defaultdict(int)
        self._pending: set[str] = set()
        self._split = 1                          # >1 nach Speichermangel: Batch wird in so viele Stücke geteilt

    @property
    def accumulation(self) -> int:
        """Effektive Akkumulationsschritte (Einstellung × automatische Batch-Teilung)."""
        return self.cfg.gradient_accumulation_steps * self._split

    # ===================================================================== Aufbau
    def log(self, text: str) -> None:
        if self.cfg.verbose:
            print(text, flush=True)

    def _layer_scale(self, name: str, n_layers: int) -> float:
        d = self.cfg.layerwise_lr_decay
        if d is None or n_layers == 0:
            return 1.0
        m = _LAYER_RX.search(name)
        if m:
            return d ** (n_layers - int(m.group(1)))
        if any(k in name for k in ("embed", "pos", "prefix", "stem")):
            return d ** (n_layers + 1)
        return 1.0

    def _param_groups(self, named: list[tuple[str, nn.Parameter]], spec: Any) -> list[dict]:
        base_lr = float(getattr(spec.args, "lr", 1e-3))
        base_wd = float(getattr(spec.args, "weight_decay", 0.0))
        patterns = [p.lower() for p in self.cfg.no_decay_patterns]
        indices = [int(m.group(1)) for n, _ in named if (m := _LAYER_RX.search(n))]
        n_layers = (max(indices) + 1) if indices else 0
        groups: dict[tuple[float, float], dict] = {}
        seen: set[int] = set()
        for name, p in named:
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            wd = 0.0 if (p.dim() < 2 or any(k in name.lower() for k in patterns)) else base_wd
            scale = self._layer_scale(name, n_layers)
            if self.surgeon is not None:
                o = self.surgeon.overrides_for(p)
                scale *= o.get("lr_scale", 1.0)
                wd = o.get("weight_decay", wd)
            key = (round(scale, 12), wd)
            g = groups.setdefault(key, {"params": [], "names": [], "lr": base_lr * scale, "weight_decay": wd})
            g["params"].append(p)
            g["names"].append(name)
        return list(groups.values())

    def _build_optimizers(self) -> None:
        overrides = self.cfg.optimizer_overrides or {}
        for key, named in self.paradigm.optimization_groups(self.model).items():
            spec = self.cfg.optimizer.copy()
            if key in overrides:
                spec.merge(overrides[key])
            groups = self._param_groups(named, spec)
            if not groups:
                continue
            self.optimizers[key] = spec.build(groups)
            self.group_params[key] = [p for g in groups for p in g["params"]]
        unknown = set(overrides) - set(self.optimizers)
        if unknown:
            raise ConfigError(f"optimizer_overrides für unbekannte Gruppen {sorted(unknown)}; "
                              f"vorhanden: {sorted(self.optimizers)}", "training.optimizer_overrides")
        if not self.optimizers:
            raise TrainingError("Keine trainierbaren Parameter gefunden (alles eingefroren?).")

    def _build_schedulers(self, total: int) -> None:
        self.schedulers = {k: self.cfg.scheduler.build(opt, total) for k, opt in self.optimizers.items()}

    # ===================================================================== Hilfen für Paradigmen
    def autocast(self) -> contextlib.AbstractContextManager:
        """Kontext für Forward-Berechnungen (gemischte Präzision, Aktivierungs-Offload)."""
        return self.mem.autocast()

    def model_aux_loss(self) -> torch.Tensor | None:
        """Hilfsverluste des Modells (MoE, VQ, Sparsity …) einsammeln."""
        target = getattr(self.model, "_orig_mod", self.model)
        return target.collect_aux_losses() if hasattr(target, "collect_aux_losses") else None

    def augment(self, inputs: Any, targets: Any) -> tuple[Any, Any]:
        """Eingaberauschen und Mixup (nur für Gleitkomma-Tensoren)."""
        r = self.cfg.regularization
        if isinstance(inputs, torch.Tensor) and inputs.is_floating_point():
            if r.input_noise_std > 0:
                inputs = inputs + torch.randn_like(inputs) * r.input_noise_std
            if r.mixup_alpha > 0 and isinstance(targets, torch.Tensor) and inputs.shape[0] > 1:
                lam = float(torch.distributions.Beta(r.mixup_alpha, r.mixup_alpha).sample())
                perm = torch.randperm(inputs.shape[0], device=inputs.device)
                inputs = lam * inputs + (1 - lam) * inputs[perm]
                if not targets.is_floating_point():
                    n_cls = self._num_classes(targets)
                    onehot = torch.nn.functional.one_hot(targets.long(), n_cls).float()
                    targets = lam * onehot + (1 - lam) * onehot[perm]
                else:
                    targets = lam * targets + (1 - lam) * targets[perm]
        return inputs, targets

    def _num_classes(self, targets: torch.Tensor) -> int:
        cfg = getattr(getattr(self.model, "_orig_mod", self.model), "config", None)
        for attr in ("output_dim", "num_classes"):
            v = getattr(cfg, attr, None) if cfg is not None else None
            if isinstance(v, int) and v > 1:
                return v
        return int(targets.max()) + 1

    def _regularization_loss(self, params: list[nn.Parameter]) -> torch.Tensor | None:
        r = self.cfg.regularization
        if not (r.l1 or r.l2 or r.orthogonality):
            return None
        total = torch.zeros((), device=self.device)
        for p in params:
            if p.dim() < 2:
                continue
            if r.l1:
                total = total + r.l1 * p.abs().sum()
            if r.l2:
                total = total + r.l2 * p.square().sum()
            if r.orthogonality:
                w = p.reshape(p.shape[0], -1)
                gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
                total = total + r.orthogonality * (gram - torch.eye(gram.shape[0], device=gram.device)).square().sum()
        return total

    # ===================================================================== Optimierung
    def optimize(self, loss: torch.Tensor, key: str = "main", closure: Callable[[], torch.Tensor] | None = None) -> bool:
        """Backward + (bei vollem Akkumulationszyklus) Optimierungsschritt. ``True`` = Schritt erfolgt."""
        if key not in self.optimizers:
            raise TrainingError(f"Unbekannte Optimierungsgruppe '{key}' (vorhanden: {sorted(self.optimizers)}).")
        r = self.cfg.regularization
        if r.flooding_level is not None:
            loss = (loss - r.flooding_level).abs() + r.flooding_level
        reg = self._regularization_loss(self.group_params[key])
        if reg is not None:
            loss = loss + reg
        if self.penalties and key == self.paradigm.primary_key:
            for fn in self.penalties:
                loss = loss + fn()
        if not torch.isfinite(loss.detach()):
            return self._handle_nan(key, "Verlust")
        scaled = loss / self.accumulation
        (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
        self._micro[key] += 1
        self._pending.add(key)
        if self._micro[key] % self.accumulation:
            return False
        return self._step(key, closure)

    def _handle_nan(self, key: str, what: str) -> bool:
        policy = self.cfg.nan_policy
        if policy == "raise":
            raise TrainingError(f"{what} ist NaN/Inf (Schritt {self.global_step}). Tipp: Lernrate senken, "
                                f"Clipping aktivieren, bf16 statt fp16.")
        if policy == "skip":
            self.optimizers[key].zero_grad(set_to_none=True)
            self._micro[key] = 0
            self._pending.discard(key)
            self.log(f"⚠ {what} NaN/Inf – Schritt übersprungen (Gruppe '{key}').")
        return False

    def _clip(self, params: list[nn.Parameter]) -> float | None:
        c = self.cfg.gradient_clipping
        grads = [p for p in params if p.grad is not None]
        if not grads or c.mode == "none":
            return None
        if c.mode == "norm":
            return float(torch.nn.utils.clip_grad_norm_(grads, c.max_norm, norm_type=c.norm_type))
        if c.mode == "value":
            torch.nn.utils.clip_grad_value_(grads, c.clip_value)
            return None
        for p in grads:  # AGC: ‖g‖ ≤ ratio·‖w‖ je Ausgabeeinheit
            w = p.detach().reshape(p.shape[0], -1) if p.dim() > 1 else p.detach().reshape(1, -1)
            g = p.grad.reshape(w.shape)
            wn = w.norm(dim=1, keepdim=True).clamp_min(1e-3)
            gn = g.norm(dim=1, keepdim=True).clamp_min(1e-6)
            g.mul_(torch.clamp(c.agc_ratio * wn / gn, max=1.0))
        return None

    def _step(self, key: str, closure: Callable[[], torch.Tensor] | None) -> bool:
        opt = self.optimizers[key]
        params = self.group_params[key]
        if self.scaler is not None:
            self.scaler.unscale_(opt)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            if self.scaler is not None and self.cfg.nan_policy != "raise":
                # fp16-Überlauf: GradScaler überspringt den Schritt selbst und senkt die Skala.
                self.scaler.step(opt)
                self.scaler.update()
                opt.zero_grad(set_to_none=True)
                self._pending.discard(key)
                return False
            return self._handle_nan(key, "Gradient")
        r = self.cfg.regularization
        if r.gradient_noise_std > 0:
            std = r.gradient_noise_std / (1 + self.global_step) ** r.gradient_noise_decay
            for p in params:
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * std)
        grad_norm = self._clip(params)
        for cb in self.callbacks:
            cb.on_before_step(self, key)
        if r.sam_rho > 0 and closure is not None:
            self._sam(params, closure, r.sam_rho)
        if self.scaler is not None:
            self.scaler.step(opt)
            self.scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        self._pending.discard(key)
        if key in self.schedulers:
            self.schedulers[key].step()
        if self.surgeon is not None:
            self.surgeon.post_step()
        if key == self.paradigm.primary_key or len(self.optimizers) == 1:
            self.global_step += 1
            self._last_grad_norm = grad_norm
            e = self.cfg.ema
            if self.ema is not None and self.global_step >= e.start_step and self.global_step % e.update_every == 0:
                self.ema.update(self.model)
            self.mem.after_step(self.global_step)
        return True

    def _sam(self, params: list[nn.Parameter], closure: Callable[[], torch.Tensor], rho: float) -> None:
        """Sharpness-Aware Minimization: Gradient an der 'schlechtesten' Nachbarstelle bestimmen."""
        grads = [p.grad for p in params if p.grad is not None]
        norm = torch.norm(torch.stack([g.norm() for g in grads])) + 1e-12
        eps = {}
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                e = p.grad * (rho / norm)
                p.add_(e)
                eps[p] = e
        for p in params:
            p.grad = None
        loss2 = closure()
        loss2.backward()
        with torch.no_grad():
            for p, e in eps.items():
                p.sub_(e)
        self._clip(params)

    def flush(self) -> None:
        """Führt angefangene Akkumulationszyklen aus (z. B. am Epochenende)."""
        for key in list(self._pending):
            if self._micro[key] % self.accumulation:
                self._micro[key] = 0
                self._step(key, None)

    def zero_grad(self, key: str | None = None) -> None:
        for k, opt in self.optimizers.items():
            if key is None or k == key:
                opt.zero_grad(set_to_none=True)

    # ===================================================================== Daten
    def _loaders(self, train: Any, val: Any) -> tuple[Any, Any]:
        c = self.cfg
        if not isinstance(train, torch.utils.data.DataLoader):
            ds = as_dataset(train)
            if val is None and c.validation_split > 0:
                ds, val = split_dataset(ds, c.validation_split, c.seed)
            sampler = None
            if c.curriculum.strategy != "none":
                diffs = [difficulty_of(ds[i], c.curriculum.strategy) for i in range(len(ds))]
                sampler = CurriculumSampler(diffs, c.curriculum.start_fraction, c.curriculum.full_after_fraction,
                                            lambda: self.global_step / max(1, self.total_steps), c.shuffle, c.seed or 0)
            train_loader = make_loader(ds, c.batch_size, c.shuffle, c.drop_last, c.num_workers, c.pin_memory, sampler)
        else:
            train_loader = train
        val_loader = None
        if val is not None:
            val_loader = make_loader(val, c.eval_batch_size or c.batch_size, False, False, c.num_workers, c.pin_memory)
        return train_loader, val_loader

    # ===================================================================== Training
    def fit(self, train_data: Any = None, val_data: Any = None) -> list[dict[str, Any]]:
        """Trainiert gemäß Konfiguration; liefert die Historie (Liste von Metrik-dicts).

        Paradigmen mit eigener Schleife (z. B. ``reinforcement``) brauchen keine Daten.
        """
        if hasattr(self.paradigm, "run"):
            return self.paradigm.run(self)
        if train_data is None:
            raise TrainingError(f"Paradigma '{self.cfg.paradigm.name}' braucht Trainingsdaten.")
        c = self.cfg
        train_loader, val_loader = self._loaders(train_data, val_data)
        steps_per_epoch = max(1, math.ceil(len(train_loader) / c.gradient_accumulation_steps))
        self.total_steps = c.max_steps or c.epochs * steps_per_epoch
        if not self.schedulers:
            self._build_schedulers(self.total_steps)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log(f"▶ Training: Paradigma '{c.paradigm.name}', {format_number(n_params)} trainierbare Parameter, "
                 f"{self.total_steps} Schritte, Gerät {self.device}")
        for cb in self.callbacks:
            cb.on_train_start(self)
        start = time.time()
        running: dict[str, list[float]] = defaultdict(list)
        epoch = self.epoch
        while not self.should_stop and self.global_step < self.total_steps:
            self.model.train()
            for batch in train_loader:
                batch = to_device(batch, self.device, non_blocking=c.pin_memory)
                before = self.global_step
                metrics = self._train_step(batch)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        running[k].append(float(v))
                if self.global_step == before:
                    continue
                self._after_step(running, start, val_loader)
                if self.should_stop or self.global_step >= self.total_steps:
                    break
            self.flush()
            epoch += 1
            self.epoch = epoch
            epoch_metrics: dict[str, Any] = {}
            if val_loader is not None and c.eval_every_steps is None:
                epoch_metrics = self._run_eval(val_loader)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch - 1, epoch_metrics)
            if c.max_steps is None and epoch >= c.epochs:
                break
        for cb in self.callbacks:
            cb.on_train_end(self)
        self.log(f"■ Training beendet nach {self.global_step} Schritten ({time.time() - start:.1f} s).")
        return self.history

    def _train_step(self, batch: Any) -> dict[str, Any]:
        """Ein Trainingsschritt – bei Speichermangel automatisch in kleinere Stücke geteilt."""
        mc = self.mem.cfg
        while True:
            try:
                if self._split == 1:
                    return self.paradigm.training_step(self, batch)
                collected: dict[str, list[float]] = defaultdict(list)
                for part in split_batch(batch, self._split):
                    for k, v in self.paradigm.training_step(self, part).items():
                        if isinstance(v, (int, float)):
                            collected[k].append(float(v))
                return {k: sum(v) / len(v) for k, v in collected.items()}
            except Exception as exc:  # noqa: BLE001 - nur Speicherfehler werden behandelt
                n = batch_len(batch)
                if mc.oom_strategy != "batch_teilen" or not is_oom(exc) or self._split >= min(mc.oom_max_splits, n):
                    raise
                self.zero_grad()
                self._micro.clear()
                self._pending.clear()
                free_memory()
                self._split *= 2
                self.log(f"⚠ Speichermangel – Batch wird ab jetzt in {self._split} Stücke geteilt "
                         f"(effektive Batchgröße bleibt gleich, Akkumulation {self.accumulation}).")

    def _after_step(self, running: dict[str, list[float]], start: float, val_loader: Any) -> None:
        c = self.cfg
        lr = next(iter(self.optimizers.values())).param_groups[0]["lr"]
        last = {k: v[-1] for k, v in running.items() if v}
        last["lr"] = lr
        for cb in self.callbacks:
            cb.on_step_end(self, last)
        if self.global_step % c.log_every_steps == 0 or self.global_step == 1:
            avg = {k: sum(v) / len(v) for k, v in running.items() if v}
            gn = getattr(self, "_last_grad_norm", None)
            parts = [f"{k}={v:.4g}" for k, v in avg.items()]
            if gn is not None:
                parts.append(f"grad_norm={gn:.3g}")
            self.log(f"  Schritt {self.global_step:>6}/{self.total_steps}  " + "  ".join(parts) +
                     f"  lr={lr:.3g}  ({time.time() - start:.0f} s)")
            self.history.append({"step": self.global_step, "epoch": self.epoch, "lr": lr, **avg})
            running.clear()
        if val_loader is not None and c.eval_every_steps and self.global_step % c.eval_every_steps == 0:
            self._run_eval(val_loader)

    def _run_eval(self, loader: Any) -> dict[str, Any]:
        metrics = self.evaluate(loader)
        self.log("  Validierung: " + "  ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
        self.history.append({"step": self.global_step, "epoch": self.epoch, **metrics})
        for sched in self.schedulers.values():
            if hasattr(sched, "step_metric") and "val_loss" in metrics:
                sched.step_metric(metrics["val_loss"])
        for cb in self.callbacks:
            cb.on_eval_end(self, metrics)
        return metrics

    @torch.no_grad()
    def evaluate(self, data: Any) -> dict[str, float]:
        """Evaluierung (EMA-Gewichte, falls aktiviert). Metriken mit Präfix ``val_``."""
        c = self.cfg
        loader = data if isinstance(data, torch.utils.data.DataLoader) else make_loader(
            data, c.eval_batch_size or c.batch_size, False)
        was = self.model.training
        self.model.eval()
        ctx = self.ema.applied(self.model) if (self.ema is not None and c.ema.use_for_eval) else contextlib.nullcontext()
        sums: dict[str, float] = defaultdict(float)
        count = 0
        with ctx:
            for i, batch in enumerate(loader):
                if c.eval_max_batches and i >= c.eval_max_batches:
                    break
                batch = to_device(batch, self.device)
                n = _batch_size(batch)
                with self.autocast():
                    m = self.paradigm.validation_step(self, batch)
                for k, v in m.items():
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        sums[k] += float(v) * n
                count += n
        self.model.train(was)
        return {f"val_{k}": v / max(1, count) for k, v in sums.items()}

    # ===================================================================== Checkpoints
    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = getattr(self.model, "_orig_mod", self.model)
        state = {
            "model": target.state_dict(),
            "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
            "schedulers": {k: s.state_dict() for k, s in self.schedulers.items()},
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "history": self.history,
            "training_config": self.cfg.to_dict(),
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str | Path, model_only: bool = False) -> dict:
        state = torch.load(path, map_location=self.device, weights_only=False)
        target = getattr(self.model, "_orig_mod", self.model)
        target.load_state_dict(state["model"])
        if not model_only:
            for k, o in self.optimizers.items():
                if k in state.get("optimizers", {}):
                    o.load_state_dict(state["optimizers"][k])
            if state.get("schedulers"):
                if not self.schedulers:
                    self._build_schedulers(max(1, self.cfg.max_steps or 1))
                for k, s in self.schedulers.items():
                    if k in state["schedulers"]:
                        s.load_state_dict(state["schedulers"][k])
            if self.scaler is not None and state.get("scaler"):
                self.scaler.load_state_dict(state["scaler"])
            if self.ema is not None and state.get("ema"):
                self.ema.load_state_dict(state["ema"])
            self.global_step = state.get("global_step", 0)
            self.epoch = state.get("epoch", 0)
            self.history = state.get("history", [])
        return state


def _batch_size(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return batch.shape[0] if batch.dim() else 1
    if isinstance(batch, dict):
        for v in batch.values():
            n = _batch_size(v)
            if n:
                return n
    if isinstance(batch, (list, tuple)) and batch:
        return _batch_size(batch[0])
    return 1
