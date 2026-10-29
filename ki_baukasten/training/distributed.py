"""
ki_baukasten.training.distributed
=================================

**Verteiltes Training** auf mehreren GPUs, Rechnern oder CPU-Prozessen (Cluster).

=============  ===================================================================================
Strategie      Was passiert
=============  ===================================================================================
``none``       ein Prozess (Standard)
``ddp``        Data Parallel: jeder Prozess hat das ganze Modell und einen Teil der Daten; Gradienten
               werden gemittelt. Für Paradigmen, die nur ``model(...)`` aufrufen, mit PyTorch-DDP
               (Rechnen und Kommunikation überlappen), sonst per Gradienten-All-Reduce (GAN & Co.).
``zero1``      wie ddp, zusätzlich werden die **Optimierer-Zustände** auf die Prozesse verteilt
               (ZeRO Stufe 1, ``ZeroRedundancyOptimizer``) – Adam-Speicher ÷ Anzahl Prozesse.
``fsdp``       Fully Sharded Data Parallel (ZeRO Stufe 3, ``fully_shard``): Gewichte, Gradienten und
               Optimierer werden verteilt; jeder Block sammelt seine Gewichte nur zum Rechnen ein.
               Für Modelle, die auf kein einzelnes Gerät passen.
=============  ===================================================================================

Start (ein Rechner, 4 GPUs)::

    python -m ki_baukasten verteilt projekt.json --prozesse 4 --strategie ddp

Mehrere Rechner: auf jedem Knoten ``--knoten 2 --knoten-rang <i> --master <adresse:port>``.
Intern wird ``torch.distributed.run`` (torchrun) gestartet; jeder Prozess erkennt Rang und
Anzahl über die Umgebungsvariablen ``RANK``, ``WORLD_SIZE`` und ``LOCAL_RANK``.

Regeln im verteilten Training
-----------------------------
* Jeder Prozess sieht über einen ``DistributedSampler`` nur seinen Teil der Trainingsdaten.
* Ausgaben und Speichern übernimmt nur Prozess 0 (Checkpoints sammeln vorher alle Teile ein).
* Gleiche Seeds auf allen Prozessen; die Startgewichte werden von Prozess 0 verteilt.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..core import BaseConfig, ConfigError, Level, Param, T

#: Paradigmen, die das Modell nur über ``model(...)`` aufrufen (dann sind DDP-/FSDP-Wrapper sicher).
FORWARD_ONLY = {"supervised", "causal_lm", "masked_lm", "seq2seq", "contrastive", "distillation", "preference",
                "dpo", "autoencoder", "diffusion"}


class DistributedConfig(BaseConfig):
    __title__ = "Verteiltes Training (Cluster)"
    __tooltip__ = T(
        "Training auf mehreren GPUs, Rechnern oder CPU-Prozessen.",
        "Mehr Geräte = schneller (Daten parallel) und/oder größere Modelle (Gewichte verteilt).",
        "Große Datensätze, große Modelle, Rechen-Cluster. Start mit 'python -m ki_baukasten verteilt …'.",
    )
    strategy = Param("none", choices={
        "none": "ein Prozess (Standard)",
        "ddp": "Daten parallel, Gradienten mitteln (DistributedDataParallel)",
        "zero1": "wie ddp + Optimierer-Zustände verteilt (ZeRO-1)",
        "fsdp": "Gewichte, Gradienten und Optimierer verteilt (FSDP / ZeRO-3)",
    }, tooltip=T(
        "Verteilungsstrategie.",
        "ddp beschleunigt, zero1 spart Optimierer-Speicher, fsdp verteilt das ganze Modell.",
        "Modell passt auf eine GPU: ddp. Adam-Speicher knapp: zero1. Modell zu groß für eine GPU: fsdp.",
        "Ohne Start über 'verteilt'/torchrun (WORLD_SIZE=1) läuft alles normal in einem Prozess.",
    ))
    backend = Param("auto", choices={"auto": "nccl bei CUDA, sonst gloo", "nccl": "NVIDIA-GPUs", "gloo": "CPU/alle"},
                    level=Level.ADVANCED, tooltip=T(
        "Kommunikations-Backend.", "nccl ist auf NVIDIA-GPUs am schnellsten, gloo läuft überall.", "auto."))
    find_unused_parameters = Param(True, level=Level.EXPERT, tooltip=T(
        "DDP: Parameter ohne Gradient in einem Schritt erlauben.",
        "Nötig, wenn nicht jeder Teil des Modells in jedem Schritt benutzt wird (z. B. MoE, ungenutzte Köpfe).",
        "Aus = minimal schneller, wenn wirklich alle Parameter immer genutzt werden."))
    fsdp_reshard_after_forward = Param(True, level=Level.EXPERT, tooltip=T(
        "FSDP: Gewichte nach dem Forward wieder verteilen (ZeRO-3) statt bis zum Backward zu behalten (ZeRO-2).",
        "An = weniger Speicher, aus = weniger Kommunikation.", "an."))
    fsdp_cpu_offload = Param(False, level=Level.EXPERT, tooltip=T(
        "FSDP: verteilte Gewichte und Optimierer-Zustände im CPU-RAM halten.",
        "Noch größere Modelle, deutlich langsamer.", "Nur wenn es sonst nicht passt."))
    sync_batchnorm = Param(False, level=Level.ADVANCED, tooltip=T(
        "BatchNorm-Statistiken über alle Prozesse berechnen.", "Wichtig bei kleinen Batches pro Gerät.",
        "CNNs mit kleinem Batch pro GPU."))
    timeout_minutes = Param(30.0, min=1.0, level=Level.EXPERT, tooltip=T(
        "Zeitlimit für Kommunikation (Minuten).", "Erkennt hängende Prozesse.", "30."))


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def active(self) -> bool:
        return self.world_size > 1

    @property
    def main(self) -> bool:
        return self.rank == 0


def env_info() -> DistInfo:
    """Rang/Anzahl aus den torchrun-Umgebungsvariablen."""
    return DistInfo(int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)),
                    int(os.environ.get("WORLD_SIZE", 1)))


def is_main_process() -> bool:
    """``True`` außerhalb verteilter Läufe und in Prozess 0."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return env_info().rank == 0


class DistributedContext:
    """Richtet Prozessgruppe, Gerät, Modell-Wrapper und Daten-Sampler für den Trainer ein."""

    def __init__(self, cfg: DistributedConfig, device: torch.device):
        self.cfg = cfg
        self.info = env_info() if cfg.strategy != "none" else DistInfo()
        self.device = device
        self.wrapped = False
        self.allreduce = False
        if not self.info.active:
            return
        import datetime

        import torch.distributed as dist

        if not dist.is_available():
            raise ConfigError("torch.distributed ist in dieser PyTorch-Installation nicht verfügbar.",
                              "training.distributed.strategy")
        if device.type == "cuda":
            self.device = torch.device("cuda", self.info.local_rank)
            torch.cuda.set_device(self.device)
        backend = cfg.backend if cfg.backend != "auto" else ("nccl" if self.device.type == "cuda" else "gloo")
        if not dist.is_initialized():
            dist.init_process_group(backend, timeout=datetime.timedelta(minutes=cfg.timeout_minutes))

    @property
    def active(self) -> bool:
        return self.info.active

    @property
    def main(self) -> bool:
        return self.info.main

    # ------------------------------------------------------------ Modell
    def wrap(self, model: nn.Module, paradigm_name: str, mixed_precision: str = "none") -> nn.Module:
        """Modell für die Strategie vorbereiten (Startgewichte von Prozess 0, DDP-/FSDP-Wrapper)."""
        if not self.active:
            return model
        import torch.distributed as dist

        if self.cfg.sync_batchnorm:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        forward_only = paradigm_name in FORWARD_ONLY
        if self.cfg.strategy == "fsdp":
            if not forward_only:
                raise ConfigError(f"FSDP braucht ein Paradigma, das das Modell nur über model(...) aufruft – "
                                  f"'{paradigm_name}' ruft Teilmodule direkt auf. Tipp: strategy='zero1'.",
                                  "training.distributed.strategy")
            return self._fsdp(model, mixed_precision)
        for t in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(t.data, src=0)
        if forward_only:
            from torch.nn.parallel import DistributedDataParallel

            ids = [self.device.index] if self.device.type == "cuda" else None
            self.wrapped = True
            return _DDP(model, device_ids=ids, find_unused_parameters=self.cfg.find_unused_parameters)
        self.allreduce = True           # GAN & Co.: Gradienten selbst mitteln (wirkt wie DDP)
        return model

    def _fsdp(self, model: nn.Module, mixed_precision: str) -> nn.Module:
        try:
            from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
        except ImportError as exc:  # pragma: no cover - ältere PyTorch-Versionen
            raise ConfigError("FSDP (fully_shard) braucht PyTorch ≥ 2.6.", "training.distributed.strategy") from exc
        kw: dict[str, Any] = {"reshard_after_forward": self.cfg.fsdp_reshard_after_forward}
        if mixed_precision in ("bf16", "fp16"):
            dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
            kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
        if self.cfg.fsdp_cpu_offload:
            kw["offload_policy"] = CPUOffloadPolicy()
        blocks = model.checkpointable_blocks() if hasattr(model, "checkpointable_blocks") else []
        for block in blocks:
            fully_shard(block, **kw)
        fully_shard(model, **kw)
        self.wrapped = True
        return model

    def unwrap(self, model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, _DDP) else model

    def no_sync(self, model: nn.Module, skip: bool):
        """Kontext: bei Gradientenakkumulation erst beim letzten Mikro-Batch synchronisieren."""
        import contextlib

        if skip and isinstance(model, _DDP):
            return model.no_sync()
        return contextlib.nullcontext()

    # ------------------------------------------------------------ Gradienten / Optimierer
    def allreduce_grads(self, params: list[nn.Parameter]) -> None:
        """Gradienten über alle Prozesse mitteln (Strategie ddp/zero1 ohne DDP-Wrapper)."""
        if not (self.active and self.allreduce):
            return
        import torch.distributed as dist

        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat)
        flat /= self.info.world_size
        offset = 0
        for g in grads:
            g.copy_(flat[offset:offset + g.numel()].view_as(g))
            offset += g.numel()

    def wrap_optimizer(self, opt: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """ZeRO-1: Optimierer-Zustände verteilen."""
        if not (self.active and self.cfg.strategy == "zero1"):
            return opt
        from torch.distributed.optim import ZeroRedundancyOptimizer

        import inspect

        accepted = set(inspect.signature(type(opt).__init__).parameters) - {"self", "params"}
        groups = [{k: v for k, v in g.items() if k != "names"} for g in opt.param_groups]
        defaults = {k: v for k, v in opt.defaults.items() if k in accepted}
        return ZeroRedundancyOptimizer(groups, optimizer_class=type(opt), **defaults)

    def consolidate(self, opt: torch.optim.Optimizer) -> None:
        if self.active and type(opt).__name__ == "ZeroRedundancyOptimizer":
            opt.consolidate_state_dict(to=0)

    def model_state(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Vollständiger state_dict (bei FSDP von allen Prozessen eingesammelt)."""
        target = self.unwrap(model)
        if self.active and self.cfg.strategy == "fsdp":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            return get_model_state_dict(target, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        return target.state_dict()

    # ------------------------------------------------------------ Daten
    def sampler(self, dataset: Any, shuffle: bool, seed: int) -> Any:
        if not self.active:
            return None
        from torch.utils.data.distributed import DistributedSampler

        return DistributedSampler(dataset, num_replicas=self.info.world_size, rank=self.info.rank, shuffle=shuffle,
                                  seed=seed)

    def barrier(self) -> None:
        if self.active:
            import torch.distributed as dist

            dist.barrier()


try:  # DDP-Wrapper, der unbekannte Attribute an das Modell weiterreicht (model.config, model.generate …)
    from torch.nn.parallel import DistributedDataParallel as _BaseDDP

    class _DDP(_BaseDDP):
        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
except ImportError:  # pragma: no cover
    class _DDP(nn.Module):  # type: ignore[no-redef]
        pass


def launch_command(source: str, processes: int = 2, strategy: str = "ddp", nodes: int = 1, node_rank: int = 0,
                   master: str | None = None, extra: list[str] | None = None,
                   command: str = "trainieren") -> list[str]:
    """Befehlszeile für ``torch.distributed.run`` (torchrun), die ``ki_baukasten <command>`` startet."""
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={processes}", f"--nnodes={nodes}",
           f"--node_rank={node_rank}"]
    if master:
        host, _, port = master.partition(":")
        cmd += [f"--master_addr={host}", f"--master_port={port or 29500}"]
    elif nodes == 1:
        cmd += ["--standalone"]
    cmd += ["-m", "ki_baukasten", command, source, "--setze", f"training.distributed.strategy={strategy}"]
    return cmd + list(extra or [])
