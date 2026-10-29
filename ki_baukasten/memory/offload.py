"""
ki_baukasten.memory.offload
===========================

Layer-Swapping: Modellblöcke liegen im CPU-RAM oder auf der Festplatte und
werden nur für ihre eigene Berechnung auf das Rechengerät geholt.

Ablauf pro Block (Forward)
--------------------------
::

    Pre-Hook:  Gewichte laden (CPU → GPU  bzw.  Festplatte → Gerät)
    Forward:   ganz normal rechnen
    Post-Hook: Gewichte wieder freigeben (sofern nicht 'keep_resident')

Grenzen (ehrlich dokumentiert)
------------------------------
* Gedacht für **Inferenz** und **eingefrorene** Blöcke. Für das **Training**
  ausgelagerter Blöcke gibt es den Modus ``stream`` (``memory.streaming``).
* Der Festplattenmodus nutzt ``torch.load(mmap=True)`` – das Betriebssystem
  lädt nur, was gebraucht wird.

Aufteilen auf zwei Geräte (``mode="split"``)
--------------------------------------------
``SplitManager`` legt die gewählten Blöcke dauerhaft auf ``split_device`` (z. B. die CPU)
und rechnet sie dort; Eingaben werden vor dem Block dorthin und Ausgaben danach zurück
verschoben. Das ist trainierbar (``Tensor.to`` ist differenzierbar) und entspricht dem
``device_map`` großer Modell-Bibliotheken: langsamer als alles auf der GPU, aber es läuft.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from torch.utils._pytree import tree_map

from ..core import ConfigError, select_indices
from .config import OffloadConfig


def select_offload_blocks(model: nn.Module, cfg: OffloadConfig) -> list[tuple[str, nn.Module]]:
    """Auszulagernde Blöcke nach ``targets`` (``auto`` + ``layers`` oder Regex auf Modulnamen)."""
    named = dict(model.named_modules())
    if cfg.targets == "auto":
        blocks = model.checkpointable_blocks() if hasattr(model, "checkpointable_blocks") else []
        if not blocks:
            raise ConfigError("Offload 'auto': Modell meldet keine Blöcke – bitte 'targets' als Regex angeben.",
                              "memory.offload.targets")
        chosen = {id(blocks[i]) for i in select_indices(cfg.layers, len(blocks))}
        return [(n, m) for n, m in named.items() if id(m) in chosen]
    rx = re.compile(cfg.targets)
    result: list[tuple[str, nn.Module]] = []
    for name, module in named.items():
        if name and rx.fullmatch(name) and not any(name.startswith(c + ".") for c, _ in result):
            result.append((name, module))
    if not result:
        raise ConfigError(f"Offload: kein Modul passt auf '{cfg.targets}'.", "memory.offload.targets")
    return result


class OffloadManager:
    """Hängt Swap-Hooks an die gewählten Blöcke eines Modells."""

    def __init__(self, model: nn.Module, cfg: OffloadConfig, device: torch.device):
        self.model, self.cfg, self.device = model, cfg, device
        self.blocks: list[tuple[str, nn.Module]] = []
        self.handles: list = []
        self.resident: "OrderedDict[int, nn.Module]" = OrderedDict()
        self.storage = torch.device("cpu")

    # ------------------------------------------------------------ Auswahl
    def _select(self) -> list[tuple[str, nn.Module]]:
        return select_offload_blocks(self.model, self.cfg)

    # ------------------------------------------------------------ Einrichten
    def attach(self) -> nn.Module:
        self.blocks = self._select()
        block_ids = {id(m) for _, m in self.blocks}
        self._move_rest(self.model, block_ids)
        for idx, (name, block) in enumerate(self.blocks):
            if self.cfg.mode == "disk":
                self._to_disk(idx, name, block)
            else:
                block.to(self.storage)
                if self.cfg.pin_memory and self.device.type == "cuda":
                    for p in block.parameters():
                        p.data = p.data.pin_memory()
            self.handles.append(block.register_forward_pre_hook(self._make_pre(idx, block)))
            self.handles.append(block.register_forward_hook(self._make_post(idx, block)))
        return self.model

    def _move_rest(self, module: nn.Module, block_ids: set[int]) -> None:
        """Alles außer den Blöcken auf das Rechengerät (ohne das ganze Modell dorthin zu laden)."""
        for p in module.parameters(recurse=False):
            p.data = p.data.to(self.device)
        for name, b in module.named_buffers(recurse=False):
            module._buffers[name] = b.to(self.device)
        for child in module.children():
            if id(child) not in block_ids:
                self._move_rest(child, block_ids)

    def _path(self, idx: int) -> Path:
        return Path(self.cfg.directory) / f"block_{idx:04d}.pt"

    def _to_disk(self, idx: int, name: str, block: nn.Module) -> None:
        path = self._path(idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in block.state_dict().items()}, path)
        for p in block.parameters():
            p.data = torch.empty(0, dtype=p.dtype)
        for bname, buf in list(block.named_buffers()):
            owner, attr = _owner(block, bname)
            owner._buffers[attr] = torch.empty(0, dtype=buf.dtype)

    # ------------------------------------------------------------ Hooks
    def _load(self, idx: int, block: nn.Module) -> None:
        if self.cfg.mode == "disk":
            state = torch.load(self._path(idx), map_location="cpu", mmap=True, weights_only=True)
            for k, v in state.items():
                owner, attr = _owner(block, k)
                if attr in owner._parameters and owner._parameters[attr] is not None:
                    owner._parameters[attr].data = v.to(self.device)
                elif attr in owner._buffers:
                    owner._buffers[attr] = v.to(self.device)
        else:
            block.to(self.device, non_blocking=True)

    def _unload(self, block: nn.Module) -> None:
        if self.cfg.mode == "disk":
            for p in block.parameters():
                p.data = torch.empty(0, dtype=p.dtype)
            for bname, buf in list(block.named_buffers()):
                owner, attr = _owner(block, bname)
                owner._buffers[attr] = torch.empty(0, dtype=buf.dtype)
        else:
            block.to(self.storage)

    def _make_pre(self, idx: int, block: nn.Module):
        def hook(module, args):
            if idx not in self.resident:
                self._load(idx, block)
            self.resident[idx] = block
            self.resident.move_to_end(idx)
        return hook

    def _make_post(self, idx: int, block: nn.Module):
        def hook(module, args, output):
            while len(self.resident) > self.cfg.keep_resident:
                old_idx, old = self.resident.popitem(last=False)
                self._unload(old)
            return output
        return hook

    def trainable_offloaded(self) -> list[str]:
        """Namen ausgelagerter Blöcke mit trainierbaren Parametern (für Warnungen)."""
        return [n for n, b in self.blocks if any(p.requires_grad for p in b.parameters())]

    def detach(self) -> None:
        """Hooks entfernen und alle Gewichte zurück auf das Gerät laden."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for idx, (_, block) in enumerate(self.blocks):
            self._load(idx, block)
        self.resident.clear()


def _owner(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


class SplitManager:
    """Gewählte Blöcke auf einem zweiten Gerät rechnen (``offload.mode = "split"``)."""

    def __init__(self, model: nn.Module, cfg: OffloadConfig, device: torch.device):
        self.model, self.cfg, self.device = model, cfg, device
        self.split_device = torch.device(cfg.split_device)
        self.blocks: list[tuple[str, nn.Module]] = []
        self.handles: list = []

    def attach(self) -> nn.Module:
        self.blocks = select_offload_blocks(self.model, self.cfg)
        ids = {id(m) for _, m in self.blocks}
        OffloadManager(self.model, self.cfg, self.device)._move_rest(self.model, ids)
        for _, block in self.blocks:
            block.to(self.split_device)
            self.handles.append(block.register_forward_pre_hook(self._to(self.split_device), with_kwargs=True))
            self.handles.append(block.register_forward_hook(self._back()))
        return self.model

    @staticmethod
    def _move(obj, device):
        return tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, obj)

    def _to(self, device):
        def hook(module, args, kwargs):
            return self._move(args, device), self._move(kwargs, device)
        return hook

    def _back(self):
        def hook(module, args, output):
            return self._move(output, self.device)
        return hook

    def placement(self) -> dict[str, str]:
        """Welcher Block auf welchem Gerät liegt."""
        return {name: str(self.split_device) for name, _ in self.blocks}

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.model.to(self.device)
