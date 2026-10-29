"""
ki_baukasten.memory.streaming
=============================

**Trainierbares** Auslagern von Modellblöcken in den CPU-RAM ("Block-Streaming").

Idee (ähnlich ZeRO-Offload / FSDP-CPU-Offload, aber für eine einzelne GPU)
-------------------------------------------------------------------------
::

    RAM (groß, langsam)                      Rechengerät (klein, schnell)
    ───────────────────                      ───────────────────────────
    Gewichte W_i   ─── Kopie vor Forward ──►  W_i' (temporär) ─► Forward von Block i
    Gradienten ∂W_i ◄── Kopie nach Backward ─ ∂W_i'           ◄─ Backward von Block i
    Optimierer-Zustände (Adam m, v) bleiben komplett im RAM; der Optimierer-Schritt
    für diese Gewichte läuft auf der CPU.

Pro Block liegt also immer nur **eine** Kopie seiner Gewichte auf dem Gerät –
und auch nur, solange er gerade rechnet. Damit lassen sich Modelle trainieren,
deren Gewichte + Gradienten + Optimierer-Zustände nicht in den GPU-Speicher
passen würden.

Ablauf im Detail
----------------
* **Forward**: Gewichte kopieren → Block ohne Autograd-Aufzeichnung rechnen →
  nur die *Eingabe* des Blocks merken (wie Gradient-Checkpointing).
* **Backward**: Gewichte erneut kopieren → Block mit identischem
  Zufallszustand (Dropout!) und identischer autocast-Einstellung neu rechnen →
  Gradienten von Eingabe und Gewichten bestimmen → Gewichts-Gradienten in den
  RAM zurückschreiben (Autograd summiert sie dort in ``param.grad``).
* **Puffer** (z. B. BatchNorm-Statistiken) werden nach dem Forward
  zurückgeschrieben; die Neuberechnung im Backward ändert sie nicht noch einmal.

Kosten & Tipps
--------------
* Pro Schritt werden die Block-Gewichte zweimal übertragen (Forward + Backward)
  und jeder Block zweimal gerechnet → typischerweise 1.3–3× langsamer.
* ``pin_memory`` beschleunigt die Übertragung auf CUDA deutlich.
* ``prefetch`` (experimentell, nur CUDA) lädt den nächsten Block auf einem
  eigenen CUDA-Stream, während der aktuelle rechnet.
* Größere Batches amortisieren die Übertragung besser (Rechnen pro Byte steigt).

Eingefrorene Gewichte auf der Festplatte (``storage="disk"``)
------------------------------------------------------------
Passt selbst der RAM nicht, legt ``offload.storage = "disk"`` alle **eingefrorenen**
Gewichte und Puffer der gestreamten Blöcke (z. B. die Basis bei LoRA/QLoRA, eingefrorene
Schichten) als Datei ab und liest sie per Memory-Map nur für die Berechnung des Blocks.
Trainierbare Gewichte, ihre Gradienten und Optimierer-Zustände bleiben im RAM. Das
Betriebssystem lädt nur die gerade benötigten Seiten und kann sie jederzeit wieder freigeben.
``state_dict()`` liefert weiterhin alle Gewichte (aus der Datei gelesen).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.func import functional_call
from torch.utils._pytree import tree_flatten, tree_unflatten


@dataclass
class StreamState:
    """Laufzeitzustand eines gestreamten Blocks (hängt als Attribut am Modul)."""

    device: torch.device
    prefetch: bool = False
    inside: bool = False                       # True während der eigentlichen Berechnung (keine Rekursion)
    next: "StreamState | None" = None          # Nachbar im Forward (für Prefetch)
    prev: "StreamState | None" = None          # Nachbar im Backward
    module: nn.Module | None = None
    _prefetched: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    _event: Any = field(default=None, repr=False)
    transfers: int = 0                         # Statistik: Anzahl Gewichts-Übertragungen
    disk: "DiskWeights | None" = None          # eingefrorene Gewichte auf der Festplatte (storage="disk")

    # ------------------------------------------------------------ Übertragung
    def _source(self) -> dict[str, torch.Tensor]:
        """Alle Gewichte + Puffer des Blocks (im RAM bzw. per Memory-Map von der Festplatte)."""
        m = self.module
        assert m is not None
        src = {n: p.detach() for n, p in m.named_parameters()}
        src.update(dict(m.named_buffers()))
        if self.disk is not None:
            params = dict(m.named_parameters())
            for n in self.disk.names:
                if n in params and params[n].requires_grad:
                    raise RuntimeError(f"'{n}' liegt eingefroren auf der Festplatte (offload.storage='disk'), soll "
                                       "aber trainiert werden – storage='ram' nutzen oder die Schicht eingefroren lassen.")
            src.update(self.disk.tensors())
        return src

    def _copy(self, requires_grad: bool) -> dict[str, torch.Tensor]:
        """Gewichte + Puffer des Blocks als Tensoren auf dem Rechengerät."""
        m = self.module
        assert m is not None
        if self._prefetched is not None:
            tensors = self._prefetched
            if self._event is not None:
                torch.cuda.current_stream(self.device).wait_event(self._event)
                for t in tensors.values():
                    t.record_stream(torch.cuda.current_stream(self.device))
            self._prefetched, self._event = None, None
        else:
            tensors = {n: t.to(self.device, non_blocking=True) for n, t in self._source().items()}
        self.transfers += 1
        out = {}
        params, buffers = dict(m.named_parameters()), dict(m.named_buffers())
        for n, t in tensors.items():
            if n in params:
                t = t.detach()
                if requires_grad and params[n].requires_grad:
                    t.requires_grad_(True)
            elif n in buffers and t.data_ptr() == buffers[n].data_ptr():
                t = t.clone()  # gleiches Gerät: Original-Puffer nicht unkontrolliert verändern
            out[n] = t
        return out

    def start_prefetch(self) -> None:
        """Gewichte asynchron auf einem eigenen CUDA-Stream vorladen (nur CUDA, experimentell)."""
        if not self.prefetch or self.device.type != "cuda" or self._prefetched is not None or self.module is None:
            return
        stream = _copy_stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            tensors = {n: t.to(self.device, non_blocking=True) for n, t in self._source().items()}
            event = torch.cuda.Event()
            event.record(stream)
        self._prefetched, self._event = tensors, event


_STREAMS: dict[str, Any] = {}


def _copy_stream(device: torch.device):
    key = str(device)
    if key not in _STREAMS:
        _STREAMS[key] = torch.cuda.Stream(device)
    return _STREAMS[key]


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda


@contextlib.contextmanager
def _restored_rng(state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device):
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield


def _autocast_state(device: torch.device) -> tuple[bool, torch.dtype | None]:
    kind = device.type if device.type in ("cuda", "cpu", "mps") else "cpu"
    try:
        return torch.is_autocast_enabled(kind), torch.get_autocast_dtype(kind)
    except (AttributeError, TypeError, RuntimeError):  # ältere PyTorch-Versionen
        return False, None


def _run(module: nn.Module, st: StreamState, tensors: dict[str, torch.Tensor], args: tuple, kwargs: dict):
    st.inside = True
    try:
        return functional_call(module, tensors, args, kwargs, strict=False)
    finally:
        st.inside = False


class _StreamFunction(torch.autograd.Function):
    """Forward/Backward eines Blocks mit Gewichten im RAM (siehe Modul-Doku)."""

    @staticmethod
    def forward(ctx, module, st, spec, flat, tensor_idx, holder, n_inputs, *rest):  # type: ignore[override]
        inputs, params = rest[:n_inputs], rest[n_inputs:]
        ctx.module, ctx.st, ctx.spec, ctx.flat, ctx.tensor_idx = module, st, spec, flat, tensor_idx
        ctx.n_inputs, ctx.n_params = n_inputs, len(params)
        ctx.rng = _rng_state(st.device)
        ctx.autocast = _autocast_state(st.device)
        if st.next is not None:
            st.next.start_prefetch()
        tensors = st._copy(requires_grad=False)
        args, kwargs = _rebuild(spec, flat, tensor_idx, inputs)
        with torch.no_grad():
            out = _run(module, st, tensors, args, kwargs)
        # Puffer (BatchNorm-Statistiken …) zurück in den RAM schreiben.
        on_disk = set(st.disk.names) if st.disk is not None else set()
        for name, buf in module.named_buffers():
            if name in tensors and name not in on_disk and tensors[name].data_ptr() != buf.data_ptr():
                buf.copy_(tensors[name])
        out_flat, out_spec = tree_flatten(out)
        if not all(isinstance(t, torch.Tensor) for t in out_flat):
            raise TypeError("Gestreamte Blöcke müssen Tensoren (oder Tupel/Listen/Dicts von Tensoren) liefern.")
        holder.append(out_spec)  # Ausgabestruktur für den Aufrufer
        ctx.save_for_backward(*inputs)
        return tuple(out_flat)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outs):  # type: ignore[override]
        st: StreamState = ctx.st
        module = ctx.module
        params = list(module.parameters())
        if st.prev is not None:
            st.prev.start_prefetch()
        inputs = [t.detach().requires_grad_(t.requires_grad) if t.is_floating_point() or t.is_complex() else t.detach()
                  for t in ctx.saved_tensors]
        tensors = st._copy(requires_grad=True)
        args, kwargs = _rebuild(ctx.spec, ctx.flat, ctx.tensor_idx, inputs)
        enabled, dtype = ctx.autocast
        kind = st.device.type if st.device.type in ("cuda", "cpu", "mps") else "cpu"
        autocast = torch.autocast(kind, dtype=dtype, enabled=True) if enabled and dtype is not None \
            else contextlib.nullcontext()
        with _restored_rng(ctx.rng, st.device), torch.enable_grad(), autocast:
            out = _run(module, st, tensors, args, kwargs)
        outs = tree_flatten(out)[0]
        pairs = [(o, g) for o, g in zip(outs, grad_outs) if g is not None and o.requires_grad]
        names = [n for n, _ in module.named_parameters()]
        wrt = [t for t in inputs if t.requires_grad] + [tensors[n] for n in names if tensors[n].requires_grad]
        grads = torch.autograd.grad([o for o, _ in pairs], wrt, [g for _, g in pairs], allow_unused=True) \
            if pairs and wrt else [None] * len(wrt)
        it = iter(grads)
        input_grads = [next(it) if t.requires_grad else None for t in inputs]
        param_grads = []
        for n, p in zip(names, params):
            g = next(it) if tensors[n].requires_grad else None
            param_grads.append(g.to(p.device, p.dtype) if g is not None else None)
        return (None,) * 7 + tuple(input_grads) + tuple(param_grads)


def _rebuild(spec, flat, tensor_idx, tensors):
    flat = list(flat)
    for i, t in zip(tensor_idx, tensors):
        flat[i] = t
    args, kwargs = tree_unflatten(flat, spec)
    return args, kwargs


# ===========================================================================
# Klassen-Tausch (Parameternamen bleiben erhalten – wie beim Checkpointing)
# ===========================================================================

_STREAM_CLASSES: dict[type, type] = {}


def _streamed_class(cls: type) -> type:
    if cls in _STREAM_CLASSES:
        return _STREAM_CLASSES[cls]

    def forward(self, *args, **kwargs):
        st: StreamState = self._ki_stream
        parent = super(new_cls, self).forward
        if st.inside:
            return parent(*args, **kwargs)
        params = list(self.parameters())
        if not (torch.is_grad_enabled() and any(p.requires_grad for p in params)):
            # Inferenz / eingefroren: einfach einmal kopieren und rechnen.
            if st.next is not None:
                st.next.start_prefetch()
            with torch.no_grad():
                return _run(self, st, st._copy(requires_grad=False), args, kwargs)
        flat, spec = tree_flatten((args, kwargs))
        tensor_idx = [i for i, v in enumerate(flat) if isinstance(v, torch.Tensor)]
        inputs = [flat[i] for i in tensor_idx]
        holder: list = []
        out = _StreamFunction.apply(self, st, spec, flat, tensor_idx, holder, len(inputs), *inputs, *params)
        return tree_unflatten(list(out), holder[0])

    new_cls = type(cls.__name__, (cls,), {"forward": forward, "_ki_streamed": True, "__module__": cls.__module__})
    _STREAM_CLASSES[cls] = new_cls
    return new_cls


class DiskWeights:
    """Eingefrorene Gewichte/Puffer eines Blocks als Datei, gelesen per Memory-Map."""

    _KEEP_IN_RAM = ("running_mean", "running_var", "num_batches_tracked")   # werden im Forward aktualisiert

    def __init__(self, module: nn.Module, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frozen = {n: p for n, p in module.named_parameters() if not p.requires_grad}
        bufs = {n: b for n, b in module.named_buffers() if not n.endswith(self._KEEP_IN_RAM)}
        torch.save({n: t.detach().cpu().contiguous() for n, t in {**frozen, **bufs}.items()}, self.path)
        self.names = list(frozen) + list(bufs)
        self._mm: dict[str, torch.Tensor] | None = None
        for n, p in frozen.items():
            p.data = torch.empty(0, dtype=p.dtype)
        for n, b in bufs.items():
            owner, attr = _owner(module, n)
            owner._buffers[attr] = torch.empty(0, dtype=b.dtype)
        module._register_state_dict_hook(self._state_dict_hook)

    def tensors(self) -> dict[str, torch.Tensor]:
        if self._mm is None:
            self._mm = torch.load(self.path, map_location="cpu", mmap=True, weights_only=True)
        return self._mm

    def _state_dict_hook(self, module: nn.Module, state: dict, prefix: str, local_metadata: Any) -> None:
        for n, t in self.tensors().items():
            if prefix + n in state:
                state[prefix + n] = t

    def restore(self, module: nn.Module, device: torch.device) -> None:
        params = dict(module.named_parameters())
        for n, t in self.tensors().items():
            if n in params:
                params[n].data = t.clone().to(device)
            else:
                owner, attr = _owner(module, n)
                owner._buffers[attr] = t.clone().to(device)


def _owner(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    mod = root
    for part in parts[:-1]:
        mod = getattr(mod, part)
    return mod, parts[-1]


def enable_block_streaming(model: nn.Module, blocks: list[nn.Module], device: torch.device,
                           pin_memory: bool = True, prefetch: bool = False, storage: str = "ram",
                           directory: str | Path = "offload_cache") -> list[StreamState]:
    """Lagert ``blocks`` trainierbar in den RAM aus (Rest des Modells auf ``device``).

    Args:
        storage: ``ram`` (Standard) oder ``disk`` – eingefrorene Gewichte der Blöcke auf die Festplatte.

    Returns:
        Die ``StreamState``-Objekte (u. a. mit Übertragungsstatistik ``transfers``).
    """
    ids = {id(b) for b in blocks}
    _move_rest(model, ids, device)
    states = []
    for i, b in enumerate(blocks):
        if getattr(b, "_ki_checkpointed", False):
            b.__class__ = b.__class__.__mro__[1]  # Streaming rechnet ohnehin neu – Checkpointing entfällt
        b.to("cpu")
        disk = DiskWeights(b, Path(directory) / f"stream_block_{i:04d}.pt") if storage == "disk" else None
        if pin_memory and device.type == "cuda":
            for p in b.parameters():
                if p.numel():
                    p.data = p.data.pin_memory()
        st = StreamState(device=device, prefetch=prefetch, module=b, disk=disk)
        b._ki_stream = st
        if not getattr(b, "_ki_streamed", False):
            b.__class__ = _streamed_class(type(b))
        states.append(st)
    for a, b in zip(states, states[1:]):
        a.next, b.prev = b, a
    return states


def disable_block_streaming(blocks: list[nn.Module], device: torch.device) -> None:
    """Macht ``enable_block_streaming`` rückgängig (Blöcke wieder normal auf ``device``)."""
    for b in blocks:
        if getattr(b, "_ki_streamed", False):
            st = b._ki_stream
            if st.disk is not None:
                st.disk.restore(b, torch.device("cpu"))
            b.__class__ = b.__class__.__mro__[1]
            del b._ki_stream
        b.to(device)


def _move_rest(module: nn.Module, block_ids: set[int], device: torch.device) -> None:
    for p in module.parameters(recurse=False):
        p.data = p.data.to(device)
    for name, b in module.named_buffers(recurse=False):
        module._buffers[name] = b.to(device)
    for child in module.children():
        if id(child) not in block_ids:
            _move_rest(child, block_ids, device)
