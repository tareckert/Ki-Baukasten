"""
ki_baukasten.analysis.export
============================

Export gebauter/trainierter Modelle zur Weiterverwendung außerhalb des Baukastens – **schlank**:
exportiert wird nur, was die Vorhersage wirklich braucht.

==============  ===============================================================
Format          Beschreibung
==============  ===============================================================
``state_dict``  nur Gewichte (``.pt``) – zum Laden in dieselbe Architektur
``export``      ``torch.export`` (``.pt2``): moderner, vollständiger Graph inkl.
                Gewichten; Grundlage für ExecuTorch, AOTInductor, TensorRT
``torchscript`` ``torch.jit.trace`` (``.ts``): ältere, weit verbreitete Variante
``onnx``        ONNX (``.onnx``) für Onnx-Runtime, TensorRT, OpenVINO
                (benötigt das Paket ``onnx``)
==============  ===============================================================

Vorbereitung (``prepare_for_export``, alles auf einer **Kopie** – das Original bleibt unverändert; jeder
Schritt einzeln schaltbar über ``deployment.export``):

=====================  =========================================================================
Schritt                Wirkung
=====================  =========================================================================
Trainings-Hüllen weg   Gradient-Checkpointing, RAM-/SSD-Streaming, Auslagerungs-Hooks, DDP,
                       ``torch.compile`` – nichts davon gehört in ein Einsatz-Modell
EMA übernehmen         geglättete Gewichte statt der letzten Trainingsgewichte (wenn EMA lief)
LoRA einrechnen        Gleitkomma-Schichten exakt; quantisierte Basis (QLoRA) wahlweise exakt
                       behalten, entquantisieren oder neu quantisieren
BatchNorm falten       BatchNorm in die vorangehende Faltung/lineare Schicht einrechnen
                       (eine Operation weniger je Schicht, gleiche Ausgabe)
Ungenutztes entfernen  Parameter und Puffer, die die Vorhersage nie berührt (abgeschaltete
                       Zweige, Trainings-Köpfe, Hilfsgrößen) – gefunden über den Rechengraphen
Präzision              optional float16/bfloat16 für den Einsatz auf GPUs
Prüfen                 Ausgabe der Kopie gegen das Original (sonst wird der Schritt zurückgenommen)
=====================  =========================================================================
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..core import BaseConfig, Level, Param, T

FORMATS = {
    "state_dict": ".pt", "export": ".pt2", "torchscript": ".ts", "onnx": ".onnx",
}

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class ExportConfig(BaseConfig):
    __title__ = "Export (schlank)"
    __tooltip__ = T(
        "Wie exportiert wird: nur das, was die Vorhersage braucht – ohne Trainings-Ballast.",
        "Ein Einsatz-Modell soll so klein und schnell sein wie ein von Hand gebautes – ohne Hüllen, ungenutzte "
        "Gewichte oder Adapter-Umwege.", "Project.export / workflow.export_format / CLI exportieren.")
    merge_lora = Param("auto", choices={
        "auto": "Gleitkomma-Schichten exakt einrechnen, quantisierte Basis exakt behalten",
        "behalten": "wie auto",
        "entquantisieren": "quantisierte Basis + Adapter → Gleitkomma-Schicht (exakt, größer)",
        "neu_quantisieren": "einrechnen und neu quantisieren (am kleinsten, verlustbehaftet)",
        "nein": "Adapter unverändert lassen (zwei Zusatz-Matmuls je Schicht)"}, tooltip=T(
        "Umgang mit LoRA-Adaptern beim Export.", "Eingerechnete Adapter kosten zur Laufzeit nichts.", "auto."),
        on="auto", off="nein")
    use_ema = Param(True, tooltip=T(
        "EMA-Gewichte exportieren, wenn im Training eine EMA mitlief.", "Geglättete Gewichte sind meist besser.",
        "an."))
    fold_batchnorm = Param(True, tooltip=T(
        "BatchNorm in die vorangehende Faltung/lineare Schicht einrechnen.",
        "Eine Operation weniger je Schicht – schneller, gleiche Ausgabe (geprüft).", "an."))
    strip_unused = Param(True, tooltip=T(
        "Parameter und Puffer entfernen, die die Vorhersage nie benutzt.",
        "Abgeschaltete Zweige, Trainings-Köpfe oder Hilfsgrößen sollen keinen Platz kosten.", "an."))
    dtype = Param("wie_modell", choices={"wie_modell": "unverändert", "float32": "32 Bit", "float16": "16 Bit (GPU)",
                                         "bfloat16": "bfloat16 (moderne GPUs/CPUs)"}, tooltip=T(
        "Präzision der exportierten Gewichte.", "Halbe Größe und auf GPUs schneller; minimal ungenauer.",
        "GPU-Einsatz: float16/bfloat16; CPU: wie_modell."))
    dynamic = Param(True, tooltip=T(
        "Variable Batchgröße (und Sequenzlänge) im Export.", "Ein Export für alle Eingabegrößen.", "an."),
        level=Level.ADVANCED)
    verify = Param(True, tooltip=T(
        "Ausgabe des Exports gegen das Original prüfen; ändert ein Schritt das Ergebnis, wird er zurückgenommen.",
        "Sicherheit, dass Verschlanken nichts kaputt macht.", "an."), level=Level.ADVANCED)
    tolerance = Param(1e-4, min=0.0, tooltip=T(
        "Erlaubte Abweichung bei der Prüfung (relativ zur Ausgabegröße).", "Rundungsunterschiede.", "1e-4."),
        level=Level.EXPERT)
    compact_config = Param(True, tooltip=T(
        "Beim Speichern nur geänderte Einstellungen in projekt.json schreiben (Rest = Standard).",
        "Kleine, lesbare Konfiguration ohne tausende Standardwerte.", "an; aus = vollständige Konfiguration."))


@dataclass
class ExportReport:
    """Was die Vorbereitung getan hat (``Project.last_export``)."""

    parameters_before: int = 0
    parameters_after: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    removed: list[str] = field(default_factory=list)
    lora_merged: int = 0
    batchnorm_folded: int = 0
    wrappers_removed: int = 0
    ema: bool = False
    dtype: str = "wie_modell"
    max_diff: float | None = None
    notes: list[str] = field(default_factory=list)
    path: Path | None = None

    def __str__(self) -> str:
        mb = (lambda b: f"{b / 2**20:.2f} MB")
        lines = [f"Export{f' → {self.path}' if self.path else ''}: {self.parameters_after:,} Parameter "
                 f"({mb(self.bytes_after)}; vorher {self.parameters_before:,} / {mb(self.bytes_before)})"
                 .replace(",", ".")]
        done = []
        if self.wrappers_removed:
            done.append(f"{self.wrappers_removed} Trainings-Hüllen entfernt")
        if self.ema:
            done.append("EMA-Gewichte übernommen")
        if self.lora_merged:
            done.append(f"{self.lora_merged} LoRA-Adapter eingerechnet")
        if self.batchnorm_folded:
            done.append(f"{self.batchnorm_folded} BatchNorm gefaltet")
        if self.removed:
            done.append(f"{len(self.removed)} ungenutzte Tensoren entfernt")
        if self.dtype != "wie_modell":
            done.append(f"Präzision {self.dtype}")
        if done:
            lines.append("  " + " · ".join(done))
        if self.max_diff is not None:
            lines.append(f"  geprüft: größte Abweichung zum Original {self.max_diff:.3g}")
        lines += [f"  ℹ {n}" for n in self.notes]
        return "\n".join(lines)


# ===========================================================================
# Hilfen
# ===========================================================================

def _as_call(example: Any) -> tuple[tuple, dict]:
    """Beispiel → (Positionsargumente, Schlüsselwortargumente)."""
    if isinstance(example, tuple):
        return example, {}
    if isinstance(example, dict):
        return (), dict(example)
    return (example,), {}


def _dims(a: Any) -> Any:
    from torch.export import Dim

    if not isinstance(a, torch.Tensor) or a.dim() == 0:
        return None
    dims = {0: Dim.AUTO}
    if a.dim() in (2, 3) and not (a.dim() == 2 and a.is_floating_point()):
        dims[1] = Dim.AUTO   # [B, T] Tokens bzw. [B, T, F] Zeitreihen
    return dims


def _dynamic_shapes(args: tuple, kwargs: dict) -> Any:
    """Batch (und bei Sequenzen die Länge) dynamisch halten – Bilder: nur der Batch."""
    if kwargs:
        return {k: _dims(v) for k, v in kwargs.items()}
    return tuple(_dims(a) for a in args)


def _double_batch(x: Any) -> Any:
    # Mit Batch 1 würde torch.export die Batch-Dimension auf 1 festlegen → Beispiel verdoppeln.
    if isinstance(x, torch.Tensor) and x.dim() > 0 and x.shape[0] == 1:
        return torch.cat([x, x])
    return x


def _to_cpu(x: Any) -> Any:
    from torch.utils._pytree import tree_map

    return tree_map(lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t, x)


def _float_leaves(out: Any) -> list[torch.Tensor]:
    from torch.utils._pytree import tree_flatten

    return [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor) and t.is_floating_point()]


def _run(model: nn.Module, args: tuple, kwargs: dict) -> Any:
    with torch.no_grad():
        return model(*args, **kwargs)


def _max_diff(a: Any, b: Any) -> float:
    la, lb = _float_leaves(a), _float_leaves(b)
    if len(la) != len(lb):
        return float("inf")
    worst = 0.0
    for x, y in zip(la, lb):
        if x.shape != y.shape:
            return float("inf")
        scale = 1.0 + float(x.float().abs().max()) if x.numel() else 1.0
        worst = max(worst, float((x.float() - y.float()).abs().max()) / scale if x.numel() else 0.0)
    return worst


def tensor_bytes(model: nn.Module) -> tuple[int, int]:
    """(Anzahl Elemente der Parameter, Bytes aller Parameter + persistenten Puffer)."""
    n = sum(p.numel() for p in model.parameters())
    sd = model.state_dict()
    seen, size = set(), 0
    for t in sd.values():
        if isinstance(t, torch.Tensor) and id(t) not in seen:
            seen.add(id(t))
            size += t.numel() * t.element_size()
    return n, size


# ===========================================================================
# Schritte der Vorbereitung
# ===========================================================================

def strip_training_wrappers(model: nn.Module, offloader: Any = None) -> int:
    """Checkpointing-, Streaming- und Auslagerungs-Hüllen entfernen; alle Gewichte auf die CPU (in place)."""
    from ..memory.streaming import disable_block_streaming

    n = 0
    streamed = [m for m in model.modules() if getattr(m, "_ki_streamed", False)]
    if streamed:
        disable_block_streaming(streamed, torch.device("cpu"))
        n += len(streamed)
    for m in model.modules():
        if getattr(m, "_ki_checkpointed", False):
            m.__class__ = m.__class__.__mro__[1]
            n += 1
        for hooks in (m._forward_pre_hooks, m._forward_hooks):
            for key, fn in list(hooks.items()):
                if any(tag in getattr(fn, "__qualname__", "") for tag in ("OffloadManager.", "SplitManager.")):
                    del hooks[key]
                    n += 1
    if offloader is not None and getattr(offloader, "cfg", None) is not None and offloader.cfg.mode == "disk":
        for idx, (name, _) in enumerate(offloader.blocks):        # ausgelagerte Gewichte von der SSD holen
            block = model.get_submodule(name)
            state = torch.load(offloader._path(idx), map_location="cpu", weights_only=True)
            for k, v in state.items():
                owner_name, _, attr = k.rpartition(".")
                owner = block.get_submodule(owner_name) if owner_name else block
                if attr in owner._parameters and owner._parameters[attr] is not None:
                    owner._parameters[attr].data = v
                elif attr in owner._buffers:
                    owner._buffers[attr] = v
    model.cpu()
    return n


_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_FOLDABLE = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


@torch.no_grad()
def fold_batchnorm(model: nn.Module) -> int:
    """BatchNorm direkt nach einer Faltung/linearen Schicht (in ``nn.Sequential``) in deren Gewichte einrechnen."""
    n = 0
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        keys = list(seq._modules)
        for a, b in zip(keys, keys[1:]):
            conv, bn = seq._modules[a], seq._modules[b]
            if not (isinstance(conv, _FOLDABLE) and isinstance(bn, _BN)) or not bn.track_running_stats \
                    or bn.running_mean is None or conv.weight.shape[0] != bn.num_features:
                continue
            scale = bn.weight.float() / (bn.running_var.float() + bn.eps).sqrt() if bn.affine else \
                1.0 / (bn.running_var.float() + bn.eps).sqrt()
            shift = (bn.bias.float() if bn.affine else 0.0) - bn.running_mean.float() * scale
            w = conv.weight.float() * scale.view(-1, *([1] * (conv.weight.dim() - 1)))
            bias = (conv.bias.float() * scale if conv.bias is not None else 0.0) + shift
            conv.weight.data = w.to(conv.weight.dtype)
            if conv.bias is None:
                conv.bias = nn.Parameter(bias.to(conv.weight.dtype), requires_grad=False)
            else:
                conv.bias.data = bias.to(conv.bias.dtype)
            seq._modules[b] = nn.Identity()
            n += 1
    return n


def _unused_from_graph(model: nn.Module, args: tuple, kwargs: dict) -> set[str] | None:
    """Namen von Parametern/Puffern, die im exportierten Rechengraphen keine Verwendung haben."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            program = torch.export.export(model, args, kwargs)
    except Exception:  # noqa: BLE001 - nicht exportierbar → anderer Weg
        return None
    sig = program.graph_signature
    names = {**sig.inputs_to_parameters, **sig.inputs_to_buffers}
    unused = set()
    for node in program.graph.nodes:
        if node.op == "placeholder" and node.name in names and not node.users:
            unused.add(names[node.name])
    return unused


def _unused_from_autograd(model: nn.Module, args: tuple, kwargs: dict) -> set[str]:
    """Fallback: Parameter ohne Gradienten nach einem Vorwärts-/Rückwärtsdurchlauf (Puffer bleiben)."""
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(p.is_floating_point())
        p.grad = None
    try:
        with torch.enable_grad():
            leaves = _float_leaves(model(*args, **kwargs))
            total = sum(t.float().sum() for t in leaves if t.requires_grad)
            if not isinstance(total, torch.Tensor):
                return set()
            total.backward()
        return {n for n, p in model.named_parameters() if p.is_floating_point() and p.grad is None}
    except Exception:  # noqa: BLE001
        return set()
    finally:
        for n, p in model.named_parameters():
            p.grad = None
            p.requires_grad_(flags.get(n, False))


def strip_unused(model: nn.Module, args: tuple, kwargs: dict) -> list[str]:
    """Ungenutzte Parameter/Puffer durch leere Tensoren ersetzen (in place); liefert ihre Namen."""
    unused = _unused_from_graph(model, args, kwargs)
    if unused is None:
        unused = _unused_from_autograd(model, args, kwargs)
    # geteilte Tensoren (z. B. gebundene Embeddings) nur entfernen, wenn *alle* Namen ungenutzt sind
    owners: dict[int, list[str]] = {}
    for name, t in list(model.named_parameters(remove_duplicate=False)) + \
            list(model.named_buffers(remove_duplicate=False)):
        owners.setdefault(id(t), []).append(name)
    removed = []
    for names in owners.values():
        if not set(names) <= unused:
            continue
        for name in names:
            owner_name, _, attr = name.rpartition(".")
            owner = model.get_submodule(owner_name) if owner_name else model
            if attr in owner._parameters and owner._parameters[attr] is not None:
                old = owner._parameters[attr]
                if old.numel() == 0:
                    continue
                owner._parameters[attr] = nn.Parameter(torch.empty(0, dtype=old.dtype), requires_grad=False)
                removed.append(name)
            elif attr in owner._buffers and owner._buffers[attr] is not None:
                old = owner._buffers[attr]
                if old.numel() == 0:
                    continue
                owner._buffers[attr] = torch.empty(0, dtype=old.dtype)
                removed.append(name)
    return sorted(removed)


def _convert_dtype(model: nn.Module, name: str) -> None:
    dtype = _DTYPES[name]
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    for mod in model.modules():
        for k, b in mod._buffers.items():
            if b is not None and b.is_floating_point():
                mod._buffers[k] = b.to(dtype)


def prepare_for_export(model: nn.Module, example: Any = None, cfg: ExportConfig | None = None,
                       ema_state: dict | None = None, offloader: Any = None) -> tuple[nn.Module, ExportReport]:
    """Schlanke, geprüfte Einsatz-Kopie von ``model`` (siehe Modul-Dokumentation)."""
    from ..memory.lora import merge_lora

    cfg = cfg or ExportConfig()
    target = getattr(model, "_orig_mod", model)
    target = getattr(target, "module", target) if isinstance(target, nn.parallel.DistributedDataParallel) else target
    report = ExportReport(dtype=cfg.dtype)
    report.parameters_before, report.bytes_before = tensor_bytes(target)
    m = copy.deepcopy(target)
    report.wrappers_removed = strip_training_wrappers(m, offloader)
    m.eval()
    for prm in m.parameters():            # Einsatz-Modell: keine Gradienten (auch kein Autograd-Ballast im Graph)
        prm.requires_grad_(False)
    args, kwargs = _as_call(_to_cpu(example)) if example is not None else ((), {})
    have_example = example is not None
    reference = None
    if have_example:
        try:
            reference = _run(m, args, kwargs)
        except Exception as exc:  # noqa: BLE001 - ohne lauffähiges Beispiel keine Prüfung
            report.notes.append(f"Beispiel-Eingabe passt nicht ({type(exc).__name__}) – ohne Prüfung exportiert.")
            have_example = False
    if cfg.use_ema and ema_state:
        params = dict(m.named_parameters())
        for k, v in ema_state.items():
            if k in params and params[k].shape == v.shape:
                params[k].data = v.detach().to(params[k].device, params[k].dtype).clone()
        report.ema = True
        if have_example:
            reference = _run(m, args, kwargs)       # Prüfen gegen das EMA-Modell (gewollte Änderung)

    def checked(step: str, fn) -> Any:
        """Schritt auf einer Sicherungskopie ausführen; ändert er die Ausgabe, zurücknehmen."""
        nonlocal m
        backup = copy.deepcopy(m) if (cfg.verify and have_example) else None
        result = fn(m)
        if backup is not None:
            try:
                diff = _max_diff(reference, _run(m, args, kwargs))
            except Exception:  # noqa: BLE001
                diff = float("inf")
            if diff > max(cfg.tolerance, 1e-6):
                m = backup
                report.notes.append(f"{step} zurückgenommen (Abweichung {diff:.3g}).")
                return None
        return result

    if cfg.merge_lora != "nein":
        policy = "behalten" if cfg.merge_lora in ("auto", "behalten") else cfg.merge_lora
        if policy == "neu_quantisieren":                    # bewusst verlustbehaftet → keine Rücknahme
            report.lora_merged = merge_lora(m, policy)
            if have_example and report.lora_merged:
                report.notes.append(f"LoRA neu quantisiert: Abweichung {_max_diff(reference, _run(m, args, kwargs)):.3g}.")
        else:
            report.lora_merged = checked("LoRA einrechnen", lambda mm: merge_lora(mm, policy)) or 0
    if cfg.fold_batchnorm:
        report.batchnorm_folded = checked("BatchNorm falten", fold_batchnorm) or 0
    if cfg.strip_unused and have_example:
        report.removed = checked("Ungenutztes entfernen", lambda mm: strip_unused(mm, args, kwargs)) or []
    if cfg.dtype != "wie_modell":
        _convert_dtype(m, cfg.dtype)
        if have_example:
            try:
                low = _run(m, tuple(a.to(_DTYPES[cfg.dtype]) if isinstance(a, torch.Tensor) and a.is_floating_point()
                                    else a for a in args), kwargs)
                report.notes.append(f"Präzision {cfg.dtype}: Abweichung {_max_diff(reference, low):.3g} "
                                    "(Rundung, erwartet).")
            except Exception as exc:  # noqa: BLE001 - z. B. float16-Operationen fehlen auf der CPU
                report.notes.append(f"{cfg.dtype} auf der CPU nicht prüfbar ({type(exc).__name__}).")
    elif have_example:
        report.max_diff = _max_diff(reference, _run(m, args, kwargs))
    report.parameters_after, report.bytes_after = tensor_bytes(m)
    return m, report


# ===========================================================================
# Export
# ===========================================================================

def export_model(model: nn.Module, path: str | Path, fmt: str = "export", example: Any = None,
                 merge_lora: bool | str = True, dynamic: bool = True, cfg: ExportConfig | None = None,
                 ema_state: dict | None = None, offloader: Any = None, return_report: bool = False) -> Any:
    """Exportiert ``model`` nach ``path`` (Endung wird ergänzt). Liefert den Zielpfad
    (mit ``return_report=True``: ``(pfad, ExportReport)``).

    ``dynamic=True`` exportiert (bei ``export``) mit variabler Batchgröße bzw. Sequenzlänge;
    gelingt das nicht, wird automatisch mit festen Formen exportiert.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unbekanntes Exportformat '{fmt}'. Erlaubt: {', '.join(FORMATS)}.")
    cfg = (cfg or ExportConfig()).copy()
    if merge_lora is False:
        cfg.merge_lora = "nein"
    elif isinstance(merge_lora, str):
        cfg.merge_lora = merge_lora
    cfg.dynamic = cfg.dynamic and dynamic
    path = Path(path)
    if path.suffix != FORMATS[fmt]:
        path = path.with_suffix(FORMATS[fmt])
    path.parent.mkdir(parents=True, exist_ok=True)
    if example is None and hasattr(model, "example_input"):
        try:
            example = getattr(model, "_orig_mod", model).example_input(1)
        except Exception:  # noqa: BLE001
            example = None
    m, report = prepare_for_export(model, example, cfg, ema_state, offloader)
    report.path = path
    if fmt == "state_dict":
        state = {k: v for k, v in m.state_dict().items() if not (isinstance(v, torch.Tensor) and v.numel() == 0
                                                                 and k in report.removed)}
        torch.save(state, path)
        return (path, report) if return_report else path
    if example is None:
        raise ValueError("Für diesen Export wird eine Beispiel-Eingabe benötigt (example=...).")
    args, kwargs = _as_call(_to_cpu(example))
    if cfg.dtype in ("float16", "bfloat16"):
        args = tuple(a.to(_DTYPES[cfg.dtype]) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                     for a in args)
        kwargs = {k: v.to(_DTYPES[cfg.dtype]) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                  for k, v in kwargs.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt == "export":
            program = None
            if cfg.dynamic:
                dyn_args = tuple(_double_batch(a) for a in args)
                dyn_kwargs = {k: _double_batch(v) for k, v in kwargs.items()}
                try:
                    program = torch.export.export(m, dyn_args, dyn_kwargs,
                                                  dynamic_shapes=_dynamic_shapes(dyn_args, dyn_kwargs))
                except Exception as exc:  # noqa: BLE001 - manche Modelle brauchen feste Formen
                    report.notes.append(f"Dynamische Formen nicht möglich ({type(exc).__name__}) – feste Formen.")
            if program is None:
                program = torch.export.export(m, args, kwargs)
            torch.export.save(program, str(path))
        elif fmt == "torchscript":
            if kwargs:
                traced = torch.jit.trace(m, example_kwarg_inputs=kwargs, check_trace=False, strict=False)
            else:
                traced = torch.jit.trace(m, args, check_trace=False, strict=False)
            traced.save(str(path))
        else:
            try:
                torch.onnx.export(m, args if not kwargs else tuple(kwargs.values()), str(path), dynamo=False)
            except Exception as exc:  # noqa: BLE001
                if "onnx" in str(exc).lower() and "install" in str(exc).lower():
                    raise ImportError("ONNX-Export benötigt 'pip install onnx'. Alternative ohne Zusatzpaket: "
                                      "fmt='export' (torch.export).") from exc
                raise
    return (path, report) if return_report else path
