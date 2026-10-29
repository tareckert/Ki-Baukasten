"""
ki_baukasten.analysis.export
============================

Export gebauter/trainierter Modelle zur Weiterverwendung außerhalb des Baukastens.

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

LoRA-Adapter werden vor dem Export standardmäßig in die Gewichte eingerechnet
(``merge_lora=True``) – auf einer Kopie, das Original bleibt unverändert.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

FORMATS = {
    "state_dict": ".pt", "export": ".pt2", "torchscript": ".ts", "onnx": ".onnx",
}


def _as_tuple(example: Any) -> tuple:
    if isinstance(example, tuple):
        return example
    if isinstance(example, dict):
        return tuple(example.values())
    return (example,)


def _dynamic_shapes(args: tuple) -> tuple | None:
    """Batch (und bei Sequenzen die Länge) dynamisch halten – Bilder: nur der Batch."""
    from torch.export import Dim

    spec = []
    for a in args:
        if not isinstance(a, torch.Tensor) or a.dim() == 0:
            spec.append(None)
            continue
        dims = {0: Dim.AUTO}
        if a.dim() in (2, 3) and not (a.dim() == 2 and a.is_floating_point()):
            dims[1] = Dim.AUTO   # [B, T] Tokens bzw. [B, T, F] Zeitreihen
        spec.append(dims)
    return tuple(spec)


def export_model(model: nn.Module, path: str | Path, fmt: str = "export", example: Any = None,
                 merge_lora: bool = True, dynamic: bool = True) -> Path:
    """Exportiert ``model`` nach ``path`` (Endung wird ergänzt). Liefert den Zielpfad.

    ``dynamic=True`` exportiert (bei ``export``) mit variabler Batchgröße bzw. Sequenzlänge;
    gelingt das nicht, wird automatisch mit festen Formen exportiert.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unbekanntes Exportformat '{fmt}'. Erlaubt: {', '.join(FORMATS)}.")
    path = Path(path)
    if path.suffix != FORMATS[fmt]:
        path = path.with_suffix(FORMATS[fmt])
    path.parent.mkdir(parents=True, exist_ok=True)
    target = getattr(model, "_orig_mod", model)
    if fmt == "state_dict":
        torch.save(target.state_dict(), path)
        return path
    m = copy.deepcopy(target).cpu().eval()
    if merge_lora:
        from ..memory.lora import merge_lora as _merge
        _merge(m)
    if example is None:
        if not hasattr(m, "example_input"):
            raise ValueError("Für diesen Export wird eine Beispiel-Eingabe benötigt (example=...).")
        example = m.example_input(1)
    args = _as_tuple(example)
    dyn_args = args
    if dynamic and fmt == "export":
        # Mit Batch 1 würde torch.export die Batch-Dimension auf 1 festlegen → Beispiel verdoppeln.
        dyn_args = tuple(torch.cat([a, a]) if isinstance(a, torch.Tensor) and a.dim() > 0 and a.shape[0] == 1 else a
                         for a in args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt == "export":
            program = None
            if dynamic:
                try:
                    program = torch.export.export(m, dyn_args, dynamic_shapes=_dynamic_shapes(dyn_args))
                except Exception as exc:  # noqa: BLE001 - manche Modelle brauchen feste Formen
                    warnings.warn(f"Dynamischer Export nicht möglich ({type(exc).__name__}) – exportiere mit festen "
                                  "Formen.")
                    program = None
            if program is None:
                program = torch.export.export(m, args)
            torch.export.save(program, str(path))
        elif fmt == "torchscript":
            traced = torch.jit.trace(m, args, check_trace=False)
            traced.save(str(path))
        else:
            try:
                torch.onnx.export(m, args, str(path), dynamo=False)
            except Exception as exc:  # noqa: BLE001
                if "onnx" in str(exc).lower() and "install" in str(exc).lower():
                    raise ImportError("ONNX-Export benötigt 'pip install onnx'. Alternative ohne Zusatzpaket: "
                                      "fmt='export' (torch.export).") from exc
                raise
    return path
