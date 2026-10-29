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


def export_model(model: nn.Module, path: str | Path, fmt: str = "export", example: Any = None,
                 merge_lora: bool = True) -> Path:
    """Exportiert ``model`` nach ``path`` (Endung wird ergänzt). Liefert den Zielpfad."""
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt == "export":
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
