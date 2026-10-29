"""
ki_baukasten.memory.quant
=========================

Eigene **Gewichts-Quantisierung** (int8 / int4 / nf4) – unabhängig von der
auslaufenden ``torch.ao.quantization``-Schnittstelle, in reinem PyTorch und
auf **jedem Gerät** (CPU, CUDA, MPS) lauffähig.

Prinzip ("weight-only")
-----------------------
::

    Speichern:  W (float32)  ─► pro Gruppe von g Werten: Skala s = max|W| / q_max
                              ─► q = round(W / s)  als int8 (bzw. 2 Werte pro Byte bei 4 Bit)
    Rechnen:    W' = q · s  (on-the-fly entpackt, im Datentyp der Eingabe)  ─►  y = x·W'ᵀ + b

Die Gewichte liegen dauerhaft komprimiert im Speicher; nur während der
Matrixmultiplikation einer Schicht existiert kurz die entpackte Kopie.

==========  ====================  =================  ===========================================
Format      Bytes/Gewicht         Fehler             Einsatz
==========  ====================  =================  ===========================================
``int8``    1 (+ Skalen)          sehr klein         Inferenz, eingefrorene Basis beim Feintuning
``int4``    0.5 (+ Skalen)        spürbar            Aggressive Kompression, kleine Gruppen nutzen
``nf4``     0.5 (+ Skalen)        kleiner als int4   QLoRA: Werte normalverteilter Gewichte optimal
==========  ====================  =================  ===========================================

* ``group_size``: Gewichte einer Zeile werden in Gruppen zu g Werten mit
  eigener Skala quantisiert (kleiner = genauer, etwas mehr Skalen-Speicher).
  ``None`` = eine Skala pro Ausgabekanal.
* Quantisierte Gewichte sind **nicht trainierbar** (Puffer). Training läuft
  über Biases, nicht quantisierte Schichten oder **LoRA-Adapter** (QLoRA).
* Backend ``torchao``: Ist das Paket ``torchao`` installiert, können
  ``torchao_int8`` / ``torchao_int4`` dessen optimierte Kernel nutzen.
"""

from __future__ import annotations

import re
import warnings

import torch
from torch import nn
import torch.nn.functional as F

# NormalFloat-4-Codebuch (Dettmers et al. 2023, QLoRA): Quantile einer Normalverteilung, auf [-1, 1] normiert.
NF4_CODE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])

FORMATS = {
    "int8": "8 Bit symmetrisch (1 Byte/Gewicht)",
    "int4": "4 Bit symmetrisch (½ Byte/Gewicht)",
    "nf4": "4 Bit NormalFloat (½ Byte/Gewicht, QLoRA)",
}


def _groups(w: torch.Tensor, group_size: int | None) -> tuple[torch.Tensor, int]:
    """``[out, in]`` → ``[out, n_groups, g]`` (mit Nullen aufgefüllt); liefert auch die Auffüllung."""
    out_f, in_f = w.shape
    g = in_f if group_size is None else min(group_size, in_f)
    pad = (-in_f) % g
    if pad:
        w = F.pad(w, (0, pad))
    return w.reshape(out_f, -1, g), pad


def _pack4(q: torch.Tensor) -> torch.Tensor:
    """Zwei 4-Bit-Werte (0…15) pro Byte."""
    q = q.to(torch.uint8)
    if q.shape[-1] % 2:
        q = F.pad(q, (0, 1))
    return q[..., 0::2] | (q[..., 1::2] << 4)


def _unpack4(p: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.stack([p & 0x0F, p >> 4], -1).flatten(-2)
    return out[..., :n]


def quantize_weight(w: torch.Tensor, fmt: str, group_size: int | None) -> dict[str, torch.Tensor]:
    """Quantisiert eine Gewichtsmatrix ``[out, in]``; liefert die zu speichernden Tensoren."""
    w = w.detach().float()
    grouped, pad = _groups(w, group_size)
    absmax = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    if fmt == "int8":
        scale = absmax / 127.0
        q = (grouped / scale).round().clamp(-127, 127).to(torch.int8)
    elif fmt == "int4":
        scale = absmax / 7.0
        q = _pack4((grouped / scale).round().clamp(-8, 7) + 8)
    elif fmt == "nf4":
        scale = absmax
        code = NF4_CODE.to(w.device)
        mids = (code[1:] + code[:-1]) / 2
        q = _pack4(torch.bucketize((grouped / scale).contiguous(), mids))
    else:
        raise ValueError(f"Unbekanntes Format '{fmt}'. Erlaubt: {', '.join(FORMATS)}.")
    return {"qweight": q, "scales": scale.squeeze(-1).to(torch.float16), "pad": torch.tensor(pad)}


def dequantize_weight(qweight: torch.Tensor, scales: torch.Tensor, fmt: str, shape: tuple[int, int], pad: int,
                      dtype: torch.dtype) -> torch.Tensor:
    out_f, in_f = shape
    g = (in_f + pad) // scales.shape[1]
    s = scales.to(dtype).unsqueeze(-1)
    if fmt == "int8":
        w = qweight.to(dtype) * s
    else:
        idx = _unpack4(qweight, g)
        w = (idx.to(dtype) - 8) * s if fmt == "int4" else NF4_CODE.to(qweight.device, dtype)[idx.long()] * s
    w = w.reshape(out_f, in_f + pad)
    return w[:, :in_f] if pad else w


class QuantLinear(nn.Module):
    """Lineare Schicht mit quantisierten (eingefrorenen) Gewichten.

    Attribute ``qweight`` und ``scales`` sind Puffer (im ``state_dict``
    enthalten, nicht trainierbar). ``bias`` bleibt ein normaler Parameter.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, fmt: str = "int8",
                 group_size: int | None = 128, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.fmt, self.group_size, self.compute_dtype = fmt, group_size, compute_dtype
        empty = quantize_weight(torch.zeros(out_features, in_features), fmt, group_size)
        self.register_buffer("qweight", empty["qweight"])
        self.register_buffer("scales", empty["scales"])
        self.pad = int(empty["pad"])
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, fmt: str = "int8", group_size: int | None = 128,
                    compute_dtype: torch.dtype | None = None) -> "QuantLinear":
        q = cls(linear.in_features, linear.out_features, linear.bias is not None, fmt, group_size, compute_dtype)
        data = quantize_weight(linear.weight, fmt, group_size)
        q.qweight, q.scales = data["qweight"].to(linear.weight.device), data["scales"].to(linear.weight.device)
        if linear.bias is not None:
            q.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=linear.bias.requires_grad)
        return q

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Entpackte Gewichtsmatrix ``[out, in]``."""
        return dequantize_weight(self.qweight, self.scales, self.fmt, (self.out_features, self.in_features),
                                 self.pad, dtype)

    @property
    def weight(self) -> torch.Tensor:  # Kompatibilität (z. B. für Analyse-Werkzeuge, DoRA)
        return self.dequantize(self.compute_dtype or torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.compute_dtype or autocast_dtype(x)
        bias = self.bias.to(dtype) if self.bias is not None else None
        return F.linear(x.to(dtype), self.dequantize(dtype), bias)

    def nbytes(self) -> int:
        return self.qweight.numel() * self.qweight.element_size() + self.scales.numel() * self.scales.element_size()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"format={self.fmt}, group_size={self.group_size}")


def autocast_dtype(x: torch.Tensor) -> torch.dtype:
    """Rechen-Datentyp: aktiver autocast-Typ des Geräts, sonst der Typ von ``x``."""
    kind = x.device.type
    try:
        if torch.is_autocast_enabled(kind):
            return torch.get_autocast_dtype(kind)
    except (AttributeError, TypeError, RuntimeError):  # ältere PyTorch-Versionen
        if kind == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        if kind == "cpu" and torch.is_autocast_cpu_enabled():
            return torch.get_autocast_cpu_dtype()
    return x.dtype


def _set(root: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, attr = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, attr, module)


def quantize_model(model: nn.Module, fmt: str = "int8", group_size: int | None = 128, skip: str | None = "lm_head",
                   only: str | None = None, compute_dtype: torch.dtype | None = None) -> list[str]:
    """Ersetzt alle ``nn.Linear`` (außer ``skip``, ggf. nur ``only``) durch ``QuantLinear``.

    Args:
        skip: Regex auf Modulnamen, die unverändert bleiben (Standard: Ausgabeschicht).
        only: Regex – nur passende Module quantisieren (``None`` = alle).

    Returns:
        Namen der quantisierten Module.
    """
    rx_skip = re.compile(skip) if skip else None
    rx_only = re.compile(only) if only else None
    done = []
    for name, module in list(model.named_modules()):
        if type(module) is not nn.Linear:  # exakt nn.Linear (Unterklassen haben oft Sonderlogik)
            continue
        if rx_skip and rx_skip.search(name):
            continue
        if rx_only and not rx_only.search(name):
            continue
        _set(model, name, QuantLinear.from_linear(module, fmt, group_size, compute_dtype))
        done.append(name)
    return done


def quantize_torchao(model: nn.Module, fmt: str = "int8", group_size: int | None = 128) -> nn.Module:
    """Quantisierung über das Paket ``torchao`` (optimierte Kernel), falls installiert."""
    try:
        import torchao.quantization as tq  # type: ignore
    except ImportError as exc:
        raise ImportError("Backend 'torchao' benötigt 'pip install torchao'. Alternative ohne Zusatzpaket: "
                          "memory.quantization = 'int8' / 'int4' / 'nf4' (eigene Implementierung).") from exc
    if fmt == "int8":
        cfg = tq.Int8WeightOnlyConfig() if hasattr(tq, "Int8WeightOnlyConfig") else tq.int8_weight_only()
    else:
        gs = group_size or 128
        cfg = tq.Int4WeightOnlyConfig(group_size=gs) if hasattr(tq, "Int4WeightOnlyConfig") \
            else tq.int4_weight_only(group_size=gs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tq.quantize_(model, cfg)
    return model


def quantized_bytes(model: nn.Module) -> tuple[int, int]:
    """(Bytes quantisierter Gewichte, Bytes, die dieselben Gewichte in float32 bräuchten)."""
    q = full = 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            q += m.nbytes()
            full += m.in_features * m.out_features * 4
    return q, full
