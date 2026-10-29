"""
ki_baukasten.components.cache
=============================

KV-Cache und Zustands-Caches für schnelle autoregressive Generierung.

Warum?
------
Ohne Cache rechnet ein Sprachmodell bei jedem neuen Token die **gesamte**
bisherige Sequenz neu durch – Aufwand O(T²) pro Token bei Attention. Mit Cache
wird nur das neue Token berechnet; alles Frühere liegt bereits vor:

==============  ====================================================  ===============================
Mixer           Was gespeichert wird                                  Speicher pro Schicht
==============  ====================================================  ===============================
``attention``   Keys + Values aller bisherigen Tokens (KV-Cache)      2 · B · T · H_kv · head_dim
  (MLA-lite)    nur der komprimierte Latent-Vektor pro Token          B · T · kv_compression_dim
``ssm``         Faltungs-Fenster + Zustand h (konstant!)              B · E · (d_conv − 1 + d_state)
``rnn``         versteckter Zustand (h bzw. h, c)                     B · D (·2 bei LSTM)
``conv``        die letzten kernel_size − 1 Eingaben                  B · D · (k − 1)
==============  ====================================================  ===============================

Speicher-Stellschrauben (``KVCacheConfig``)
-------------------------------------------
* ``dtype``: Cache in float16/bfloat16 oder **int8** (pro Token und Kopf
  quantisiert) → 2–4× weniger Speicher als float32.
* ``max_length`` + ``sink_tokens``: Cache-Fenster begrenzen (StreamingLLM):
  die ersten ``sink_tokens`` Tokens bleiben ("Attention Sinks"), sonst nur die
  letzten Einträge → konstanter Speicher auch bei sehr langen Generierungen.
* Grouped-Query-Attention (``num_kv_heads``) und MLA-lite
  (``kv_compression_dim``) verkleinern den Cache bereits beim Modellbau.

Protokoll für Mixer
-------------------
Ein Mixer, der ``supports_cache = True`` setzt, erhält im Forward das
Argument ``cache`` – ein ``LayerCache`` (Wörterbuch) **seiner eigenen
Schicht**. Er liest daraus den bisherigen Zustand, verarbeitet nur die neuen
Positionen und legt den aktualisierten Zustand wieder ab. Eigene Mixer können
das Protokoll ebenso umsetzen; Mixer ohne Unterstützung führen automatisch zur
(langsameren, aber korrekten) Neuberechnung ohne Cache.
"""

from __future__ import annotations

from typing import Any

import torch

from ..core import BaseConfig, Level, Param, T


class KVCacheConfig(BaseConfig):
    __title__ = "KV-/Zustands-Cache"
    __tooltip__ = T(
        "Zwischenspeicher für Keys/Values (Attention) und Zustände (SSM, RNN, Faltung) bei der Generierung.",
        "Jedes neue Token kostet dann nur noch einen Schritt statt einer kompletten Neuberechnung.",
        "Text-/Sequenzgenerierung mit decoder_only-Modellen.",
    )
    dtype = Param("auto", choices={
        "auto": "wie die Rechnung (meist float32 bzw. autocast-Typ)",
        "float32": "32 Bit (genau)",
        "float16": "16 Bit (halber Speicher)",
        "bfloat16": "16 Bit Brain-Float (halber Speicher, großer Wertebereich)",
        "int8": "8 Bit, pro Token und Kopf skaliert (¼ Speicher, minimal ungenauer)",
    }, tooltip=T(
        "Datentyp, in dem Keys/Values im Cache liegen.",
        "Der KV-Cache ist bei langen Kontexten oft größer als das Modell selbst – kleinere Typen sparen viel.",
        "Lange Kontexte, große Batches; int8 ist meist praktisch verlustfrei.",
        "Gilt für Attention-Caches; SSM-/RNN-Zustände bleiben in voller Genauigkeit (sie sind klein).",
    ))
    max_length = Param(None, type=int, min=1, tooltip=T(
        "Höchstzahl gespeicherter Positionen pro Schicht (None = unbegrenzt).",
        "Begrenzt den Speicher: ältere Einträge werden verworfen (gleitendes Fenster).",
        "Sehr lange Generierungen, Streaming; mit sink_tokens kombinieren.",
        "Positions-Bias (ALiBi/T5) und Sliding-Window nutzen dann Abstände innerhalb des Caches.",
    ), level=Level.ADVANCED)
    sink_tokens = Param(4, min=0, tooltip=T(
        "Anzahl der ersten Tokens, die bei begrenztem Cache nie verworfen werden ('Attention Sinks').",
        "Modelle legen viel Attention-Gewicht auf die ersten Tokens – sie zu behalten hält die Qualität stabil.",
        "StreamingLLM (Xiao et al. 2023): 4.",
    ), level=Level.EXPERT)
    overflow_keep = Param(0.5, min=0.0, max=1.0, tooltip=T(
        "Wird max_seq_len überschritten, wird der Cache aus diesem Anteil der letzten Tokens neu aufgebaut.",
        "1.0 = exakt wie ohne Cache (aber Neuberechnung bei jedem Schritt); 0.5 = nur selten neu aufbauen.",
        "Generierungen länger als der Trainingskontext.",
    ), level=Level.EXPERT)

    def check(self) -> list[str]:
        if self.max_length is not None and self.sink_tokens >= self.max_length:
            return ["sink_tokens muss kleiner als max_length sein."]
        return []


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class LayerCache(dict):
    """Cache einer einzelnen Schicht (beliebige Einträge des jeweiligen Mixers).

    Hilfsmethode ``append`` kümmert sich um Datentyp, int8-Quantisierung und
    die Fensterbegrenzung – Mixer müssen das nicht selbst implementieren.
    """

    def __init__(self, cfg: KVCacheConfig):
        super().__init__()
        self.cfg = cfg

    # ------------------------------------------------------------ Speichern
    def append(self, key: str, new: torch.Tensor, dim: int) -> torch.Tensor:
        """Hängt ``new`` entlang ``dim`` an den gespeicherten Tensor ``key`` an.

        Returns:
            Den **vollständigen** Tensor (alt + neu) im Datentyp von ``new`` – für
            die Rechnung dieses Schritts. Gespeichert wird (ggf. begrenzt und
            komprimiert) für die folgenden Schritte.
        """
        old = self.read(key, new.dtype)
        full = new if old is None else torch.cat([old, new], dim)
        new = new.detach()
        if self.cfg.dtype == "int8":
            # Symmetrisch pro Vektor (letzte Dimension = Merkmale eines Tokens/Kopfes); nur Neues quantisieren.
            scale = new.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
            q = (new / scale).round().clamp(-127, 127).to(torch.int8)
            scale = scale.to(torch.float16 if new.dtype == torch.float32 else new.dtype)
            if key + ":q" in self:
                q = torch.cat([self[key + ":q"], q], dim)
                scale = torch.cat([self[key + ":s"], scale], dim)
            self[key + ":q"], self[key + ":s"] = self.truncate(q, dim), self.truncate(scale, dim)
        else:
            stored = new.to(_DTYPES.get(self.cfg.dtype, new.dtype))
            if key in self:
                stored = torch.cat([self[key], stored], dim)
            self[key] = self.truncate(stored, dim)
        return full

    def append_mask(self, new: torch.Tensor | None, batch: int, length: int, device: torch.device) -> torch.Tensor | None:
        """Padding-Maske fortschreiben (``True`` = gültig). ``None`` solange nirgends Padding vorkam."""
        old = self.get("mask")
        if new is None and old is None:
            return None
        if new is None:
            new = torch.ones(batch, length, dtype=torch.bool, device=device)
        new = new.bool()
        full = new if old is None else torch.cat([old, new], 1)
        self["mask"] = self.truncate(full, 1)
        return full

    def read(self, key: str, dtype: torch.dtype) -> torch.Tensor | None:
        if key in self:
            return self[key].to(dtype)
        if key + ":q" in self:
            return (self[key + ":q"].to(dtype) * self[key + ":s"].to(dtype))
        return None

    def truncate(self, full: torch.Tensor, dim: int) -> torch.Tensor:
        limit = self.cfg.max_length
        if limit is None or full.shape[dim] <= limit:
            return full
        sinks = self.cfg.sink_tokens
        tail = full.narrow(dim, full.shape[dim] - (limit - sinks), limit - sinks)
        return torch.cat([full.narrow(dim, 0, sinks), tail], dim) if sinks else tail

    # ------------------------------------------------------------ Auskunft
    def nbytes(self) -> int:
        total = 0
        for v in self.values():
            for t in (v if isinstance(v, (tuple, list)) else (v,)):
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total


class ModelCache:
    """Caches aller Schichten eines Modells plus Zähler der bereits verarbeiteten Positionen."""

    def __init__(self, n_layers: int, cfg: KVCacheConfig | dict | None = None):
        if isinstance(cfg, dict):
            cfg = KVCacheConfig.from_dict(cfg)
        self.cfg = (cfg or KVCacheConfig()).validate()
        self.layers = [LayerCache(self.cfg) for _ in range(n_layers)]
        self.seen = 0  # Anzahl bereits verarbeiteter Positionen (inkl. Präfix-/Register-Tokens)

    def __getitem__(self, i: int) -> LayerCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.clear()
        self.seen = 0

    def nbytes(self) -> int:
        """Aktueller Speicherbedarf des Caches in Bytes."""
        return sum(layer.nbytes() for layer in self.layers)

    def summary(self) -> dict[str, Any]:
        return {"schichten": len(self.layers), "positionen": self.seen, "bytes": self.nbytes(),
                "dtype": self.cfg.dtype, "max_length": self.cfg.max_length}


def module_supports_cache(module: torch.nn.Module) -> bool:
    return bool(getattr(module, "supports_cache", False))
