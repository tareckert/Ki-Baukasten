"""
ki_baukasten.models.external
============================

**Fremde Modelle importieren, verändern und wieder exportieren.**

Das Modell ``extern`` umhüllt ein von außen geladenes PyTorch-Modell, sodass alle
Werkzeuge des Baukastens darauf funktionieren: Aufbau-Darstellung, Tensor-Regeln
(nach Parameternamen), Modul-Regeln, LoRA, Quantisierung, Kompression, Training,
Distillation und Export.

Formate (``format``)
--------------------
===============  ================================================================================
``export``       ``torch.export``-Datei (``.pt2``) – auch aus diesem Baukasten exportiert
``torchscript``  TorchScript (``.pt``/``.ts`` aus ``torch.jit``) – trainierbar, Struktur fest
``module``       komplettes, mit ``torch.save(model)`` gespeichertes Modul (Pickle!) – nur mit
                 ``allow_pickle = True``, da beim Laden beliebiger Code ausgeführt werden kann
``state_dict``   reine Gewichte (``.pt``/``.bin``) in eine **Baukasten-Architektur** laden
                 (``architecture``), Schlüssel per ``key_map`` umbenennen
``safetensors``  wie ``state_dict``, aus ``.safetensors`` (Paket ``safetensors``)
``huggingface``  Modell-Ordner/-Name von Hugging Face (Paket ``transformers``)
``auto``         nach Dateiendung bzw. Ordnerinhalt
===============  ================================================================================

Ändern
------
* Tensor-Regeln und LoRA/Quantisierung wirken über Parameternamen bzw. ``nn.Linear`` –
  bei ``state_dict``/``safetensors``/``huggingface``/``module`` uneingeschränkt.
* ``export``-Module (GraphModule) behalten die Modulnamen, enthalten aber keine
  ``nn.Linear``-Objekte mehr: Tensor-Regeln funktionieren, LoRA/Quantisierung nicht.
  Für volle Umbaufreiheit: Gewichte als ``state_dict`` in eine Baukasten-Architektur laden.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..core import ComponentParam, ConfigError, Level, Param, T
from .base import MODELS, BaseModel, ModelConfig


class ExternalModelConfig(ModelConfig):
    __title__ = "Externes Modell (Import)"
    __tooltip__ = T(
        "Ein von außen geladenes Modell (torch.export, TorchScript, state_dict, safetensors, Hugging Face …).",
        "Vorhandene KIs mit allen Werkzeugen des Baukastens analysieren, verändern, nachtrainieren und exportieren.",
        "Fremde/ältere Modelle weiterverwenden, Feintuning, Kompression für neue Zielsysteme.",
    )
    path = Param("", tooltip=T(
        "Pfad zur Modelldatei bzw. zum Modell-Ordner (Hugging Face: auch Modellname).",
        "Quelle des Imports.", "z. B. 'modelle/klassifikator.pt2'."))
    format = Param("auto", choices={
        "auto": "nach Endung/Inhalt erkennen",
        "export": "torch.export (.pt2)",
        "torchscript": "TorchScript (torch.jit)",
        "module": "komplettes gepickeltes Modul (nur mit allow_pickle)",
        "state_dict": "Gewichte in eine Baukasten-Architektur laden",
        "safetensors": "Gewichte aus .safetensors in eine Baukasten-Architektur",
        "huggingface": "Hugging-Face-Modell (Paket transformers)",
    }, tooltip=T("Dateiformat.", "Bestimmt, wie geladen wird.", "auto."))
    allow_pickle = Param(False, tooltip=T(
        "Laden gepickelter Dateien erlauben (format 'module', alte .pt-Dateien).",
        "Pickle kann beim Laden beliebigen Code ausführen – nur für vertrauenswürdige Dateien einschalten!",
        "Eigene, selbst gespeicherte Modelle."), level=Level.ADVANCED)
    architecture = ComponentParam("model", None, tooltip=T(
        "Baukasten-Architektur, in die state_dict/safetensors-Gewichte geladen werden.",
        "Danach ist das Modell ein normales Baukasten-Modell – mit voller Umbaufreiheit.",
        "Gleiche Form wie das Original (Schlüssel ggf. mit key_map umbenennen)."))
    key_map = Param({}, tooltip=T(
        "Umbenennung der Gewichtsschlüssel: {Regex: Ersatz}, der Reihe nach angewendet.",
        "Fremde Namen auf Baukasten-Namen abbilden (z. B. 'transformer.h.' → 'blocks.').",
        "{'^model\\\\.': '', 'attn.c_proj': 'mixer.attn.o_proj'}"), level=Level.ADVANCED)
    strict = Param(True, tooltip=T(
        "Fehlende/überzählige Gewichte als Fehler behandeln.",
        "Aus: teilweise laden (z. B. neuer Kopf); der Bericht zeigt, was fehlte.", "an."), level=Level.ADVANCED)
    hf_task = Param("auto", choices={
        "auto": "Kopf aus der Hugging-Face-Konfiguration", "causal_lm": "Sprachmodell", "base": "nur Rumpf",
        "sequence_classification": "Klassifikation"}, tooltip=T(
        "Welche Hugging-Face-Modellklasse geladen wird.", "Bestimmt den Ausgabekopf.", "auto."), level=Level.EXPERT)
    input_shape = Param([1, 16], item_type=int, min=1, tooltip=T(
        "Form einer Beispiel-Eingabe (für Darstellung, Messung, Export).", "Fremde Modelle kennen ihre Eingabe nicht.",
        "Text: [1, 16]; Bild: [1, 3, 224, 224]; Tabelle: [1, 20]."))
    input_dtype = Param("long", choices={"long": "Ganzzahlen (Tokens/IDs)", "float32": "Gleitkomma"}, tooltip=T(
        "Datentyp der Beispiel-Eingabe.", "Tokens: long; Messwerte/Bilder: float32.", "Je nach Modell."))
    input_high = Param(100, min=1, tooltip=T(
        "Obergrenze zufälliger Token-IDs der Beispiel-Eingabe (≤ Vokabulargröße).", "Gültige Beispiel-Tokens.", "100."),
        level=Level.EXPERT)
    output_key = Param(None, type=str, nullable=True, tooltip=T(
        "Schlüssel der Hauptausgabe, falls das Modell ein dict liefert (None = 'logits' bzw. erster Eintrag).",
        "Einheitliche Tensor-Ausgabe für Training und Analyse.", "'logits'."), level=Level.EXPERT)


def _detect(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    if path.is_dir():
        if (path / "projekt.json").exists():
            raise ConfigError(f"'{path}' ist ein Baukasten-Projekt – mit Project.load öffnen.")
        return "huggingface"
    suffix = path.suffix.lower()
    if suffix == ".pt2":
        return "export"
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in (".ts", ".torchscript"):
        return "torchscript"
    if suffix in (".pt", ".pth", ".bin"):
        try:
            torch.jit.load(str(path), map_location="cpu")
            return "torchscript"
        except Exception:  # noqa: BLE001 - kein TorchScript → Gewichte
            return "state_dict"
    if not path.exists():
        return "huggingface"  # Modellname aus dem Hugging-Face-Hub
    raise ConfigError(f"Format von '{path}' nicht erkennbar – 'format' angeben.")


def _remap(state: dict[str, torch.Tensor], key_map: dict[str, str]) -> dict[str, torch.Tensor]:
    if not key_map:
        return state
    out = {}
    for k, v in state.items():
        for pattern, repl in key_map.items():
            k = re.sub(pattern, repl, k)
        out[k] = v
    return out


def _load_state(path: Path, fmt: str, allow_pickle: bool) -> dict[str, torch.Tensor]:
    if fmt == "safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:
            raise ImportError("safetensors-Import braucht 'pip install safetensors'.") from exc
        return load_file(str(path))
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        if not allow_pickle:
            raise ConfigError(f"'{path}' enthält mehr als reine Tensoren (Pickle). Nur bei vertrauenswürdiger Quelle "
                              "model.allow_pickle = true setzen.") from exc
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]  # Trainer-Checkpoint
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise ConfigError(f"'{path}' enthält kein state_dict.")
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


def _load_hf(path: str, task: str) -> nn.Module:
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise ImportError("Hugging-Face-Import braucht 'pip install transformers'.") from exc
    classes = {"causal_lm": "AutoModelForCausalLM", "sequence_classification": "AutoModelForSequenceClassification",
               "base": "AutoModel"}
    if task == "auto":
        cfg = transformers.AutoConfig.from_pretrained(path)
        archs = getattr(cfg, "architectures", None) or []
        task = "causal_lm" if any("CausalLM" in a or "LMHead" in a for a in archs) else (
            "sequence_classification" if any("SequenceClassification" in a for a in archs) else "base")
    return getattr(transformers, classes[task]).from_pretrained(path)


class ExternalModel(BaseModel):
    """Hülle um ein importiertes Modell (``self.inner``). Siehe Modul-Dokumentation."""

    def __init__(self, config: ExternalModelConfig):
        super().__init__(config)
        c = config
        if not c.path and c.architecture is None:
            raise ConfigError("model.path angeben (Datei/Ordner des zu importierenden Modells).", "model.path")
        path = Path(c.path)
        fmt = _detect(path, c.format)
        self.import_format = fmt
        self.import_report: dict[str, Any] = {"format": fmt}
        if fmt == "export":
            self.inner = torch.export.load(str(path)).module()
        elif fmt == "torchscript":
            self.inner = torch.jit.load(str(path), map_location="cpu")
        if fmt in ("export", "torchscript"):
            # Einsatz-Exporte speichern Gewichte ohne Gradienten – zum Weitertrainieren wieder trainierbar machen
            for prm in self.inner.parameters():
                if prm.is_floating_point() and prm.numel():
                    prm.requires_grad_(True)
        elif fmt == "module":
            if not c.allow_pickle:
                raise ConfigError("format 'module' lädt Pickle (führt Code aus) – nur mit model.allow_pickle = true.")
            obj = torch.load(str(path), map_location="cpu", weights_only=False)
            if not isinstance(obj, nn.Module):
                raise ConfigError(f"'{path}' enthält kein nn.Module, sondern {type(obj).__name__}.")
            self.inner = obj
        elif fmt in ("state_dict", "safetensors"):
            if c.architecture is None:
                raise ConfigError("Für reine Gewichte wird model.architecture (Baukasten-Modell gleicher Form) gebraucht.",
                                  "model.architecture")
            self.inner = c.architecture.build()
            if c.path:
                state = _remap(_load_state(path, fmt, c.allow_pickle), c.key_map)
                missing, unexpected = self.inner.load_state_dict(state, strict=False)
                self.import_report.update(missing=list(missing), unexpected=list(unexpected), loaded=len(state))
                if c.strict and (missing or unexpected):
                    raise ConfigError(f"Gewichte passen nicht: fehlend {list(missing)[:5]}, überzählig "
                                      f"{list(unexpected)[:5]} – key_map anpassen oder strict = false.")
        elif fmt == "huggingface":
            self.inner = _load_hf(c.path, c.hf_task)
        else:
            raise ConfigError(f"Unbekanntes Format '{fmt}'.")
        try:
            params = inspect.signature(self.inner.forward).parameters
            self._inner_kwargs = {n for n in params} | ({"*"} if any(
                p.kind == p.VAR_KEYWORD for p in params.values()) else set())
        except (TypeError, ValueError):
            self._inner_kwargs = {"*"}
        # Kein finish(): die importierten Gewichte dürfen nicht neu initialisiert werden.

    def forward(self, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        allowed = self._inner_kwargs
        kwargs = {k: v for k, v in kwargs.items() if "*" in allowed or k in allowed}
        out = self.inner(inputs, *args, **kwargs)
        key = self.config.output_key
        if key is not None:
            return out[key] if isinstance(out, dict) else getattr(out, key)
        if hasattr(out, "logits") and not isinstance(out, torch.Tensor):
            return out.logits  # Hugging Face
        return out

    def train(self, mode: bool = True) -> "ExternalModel":
        """Exportierte Graphen (torch.export) haben einen festen Modus – dann nur die Hülle umschalten."""
        self.training = mode
        try:
            self.inner.train(mode)
        except NotImplementedError:
            pass
        return self

    def depth(self) -> int:
        return max(1, len(self.checkpointable_blocks()))

    def checkpointable_blocks(self) -> list[nn.Module]:
        if hasattr(self.inner, "checkpointable_blocks"):
            return list(self.inner.checkpointable_blocks())
        lists = [m for m in self.inner.modules() if isinstance(m, nn.ModuleList) and len(m) > 1]
        return list(max(lists, key=len)) if lists else []

    def example_input(self, batch_size: int = 2) -> Any:
        if hasattr(self.inner, "example_input") and self.import_format in ("state_dict", "safetensors"):
            return self.inner.example_input(batch_size)
        c = self.config
        shape = [batch_size] + list(c.input_shape[1:])
        if c.input_dtype == "long":
            return torch.randint(0, c.input_high, shape)
        return torch.randn(*shape)


MODELS.register("extern", ExternalModel, config=ExternalModelConfig, aliases=("import", "external"),
                level=Level.ADVANCED, tags=("import",), tooltip=T(
    "Importiertes Modell (torch.export, TorchScript, state_dict, safetensors, Hugging Face).",
    "Fremde KIs mit allen Baukasten-Werkzeugen ändern, nachtrainieren, komprimieren und neu exportieren.",
    "Weiterverwendung vorhandener Modelle."))
