"""
ki_baukasten.data.artifacts
===========================

**Vorverarbeitung mitspeichern** – damit ein gespeichertes Modell später *ohne* die Trainingsdaten
benutzt werden kann (``Project.save`` → ``vorverarbeitung.json``, ``Project.load``):

* Tokenizer (Zeichen/Wort/BPE samt gelernter Merges)
* Tabellen-Vorverarbeitung (Spalten, Kategorien, Ergänzungswerte, Skalen)
* Klassennamen, Ziel-Skalierung, Bild-Normierung, Audio-Merkmale, Vorlage für Anweisungen

``prepare_raw`` wandelt Rohdaten (Tabellenzeilen, Texte, Bild-/WAV-Dateien) in Modelleingaben um –
genau wie im Training. Alles ist reines JSON (keine ausführbaren Objekte).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_PLAIN = ("label_names", "features", "task", "target_transform", "prompt_template", "eos_token_id", "eos_text",
          "seq_len", "input_size", "in_channels", "sample_rate", "audio_features", "duration", "audio", "normalize",
          "vocab_size", "output_dim", "input_dim")


def export_artifacts(info: dict[str, Any]) -> dict[str, Any]:
    out = {k: info[k] for k in _PLAIN if k in info}
    if info.get("tokenizer") is not None and hasattr(info["tokenizer"], "to_dict"):
        out["tokenizer"] = info["tokenizer"].to_dict()
    if info.get("preprocessor") is not None and hasattr(info["preprocessor"], "to_dict"):
        out["preprocessor"] = info["preprocessor"].to_dict()
    return out


def import_artifacts(d: dict[str, Any]) -> dict[str, Any]:
    from .files import TabularPreprocessor
    from .tokenizers import tokenizer_from_dict

    out = dict(d)
    if isinstance(d.get("tokenizer"), dict):
        out["tokenizer"] = tokenizer_from_dict(d["tokenizer"])
    if isinstance(d.get("preprocessor"), dict):
        out["preprocessor"] = TabularPreprocessor.from_dict(d["preprocessor"])
    return out


def _source(item: Any) -> Any:
    """Dateipfad oder Datei-Inhalt (``bytes``, z. B. aus einer HTTP-Anfrage)."""
    import io

    return io.BytesIO(item) if isinstance(item, (bytes, bytearray)) else Path(item)


def kind_of(info: dict[str, Any]) -> str:
    """Welche Rohdaten passen: tabelle | text | sprachmodell | bild | audio | anweisung | tensor."""
    if info.get("preprocessor") is not None:
        return "tabelle"
    if info.get("prompt_template"):
        return "anweisung"
    if info.get("tokenizer") is not None:
        return "sprachmodell" if info.get("task") == "language_modeling" else "text"
    if info.get("sample_rate"):
        return "audio"
    if info.get("normalize"):
        return "bild"
    return "tensor"


def prepare_raw(info: dict[str, Any], records: Any) -> Any:
    """Rohdaten → Modelleingaben (Tensor bzw. dict mit ``inputs``/``attention_mask``)."""
    if isinstance(records, torch.Tensor):
        return records
    if isinstance(records, (str, dict)):
        records = [records]
    kind = kind_of(info)
    if kind == "tabelle":
        prep = info["preprocessor"]
        cols = prep.input_columns
        rows = []
        for r in records:
            if isinstance(r, dict):
                missing = [c for c in cols if c not in r]
                if len(missing) == len(cols):
                    raise ValueError(f"Keine der erwarteten Spalten gefunden. Erwartet: {', '.join(cols)}.")
                rows.append(["" if r.get(c) is None else str(r.get(c)) for c in cols])
            else:
                rows.append([str(v) for v in r])
        return prep.transform(rows)
    if kind == "text":
        from .tokenizers import PAD

        tok, length = info["tokenizer"], int(info.get("seq_len", 64))
        ids = torch.full((len(records), length), PAD, dtype=torch.long)
        for i, text in enumerate(records):
            enc = tok.encode(str(text))[:length]
            if enc:
                ids[i, :len(enc)] = torch.tensor(enc)
        return {"inputs": ids, "attention_mask": ids != PAD}
    if kind == "bild":
        from .files import _load_image

        size, ch = info["input_size"], int(info.get("in_channels", 3))
        x = torch.stack([_load_image(_source(p), size, ch) for p in records])
        mean = torch.tensor(info["normalize"]["mean"]).view(1, -1, 1, 1)
        std = torch.tensor(info["normalize"]["std"]).view(1, -1, 1, 1)
        return (x - mean) / std
    if kind == "audio":
        import torch.nn.functional as F

        from .domains import log_mel_spectrogram, read_wav

        sr = int(info["sample_rate"])
        n = int(sr * float(info.get("duration", 1.0)))
        waves = []
        for p in records:
            w = read_wav(_source(p), sr)[:n]
            waves.append(F.pad(w, (0, n - len(w))))
        x = torch.stack(waves)
        if info.get("audio_features") == "spektrogramm":
            a = info["audio"]
            x = (log_mel_spectrogram(x, sr, a["n_fft"], a["hop"], a["n_mels"]) - a["mean"]) / a["std"]
        return x.unsqueeze(1)
    raise ValueError("Für diese Daten gibt es keine gespeicherte Vorverarbeitung – Tensoren übergeben.")
