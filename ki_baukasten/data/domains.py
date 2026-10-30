"""
ki_baukasten.data.domains
=========================

Datensätze für weitere Anwendungsgebiete – jeweils mit eingebautem Beispiel und eigenen Dateien:

======================  ===========================================================================
Registry-Name           Inhalt
======================  ===========================================================================
segmentation            Bilder + Masken (eine Klasse je Pixel): synthetische Formen oder eigene
                        Ordner ``bilder/`` + ``masken/`` (gleiche Dateinamen)
audio                   Audio-Klassifikation: WAV-Dateien aus ``ordner/<klasse>/*.wav`` (ohne
                        Zusatzpakete gelesen) oder synthetische Töne; als Log-Mel-Spektrogramm
                        (wie Spracherkennung) oder Wellenform
instructions            Instruction-Tuning (SFT): Frage → Antwort aus JSONL/CSV oder Beispielaufgaben;
                        der Verlust zählt nur auf den **Antworten** (``loss_mask``)
======================  ===========================================================================
"""

from __future__ import annotations

import csv
import json
import math
import random
import wave
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..core import BaseConfig, Level, Param, T
from .files import split_indices
from .tokenizers import KINDS as TOKENIZERS, PAD, make_tokenizer

EOS = "␄"          # Ende-Zeichen der Antworten (wird vom Tokenizer mitgelernt)


# ===========================================================================
# Segmentierung
# ===========================================================================

class SegmentationDataConfig(BaseConfig):
    __title__ = "Segmentierung"
    images = Param(None, type=str, nullable=True, tooltip=T(
        "Ordner mit Bildern (None = synthetische Formen).", "Eigene Segmentierungsdaten.", "Pfad."))
    masks = Param(None, type=str, nullable=True, tooltip=T(
        "Ordner mit Masken gleichen Namens (Graustufen: Pixelwert = Klasse; 255 wird als 1 gelesen).",
        "Zielklasse je Pixel.", "Pfad."))
    size = Param([32, 32], item_type=int, length=2, min=8, tooltip=T(
        "Bildgröße.", "Größer = feinere Details, mehr Rechenzeit.", "32–256."))
    n_samples = Param(600, min=10, tooltip=T("Anzahl synthetischer Bilder.", "Nur ohne eigene Daten.", "600."))
    noise = Param(0.15, min=0.0, tooltip=T("Bildrauschen (synthetisch).", "Schwierigkeit.", "0.1–0.3."))
    val_fraction = Param(0.15, min=0.0, max=0.9, tooltip=T("Validierungsanteil.", "Ehrliche Bewertung.", "0.15."))
    seed = Param(0, min=0, tooltip=T("Seed.", "Reproduzierbar.", "0."), level=Level.ADVANCED)


def _synthetic_segmentation(a: SegmentationDataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(a.seed)
    h, w = a.size
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    imgs = torch.zeros(a.n_samples, 1, h, w)
    masks = torch.zeros(a.n_samples, h, w, dtype=torch.long)
    for i in range(a.n_samples):
        for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
            kind = int(torch.randint(1, 3, (1,), generator=g))          # 1 = Kreis, 2 = Quadrat
            r = float(torch.empty(1).uniform_(min(h, w) * 0.1, min(h, w) * 0.22, generator=g))
            cy, cx = (float(torch.empty(1).uniform_(r, s - r, generator=g)) for s in (h, w))
            region = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) if kind == 1 else \
                ((yy - cy).abs() <= r * 0.85) & ((xx - cx).abs() <= r * 0.85)
            masks[i][region] = kind
            imgs[i, 0][region] = 0.5 + 0.5 * (kind == 1)
    imgs += a.noise * torch.randn(imgs.shape, generator=g)
    return imgs, masks


def _read_gray(path: Path, size: list[int], channels: int) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path).convert("L" if channels == 1 else "RGB").resize((size[1], size[0]))
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).clone()
    return t.reshape(size[0], size[1], channels).permute(2, 0, 1)


def load_segmentation(a: SegmentationDataConfig) -> Any:
    from .datasets import DatasetBundle

    if a.images:
        if not a.masks:
            raise FileNotFoundError("segmentation: 'masks' (Ordner mit Masken) fehlt.")
        files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        imgs, masks = [], []
        for f in files:
            m = next((Path(a.masks) / (f.stem + ext) for ext in (".png", ".bmp", f.suffix)
                      if (Path(a.masks) / (f.stem + ext)).exists()), None)
            if m is None:
                continue
            imgs.append(_read_gray(f, a.size, 3).float() / 255)
            raw = _read_gray(m, a.size, 1)[0].long()
            masks.append(torch.where(raw == 255, torch.ones_like(raw), raw))
        if not imgs:
            raise FileNotFoundError(f"Keine Bild/Masken-Paare in {a.images} und {a.masks} gefunden.")
        x, y = torch.stack(imgs), torch.stack(masks)
    else:
        x, y = _synthetic_segmentation(a)
    train_idx, val_idx = split_indices(len(x), a.val_fraction, "zufall", None, a.seed)
    n_classes = int(y.max()) + 1
    info = {"in_channels": x.shape[1], "input_size": list(a.size), "output_dim": max(2, n_classes),
            "task": "segmentation", "label_names": ["hintergrund", "kreis", "quadrat"][:n_classes]
            if not a.images else [str(k) for k in range(n_classes)]}
    pick = (lambda idx: {"inputs": x[idx], "targets": y[idx]})
    return DatasetBundle(pick(train_idx), pick(val_idx) if val_idx else None, info)


# ===========================================================================
# Audio
# ===========================================================================

def mel_filterbank(n_freqs: int, n_mels: int, sample_rate: int, f_min: float = 0.0,
                   f_max: float | None = None) -> torch.Tensor:
    """Dreieckige Mel-Filter ``[n_mels, n_freqs]`` (HTK-Formel)."""
    f_max = f_max or sample_rate / 2
    mel = (lambda f: 2595 * math.log10(1 + f / 700))
    hz = (lambda m: 700 * (10 ** (m / 2595) - 1))
    points = torch.tensor([hz(m) for m in torch.linspace(mel(f_min), mel(f_max), n_mels + 2).tolist()])
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    lo, mid, hi = points[:-2, None], points[1:-1, None], points[2:, None]
    up = (freqs - lo) / (mid - lo).clamp_min(1e-6)
    down = (hi - freqs) / (hi - mid).clamp_min(1e-6)
    return torch.clamp(torch.minimum(up, down), min=0)


def log_mel_spectrogram(wave_: torch.Tensor, sample_rate: int, n_fft: int = 256, hop: int = 128,
                        n_mels: int = 40) -> torch.Tensor:
    """Log-Mel-Spektrogramm ``[..., n_mels, frames]`` – Standard-Merkmal für Sprache/Geräusche."""
    spec = torch.stft(wave_.reshape(-1, wave_.shape[-1]).float(), n_fft, hop, window=torch.hann_window(n_fft),
                      return_complex=True).abs().square()
    mel = mel_filterbank(spec.shape[-2], n_mels, sample_rate) @ spec
    return (mel + 1e-6).log().reshape(*wave_.shape[:-1], n_mels, -1)


def read_wav(path: Any, sample_rate: int) -> torch.Tensor:
    """16-Bit-PCM-WAV (mono/stereo; Pfad oder Datei-Objekt) lesen und auf ``sample_rate`` umrechnen."""
    with wave.open(path if hasattr(path, "read") else str(path), "rb") as f:
        sr, ch, width, n = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
        raw = f.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: nur 16-Bit-PCM wird unterstützt (gefunden: {8 * width} Bit).")
    x = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768
    x = x.reshape(-1, ch).mean(1)
    if sr != sample_rate:
        x = F.interpolate(x[None, None], size=max(1, round(len(x) * sample_rate / sr)), mode="linear",
                          align_corners=False)[0, 0]
    return x


class AudioDataConfig(BaseConfig):
    __title__ = "Audio-Klassifikation"
    path = Param(None, type=str, nullable=True, tooltip=T(
        "Ordner mit Unterordnern je Klasse voller WAV-Dateien (None = synthetische Töne/Geräusche).",
        "Eigene Geräusch-/Sprachbefehl-Daten.", "Pfad."))
    sample_rate = Param(8000, min=1000, tooltip=T(
        "Abtastrate in Hz (Dateien werden umgerechnet).", "Sprache: 16000; einfache Geräusche: 8000.", "8000–16000."))
    duration = Param(0.5, min=0.05, tooltip=T(
        "Länge je Beispiel in Sekunden (kürzen/auffüllen).", "Einheitliche Eingabegröße.", "0.5–2."))
    features = Param("spektrogramm", choices={
        "spektrogramm": "Log-Mel-Spektrogramm [1, Mel, Zeit] → 2D-CNN (Standard für Sprache/Geräusche)",
        "wellenform": "rohe Wellenform [1, Samples] → 1D-CNN"}, tooltip=T(
        "Eingabedarstellung.", "Spektrogramme machen Frequenzen sichtbar – meist besser und kleiner.",
        "spektrogramm."))
    n_mels = Param(40, min=8, tooltip=T("Mel-Bänder.", "Frequenzauflösung.", "40–128."), level=Level.ADVANCED)
    n_fft = Param(256, min=16, tooltip=T("FFT-Fenster.", "Zeit- vs. Frequenzauflösung.", "256–1024."),
                  level=Level.ADVANCED)
    hop = Param(128, min=1, tooltip=T("Fenster-Schritt.", "Zeitauflösung.", "n_fft/2."), level=Level.ADVANCED)
    n_samples = Param(800, min=10, tooltip=T("Anzahl synthetischer Beispiele.", "Nur ohne path.", "800."))
    noise = Param(0.1, min=0.0, tooltip=T("Hintergrundrauschen (synthetisch).", "Schwierigkeit.", "0.05–0.3."))
    val_fraction = Param(0.15, min=0.0, max=0.9, tooltip=T("Validierungsanteil.", "Ehrliche Bewertung.", "0.15."))
    seed = Param(0, min=0, tooltip=T("Seed.", "Reproduzierbar.", "0."), level=Level.ADVANCED)


_SOUNDS = ["ton_tief", "ton_hoch", "zirpen", "pochen"]


def _synthetic_audio(a: AudioDataConfig) -> tuple[torch.Tensor, list[int]]:
    g = torch.Generator().manual_seed(a.seed)
    n = int(a.sample_rate * a.duration)
    t = torch.arange(n) / a.sample_rate
    waves, labels = [], []
    for i in range(a.n_samples):
        k = i % len(_SOUNDS)
        jitter = float(torch.empty(1).uniform_(0.9, 1.1, generator=g))
        phase = float(torch.rand(1, generator=g)) * 6.28
        if k == 0:
            x = torch.sin(2 * math.pi * 300 * jitter * t + phase)
        elif k == 1:
            x = torch.sin(2 * math.pi * 1500 * jitter * t + phase)
        elif k == 2:
            x = torch.sin(2 * math.pi * (300 + 2500 * t / a.duration) * jitter * t + phase)
        else:
            x = torch.zeros(n)
            hit = min(200, n)
            for start in torch.randint(0, max(1, n - hit), (3,), generator=g).tolist():
                x[start:start + hit] += torch.randn(hit, generator=g) * torch.linspace(1, 0, hit)
        amp = float(torch.empty(1).uniform_(0.3, 1.0, generator=g))
        waves.append(amp * x + a.noise * torch.randn(n, generator=g))
        labels.append(k)
    return torch.stack(waves), labels


def load_audio(a: AudioDataConfig) -> Any:
    from .datasets import DatasetBundle

    n = int(a.sample_rate * a.duration)
    if a.path:
        classes = sorted(d.name for d in Path(a.path).iterdir() if d.is_dir())
        waves, labels = [], []
        for k, c in enumerate(classes):
            for f in sorted((Path(a.path) / c).glob("*.wav")):
                x = read_wav(f, a.sample_rate)[:n]
                waves.append(F.pad(x, (0, n - len(x))))
                labels.append(k)
        if not waves:
            raise FileNotFoundError(f"Keine WAV-Dateien in {a.path}/<klasse>/ gefunden.")
        x = torch.stack(waves)
    else:
        classes = list(_SOUNDS)
        x, labels = _synthetic_audio(a)
    y = torch.tensor(labels)
    train_idx, val_idx = split_indices(len(x), a.val_fraction, "stratifiziert", labels, a.seed)
    if a.features == "spektrogramm":
        x = log_mel_spectrogram(x, a.sample_rate, a.n_fft, a.hop, a.n_mels).unsqueeze(1)
        m, s = x[train_idx].mean(), x[train_idx].std().clamp_min(1e-6)
        x = (x - m) / s
        info = {"in_channels": 1, "input_size": list(x.shape[-2:]),
                "audio": {"mean": float(m), "std": float(s), "n_fft": a.n_fft, "hop": a.hop, "n_mels": a.n_mels}}
    else:
        x = x.unsqueeze(1)
        info = {"in_channels": 1, "input_size": [n]}
    info.update({"output_dim": len(classes), "label_names": classes, "task": "classification",
                 "sample_rate": a.sample_rate, "audio_features": a.features, "duration": a.duration})
    pick = (lambda idx: {"inputs": x[idx], "targets": y[idx]})
    return DatasetBundle(pick(train_idx), pick(val_idx) if val_idx else None, info)


# ===========================================================================
# Instruction-Tuning (SFT)
# ===========================================================================

_OPPOSITES = {"groß": "klein", "hell": "dunkel", "warm": "kalt", "schnell": "langsam", "laut": "leise", "alt": "jung",
              "oben": "unten", "voll": "leer", "hart": "weich", "nah": "fern", "früh": "spät", "stark": "schwach"}
_WORDS = ["Haus", "Baum", "Katze", "Wasser", "Sonne", "Buch", "Licht", "Musik", "Brot", "Stein", "Wolke", "Fluss"]


def _demo_instructions(n: int, seed: int) -> list[tuple[str, str]]:
    """Eindeutige Beispielaufgaben (keine Wiederholungen → Training und Validierung überschneiden sich nicht)."""
    pairs = [(f"Was ist {a} plus {b}?", f"{a + b}.") for a in range(30) for b in range(30)]
    pairs += [(f"Was ist {a} minus {b}?", f"{a - b}.") for a in range(30) for b in range(a + 1)]
    pairs += [(f"Nachfolger von {k}?", f"{k + 1}.") for k in range(200)]
    pairs += [(f"Gegenteil von {w}?", f"{v}.") for w, v in _OPPOSITES.items()]
    pairs += [(f"Gegenteil von {v}?", f"{w}.") for w, v in _OPPOSITES.items()]
    pairs += [(f"Wiederhole: {w}", w) for w in _WORDS]
    pairs += [(f"Erster Buchstabe von {w}?", f"{w[0]}.") for w in _WORDS]
    pairs += [(f"Zähle bis {k}.", " ".join(str(i) for i in range(1, k + 1)) + ".") for k in range(1, 10)]
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


class InstructionDataConfig(BaseConfig):
    __title__ = "Anweisungen (Instruction-Tuning)"
    path = Param(None, type=str, nullable=True, tooltip=T(
        "JSONL (eine Zeile {'prompt': …, 'response': …}) oder CSV mit Spalten prompt/response "
        "(None = eingebaute Beispielaufgaben).", "Eigene Frage-Antwort-/Chat-Daten.", "Pfad."))
    prompt_key = Param("prompt", tooltip=T("Feld/Spalte der Anweisung.", "Eingabe.", "prompt (auch: instruction)."))
    response_key = Param("response", tooltip=T("Feld/Spalte der Antwort.", "Ziel.", "response (auch: output)."))
    template = Param("Frage: {prompt}\nAntwort: ", tooltip=T(
        "Vorlage vor der Antwort ({prompt} wird ersetzt).", "Das Modell lernt das Format – bei Benutzung "
        "(Project.ask) wird dieselbe Vorlage verwendet.", "Chat-Formate: '<|user|>{prompt}<|assistant|>'."))
    train_on_prompt = Param(False, tooltip=T(
        "Verlust auch auf der Anweisung (aus = nur auf der Antwort, Standard beim SFT).",
        "Das Modell soll antworten lernen, nicht Fragen nachplappern.", "aus."))
    tokenizer = Param("zeichen", choices=TOKENIZERS, tooltip=T(
        "Tokenizer.", "zeichen: sicher bei kleinen Daten; bpe: kompakter bei viel Text.", "zeichen/bpe."))
    vocab_size = Param(1000, min=8, tooltip=T("Vokabular (wort/bpe).", "Größe.", "500–30.000."), level=Level.ADVANCED)
    seq_len = Param(64, min=8, tooltip=T(
        "Maximale Länge (Anweisung + Antwort).", "Längere Beispiele werden gekürzt.", "64–2048."))
    n_samples = Param(2000, min=10, tooltip=T(
        "Anzahl Beispielaufgaben (höchstens so viele, wie es verschiedene gibt: ≈ 1.600).", "Nur ohne path.", "2000."))
    val_fraction = Param(0.1, min=0.0, max=0.9, tooltip=T("Validierungsanteil.", "Ehrliche Bewertung.", "0.1."))
    seed = Param(0, min=0, tooltip=T("Seed.", "Reproduzierbar.", "0."), level=Level.ADVANCED)


def _read_pairs(a: InstructionDataConfig) -> list[tuple[str, str]]:
    path = Path(a.path)
    if path.suffix.lower() in (".jsonl", ".json"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    keys_p = (a.prompt_key, "instruction", "frage", "input")
    keys_r = (a.response_key, "output", "antwort", "completion")
    out = []
    for r in rows:
        p = next((r[k] for k in keys_p if k in r and r[k]), None)
        s = next((r[k] for k in keys_r if k in r and r[k] is not None), None)
        if p is not None and s is not None:
            out.append((str(p), str(s)))
    if not out:
        raise ValueError(f"{path}: keine Paare mit '{a.prompt_key}'/'{a.response_key}' gefunden.")
    return out


def load_instructions(a: InstructionDataConfig) -> Any:
    from .datasets import DatasetBundle

    pairs = _read_pairs(a) if a.path else _demo_instructions(a.n_samples, a.seed)
    train_idx, val_idx = split_indices(len(pairs), a.val_fraction, "zufall", None, a.seed)
    texts = [a.template.format(prompt=p) + r + EOS for p, r in pairs]
    tok = make_tokenizer(a.tokenizer, "\n".join(texts[i] for i in train_idx) + EOS, a.vocab_size)

    def encode(idx: list[int]) -> dict[str, torch.Tensor]:
        ids = torch.full((len(idx), a.seq_len), PAD, dtype=torch.long)
        loss = torch.zeros(len(idx), a.seq_len, dtype=torch.bool)
        for row, i in enumerate(idx):
            p, r = pairs[i]
            head, tail = tok.encode(a.template.format(prompt=p)), tok.encode(r + EOS)
            seq = (head + tail)[: a.seq_len]
            ids[row, : len(seq)] = torch.tensor(seq)
            loss[row, (0 if a.train_on_prompt else len(head)): len(seq)] = True
        return {"inputs": ids, "attention_mask": ids != PAD, "loss_mask": loss}

    eos_ids = tok.encode(EOS)
    info = {"tokenizer": tok, "vocab_size": tok.vocab_size, "seq_len": a.seq_len, "task": "language_modeling",
            "prompt_template": a.template, "eos_token_id": eos_ids[-1] if eos_ids else None, "eos_text": EOS}
    return DatasetBundle(encode(train_idx), encode(val_idx) if val_idx else None, info)


__all__ = ["SegmentationDataConfig", "AudioDataConfig", "InstructionDataConfig", "load_segmentation", "load_audio",
           "load_instructions", "log_mel_spectrogram", "mel_filterbank", "read_wav", "EOS"]
