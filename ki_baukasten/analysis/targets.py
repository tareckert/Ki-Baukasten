"""
ki_baukasten.analysis.targets
=============================

**Zielsysteme**: Auf welchem Gerät soll die KI laufen – und was muss dafür passieren?

Jedes Profil beschreibt grob, wie viel Speicher ein Modell dort belegen darf, wie viel
Rechenleistung bei kleinen Batches realistisch ankommt, welche Datentypen schnell sind
und welches Exportformat dort üblich ist. Daraus berechnet ``recommend`` ein
konkretes **Reduktions-Rezept** (Kompressions-Schritte, ggf. Schüler-Modell, Export).

Die Zahlen sind bewusst **grobe Richtwerte** (effektive Leistung bei Batch 1, nicht
Datenblatt-Spitzen) – sie sollen die Größenordnung treffen, nicht das Benchmarking ersetzen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from ..core import BaseConfig, Level, Param, T, format_number


@dataclass(frozen=True)
class TargetProfile:
    key: str
    title: str
    memory_mb: float          # Speicher, den das Modell (Gewichte + Laufzeit) belegen darf
    gflops: float             # effektive Rechenleistung bei Batch 1 (GFLOP/s)
    dtypes: tuple[str, ...]   # schnelle/unterstützte Gewichtsformate, bevorzugtes zuerst
    export: str               # übliches Exportformat
    latency_ms: float         # angestrebte Latenz pro Eingabe
    notes: str = ""


TARGETS: dict[str, TargetProfile] = {p.key: p for p in [
    TargetProfile("rechenzentrum_gpu", "Rechenzentrum-GPU (A100/H100, 40–80 GB)", 60000, 150000,
                  ("bfloat16", "int8", "int4"), "torch.export → AOTInductor / TensorRT", 50,
                  "bf16 + torch.compile; Kompression meist nur für Kosten/Durchsatz nötig."),
    TargetProfile("gaming_gpu", "Gaming-/Workstation-GPU (8–24 GB)", 8000, 40000,
                  ("float16", "bfloat16", "int8", "int4"), "torch.export / ONNX → TensorRT", 100,
                  "Große Sprachmodelle in int4/nf4; KV-Cache int8."),
    TargetProfile("laptop_gpu", "Laptop-/Einsteiger-GPU (4–6 GB)", 3000, 10000,
                  ("float16", "int8", "int4"), "torch.export / ONNX", 150, "fp16 oder int8; Batch klein halten."),
    TargetProfile("apple_silicon", "Apple Silicon (M1–M4, gemeinsamer Speicher)", 6000, 5000,
                  ("float16", "int8", "pal4"), "torch.export → CoreML / ExecuTorch (MPS)", 100,
                  "Palettisierung (pal4) ist auf Apple-Hardware besonders effizient."),
    TargetProfile("server_cpu", "Server-CPU (x86, viele Kerne)", 16000, 1000,
                  ("int8", "bfloat16"), "ONNX (onnxruntime) / torch.export", 200,
                  "int8 + mehrere Threads; Durchsatz über Batching."),
    TargetProfile("laptop_cpu", "Laptop-/Desktop-CPU", 4000, 200,
                  ("int8",), "ONNX (onnxruntime) / torch.export", 300, "int8, wenige Threads, kleine Modelle bevorzugen."),
    TargetProfile("smartphone", "Smartphone (ARM, 6–12 GB RAM, Akku)", 1000, 100,
                  ("int8", "int4", "float16"), "ExecuTorch (.pte aus torch.export) / ONNX Mobile / CoreML", 100,
                  "Energie: Rechenaufwand zählt doppelt; Modell < 1 GB, besser < 300 MB."),
    TargetProfile("raspberry_pi", "Raspberry Pi 4/5 (ARM-CPU, 4–8 GB)", 500, 15,
                  ("int8",), "ONNX (onnxruntime) / ExecuTorch", 500, "Kleine Schüler-Modelle + int8."),
    TargetProfile("mikrocontroller", "Mikrocontroller (Cortex-M, ≤ 1 MB Flash)", 0.5, 0.1,
                  ("int8",), "ONNX → TFLite Micro / ExecuTorch für MCUs", 1000,
                  "Nur winzige Modelle (< 500 KB): Schüler + int8 sind Pflicht."),
    TargetProfile("browser", "Browser (WebGPU/WebAssembly)", 200, 50,
                  ("int8", "float16"), "ONNX (onnxruntime-web, WebGPU)", 200,
                  "Download-Größe zählt: < 100 MB anstreben."),
]}

class DeploymentConfig(BaseConfig):
    __title__ = "Zielsysteme (Deployment)"
    __tooltip__ = T(
        "Auf welchen Geräten die KI laufen soll – für Reduktions-Empfehlungen nach dem Training.",
        "Zeigt sofort, ob das Modell auf Smartphone, Raspberry Pi, CPU oder GPU passt und wie man es dorthin bringt.",
        "Planung des Einsatzes; Project.recommend_for / reduce_for.",
    )
    targets = Param(["gaming_gpu", "laptop_cpu", "smartphone", "raspberry_pi"], item_type=str, tooltip=T(
        "Zielsysteme für die Übersicht nach dem Training.",
        "Vergleich mehrerer Geräteklassen auf einen Blick.",
        "Verfügbar: rechenzentrum_gpu, gaming_gpu, laptop_gpu, apple_silicon, server_cpu, laptop_cpu, smartphone, "
        "raspberry_pi, mikrocontroller, browser.",
    ))
    show_after_training = Param(True, tooltip=T(
        "Nach train() die Reduktions-Übersicht anzeigen.", "Man sieht beim Fertigstellen sofort die Optionen.", "an."))
    main_target = Param(None, type=str, nullable=True, tooltip=T(
        "Hauptziel: nach dem Training zusätzlich die ausführliche Empfehlung zeigen.",
        "Detailliertes Rezept + Exportformat für genau dieses Gerät.", "z. B. 'smartphone'."))
    heal_steps = Param(300, min=0, tooltip=T(
        "Heil-Schritte (Distillation vom Original) in reduce_for.", "Mehr = bessere Qualität nach dem Verkleinern.",
        "100–1000."), level=Level.ADVANCED)
    student_steps = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Trainingsschritte eines empfohlenen Schüler-Modells (None = wie training).", "Distillation.", "None."),
        level=Level.ADVANCED)

    def check(self) -> list[str]:
        bad = [t for t in self.targets + ([self.main_target] if self.main_target else []) if t not in TARGETS]
        return [f"Unbekannte Zielsysteme: {bad}. Verfügbar: {', '.join(TARGETS)}."] if bad else []


_BYTES = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "int8": 1.0, "int4": 0.5, "nf4": 0.5, "pal4": 0.5}


@dataclass
class Recommendation:
    target: TargetProfile
    params: int
    size_now_mb: float
    macs: int | None
    latency_now_ms: float | None
    fits_now: bool
    steps: list[dict] = field(default_factory=list)
    student: dict | None = None        # {"width": .., "depth": ..} falls ein Schüler-Modell nötig ist
    size_after_mb: float = 0.0
    latency_after_ms: float | None = None
    fits_after: bool = True
    notes: list[str] = field(default_factory=list)

    def short(self) -> str:
        parts = [s["name"] + (f"({next(iter(s.get('args', {}).values()))})" if s.get("args") else "")
                 for s in self.steps]
        if self.student:
            parts.insert(0, f"Schüler(Breite {self.student['width']:.2f}, Tiefe {self.student['depth']:.2f})")
        return " → ".join(parts) or "nichts nötig"

    def __str__(self) -> str:
        t = self.target
        lat = (lambda v: f"{v:.1f} ms" if v is not None else "?")
        lines = [f"Zielsystem: {t.title}",
                 f"  Jetzt:      {format_number(self.params)} Parameter, {format_number(self.size_now_mb, 1)} MB, "
                 f"≈ {lat(self.latency_now_ms)} pro Eingabe (Budget {format_number(t.memory_mb, 1)} MB, "
                 f"Ziel {t.latency_ms:.0f} ms) → {'passt ✓' if self.fits_now else 'passt nicht ✗'}",
                 f"  Empfehlung: {self.short()}",
                 f"  Danach:     ≈ {format_number(self.size_after_mb, 1)} MB, ≈ {lat(self.latency_after_ms)} → "
                 f"{'passt ✓' if self.fits_after else 'weiterhin zu groß/langsam – kleineres Grundmodell wählen'}",
                 f"  Export:     {t.export}"]
        for n in self.notes + ([t.notes] if t.notes else []):
            lines.append(f"  Hinweis:    {n}")
        lines.append(f"  Anwenden:   projekt.reduce_for(\"{t.key}\")")
        return "\n".join(lines)


def _has(model: nn.Module, cls_name: str) -> bool:
    return any(type(m).__name__ == cls_name for m in model.modules())


def recommend(model: nn.Module, target: str | TargetProfile, macs: int | None = None, can_heal: bool = True,
              size_bytes: int | None = None) -> Recommendation:
    """Reduktions-Empfehlung für ``model`` auf ``target`` (Profil-Schlüssel oder ``TargetProfile``)."""
    from .benchmark import model_size_bytes

    t = TARGETS[target] if isinstance(target, str) else target
    if isinstance(target, str) and target not in TARGETS:
        raise KeyError(f"Unbekanntes Zielsystem '{target}'. Verfügbar: {', '.join(TARGETS)}.")
    params = sum(p.numel() for p in model.parameters())
    params += sum(m.in_features * m.out_features for m in model.modules() if type(m).__name__ == "QuantLinear")
    size_mb = (size_bytes if size_bytes is not None else model_size_bytes(model)) / 2**20
    lat = (2 * macs / (t.gflops * 1e9) * 1000) if macs else None
    fits_now = size_mb <= t.memory_mb and (lat is None or lat <= t.latency_ms)
    rec = Recommendation(t, params, size_mb, macs, lat, fits_now)

    speed_factor = (lat / t.latency_ms) if lat else 1.0
    steps: list[dict] = []
    transformer_like = _has(model, "TransformerBlock")
    has_attention = _has(model, "MultiHeadAttention")
    is_mlp = type(model).__name__ == "MLP"

    # --- Wie weit kommt man allein mit dem kleinsten Format des Ziels? -------------------------------
    min_bytes = min(_BYTES[d] for d in t.dtypes)
    rest_factor = params * min_bytes / 2**20 / t.memory_mb       # > 1: Präzision allein reicht nicht

    # --- Schüler-Modell, wenn selbst ~60 % Pruning nicht reichen ------------------------------------
    if rest_factor > 2.5 or speed_factor > 5:
        need = max(rest_factor / 2.5, speed_factor / 5, 1.0)
        shrink = min(0.9, max(0.15, 1 / math.sqrt(need)))
        rec.student = {"width": round(shrink, 2), "depth": round(min(1.0, shrink * 1.2), 2)}
        rec.notes.append("Ein kleineres Schüler-Modell per Distillation (Project.distill) ist der wirksamste Hebel.")
    keep = (rec.student["width"] ** 2 * rec.student["depth"]) if rec.student else 1.0
    params_after = params * keep
    speed_left, rest_left = speed_factor * keep, rest_factor * keep

    # --- Struktur (Rechenaufwand) ---------------------------------------------------------------
    if speed_left > 1.2 or rest_left > 1.0:
        structural = min(0.6, max(0.2, 1 - 1 / max(speed_left, rest_left, 1.25)))
        if has_attention:
            steps.append({"name": "prune_heads", "args": {"amount": round(min(0.5, structural), 2)}})
        if transformer_like:
            steps.append({"name": "prune_ffn", "args": {"amount": round(structural, 2)}})
            if structural >= 0.4:
                steps.append({"name": "drop_layers", "args": {"amount": 0.15}})
        elif is_mlp:
            steps.append({"name": "prune_mlp", "args": {"amount": round(structural, 2)}})
        else:
            steps.append({"name": "low_rank", "args": {"energy": 0.85}})
        if can_heal:
            steps.append({"name": "heal", "args": {"steps": 300}})
        else:
            rec.notes.append("Ohne Trainingsdaten kann nicht geheilt werden – Qualitätsverlust ist größer.")
        params_after *= (1 - structural * 0.8)

    # --- Präzision ----------------------------------------------------------------------------------
    need_bytes = (t.memory_mb * 2**20) / max(1.0, params_after)    # Bytes pro Gewicht, die passen
    chosen = None
    for d in t.dtypes:  # bevorzugtes Format zuerst – aber nur, wenn es passt
        if _BYTES[d] <= need_bytes:
            chosen = d
            break
    if chosen is None:
        chosen = min(t.dtypes, key=lambda d: _BYTES[d])
    if chosen in ("float16", "bfloat16"):
        steps.append({"name": "dtype", "args": {"dtype": chosen}})
    elif chosen in ("int8", "int4", "nf4", "pal4"):
        steps.append({"name": "quantize", "args": {"format": chosen}})
    rec.steps = steps
    rec.size_after_mb = params_after * _BYTES[chosen] / 2**20
    rec.latency_after_ms = (lat * (params_after / max(1, params))) if lat is not None else None
    rec.fits_after = rec.size_after_mb <= t.memory_mb and (rec.latency_after_ms is None
                                                           or rec.latency_after_ms <= t.latency_ms * 1.1)
    if _has(model, "AttentionMixer") and t.key not in ("rechenzentrum_gpu",):
        rec.notes.append("Bei der Generierung KV-Cache in int8: generate(..., cache={'dtype': 'int8'}).")
    return rec


def overview(model: nn.Module, targets: list[str], macs: int | None = None, can_heal: bool = True) -> str:
    """Kurze Tabelle über mehrere Zielsysteme (wird nach dem Training angezeigt)."""
    rows = [recommend(model, k, macs, can_heal) for k in targets]
    lines = ["Reduktion für Zielsysteme (Schätzung – Details: projekt.recommend_for(ziel), anwenden: reduce_for):",
             f"  {'Zielsystem':<42} {'jetzt':>11} {'Latenz':>9}  {'passt':<6} {'Empfehlung':<60} {'danach':>12}"]
    for r in rows:
        lat = f"{r.latency_now_ms:.1f} ms" if r.latency_now_ms is not None else "?"
        lines.append(f"  {r.target.title[:42]:<42} {format_number(r.size_now_mb, 1):>8} MB {lat:>9}  "
                     f"{'ja' if r.fits_now else 'nein':<6} {r.short()[:60]:<60} "
                     f"{format_number(r.size_after_mb, 1):>7} MB {'✓' if r.fits_after else '✗'}")
    return "\n".join(lines)
