"""
ki_baukasten.analysis.fallback
==============================

**Notfall-Ausführung**: Was tun, wenn ein Modell *trotz* Verkleinern nicht auf die Hardware passt?

Statt aufzugeben, wird das Modell **stückweise** ausgeführt – Schicht für Schicht, Batch-Stück
für Batch-Stück, oder auf mehrere Geräte bzw. Speicherarten verteilt. Jede Option tauscht
Geschwindigkeit gegen Speicher. ``execution_options`` schätzt für jede Möglichkeit den
Spitzenbedarf auf dem Rechengerät, den zusätzlichen RAM-/Festplattenbedarf und die
Verlangsamung – und liefert die Einstellungen, die sie einschalten (``Project.apply_fallback``).

=====================  ==========================================================================
Inferenz               Wie es funktioniert
=====================  ==========================================================================
micro_batch            Eingaben in kleinen Stücken verarbeiten (``predict(chunk_size=…)``)
split                  hintere Blöcke dauerhaft auf der CPU rechnen (``offload.mode = split``)
ram_offload            Blöcke liegen im RAM, werden einzeln aufs Gerät geholt (``cpu``)
disk_offload           Blöcke liegen auf der SSD (Memory-Map), Schicht für Schicht geladen (``disk``)
layerweise_stapel      wie disk_offload, aber viele Eingaben pro Durchgang → jede Schicht nur einmal geladen
cpu                    alles auf der CPU (RAM ist meist viel größer als GPU-Speicher)
quantisieren_nf4       4-Bit-Gewichte ohne Training (Qualität leicht geringer)
=====================  ==========================================================================

=====================  ==========================================================================
Training               Wie es funktioniert
=====================  ==========================================================================
sparsam                Checkpointing + kleine Mikro-Batches + Akkumulation + bf16 (gleiches Ergebnis)
optimierer_8bit        8-Bit-AdamW: Optimierer-Zustände ÷ 4
lisa                   nur 2 zufällig wechselnde Blöcke gleichzeitig trainieren
lora / qlora           nur kleine Adapter trainieren (QLoRA: Basis zusätzlich in 4 Bit)
stream                 Gewichte, Gradienten, Optimierer im RAM; Blöcke einzeln aufs Gerät
stream_disk_lora       Basis eingefroren auf der SSD, Adapter im RAM – größer als der RAM möglich
cpu                    Training auf der CPU
cluster                auf mehrere GPUs/Rechner verteilen (FSDP teilt Gewichte und Optimierer)
=====================  ==========================================================================

Die Zahlen sind grobe Schätzungen (Größenordnung), keine Messungen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from ..core import format_number


@dataclass
class ExecutionOption:
    """Eine Möglichkeit, das Modell trotz Speichermangel auszuführen."""

    key: str
    title: str
    mode: str                         # inferenz | training
    peak_mb: float                    # Spitzenbedarf auf dem Rechengerät
    slowdown: float                   # grob: Faktor gegenüber "alles passt"
    how: str
    changes: dict[str, Any] = field(default_factory=dict)
    ram_mb: float = 0.0               # zusätzlich im CPU-RAM
    disk_mb: float = 0.0              # auf der Festplatte
    caveats: str = ""
    usage: str = ""                   # Hinweis zur Benutzung (z. B. predict(chunk_size=…))
    fits: bool = False
    quality_loss: bool = False

    def __str__(self) -> str:
        mark = "✓" if self.fits else "✗"
        extra = []
        if self.ram_mb:
            extra.append(f"RAM {format_number(self.ram_mb, 1)} MB")
        if self.disk_mb:
            extra.append(f"SSD {format_number(self.disk_mb, 1)} MB")
        lines = [f"{mark} {self.title}  (Kennung: {self.key})",
                 f"    Gerät ≈ {format_number(self.peak_mb, 1)} MB" + (" · " + " · ".join(extra) if extra else "")
                 + f" · Tempo ≈ {self.slowdown:.1f}× langsamer" + (" · leichter Qualitätsverlust"
                                                                    if self.quality_loss else ""),
                 f"    Wie: {self.how}"]
        if self.changes:
            lines.append("    Ändern: " + ", ".join(f"{k} = {v!r}" for k, v in self.changes.items()))
        if self.usage:
            lines.append(f"    Nutzung: {self.usage}")
        if self.caveats:
            lines.append(f"    Achtung: {self.caveats}")
        return "\n".join(lines)


def _blocks(model: nn.Module) -> list[nn.Module]:
    target = getattr(model, "_orig_mod", model)
    try:
        return list(target.checkpointable_blocks()) if hasattr(target, "checkpointable_blocks") else []
    except Exception:  # noqa: BLE001
        return []


def _mb(n: float) -> float:
    return n / 2**20


def execution_options(model: nn.Module, budget_mb: float, training: bool = False, activations_mb: float = 0.0,
                      trainable_fraction: float | None = None, scale: float = 1.0, batch_size: int = 1,
                      gpu_target: bool = True, n_devices: int = 4, callbacks: list | None = None,
                      gradient_accumulation_steps: int = 1) -> list[ExecutionOption]:
    """Alle Notfall-Optionen mit Schätzung, sortiert: passende zuerst, dann nach Tempo.

    Args:
        budget_mb:      verfügbarer Speicher des Rechengeräts.
        training:       Training (Gewichte + Gradienten + Optimierer + Aktivierungen) statt Inferenz.
        activations_mb: Aktivierungsspeicher beim aktuellen Batch (Schätzung des Aufrufers).
        scale:          Faktor auf alle Gewichtsgrößen (z. B. Größe nach geplanter Verkleinerung / jetzt).
        gpu_target:     Rechengerät ist ein Beschleuniger (dann ist CPU-Rechnen eine Option).
    """
    from .benchmark import model_size_bytes

    total = _mb(model_size_bytes(model)) * scale
    blocks = _blocks(model)
    block_mb = [_mb(model_size_bytes(b)) * scale for b in blocks]
    in_blocks = sum(block_mb)
    rest = max(0.0, total - in_blocks)
    largest = max(block_mb) if block_mb else total
    n_blocks = len(blocks)
    act = activations_mb
    act_one = act / max(1, batch_size)
    params = sum(p.numel() for p in model.parameters())
    frac = trainable_fraction if trainable_fraction is not None else (
        sum(p.numel() for p in model.parameters() if p.requires_grad) / max(1, params))
    opts: list[ExecutionOption] = []
    add = opts.append

    if not training:
        add(ExecutionOption("normal", "Alles im Speicher (keine Notlösung)", "inferenz", total + act, 1.0,
                            "Gewichte und Aktivierungen liegen vollständig auf dem Rechengerät."))
        add(ExecutionOption("micro_batch", "Eingaben stückweise verarbeiten", "inferenz", total + act_one, 1.3,
                            "Nur ein Beispiel (bzw. wenige) gleichzeitig – Aktivierungsspeicher minimal.",
                            usage="projekt.predict(daten, chunk_size=1) bzw. memory.oom_strategy='batch_teilen'"))
        if n_blocks >= 2:
            free = budget_mb - rest - act_one
            keep = 0
            used = 0.0
            for mb in block_mb:                       # vordere Blöcke aufs Gerät, solange Platz ist
                if used + mb > free:
                    break
                used += mb
                keep += 1
            k = max(1, n_blocks - keep)
            off_mb = sum(block_mb[-k:])
            add(ExecutionOption("split", f"Hintere {k} von {n_blocks} Blöcken auf der CPU rechnen", "inferenz",
                                total - off_mb + act_one, 1.0 + 4.0 * k / n_blocks,
                                "Teile des Modells rechnen dauerhaft auf einem zweiten Gerät (device_map).",
                                {"memory.offload.mode": "split", "memory.offload.split_device": "cpu",
                                 "memory.offload.layers": f"-{k}:"}, ram_mb=off_mb,
                                caveats="CPU-Blöcke sind deutlich langsamer als GPU-Blöcke."))
            add(ExecutionOption("ram_offload", "Blöcke im RAM, einzeln aufs Gerät laden", "inferenz",
                                rest + largest + act_one, 2.5,
                                "Jeder Block wird nur für seine Berechnung aufs Gerät kopiert (Layer-Swapping).",
                                {"memory.offload.mode": "cpu", "memory.offload.keep_resident": 1}, ram_mb=in_blocks))
            add(ExecutionOption("disk_offload", "Schicht für Schicht von der SSD laden", "inferenz",
                                rest + largest + act_one, 15.0,
                                "Blöcke liegen als Dateien auf der SSD und werden per Memory-Map einzeln geladen.",
                                {"memory.offload.mode": "disk", "memory.offload.keep_resident": 0},
                                ram_mb=largest, disk_mb=in_blocks,
                                caveats="Stark abhängig von der SSD-Geschwindigkeit."))
            stack = 64       # Beispiele pro Stapel: das Laden jeder Schicht verteilt sich auf alle
            add(ExecutionOption("layerweise_stapel", "Schichtweise mit großen Stapeln (je Schicht nur einmal laden)",
                                "inferenz", rest + largest + act_one * stack, 15.0 / stack + 1.0,
                                "Wie disk_offload, aber viele Eingaben auf einmal: jede Schicht wird pro Stapel nur "
                                "einmal gelesen – ideal für Batch-Auswertungen großer Modelle (AirLLM-Prinzip).",
                                {"memory.offload.mode": "disk", "memory.offload.keep_resident": 0},
                                ram_mb=largest, disk_mb=in_blocks,
                                usage=f"projekt.predict(viele_eingaben) – Tempo gilt pro Beispiel bei Stapeln von {stack}"))
        if gpu_target:
            add(ExecutionOption("cpu", "Auf der CPU rechnen (RAM statt Grafikspeicher)", "inferenz", 0.0, 10.0,
                                "Der Arbeitsspeicher ist meist viel größer als der Grafikspeicher.",
                                {"memory.device": "cpu"}, ram_mb=total + act))
        add(ExecutionOption("quantisieren_nf4", "Gewichte in 4 Bit (nf4) – ohne Nachtraining", "inferenz",
                            total * 0.16 + act_one, 1.3,
                            "Lineare Schichten werden auf 4 Bit gepackt und beim Rechnen entpackt.",
                            {"memory.quantization": "nf4"}, quality_loss=True))
    else:
        grads = total * frac
        optim = total * frac * 2
        full = total + grads + optim + act
        add(ExecutionOption("normal", "Normales Training (keine Notlösung)", "training", full, 1.0,
                            "Gewichte, Gradienten, Optimierer-Zustände und Aktivierungen auf dem Gerät."))
        add(ExecutionOption("sparsam", "Checkpointing + Mikro-Batches + Akkumulation + bf16", "training",
                            total + grads + optim + act_one * 0.5, 1.5,
                            "Aktivierungen werden neu berechnet statt gespeichert, Batches in kleinen Stücken "
                            "mit Akkumulation – die effektive Batchgröße bleibt gleich.",
                            {"memory.gradient_checkpointing": True, "memory.mixed_precision": "bf16",
                             "training.batch_size": 1,
                             "training.gradient_accumulation_steps": max(1, batch_size) * gradient_accumulation_steps,
                             "memory.oom_strategy": "batch_teilen"}))
        add(ExecutionOption("optimierer_8bit", "8-Bit-AdamW + Checkpointing", "training",
                            total + grads + optim / 4 + act_one, 1.4,
                            "Optimierer-Zustände in 8 Bit (blockweise skaliert).",
                            {"training.optimizer.name": "adamw8bit", "memory.gradient_checkpointing": True,
                             "training.batch_size": 1,
                             "training.gradient_accumulation_steps": max(1, batch_size) * gradient_accumulation_steps}))
        if n_blocks >= 2:
            k = min(2, n_blocks)
            active = sum(sorted(block_mb)[-k:]) + rest
            add(ExecutionOption("lisa", f"LISA: nur {k} von {n_blocks} Blöcken gleichzeitig trainieren", "training",
                                total + active * frac * 3 + act_one, 1.2,
                                "Zufällig wechselnde Blöcke sind trainierbar – volle Gewichte, aber Gradienten und "
                                "Optimierer nur für wenige Blöcke.",
                                {"training.callbacks": list(callbacks or []) + [{"name": "lisa",
                                                                                 "args": {"active_blocks": k}}],
                                 "memory.gradient_checkpointing": True}))
        add(ExecutionOption("lora", "LoRA: nur kleine Adapter trainieren", "training",
                            total + total * 0.03 + act_one, 1.1,
                            "Die Basis bleibt eingefroren, trainiert werden Adapter mit ~1 % der Parameter.",
                            {"memory.lora.enabled": True, "memory.gradient_checkpointing": True}))
        add(ExecutionOption("qlora", "QLoRA: Basis in 4 Bit + Adapter", "training",
                            total * 0.16 + total * 0.03 + act_one, 1.4,
                            "Wie LoRA, die eingefrorene Basis liegt zusätzlich in 4 Bit (nf4).",
                            {"memory.quantization": "nf4", "memory.lora.enabled": True,
                             "memory.gradient_checkpointing": True}, quality_loss=True))
        if n_blocks >= 2:
            add(ExecutionOption("stream", "Blöcke trainierbar im RAM (Streaming)", "training",
                                rest * (1 + 3 * frac) + 2 * largest + act_one, 2.5,
                                "Gewichte, Gradienten und Optimierer-Zustände der Blöcke liegen im RAM; auf dem Gerät "
                                "rechnet immer nur ein Block (Forward und Backward).",
                                {"memory.offload.mode": "stream"}, ram_mb=in_blocks * (1 + 3 * frac)))
            add(ExecutionOption("stream_disk_lora", "Basis auf der SSD, Adapter im RAM (Streaming + LoRA)", "training",
                                rest + 2 * largest + act_one, 8.0,
                                "Eingefrorene Basisgewichte liegen auf der SSD und werden blockweise per Memory-Map "
                                "gelesen; nur die LoRA-Adapter samt Optimierer liegen im RAM.",
                                {"memory.offload.mode": "stream", "memory.offload.storage": "disk",
                                 "memory.lora.enabled": True}, ram_mb=largest + in_blocks * 0.03, disk_mb=in_blocks,
                                caveats="Schnelle SSD nötig; trainiert werden nur Adapter (und Teile außerhalb der Blöcke)."))
        if gpu_target:
            add(ExecutionOption("cpu", "Auf der CPU trainieren", "training", 0.0, 15.0,
                                "Training im Arbeitsspeicher (langsam, aber meist viel mehr Platz).",
                                {"memory.device": "cpu"}, ram_mb=full))
        add(ExecutionOption("cluster", f"Auf {n_devices} Geräte verteilen (FSDP)", "training",
                            (total + grads + optim) / n_devices + act_one, 1.3,
                            "Fully Sharded Data Parallel: Gewichte, Gradienten und Optimierer werden auf alle "
                            "Geräte aufgeteilt (Start: python -m ki_baukasten verteilt …).",
                            {"training.distributed.strategy": "fsdp"},
                            caveats=f"Braucht {n_devices} GPUs oder Rechner (torchrun)."))

    for o in opts:
        o.fits = o.peak_mb <= budget_mb
    # Passende nach Tempo, nicht passende danach – die dem Budget am nächsten kommen zuerst.
    return sorted(opts, key=lambda o: (not o.fits, o.quality_loss if o.fits else 0,
                                       o.slowdown if o.fits else o.peak_mb))


def best_option(options: list[ExecutionOption], allow_quality_loss: bool = False) -> ExecutionOption | None:
    """Schnellste passende Option (ohne Qualitätsverlust, sofern nicht erlaubt) – ``None``, wenn keine passt."""
    for o in options:
        if o.fits and (allow_quality_loss or not o.quality_loss) and o.key != "normal":
            return o
    return None


def options_text(options: list[ExecutionOption], budget_mb: float, limit: int | None = None) -> str:
    head = (f"Notfall-Ausführung (Budget {format_number(budget_mb, 1)} MB) – passende zuerst, anwenden mit "
            f"projekt.apply_fallback('<kennung>'):")
    shown = options if limit is None else options[:limit]
    return "\n".join([head] + [str(o) for o in shown])
