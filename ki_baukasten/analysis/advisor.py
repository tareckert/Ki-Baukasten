"""
ki_baukasten.analysis.advisor
=============================

**Optimierungs- und Speicher-Ratgeber**: untersucht Konfiguration, gebautes
Modell und Speicherschätzung und schlägt konkrete, **direkt anwendbare**
Änderungen vor – jeweils mit Begründung und geschätztem Effekt.

Kategorien
----------
==============  ==============================================================
``speicher``    weniger GPU-/RAM-Bedarf (Optimierer, Aktivierungen, Gewichte, Cache)
``tempo``       schnelleres Training/Inferenz
``stabilitaet`` robusteres Training (Clipping, Warmup, Präzision)
``qualitaet``   bessere Ergebnisse bei gleichem Aufwand
``struktur``    Architektur-Hinweise (GQA, Weight Tying, Kopfgrößen …)
==============  ==============================================================

Nutzung::

    for tipp in projekt.advise():
        print(tipp)
    projekt.apply_advice(["adamw8bit", "gqa"])      # nach Kennung anwenden
    projekt.apply_advice(kategorie="speicher")      # alle Speichertipps

Jeder Vorschlag hat eine Kennung (``key``), eine Wichtigkeit (1 = hoch …
3 = niedrig) und ``changes`` – ein dict ``{pfad: wert}``, das ``Project.update``
direkt versteht.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..core import format_number


@dataclass
class Advice:
    key: str
    category: str
    priority: int
    title: str
    reason: str
    effect: str = ""
    changes: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        stars = {1: "‼", 2: "!", 3: "·"}.get(self.priority, "·")
        lines = [f"{stars} [{self.category}] {self.title}  (Kennung: {self.key})", f"    Warum: {self.reason}"]
        if self.effect:
            lines.append(f"    Effekt: {self.effect}")
        if self.changes:
            lines.append("    Ändern: " + ", ".join(f"{k} = {v!r}" for k, v in self.changes.items()))
        return "\n".join(lines)


def _mb(n: float) -> str:
    return f"{format_number(n / 2**20, 1)} MiB"


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    try:
        return cfg.get(path)
    except Exception:  # noqa: BLE001 - Pfad existiert bei diesem Modell nicht
        return default


def advise(project: Any, batch_size: int | None = None, measure: bool = True) -> list[Advice]:
    """Analysiert ein ``Project`` (baut das Modell bei Bedarf) und liefert sortierte Vorschläge."""
    cfg = project.config
    model: nn.Module = project.model if project.model is not None else project.build()
    t, mem = cfg.training, cfg.memory
    tips: list[Advice] = []
    add = tips.append
    on_gpu = torch.cuda.is_available() and (mem.device == "auto" or str(mem.device).startswith("cuda"))
    model_name = cfg.model.name
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ------------------------------------------------------------ Speicher
    report = None
    try:
        report = project.memory_report(batch_size=batch_size) if measure else None
    except Exception:  # noqa: BLE001
        report = None
    opt_name = t.optimizer.name
    if report is not None:
        posts = {"Optimierer": report.optimizer_bytes, "Aktivierungen": report.activations_bytes or 0.0,
                 "Gewichte": report.weights_bytes, "Gradienten": report.gradients_bytes}
        biggest = max(posts, key=posts.get)
        tips_text = f"Größter Posten: {biggest} ({_mb(posts[biggest])} von {_mb(report.total_bytes)} gesamt)."
        tight = report.fits() is False
        if opt_name in ("adam", "adamw") and report.optimizer_bytes > (0.1 if tight else 0.2) * report.total_bytes:
            add(Advice("adamw8bit", "speicher", 1 if tight else 2, "8-Bit-AdamW verwenden",
                       f"AdamW hält zwei 32-Bit-Kopien pro Parameter. {tips_text}",
                       f"spart ≈ {_mb(report.optimizer_bytes * 0.75)} Optimierer-Speicher",
                       {"training.optimizer": "adamw8bit"}))
        act = report.activations_bytes or 0.0
        if act > 0.3 * report.total_bytes and not mem.gradient_checkpointing and mem.offload.mode != "stream":
            add(Advice("checkpointing", "speicher", 1 if tight else 2, "Gradient-Checkpointing einschalten",
                       f"Aktivierungen dominieren ({_mb(act)}). {tips_text}",
                       f"spart ≈ {_mb(act * 0.65)} (kostet ~30 % Rechenzeit)",
                       {"memory.gradient_checkpointing": True}))
        if act > 0.3 * report.total_bytes and mem.mixed_precision == "none" and on_gpu:
            add(Advice("bf16", "speicher", 2, "Gemischte Präzision bf16",
                       "Aktivierungen in 16 Bit halbieren den Aktivierungsspeicher und nutzen Tensor-Cores.",
                       f"spart ≈ {_mb(act * 0.5)}, oft 1.5–3× schneller", {"memory.mixed_precision": "bf16"}))
        if tight and act > 0.5 * report.total_bytes and t.batch_size > 4:
            add(Advice("akkumulation", "speicher", 1, "Kleinere Batches mit Gradient-Akkumulation",
                       "Aktivierungen wachsen linear mit der Batchgröße; Akkumulation erhält die effektive Batchgröße.",
                       "halbiert Aktivierungsspeicher",
                       {"training.batch_size": max(1, t.batch_size // 2),
                        "training.gradient_accumulation_steps": t.gradient_accumulation_steps * 2}))
        if tight and report.weights_bytes + report.gradients_bytes + report.optimizer_bytes > 0.5 * report.total_bytes \
                and mem.offload.mode == "none":
            add(Advice("streaming", "speicher", 1, "Blöcke trainierbar in den RAM auslagern",
                       "Gewichte + Gradienten + Optimierer-Zustände passen nicht – beim Streaming liegen sie im "
                       "CPU-RAM, auf dem Gerät nur der gerade rechnende Block.",
                       "Geräte-Speicher für Parameter ≈ ein Block", {"memory.offload.mode": "stream"}))
    if n_params > 5e7 and n_train > 0.9 * n_params and not mem.lora.enabled:
        add(Advice("qlora", "speicher", 2, "Feintuning mit QLoRA (nf4-Basis + LoRA-Adapter)",
                   "Bei großen vortrainierten Modellen reicht es meist, kleine Adapter zu trainieren.",
                   "Gewichte ⅛, Gradienten/Optimierer nur für < 1 % der Parameter",
                   {"memory.quantization": "nf4", "memory.lora.enabled": True}))
    if n_train < 0.5 * n_params and mem.quantization == "none" and n_params > 1e6:
        add(Advice("int8_basis", "speicher", 3, "Eingefrorene Gewichte quantisieren (int8)",
                   f"{format_number(n_params - n_train)} Parameter werden nicht trainiert – sie müssen nicht in "
                   "32 Bit vorliegen.", "Gewichtsspeicher der eingefrorenen Teile ÷ 4",
                   {"memory.quantization": "int8"}))

    # ------------------------------------------------------------ Struktur (Transformer & Co.)
    blocks = list(getattr(model, "blocks", []))
    attn = [b.mixer.attn for b in blocks if hasattr(getattr(b, "mixer", None), "attn")]
    if attn:
        a0 = attn[0]
        if a0.num_kv_heads == a0.num_heads and a0.num_heads >= 8 and _get(cfg, "model.block.mixer.num_kv_heads", 0) is None:
            kv = max(1, a0.num_heads // 4)
            add(Advice("gqa", "struktur", 2, f"Grouped-Query-Attention ({kv} KV-Köpfe statt {a0.num_heads})",
                       "Weniger Key/Value-Köpfe verkleinern den KV-Cache und die K/V-Projektionen kaum spürbar "
                       "für die Qualität (LLaMA-2/3, Mistral).",
                       f"KV-Cache ÷ {a0.num_heads // kv}, Parameter der K/V-Projektionen ÷ {a0.num_heads // kv}",
                       {"model.block.mixer.num_kv_heads": kv}))
        if a0.head_dim % 8:
            add(Advice("head_dim", "tempo", 3, f"Kopfdimension {a0.head_dim} ist kein Vielfaches von 8",
                       "Optimierte Attention-Kernel (Flash) und Tensor-Cores arbeiten mit Vielfachen von 8/16 "
                       "deutlich schneller.", "schnellere Attention",
                       {"model.block.mixer.num_heads": max(1, cfg.model.args.d_model // 64)}
                       if hasattr(cfg.model.args, "d_model") and cfg.model.args.d_model % 64 == 0 else {}))
        seq = _get(cfg, "model.max_seq_len", 0) or 0
        batch = batch_size or t.batch_size
        if seq >= 1024:
            kv_bytes = sum(2 * batch * seq * x.num_kv_heads * x.head_dim * 4 for x in attn)
            add(Advice("kv_cache_int8", "speicher", 3, "KV-Cache für die Generierung in int8",
                       f"Bei Kontext {seq} und Batch {batch} bräuchte der KV-Cache in float32 ≈ {_mb(kv_bytes)}.",
                       f"KV-Cache ≈ {_mb(kv_bytes / 4)} (generate(..., cache={{'dtype': 'int8'}}))"))
            if all(x.cfg.window_size is None for x in attn) and len(attn) >= 4:
                add(Advice("sliding_window", "tempo", 3, "Sliding-Window-Attention in einem Teil der Schichten",
                           "Bei langen Kontexten wächst Attention quadratisch; lokale Fenster in jeder zweiten Schicht "
                           "behalten globale Reichweite über die Tiefe (Gemma 2/3, Mistral).",
                           "weniger Rechenaufwand und Speicher bei langen Sequenzen",
                           {"model.layer_overrides": {**(_get(cfg, "model.layer_overrides", {}) or {}),
                                                      "::2": {"mixer.window_size": max(128, seq // 8)}}}))
    tie = _get(cfg, "model.tie_embeddings")
    vocab = _get(cfg, "model.vocab_size", 0) or 0
    if tie is False and vocab >= 1000 and _get(cfg, "model.input_type") == "tokens":
        d = _get(cfg, "model.d_model", 0) or 0
        add(Advice("tie_embeddings", "struktur", 2, "Ein- und Ausgabe-Embedding teilen (Weight Tying)",
                   "Bei Sprachmodellen sind Embedding und Ausgabeschicht oft die größten Matrizen.",
                   f"spart {format_number(vocab * d)} Parameter", {"model.tie_embeddings": True}))
    for b in blocks:
        mixer = getattr(b, "mixer", None)
        c = getattr(mixer, "cfg", None)
        if c is not None and hasattr(c, "scan_mode") and (c.scan_mode == "sequential" or
                                                           getattr(c, "scan_backward", "fused") == "autograd"):
            add(Advice("ssm_scan", "tempo", 1, "Schnellen SSM-Scan verwenden",
                       "Der aufgezeichnete Python-Scan ist um ein Vielfaches langsamer und speicherhungriger.",
                       "Mamba-Training ~10× schneller, Aktivierungsspeicher ÷ d_state",
                       {"model.block.mixer.scan_mode": "auto", "model.block.mixer.scan_backward": "fused"}))
            break

    # ------------------------------------------------------------ Stabilität
    if t.gradient_clipping.mode == "none" and model_name in ("transformer", "ssm", "hybrid", "rnn"):
        add(Advice("clipping", "stabilitaet", 2, "Gradient-Clipping einschalten",
                   "Transformer und RNNs neigen zu seltenen, sehr großen Gradienten.", "verhindert Ausreißer-Schritte",
                   {"training.gradient_clipping.mode": "norm", "training.gradient_clipping.max_norm": 1.0}))
    lr = _get(t, "optimizer.lr", None)
    d_model = _get(cfg, "model.d_model", 0) or 0
    if lr and opt_name in ("adam", "adamw", "adamw8bit") and d_model >= 512 and lr > 1e-3:
        add(Advice("lr", "stabilitaet", 2, f"Lernrate {lr:g} ist für d_model={d_model} hoch",
                   "Breite Transformer werden mit Adam-Lernraten > 1e-3 oft instabil.", "stabileres Training",
                   {"training.optimizer.lr": 3e-4}))
    if t.scheduler.name in ("constant",) and model_name in ("transformer", "ssm", "hybrid"):
        add(Advice("warmup", "stabilitaet", 3, "Lernraten-Plan mit Warmup",
                   "Ein kurzes Aufwärmen verhindert frühe Instabilität adaptiver Optimierer.", "robusterer Start",
                   {"training.scheduler": "warmup_cosine"}))
    if mem.mixed_precision == "fp16" and mem.grad_scaler == "off":
        add(Advice("grad_scaler", "stabilitaet", 1, "Loss-Scaling für fp16 einschalten",
                   "Ohne Scaling verschwinden kleine fp16-Gradienten (Unterlauf).", "verhindert stilles Nicht-Lernen",
                   {"memory.grad_scaler": "auto"}))

    # ------------------------------------------------------------ Tempo
    if on_gpu:
        if not mem.compile:
            add(Advice("compile", "tempo", 3, "torch.compile verwenden",
                       "Fusioniert Operationen zu wenigen Kerneln.", "oft 1.3–2× schneller nach Aufwärmen",
                       {"memory.compile": True}))
        if mem.matmul_precision == "highest":
            add(Advice("tf32", "tempo", 3, "TF32-Matrixmultiplikation erlauben",
                       "Ampere+-GPUs rechnen fp32-Matrizen mit TF32 mehrfach schneller bei kaum messbarem Fehler.",
                       "bis zu 3× schnellere Matmuls", {"memory.matmul_precision": "high"}))
        if model_name == "cnn" and not mem.channels_last:
            add(Advice("channels_last", "tempo", 3, "Speicherlayout channels_last für Faltungen",
                       "Tensor-Core-Faltungen sind in NHWC schneller.", "10–30 % schneller",
                       {"memory.channels_last": True}))
    if t.num_workers == 0 and cfg.data.name in ("image_folder", "csv", "text_file"):
        add(Advice("workers", "tempo", 3, "Daten parallel laden",
                   "Mit 0 Workern wartet die GPU auf das Laden/Augmentieren.", "höhere GPU-Auslastung",
                   {"training.num_workers": 4}))
    return sorted(tips, key=lambda a: (a.priority, a.category))


def apply_advice(project: Any, tips: list[Advice], keys: list[str] | None = None,
                 category: str | None = None) -> list[Advice]:
    """Wendet die gewählten Vorschläge an (nach Kennung und/oder Kategorie); liefert die angewandten."""
    chosen = [a for a in tips if a.changes and (keys is None or a.key in keys)
              and (category is None or a.category == category)]
    for a in chosen:
        project.update(a.changes)
    return chosen
