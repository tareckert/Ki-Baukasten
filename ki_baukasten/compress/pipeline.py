"""
ki_baukasten.compress.pipeline
==============================

**Kompressions-Pipeline**: beliebige Schritte hintereinander, mit Messung vorher/nachher.

::

    projekt.compress([
        {"name": "prune_heads", "args": {"amount": 0.25}},
        {"name": "prune_ffn",   "args": {"amount": 0.4}},
        {"name": "drop_layers", "args": {"count": 1}},
        {"name": "heal",        "args": {"steps": 200}},       # Distillation vom Original
        {"name": "quantize",    "args": {"format": "int8"}},
    ])

Jeder Schritt ist eine Komponente der Registry ``compress_step`` (mit Tooltips) –
eigene Schritte lassen sich registrieren. Nach jedem Schritt werden Parameter,
Größe, Rechenaufwand, Latenz und Qualität gemessen (``CompressionReport``).

Empfohlene Reihenfolge
----------------------
1. **Struktur** (Köpfe, Neuronen, Blöcke) – entfernt echte Rechenarbeit
2. **Heilen** (Distillation vom Original) – holt verlorene Qualität zurück
3. **Low-Rank / Sparsity** – optional
4. **Quantisierung / Präzision** – zuletzt (verträgt sich nicht mit weiterem Umbau)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..core import BaseConfig, ComponentListParam, ConfigError, Level, Param, T, create_registry, format_number
from . import methods

STEPS = create_registry(
    "compress_step",
    "Kompressions-Schritte",
    T(
        "Einzelne Schritte der Modell-Kompression ('eindampfen').",
        "Kleinere, schnellere Modelle für schwächere Geräte – mit messbarem Qualitätsverlust bzw. Heilung.",
        "Deployment auf Smartphone, Raspberry Pi, CPU-Servern, Browser.",
    ),
    context_doc="factory(args) -> callable(ctx: CompressionContext) -> str (Beschreibung)",
)


# ===========================================================================
# Kontext & Bericht
# ===========================================================================

@dataclass
class CompressionContext:
    model: nn.Module
    project: Any = None                       # optionales Project (Daten, Training, Speichern)
    teacher: nn.Module | None = None          # unverändertes Original (für Heilen)
    calibration: list[Any] = field(default_factory=list)
    eval_batches: list[Any] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)
    masks: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class StageMetrics:
    stage: str
    parameters: int
    size_bytes: int
    macs: int | None
    latency_ms: float | None
    loss: float | None
    accuracy: float | None
    sparsity: float
    note: str = ""


@dataclass
class CompressionReport:
    stages: list[StageMetrics]

    @property
    def before(self) -> StageMetrics:
        return self.stages[0]

    @property
    def after(self) -> StageMetrics:
        return self.stages[-1]

    def ratio(self, attr: str) -> float | None:
        a, b = getattr(self.before, attr), getattr(self.after, attr)
        return (a / b) if a and b else None

    def __str__(self) -> str:
        def mb(n):
            return f"{format_number(n / 2**20, 2)} MB"

        lines = ["Kompression – Messung je Schritt",
                 f"  {'Schritt':<28} {'Parameter':>12} {'Größe':>11} {'MACs':>12} {'Latenz':>9} {'Verlust':>8} "
                 f"{'Genauigk.':>9} {'Nullen':>7}",
                 "  " + "─" * 104]
        for s in self.stages:
            lat = f"{s.latency_ms:.2f} ms" if s.latency_ms is not None else "–"
            loss = f"{s.loss:.4f}" if s.loss is not None else "–"
            acc = f"{100 * s.accuracy:.1f} %" if s.accuracy is not None else "–"
            macs = format_number(s.macs) if s.macs is not None else "–"
            lines.append(f"  {s.stage[:28]:<28} {format_number(s.parameters):>12} {mb(s.size_bytes):>11} {macs:>12} "
                         f"{lat:>9} {loss:>8} {acc:>9} {100 * s.sparsity:>6.1f}%")
            if s.note:
                lines.append(f"      ↳ {s.note}")
        r_size, r_lat = self.ratio("size_bytes"), self.ratio("latency_ms")
        summary = []
        if r_size:
            summary.append(f"Größe ÷ {r_size:.2f}")
        if self.ratio("macs"):
            summary.append(f"Rechenaufwand ÷ {self.ratio('macs'):.2f}")
        if r_lat:
            summary.append(f"Latenz ÷ {r_lat:.2f}")
        if self.before.loss is not None and self.after.loss is not None:
            summary.append(f"Verlust {self.before.loss:.4f} → {self.after.loss:.4f}")
        if summary:
            lines.append("  Ergebnis: " + " · ".join(summary))
        return "\n".join(lines)


def _example(ctx: CompressionContext) -> Any:
    from ..training.data_utils import to_device

    if ctx.eval_batches:
        b = ctx.eval_batches[0]
        x = b["inputs"] if isinstance(b, dict) else b
        return to_device(x[:1], methods._device(ctx.model))
    return ctx.model.example_input(1) if hasattr(ctx.model, "example_input") else None


@torch.no_grad()
def _quality(ctx: CompressionContext) -> tuple[float | None, float | None]:
    from ..training.paradigms import accuracy

    if not ctx.eval_batches:
        return None, None
    model = ctx.model
    was = model.training
    model.eval()
    losses, accs = [], []
    try:
        for b in ctx.eval_batches:
            losses.append(float(methods.calibration_loss(model, b)))
            if isinstance(b, dict) and b.get("targets") is not None:
                out = methods._call(model, b)
                acc = accuracy(out, b["targets"].to(out.device))
                if acc is not None:
                    accs.append(acc)
            elif isinstance(b, dict):
                x = b["inputs"]
                if isinstance(x, torch.Tensor) and not x.is_floating_point() and x.dim() == 2:
                    out = methods._call(model, b)
                    if out.dim() == 3 and out.shape[1] == x.shape[1]:
                        accs.append(float((out[:, :-1].argmax(-1).cpu() == x[:, 1:].cpu()).float().mean()))
    finally:
        model.train(was)
    return sum(losses) / len(losses), (sum(accs) / len(accs) if accs else None)


def measure(ctx: CompressionContext, stage: str, latency: bool = True, note: str = "") -> StageMetrics:
    from ..analysis.benchmark import count_macs, measure_latency, model_size_bytes

    ex = _example(ctx)
    macs = lat = None
    if ex is not None:
        try:
            macs = count_macs(ctx.model, ex)
            lat = measure_latency(ctx.model, ex, runs=5, warmup=1) * 1000 if latency else None
        except Exception:  # noqa: BLE001 - Messung ist optional
            macs = lat = None
    loss, acc = _quality(ctx)
    params = sum(p.numel() for p in ctx.model.parameters())
    params += sum(m.in_features * m.out_features for m in ctx.model.modules() if type(m).__name__ == "QuantLinear")
    return StageMetrics(stage, params, model_size_bytes(ctx.model), macs, lat, loss, acc,
                        methods.sparsity_of(ctx.model), note)


# ===========================================================================
# Schritt-Konfigurationen
# ===========================================================================

_IMPORTANCE = {
    "magnitude": "Gewichtsnormen (ohne Daten)",
    "activation": "Aktivierung × Gewichtsnorm auf Kalibrierungsdaten (empfohlen)",
    "taylor": "|Gewicht × Gradient| (Taylor-Näherung)",
    "random": "Zufall (Kontrollexperiment)",
}


def _importance(default: str = "activation") -> Param:
    return Param(default, choices=_IMPORTANCE, tooltip=T(
        "Wie die Wichtigkeit bestimmt wird.", "Entscheidet, WAS entfernt wird – großer Einfluss auf die Qualität.",
        "activation (mit Daten), magnitude (ohne Daten)."))


class PruneFFNConfig(BaseConfig):
    amount = Param(0.3, min=0.0, max=0.95, tooltip=T(
        "Anteil der FFN-Neuronen, die entfernt werden.", "FFNs enthalten meist ~⅔ der Parameter.", "0.2–0.5."))
    importance = _importance()
    layers = Param(None, type=str, nullable=True, tooltip=T(
        "Nur in diesen Blöcken (Selektor).", "Z. B. hintere Schichten stärker beschneiden.", "None = alle."))
    multiple_of = Param(8, min=1, tooltip=T(
        "Neue Breite auf ein Vielfaches runden.", "Vielfache von 8/64 rechnen auf GPUs schneller.", "8."),
        level=Level.EXPERT)


class PruneHeadsConfig(BaseConfig):
    amount = Param(0.25, min=0.0, max=0.95, tooltip=T(
        "Anteil der Attention-Köpfe, die entfernt werden.", "Viele Köpfe sind redundant (Michel et al. 2019).",
        "0.2–0.5."))
    importance = _importance()
    layers = Param(None, type=str, nullable=True, tooltip=T("Nur in diesen Blöcken.", "Selektiv.", "None = alle."))
    granularity = Param("auto", choices={
        "auto": "group bei GQA (≥ 2 KV-Gruppen), sonst query",
        "query": "Query-Köpfe gleichmäßig je Gruppe entfernen",
        "group": "ganze KV-Gruppen entfernen (verkleinert auch den KV-Cache)",
    }, tooltip=T("Was als Einheit entfernt wird.", "group spart zusätzlich KV-Cache.", "auto."), level=Level.ADVANCED)


class DropLayersConfig(BaseConfig):
    count = Param(1, type=int, min=0, nullable=True, tooltip=T(
        "Anzahl zu entfernender Blöcke (None = 'amount' verwenden).", "Weniger Tiefe = proportional schneller.",
        "1–25 % der Blöcke."))
    amount = Param(None, type=float, min=0.0, max=0.9, nullable=True, tooltip=T(
        "Anteil zu entfernender Blöcke (wenn count = None).", "Relativ zur Tiefe.", "0.25."))
    importance = Param("block_influence", choices={
        "block_influence": "Blöcke, die ihre Eingabe am wenigsten verändern (ShortGPT, braucht Daten)",
        "last": "hintere Blöcke zuerst", "magnitude": "kleinste Gewichtsnorm", "random": "Zufall"},
        tooltip=T("Auswahl der Blöcke.", "Block-Einfluss ist meist am besten.", "block_influence."))
    keep_first_last = Param(True, tooltip=T(
        "Ersten und letzten Block nie entfernen.", "Sie sind fast immer besonders wichtig.", "an."))


class PruneMLPConfig(BaseConfig):
    amount = Param(0.3, min=0.0, max=0.95, tooltip=T(
        "Anteil der Neuronen je versteckter Schicht (Modell 'mlp').", "Schmalere Schichten.", "0.2–0.5."))
    importance = _importance("magnitude")
    multiple_of = Param(1, min=1, tooltip=T("Rundung der Breite.", "GPU-freundliche Breiten.", "1."), level=Level.EXPERT)


class LowRankConfig(BaseConfig):
    energy = Param(0.9, min=0.1, max=1.0, tooltip=T(
        "Anteil der 'Energie' (Σ Singulärwerte²), der erhalten bleibt.", "Höher = genauer, weniger Ersparnis.",
        "0.85–0.95."))
    rank_ratio = Param(None, type=float, min=0.01, max=1.0, nullable=True, tooltip=T(
        "Fester Rang als Anteil von min(in, out) (statt energy).", "Planbare Größe.", "0.25–0.5."),
        level=Level.ADVANCED)
    targets = Param(None, type=str, nullable=True, tooltip=T(
        "Regex: nur diese Schichten.", "Z. B. nur FFN ('ffn').", "None = alle linearen."), level=Level.ADVANCED)
    exclude = Param("lm_head|embed|router", type=str, nullable=True, tooltip=T(
        "Regex: nie zerlegen.", "Empfindliche Teile schützen.", "lm_head|embed|router."), level=Level.ADVANCED)
    min_saving = Param(0.1, min=0.0, max=0.9, tooltip=T(
        "Nur zerlegen, wenn mindestens dieser Anteil Parameter gespart wird.", "Verhindert Zerlegung ohne Nutzen.",
        "0.1."), level=Level.EXPERT)


class SparsifyConfig(BaseConfig):
    sparsity = Param(0.5, min=0.0, max=0.99, tooltip=T(
        "Anteil der Gewichte, die auf 0 gesetzt werden.", "Mit Sparse-Formaten/Zip kleiner; bis ~50 % oft fast verlustfrei.",
        "0.3–0.7."))
    scope = Param("global", choices={"global": "eine Schwelle für alle Schichten", "layer": "je Schicht gleicher Anteil"},
                  tooltip=T("Wie die Schwelle bestimmt wird.", "global schont wichtige Schichten.", "global."))
    targets = Param(None, type=str, nullable=True, tooltip=T("Regex: nur diese Gewichte.", "Selektiv.", "None."),
                    level=Level.ADVANCED)
    exclude = Param("embed|norm|lm_head", type=str, nullable=True, tooltip=T(
        "Regex: auslassen.", "Empfindliche Teile schützen.", "embed|norm|lm_head."), level=Level.ADVANCED)


class QuantizeStepConfig(BaseConfig):
    format = Param("int8", choices=["int8", "int4", "nf4", "pal4"], tooltip=T(
        "Format der Gewichte.", "int8 ≈ ¼, int4/nf4/pal4 ≈ ⅛ des float32-Speichers.",
        "int8 (sicher), nf4 (normalverteilte Gewichte), pal4 (k-means-Palette, Apple-artig)."))
    group_size = Param(64, type=int, min=8, nullable=True, tooltip=T(
        "Gruppengröße der Skalen (None = pro Kanal).", "Kleinere Gruppen = genauer.", "64."))
    skip = Param("lm_head", type=str, nullable=True, tooltip=T(
        "Regex: nicht quantisieren.", "Ausgabeschicht schützen.", "lm_head."))


class DtypeStepConfig(BaseConfig):
    dtype = Param("bfloat16", choices=["float16", "bfloat16"], tooltip=T(
        "Ziel-Datentyp aller Gewichte.", "Halber Speicher; bfloat16 robust, float16 für Mobil-GPUs/Apple.",
        "GPU: bfloat16, Smartphone/Apple: float16."))


class HealConfig(BaseConfig):
    steps = Param(200, min=1, tooltip=T(
        "Trainingsschritte der Heilung.", "Mehr Schritte holen mehr Qualität zurück.", "100–2000."))
    lr = Param(5e-4, min=0.0, tooltip=T("Lernrate.", "Klein halten – das Modell ist schon trainiert.", "1e-4–1e-3."))
    temperature = Param(2.0, min=1e-3, tooltip=T("Distillations-Temperatur.", "Weiche Ziele.", "1–4."))
    alpha = Param(0.7, min=0.0, max=1.0, tooltip=T(
        "Anteil Lehrer-Verlust (Rest: echte Labels/nächstes Token).", "Balance.", "0.5–0.9."))
    feature_weight = Param(0.5, min=0.0, tooltip=T(
        "Gewicht der Feature-Distillation (innere Zustände).", "Hilft nach starkem Strukturumbau.", "0.1–1."))
    kd_loss = Param("kl", choices=["kl", "reverse_kl", "js", "mse"], tooltip=T(
        "Distillationsverlust.", "kl Standard; reverse_kl für Sprachmodelle.", "kl."))
    batch_size = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Batchgröße (None = wie im Projekt).", "Speicher vs. Tempo.", "None."), level=Level.ADVANCED)


class FinetuneConfig(BaseConfig):
    steps = Param(200, min=1, tooltip=T("Trainingsschritte mit dem normalen Paradigma.", "Nachtraining.", "100–1000."))
    lr = Param(3e-4, min=0.0, tooltip=T("Lernrate.", "Klein halten.", "1e-4–5e-4."))


# ===========================================================================
# Schritt-Implementierungen
# ===========================================================================

def _structure(fn):
    def run(ctx: CompressionContext) -> str:
        before = sum(p.numel() for p in ctx.model.parameters())
        rules = fn(ctx)
        ctx.rules += rules
        after = sum(p.numel() for p in ctx.model.parameters())
        return f"{len(rules)} Formänderungen, {format_number(before - after)} Parameter entfernt"
    return run


def _train_steps(ctx: CompressionContext, steps: int, lr: float, distill: dict | None, batch_size: int | None) -> str:
    """Kurzes (Nach-)Training mit dem Trainer des Projekts – optional als Distillation vom Original."""
    from ..training import Trainer
    from ..training.callbacks import Callback

    p = ctx.project
    if p is None:
        raise ConfigError("Heilen/Nachtrainieren braucht ein Project (Daten). Für reine Modelle: weglassen.")
    t = p.config.training.copy()
    t.max_steps, t.verbose, t.callbacks = steps, False, []
    t.ema.enabled = False
    if batch_size:
        t.batch_size = batch_size
    t.set("optimizer.lr", lr)
    if distill is not None:
        t.paradigm = {"name": "distillation", "args": distill}
    masks = ctx.masks

    class KeepMasks(Callback):
        def on_step_end(self, trainer, metrics):  # Sparsity-Masken nach jedem Schritt erzwingen
            with torch.no_grad():
                params = dict(trainer.model.named_parameters())
                for n, m in masks.items():
                    if n in params:
                        params[n].mul_(m.to(params[n].device, params[n].dtype))

    mem = p.config.memory.copy()
    mem.quantization, mem.lora.enabled, mem.offload.mode = "none", False, mem.offload.mode
    trainer = Trainer(ctx.model, t, mem, teacher=ctx.teacher if distill is not None else None,
                      callbacks=[KeepMasks()] if masks else [])
    bundle = p.data()
    trainer.fit(bundle.train, None)
    ctx.model = trainer.model
    last = next((h for h in reversed(trainer.history) if "loss" in h), {})
    return f"{steps} Schritte, letzter Verlust {last.get('loss', float('nan')):.4f}"


def _heal(a: HealConfig):
    def run(ctx: CompressionContext) -> str:
        if ctx.teacher is None:
            raise ConfigError("heal braucht das Originalmodell als Lehrer (wird automatisch kopiert).")
        distill = {"temperature": a.temperature, "alpha": a.alpha, "kd_loss": a.kd_loss,
                   "feature_weight": a.feature_weight}
        return "Distillation vom Original: " + _train_steps(ctx, a.steps, a.lr, distill, a.batch_size)
    return run


def _quantize(a: QuantizeStepConfig):
    def run(ctx: CompressionContext) -> str:
        from ..memory.quant import quantize_model
        done = quantize_model(ctx.model, a.format, a.group_size, a.skip)
        ctx.rules += [{"label": "Kompression: Quantisierung", "target": n, "match": "exact",
                       "op": {"name": "quantize", "format": a.format, "group_size": a.group_size}} for n in done]
        return f"{len(done)} lineare Schichten → {a.format}"
    return run


def _dtype(a: DtypeStepConfig):
    def run(ctx: CompressionContext) -> str:
        ctx.model.to({"float16": torch.float16, "bfloat16": torch.bfloat16}[a.dtype])
        if ctx.project is not None:
            ctx.project.config.memory.param_dtype = a.dtype
        return f"Gewichte in {a.dtype}"
    return run


def _sparsify(a: SparsifyConfig):
    def run(ctx: CompressionContext) -> str:
        masks = methods.sparsify(ctx.model, a.sparsity, a.scope, a.targets, a.exclude)
        ctx.masks.update(masks)
        return f"{len(masks)} Gewichtsmatrizen, Nullanteil {100 * methods.sparsity_of(ctx.model):.1f} %"
    return run


def _merge(a):
    def run(ctx: CompressionContext) -> str:
        from ..memory.lora import merge_lora
        return f"{merge_lora(ctx.model)} LoRA-Adapter eingerechnet"
    return run


R = STEPS.register
R("prune_ffn", lambda a: _structure(lambda c: methods.prune_ffn(c.model, a.amount, a.importance, c.calibration,
                                                                 a.layers, a.multiple_of)),
  config=PruneFFNConfig, aliases=("ffn_pruning",), tooltip=T(
      "FFN-Neuronen entfernen (strukturiert).", "Echte Ersparnis an Parametern und Rechenzeit.",
      "Transformer/SSM/Hybrid (MLP-, GLU- und MoE-Experten-FFNs)."))
R("prune_heads", lambda a: _structure(lambda c: methods.prune_heads(c.model, a.amount, a.importance, c.calibration,
                                                                     a.layers, a.granularity)),
  config=PruneHeadsConfig, aliases=("head_pruning",), tooltip=T(
      "Attention-Köpfe entfernen.", "Weniger Rechenaufwand in der Attention, bei 'group' auch kleinerer KV-Cache.",
      "Transformer, Hybride."))
R("drop_layers", lambda a: _structure(lambda c: methods.drop_layers(c.model, a.count, a.amount, a.importance,
                                                                     c.calibration, a.keep_first_last)),
  config=DropLayersConfig, aliases=("layer_pruning",), tooltip=T(
      "Ganze Blöcke entfernen.", "Proportional schneller (Tiefe) – oft erstaunlich verlustarm.", "Tiefe Modelle."))
R("prune_mlp", lambda a: _structure(lambda c: methods.prune_mlp(c.model, a.amount, a.importance, c.calibration,
                                                                 a.multiple_of)),
  config=PruneMLPConfig, tooltip=T("Neuronen eines MLP-Modells entfernen.", "Schmalere Schichten.", "Modell 'mlp'."))
R("low_rank", lambda a: _structure(lambda c: methods.low_rank(c.model, a.energy, a.rank_ratio, a.targets, a.exclude,
                                                               a.min_saving)),
  config=LowRankConfig, aliases=("svd",), tooltip=T(
      "Lineare Schichten per SVD zerlegen.", "Spart Parameter, wo Matrizen 'eigentlich' niedrigrangig sind.",
      "Große Projektionen, nach dem Training."))
R("sparsify", lambda a: _sparsify(a), config=SparsifyConfig, aliases=("magnitude_pruning",), tooltip=T(
    "Unstrukturiertes Pruning (kleinste Gewichte = 0).", "Hohe Kompressionsraten mit Sparse-Formaten/Zip.",
    "Speichergröße, Vorbereitung auf Sparse-Kernel."))
R("heal", lambda a: _heal(a), config=HealConfig, aliases=("distill", "heilen"), tooltip=T(
    "Kurze Distillation vom unkomprimierten Original.",
    "Holt einen Großteil der durch Pruning verlorenen Qualität zurück.", "Nach jedem stärkeren Strukturschritt."))
R("finetune", lambda a: (lambda ctx: "Nachtraining: " + _train_steps(ctx, a.steps, a.lr, None, None)),
  config=FinetuneConfig, tooltip=T("Normales Nachtraining.", "Ohne Lehrer.", "Wenn kein Original vorliegt."))
R("quantize", lambda a: _quantize(a), config=QuantizeStepConfig, tooltip=T(
    "Gewichte quantisieren (int8/int4/nf4/pal4).", "¼ bis ⅛ des Speichers.", "Als letzter Schritt."))
R("dtype", lambda a: _dtype(a), config=DtypeStepConfig, aliases=("half",), tooltip=T(
    "Alle Gewichte in 16 Bit.", "Halber Speicher, auf GPUs/Apple oft schneller.", "GPU-/Mobil-Deployment."))
R("merge_lora", lambda a: _merge(a), tooltip=T(
    "LoRA-Adapter in die Gewichte einrechnen.", "Keine Zusatzkosten bei der Inferenz.", "Vor dem Export."))


class CompressionConfig(BaseConfig):
    __title__ = "Kompression (Modell eindampfen)"
    __tooltip__ = T(
        "Rezept zum Verkleinern eines (trainierten) Modells: Schritte + Messung.",
        "Kleinere, schnellere Modelle für schwächere Zielsysteme – mit Vorher/Nachher-Bericht.",
        "Deployment; Project.compress() bzw. Project.reduce_for(ziel).",
    )
    steps = ComponentListParam("compress_step", tooltip=T(
        "Kompressions-Schritte in Reihenfolge.", "Freie Kombination aller Verfahren.",
        "z. B. prune_heads → prune_ffn → heal → quantize."))
    calibration_batches = Param(8, min=0, tooltip=T(
        "Anzahl Trainingsbatches zur Kalibrierung (Wichtigkeit, Block-Einfluss).",
        "Mehr = stabilere Auswahl.", "4–32."))
    eval_batches = Param(4, min=0, tooltip=T(
        "Validierungsbatches für die Qualitätsmessung nach jedem Schritt (0 = keine Messung).",
        "Zeigt sofort, welcher Schritt wie viel kostet.", "4–16."))
    measure_latency = Param(True, tooltip=T(
        "Latenz (Batch 1) nach jedem Schritt messen.", "Zeigt die echte Beschleunigung auf diesem Gerät.", "an."),
        level=Level.ADVANCED)


def run_compression(model: nn.Module, steps: list, project: Any = None, calibration: list | None = None,
                    eval_batches: list | None = None, measure_latency: bool = True,
                    keep_teacher: bool = True) -> tuple[nn.Module, CompressionReport, list[dict]]:
    """Führt die Schritte aus. Liefert ``(komprimiertes Modell, Bericht, Struktur-Regeln)``."""
    from ..core.component import ComponentSpec

    specs = [s if isinstance(s, ComponentSpec) else ComponentSpec.from_value("compress_step", s) for s in steps]
    teacher = None
    if keep_teacher and any(s.name in ("heal",) for s in specs):
        teacher = copy.deepcopy(model).eval()
        teacher.requires_grad_(False)
    ctx = CompressionContext(model, project, teacher, list(calibration or []), list(eval_batches or []))
    stages = [measure(ctx, "Original", measure_latency)]
    for spec in specs:
        note = spec.build()(ctx)
        stages.append(measure(ctx, spec.name, measure_latency, note))
    return ctx.model, CompressionReport(stages), ctx.rules


__all__ = ["STEPS", "CompressionConfig", "CompressionContext", "CompressionReport", "StageMetrics", "run_compression",
           "measure"]
