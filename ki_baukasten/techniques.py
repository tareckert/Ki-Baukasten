"""
ki_baukasten.techniques
=======================

**Techniken-Katalog**: Optimierungen und Verfahren, die Communitys, Forschung und Firmen nutzen –
jede mit einem Klick **an- und abwählbar**.

Eine Technik ist ein benanntes Bündel von Einstellungen (z. B. "QLoRA" = nf4-Quantisierung +
LoRA-Adapter). ``an`` setzt die Einstellungen, ``aus`` stellt den vorherigen Zustand wieder her
(bzw. schaltet die Einstellungen aus). Der **Status** wird immer aus der aktuellen Konfiguration
abgeleitet: ``an`` · ``aus`` · ``teilweise`` · ``nicht anwendbar`` (passt nicht zu diesem Modell).

Kategorien
----------
Stabilität · gegen Auswendiglernen · Performance · große Modelle auf kleiner Hardware · Cluster ·
automatische Suche · Schwarmintelligenz · Evolution · Daten · Datenschutz · Überwachung ·
Kompression & Einsatz · Bewertung & Vertrauen · experimentell

Nutzung::

    p.techniques("stabilitaet")              # Liste mit Status
    p.enable("qlora", "gradient_clipping")   # einschalten
    p.disable("qlora")                       # ausschalten (vorheriger Zustand kommt zurück)
    print(p.technique_overview())

Schreibweise der Einstellungen einer Technik
--------------------------------------------
* ``"pfad": wert``                – genau diese Einstellung
* ``"model.**.dropout": wert``    – alle passenden Einstellungen (Platzhalter)
* ``"+training.callbacks": {...}`` – Element zu einer Komponentenliste hinzufügen (aus = entfernen)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .core import UNSET, BaseConfig, ConfigError, Level, T, Tooltip
from .core.config import _is_spec
from .core.switches import _same, off_value, set_state

CATEGORIES: dict[str, tuple[str, str]] = {
    "stabilitaet": ("Stabilität", "Training läuft sicher durch: keine Explosionen, NaNs oder Abstürze."),
    "generalisierung": ("Gegen Auswendiglernen", "Besser auf neuen Daten statt nur auf den Trainingsdaten."),
    "tempo": ("Performance", "Schneller trainieren und rechnen."),
    "grosse_modelle": ("Große Modelle, kleine Hardware", "Speicher sparen, auslagern, stückweise rechnen."),
    "cluster": ("Cluster & verteilt", "Mehrere GPUs, Rechner oder Prozesse."),
    "automl": ("Automatische Suche", "Gute Einstellungen automatisch finden."),
    "schwarm": ("Schwarmintelligenz", "Viele Modelle oder Teilnehmer arbeiten zusammen."),
    "evolution": ("Evolution", "Auslese, Kreuzung und Mutation als Optimierer."),
    "daten": ("Daten", "Augmentierung, Ausgleich seltener Klassen, Curriculum."),
    "datenschutz": ("Datenschutz", "Sensible Daten schützen."),
    "ueberwachung": ("Überwachung & Reproduzierbarkeit", "Protokolle, Checkpoints, Seeds."),
    "kompression": ("Kompression & Einsatz", "Nach dem Training verkleinern und exportieren."),
    "vertrauen": ("Bewertung & Vertrauen", "Ehrlich messen, Wahrscheinlichkeiten kalibrieren, Unsicherheit zeigen, "
                                          "robust gegen Störungen."),
    "experimentell": ("Experimentell", "Neue Verfahren aus der Forschung – Nutzen nicht immer gesichert."),
}


@dataclass(frozen=True)
class Technique:
    key: str
    title: str
    category: str
    tooltip: Tooltip
    changes: dict[str, Any] | Callable[[BaseConfig], dict[str, Any]]
    used_by: str = ""
    hint: str = ""
    level: Level = Level.BASIC
    also: tuple[str, ...] = ()
    check: Callable[[BaseConfig], bool | None] | None = None   # eigener Status-Test (für berechnete Änderungen)
    applies: Callable[[BaseConfig], bool] | None = None        # passt die Technik zu Modell/Daten/Paradigma?

    def changes_for(self, cfg: BaseConfig) -> dict[str, Any]:
        return self.changes(cfg) if callable(self.changes) else dict(self.changes)

    def in_category(self, cat: str) -> bool:
        return self.category == cat or cat in self.also


@dataclass
class TechniqueState:
    technique: Technique
    status: str                      # an | aus | teilweise | nicht_anwendbar
    paths: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        mark = {"an": "●", "aus": "○", "teilweise": "◐", "nicht_anwendbar": "–"}[self.status]
        t = self.technique
        extra = f"  ({t.used_by})" if t.used_by else ""
        return f"{mark} {t.key:<24} {t.title}{extra}"


# ===========================================================================
# Der Katalog
# ===========================================================================

def _gqa(cfg: BaseConfig) -> dict[str, Any]:
    heads = cfg.get("model.block.mixer.num_heads")
    return {"model.block.mixer.num_kv_heads": max(1, int(heads) // 4)}


def _gqa_on(cfg: BaseConfig) -> bool | None:
    try:
        kv, heads = cfg.get("model.block.mixer.num_kv_heads"), cfg.get("model.block.mixer.num_heads")
    except ConfigError:
        return None
    return kv is not None and kv < heads


def _accumulate(cfg: BaseConfig) -> dict[str, Any]:
    t = cfg.get("training")
    return {"training.batch_size": max(1, t.batch_size // 4),
            "training.gradient_accumulation_steps": t.gradient_accumulation_steps * 4}


def _accumulate_on(cfg: BaseConfig) -> bool:
    return cfg.get("training.gradient_accumulation_steps") > 1


def _arg(cfg: BaseConfig, name: str) -> Any:
    return getattr(cfg.get("model").args, name, None)


def _images(cfg: BaseConfig) -> bool:
    if cfg.get("data").name == "audio":                  # Spektrogramme: Spiegeln wäre Zeitumkehr
        return False
    return cfg.get("model").name in ("cnn", "unet") or _arg(cfg, "input_type") in ("image", "patches") or \
        cfg.get("data").name in ("shapes", "image_folder", "segmentation")


_TOKEN_DATA = ("text", "sequence_classification", "seq2seq_toy", "text_classification", "instructions")


def _tokens(cfg: BaseConfig) -> bool:
    return _arg(cfg, "input_type") == "tokens" or cfg.get("data").name in _TOKEN_DATA


def _supervised(cfg: BaseConfig) -> bool:
    return cfg.get("training.paradigm").name == "supervised"


def _float_supervised(cfg: BaseConfig) -> bool:
    """Überwacht mit Zahlen-Eingaben (Tabellen, Bilder, Signale) – Voraussetzung für Angriffe/Gradienten."""
    return _supervised(cfg) and not _tokens(cfg) and cfg.get("data").name not in ("graphs", "multimodal", "none")


def _generator(cfg: BaseConfig) -> bool:
    """Textgenerierung möglich (Decoder-Transformer auf Tokens)."""
    return cfg.get("model").name in ("transformer", "ssm", "hybrid") and _arg(cfg, "architecture") in (
        None, "decoder_only") and _arg(cfg, "input_type") in (None, "tokens") and _arg(cfg, "pooling") in (None, "none")


def _not_rl(cfg: BaseConfig) -> bool:
    return cfg.get("training.paradigm").name != "reinforcement"


#: Modelle mit Hauptblöcken (für Auslagern, Checkpointing, LISA, FSDP-Aufteilung).
_BLOCK_MODELS = ("transformer", "ssm", "hybrid", "cnn", "diffusion", "extern")


def _blocks(cfg: BaseConfig) -> bool:
    return cfg.get("model").name in _BLOCK_MODELS


def _linear_blocks(cfg: BaseConfig) -> bool:
    return cfg.get("model").name not in ("cnn", "gnn")


def _data_paradigm(cfg: BaseConfig) -> bool:
    """Paradigmen, deren Eingaben vollständig aus den Daten kommen (kein intern erzeugtes Rauschen)."""
    return cfg.get("training.paradigm").name in ("supervised", "causal_lm", "masked_lm", "seq2seq", "distillation",
                                                 "contrastive", "preference", "dpo")


def _classification(cfg: BaseConfig) -> bool:
    return cfg.get("training.paradigm").name == "supervised" and cfg.get("data").name not in (
        "regression", "time_series", "none")


def _K(key, title, cat, tip, changes, used_by="", hint="", level=Level.BASIC, also=(), check=None,
       applies=None) -> Technique:
    return Technique(key, title, cat, tip, changes, used_by, hint, level, also, check, applies)


CATALOG: list[Technique] = [
    # ------------------------------------------------------------------ Stabilität
    _K("gradient_clipping", "Gradient-Clipping (Gesamtnorm 1.0)", "stabilitaet", T(
        "Begrenzt die Länge des Gradienten pro Schritt.", "Einzelne Ausreißer-Batches zerstören das Modell nicht.",
        "Transformer, RNNs, RL – fast immer sinnvoll."),
        {"training.gradient_clipping.mode": "norm", "training.gradient_clipping.max_norm": 1.0},
        "GPT-3, LLaMA, fast alle großen Modelle"),
    _K("agc", "Adaptives Gradient-Clipping (AGC)", "stabilitaet", T(
        "Clipping relativ zur Größe der Gewichte jeder Einheit.", "Stabil auch ohne BatchNorm und bei großen Batches.",
        "Sehr tiefe CNNs, große Batches."), {"training.gradient_clipping.mode": "adaptive"}, "NFNets (DeepMind)",
        level=Level.ADVANCED),
    _K("warmup", "Lernraten-Warmup", "stabilitaet", T(
        "Lernrate in den ersten Prozent des Trainings langsam hochfahren.",
        "Adaptive Optimierer sind am Anfang instabil – Warmup verhindert frühe Divergenz.", "Transformer, große Batches."),
        {"training.scheduler.warmup_ratio": 0.05}, "BERT, GPT, ViT"),
    _K("nan_schutz", "NaN/Inf-Schutz", "stabilitaet", T(
        "Schritte mit ungültigen Werten überspringen statt die Gewichte zu zerstören.", "Robust bei fp16 und Ausreißern.",
        "Immer."), {"training.nan_policy": "skip"}, "Standard in Trainings-Frameworks"),
    _K("spike_schutz", "Schutz vor Verlust-Spitzen", "stabilitaet", T(
        "Erkennt plötzliche Verlust-Spitzen und überspringt diese Schritte.",
        "Ein schlechter Batch kann sonst Stunden an Training kosten.", "Lange Läufe, verrauschte Daten."),
        {"training.stability.spike_guard": True}, "große LLM-Läufe (PaLM, OPT)", applies=_not_rl),
    _K("spike_rueckspung", "Verlust-Spitzen: Rücksprung + kleinere Lernrate", "stabilitaet", T(
        "Springt nach einer Spitze auf den letzten guten Stand zurück und senkt die Lernrate.",
        "Repariert ein bereits 'abgerutschtes' Training automatisch.", "Teure Läufe, hohe Lernraten."),
        {"training.stability.spike_guard": True, "training.stability.spike_action": "rollback"},
        "OPT-175B-Logbuch (Meta): manuelle Rücksprünge – hier automatisch", level=Level.ADVANCED),
    _K("ema", "EMA der Gewichte", "stabilitaet", T(
        "Gleitender Mittelwert der Gewichte für Auswertung und Einsatz.", "Glattere, oft bessere Modelle.",
        "Diffusion, GANs, Bilderkennung."), {"training.ema.enabled": True}, "Stable Diffusion, YOLO, timm",
        also=("generalisierung",)),
    _K("qk_norm", "QK-Normalisierung in der Attention", "stabilitaet", T(
        "Queries und Keys vor dem Skalarprodukt normalisieren.", "Verhindert explodierende Attention-Logits.",
        "Große/tiefe Transformer, hohe Lernraten."), {"model.**.qk_norm": "rmsnorm"}, "ViT-22B, Gemma 3, OLMo 2",
        level=Level.ADVANCED),
    _K("z_loss", "z-Loss für Ausgabe-Logits", "stabilitaet", T(
        "Kleine Strafe auf die Größe der Logits (log Z²).", "Hält die Ausgabeschicht numerisch stabil.",
        "Sprachmodelle, bf16-Training."), {"training.paradigm.z_loss_coef": 1e-4}, "PaLM, OLMo", level=Level.ADVANCED),
    _K("logit_softcap", "Logit-Soft-Capping", "stabilitaet", T(
        "Logits weich auf ±30 begrenzen (tanh).", "Verhindert Überkonfidenz und Instabilität.", "Sprachmodelle."),
        {"model.logit_softcap": 30.0}, "Gemma 2", level=Level.EXPERT),
    _K("plateau_lr", "Lernrate bei Stillstand senken", "stabilitaet", T(
        "Lernrate halbieren, wenn sich die Validierung nicht mehr verbessert.", "Einfach und robust.",
        "Klassische CNNs, Tabellen."), {"training.scheduler": "plateau"}, "ReduceLROnPlateau (Keras/PyTorch)",
        level=Level.ADVANCED),

    # ------------------------------------------------------------------ Gegen Auswendiglernen
    _K("dropout", "Dropout (0.1)", "generalisierung", T(
        "Zufällig Teile der Aktivierungen im Training abschalten.", "Das Netz verlässt sich nicht auf Einzelne.",
        "Kleine/mittlere Datensätze."), {"model.**.dropout": 0.1}, "AlexNet, BERT, fast alle"),
    _K("drop_path", "Stochastic Depth / DropPath (0.1)", "generalisierung", T(
        "Ganze Residual-Blöcke zufällig überspringen.", "Starke Regularisierung tiefer Netze.", "ViT, ConvNeXt."),
        {"model.**.drop_path": 0.1}, "ViT, ConvNeXt, Swin", level=Level.ADVANCED),
    _K("weight_decay", "Weight Decay (0.05)", "generalisierung", T(
        "Gewichte bei jedem Schritt leicht schrumpfen lassen (AdamW).", "Einfachere, robustere Lösungen.",
        "Fast immer."), {"training.optimizer.weight_decay": 0.05}, "AdamW-Standard (Transformer, ViT)"),
    _K("label_smoothing", "Label Smoothing (0.1)", "generalisierung", T(
        "Ziele leicht 'aufweichen' (90 % richtige Klasse, Rest verteilt).", "Weniger Überkonfidenz, bessere Kalibrierung.",
        "Klassifikation, Übersetzung."), {"training.**.label_smoothing": 0.1}, "Inception, Transformer (Vaswani)"),
    _K("mixup", "Mixup", "generalisierung", T(
        "Beispiele und Ziele paarweise mischen.", "Glattere Entscheidungsgrenzen.", "Bilder, Tabellen."),
        {"training.regularization.mixup_alpha": 0.2}, "timm-/ImageNet-Rezepte", also=("daten",), applies=_not_rl),
    _K("cutmix", "CutMix", "generalisierung", T(
        "Bildausschnitt eines anderen Beispiels einsetzen, Ziele anteilig mischen.", "Stärker als Mixup bei Bildern.",
        "Bildklassifikation."), {"training.augmentation.cutmix_alpha": 1.0}, "timm, YOLO", also=("daten",), applies=_images),
    _K("early_stopping", "Early Stopping", "generalisierung", T(
        "Training beenden, wenn sich die Validierung nicht mehr verbessert; beste Gewichte behalten.",
        "Stoppt, bevor das Auswendiglernen beginnt.", "Fast jedes überwachte Training."),
        {"+training.callbacks": {"name": "early_stopping"}}, "Keras/Lightning-Standard"),
    _K("sam", "Sharpness-Aware Minimization (SAM)", "generalisierung", T(
        "Sucht flache Minima, indem auch die 'schlechteste' Nachbarstelle betrachtet wird.",
        "Deutlich bessere Generalisierung; doppelte Rechenzeit.", "Bilder ohne Vortraining."),
        {"training.regularization.sam_rho": 0.05}, "Google (ViT/ResNet ohne Vortraining)", level=Level.ADVANCED),
    _K("swa", "Stochastic Weight Averaging (SWA)", "generalisierung", T(
        "Gewichte der späten Trainingsphase mitteln.", "Flachere Minima, fast kostenlos.", "Bilder, Tabellen."),
        {"+training.callbacks": {"name": "swa"}}, "PyTorch SWA, Kaggle-Lösungen", level=Level.ADVANCED),
    _K("flooding", "Flooding", "generalisierung", T(
        "Trainingsverlust darf einen Pegel nicht unterschreiten.", "Verhindert Auswendiglernen kleiner Datensätze.",
        "Sehr kleine Datensätze."), {"training.regularization.flooding_level": 0.05}, "Ishida et al. (2020)",
        level=Level.EXPERT),
    _K("eingaberauschen", "Rauschen auf den Eingaben", "generalisierung", T(
        "Gauß-Rauschen auf Eingaben im Training.", "Robust gegen Messrauschen.", "Sensor- und Tabellendaten."),
        {"training.regularization.input_noise_std": 0.05}, "klassische Regularisierung", also=("daten",),
        applies=lambda c: _not_rl(c) and not _tokens(c)),
    _K("l1_sparsity", "L1-Regularisierung (spärliche Gewichte)", "generalisierung", T(
        "Strafe auf |w| – viele Gewichte werden exakt 0.", "Einfache, kompakte Modelle.", "Merkmalsauswahl."),
        {"training.regularization.l1": 1e-5}, "Lasso (Statistik)", level=Level.ADVANCED),
    _K("kreuzvalidierung", "k-fache Kreuzvalidierung (5)", "generalisierung", T(
        "Fünf Modelle auf je vier Fünfteln trainieren und auf dem Rest messen.",
        "Ehrliche Einschätzung, wie gut das Modell wirklich verallgemeinert.", "Kleine Datensätze, Berichte."),
        {"workflow.cross_validation": 5}, "Kaggle, Medizin-Statistik", applies=_not_rl),

    # ------------------------------------------------------------------ Performance
    _K("bf16", "Gemischte Präzision bf16", "tempo", T(
        "Rechnen in 16 Bit (bfloat16), Gewichte in 32 Bit.", "Bis zu 2–3× schneller, halber Aktivierungsspeicher.",
        "GPU-Training (Ampere+), moderne CPUs."), {"memory.mixed_precision": "bf16"}, "Standard für LLM-Training",
        "GPU empfohlen", also=("grosse_modelle",)),
    _K("fp16", "Gemischte Präzision fp16 + Loss-Scaling", "tempo", T(
        "Rechnen in float16 mit automatischem Loss-Scaling.", "Schnell auf älteren GPUs ohne bf16.",
        "Volta/Turing-GPUs."), {"memory.mixed_precision": "fp16"}, "NVIDIA Apex/AMP", "nur GPU", level=Level.ADVANCED),
    _K("compile", "torch.compile", "tempo", T(
        "Modell zu optimierten Kerneln übersetzen.", "Oft 1.3–2× schneller nach dem Aufwärmen.", "Lange Trainings."),
        {"memory.compile": True}, "PyTorch 2", "GPU empfohlen", level=Level.ADVANCED),
    _K("tf32", "TF32-Matrixmultiplikation", "tempo", T(
        "fp32-Matrixmultiplikation in TF32 auf Tensor-Cores.", "Bis zu 3× schneller, kaum Genauigkeitsverlust.",
        "NVIDIA Ampere+."), {"memory.matmul_precision": "high"}, "NVIDIA-Standard", "GPU", level=Level.ADVANCED),
    _K("channels_last", "Speicherlayout channels_last", "tempo", T(
        "Bilder im Format NHWC speichern.", "Schnellere Faltungen auf Tensor-Cores.", "CNNs auf GPU."),
        {"memory.channels_last": True}, "NVIDIA, timm", "GPU, CNN", level=Level.EXPERT, applies=_images),
    _K("cudnn_benchmark", "cuDNN-Benchmark", "tempo", T(
        "Schnellsten Faltungs-Algorithmus je Eingabeform ausprobieren.", "Schnellere CNNs bei festen Formen.",
        "CNN-Training auf NVIDIA."), {"memory.cudnn_benchmark": True}, "torchvision-Referenzskripte", "GPU",
        level=Level.ADVANCED),
    _K("daten_parallel_laden", "Daten parallel laden", "tempo", T(
        "Mehrere Hintergrundprozesse laden und gepinnter Speicher.", "Die GPU wartet nicht auf Daten.",
        "Große Datensätze, Bilder von der Platte."), {"training.num_workers": 4, "training.pin_memory": True},
        "PyTorch-DataLoader-Praxis", level=Level.ADVANCED),
    _K("fused_optimizer", "Fusionierter Optimierer", "tempo", T(
        "Optimierer-Schritt in einem einzigen GPU-Kernel.", "Spürbar schneller bei vielen Parametern.", "CUDA."),
        {"training.optimizer.fused": True}, "PyTorch AdamW(fused=True)", "nur CUDA", level=Level.EXPERT),
    _K("gqa", "Grouped-Query-Attention", "tempo", T(
        "Mehrere Query-Köpfe teilen sich Key/Value-Köpfe.", "KV-Cache und Projektionen ÷ 4 – schnellere Generierung.",
        "Sprachmodelle."), _gqa, "LLaMA 2/3, Mistral, Gemma", also=("grosse_modelle",), check=_gqa_on,
        level=Level.ADVANCED),

    # ------------------------------------------------------------------ Große Modelle, kleine Hardware
    _K("checkpointing", "Gradient-Checkpointing", "grosse_modelle", T(
        "Aktivierungen im Backward neu berechnen statt speichern.", "60–80 % weniger Aktivierungsspeicher, ~30 % "
        "langsamer.", "Tiefe Modelle, lange Sequenzen."), {"memory.gradient_checkpointing": True},
        "Standard für große Modelle (Hugging Face, Megatron)"),
    _K("akkumulation", "Kleine Batches + Gradientenakkumulation", "grosse_modelle", T(
        "Batch vierteln und 4 Mikro-Batches akkumulieren.", "Gleiche effektive Batchgröße mit ¼ Aktivierungsspeicher.",
        "Wenig GPU-Speicher."), _accumulate, "überall, wo der Speicher knapp ist", check=_accumulate_on),
    _K("adamw8bit", "8-Bit-AdamW", "grosse_modelle", T(
        "Optimierer-Zustände in 8 Bit.", "Optimierer-Speicher ÷ 4 bei gleicher Qualität.", "Große Modelle."),
        {"training.optimizer.name": "adamw8bit"}, "bitsandbytes (Hugging Face)"),
    _K("adafactor", "Adafactor", "grosse_modelle", T(
        "Faktorisierte zweite Momente statt voller Matrizen.", "Optimierer-Speicher fast null.", "Sehr große Modelle."),
        {"training.optimizer.name": "adafactor"}, "T5, PaLM (Google)", level=Level.ADVANCED),
    _K("lora", "LoRA-Adapter", "grosse_modelle", T(
        "Basis einfrieren, kleine Zusatzmatrizen trainieren.", "Gradienten/Optimierer nur für ~1 % der Parameter.",
        "Feintuning großer Modelle."), {"memory.lora.enabled": True}, "Microsoft LoRA, Hugging Face PEFT",
        applies=_linear_blocks),
    _K("qlora", "QLoRA (4-Bit-Basis + LoRA)", "grosse_modelle", T(
        "Basis in 4 Bit (nf4), Adapter in voller Präzision.", "Feintuning von Modellen, die sonst nie passen würden.",
        "Große Sprachmodelle auf einer GPU."), {"memory.quantization": "nf4", "memory.lora.enabled": True},
        "QLoRA (Dettmers et al. 2023)", applies=_linear_blocks),
    _K("int8_gewichte", "Gewichte in int8", "grosse_modelle", T(
        "Lineare Schichten in 8 Bit speichern.", "Gewichtsspeicher ÷ 4.", "Inferenz, eingefrorene Teile."),
        {"memory.quantization": "int8"}, "LLM.int8 (bitsandbytes)", also=("kompression",)),
    _K("aktivierungen_auslagern", "Aktivierungen in den RAM auslagern", "grosse_modelle", T(
        "Für den Backward gespeicherte Aktivierungen im CPU-RAM ablegen.", "Spart GPU-Speicher ohne Neuberechnung.",
        "Lange Sequenzen auf kleiner GPU."), {"memory.activation_offload": True}, "PyTorch save_on_cpu",
        level=Level.ADVANCED),
    _K("stream_ram", "Blöcke trainierbar im RAM (Streaming)", "grosse_modelle", T(
        "Gewichte, Gradienten und Optimierer der Blöcke im RAM; aufs Gerät nur der rechnende Block.",
        "Training von Modellen größer als der GPU-Speicher.", "Große Modelle, kleine GPU."),
        {"memory.offload.mode": "stream"}, "Prinzip von DeepSpeed ZeRO-Offload", level=Level.ADVANCED,
        applies=_blocks),
    _K("stream_ssd", "Eingefrorene Basis auf der SSD (Streaming)", "grosse_modelle", T(
        "Eingefrorene Gewichte liegen als Datei auf der SSD und werden blockweise gelesen.",
        "Modelle größer als der RAM (z. B. mit LoRA) trainierbar.", "Sehr große Modelle."),
        {"memory.offload.mode": "stream", "memory.offload.storage": "disk"}, "Prinzip von ZeRO-Infinity",
        level=Level.EXPERT, applies=_blocks),
    _K("split_cpu", "Teile auf der CPU rechnen (device_map)", "grosse_modelle", T(
        "Ausgewählte Blöcke rechnen dauerhaft auf der CPU, der Rest auf der GPU.", "Es läuft, auch wenn es nicht passt.",
        "GPU knapp zu klein."), {"memory.offload.mode": "split"}, "Hugging Face accelerate device_map",
        level=Level.ADVANCED, applies=_blocks),
    _K("layerweise", "Schicht für Schicht von der SSD (Inferenz)", "grosse_modelle", T(
        "Jede Schicht wird nur zum Rechnen geladen und sofort wieder freigegeben.",
        "Riesige Modelle auch auf kleinen Rechnern ausführen (langsam).", "Batch-Auswertungen großer Modelle."),
        {"memory.offload.mode": "disk", "memory.offload.keep_resident": 0}, "AirLLM", "nur Inferenz (vor dem Training "
        "ausschalten oder Blöcke einfrieren)", level=Level.ADVANCED, applies=_blocks),
    _K("lisa", "LISA: wenige Blöcke gleichzeitig trainieren", "grosse_modelle", T(
        "Immer nur zwei zufällig wechselnde Blöcke sind trainierbar.", "Speicher fast wie LoRA, volle Gewichte.",
        "Feintuning großer Modelle."), {"+training.callbacks": {"name": "lisa"}}, "LISA (Pan et al. 2024)",
        level=Level.ADVANCED, applies=_blocks),
    _K("oom_schutz", "Speichermangel: Batch automatisch teilen", "grosse_modelle", T(
        "Bei Out-of-Memory den Batch teilen und per Akkumulation weiterrechnen.", "Kein Abbruch mitten im Training.",
        "Unbekannte Hardware."), {"memory.oom_strategy": "batch_teilen"}, "Lightning auto_scale (Prinzip)",
        also=("stabilitaet",)),
    _K("bf16_gewichte", "Gewichte komplett in bf16", "grosse_modelle", T(
        "Gewichte und Gradienten in 16 Bit.", "Halber Speicher für Gewichte und Gradienten.", "Große Modelle."),
        {"memory.param_dtype": "bfloat16"}, "reines bf16-Training großer LLMs", level=Level.ADVANCED,
        applies=_data_paradigm),

    # ------------------------------------------------------------------ Cluster
    _K("ddp", "Data Parallel (DDP)", "cluster", T(
        "Jede GPU/jeder Prozess rechnet einen Teil der Daten, Gradienten werden gemittelt.",
        "Fast lineare Beschleunigung mit mehr GPUs.", "Mehrere GPUs/Rechner."),
        {"training.distributed.strategy": "ddp"}, "PyTorch-Standard", "Start mit 'verteilt'"),
    _K("zero1", "ZeRO-1 (Optimierer verteilt)", "cluster", T(
        "Wie DDP, Optimierer-Zustände auf die Prozesse verteilt.", "Adam-Speicher ÷ Anzahl GPUs.", "Große Modelle."),
        {"training.distributed.strategy": "zero1"}, "DeepSpeed ZeRO", "Start mit 'verteilt'", level=Level.ADVANCED,
        also=("grosse_modelle",)),
    _K("fsdp", "FSDP (alles verteilt)", "cluster", T(
        "Gewichte, Gradienten und Optimierer werden verteilt.", "Modelle, die auf keine einzelne GPU passen.",
        "Sehr große Modelle."), {"training.distributed.strategy": "fsdp"}, "Meta (LLaMA-Training)",
        "Start mit 'verteilt'", level=Level.ADVANCED, also=("grosse_modelle",)),
    _K("fsdp_cpu", "FSDP + CPU-Auslagerung", "cluster", T(
        "FSDP mit verteilten Gewichten/Optimierern im CPU-RAM.", "Noch größere Modelle, langsamer.", "Extremfälle."),
        {"training.distributed.strategy": "fsdp", "training.distributed.fsdp_cpu_offload": True}, "PyTorch FSDP",
        level=Level.EXPERT),
    _K("sync_batchnorm", "Synchronisierte BatchNorm", "cluster", T(
        "BatchNorm-Statistiken über alle GPUs.", "Korrekt bei kleinen Batches pro GPU.", "CNNs verteilt."),
        {"training.distributed.sync_batchnorm": True}, "Detectron2, MMDetection", level=Level.ADVANCED),

    # ------------------------------------------------------------------ Automatische Suche
    _K("zufallssuche", "Zufallssuche", "automl", T(
        "Zufällige Einstellungen ausprobieren, die beste übernehmen.", "Einfach und erstaunlich stark.",
        "Erste Abstimmung."), {"workflow.search.method": "random"}, "Bergstra & Bengio (2012)"),
    _K("gittersuche", "Gittersuche", "automl", T(
        "Alle Kombinationen weniger Werte ausprobieren.", "Vollständig und nachvollziehbar.", "Wenige Einstellungen."),
        {"workflow.search.method": "grid"}, "scikit-learn GridSearch"),

    # ------------------------------------------------------------------ Schwarmintelligenz
    _K("partikelschwarm", "Partikelschwarm-Optimierung (PSO)", "schwarm", T(
        "Ein Schwarm von Einstellungs-'Partikeln' fliegt zum eigenen und zum besten Fund des Schwarms.",
        "Nutzt kollektives Wissen; findet gute Bereiche schneller als Zufall.", "Hyperparameter-Suche."),
        {"workflow.search.method": "pso"}, "Kennedy & Eberhart (1995)", also=("automl",)),
    _K("foederiert", "Föderiertes Lernen (FedAvg)", "schwarm", T(
        "Viele Teilnehmer trainieren lokal, nur Gewichte werden zusammengeführt.", "Daten verlassen nie das Gerät.",
        "Verteilte, sensible Daten."), {"workflow.federated.enabled": True}, "Google Gboard, Apple",
        also=("datenschutz",), applies=_not_rl),
    _K("fedprox", "FedProx (robust bei ungleichen Teilnehmern)", "schwarm", T(
        "Föderiert mit Strafterm, der lokale Modelle nahe am globalen hält.", "Stabil bei sehr unterschiedlichen Daten.",
        "Nicht-iid-Daten."), {"workflow.federated.enabled": True, "workflow.federated.algorithm": "fedprox"},
        "Li et al. (2020)", level=Level.ADVANCED, applies=_not_rl),
    _K("model_soup", "Model Soup (mehrere Läufe mitteln)", "schwarm", T(
        "Mehrere Läufe vom selben Startpunkt trainieren und die Gewichte mitteln.",
        "Genauer und robuster – ohne Mehrkosten beim Einsatz.", "Feintuning."), {"workflow.soup_runs": 4},
        "Google (Wortsman et al. 2022)", also=("generalisierung",), level=Level.ADVANCED, applies=_not_rl),

    # ------------------------------------------------------------------ Evolution
    _K("genetisch", "Genetischer Algorithmus", "evolution", T(
        "Die besten Einstellungen überleben, werden gekreuzt und mutiert.", "Findet Kombinationen jenseits der Intuition.",
        "Hyperparameter-/Architektursuche."), {"workflow.search.method": "evolution"}, "Neuroevolution (Uber, Google)",
        also=("automl",)),
    _K("pbt", "Population Based Training", "evolution", T(
        "Eine Population trainiert gleichzeitig; Schlechte übernehmen Gewichte und Einstellungen der Guten.",
        "Einstellungen passen sich während des Trainings an – ein Lauf statt vieler.", "Lange Trainings."),
        {"workflow.search.method": "pbt"}, "DeepMind (AlphaStar), Waymo", also=("automl",), level=Level.ADVANCED),
    _K("evolution_strategies", "Evolution Strategies (ohne Backward)", "evolution", T(
        "Gewichte zufällig stören und aus der Fitness die Richtung schätzen.",
        "Gradientenfrei – auch für nicht differenzierbare Ziele, trivial parallel.", "Kleine Modelle, Forschung."),
        {"training.paradigm": "evolution_strategies"}, "OpenAI-ES (Salimans et al. 2017)", level=Level.EXPERIMENTAL,
        applies=lambda c: c.get("training.paradigm").name in ("supervised", "causal_lm", "evolution_strategies")),

    # ------------------------------------------------------------------ Daten
    _K("bild_augmentierung", "Bild-Augmentierung (Spiegeln + Ausschnitt)", "daten", T(
        "Bilder zufällig spiegeln und verschieben.", "Mehr Vielfalt aus denselben Daten.", "Bildklassifikation."),
        {"training.augmentation.hflip": True, "training.augmentation.crop_padding": 4}, "CIFAR/ImageNet-Standard",
        also=("generalisierung",), applies=_images),
    _K("random_erasing", "Random Erasing (Cutout)", "daten", T(
        "Zufällige Rechtecke im Bild löschen.", "Robust gegen Verdeckung.", "Bilder."),
        {"training.augmentation.random_erasing": 0.25}, "Zhong et al., timm", level=Level.ADVANCED,
        also=("generalisierung",), applies=_images),
    _K("token_rauschen", "Token-Rauschen", "daten", T(
        "Einige Eingabe-Tokens zufällig ersetzen.", "Robust gegen Tippfehler.", "Textklassifikation."),
        {"training.augmentation.token_noise": 0.05}, "BERT-Maskierung (Prinzip)", level=Level.ADVANCED,
        applies=lambda c: _tokens(c) and _classification(c)),
    _K("klassen_ausgleich", "Seltene Klassen ausgleichen", "daten", T(
        "Seltene Klassen häufiger ziehen.", "Unausgewogene Daten werden fair gelernt.", "Betrug, Diagnosen."),
        {"training.balance_classes": True}, "Standard bei unbalancierten Daten", applies=_classification),
    _K("bpe_tokenizer", "BPE-Tokenizer", "daten", T(
        "Häufige Zeichenfolgen werden zu einem Token zusammengefasst (wie GPT/LLaMA).",
        "Kürzere Folgen als Zeichen-Tokens, keine unbekannten Wörter wie beim Wort-Tokenizer.",
        "Texte mit genug Trainingsmaterial."), {"data.tokenizer": "bpe"}, "GPT-2, LLaMA, RoBERTa",
        applies=lambda cfg: hasattr(cfg.get("data").args, "tokenizer")),
    _K("curriculum", "Curriculum (leicht → schwer)", "daten", T(
        "Zuerst kurze/leichte Beispiele, dann alle.", "Schnellerer, stabilerer Start.", "Sprachmodelle, Logik."),
        {"training.curriculum.strategy": "length"}, "Bengio et al. (2009)", level=Level.ADVANCED, applies=_not_rl),

    # ------------------------------------------------------------------ Datenschutz
    _K("dp_sgd", "Differential Privacy (DP-SGD)", "datenschutz", T(
        "Gradient jedes Beispiels begrenzen und verrauschen.", "Das Modell merkt sich keine Einzelpersonen.",
        "Medizin, Finanzen, Nutzerdaten."), {"training.privacy.enabled": True}, "Google, Apple, US-Zensus",
        "langsam (jedes Beispiel einzeln)", level=Level.ADVANCED,
        applies=lambda c: c.get("training.paradigm").name in ("supervised", "causal_lm", "masked_lm", "seq2seq")),

    # ------------------------------------------------------------------ Überwachung & Reproduzierbarkeit
    _K("checkpoints", "Checkpoints (bestes Modell sichern)", "ueberwachung", T(
        "Bestes Modell während des Trainings speichern.", "Kein Verlust bei Abbrüchen.", "Längere Läufe."),
        {"+training.callbacks": {"name": "checkpoint"}}, "Standard"),
    _K("seed", "Fester Zufalls-Seed", "ueberwachung", T(
        "Alle Zufallsquellen festlegen.", "Ergebnisse wiederholbar.", "Vergleiche, Berichte."),
        {"training.seed": 42}, "wissenschaftliche Praxis"),
    _K("deterministisch", "Deterministische Algorithmen", "ueberwachung", T(
        "Nur bitgenau wiederholbare Kernel verwenden.", "Exakte Reproduzierbarkeit (langsamer).", "Fehlersuche."),
        {"training.deterministic": True}, "Audit-/Medizin-Anforderungen", level=Level.EXPERT),
    _K("log_datei", "Protokoll-Datei mit Rotation", "ueberwachung", T(
        "Textprotokoll mit Zeitstempeln, begrenzte Dateigröße.", "Nachlesen auf Servern ohne Terminal.",
        "Lange Läufe."), {"+training.logging.backends": "datei"}, "Server-Betrieb"),
    _K("log_jsonl", "Maschinenlesbares Protokoll (JSONL)", "ueberwachung", T(
        "Alle Metriken und Meldungen als JSON-Zeilen.", "Eigene Auswertungen, Dashboards.", "Analyse."),
        {"+training.logging.backends": "jsonl"}, "MLOps-Praxis"),
    _K("tensorboard", "TensorBoard", "ueberwachung", T(
        "Kurven, Histogramme und Einstellungen im Browser.", "Interaktiver Vergleich von Läufen.", "Experimente."),
        {"+training.logging.backends": "tensorboard"}, "Google/TensorFlow, PyTorch", "Paket 'tensorboard'",
        level=Level.ADVANCED),
    _K("wandb", "Weights & Biases", "ueberwachung", T(
        "Läufe im Team-Dashboard.", "Vergleich vieler Läufe, Teilen von Ergebnissen.", "Teams."),
        {"+training.logging.backends": "wandb"}, "OpenAI, viele Forschungsteams", "Paket 'wandb'", level=Level.ADVANCED),
    _K("mlflow", "MLflow-Tracking", "ueberwachung", T(
        "Experimente und Modelle zentral verwalten.", "Nachvollziehbarkeit im Unternehmen.", "MLOps."),
        {"+training.logging.backends": "mlflow"}, "Databricks, Unternehmen", "Paket 'mlflow'", level=Level.ADVANCED),
    _K("gradienten_ueberwachung", "Gradienten je Schicht überwachen", "ueberwachung", T(
        "Gradientennormen jedes Tensors regelmäßig protokollieren.", "Findet tote oder explodierende Schichten.",
        "Fehlersuche."), {"training.logging.gradients_every_n_steps": 100}, "W&B watch, TensorBoard",
        level=Level.ADVANCED),
    _K("gewichts_ueberwachung", "Gewichte je Schicht überwachen", "ueberwachung", T(
        "Norm/Streuung jedes Gewichtstensors protokollieren (TensorBoard: Histogramme).", "Sieht, welche Schichten "
        "lernen.", "Analyse."), {"training.logging.weights_every_n_steps": 200}, "TensorBoard-Histogramme",
        level=Level.ADVANCED),
    _K("zeitlimit", "Zeitlimit", "ueberwachung", T(
        "Training nach fester Zeit sauber beenden.", "Rechenzeit-Budgets einhalten.", "Cluster-Slots."),
        {"+training.callbacks": {"name": "time_limit"}}, "SLURM-Praxis", level=Level.ADVANCED),

    # ------------------------------------------------------------------ Kompression & Einsatz
    _K("komprimieren_nach_training", "Nach dem Training komprimieren", "kompression", T(
        "compression.steps (oder die Empfehlung für das Hauptziel) automatisch anwenden.",
        "Fertiges Modell direkt verkleinert.", "Einsatz auf Geräten."), {"workflow.compress_after_training": True},
        "Edge-Deployment"),
    _K("export_nach_training", "Nach dem Training exportieren", "kompression", T(
        "Am Ende als torch.export-Datei speichern.", "Einsatzfertige Datei.", "Produktion."),
        {"workflow.export_format": "export"}, "PyTorch 2"),
    _K("qat", "Quantization-Aware Training", "kompression", T(
        "Im Training bereits mit gerundeten Gewichten rechnen.", "Kaum Qualitätsverlust bei späterer Quantisierung.",
        "Modelle für int8-Geräte."), {"memory.quantization": "fake_quant"}, "Google, Qualcomm (TFLite, AIMET)",
        level=Level.ADVANCED),

    # ------------------------------------------------------------------ Bewertung & Vertrauen
    _K("testmenge", "Testmenge zurückhalten (15 %)", "vertrauen", T(
        "Ein Teil der Daten wird nie fürs Training oder die Auswahl benutzt und erst am Ende geprüft (Project.test).",
        "Validierungswerte sind durch Early Stopping und Suche leicht geschönt – nur der Test zeigt die echte "
        "Qualität.", "Jedes ernsthafte Projekt mit genug Daten."),
        {"training.test_split": 0.15}, "Standard in Forschung und Industrie", applies=_data_paradigm),
    _K("metriken_auto", "Passende Metriken automatisch", "vertrauen", T(
        "F1, AUROC, Kalibrierungsfehler (Klassifikation), MAE/RMSE/R² (Regression), Perplexität, BLEU, IoU …",
        "Der Verlust allein sagt wenig über die echte Qualität.", "Immer."),
        {"training.auto_metrics": True}, "scikit-learn/torchmetrics-Praxis", applies=_data_paradigm),
    _K("stratifiziert", "Stratifizierte Aufteilung", "vertrauen", T(
        "Klassenanteile in Training, Validierung und Test gleich halten.", "Faire Bewertung, besonders bei seltenen "
        "Klassen.", "Klassifikation."), {"training.validation_strategy": "stratifiziert"}, "scikit-learn "
        "StratifiedKFold", also=("daten",), applies=_classification),
    _K("zeitliche_aufteilung", "Zeitliche Aufteilung (keine Zukunft im Training)", "vertrauen", T(
        "Die letzten Beispiele werden Validierung/Test.", "Zufälliges Mischen verrät bei Zeitreihen die Zukunft – "
        "die Bewertung wäre geschönt.", "Prognosen, Zeitreihen, Logs."), {"training.validation_strategy": "zeitlich"},
        "Prognose-Wettbewerbe (M4/M5)", level=Level.ADVANCED, applies=_data_paradigm),
    _K("kalibrierung", "Wahrscheinlichkeiten kalibrieren (Temperature Scaling)", "vertrauen", T(
        "Nach dem Training eine Temperatur auf der Validierung einstellen.",
        "'90 % sicher' soll in 90 % der Fälle stimmen – moderne Netze sind oft überkonfident.",
        "Entscheidungen mit Risiko, Schwellenwerte."), {"inference.auto_calibrate": True}, "Guo et al. (2017)",
        applies=_classification),
    _K("schwelle_optimieren", "Entscheidungsschwelle optimieren (F1)", "vertrauen", T(
        "Bei zwei Klassen die Schwelle wählen, die F1 auf der Validierung maximiert.", "0.5 ist bei seltenen "
        "Positiven selten optimal.", "Betrug, Diagnosen, Fehlererkennung."),
        {"inference.auto_calibrate": True, "inference.optimize_threshold": "f1"}, "Praxis in Kreditscoring/Medizin",
        level=Level.ADVANCED, applies=_classification),
    _K("enthaltung", "Bei Unsicherheit enthalten ('unsicher')", "vertrauen", T(
        "Liegt die Konfidenz unter 70 %, lautet die Antwort 'unsicher' statt zu raten.",
        "Unsichere Fälle gehen an einen Menschen – weniger teure Fehler.", "Medizin, Qualitätskontrolle, Support."),
        {"inference.abstain_below": 0.7}, "Selective Prediction / Reject Option", level=Level.ADVANCED,
        applies=_classification),
    _K("conformal", "Conformal Prediction (90 % garantiert)", "vertrauen", T(
        "Statt einer Antwort eine Menge (Klassen) bzw. ein Intervall (Zahlen), das in 90 % der Fälle stimmt.",
        "Statistisch garantierte Unsicherheitsangabe – für jedes Modell, ohne Annahmen.", "Risiko-Entscheidungen."),
        {"inference.conformal_alpha": 0.1, "inference.auto_calibrate": True}, "Angelopoulos & Bates (2021)",
        level=Level.ADVANCED, applies=_supervised),
    _K("mc_dropout", "Unsicherheit per MC-Dropout", "vertrauen", T(
        "Mehrere Vorhersagen mit aktivem Dropout; die Streuung zeigt die Unsicherheit.",
        "Erkennt Eingaben, die das Modell nicht kennt.", "Modelle mit Dropout."),
        {"inference.uncertainty": "mc_dropout"}, "Gal & Ghahramani (2016)", level=Level.ADVANCED, applies=_supervised),
    _K("tta", "Test-Time Augmentation", "vertrauen", T(
        "Vorhersage über das Original und gespiegelte Varianten mitteln.", "Meist 0,5–2 Punkte genauer und "
        "stabiler.", "Bilder."), {"inference.uncertainty": "tta"}, "Kaggle-Standard", level=Level.ADVANCED,
        applies=_images),
    _K("adversarial_training", "Adversariales Training (PGD)", "vertrauen", T(
        "Zusätzlich auf gezielt gestörten Eingaben trainieren, die das Modell am stärksten täuschen.",
        "Robust gegen Angriffe und Rauschen; wirkt auch als Regularisierung.", "Sicherheitskritische Modelle."),
        {"training.adversarial.method": "pgd"}, "Madry et al. (2018)", level=Level.ADVANCED,
        also=("generalisierung",), applies=_float_supervised),
    _K("fgsm_training", "Adversariales Training (FGSM, schnell)", "vertrauen", T(
        "Ein-Schritt-Variante des adversarialen Trainings.", "Fast so robust bei viel weniger Rechenzeit.",
        "Einstieg in Robustheit."), {"training.adversarial.method": "fgsm"}, "Goodfellow et al. (2015), "
        "Wong et al. (2020)", level=Level.ADVANCED, applies=_float_supervised),
    _K("klassengewichte", "Klassengewichte im Verlust", "daten", T(
        "Seltene Klassen zählen im Verlust stärker (N / (K · n_Klasse)).", "Unausgewogene Daten werden fair gelernt "
        "– ohne die Daten zu verändern.", "Betrug, Diagnosen, seltene Fehler."),
        {"training.class_weights": "ausgeglichen"}, "scikit-learn class_weight='balanced'",
        also=("vertrauen",), applies=_classification),
    _K("beam_search", "Beam Search bei der Generierung", "vertrauen", T(
        "Mehrere Fortsetzungen parallel verfolgen und die wahrscheinlichste Gesamtfolge wählen.",
        "Bessere Antworten bei Übersetzung, Zusammenfassung und Aufgaben mit fester Lösung.", "Nicht für kreative "
        "Texte."), {"generation.num_beams": 4}, "Übersetzungssysteme (Marian, T5)", level=Level.ADVANCED,
        applies=_generator),
    _K("keine_wiederholungen", "Keine Wiederholungs-Schleifen (n-Gramm-Sperre)", "vertrauen", T(
        "Kein 3-Gramm darf sich in der Ausgabe wiederholen.", "Verhindert die typischen Endlosschleifen kleiner "
        "Sprachmodelle.", "Textgenerierung."), {"generation.no_repeat_ngram_size": 3}, "Hugging Face generate()",
        level=Level.ADVANCED, applies=_generator),
    _K("datenpruefung", "Datenprüfung vor dem Training", "vertrauen", T(
        "Project.run prüft die Daten zuerst (Klassen, Lecks, Duplikate, NaN, Skalen …).",
        "Die meisten KI-Probleme sind Datenprobleme.", "Jedes Projekt."), {"workflow.check_data": True},
        "Google 'Data Validation for ML'", also=("daten",), applies=_data_paradigm),
    _K("robustheitspruefung", "Robustheitsprüfung nach dem Training", "vertrauen", T(
        "Project.run misst die Qualität unter Rauschen und gezielten Angriffen (FGSM/PGD).",
        "Zeigt, ob kleine Störungen das Modell aus der Bahn werfen.", "Sicherheitskritische Modelle."),
        {"workflow.robustness_check": True}, "Robustheits-Benchmarks (RobustBench)", level=Level.ADVANCED,
        applies=_float_supervised),
    _K("modellkarte", "Modellkarte beim Speichern", "ueberwachung", T(
        "MODELLKARTE.md und umgebung.json neben dem Modell ablegen.", "Zweck, Daten, Ergebnisse, Grenzen und "
        "Versionen dokumentiert – nachvollziehbar und prüfbar.", "Immer, besonders bei Weitergabe."),
        {"deployment.model_card": True}, "Mitchell et al. (2019), Hugging Face Model Cards"),
    _K("fortsetzen", "Automatisch fortsetzen (neuester Checkpoint)", "ueberwachung", T(
        "Das Training startet beim neuesten Checkpoint statt von vorn.", "Nach Abbruch, Stromausfall oder "
        "Zeitlimit geht keine Arbeit verloren.", "Lange Läufe, Cluster mit Zeitslots."),
        {"training.resume_from": "auto", "+training.callbacks": {"name": "checkpoint", "every_n_steps": 500,
                                                               "save_best_only": False}},
        "Lightning/Hugging Face 'resume_from_checkpoint'", level=Level.ADVANCED, applies=_not_rl),

    # ------------------------------------------------------------------ Experimentell
    _K("lookahead", "Lookahead-Optimierer", "experimentell", T(
        "Langsame Gewichte folgen den schnellen alle k Schritte ein Stück.", "Glatter, weniger lernratenabhängig.",
        "Schwierige Optimierung."), {"training.optimizer_extras.lookahead_k": 5}, "Zhang et al. (2019)",
        level=Level.ADVANCED),
    _K("gradient_centralization", "Gradient Centralization", "experimentell", T(
        "Mittelwert jeder Gradientenzeile abziehen.", "Oft schneller und stabiler bei CNNs.", "CNNs, MLPs."),
        {"training.optimizer_extras.gradient_centralization": True}, "Yong et al. (2020)", level=Level.ADVANCED),
    _K("grokfast", "Grokfast (schnelleres Grokking)", "experimentell", T(
        "Langsame Gradientenanteile verstärken.", "Späte Generalisierung kommt viel früher.", "Kleine Algorithmik-Daten."),
        {"training.optimizer_extras.grokfast_alpha": 0.98}, "Lee et al. (2024)", level=Level.EXPERIMENTAL,
        also=("generalisierung",)),
    _K("lion", "Lion-Optimierer", "experimentell", T(
        "Vorzeichen-basierter Optimierer aus einer automatischen Suche.", "Nur ein Moment – weniger Speicher als Adam.",
        "Große Bild-/Sprachmodelle."), {"training.optimizer.name": "lion"}, "Google (Chen et al. 2023)",
        level=Level.ADVANCED, also=("grosse_modelle",)),
    _K("muon", "Muon-Optimierer", "experimentell", T(
        "Orthogonalisierte Momentum-Updates für Gewichtsmatrizen.", "Schnelleres Training großer Transformer.",
        "Sprachmodelle."), {"training.optimizer.name": "muon"}, "Moonshot AI / Kimi (2025)", level=Level.EXPERIMENTAL),
    _K("gradientenrauschen", "Gradientenrauschen", "experimentell", T(
        "Abklingendes Rauschen auf die Gradienten.", "Hilft aus schlechten Minima.", "Sehr tiefe Netze."),
        {"training.regularization.gradient_noise_std": 0.01}, "Neelakantan et al. (2015)", level=Level.EXPERIMENTAL),
    _K("orthogonalitaet", "Orthogonalitäts-Regularisierung", "experimentell", T(
        "Gewichtsmatrizen nahe orthogonal halten.", "Normerhaltende Signale, stabile GANs.", "GANs, tiefe Netze."),
        {"training.regularization.orthogonality": 1e-4}, "BigGAN (DeepMind)", level=Level.EXPERT),
    _K("schrittweise_auftauen", "Schichten schrittweise auftauen", "experimentell", T(
        "Erst den Kopf, dann nach und nach tiefere Schichten trainieren.", "Schont vortrainiertes Wissen.",
        "Transfer-Learning."), {"+training.callbacks": {"name": "gradual_unfreeze"}}, "ULMFiT (fast.ai)",
        level=Level.EXPERIMENTAL),
    _K("schichtweise_lr", "Schichtweise Lernraten (LLRD)", "experimentell", T(
        "Frühere Schichten lernen langsamer.", "Besseres Feintuning vortrainierter Modelle.", "Feintuning."),
        {"training.layerwise_lr_decay": 0.8}, "BEiT, ELECTRA", level=Level.EXPERT),
    _K("wsd", "Warmup-Stable-Decay-Zeitplan", "experimentell", T(
        "Konstante Lernrate mit kurzem Abklingen am Ende.", "Training jederzeit verlängerbar.", "Sprachmodelle."),
        {"training.scheduler": "wsd"}, "MiniCPM, DeepSeek", level=Level.ADVANCED),
]

BY_KEY = {t.key: t for t in CATALOG}


# ===========================================================================
# Status, Ein- und Ausschalten
# ===========================================================================

def _list_names(value: Any) -> list[str]:
    return [v.name for v in (value or []) if _is_spec(v)]


def _item_name(item: Any) -> str:
    return item if isinstance(item, str) else item["name"]


def status(cfg: BaseConfig, tech: Technique) -> TechniqueState:
    """``an`` | ``aus`` | ``teilweise`` | ``nicht_anwendbar`` aus der aktuellen Konfiguration."""
    try:
        if tech.applies is not None and not tech.applies(cfg):
            return TechniqueState(tech, "nicht_anwendbar")
        changes = tech.changes_for(cfg)
    except ConfigError:
        return TechniqueState(tech, "nicht_anwendbar")
    hits, total, paths = 0, 0, []
    for key, value in changes.items():
        if key.startswith("+"):
            path = key[1:]
            try:
                names = _list_names(cfg.get(path))
            except ConfigError:
                return TechniqueState(tech, "nicht_anwendbar")
            total += 1
            hits += _item_name(value) in names
            paths.append(path)
            continue
        matches = cfg.find(key) if "*" in key else [key]
        if not matches:
            return TechniqueState(tech, "nicht_anwendbar")
        for path in matches:
            try:
                current = cfg.get(path)
            except ConfigError:
                return TechniqueState(tech, "nicht_anwendbar")
            total += 1
            hits += _same(current, value)
            paths.append(path)
    if tech.check is not None:
        custom = tech.check(cfg)
        if custom is None:
            return TechniqueState(tech, "nicht_anwendbar", paths)
        return TechniqueState(tech, "an" if custom else "aus", paths)
    state = "an" if total and hits == total else ("aus" if hits == 0 else "teilweise")
    return TechniqueState(tech, state, paths)


def enable(cfg: BaseConfig, tech: Technique, memory: dict[str, Any]) -> list[str]:
    """Schaltet ``tech`` in ``cfg`` ein; merkt vorherige Werte in ``memory`` (für ``disable``)."""
    if tech.applies is not None and not tech.applies(cfg):
        raise ConfigError(f"Technik '{tech.key}' passt nicht zu diesem Projekt (Modell/Daten/Paradigma).")
    if tech.check is not None and tech.check(cfg):
        return []                       # bereits an
    changed = []
    remembered = memory.setdefault(tech.key, {})
    for key, value in tech.changes_for(cfg).items():
        if key.startswith("+"):
            path = key[1:]
            items = list(cfg.get(path))
            if _item_name(value) not in _list_names(items):
                items.append(value)
                cfg.set(path, items)
                remembered[key] = True            # selbst hinzugefügt → beim Ausschalten wieder entfernen
                changed.append(path)
            continue
        matches = cfg.find(key) if "*" in key else [key]
        if not matches:
            raise ConfigError(f"Technik '{tech.key}' passt nicht zu diesem Projekt (keine Einstellung '{key}').")
        for path in matches:
            current = cfg.get(path)
            if not _same(current, value):
                if path.endswith(".name"):       # Komponentenwechsel: ganze Komponente samt Argumenten merken
                    parent = path[: -len(".name")]
                    remembered.setdefault(parent, cfg.get(parent).to_dict())
                else:
                    remembered.setdefault(path, current.to_dict() if _is_spec(current) else current)
                cfg.set(path, value)
                changed.append(path)
    return changed


def disable(cfg: BaseConfig, tech: Technique, memory: dict[str, Any]) -> list[str]:
    """Schaltet ``tech`` aus: vorherige Werte zurück, sonst die Einstellungen ausschalten bzw. auf Standard."""
    changed = []
    known = tech.key in memory          # eingeschaltet in dieser Sitzung → exakt den vorherigen Zustand herstellen
    remembered = memory.pop(tech.key, {})
    if callable(tech.changes):          # berechnete Änderungen: gemerkte Werte zurück, sonst Standard
        for path in remembered or tech.changes_for(cfg):
            if path in remembered:
                cfg.set(path, remembered[path])
            else:
                set_state(cfg, path, "standard")
            changed.append(path)
        return changed
    for key, value in tech.changes_for(cfg).items():
        if key.startswith("+"):
            path = key[1:]
            if known and key not in remembered:
                continue                          # war schon vor dem Einschalten vorhanden
            items = list(cfg.get(path))
            kept = [i for i in items if not (_is_spec(i) and i.name == _item_name(value))]
            if len(kept) != len(items):
                cfg.set(path, kept)
                changed.append(path)
            continue
        for path in (cfg.find(key) if "*" in key else [key]):
            if not _same(cfg.get(path), value):
                continue                                  # wurde inzwischen anders eingestellt → nicht anfassen
            parent = path[: -len(".name")] if path.endswith(".name") else None
            if parent is not None and parent in remembered:
                cfg.set(parent, remembered[parent])
            elif path in remembered:
                cfg.set(path, remembered[path])
            elif not known:
                _switch_off(cfg, path)
            else:
                continue                                  # war schon vor dem Einschalten so → unverändert lassen
            changed.append(path)
    return changed


def _switch_off(cfg: BaseConfig, path: str) -> None:
    ref = cfg.param_at(path)
    p = ref.param
    if path.endswith(".name") and getattr(p, "kind", "") == "component":
        cfg.set(path, p.default_name)
    elif getattr(p, "kind", "") == "component" or off_value(p) is UNSET:
        set_state(cfg, path, "standard")
    else:
        set_state(cfg, path, "aus")


def overview(cfg: BaseConfig, only_active: bool = False, max_level: Level | str | None = None) -> str:
    """Katalog nach Kategorien mit Status (● an · ○ aus · ◐ teilweise · – passt nicht)."""
    limit = Level.parse(max_level)
    lines = ["Techniken (● an · ○ aus · ◐ teilweise · – passt nicht zu diesem Modell):"]
    n_on = 0
    for cat, (title, desc) in CATEGORIES.items():
        rows = [status(cfg, t) for t in CATALOG if t.category == cat and t.level.visible_at(limit)]
        if only_active:
            rows = [r for r in rows if r.status in ("an", "teilweise")]
        if not rows:
            continue
        n_on += sum(r.status == "an" for r in rows)
        lines.append(f"\n■ {title} – {desc}")
        lines.extend(f"  {r}" for r in rows)
    lines.append(f"\n{n_on} Techniken eingeschaltet · einschalten: projekt.enable('<kennung>') · "
                 f"ausschalten: projekt.disable('<kennung>')")
    return "\n".join(lines)
