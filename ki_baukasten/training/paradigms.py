"""
ki_baukasten.training.paradigms
===============================

Trainings-Paradigmen – **welche Art** von Lernen stattfindet.

Ein Paradigma übersetzt einen Daten-Batch in Verluste und ruft dafür den
Trainer auf. Dadurch bleibt der Trainer selbst völlig allgemein (Präzision,
Akkumulation, Clipping, Zeitpläne, EMA, Callbacks ...), während das
Paradigma die *Lernlogik* bestimmt.

=================  ============================================================
Paradigma          Lernlogik
=================  ============================================================
supervised         Eingabe → Vorhersage ↔ Ziel (Klassifikation, Regression,
                   Multi-Label, Multi-Task, Token-Klassifikation)
causal_lm          nächstes Token vorhersagen (GPT-artiges Vortraining)
masked_lm          maskierte Tokens rekonstruieren (BERT-artig)
seq2seq            Eingabesequenz → Ausgabesequenz mit Teacher Forcing
autoencoder        Rekonstruktion (+ KL-Annealing für VAEs)
contrastive        selbstüberwacht: zwei Sichten desselben Beispiels angleichen
distillation       Schüler lernt von einem Lehrer-Modell
gan                Generator ↔ Diskriminator (5 Verlustarten, GP, R1 ...)
diffusion          Rauschen vorhersagen (Min-SNR-Gewichtung ...)
preference         Reward-Modell aus Paaren (bevorzugt/abgelehnt, Bradley-Terry)
dpo                Direct Preference Optimization für Sprachmodelle
reinforcement      Reinforcement Learning (siehe ``training.rl``)
=================  ============================================================

Batch-Format
------------
Batches sind ``dict``\\s. Übliche Schlüssel: ``inputs``, ``targets``,
``attention_mask``; paradigmenspezifisch z. B. ``chosen``/``rejected``.

Registry ``paradigm``
---------------------
Fabrik-Signatur: ``factory(args) -> Paradigm``.
"""

from __future__ import annotations

import copy
import inspect
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, ComponentParam, Level, Param, T, TrainingError, create_registry

PARADIGMS = create_registry(
    "paradigm",
    "Trainings-Paradigmen",
    T(
        "Art des Lernens (überwacht, generativ, selbstüberwacht, präferenzbasiert, RL ...).",
        "Bestimmt, wie aus Daten ein Lernsignal entsteht.",
        "Jedes Training.",
    ),
    context_doc="factory(args) -> Paradigm",
)


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================

_SIG_CACHE: dict[type, tuple[set[str], bool]] = {}


def _accepts(model: nn.Module, name: str) -> bool:
    cls = type(model)
    if cls not in _SIG_CACHE:
        params = inspect.signature(model.forward).parameters
        _SIG_CACHE[cls] = (set(params), any(p.kind == p.VAR_KEYWORD for p in params.values()))
    names, var_kw = _SIG_CACHE[cls]
    return name in names or var_kw


def call_model(model: nn.Module, inputs: Any, **extra: Any) -> Any:
    """Ruft ein Modell auf; zusätzliche Argumente nur, wenn ``forward`` sie kennt.

    So funktionieren Paradigmen mit allen Modellen: ``attention_mask`` z. B.
    wird an Transformer übergeben, an ein RNN aber nicht.
    """
    target = getattr(model, "_orig_mod", model)  # torch.compile
    extra = {k: v for k, v in extra.items() if v is not None and _accepts(target, k)}
    if isinstance(inputs, dict):
        return model(**inputs, **extra)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs, **extra)
    return model(inputs, **extra)


def main_output(y: Any) -> torch.Tensor:
    """Hauptausgabe (bei dict-Ausgaben 'logits'/'output'/'reconstruction' oder erster Eintrag)."""
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, dict):
        for key in ("logits", "output", "reconstruction"):
            if key in y:
                return y[key]
        return next(iter(y.values()))
    if isinstance(y, (tuple, list)):
        return y[0]
    raise TrainingError(f"Unbekannte Modellausgabe vom Typ {type(y).__name__}.")


def _select(obj: Any, key: str | None) -> Any:
    if key is None:
        return main_output(obj) if isinstance(obj, (dict, tuple, list)) and not isinstance(obj, torch.Tensor) else obj
    if not isinstance(obj, dict) or key not in obj:
        raise TrainingError(f"Schlüssel '{key}' nicht in Ausgabe/Ziel vorhanden.")
    return obj[key]


def is_segmentation(pred: Any, target: Any) -> bool:
    """Logits ``[N, C, H, W]`` gegen Klassenmasken ``[N, H, W]`` (Kanäle vorne, wie bei Bildern)."""
    return (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor) and pred.dim() == 4
            and target.dim() == 3 and not target.is_floating_point() and pred.shape[2:] == target.shape[1:])


def auto_loss(pred: torch.Tensor, target: torch.Tensor, task: str = "auto", label_smoothing: float = 0.0,
              soft: bool = False, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Wählt einen sinnvollen Standardverlust anhand von Aufgabe und Datentypen.

    ``soft=True``: Ziele sind Wahrscheinlichkeiten (z. B. nach Mixup/CutMix) → Kreuzentropie mit weichen Zielen.
    ``weight``: Klassengewichte für die Kreuzentropie (``training.class_weights``).
    """
    if weight is not None:
        weight = weight.to(pred.device, torch.float32)
        if weight.numel() != pred.shape[1 if is_segmentation(pred, target) else -1]:
            weight = None
    if soft and target.shape == pred.shape:
        return F.cross_entropy(pred.float().reshape(-1, pred.shape[-1]), target.float().reshape(-1, pred.shape[-1]),
                               weight=weight, label_smoothing=label_smoothing)
    if is_segmentation(pred, target) and task in ("auto", "classification"):
        return F.cross_entropy(pred.float(), target.long(), weight=weight, ignore_index=-100,
                               label_smoothing=label_smoothing)
    if task == "classification" or (task == "auto" and not target.is_floating_point()
                                    and pred.dim() == target.dim() + 1):
        logits = pred.reshape(-1, pred.shape[-1])
        return F.cross_entropy(logits.float(), target.reshape(-1).long(), weight=weight, ignore_index=-100,
                               label_smoothing=label_smoothing)
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(pred.float(), target.float().reshape(pred.shape))
    t = target.float()
    if t.shape != pred.shape and t.numel() == pred.numel():
        t = t.reshape(pred.shape)
    return F.mse_loss(pred.float(), t)


def accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float | None:
    """Klassifikationsgenauigkeit (oder ``None``, wenn nicht anwendbar)."""
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        return None
    if target.is_floating_point() or pred.dim() != target.dim() + 1:
        return None
    valid = target != ignore_index
    if valid.sum() == 0:
        return None
    guess = pred.argmax(1) if is_segmentation(pred, target) else pred.argmax(-1)
    return float((guess[valid] == target[valid]).float().mean())


class Paradigm:
    """Basisklasse aller Paradigmen."""

    #: Optimierungsgruppe, deren Schritte als 'globaler Schritt' zählen.
    primary_key = "main"

    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg

    def setup(self, trainer: Any) -> None:
        """Einmalig nach Aufbau des Trainers (z. B. Lehrer/Referenzmodell laden)."""

    def optimization_groups(self, model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
        """Parameter je Optimierer. Standard: ein Optimierer für alles."""
        return {"main": list(model.named_parameters())}

    def training_step(self, trainer: Any, batch: dict) -> dict[str, float]:
        loss, metrics = self.compute(trainer, batch, train=True)
        trainer.optimize(loss, closure=lambda: self.compute(trainer, batch, train=True)[0])
        return metrics

    @torch.no_grad()
    def validation_step(self, trainer: Any, batch: dict) -> dict[str, float]:
        _, metrics = self.compute(trainer, batch, train=False)
        return metrics

    def compute(self, trainer: Any, batch: dict, train: bool) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError


# ===========================================================================
# Überwacht
# ===========================================================================

class SupervisedConfig(BaseConfig):
    task = Param("auto", choices={
        "auto": "aus Datentypen ableiten (Ganzzahl-Ziele → Klassifikation, sonst Regression)",
        "classification": "Klassifikation (Kreuzentropie)",
        "regression": "Regression (MSE)",
        "multilabel": "Multi-Label (binäre Kreuzentropie je Ausgabe)",
    }, tooltip=T(
        "Art der überwachten Aufgabe (nur relevant, wenn keine eigenen Verluste gesetzt sind).",
        "Bestimmt Standardverlust und Metriken.", "Meist 'auto'."))
    label_smoothing = Param(0.0, min=0.0, max=0.5, tooltip=T(
        "Label Smoothing für den Standard-Klassifikationsverlust (Anteil der Zielwahrscheinlichkeit, der gleichmäßig "
        "auf alle Klassen verteilt wird).", "Verhindert übermäßig selbstsichere Vorhersagen, bessere Kalibrierung.",
        "Klassifikation: 0.05–0.1."))


_TASK_KIND = {"classification": "klassifikation", "regression": "regression", "multilabel": "multilabel"}


class Supervised(Paradigm):
    """Eingabe → Vorhersage ↔ Ziel. Unterstützt dict-Ausgaben/Ziele (Multi-Task)."""

    def loss(self, trainer: Any, out: Any, targets: Any) -> torch.Tensor:
        if trainer.losses:
            total = 0.0
            for fn, a in trainer.losses:
                total = total + a.weight * fn(_select(out, a.output_key), _select(targets, a.target_key))
            return total
        if isinstance(out, dict) and isinstance(targets, dict):
            keys = [k for k in out if k in targets]
            if not keys:
                raise TrainingError("Ausgabe und Ziele haben keine gemeinsamen Schlüssel.")
            return sum(auto_loss(out[k], targets[k], self.cfg.task, self.cfg.label_smoothing) for k in keys)
        return auto_loss(main_output(out), targets if not isinstance(targets, dict) else main_output(targets),
                         self.cfg.task, self.cfg.label_smoothing, getattr(self, "_soft", False),
                         getattr(trainer, "class_weight", None))

    def compute(self, trainer, batch, train):
        inputs, targets = batch["inputs"], batch.get("targets")
        if targets is None:
            raise TrainingError("Paradigma 'supervised' braucht 'targets' im Batch.")
        self._soft = False
        if train:
            was_int = isinstance(targets, torch.Tensor) and not targets.is_floating_point()
            inputs, targets = trainer.augment(inputs, targets)
            self._soft = was_int and isinstance(targets, torch.Tensor) and targets.is_floating_point()
        with trainer.autocast():
            out = call_model(trainer.model, inputs, attention_mask=batch.get("attention_mask"),
                             lengths=batch.get("lengths"))
            loss = self.loss(trainer, out, targets)
            aux = trainer.model_aux_loss()
            if aux is not None:
                loss = loss + aux
        metrics = {"loss": float(loss.detach())}
        if not isinstance(targets, dict):
            pred = main_output(out).detach()
            acc = accuracy(pred, targets)
            if acc is not None:
                metrics["accuracy"] = acc
            if not train:
                trainer.collect(pred, targets, _TASK_KIND.get(self.cfg.task, "auto"))
        return loss, metrics


# ===========================================================================
# Sprachmodelle
# ===========================================================================

class CausalLMConfig(BaseConfig):
    pad_token_id = Param(None, type=int, tooltip=T(
        "Padding-Token, das im Verlust ignoriert wird.", "Unterschiedlich lange Sequenzen.", "Wenn gepaddet wird."))
    z_loss_coef = Param(0.0, min=0.0, tooltip=T(
        "z-Loss: bestraft große log Σ exp(Logits) (PaLM).", "Stabilisiert große Sprachmodelle numerisch.",
        "Große LMs: 1e-4."), level=Level.EXPERT)
    label_smoothing = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Label Smoothing auf die Token-Ziele.", "Weniger Überkonfidenz.", "Selten bei LMs."), level=Level.ADVANCED)


class CausalLM(Paradigm):
    """Nächstes-Token-Vorhersage: Eingabe ``[B, T]`` → Ziele um eins verschoben."""

    def compute(self, trainer, batch, train):
        x = batch["inputs"]
        mask = batch.get("attention_mask")
        inp, tgt = x[:, :-1], x[:, 1:].clone()
        if self.cfg.pad_token_id is not None:
            tgt[tgt == self.cfg.pad_token_id] = -100
        if mask is not None:
            tgt[~mask[:, 1:].bool()] = -100
            mask = mask[:, :-1]
        with trainer.autocast():
            logits = main_output(call_model(trainer.model, inp, attention_mask=mask))
            if trainer.losses:
                loss = sum(a.weight * fn(logits, tgt) for fn, a in trainer.losses)
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
                                       ignore_index=-100, label_smoothing=self.cfg.label_smoothing)
            if self.cfg.z_loss_coef > 0:
                loss = loss + self.cfg.z_loss_coef * torch.logsumexp(logits.float(), -1).square().mean()
            aux = trainer.model_aux_loss()
            if aux is not None:
                loss = loss + aux
        lv = float(loss.detach())
        if not train:
            trainer.collect(logits.detach(), tgt, "sprache")
        return loss, {"loss": lv, "perplexity": math.exp(min(lv, 30)), "accuracy": accuracy(logits.detach(), tgt) or 0.0}


class MaskedLMConfig(BaseConfig):
    mask_token_id = Param(None, type=int, tooltip=T(
        "ID des [MASK]-Tokens (None = letzte Vokabular-ID).", "Ersetzt maskierte Tokens.", "Je nach Tokenizer."))
    mask_prob = Param(0.15, min=0.0, max=1.0, tooltip=T(
        "Anteil maskierter Tokens.", "Mehr = schwierigere Aufgabe, mehr Lernsignal pro Beispiel.", "BERT: 0.15."))
    random_ratio = Param(0.1, min=0.0, max=1.0, tooltip=T(
        "Anteil der ausgewählten Tokens, die durch zufällige Tokens ersetzt werden.", "Robustheit.", "0.1."),
        level=Level.ADVANCED)
    keep_ratio = Param(0.1, min=0.0, max=1.0, tooltip=T(
        "Anteil der ausgewählten Tokens, die unverändert bleiben.", "Verhindert Train/Test-Diskrepanz.", "0.1."),
        level=Level.ADVANCED)
    pad_token_id = Param(None, type=int, tooltip=T("Padding-Token (wird nie maskiert).", "Nur echte Tokens.", "Bei Padding."))
    vocab_size = Param(None, type=int, min=1, tooltip=T(
        "Vokabulargröße (None = aus der Modellausgabe).", "Für zufällige Ersatz-Tokens.", "Meist None."),
        level=Level.EXPERT)


class MaskedLM(Paradigm):
    def compute(self, trainer, batch, train):
        c = self.cfg
        x = batch["inputs"]
        mask = batch.get("attention_mask")
        with torch.no_grad():
            probe = c.vocab_size
            if probe is None:
                probe = getattr(getattr(trainer.model, "config", None), "vocab_size", None) or int(x.max()) + 2
            mask_id = c.mask_token_id if c.mask_token_id is not None else probe - 1
            cand = torch.ones_like(x, dtype=torch.bool)
            if c.pad_token_id is not None:
                cand &= x != c.pad_token_id
            if mask is not None:
                cand &= mask.bool()
            selected = (torch.rand(x.shape, device=x.device) < c.mask_prob) & cand
            tgt = torch.where(selected, x, torch.full_like(x, -100))
            r = torch.rand(x.shape, device=x.device)
            inp = x.clone()
            to_mask = selected & (r < 1 - c.random_ratio - c.keep_ratio)
            to_rand = selected & (r >= 1 - c.random_ratio - c.keep_ratio) & (r < 1 - c.keep_ratio)
            inp[to_mask] = mask_id
            inp[to_rand] = torch.randint(0, probe, (int(to_rand.sum()),), device=x.device)
        with trainer.autocast():
            logits = main_output(call_model(trainer.model, inp, attention_mask=mask))
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):  # kein Token maskiert
                loss = logits.sum() * 0.0
            aux = trainer.model_aux_loss()
            if aux is not None:
                loss = loss + aux
        if not train:
            trainer.collect(logits.detach(), tgt, "sprache")
        return loss, {"loss": float(loss.detach()), "accuracy": accuracy(logits.detach(), tgt) or 0.0}


class Seq2SeqConfig(BaseConfig):
    bos_token_id = Param(0, tooltip=T(
        "Start-Token, mit dem der Decoder beginnt (nur Token-Ziele).", "Teacher Forcing: Ziele um eins verschoben.",
        "Je nach Tokenizer."))
    pad_token_id = Param(None, type=int, tooltip=T("Padding im Ziel (wird ignoriert).", "Unterschiedliche Längen.", "Bei Padding."))
    label_smoothing = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Label Smoothing.", "Bessere Übersetzungen (0.1).", "Übersetzung."), level=Level.ADVANCED)


class Seq2Seq(Paradigm):
    """Encoder-Decoder mit Teacher Forcing; Token- oder kontinuierliche Ziele."""

    def compute(self, trainer, batch, train):
        src, tgt = batch["inputs"], batch["targets"]
        if tgt.is_floating_point():
            if tgt.dim() == 2:
                tgt = tgt.unsqueeze(-1)
            dec_in = torch.cat([torch.zeros_like(tgt[:, :1]), tgt[:, :-1]], 1)
        else:
            dec_in = torch.cat([torch.full_like(tgt[:, :1], self.cfg.bos_token_id), tgt[:, :-1]], 1)
        with trainer.autocast():
            out = main_output(call_model(trainer.model, src, decoder_inputs=dec_in,
                                         attention_mask=batch.get("attention_mask")))
            if tgt.is_floating_point():
                loss = F.mse_loss(out.float(), tgt.float())
                acc = None
            else:
                labels = tgt.clone()
                if self.cfg.pad_token_id is not None:
                    labels[labels == self.cfg.pad_token_id] = -100
                loss = F.cross_entropy(out.reshape(-1, out.shape[-1]).float(), labels.reshape(-1), ignore_index=-100,
                                       label_smoothing=self.cfg.label_smoothing)
                acc = accuracy(out.detach(), labels)
            aux = trainer.model_aux_loss()
            if aux is not None:
                loss = loss + aux
        m = {"loss": float(loss.detach())}
        if acc is not None:
            m["accuracy"] = acc
        if not train:
            if tgt.is_floating_point():
                trainer.collect(out.detach(), tgt, "regression")
            else:
                trainer.collect(out.detach(), labels, "seq2seq")
        return loss, m


# ===========================================================================
# Autoencoder & kontrastiv
# ===========================================================================

class AutoencoderParadigmConfig(BaseConfig):
    reconstruction_loss = Param("mse", choices={
        "mse": "quadratischer Fehler (kontinuierliche Daten)",
        "bce": "binäre Kreuzentropie (Daten in [0, 1], Ausgabe mit sigmoid)",
        "l1": "absoluter Fehler (robust)",
    }, tooltip=T("Rekonstruktionsverlust.", "Passend zum Datentyp.", "Bilder in [0,1]: bce; sonst mse."))
    reduction = Param("auto", choices={
        "auto": "VAE: Summe je Beispiel (korrekte ELBO), sonst Mittelwert",
        "mean": "Mittelwert über alle Elemente",
        "sum": "Summe je Beispiel, gemittelt über den Batch",
    }, tooltip=T(
        "Reduktion des Rekonstruktionsverlusts.",
        "Beim VAE muss die Rekonstruktion je Beispiel summiert werden, sonst erdrückt der KL-Term sie "
        "(Posterior Collapse: Latentraum wird ignoriert).", "auto."), level=Level.ADVANCED)
    kl_schedule = Param("linear", choices={
        "constant": "β sofort voll", "linear": "β linear von 0 hochfahren (KL-Annealing)",
        "cyclical": "β zyklisch 0 → voll (Fu et al. 2019)",
    }, tooltip=T("KL-Annealing (nur VAE).", "Verhindert Posterior Collapse zu Beginn.", "linear."), level=Level.ADVANCED)
    kl_warmup_steps = Param(500, min=0, tooltip=T(
        "Schritte bis β voll erreicht ist (bzw. Zykluslänge).", "Dauer des Annealings.", "500–10000."),
        level=Level.ADVANCED)


class AutoencoderParadigm(Paradigm):
    def compute(self, trainer, batch, train):
        x = batch["inputs"]
        target = batch.get("targets", x)
        if not isinstance(target, torch.Tensor) or not target.is_floating_point() or target.shape != x.shape:
            target = x
        with trainer.autocast():
            out = trainer.model(x)
            rec = out["reconstruction"]
            kind = self.cfg.reconstruction_loss
            if kind == "bce":
                per = F.binary_cross_entropy(rec.float().clamp(1e-6, 1 - 1e-6), target.float(), reduction="none")
            elif kind == "l1":
                per = (rec.float() - target.float()).abs()
            else:
                per = (rec.float() - target.float()).square()
            red = self.cfg.reduction
            if red == "auto":
                red = "sum" if "kl" in out else "mean"
            r = per.flatten(1).sum(1).mean() if red == "sum" else per.mean()
            loss = r
            metrics = {"reconstruction": float(r.detach())}
            if "kl" in out:
                beta = getattr(trainer.model.config, "beta", 1.0) * self._kl_factor(trainer.global_step)
                loss = loss + beta * out["kl"]
                metrics["kl"] = float(out["kl"].detach())
            aux = trainer.model_aux_loss()
            if aux is not None:
                loss = loss + aux
        metrics["loss"] = float(loss.detach())
        return loss, metrics

    def _kl_factor(self, step: int) -> float:
        w = self.cfg.kl_warmup_steps
        if self.cfg.kl_schedule == "constant" or w == 0:
            return 1.0
        if self.cfg.kl_schedule == "cyclical":
            return min(1.0, 2 * (step % w) / w)
        return min(1.0, step / w)


class ContrastiveConfig(BaseConfig):
    temperature = Param(0.1, min=1e-4, tooltip=T(
        "Temperatur des InfoNCE-Verlusts.", "Schärfe der Unterscheidung.", "0.05–0.5."))
    noise_std = Param(0.1, min=0.0, tooltip=T(
        "Gauß-Rauschen als Augmentierung (Gleitkomma-Eingaben).", "Erzeugt zwei Sichten.", "0.05–0.2."))
    feature_dropout = Param(0.1, min=0.0, max=0.9, tooltip=T(
        "Zufälliges Nullsetzen von Eingabewerten als Augmentierung.", "Zweite Sicht-Variation.", "0.1–0.3."))
    token_mask_prob = Param(0.1, min=0.0, max=0.9, tooltip=T(
        "Anteil zufällig ersetzter Tokens (Ganzzahl-Eingaben).", "Augmentierung für Text.", "0.1."),
        level=Level.ADVANCED)
    mask_token_id = Param(0, tooltip=T("Ersatz-Token für token_mask_prob.", "Augmentierung.", "0 oder [MASK]."),
                          level=Level.ADVANCED)


class Contrastive(Paradigm):
    """SimCLR-artig: zwei augmentierte Sichten (oder ``positives`` im Batch) per InfoNCE angleichen."""

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if x.is_floating_point():
            if c.noise_std:
                x = x + torch.randn_like(x) * c.noise_std
            if c.feature_dropout:
                x = x * (torch.rand_like(x) >= c.feature_dropout)
            return x
        if c.token_mask_prob:
            x = torch.where(torch.rand(x.shape, device=x.device) < c.token_mask_prob,
                            torch.full_like(x, c.mask_token_id), x)
        return x

    def compute(self, trainer, batch, train):
        x = batch["inputs"]
        pos = batch.get("positives")
        with trainer.autocast():
            a = main_output(call_model(trainer.model, self._augment(x)))
            b = main_output(call_model(trainer.model, self._augment(pos if pos is not None else x)))
            za, zb = F.normalize(a.float().flatten(1), dim=-1), F.normalize(b.float().flatten(1), dim=-1)
            logits = za @ zb.T / self.cfg.temperature
            labels = torch.arange(za.shape[0], device=za.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        acc = float((logits.argmax(-1) == labels).float().mean())
        return loss, {"loss": float(loss.detach()), "accuracy": acc}


# ===========================================================================
# Distillation
# ===========================================================================

class DistillationConfig(BaseConfig):
    __title__ = "Distillation (Lehrer → Schüler)"
    __tooltip__ = T(
        "Ein (meist kleinerer) Schüler lernt die Ausgaben – und optional die inneren Zustände – eines Lehrers.",
        "Weiche Lehrer-Ziele enthalten mehr Information als harte Labels ('dunkles Wissen'): kleine Modelle werden "
        "deutlich besser als mit Labels allein.",
        "Modellkompression, Heilen nach Pruning/Quantisierung, Weiterlernen ohne Vergessen (Lehrer = alte Version).",
    )
    teacher = ComponentParam("model", None, tooltip=T(
        "Lehrer-Modell (beliebiges Modell aus der Registry; meist größer).",
        "Liefert 'weiche' Ziele mit mehr Information als Labels.", "Modellkompression.",
        "Alternativ teacher_project / teacher_export – oder Project.distill(), das den Lehrer direkt übergibt."))
    teacher_checkpoint = Param(None, type=str, tooltip=T(
        "Pfad zu trainierten Lehrer-Gewichten (.pt) für 'teacher'.", "Ohne Gewichte lehrt der Lehrer Zufall!",
        "Pflicht in der Praxis, wenn 'teacher' gesetzt ist."))
    teacher_project = Param(None, type=str, tooltip=T(
        "Ordner eines gespeicherten Projekts als Lehrer (Project.save).",
        "Bequemster Weg: trainiertes Projekt komplett als Lehrer verwenden.", "ausgaben/mein_grosses_modell."))
    teacher_export = Param(None, type=str, tooltip=T(
        "torch.export-Datei (.pt2) als Lehrer – auch Modelle von außerhalb des Baukastens.",
        "Lehrer muss nicht mit dem Baukasten gebaut sein.", "Importierte/exportierte Modelle."), level=Level.ADVANCED)
    task = Param("auto", choices={
        "auto": "aus Daten ableiten (Labels → überwacht; nur Token-Eingaben → nächstes Token)",
        "supervised": "harter Verlust mit Labels (Klassifikation/Regression)",
        "causal_lm": "harter Verlust = nächstes Token vorhersagen (Sprachmodelle)",
        "none": "nur vom Lehrer lernen",
    }, tooltip=T("Wozu der harte Verlust (Anteil 1-α) dient.", "Kombiniert Lehrerwissen mit echten Daten.", "auto."))
    temperature = Param(2.0, min=1e-3, tooltip=T(
        "Temperatur zum Weichzeichnen der Lehrer-Verteilung.", "Macht 'dunkles Wissen' über Nebenklassen sichtbar.",
        "2–5; Sprachmodelle 1–2."))
    alpha = Param(0.5, min=0.0, max=1.0, tooltip=T(
        "Gewicht des Distillationsverlusts (1-α für echte Labels).", "Balance Lehrer vs. Daten.", "0.5–0.9."))
    kd_loss = Param("kl", choices={
        "kl": "KL(Lehrer ‖ Schüler) – klassisch (Hinton), deckt die ganze Verteilung ab",
        "reverse_kl": "KL(Schüler ‖ Lehrer) – modussuchend, gut für Sprachmodelle (MiniLLM)",
        "js": "Jensen-Shannon – symmetrisch, robust",
        "mse": "MSE der Logits (Regression, Logit-Matching)",
        "cosine": "Kosinus-Abstand der Logits",
    }, tooltip=T("Art des Distillationsverlusts.", "kl für Klassifikation; reverse_kl/js für Sprachmodelle.", "kl."))
    feature_weight = Param(0.0, min=0.0, tooltip=T(
        "Gewicht der Feature-Distillation (innere Block-Ausgaben angleichen; 0 = aus).",
        "Der Schüler lernt nicht nur WAS, sondern WIE der Lehrer rechnet – hilft besonders bei starkem Pruning.",
        "0.1–1. Blöcke werden proportional zugeordnet (Schüler-Block i ↔ Lehrer-Block i·N_L/N_S)."),
        group="Features")
    feature_loss = Param("auto", choices={
        "auto": "mse bei gleicher Breite, sonst gram",
        "mse": "MSE der normierten Zustände (gleiche Breite nötig)",
        "cosine": "Kosinus-Abstand (gleiche Breite nötig)",
        "gram": "Ähnlichkeiten zwischen Positionen angleichen – funktioniert bei beliebiger Breite",
    }, tooltip=T("Wie innere Zustände verglichen werden.", "gram braucht keine Projektionsschicht.", "auto."),
        group="Features", level=Level.ADVANCED)


def _kd_divergence(s: torch.Tensor, t: torch.Tensor, kind: str, temp: float, valid: torch.Tensor | None):
    """Distillationsverlust pro Position, gemittelt über gültige Positionen."""
    s2, t2 = s.float().reshape(-1, s.shape[-1]), t.float().reshape(-1, t.shape[-1])
    if kind == "mse":
        per = (s2 - t2).square().mean(-1)
    elif kind == "cosine":
        per = 1 - F.cosine_similarity(s2, t2, dim=-1)
    else:
        ls, lt = F.log_softmax(s2 / temp, -1), F.log_softmax(t2 / temp, -1)
        if kind == "kl":
            per = (lt.exp() * (lt - ls)).sum(-1)
        elif kind == "reverse_kl":
            per = (ls.exp() * (ls - lt)).sum(-1)
        else:  # js
            m = torch.logsumexp(torch.stack([ls, lt]), 0) - math.log(2.0)
            per = 0.5 * ((ls.exp() * (ls - m)).sum(-1) + (lt.exp() * (lt - m)).sum(-1))
        per = per * temp ** 2
    if valid is not None:
        v = valid.reshape(-1).to(per.dtype)
        if v.numel() == per.numel():
            return (per * v).sum() / v.sum().clamp_min(1.0)
    return per.mean()


def _feature_distance(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
    a, b = a.float(), b.float()
    if kind == "auto":
        kind = "mse" if a.shape[-1] == b.shape[-1] else "gram"
    if kind in ("mse", "cosine") and a.shape[-1] != b.shape[-1]:
        raise TrainingError(f"feature_loss='{kind}' braucht gleiche Breiten ({a.shape[-1]} vs. {b.shape[-1]}) – "
                            "nutze 'gram'.")
    if kind == "mse":
        return (F.layer_norm(a, a.shape[-1:]) - F.layer_norm(b, b.shape[-1:])).square().mean()
    if kind == "cosine":
        return (1 - F.cosine_similarity(a, b, dim=-1)).mean()
    if a.dim() == 2:  # [B, D] → Ähnlichkeiten zwischen Beispielen
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    ga = F.normalize(a, dim=-1) @ F.normalize(a, dim=-1).transpose(-1, -2)
    gb = F.normalize(b, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)
    return (ga - gb).square().mean()


def load_teacher(cfg: DistillationConfig, device: torch.device) -> nn.Module:
    """Lehrer aus Komponente+Checkpoint, gespeichertem Projekt oder torch.export-Datei."""
    if cfg.teacher_project:
        from ..project import Project
        teacher = Project.load(cfg.teacher_project).model
    elif cfg.teacher_export:
        teacher = torch.export.load(cfg.teacher_export).module()
    elif cfg.teacher is not None:
        teacher = cfg.teacher.build()
        if cfg.teacher_checkpoint:
            state = torch.load(cfg.teacher_checkpoint, map_location="cpu", weights_only=False)
            teacher.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    else:
        raise TrainingError("Distillation braucht einen Lehrer: args.teacher (+ teacher_checkpoint), "
                            "teacher_project, teacher_export oder Project.distill(...).")
    teacher.requires_grad_(False)
    return teacher.to(device).eval()


class Distillation(Paradigm):
    def setup(self, trainer):
        given = getattr(trainer, "teacher_model", None)
        if given is not None:
            given.requires_grad_(False)
            self.teacher = given.to(trainer.device).eval()
        else:
            self.teacher = load_teacher(self.cfg, trainer.device)
        self._pairs: list[tuple[nn.Module, nn.Module]] = []
        self._feats: dict[str, list[torch.Tensor]] = {"s": [], "t": []}
        self._capturing = False
        if self.cfg.feature_weight > 0:
            self._hook_features(trainer)

    def _hook_features(self, trainer) -> None:
        student = getattr(trainer.model, "_orig_mod", trainer.model)
        sb = student.checkpointable_blocks() if hasattr(student, "checkpointable_blocks") else []
        tb = self.teacher.checkpointable_blocks() if hasattr(self.teacher, "checkpointable_blocks") else []
        if not sb or not tb:
            raise TrainingError("feature_weight > 0 braucht Modelle mit Blöcken (checkpointable_blocks).")
        for i, block in enumerate(sb):
            j = min(len(tb) - 1, round((i + 1) * len(tb) / len(sb)) - 1)
            self._pairs.append((block, tb[j]))

        def capture(key):
            def hook(module, args, output):
                if self._capturing:  # nur während compute() aufzeichnen (nicht bei späterer Inferenz)
                    self._feats[key].append(output[0] if isinstance(output, tuple) else output)
            return hook

        for s_block, t_block in self._pairs:
            s_block.register_forward_hook(capture("s"))
            t_block.register_forward_hook(capture("t"))

    def _task(self, batch) -> str:
        if self.cfg.task != "auto":
            return self.cfg.task
        if batch.get("targets") is not None:
            return "supervised"
        x = batch["inputs"]
        return "causal_lm" if isinstance(x, torch.Tensor) and not x.is_floating_point() and x.dim() == 2 else "none"

    def compute(self, trainer, batch, train):
        c = self.cfg
        inputs, targets = batch["inputs"], batch.get("targets")
        mask = batch.get("attention_mask")
        task = self._task(batch)
        self._feats = {"s": [], "t": []}
        self._capturing = c.feature_weight > 0
        with trainer.autocast():
            try:
                s = main_output(call_model(trainer.model, inputs, attention_mask=mask))
                with torch.no_grad():
                    t = main_output(call_model(self.teacher, inputs, attention_mask=mask))
            finally:
                self._capturing = False
            if s.shape[-1] != t.shape[-1] and c.kd_loss not in ("mse", "cosine"):
                raise TrainingError(f"Schüler ({s.shape[-1]}) und Lehrer ({t.shape[-1]}) haben verschiedene "
                                    "Ausgabegrößen.")
            kd = _kd_divergence(s, t, c.kd_loss, c.temperature, mask if s.dim() == 3 else None)
            loss = c.alpha * kd
            m = {"distill": float(kd.detach())}
            if c.feature_weight > 0 and self._feats["s"]:
                feat = sum(_feature_distance(a, b.detach(), c.feature_loss)
                           for a, b in zip(self._feats["s"], self._feats["t"])) / len(self._feats["s"])
                loss = loss + c.feature_weight * feat
                m["features"] = float(feat.detach())
            if c.alpha < 1 and task != "none":
                if task == "causal_lm":
                    hard = F.cross_entropy(s[:, :-1].reshape(-1, s.shape[-1]).float(), inputs[:, 1:].reshape(-1),
                                           ignore_index=-100)
                elif targets is not None:
                    hard = (sum(a.weight * fn(s, targets) for fn, a in trainer.losses) if trainer.losses
                            else auto_loss(s, targets))
                else:
                    hard = None
                if hard is not None:
                    loss = loss + (1 - c.alpha) * hard
                    m["hard"] = float(hard.detach())
            if targets is not None:
                acc = accuracy(s.detach(), targets)
                if acc is not None:
                    m["accuracy"] = acc
                if not train and isinstance(targets, torch.Tensor):
                    trainer.collect(s.detach(), targets)
            elif task == "causal_lm":
                m["accuracy"] = float((s[:, :-1].argmax(-1) == inputs[:, 1:]).float().mean())
            m["agreement"] = float((s.detach().argmax(-1) == t.argmax(-1)).float().mean()) \
                if s.shape[-1] == t.shape[-1] else float("nan")
        m["loss"] = float(loss.detach())
        return loss, m


# ===========================================================================
# GAN
# ===========================================================================

class GANParadigmConfig(BaseConfig):
    loss = Param("non_saturating", choices={
        "standard": "Original-Minimax (sättigt früh – eher historisch)",
        "non_saturating": "nicht-sättigende Variante (Standard)",
        "wasserstein": "Wasserstein-GAN (mit gradient_penalty → WGAN-GP)",
        "hinge": "Hinge-Verlust (BigGAN, SAGAN)",
        "least_squares": "Least-Squares-GAN (stabile Gradienten)",
    }, tooltip=T("GAN-Verlustart.", "Großer Einfluss auf Stabilität und Mode-Collapse.", "non_saturating oder hinge."))
    n_critic = Param(1, min=1, tooltip=T(
        "Diskriminator-Schritte pro Generator-Schritt.", "WGAN braucht einen starken Kritiker.", "1; WGAN: 5."))
    gradient_penalty = Param(0.0, min=0.0, tooltip=T(
        "Gewicht der Gradientenstrafe (WGAN-GP).", "Erzwingt Lipschitz-Stetigkeit des Kritikers.", "WGAN-GP: 10."))
    r1_penalty = Param(0.0, min=0.0, tooltip=T(
        "R1-Strafe (Gradient auf echten Daten).", "Stabilisiert (StyleGAN).", "1–10."), level=Level.ADVANCED)
    label_smoothing = Param(0.0, min=0.0, max=0.5, tooltip=T(
        "Einseitiges Label Smoothing für echte Daten.", "Weniger überkonfidenter Diskriminator.", "0.1."),
        level=Level.ADVANCED)
    instance_noise_std = Param(0.0, min=0.0, tooltip=T(
        "Rauschen auf Diskriminator-Eingaben (klingt linear ab).", "Überlappende Verteilungen, stabileres Training.",
        "0.1."), level=Level.EXPERT)
    instance_noise_steps = Param(5000, min=1, tooltip=T(
        "Schritte bis das Instanzrauschen 0 ist.", "Abklingdauer.", "Trainingslänge/2."), level=Level.EXPERT)


class GANParadigm(Paradigm):
    primary_key = "generator"

    def optimization_groups(self, model):
        d_names = ("discriminator.", "d_label.")
        named = list(model.named_parameters())
        return {
            "discriminator": [(n, p) for n, p in named if n.startswith(d_names)],
            "generator": [(n, p) for n, p in named if not n.startswith(d_names)],
        }

    def _d_params(self, model):
        return [p for n, p in model.named_parameters() if n.startswith(("discriminator.", "d_label."))]

    def _noise(self, x, step):
        s = self.cfg.instance_noise_std * max(0.0, 1 - step / self.cfg.instance_noise_steps)
        return x + torch.randn_like(x) * s if s > 0 else x

    def _d_loss(self, real, fake):
        k, eps = self.cfg.loss, self.cfg.label_smoothing
        if k in ("standard", "non_saturating"):
            return F.binary_cross_entropy_with_logits(real, torch.full_like(real, 1 - eps)) + \
                F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        if k == "wasserstein":
            return fake.mean() - real.mean()
        if k == "hinge":
            return F.relu(1 - real).mean() + F.relu(1 + fake).mean()
        return 0.5 * ((real - 1).square().mean() + fake.square().mean())

    def _g_loss(self, fake):
        k = self.cfg.loss
        if k == "standard":
            return -F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        if k == "non_saturating":
            return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
        if k in ("wasserstein", "hinge"):
            return -fake.mean()
        return 0.5 * (fake - 1).square().mean()

    def training_step(self, trainer, batch):
        model, c = trainer.model, self.cfg
        real = batch["inputs"]
        labels = batch.get("targets") if getattr(model.config, "num_classes", 0) else None
        n = real.shape[0]
        step = trainer.global_step
        m: dict[str, float] = {}
        for _ in range(c.n_critic):
            with trainer.autocast():
                fake = model(model.sample_noise(n, real.device), labels).detach().reshape(real.shape)
                real_in = self._noise(real, step).requires_grad_(c.r1_penalty > 0)
                d_real = model.discriminate(real_in, labels)
                d_fake = model.discriminate(self._noise(fake, step), labels)
                loss_d = self._d_loss(d_real.float(), d_fake.float())
            if c.gradient_penalty > 0:
                eps = torch.rand(n, *([1] * (real.dim() - 1)), device=real.device)
                mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
                g = torch.autograd.grad(model.discriminate(mix, labels).sum(), mix, create_graph=True)[0]
                loss_d = loss_d + c.gradient_penalty * (g.flatten(1).norm(dim=1) - 1).square().mean()
            if c.r1_penalty > 0:
                g = torch.autograd.grad(d_real.sum(), real_in, create_graph=True)[0]
                loss_d = loss_d + 0.5 * c.r1_penalty * g.flatten(1).square().sum(1).mean()
            trainer.optimize(loss_d, key="discriminator")
        m.update(loss_d=float(loss_d.detach()), d_real=float(d_real.detach().mean()), d_fake=float(d_fake.detach().mean()))

        d_params = self._d_params(model)
        for p in d_params:  # Diskriminator einfrieren → keine Gradienten beim Generator-Schritt
            p.requires_grad_(False)
        with trainer.autocast():
            fake = model(model.sample_noise(n, real.device), labels).reshape(real.shape)
            loss_g = self._g_loss(model.discriminate(fake, labels).float())
        trainer.optimize(loss_g, key="generator")
        for p in d_params:
            p.requires_grad_(True)
        m["loss_g"] = float(loss_g.detach())
        m["loss"] = m["loss_g"]
        return m

    @torch.no_grad()
    def validation_step(self, trainer, batch):
        model = trainer.model
        real = batch["inputs"]
        labels = batch.get("targets") if getattr(model.config, "num_classes", 0) else None
        fake = model(model.sample_noise(real.shape[0], real.device), labels).reshape(real.shape)
        d_real, d_fake = model.discriminate(real, labels), model.discriminate(fake, labels)
        # Einfache Qualitätsmaße: Abstand der Mittelwerte/Streuungen echter vs. erzeugter Daten.
        mean_gap = float((real.flatten(1).mean(0) - fake.flatten(1).mean(0)).abs().mean())
        std_gap = float((real.flatten(1).std(0) - fake.flatten(1).std(0)).abs().mean())
        return {"loss": mean_gap + std_gap, "mean_gap": mean_gap, "std_gap": std_gap,
                "d_real": float(d_real.mean()), "d_fake": float(d_fake.mean())}


# ===========================================================================
# Diffusion
# ===========================================================================

class DiffusionParadigmConfig(BaseConfig):
    loss = Param("mse", choices=["mse", "l1", "huber"], tooltip=T(
        "Verlust zwischen Vorhersage und Ziel.", "mse ist Standard; huber robuster.", "mse."))
    weighting = Param("none", choices={
        "none": "alle Rauschstufen gleich",
        "min_snr": "Min-SNR-γ (Hang et al. 2023): schnellere Konvergenz",
        "snr": "Gewicht = SNR (entspricht x₀-Verlust)",
    }, tooltip=T("Gewichtung der Zeitschritte.", "Balanciert leichte/schwere Rauschstufen.", "min_snr."),
        level=Level.ADVANCED)
    min_snr_gamma = Param(5.0, min=0.0, tooltip=T("γ für Min-SNR.", "Obergrenze der Gewichte.", "5."), level=Level.EXPERT)


class DiffusionParadigm(Paradigm):
    def compute(self, trainer, batch, train):
        model, c = trainer.model, self.cfg
        x0 = batch["inputs"].float()
        labels = batch.get("targets") if getattr(model.config, "num_classes", 0) else None
        t = torch.randint(0, model.config.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        with trainer.autocast():
            x_t = model.q_sample(x0, t, noise)
            pred = model(x_t, t, labels)
            target = model.target(x0, noise, t)
            if c.loss == "l1":
                per = (pred.float() - target).abs()
            elif c.loss == "huber":
                per = F.huber_loss(pred.float(), target, reduction="none")
            else:
                per = (pred.float() - target).square()
            per = per.flatten(1).mean(1)
            if c.weighting != "none":
                ac = model.alphas_cumprod[t]
                snr = ac / (1 - ac)
                if c.weighting == "snr":
                    w = snr
                else:
                    clipped = snr.clamp_max(c.min_snr_gamma)
                    p = model.config.prediction
                    w = clipped / snr if p == "eps" else (clipped / (snr + 1) if p == "v" else clipped)
                per = per * w
            loss = per.mean()
        return loss, {"loss": float(loss.detach())}


# ===========================================================================
# Präferenzen: Reward-Modell & DPO
# ===========================================================================

class PreferenceConfig(BaseConfig):
    margin = Param(0.0, tooltip=T(
        "Mindestabstand der Belohnungen bevorzugt vs. abgelehnt.", "Erzwingt deutlichere Trennung.", "0–1."))
    center_coef = Param(0.0, min=0.0, tooltip=T(
        "Strafe auf (r_c + r_r)² – zentriert Belohnungen um 0.", "Verhindert Drift der Belohnungsskala.", "1e-3."),
        level=Level.ADVANCED)


class Preference(Paradigm):
    """Reward-Modell (Bradley-Terry): Batch mit ``chosen`` und ``rejected``."""

    @staticmethod
    def _reward(trainer, x):
        r = main_output(call_model(trainer.model, x)).float()
        return r.reshape(r.shape[0], -1)[:, -1]

    def compute(self, trainer, batch, train):
        with trainer.autocast():
            rc, rr = self._reward(trainer, batch["chosen"]), self._reward(trainer, batch["rejected"])
            margin = batch.get("margin", self.cfg.margin)
            loss = -F.logsigmoid(rc - rr - margin).mean()
            if self.cfg.center_coef:
                loss = loss + self.cfg.center_coef * (rc + rr).square().mean()
        rc, rr = rc.detach(), rr.detach()
        return loss, {"loss": float(loss.detach()), "accuracy": float((rc > rr).float().mean()),
                      "reward_margin": float((rc - rr).mean())}


class DPOConfig(BaseConfig):
    beta = Param(0.1, min=0.0, tooltip=T(
        "Stärke der Bindung an das Referenzmodell (β).", "Klein = stärkere Anpassung an Präferenzen.", "0.1–0.5."))
    variant = Param("sigmoid", choices={
        "sigmoid": "DPO (Rafailov et al. 2023)", "ipo": "IPO (robuster gegen Überanpassung)",
        "hinge": "Hinge-Variante (SLiC)",
    }, tooltip=T("DPO-Verlustvariante.", "Unterschiedliche Robustheit.", "sigmoid."))
    label_smoothing = Param(0.0, min=0.0, max=0.5, tooltip=T(
        "Annahme verrauschter Präferenzen (cDPO).", "Robust gegen falsch gelabelte Paare.", "0–0.1."),
        level=Level.ADVANCED)
    reference_checkpoint = Param(None, type=str, tooltip=T(
        "Gewichte des Referenzmodells (None = Kopie des Start-Modells).", "Ankerpunkt der Optimierung.", "Meist None."),
        level=Level.ADVANCED)


class DPO(Paradigm):
    """DPO: Batch mit ``chosen``/``rejected`` Token-Folgen ``[B, T]`` (+ optional ``*_mask`` für Antwort-Tokens)."""

    def setup(self, trainer):
        ref = copy.deepcopy(trainer.model)
        if self.cfg.reference_checkpoint:
            state = torch.load(self.cfg.reference_checkpoint, map_location="cpu", weights_only=False)
            ref.load_state_dict(state.get("model", state))
        ref.requires_grad_(False)
        self.reference = ref.eval()

    @staticmethod
    def _logp(model, seq, mask):
        logits = main_output(model(seq[:, :-1])).float()
        lp = torch.gather(F.log_softmax(logits, -1), 2, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:].float() if mask is not None else torch.ones_like(lp)
        return (lp * m).sum(-1)

    def compute(self, trainer, batch, train):
        c = self.cfg
        ch, rj = batch["chosen"], batch["rejected"]
        mc, mr = batch.get("chosen_mask"), batch.get("rejected_mask")
        with trainer.autocast():
            pc, pr = self._logp(trainer.model, ch, mc), self._logp(trainer.model, rj, mr)
            with torch.no_grad():
                rc, rr = self._logp(self.reference, ch, mc), self._logp(self.reference, rj, mr)
            delta = (pc - rc) - (pr - rr)
            if c.variant == "ipo":
                loss = (delta - 1 / (2 * c.beta)).square().mean()
            elif c.variant == "hinge":
                loss = F.relu(1 - c.beta * delta).mean()
            else:
                loss = (-(1 - c.label_smoothing) * F.logsigmoid(c.beta * delta)
                        - c.label_smoothing * F.logsigmoid(-c.beta * delta)).mean()
        delta = delta.detach()
        return loss, {"loss": float(loss.detach()), "accuracy": float((delta > 0).float().mean()),
                      "reward_margin": float((c.beta * delta).mean())}


# ===========================================================================
# Registrierung
# ===========================================================================

class EvolutionStrategiesConfig(BaseConfig):
    __title__ = "Evolution Strategies (gradientenfrei)"
    base = Param("auto", choices={
        "auto": "aus dem Batch ableiten (targets vorhanden → supervised, sonst causal_lm)",
        "supervised": "Verlust wie beim überwachten Lernen", "causal_lm": "Verlust wie beim Sprachmodell"},
        tooltip=T("Woraus die Fitness (−Verlust) berechnet wird.", "ES optimiert jede berechenbare Zielgröße.", "auto."))
    pairs = Param(8, min=1, tooltip=T(
        "Anzahl Störungspaare (±ε) pro Schritt – Population = 2 × pairs.",
        "Mehr Paare = genauere Schätzung der Richtung, mehr Forward-Durchläufe.", "4–64."))
    sigma = Param(0.02, min=0.0, tooltip=T(
        "Stärke der zufälligen Störung der Gewichte.", "Zu klein: verrauscht; zu groß: ungenau.", "0.01–0.05."))
    rank_shaping = Param(True, tooltip=T(
        "Fitness-Formung über Ränge statt Rohwerte (OpenAI-ES).", "Robust gegen Ausreißer und Skalen.", "an."),
        level=Level.ADVANCED)


class EvolutionStrategies(Paradigm):
    """OpenAI-ES (Salimans et al. 2017): Gradient aus antithetischen Zufallsstörungen schätzen – kein Backward.

    Funktioniert auch mit nicht differenzierbaren Zielen und ist trivial parallelisierbar. Der geschätzte
    Gradient läuft durch den normalen Optimierer (Adam & Co.), Zeitplan, Clipping usw.
    """

    def _loss(self, trainer, batch) -> torch.Tensor:
        base = self.cfg.base
        if base == "auto":
            base = "supervised" if isinstance(batch, dict) and batch.get("targets") is not None else "causal_lm"
        if not hasattr(self, "_inner") or self._inner_name != base:
            self._inner = PARADIGMS.build(base)
            self._inner_name = base
        loss, _ = self._inner.compute(trainer, batch, False)
        return loss.detach().float()

    @torch.no_grad()
    def training_step(self, trainer, batch):
        params = [p for p in trainer.group_params[self.primary_key] if p.requires_grad]
        sigma, pairs = self.cfg.sigma, self.cfg.pairs
        base_loss = float(self._loss(trainer, batch))
        seeds = torch.randint(0, 2**31 - 1, (pairs,)).tolist()
        diffs = []
        for seed in seeds:
            gen = torch.Generator(device=params[0].device).manual_seed(seed)
            eps = [torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params]
            for p, e in zip(params, eps):
                p.add_(e, alpha=sigma)
            plus = float(self._loss(trainer, batch))
            for p, e in zip(params, eps):
                p.add_(e, alpha=-2 * sigma)
            minus = float(self._loss(trainer, batch))
            for p, e in zip(params, eps):
                p.add_(e, alpha=sigma)
            diffs.append((plus, minus))
        if self.cfg.rank_shaping:
            flat = [v for pm in diffs for v in pm]
            order = sorted(range(len(flat)), key=flat.__getitem__)
            ranks = [0.0] * len(flat)
            for r, i in enumerate(order):
                ranks[i] = r / max(1, len(flat) - 1) - 0.5
            weights = [ranks[2 * i] - ranks[2 * i + 1] for i in range(pairs)]
        else:
            weights = [plus - minus for plus, minus in diffs]
        grads = [torch.zeros_like(p) for p in params]
        for seed, w in zip(seeds, weights):
            gen = torch.Generator(device=params[0].device).manual_seed(seed)
            for g, p in zip(grads, params):
                g.add_(torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype), alpha=w)
        scale = 1.0 / (2 * pairs * sigma) if not self.cfg.rank_shaping else 1.0 / (pairs * sigma)
        for p, g in zip(params, grads):
            p.grad = g * scale
        trainer.step_with_grads(self.primary_key)
        spread = sum(abs(a - b) for a, b in diffs) / pairs
        return {"loss": base_loss, "es_streuung": spread}

    def validation_step(self, trainer, batch):
        return {"loss": float(self._loss(trainer, batch))}


R = PARADIGMS.register
R("supervised", lambda a: Supervised(a), config=SupervisedConfig, aliases=("ueberwacht",), tooltip=T(
    "Überwachtes Lernen: Eingabe → Ziel.", "Der Klassiker für Klassifikation und Regression.",
    "Tabellen, Bilder, Zeitreihen, Multi-Task."))
R("causal_lm", lambda a: CausalLM(a), config=CausalLMConfig, aliases=("language_modeling", "lm"), tags=("nlp",),
  tooltip=T("Kausales Sprachmodell-Training (nächstes Token).", "Selbstüberwacht auf Rohtext.", "GPT-artige Modelle."))
R("masked_lm", lambda a: MaskedLM(a), config=MaskedLMConfig, aliases=("mlm",), tags=("nlp",), tooltip=T(
    "Maskiertes Sprachmodell (BERT).", "Bidirektionales Vortraining.", "Encoder-Modelle."))
R("seq2seq", lambda a: Seq2Seq(a), config=Seq2SeqConfig, tooltip=T(
    "Sequenz-zu-Sequenz mit Teacher Forcing.", "Eingabe- und Ausgabesequenz verschieden.",
    "Übersetzung, Zusammenfassung, Mehrschritt-Prognosen."))
R("autoencoder", lambda a: AutoencoderParadigm(a), config=AutoencoderParadigmConfig, aliases=("reconstruction",),
  tooltip=T("Rekonstruktionstraining (+ KL-Annealing).", "Unüberwacht.", "Autoencoder, VAEs."))
R("contrastive", lambda a: Contrastive(a), config=ContrastiveConfig, aliases=("simclr",), level=Level.ADVANCED,
  tooltip=T("Kontrastives selbstüberwachtes Lernen.", "Repräsentationen ohne Labels.", "Vortraining, Embeddings."))
R("distillation", lambda a: Distillation(a), config=DistillationConfig, aliases=("kd", "lehrer_schueler"),
  level=Level.ADVANCED, tooltip=T(
      "Wissensdestillation: Schüler lernt Ausgaben (und optional innere Zustände) eines Lehrers.",
      "Kleine Modelle mit großem Wissen; heilt komprimierte Modelle; Weiterlernen ohne Vergessen.",
      "Kompression, Deployment auf kleinen Geräten, kontinuierliches Lernen."))
R("gan", lambda a: GANParadigm(a), config=GANParadigmConfig, tags=("generativ",), tooltip=T(
    "Adversariales Training (Generator vs. Diskriminator).", "Generative Modelle.", "Modell 'gan'."))
R("diffusion", lambda a: DiffusionParadigm(a), config=DiffusionParadigmConfig, tags=("generativ",), tooltip=T(
    "Diffusions-Training (Rauschen vorhersagen).", "Stabil, hohe Qualität.", "Modell 'diffusion'."))
R("preference", lambda a: Preference(a), config=PreferenceConfig, aliases=("reward_model", "bradley_terry"),
  level=Level.ADVANCED, tooltip=T("Reward-Modell aus Präferenzpaaren.", "Lernt menschliche Vorlieben als Belohnung.",
                                  "RLHF-Pipelines."))
R("dpo", lambda a: DPO(a), config=DPOConfig, level=Level.ADVANCED, tags=("nlp",), tooltip=T(
    "Direct Preference Optimization.", "Präferenzlernen ohne separates Reward-Modell/RL.",
    "Ausrichtung von Sprachmodellen."))
R("evolution_strategies", lambda a: EvolutionStrategies(a), config=EvolutionStrategiesConfig, aliases=("es",),
  level=Level.EXPERIMENTAL, tooltip=T(
      "Evolution Strategies: Gewichte zufällig stören, aus der Fitness die Richtung schätzen (ohne Backward).",
      "Gradientenfrei – für nicht differenzierbare Ziele, sehr gut parallelisierbar (OpenAI-ES).",
      "Kleine Modelle, Forschung, robuste Optimierung; langsamer als Backprop bei großen Netzen."))
