"""
ki_baukasten.models.composite
=============================

**Kombinationen von Modellen** – KI-Systeme aus mehreren KIs.

Weil jedes Modell selbst eine Komponente der Registry ``model`` ist, lassen
sich Modelle beliebig verschachteln: ein Ensemble aus Pipelines aus
Transformern ist genauso möglich wie ein Mixture-of-Models aus CNN und MLP.

=====================  =======================================================
Modell                 Prinzip
=====================  =======================================================
``ensemble``           mehrere Modelle, gleiche Eingabe → Ausgaben kombiniert
                       (Mittel, gewichtet, Median, Maximum, Mehrheitsvotum,
                       gelerntes Stacking)
``pipeline``           Modelle hintereinander (Ausgabe → Eingabe), optional
                       mit automatischen Adapter-Schichten
``fusion``             mehrere Zweige mit **eigenen Eingaben** (multimodal:
                       Bild + Text + Tabelle) → Fusion → Kopf
``mixture_of_models``  Router entscheidet je Beispiel, welche Modelle
                       (Experten) wie stark zählen
``multitask``          ein gemeinsamer Rumpf, mehrere Köpfe (Multi-Task-Learning)
=====================  =======================================================
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..core import (BaseConfig, ComponentListParam, ComponentParam, ConfigListParam, Level, Param, T)
from .base import MODELS, BaseModel, ModelConfig


def _first_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        return _first_tensor(next(iter(x.values())))
    if isinstance(x, (tuple, list)):
        return _first_tensor(x[0])
    raise TypeError(f"Kein Tensor in {type(x).__name__}")


def _call(model: nn.Module, x: Any) -> Any:
    """Ruft ein Modell mit Tensor, dict (→ **kwargs) oder Tupel (→ *args) auf."""
    if isinstance(x, dict):
        return model(**x)
    if isinstance(x, tuple):
        return model(*x)
    return model(x)


def _main_output(y: Any) -> torch.Tensor:
    """Hauptausgabe eines Modells (dict-Ausgaben: 'reconstruction'/'output' oder erster Eintrag)."""
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, dict):
        for key in ("output", "logits", "reconstruction"):
            if key in y:
                return y[key]
        return next(iter(y.values()))
    if isinstance(y, (tuple, list)):
        return y[0]
    raise TypeError(f"Unbekannte Ausgabe {type(y).__name__}")


# ===========================================================================
# Ensemble
# ===========================================================================

class EnsembleConfig(ModelConfig):
    __title__ = "Ensemble"
    __tooltip__ = T(
        "Mehrere Modelle bekommen dieselbe Eingabe; ihre Ausgaben werden kombiniert.",
        "Ensembles sind fast immer genauer und robuster als jedes Einzelmodell und liefern Unsicherheitsmaße.",
        "Wettbewerbe, sicherheitskritische Vorhersagen, Unsicherheitsschätzung.",
    )
    members = ComponentListParam("model", default=["mlp", {"name": "mlp", "hidden_dims": [64]}], min_items=1, tooltip=T(
        "Mitglieder des Ensembles (beliebige Modelle mit gleicher Ein-/Ausgabeform).",
        "Vielfalt der Mitglieder (Architektur, Init, Daten) macht Ensembles stark.",
        "2–10 Mitglieder.",
    ))
    combine = Param("mean", choices={
        "mean": "Mittelwert", "weighted": "gewichteter Mittelwert (weights)", "median": "Median (robust gegen Ausreißer)",
        "max": "elementweises Maximum", "vote": "Mehrheitsvotum (Klassifikation; Ausgabe = Stimmenanteile)",
        "stacking": "gelernte lineare Kombination aller Ausgaben",
        "softmax_mean": "Mittel der Wahrscheinlichkeiten (Klassifikation; Ausgabe = log-Wahrscheinlichkeiten)",
    }, tooltip=T("Kombinationsregel.", "Bestimmt, wie Einzelvorhersagen verschmelzen.",
                 "Regression: mean/median; Klassifikation: softmax_mean/vote."))
    weights = Param([], item_type=float, min=0.0, tooltip=T(
        "Gewichte für 'weighted' (leer = gleich).", "Bessere Modelle stärker gewichten.", "Länge = Anzahl Mitglieder."))
    learnable_weights = Param(False, tooltip=T(
        "Gewichte für 'weighted' mittrainieren (Softmax-normiert).", "Datengetriebene Gewichtung.", "Experimente."),
        level=Level.ADVANCED)
    member_dropout = Param(0.0, min=0.0, max=0.95, tooltip=T(
        "Im Training zufällig Mitglieder weglassen.", "Erzwingt, dass jedes Mitglied alleine gut ist.", "Experimente."),
        level=Level.EXPERIMENTAL)
    output_dim = Param(None, type=int, min=1, tooltip=T(
        "Ausgabebreite (nur für 'stacking'; None = automatisch).", "Größe der Stacking-Schicht.", "Selten nötig."),
        level=Level.EXPERT)

    def check(self) -> list[str]:
        if self.combine == "weighted" and self.weights and len(self.weights) != len(self.members):
            return [f"weights hat {len(self.weights)} Einträge, es gibt {len(self.members)} Mitglieder."]
        return []


class Ensemble(BaseModel):
    def __init__(self, config: EnsembleConfig):
        super().__init__(config)
        config.validate()
        self.members = nn.ModuleList([m.build() for m in config.members])
        n = len(self.members)
        w = torch.tensor(config.weights or [1.0] * n)
        if config.learnable_weights:
            self.weight_logits = nn.Parameter(torch.log(w / w.sum()))
        else:
            self.register_buffer("fixed_weights", w / w.sum(), persistent=False)
        self.stack: nn.Module | None = None
        if config.combine == "stacking":
            out = config.output_dim or _main_output(_call(self.members[0], self.members[0].example_input())).shape[-1]
            self.stack = nn.Linear(n * out, out)
        self.finish()

    def forward(self, x: Any) -> torch.Tensor:
        c = self.config
        outs = [_main_output(_call(m, x)) for m in self.members]
        y = torch.stack(outs)  # [M, ...]
        if self.training and c.member_dropout > 0 and len(outs) > 1:
            keep = torch.rand(len(outs), device=y.device) >= c.member_dropout
            keep[torch.randint(len(outs), (1,))] = True
            y = y[keep]
        if c.combine == "mean":
            return y.mean(0)
        if c.combine == "weighted":
            w = self.weight_logits.softmax(0) if c.learnable_weights else self.fixed_weights
            w = w[: y.shape[0]] / w[: y.shape[0]].sum()
            return (w.view(-1, *([1] * (y.dim() - 1))) * y).sum(0)
        if c.combine == "median":
            return y.median(0).values
        if c.combine == "max":
            return y.amax(0)
        if c.combine == "vote":
            votes = F.one_hot(y.argmax(-1), y.shape[-1]).float()
            return votes.mean(0)
        if c.combine == "softmax_mean":
            return torch.log(y.softmax(-1).mean(0).clamp_min(1e-12))
        return self.stack(torch.cat(outs, dim=-1))  # stacking

    def disagreement(self, x: Any) -> torch.Tensor:
        """Uneinigkeit der Mitglieder (Standardabweichung) – ein einfaches Unsicherheitsmaß."""
        with torch.no_grad():
            return torch.stack([_main_output(_call(m, x)) for m in self.members]).std(0)

    def example_input(self, batch_size: int = 2) -> Any:
        return self.members[0].example_input(batch_size)


# ===========================================================================
# Pipeline
# ===========================================================================

class PipelineConfig(ModelConfig):
    __title__ = "Pipeline"
    __tooltip__ = T(
        "Modelle hintereinander: die Ausgabe eines Modells ist die Eingabe des nächsten.",
        "Baut Systeme aus spezialisierten Stufen (z. B. Merkmalsextraktor → Klassifikator).",
        "Feature-Extraktion + Kopf, Encoder + Prädiktor, mehrstufige Verarbeitung.",
    )
    stages = ComponentListParam("model", default=[{"name": "mlp", "output_dim": 32},
                                                  {"name": "mlp", "input_dim": 32, "output_dim": 1}],
                                min_items=1, tooltip=T(
        "Stufen in Reihenfolge.", "Jede Stufe darf ein beliebiges Modell sein.", "2–5 Stufen."))
    adapters = Param(True, tooltip=T(
        "Automatisch lineare Adapter einfügen, wenn Ausgabe- und Eingabebreite nicht passen.",
        "Verbindet Modelle unterschiedlicher Breite ohne Handarbeit.",
        "Fast immer an.",
    ))
    adapter_activation = ComponentParam("activation", "identity", tooltip=T(
        "Aktivierung nach Adaptern.", "Zusätzliche Nichtlinearität zwischen Stufen.", "identity oder gelu."),
        level=Level.ADVANCED)
    detach_between = Param(False, tooltip=T(
        "Gradientenfluss zwischen Stufen kappen.", "Frühere Stufen werden dann nicht mittrainiert (eingefroren).",
        "Vortrainierte Merkmalsextraktoren."), level=Level.ADVANCED)


class Pipeline(BaseModel):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.stages = nn.ModuleList([s.build() for s in config.stages])
        self.adapters = nn.ModuleList()
        for i in range(len(self.stages) - 1):
            adapter: nn.Module = nn.Identity()
            if config.adapters:
                with torch.no_grad():
                    y = _main_output(_call(self.stages[i].eval(), self.stages[i].example_input()))
                    nxt = _first_tensor(self.stages[i + 1].example_input())
                self.stages[i].train()
                if nxt.is_floating_point() and y.shape[-1] != nxt.shape[-1]:
                    adapter = nn.Sequential(nn.Linear(y.shape[-1], nxt.shape[-1]), config.adapter_activation.build())
            self.adapters.append(adapter)
        self.finish()

    def forward(self, x: Any) -> Any:
        h = _call(self.stages[0], x)
        for adapter, stage in zip(self.adapters, self.stages[1:]):
            h = adapter(_main_output(h))
            if self.config.detach_between:
                h = h.detach()
            h = stage(h)
        return h

    def example_input(self, batch_size: int = 2) -> Any:
        return self.stages[0].example_input(batch_size)


# ===========================================================================
# Fusion (multimodal)
# ===========================================================================

class FusionBranchConfig(BaseConfig):
    __title__ = "Fusionszweig"
    name = Param("zweig", tooltip=T("Name des Zweigs.", "Zur Übersicht.", "z. B. 'bild', 'text'."))
    input_key = Param("x", tooltip=T(
        "Schlüssel der Eingabe im Eingabe-dict.", "Welche Daten dieser Zweig bekommt.", "z. B. 'bild'."))
    model = ComponentParam("model", "mlp", tooltip=T(
        "Modell des Zweigs (Ausgabe = Merkmalsvektor).",
        "Jede Modalität mit passender Architektur (CNN für Bilder, Transformer für Text ...).",
        "output_dim des Modells = Merkmalsgröße.",
    ))
    freeze = Param(False, tooltip=T("Zweig einfrieren (nicht trainieren).", "Für vortrainierte Zweige.", "Transfer."),
                   level=Level.ADVANCED)


def _default_branches() -> list[FusionBranchConfig]:
    return [FusionBranchConfig(name="a", input_key="a", model={"name": "mlp", "input_dim": 8, "output_dim": 32}),
            FusionBranchConfig(name="b", input_key="b", model={"name": "mlp", "input_dim": 4, "output_dim": 32})]


class FusionConfig(ModelConfig):
    __title__ = "Fusion (multimodal)"
    __tooltip__ = T(
        "Mehrere Zweige mit eigenen Eingaben, deren Merkmale fusioniert werden.",
        "Verbindet unterschiedliche Datenarten in einem Modell.",
        "Bild + Text, Sensoren + Metadaten, Audio + Video.",
    )
    branches = ConfigListParam(FusionBranchConfig, default_factory=_default_branches, min_items=1, tooltip=T(
        "Zweige (je eine Modalität).", "Jeder Zweig ein eigenes Modell.", "2–4 Zweige."))
    fusion = Param("concat", choices={
        "concat": "Merkmale aneinanderhängen",
        "sum": "auf fusion_dim projizieren und addieren",
        "gated": "gelernte, beispielabhängige Gewichtung der Zweige",
        "attention": "Zweige als Tokens; Self-Attention zwischen Modalitäten",
    }, tooltip=T("Fusionsverfahren.", "Wie Modalitäten interagieren.", "concat einfach; attention ausdrucksstark."))
    fusion_dim = Param(64, min=1, tooltip=T(
        "Gemeinsame Breite (sum/gated/attention).", "Größe des gemeinsamen Raums.", "64–512."))
    attention_heads = Param(4, min=1, tooltip=T("Köpfe (attention).", "Interaktionsmuster.", "4."), level=Level.ADVANCED)
    head_hidden_dims = Param([64], item_type=int, min=1, tooltip=T(
        "Versteckte Schichten des gemeinsamen Kopfs.", "Verarbeitung nach der Fusion.", "[64]."))
    activation = ComponentParam("activation", "gelu", tooltip=T("Aktivierung im Kopf.", "Nichtlinearität.", "gelu."))
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T("Dropout im Kopf.", "Regularisierung.", "0.1."), group="Regularisierung")
    modality_dropout = Param(0.0, min=0.0, max=0.95, tooltip=T(
        "Im Training ganze Modalitäten zufällig auf 0 setzen.",
        "Robustheit gegen fehlende Modalitäten zur Laufzeit.", "Multimodale Systeme mit Ausfällen."),
        level=Level.EXPERIMENTAL, group="Regularisierung")
    output_dim = Param(1, min=1, tooltip=T("Ausgabegröße.", "Klassen/Zielwerte.", "Je nach Aufgabe."))


class Fusion(BaseModel):
    """Aufruf mit dict: ``model({"bild": ..., "text": ...})`` oder ``model(bild=..., text=...)``."""

    def __init__(self, config: FusionConfig):
        super().__init__(config)
        c = config
        self.branches = nn.ModuleList()
        dims = []
        for b in c.branches:
            m = b.model.build()
            if b.freeze:
                m.requires_grad_(False)
            self.branches.append(m)
            with torch.no_grad():
                dims.append(_main_output(_call(m.eval(), m.example_input())).shape[-1])
            m.train()
        self.keys = [b.input_key for b in c.branches]
        if c.fusion == "concat":
            self.proj = nn.ModuleList([nn.Identity() for _ in dims])
            feat = sum(dims)
        else:
            self.proj = nn.ModuleList([nn.Linear(d, c.fusion_dim) for d in dims])
            feat = c.fusion_dim
        if c.fusion == "gated":
            self.gate = nn.Linear(c.fusion_dim * len(dims), len(dims))
        if c.fusion == "attention":
            self.attn = nn.TransformerEncoderLayer(c.fusion_dim, c.attention_heads, 2 * c.fusion_dim,
                                                   dropout=c.dropout, batch_first=True, norm_first=True)
        layers: list[nn.Module] = []
        hd = [feat] + list(c.head_hidden_dims)
        for i in range(len(c.head_hidden_dims)):
            layers += [nn.Linear(hd[i], hd[i + 1]), c.activation.build(), nn.Dropout(c.dropout)]
        layers.append(nn.Linear(hd[-1], c.output_dim))
        self.head = nn.Sequential(*layers)
        self.finish()

    def forward(self, inputs: dict[str, Any] | None = None, **kwargs: Any) -> torch.Tensor:
        data = dict(inputs or {})
        data.update(kwargs)
        c = self.config
        feats = []
        for key, branch, proj in zip(self.keys, self.branches, self.proj):
            f = proj(_main_output(_call(branch, data[key])))
            if self.training and c.modality_dropout > 0:
                keep = (torch.rand(f.shape[0], 1, device=f.device) >= c.modality_dropout).to(f.dtype)
                f = f * keep
            feats.append(f)
        if c.fusion == "concat":
            h = torch.cat(feats, dim=-1)
        elif c.fusion == "sum":
            h = torch.stack(feats).sum(0)
        elif c.fusion == "gated":
            g = self.gate(torch.cat(feats, -1)).softmax(-1)  # [B, M]
            h = (torch.stack(feats, 1) * g.unsqueeze(-1)).sum(1)
        else:
            h = self.attn(torch.stack(feats, 1)).mean(1)
        return self.head(h)

    def example_input(self, batch_size: int = 2) -> dict[str, Any]:
        return {"inputs": {k: b.example_input(batch_size) for k, b in zip(self.keys, self.branches)}}


# ===========================================================================
# Mixture of Models
# ===========================================================================

class MixtureOfModelsConfig(ModelConfig):
    __title__ = "Mixture of Models"
    __tooltip__ = T(
        "Ein Router gewichtet je Beispiel mehrere vollständige Modelle (Experten).",
        "Spezialisierung: verschiedene Modelle für verschiedene Datenbereiche.",
        "Heterogene Daten, Regime-Wechsel (z. B. Märkte), Kombination verschiedener Architekturen.",
    )
    experts = ComponentListParam("model", default=["mlp", {"name": "mlp", "hidden_dims": [32, 32, 32]}], min_items=1,
                                 tooltip=T("Experten-Modelle (gleiche Ein-/Ausgabeform).",
                                           "Beliebige Architekturen.", "2–8."))
    router_hidden_dims = Param([32], item_type=int, min=1, tooltip=T(
        "Versteckte Schichten des Routers.", "Kapazität der Zuteilungsentscheidung.", "[32]."))
    top_k = Param(None, type=int, min=1, tooltip=T(
        "Nur die k besten Experten nutzen (None = alle gewichtet).", "Sparsere, spezialisiertere Zuteilung.", "1–2."))
    router_noise_std = Param(0.0, min=0.0, tooltip=T(
        "Rauschen auf Router-Logits im Training.", "Erkundung, verhindert Kollaps.", "0.1."), level=Level.EXPERT)
    balance_loss_coef = Param(0.01, min=0.0, tooltip=T(
        "Gewicht des Lastverteilungs-Verlusts.", "Verhindert, dass nur ein Experte genutzt wird.", "0.01."),
        group="Verluste")


class MixtureOfModels(BaseModel):
    """Router (MLP auf der abgeflachten Eingabe) gewichtet die Experten-Ausgaben."""

    def __init__(self, config: MixtureOfModelsConfig):
        super().__init__(config)
        c = config
        self.experts = nn.ModuleList([e.build() for e in c.experts])
        in_dim = _first_tensor(self.experts[0].example_input()).flatten(1).shape[1]
        dims = [in_dim] + list(c.router_hidden_dims)
        layers: list[nn.Module] = []
        for i in range(len(c.router_hidden_dims)):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        layers.append(nn.Linear(dims[-1], len(self.experts)))
        self.router = nn.Sequential(*layers)
        self.aux_loss: torch.Tensor | None = None
        self.finish()

    def forward(self, x: Any) -> torch.Tensor:
        c = self.config
        logits = self.router(_first_tensor(x).flatten(1).float())
        if self.training and c.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * c.router_noise_std
        probs = logits.softmax(-1)  # [B, E]
        w = probs
        if c.top_k is not None and c.top_k < probs.shape[-1]:
            top = probs.topk(c.top_k, -1)
            w = torch.zeros_like(probs).scatter(-1, top.indices, top.values)
            w = w / w.sum(-1, keepdim=True)
        outs = torch.stack([_main_output(_call(e, x)) for e in self.experts], 1)  # [B, E, ...]
        y = (outs * w.view(*w.shape, *([1] * (outs.dim() - 2))).to(outs.dtype)).sum(1)
        if c.balance_loss_coef > 0:
            load = (w > 0).float().mean(0)
            self.aux_loss = c.balance_loss_coef * len(self.experts) * (load * probs.mean(0)).sum()
        return y

    def example_input(self, batch_size: int = 2) -> Any:
        return self.experts[0].example_input(batch_size)


# ===========================================================================
# Multi-Task
# ===========================================================================

class TaskHeadConfig(BaseConfig):
    __title__ = "Aufgaben-Kopf"
    name = Param("aufgabe", tooltip=T("Name der Aufgabe (Schlüssel der Ausgabe).", "Für Verluste/Targets.", "z. B. 'klasse'."))
    output_dim = Param(1, min=1, tooltip=T("Ausgabegröße dieser Aufgabe.", "Klassen/Zielwerte.", "Je Aufgabe."))
    hidden_dims = Param([], item_type=int, min=1, tooltip=T("Versteckte Schichten des Kopfs.", "Kapazität.", "Meist leer."))
    activation = ComponentParam("activation", "relu", tooltip=T("Aktivierung.", "Nichtlinearität.", "relu."))


class MultiTaskConfig(ModelConfig):
    __title__ = "Multi-Task"
    __tooltip__ = T(
        "Gemeinsamer Rumpf mit mehreren aufgabenspezifischen Köpfen.",
        "Aufgaben helfen sich gegenseitig (geteilte Repräsentation), spart Rechenzeit.",
        "Mehrere Vorhersagen aus denselben Daten (z. B. Klasse + Position + Qualität).",
    )
    backbone = ComponentParam("model", "mlp", default_args={"output_dim": 64}, tooltip=T(
        "Gemeinsamer Rumpf (Ausgabe = gemeinsame Merkmale).", "Lernt die geteilte Repräsentation.", "Beliebiges Modell."))
    heads = ConfigListParam(TaskHeadConfig, default_factory=lambda: [
        TaskHeadConfig(name="klasse", output_dim=3), TaskHeadConfig(name="wert", output_dim=1)], min_items=1,
        tooltip=T("Köpfe (je Aufgabe einer).", "Aufgabenspezifische Ausgaben.", "2–5."))
    backbone_activation = ComponentParam("activation", "relu", tooltip=T(
        "Aktivierung zwischen Rumpf und Köpfen.", "Nichtlinearität.", "relu oder identity."), level=Level.ADVANCED)


class MultiTask(BaseModel):
    """Liefert ein dict ``{aufgabe: ausgabe}``."""

    def __init__(self, config: MultiTaskConfig):
        super().__init__(config)
        self.backbone = config.backbone.build()
        with torch.no_grad():
            feat = _main_output(_call(self.backbone.eval(), self.backbone.example_input())).shape[-1]
        self.backbone.train()
        self.act = config.backbone_activation.build()
        self.heads = nn.ModuleDict()
        for h in config.heads:
            dims = [feat] + list(h.hidden_dims)
            layers: list[nn.Module] = []
            for i in range(len(h.hidden_dims)):
                layers += [nn.Linear(dims[i], dims[i + 1]), h.activation.build()]
            layers.append(nn.Linear(dims[-1], h.output_dim))
            self.heads[h.name] = nn.Sequential(*layers)
        self.finish()

    def forward(self, x: Any) -> dict[str, torch.Tensor]:
        h = self.act(_main_output(_call(self.backbone, x)))
        return {name: head(h) for name, head in self.heads.items()}

    def example_input(self, batch_size: int = 2) -> Any:
        return self.backbone.example_input(batch_size)


R = MODELS.register
R("ensemble", Ensemble, config=EnsembleConfig, tags=("kombination",), tooltip=T(
    "Ensemble mehrerer Modelle.", "Genauer, robuster, mit Unsicherheitsmaß.", "Kombination beliebiger Modelle."))
R("pipeline", Pipeline, config=PipelineConfig, tags=("kombination",), tooltip=T(
    "Modelle hintereinander geschaltet.", "Mehrstufige Systeme.", "Extraktor + Kopf."))
R("fusion", Fusion, config=FusionConfig, aliases=("multimodal",), tags=("kombination",), tooltip=T(
    "Multimodale Fusion mehrerer Zweige.", "Verbindet Datenarten.", "Bild + Text + Tabelle."))
R("mixture_of_models", MixtureOfModels, config=MixtureOfModelsConfig, aliases=("mom",), level=Level.ADVANCED,
  tags=("kombination",), tooltip=T("Router über ganze Modelle.", "Spezialisierte Experten.", "Heterogene Daten."))
R("multitask", MultiTask, config=MultiTaskConfig, aliases=("multi_task",), tags=("kombination",), tooltip=T(
    "Gemeinsamer Rumpf, mehrere Köpfe.", "Multi-Task-Learning.", "Mehrere Ziele gleichzeitig."))
