"""
ki_baukasten.models.base
========================

Gemeinsame Grundlage aller Modelle: ``ModelConfig``, ``InitConfig``,
``BaseModel`` und die Registry ``model``.

Was jedes Modell kann (``BaseModel``)
-------------------------------------
* ``config``                – die vollständige, speicherbare Konfiguration
* ``finish()``              – am Ende des Konstruktors: Initialisierung anwenden
* ``collect_aux_losses()``  – Hilfsverluste (z. B. MoE-Lastverteilung, VAE-KL)
  einsammeln, die Module während des Forward-Passes abgelegt haben
* ``checkpointable_blocks()`` – Blöcke, die Gradient-Checkpointing erlauben
  (genutzt von ``memory``)
* ``example_input()``       – Dummy-Eingabe passender Form (Tests, Speicher-
  schätzung, Formprüfung)
* ``num_parameters()`` / ``summary()`` – Übersicht über Aufbau und Größe

Registry ``model``
------------------
Fabrik-Signatur: ``factory(args) -> BaseModel``. Weil Modelle selbst
Komponenten sind, können sie **beliebig kombiniert** werden: ein Ensemble
aus Transformern, ein GAN mit CNN-Diskriminator, eine Pipeline aus
Autoencoder + MLP ... (siehe ``models.composite``).
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch import nn

from ..components.initializers import apply_initializer
from ..core import BaseConfig, ComponentParam, Level, NestedParam, Param, T, create_registry

MODELS = create_registry(
    "model",
    "Modelle",
    T(
        "Vollständige, vorgefertigte Modellarchitekturen.",
        "Jede ist bis ins Detail konfigurierbar und mit allen anderen kombinierbar.",
        "Ausgangspunkt jeder KI im Baukasten.",
    ),
    context_doc="factory(args) -> BaseModel",
)


# ===========================================================================
# Initialisierung
# ===========================================================================

def _init_param(default: str | None, what: str, tip: str = "", level: Level = Level.EXPERT) -> ComponentParam:
    return ComponentParam("initializer", default, level=level, tooltip=T(
        f"Initialisierer für {what}.",
        "Startwerte bestimmen, ob das Training stabil startet.",
        "Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.",
        tip,
    ))


class InitConfig(BaseConfig):
    """Standard-Initialisierung je Modultyp (feiner: Tensor-Chirurgie)."""

    __title__ = "Initialisierung"
    __tooltip__ = T(
        "Startwerte der Gewichte, getrennt nach Modultyp.",
        "Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.",
        "Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden.",
    )

    linear_weight = _init_param("keep", "Gewichte linearer Schichten", "Transformer: normal(std=0.02).",
                                level=Level.ADVANCED)
    linear_bias = _init_param("keep", "Biases linearer Schichten", "Häufig 'zeros'.")
    conv_weight = _init_param("keep", "Faltungsgewichte", "ResNet: kaiming_normal(mode=fan_out).", level=Level.ADVANCED)
    conv_bias = _init_param("keep", "Faltungs-Biases")
    embedding = _init_param("keep", "Embedding-Tabellen", "GPT/BERT: normal(std=0.02).", level=Level.ADVANCED)
    norm_weight = _init_param("keep", "Skalen von Normalisierungen", "Normalerweise 1 ('keep').")
    norm_bias = _init_param("keep", "Verschiebungen von Normalisierungen")
    rnn_weight_ih = _init_param("keep", "Eingangsgewichte rekurrenter Schichten", "Oft xavier_uniform.")
    rnn_weight_hh = _init_param("keep", "rekurrente Gewichte (Zustand→Zustand)", "Klassiker: orthogonal.")
    rnn_bias = _init_param("keep", "Biases rekurrenter Schichten")
    residual_projection = _init_param(None, "Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down)",
                                      "GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität.")


_NORM_TYPES: tuple[type, ...] = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._NormBase)


def apply_init(model: nn.Module, cfg: InitConfig, num_layers: int = 1) -> None:
    """Wendet ``InitConfig`` auf alle Module von ``model`` an."""

    def use(spec: Any, tensor: torch.Tensor | None) -> None:
        if spec is None or tensor is None or spec.name == "keep":
            return
        apply_initializer(spec, tensor.data, num_layers=num_layers)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            use(cfg.linear_weight, module.weight)
            use(cfg.linear_bias, module.bias)
        elif isinstance(module, nn.modules.conv._ConvNd):
            use(cfg.conv_weight, module.weight)
            use(cfg.conv_bias, module.bias)
        elif isinstance(module, nn.Embedding):
            use(cfg.embedding, module.weight)
        elif isinstance(module, (nn.RNNBase,)):
            for name, p in module.named_parameters(recurse=False):
                if name.startswith("weight_ih"):
                    use(cfg.rnn_weight_ih, p)
                elif name.startswith("weight_hh"):
                    use(cfg.rnn_weight_hh, p)
                elif name.startswith("bias"):
                    use(cfg.rnn_bias, p)
        elif isinstance(module, _NORM_TYPES) or type(module).__name__ in ("RMSNorm", "DynamicTanh"):
            use(cfg.norm_weight, getattr(module, "weight", None))
            use(cfg.norm_bias, getattr(module, "bias", None))
    if cfg.residual_projection is not None:
        for module in model.modules():
            if getattr(module, "_residual_proj", False) and isinstance(module, nn.Linear):
                use(cfg.residual_projection, module.weight)


def mark_residual_projections(module: nn.Module, names: tuple[str, ...] = ("o_proj", "down", "out_proj")) -> None:
    """Markiert Ausgangsprojektionen in Residual-Zweigen (für ``residual_projection``-Init)."""
    for name, sub in module.named_modules():
        if name.split(".")[-1] in names and isinstance(sub, nn.Linear):
            sub._residual_proj = True  # type: ignore[attr-defined]


# ===========================================================================
# ModelConfig / BaseModel
# ===========================================================================

class ModelConfig(BaseConfig):
    """Basisklasse aller Modell-Konfigurationen."""

    init = NestedParam(InitConfig, level=Level.ADVANCED, group="Initialisierung")
    extras = Param({}, level=Level.EXPERIMENTAL, group="Experimente", tooltip=T(
        "Freies dict für eigene, experimentelle Einstellungen.",
        "Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.",
        "Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben.",
    ))


class BaseModel(nn.Module):
    """Basisklasse aller Baukasten-Modelle (siehe Modul-Dokumentation)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    # ------------------------------------------------------------ Aufbau
    def finish(self) -> "BaseModel":
        """Muss am Ende jedes Modell-Konstruktors aufgerufen werden."""
        apply_init(self, self.config.init, num_layers=max(1, self.depth()))
        return self

    def depth(self) -> int:
        """Anzahl Hauptblöcke (für tiefenskalierte Initialisierung)."""
        return 1

    def checkpointable_blocks(self) -> list[nn.Module]:
        """Blöcke, die per Gradient-Checkpointing neu berechnet werden dürfen."""
        return []

    # ------------------------------------------------------------ Hilfsverluste
    def collect_aux_losses(self, reset: bool = True) -> torch.Tensor | None:
        """Summiert alle von Untermodulen abgelegten ``aux_loss``-Werte."""
        total = None
        for module in self.modules():
            loss = getattr(module, "aux_loss", None)
            if isinstance(loss, torch.Tensor):
                total = loss if total is None else total + loss
                if reset:
                    module.aux_loss = None
        return total

    # ------------------------------------------------------------ Eingaben
    def example_input(self, batch_size: int = 2) -> Any:
        """Dummy-Eingabe passender Form. Unterklassen überschreiben dies."""
        raise NotImplementedError(f"{type(self).__name__} definiert keine example_input().")

    # ------------------------------------------------------------ Übersicht
    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def summary(self, max_depth: int = 2, width: int = 100) -> str:
        """Tabellarische Übersicht der Module bis ``max_depth`` mit Parameterzahlen."""
        lines = [f"{type(self).__name__}: {self.num_parameters():,} Parameter "
                 f"({self.num_parameters(True):,} trainierbar)".replace(",", ".")]
        lines.append(f"{'Modul':<52} {'Typ':<26} {'Parameter':>14}")
        lines.append("-" * min(width, 94))
        for name, module in self.named_modules():
            depth = name.count(".") + 1 if name else 0
            if not name or depth > max_depth:
                continue
            n = sum(p.numel() for p in module.parameters())
            shown = ("  " * (depth - 1) + name.split(".")[-1])[:51]
            lines.append(f"{shown:<52} {type(module).__name__[:25]:<26} {n:>14,}".replace(",", "."))
        return "\n".join(lines)

    def parameter_table(self) -> Iterator[tuple[str, tuple[int, ...], int, bool]]:
        """(Name, Form, Anzahl, trainierbar) für jeden Parameter – Basis für Tensor-Chirurgie."""
        for name, p in self.named_parameters():
            yield name, tuple(p.shape), p.numel(), p.requires_grad


def build_model(spec: Any) -> BaseModel:
    """Baut ein Modell aus Name, dict oder ``ComponentSpec`` der Registry ``model``."""
    return MODELS.build(spec)
