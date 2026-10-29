"""
ki_baukasten.models.mlp
=======================

Mehrschichtiges Perzeptron (MLP) – das universelle Grundmodell.

Einsatz
-------
Tabellarische Daten, kleine Regressions-/Klassifikationsaufgaben, Köpfe
anderer Modelle, Policies im Reinforcement Learning, Generator/Diskriminator
in einfachen GANs, Denoiser in Vektor-Diffusionsmodellen ...

Freiheitsgrade
--------------
* Breite **jeder** Schicht (``hidden_dims``)
* **Reihenfolge** der Operationen pro Schicht (``layer.order``), z. B.
  ``["linear", "norm", "activation", "dropout"]`` oder Pre-Norm-artig
  ``["norm", "linear", "activation"]``
* Residualverbindungen (auch bei Breitenwechsel per Projektion)
* Gewichts-Parametrisierung (Weight-Norm, Spectral-Norm)
* **Schichtweise Abweichungen** über ``layer_overrides``
  (z. B. andere Aktivierung nur in Schicht 2)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrizations

from ..components.layers import LAYERS
from ..components.normalization import build_norm
from ..core import (BaseConfig, ComponentParam, Level, NestedParam, Param, T, resolve_layer_config,
                    validate_overrides)
from .base import MODELS, BaseModel, ModelConfig

_ORDER_ITEMS = ("linear", "norm", "activation", "dropout")


def _check_order(order: list[str]) -> str | None:
    if "linear" not in order:
        return "order muss 'linear' enthalten."
    bad = [o for o in order if o not in _ORDER_ITEMS]
    if bad:
        return f"unbekannte Elemente {bad}; erlaubt: {list(_ORDER_ITEMS)}"
    return None


class MLPLayerConfig(BaseConfig):
    """Einstellungen *einer* versteckten Schicht (Basis für alle; per Override änderbar)."""

    __title__ = "MLP-Schicht"
    __tooltip__ = T(
        "Bauplan einer versteckten MLP-Schicht.",
        "Gilt für alle Schichten; Abweichungen über layer_overrides.",
        "Feintuning jeder einzelnen Schicht.",
    )

    activation = ComponentParam("activation", "relu", tooltip=T(
        "Aktivierungsfunktion der Schicht.", "Nichtlinearität.", "relu/gelu/silu sind gute Standards."))
    norm = ComponentParam("normalization", None, tooltip=T(
        "Normalisierung (None = keine).",
        "Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.",
        "Tiefe MLPs (> 4 Schichten).",
    ))
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout-Rate.", "Regularisierung gegen Überanpassung.", "Kleine Datensätze: 0.1–0.5."), group="Regularisierung")
    bias = Param(True, tooltip=T("Bias in der linearen Schicht.", "Verschiebung der Ausgabe.", "Vor BatchNorm überflüssig."),
                 level=Level.ADVANCED)
    order = Param(["linear", "norm", "activation", "dropout"], item_type=str, validator=_check_order, tooltip=T(
        "Reihenfolge der Operationen innerhalb der Schicht.",
        "Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.",
        "Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.",
        "Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal).",
    ), level=Level.EXPERT)
    residual = Param("none", choices={
        "none": "keine Residualverbindung",
        "identity": "x + f(x), nur wenn Ein- und Ausgabebreite gleich sind (sonst ignoriert)",
        "projection": "x·W + f(x) – lineare Projektion bei Breitenwechsel",
    }, tooltip=T(
        "Residual-/Skip-Verbindung um die Schicht.",
        "Erleichtert den Gradientenfluss in tiefen MLPs erheblich.",
        "MLPs mit vielen Schichten.",
    ), level=Level.ADVANCED)
    parametrization = Param("none", choices={
        "none": "normale Gewichte",
        "weight_norm": "Weight Normalization: w = g · v/||v|| (Salimans & Kingma 2016)",
        "spectral_norm": "Spektralnorm ≤ 1 (Lipschitz-Beschränkung)",
    }, tooltip=T(
        "Umparametrisierung der Gewichtsmatrix.",
        "Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.",
        "GANs, Lipschitz-beschränkte Netze, RL-Kritiker.",
    ), level=Level.EXPERT)


class MLPConfig(ModelConfig):
    __title__ = "MLP"
    __tooltip__ = T(
        "Mehrschichtiges Perzeptron (vollständig verbundenes Netz).",
        "Universeller Funktionsapproximator mit wenig Annahmen über die Daten.",
        "Tabellendaten, Köpfe, RL-Policies, einfache Generatoren.",
    )

    input_dim = Param(16, min=1, tooltip=T(
        "Anzahl Eingabe-Features.", "Muss zur Datenbreite passen.", "Spaltenzahl der Tabelle / Größe des Vektors."))
    output_dim = Param(1, min=1, tooltip=T(
        "Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).",
        "Bestimmt die Form der Vorhersage.",
        "Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen.",
    ))
    hidden_dims = Param([128, 128], item_type=int, min=1, tooltip=T(
        "Breiten der versteckten Schichten (Liste; Länge = Tiefe).",
        "Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.",
        "Tabellendaten: 2–4 Schichten à 64–512.",
        "Leere Liste = lineares Modell (logistische/lineare Regression).",
    ))
    layer = NestedParam(MLPLayerConfig)
    layer_overrides = Param({}, tooltip=T(
        "Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.",
        "Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.",
        "Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...",
        "Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}}",
    ), level=Level.ADVANCED)
    input_dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout direkt auf die Eingabe.", "Robustheit gegen fehlende/verrauschte Features.", "Tabellendaten, Denoising."),
        level=Level.ADVANCED, group="Regularisierung")
    output_activation = ComponentParam("activation", "identity", tooltip=T(
        "Aktivierung der Ausgabeschicht.",
        "identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.",
        "Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh.",
    ))
    output_bias = Param(True, tooltip=T("Bias der Ausgabeschicht.", "Verschiebung der Ausgabe.", "Standard."),
                        level=Level.ADVANCED)

    def check(self) -> list[str]:
        return validate_overrides(self.layer, self.layer_overrides, max(1, len(self.hidden_dims)))


class MLPLayer(nn.Module):
    """Eine versteckte Schicht gemäß ``MLPLayerConfig``."""

    def __init__(self, in_dim: int, out_dim: int, cfg: MLPLayerConfig):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim, bias=cfg.bias)
        if cfg.parametrization == "weight_norm":
            linear = parametrizations.weight_norm(linear)
        elif cfg.parametrization == "spectral_norm":
            linear = parametrizations.spectral_norm(linear)
        ops: dict[str, nn.Module] = {
            "linear": linear,
            "norm": build_norm(cfg.norm, out_dim if cfg.order.index("linear") < cfg.order.index("norm") else in_dim)
            if "norm" in cfg.order else nn.Identity(),
            "activation": cfg.activation.build(),
            "dropout": nn.Dropout(cfg.dropout),
        }
        self.ops = nn.ModuleDict({k: ops[k] for k in cfg.order})
        self.order = list(cfg.order)
        self.skip: nn.Module | None = None
        if cfg.residual == "identity" and in_dim == out_dim:
            self.skip = nn.Identity()
        elif cfg.residual == "projection":
            self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for key in self.order:
            h = self.ops[key](h)
        return h + self.skip(x) if self.skip is not None else h


class MLP(BaseModel):
    """MLP gemäß ``MLPConfig``. Eingabe ``[..., input_dim]`` → ``[..., output_dim]``."""

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        config.validate()
        dims = [config.input_dim] + list(config.hidden_dims)
        n = len(config.hidden_dims)
        self.input_dropout = nn.Dropout(config.input_dropout)
        self.layers = nn.ModuleList([
            MLPLayer(dims[i], dims[i + 1], resolve_layer_config(config.layer, config.layer_overrides, i, n))
            for i in range(n)
        ])
        self.out = nn.Linear(dims[-1], config.output_dim, bias=config.output_bias)
        self.out_act = config.output_activation.build()
        self.finish()

    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_dropout(x)
        for layer in self.layers:
            h = layer(h)
        return self.out_act(self.out(h))

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        return torch.randn(batch_size, self.config.input_dim)


MODELS.register("mlp", MLP, config=MLPConfig, aliases=("perzeptron", "dense_net"), tags=("tabular", "basis"), tooltip=T(
    "Mehrschichtiges Perzeptron.",
    "Einfach, schnell, universell.",
    "Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe.",
))


# Als Schicht für freie Graph-Modelle verfügbar machen (Eingabebreite wird automatisch ermittelt).
class MLPLayerSpecConfig(MLPConfig):
    __title__ = "MLP (als Schicht)"
    input_dim = Param(None, type=int, min=1, tooltip=T(
        "Eingabebreite (None = automatisch aus der Eingabe).", "Wird im Graph automatisch bestimmt.", "Graph-Modelle."))


def _mlp_layer(a: MLPLayerSpecConfig, in_shape: tuple[int, ...]) -> nn.Module:
    cfg = MLPConfig.from_dict({**a.to_dict(), "input_dim": a.input_dim or in_shape[-1]})
    return MLP(cfg)


LAYERS.register("mlp", _mlp_layer, config=MLPLayerSpecConfig, tooltip=T(
    "Komplettes MLP als eine Schicht.", "Kompakter Teilblock in freien Netzen.", "Graph-Modelle, Köpfe."))
