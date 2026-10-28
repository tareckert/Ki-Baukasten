"""
ki_baukasten.models.graph
=========================

**Freies Netz** – ein beliebiger gerichteter Graph aus Schichten.

Wofür?
------
Wenn keine vorgefertigte Architektur passt, baut man das Netz Knoten für
Knoten selbst – mit jeder Schicht aus der Registry ``layer`` (Linear,
Faltung, LSTM, Attention, Transformer-Block, MLP, Normalisierung, Tensor-
Funktionen ...). Verzweigungen, Residualverbindungen, mehrere Eingänge und
Ausgänge, geteilte Gewichte und wiederholte Blöcke sind möglich.

Beispiel (als JSON)::

    {
      "inputs": [{"name": "bild", "shape": [3, 32, 32]}, {"name": "meta", "shape": [5]}],
      "nodes": [
        {"name": "c1",   "layer": {"name": "conv", "out_channels": 16}, "inputs": ["bild"]},
        {"name": "a1",   "layer": "activation"},
        {"name": "pool", "layer": "global_pool"},
        {"name": "m",    "layer": {"name": "linear", "out_features": 16}, "inputs": ["meta"]},
        {"name": "fuse", "layer": {"name": "mlp", "output_dim": 3}, "inputs": ["pool", "m"], "merge": "concat"}
      ],
      "outputs": ["fuse"]
    }

Regeln
------
* Knoten werden **in der angegebenen Reihenfolge** gebaut und ausgeführt;
  ``inputs`` dürfen nur Eingänge oder *frühere* Knoten nennen (→ garantiert
  zyklenfrei).
* ``inputs`` leer → Ausgabe des vorherigen Knotens (bzw. erster Eingang).
* Mehrere Eingänge werden mit ``merge`` vereint (concat, add, mul, gated ...).
* Eingangsgrößen jeder Schicht werden **automatisch** über einen Probelauf
  mit Dummy-Daten bestimmt – man gibt nur Ausgangsgrößen an.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..core import BaseConfig, ComponentParam, ConfigError, ConfigListParam, Level, Param, T
from .base import MODELS, BaseModel, ModelConfig


class GraphInputConfig(BaseConfig):
    __title__ = "Eingang"
    name = Param("x", tooltip=T("Name des Eingangs.", "Wird von Knoten in 'inputs' referenziert.", "z. B. 'bild', 'text'."))
    shape = Param([16], item_type=int, min=1, tooltip=T(
        "Form eines Beispiels OHNE Batch-Achse.", "Für den automatischen Probelauf.", "[16], [3, 32, 32], [128] ..."))
    dtype = Param("float", choices={"float": "Gleitkommazahlen", "long": "Ganzzahlen (z. B. Token-IDs)"},
                  tooltip=T("Datentyp des Eingangs.", "long für Embeddings.", "Text-IDs: long."))
    max_value = Param(1, min=1, tooltip=T(
        "Obergrenze (exklusiv) für Dummy-Ganzzahlen im Probelauf (nur long).",
        "Muss ≤ Vokabulargröße folgender Embeddings sein.", "Selten zu ändern."), level=Level.EXPERT)


class GraphNodeConfig(BaseConfig):
    __title__ = "Knoten"
    __tooltip__ = T("Ein Knoten des freien Netzes (eine Schicht mit Eingängen).",
                    "Grundbaustein frei definierter Architekturen.", "Modell 'graph'.")
    name = Param("knoten", tooltip=T("Eindeutiger Name.", "Für Verweise anderer Knoten.", "Sprechende Namen helfen."))
    layer = ComponentParam("layer", "linear", tooltip=T(
        "Schicht aus der Registry 'layer'.", "Was dieser Knoten berechnet.", "Beliebig."))
    inputs = Param([], item_type=str, tooltip=T(
        "Namen der Eingänge/Knoten, die hier hineinfließen (leer = vorheriger Knoten).",
        "Definiert die Kanten des Graphen.", "Verzweigungen, Skip-Verbindungen."))
    merge = ComponentParam("merge", "concat", tooltip=T(
        "Zusammenführung bei mehreren Eingängen.", "concat, add (Residual), mul, gated ...", "Nur bei > 1 Eingang."))
    repeat = Param(1, min=1, tooltip=T(
        "Schicht n-mal hintereinander (je eigene Gewichte).", "Kompakt für wiederholte Blöcke.", "6× transformer_block."))
    share_weights_with = Param(None, type=str, tooltip=T(
        "Gewichte mit einem früheren Knoten teilen (gleiche Schicht, gleiche Parameter).",
        "Siamesische Netze, Gewichtsbindung.", "Muss formkompatibel sein."), level=Level.EXPERT)


def _default_nodes() -> list[GraphNodeConfig]:
    return [
        GraphNodeConfig(name="dense1", layer={"name": "linear", "out_features": 64}),
        GraphNodeConfig(name="act1", layer={"name": "activation", "activation": "gelu"}),
        GraphNodeConfig(name="out", layer={"name": "linear", "out_features": 1}),
    ]


class GraphModelConfig(ModelConfig):
    __title__ = "Freies Netz (Graph)"
    __tooltip__ = T(
        "Frei definierter Netzgraph aus beliebigen Schichten.",
        "Maximale Freiheit: jede denkbare Architektur ohne neuen Code.",
        "Prototypen, Forschung, ungewöhnliche Datenflüsse, Multi-Input/-Output.",
    )
    inputs = ConfigListParam(GraphInputConfig, default_factory=lambda: [GraphInputConfig()], min_items=1, tooltip=T(
        "Eingänge des Netzes.", "Mehrere Eingänge für multimodale Daten möglich.", "Mindestens einer."))
    nodes = ConfigListParam(GraphNodeConfig, default_factory=_default_nodes, min_items=1, tooltip=T(
        "Knoten in Ausführungsreihenfolge.", "Die eigentliche Architektur.", "Beliebig viele."))
    outputs = Param([], item_type=str, tooltip=T(
        "Ausgabeknoten (leer = letzter Knoten; mehrere → dict).", "Multi-Output-Netze.", "Multi-Task."))

    def check(self) -> list[str]:
        problems, known = [], [i.name for i in self.inputs]
        for node in self.nodes:
            if node.name in known:
                problems.append(f"Name '{node.name}' doppelt vergeben.")
            for src in node.inputs:
                if src not in known:
                    problems.append(f"Knoten '{node.name}': Eingang '{src}' unbekannt oder erst später definiert.")
            if node.share_weights_with is not None and node.share_weights_with not in known:
                problems.append(f"Knoten '{node.name}': share_weights_with '{node.share_weights_with}' unbekannt.")
            known.append(node.name)
        for out in self.outputs:
            if out not in known:
                problems.append(f"Ausgang '{out}' existiert nicht.")
        return problems


class GraphModel(BaseModel):
    """Führt den Knotengraph aus. Aufruf: ``model(x)`` oder ``model(name1=..., name2=...)``."""

    def __init__(self, config: GraphModelConfig):
        super().__init__(config)
        config.validate()
        self.input_names = [i.name for i in config.inputs]
        self.layers = nn.ModuleDict()
        self.merges = nn.ModuleDict()
        self.plan: list[tuple[str, list[str]]] = []
        values = self._dummy_inputs(2)
        prev = self.input_names[0]
        for node in config.nodes:
            srcs = list(node.inputs) or [prev]
            tensors = [values[s] for s in srcs]
            if len(tensors) > 1:
                merge = node.merge.build([tuple(t.shape[1:]) for t in tensors])
                self.merges[node.name] = merge
                with torch.no_grad():
                    h = merge(*tensors)
            else:
                h = tensors[0]
            if node.share_weights_with is not None:
                module = self.layers[node.share_weights_with]
            else:
                chain = []
                for _ in range(node.repeat):
                    layer = node.layer.build(tuple(h.shape[1:]))
                    chain.append(layer)
                    h = self._probe(layer, h)
                module = chain[0] if len(chain) == 1 else nn.Sequential(*chain)
                values[node.name] = h
            if node.share_weights_with is not None:
                h = self._probe(module, h)
                values[node.name] = h
            self.layers[node.name] = module
            self.plan.append((node.name, srcs))
            prev = node.name
        self.output_names = list(config.outputs) or [prev]
        self._shapes = {k: tuple(v.shape[1:]) for k, v in values.items()}
        self.finish()

    @staticmethod
    def _probe(layer: nn.Module, h: torch.Tensor) -> torch.Tensor:
        """Probelauf im Eval-Modus (verfälscht keine BatchNorm-Statistik)."""
        was = layer.training
        layer.eval()
        try:
            with torch.no_grad():
                out = layer(h)
        except Exception as exc:  # verständliche Fehlermeldung mit Form
            raise ConfigError(f"Schicht {type(layer).__name__} passt nicht zur Eingabeform {tuple(h.shape)}: {exc}") from exc
        layer.train(was)
        return out

    def _dummy_inputs(self, batch: int) -> dict[str, torch.Tensor]:
        out = {}
        for inp in self.config.inputs:
            if inp.dtype == "long":
                out[inp.name] = torch.randint(0, inp.max_value, (batch, *inp.shape))
            else:
                out[inp.name] = torch.randn(batch, *inp.shape)
        return out

    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Form (ohne Batch) jedes Eingangs/Knotens – hilfreich beim Entwerfen."""
        return dict(self._shapes)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Any:
        values: dict[str, torch.Tensor] = dict(zip(self.input_names, args))
        values.update(kwargs)
        missing = [n for n in self.input_names if n not in values]
        if missing:
            raise ValueError(f"Fehlende Eingänge: {missing}")
        for name, srcs in self.plan:
            tensors = [values[s] for s in srcs]
            h = self.merges[name](*tensors) if len(tensors) > 1 else tensors[0]
            values[name] = self.layers[name](h)
        if len(self.output_names) == 1:
            return values[self.output_names[0]]
        return {n: values[n] for n in self.output_names}

    def example_input(self, batch_size: int = 2) -> Any:
        dummy = self._dummy_inputs(batch_size)
        return dummy[self.input_names[0]] if len(dummy) == 1 else dummy


MODELS.register("graph", GraphModel, config=GraphModelConfig, aliases=("frei", "custom", "dag"), tags=("frei",),
                tooltip=T(
    "Freies Netz aus beliebigen Schichten (gerichteter Graph).",
    "Jede denkbare Architektur ohne eigenen Code.",
    "Prototypen, Forschung, Multi-Input/-Output.",
))
