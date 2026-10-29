"""
ki_baukasten.models.gnn
=======================

Graph-Neuronale Netze (GNN) in reinem PyTorch – ohne Zusatzbibliotheken.

Datenformat
-----------
* ``x``:          Knotenmerkmale ``[N, F]`` (alle Graphen eines Batches hintereinander)
* ``edge_index``: Kanten ``[2, E]`` als (Quelle, Ziel)-Paare (Ganzzahlen)
* ``batch``:      ``[N]`` – zu welchem Graphen gehört jeder Knoten (für Graph-Aufgaben)

Faltungsarten (``conv``)
------------------------
* ``gcn``  – Graph Convolutional Network (Kipf & Welling): normalisierte Nachbarschaftsmittelung
* ``sage`` – GraphSAGE: eigene Merkmale + aggregierte Nachbarn getrennt gewichtet
* ``gin``  – Graph Isomorphism Network: (1+ε)·x + Σ Nachbarn → MLP (sehr ausdrucksstark)
* ``gat``  – Graph Attention Network: gelernte Gewichtung der Nachbarn per Attention

Aufgaben
--------
* ``readout="none"`` → Vorhersage **pro Knoten** (Knotenklassifikation)
* ``readout="mean"/"sum"/"max"`` → Vorhersage **pro Graph** (Molekül-Eigenschaften ...)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..components.normalization import build_norm
from ..core import (BaseConfig, ComponentParam, Level, NestedParam, Param, T, resolve_layer_config,
                    validate_overrides)
from .base import MODELS, BaseModel, ModelConfig
from ..core.runtime import is_exporting


def scatter(src: torch.Tensor, index: torch.Tensor, n: int, reduce: str) -> torch.Tensor:
    """Aggregiert Zeilen von ``src`` nach ``index`` auf ``n`` Ziele (sum/mean/max)."""
    out = src.new_zeros(n, *src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    if reduce == "sum":
        return out.scatter_add_(0, idx, src)
    if reduce == "mean":
        return out.scatter_reduce_(0, idx, src, reduce="mean", include_self=False)
    return out.scatter_reduce_(0, idx, src, reduce="amax", include_self=False)


def add_self_loops(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    loops = torch.arange(n, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loops], dim=1)


class GNNLayerConfig(BaseConfig):
    __title__ = "GNN-Schicht"
    conv = Param("gcn", choices={
        "gcn": "Graph Convolution (normalisierte Mittelung)",
        "sage": "GraphSAGE (selbst + Nachbarn getrennt)",
        "gin": "Graph Isomorphism Network (maximal ausdrucksstark)",
        "gat": "Graph Attention (gelernte Nachbargewichte)",
    }, tooltip=T("Art der Graph-Faltung.", "Bestimmt, wie Nachbarinformation kombiniert wird.",
                 "gcn: Standard; gin: Graph-Klassifikation; gat: heterogene Nachbarschaften."))
    aggregation = Param("mean", choices=["sum", "mean", "max"], tooltip=T(
        "Aggregation der Nachbarn (sage/gin).", "sum unterscheidet Knotengrade, mean ist robuster.", "gin: sum."),
        level=Level.ADVANCED)
    heads = Param(4, min=1, tooltip=T("Attention-Köpfe (gat).", "Mehrere Sichtweisen.", "4–8."), level=Level.ADVANCED)
    gat_concat = Param(True, tooltip=T(
        "GAT-Köpfe konkatenieren (sonst mitteln).", "Konkatenation = mehr Kapazität.", "Letzte Schicht: mitteln."),
        level=Level.EXPERT)
    self_loops = Param(True, tooltip=T(
        "Jeder Knoten ist auch sein eigener Nachbar (gcn/gat).", "Eigene Merkmale fließen mit ein.", "Fast immer an."),
        level=Level.ADVANCED)
    activation = ComponentParam("activation", "relu", tooltip=T("Aktivierung.", "Nichtlinearität.", "relu/elu."))
    norm = ComponentParam("normalization", None, tooltip=T("Normalisierung.", "Stabilität tiefer GNNs.", "layernorm."),
                          level=Level.ADVANCED)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T("Dropout.", "Regularisierung.", "0.1–0.5."), group="Regularisierung")
    residual = Param(False, tooltip=T(
        "Residualverbindung (bei gleicher Breite).", "Gegen Over-Smoothing in tiefen GNNs.", "> 3 Schichten."),
        level=Level.ADVANCED)
    learn_eps = Param(True, tooltip=T("ε in GIN lernbar.", "Flexibles Gewicht des eigenen Knotens.", "gin."),
                      level=Level.EXPERT)


class GNNConfig(ModelConfig):
    __title__ = "GNN"
    __tooltip__ = T(
        "Graph-Neuronales Netz für Daten aus Knoten und Kanten.",
        "Nutzt Beziehungsstruktur direkt – Nachrichten fließen entlang der Kanten.",
        "Moleküle, soziale Netze, Wissensgraphen, Verkehrs-/Stromnetze, Empfehlungssysteme.",
    )
    input_dim = Param(8, min=1, tooltip=T("Merkmale pro Knoten.", "Eingabebreite.", "Je nach Daten."))
    hidden_dims = Param([64, 64], item_type=int, min=1, tooltip=T(
        "Breiten der GNN-Schichten (Länge = Anzahl Nachrichtenrunden).",
        "Jede Schicht erweitert die 'Sichtweite' um einen Kantenschritt.", "2–4 Schichten (sonst Over-Smoothing)."))
    layer = NestedParam(GNNLayerConfig)
    layer_overrides = Param({}, level=Level.ADVANCED, tooltip=T(
        "Abweichungen einzelner Schichten.", "Z. B. GAT in der ersten, GIN in späteren Schichten.", "Experimente."))
    output_dim = Param(2, min=1, tooltip=T("Ausgabegröße.", "Klassen oder Zielwerte.", "Je nach Aufgabe."))
    readout = Param("mean", choices={
        "none": "Ausgabe pro Knoten", "mean": "Mittel über Knoten je Graph",
        "sum": "Summe je Graph", "max": "Maximum je Graph",
    }, tooltip=T("Verdichtung auf Graph-Ebene.", "Graph- statt Knotenvorhersagen.", "Moleküle: mean/sum."))
    jumping_knowledge = Param("last", choices={
        "last": "nur letzte Schicht", "concat": "alle Schichten konkatenieren", "max": "elementweises Maximum aller Schichten",
    }, tooltip=T("Kombination der Schicht-Repräsentationen.", "Nutzt mehrere Reichweiten gleichzeitig.", "Tiefe GNNs."),
        level=Level.ADVANCED)
    edge_dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Anteil zufällig entfernter Kanten im Training (DropEdge).", "Regularisierung, gegen Over-Smoothing.", "0.1–0.3."),
        level=Level.ADVANCED, group="Regularisierung")

    def check(self) -> list[str]:
        return validate_overrides(self.layer, self.layer_overrides, len(self.hidden_dims))


class GNNLayer(nn.Module):
    def __init__(self, din: int, dout: int, cfg: GNNLayerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.conv == "gat":
            per_head = dout // cfg.heads if cfg.gat_concat else dout
            if cfg.gat_concat and dout % cfg.heads:
                raise ValueError(f"GAT: Breite {dout} nicht durch heads={cfg.heads} teilbar.")
            self.lin = nn.Linear(din, per_head * cfg.heads, bias=False)
            self.att_src = nn.Parameter(torch.randn(cfg.heads, per_head) * 0.1)
            self.att_dst = nn.Parameter(torch.randn(cfg.heads, per_head) * 0.1)
            self.per_head = per_head
            self.bias = nn.Parameter(torch.zeros(dout))
        elif cfg.conv == "sage":
            self.lin_self = nn.Linear(din, dout)
            self.lin_neigh = nn.Linear(din, dout, bias=False)
        elif cfg.conv == "gin":
            self.mlp = nn.Sequential(nn.Linear(din, dout), cfg.activation.build(), nn.Linear(dout, dout))
            self.eps = nn.Parameter(torch.zeros(1), requires_grad=cfg.learn_eps)
        else:
            self.lin = nn.Linear(din, dout)
        self.norm = build_norm(cfg.norm, dout) if cfg.norm is not None else nn.Identity()
        self.act = cfg.activation.build()
        self.drop = nn.Dropout(cfg.dropout)
        self.residual = cfg.residual and din == dout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        c, n = self.cfg, x.shape[0]
        src, dst = edge_index
        if c.conv == "gcn":
            ei = add_self_loops(edge_index, n) if c.self_loops else edge_index
            s, d = ei
            deg = scatter(torch.ones_like(d, dtype=x.dtype), d, n, "sum").clamp_min(1)
            w = deg.pow(-0.5)[s] * deg.pow(-0.5)[d]
            h = self.lin(x)
            h = scatter(h[s] * w.unsqueeze(-1), d, n, "sum")
        elif c.conv == "sage":
            h = self.lin_self(x) + self.lin_neigh(scatter(x[src], dst, n, c.aggregation))
        elif c.conv == "gin":
            h = self.mlp((1 + self.eps) * x + scatter(x[src], dst, n, c.aggregation))
        else:
            ei = add_self_loops(edge_index, n) if c.self_loops else edge_index
            s, d = ei
            hh = self.lin(x).view(n, c.heads, self.per_head)
            e = F.leaky_relu((hh[s] * self.att_src).sum(-1) + (hh[d] * self.att_dst).sum(-1), 0.2)  # [E, H]
            e = e - scatter(e, d, n, "max")[d]  # numerisch stabile Softmax je Zielknoten
            a = torch.exp(e)
            a = a / scatter(a, d, n, "sum")[d].clamp_min(1e-16)
            a = F.dropout(a, c.dropout, self.training)
            h = scatter(hh[s] * a.unsqueeze(-1), d, n, "sum")
            h = h.reshape(n, -1) if c.gat_concat else h.mean(1)
            h = h + self.bias
        h = self.drop(self.act(self.norm(h)))
        return x + h if self.residual else h


class GNN(BaseModel):
    """GNN. Aufruf: ``model(x, edge_index, batch=None)``."""

    def __init__(self, config: GNNConfig):
        super().__init__(config)
        config.validate()
        c = config
        dims = [c.input_dim] + list(c.hidden_dims)
        n = len(c.hidden_dims)
        self.layers = nn.ModuleList([
            GNNLayer(dims[i], dims[i + 1], resolve_layer_config(c.layer, c.layer_overrides, i, n)) for i in range(n)
        ])
        feat = sum(c.hidden_dims) if c.jumping_knowledge == "concat" else c.hidden_dims[-1]
        self.head = nn.Linear(feat, c.output_dim)
        self.finish()

    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        c = self.config
        edge_index = edge_index.long()
        if self.training and c.edge_dropout > 0:
            keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= c.edge_dropout
            edge_index = edge_index[:, keep]
        h, states = x, []
        for layer in self.layers:
            h = layer(h, edge_index)
            states.append(h)
        if c.jumping_knowledge == "concat":
            h = torch.cat(states, dim=-1)
        elif c.jumping_knowledge == "max":
            h = torch.stack(states).amax(0)
        if c.readout != "none":
            if batch is None:
                batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
            if is_exporting():                       # Graphanzahl datenabhängig, aber exportierbar
                n_graphs = batch.max().item() + 1
                torch._check_is_size(n_graphs)
            else:
                n_graphs = int(batch.max()) + 1
            h = scatter(h, batch.long(), n_graphs, c.readout)
        return self.head(h)

    def example_input(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        n = 6 * batch_size
        src = torch.arange(n)
        dst = (src + 1) % n
        return {"x": torch.randn(n, self.config.input_dim),
                "edge_index": torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]),
                "batch": torch.arange(batch_size).repeat_interleave(6)}


MODELS.register("gnn", GNN, config=GNNConfig, aliases=("graph_nn",), tags=("graph",), tooltip=T(
    "Graph-Neuronales Netz (GCN, GraphSAGE, GIN, GAT).", "Lernt auf Beziehungsstrukturen.",
    "Moleküle, Netzwerke, Empfehlungen.",
))
