"""
ki_baukasten.models.rnn
=======================

Rekurrente Netze (RNN, LSTM, GRU, minGRU) – Schicht für Schicht frei kombinierbar.

Besonderheit dieses Baukastens
------------------------------
Statt eines monolithischen ``nn.LSTM(num_layers=4)`` wird jede rekurrente
Schicht **einzeln** gebaut. Dadurch kann jede Schicht

* eine **eigene Zellart** haben (z. B. LSTM → GRU → minGRU),
* eine **eigene Breite** (``hidden_sizes``),
* eigene Richtung, Normalisierung, Residualverbindung und Dropout.

Ausgabe
-------
* ``pooling="none"``: eine Ausgabe pro Zeitschritt ``[B, T, output_dim]``
  (Tagging, Sprachmodell, Sequenz-zu-Sequenz gleicher Länge)
* sonst ein Vektor pro Sequenz ``[B, output_dim]`` (Klassifikation, Prognose)
"""

from __future__ import annotations

import torch
from torch import nn

from ..components.normalization import build_norm
from ..core import (BaseConfig, ComponentParam, Level, NestedParam, Param, T, resolve_layer_config,
                    validate_overrides)
from ..components.mixers import MinGRU  # noqa: F401  (paralleler Scan, auch als Mixer nutzbar)
from .base import MODELS, BaseModel, ModelConfig


class RecurrentLayerConfig(BaseConfig):
    __title__ = "Rekurrente Schicht"
    cell = Param("lstm", choices={
        "lstm": "Long Short-Term Memory – Gates für Gedächtnis (robust, Standard)",
        "gru": "Gated Recurrent Unit – schlanker als LSTM, oft gleich gut",
        "rnn": "einfaches Elman-RNN – vergisst schnell, nur für kurze Abhängigkeiten",
        "mingru": "minGRU – vereinfachte, parallelisierbare GRU (2024)",
    }, tooltip=T("Zellart.", "Bestimmt Gedächtnisfähigkeit und Kosten.", "LSTM/GRU sind sichere Standards."))
    bidirectional = Param(False, tooltip=T(
        "Zusätzlich rückwärts verarbeiten.",
        "Jeder Zeitschritt kennt Vergangenheit UND Zukunft.",
        "Klassifikation, Tagging. Nicht für Generierung/Echtzeit!",
    ))
    nonlinearity = Param("tanh", choices=["tanh", "relu"], tooltip=T(
        "Aktivierung des einfachen RNN.", "relu kann explodieren, tanh sättigt.", "Nur cell='rnn'."), level=Level.EXPERT)
    proj_size = Param(0, min=0, tooltip=T(
        "Projektion des LSTM-Zustands auf kleinere Größe (0 = aus).",
        "Spart Parameter in großen LSTMs (LSTMP).", "Spracherkennung."), level=Level.EXPERT)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout auf die Ausgabe der Schicht.", "Regularisierung zwischen Schichten.", "0.1–0.5."), group="Regularisierung")
    norm = ComponentParam("normalization", None, tooltip=T(
        "Normalisierung nach der Schicht.", "Stabilisiert tiefe RNN-Stapel.", "layernorm bei > 2 Schichten."),
        level=Level.ADVANCED)
    residual = Param(False, tooltip=T(
        "Residualverbindung um die Schicht (bei gleicher Breite).",
        "Bessere Gradienten in tiefen Stapeln (GNMT).", "≥ 4 Schichten."), level=Level.ADVANCED)
    forget_bias = Param(None, type=float, tooltip=T(
        "Startwert des LSTM-Forget-Gate-Bias (None = PyTorch-Standard 0).",
        "1.0 lässt das LSTM anfangs eher erinnern – klassischer Trick für lange Abhängigkeiten.",
        "LSTMs auf langen Sequenzen.",
    ), level=Level.EXPERT)


class RecurrentConfig(ModelConfig):
    __title__ = "Rekurrentes Netz"
    __tooltip__ = T(
        "Rekurrentes Netz (LSTM/GRU/RNN/minGRU) mit frei konfigurierbaren Schichten.",
        "Verarbeitet Sequenzen Schritt für Schritt mit einem Gedächtniszustand – natürlich für Streaming.",
        "Zeitreihen, Sensordaten, Steuerung, kleine Sprachmodelle, Sequenz-Tagging.",
    )
    input_type = Param("continuous", choices={"continuous": "Vektoren pro Schritt", "tokens": "Ganzzahl-IDs (Text)"},
                       tooltip=T("Art der Eingabe.", "Tokens bekommen ein Embedding.", "Zeitreihen: continuous."))
    input_dim = Param(8, min=1, tooltip=T("Features pro Zeitschritt (continuous).", "Eingabebreite.", "Anzahl Sensoren."))
    vocab_size = Param(1000, min=1, tooltip=T("Vokabular (tokens).", "Anzahl Symbole.", "Zeichen: ~100."))
    embedding_dim = Param(64, min=1, tooltip=T("Embedding-Größe (tokens).", "Repräsentation der Symbole.", "32–512."))
    hidden_sizes = Param([128, 128], item_type=int, min=1, tooltip=T(
        "Zustandsgröße jeder Schicht (Länge = Anzahl Schichten).",
        "Gedächtniskapazität pro Schicht.", "1–3 Schichten à 64–512.",
    ))
    layer = NestedParam(RecurrentLayerConfig)
    layer_overrides = Param({}, level=Level.ADVANCED, tooltip=T(
        "Abweichungen einzelner Schichten (z. B. andere Zellart).",
        "Jede Schicht individuell.", "{'0': {'bidirectional': True}, '-1': {'cell': 'gru'}}"))
    pooling = Param("last", choices={
        "none": "Ausgabe pro Zeitschritt",
        "last": "letzter Zeitschritt",
        "mean": "Mittelwert über die Zeit",
        "max": "Maximum über die Zeit",
        "attention": "gelernte Attention-Gewichtung über die Zeit",
    }, tooltip=T("Verdichtung über die Zeit.", "Sequenz → Vektor für sequenzweise Vorhersagen.",
                 "Klassifikation: last/attention; Tagging/LM: none."))
    output_dim = Param(1, min=1, tooltip=T("Ausgabegröße.", "Klassen oder Zielgrößen.", "Je nach Aufgabe."))
    head_hidden_dims = Param([], item_type=int, min=1, tooltip=T(
        "Versteckte Schichten des Ausgabekopfs.", "Mehr Kapazität im Kopf.", "Meist leer."), level=Level.ADVANCED)
    head_activation = ComponentParam("activation", "relu", tooltip=T(
        "Aktivierung im Kopf.", "Nichtlinearität.", "Mit head_hidden_dims."), level=Level.ADVANCED)
    embedding_dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout nach dem Embedding / auf die Eingabe.", "Regularisierung.", "Text: 0.1–0.3."), group="Regularisierung")

    def check(self) -> list[str]:
        problems = validate_overrides(self.layer, self.layer_overrides, len(self.hidden_sizes))
        if not self.hidden_sizes:
            problems.append("hidden_sizes darf nicht leer sein.")
        return problems


class _RecurrentBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, cfg: RecurrentLayerConfig):
        super().__init__()
        if cfg.cell == "mingru":
            self.fwd = MinGRU(in_dim, hidden)
            self.bwd = MinGRU(in_dim, hidden) if cfg.bidirectional else None
            out = hidden * (2 if cfg.bidirectional else 1)
            self.native = False
        else:
            kw: dict = dict(batch_first=True, bidirectional=cfg.bidirectional)
            if cfg.cell == "lstm":
                rnn = nn.LSTM(in_dim, hidden, proj_size=cfg.proj_size, **kw)
                if cfg.forget_bias is not None:
                    with torch.no_grad():
                        for name, p in rnn.named_parameters():
                            if name.startswith("bias_ih"):
                                p[hidden: 2 * hidden].fill_(cfg.forget_bias)  # Gate-Reihenfolge: i, f, g, o
                            elif name.startswith("bias_hh"):
                                p[hidden: 2 * hidden].zero_()
            elif cfg.cell == "gru":
                rnn = nn.GRU(in_dim, hidden, **kw)
            else:
                rnn = nn.RNN(in_dim, hidden, nonlinearity=cfg.nonlinearity, **kw)
            self.rnn = rnn
            unit = cfg.proj_size if (cfg.cell == "lstm" and cfg.proj_size) else hidden
            out = unit * (2 if cfg.bidirectional else 1)
            self.native = True
        self.out_dim = out
        self.norm = build_norm(cfg.norm, out) if cfg.norm is not None else nn.Identity()
        self.drop = nn.Dropout(cfg.dropout)
        self.residual = cfg.residual and in_dim == out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.native:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                h, _ = self.rnn(packed)
                h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
            else:
                h, _ = self.rnn(x)
        else:
            h, _ = self.fwd(x)
            if self.bwd is not None:
                back, _ = self.bwd(x.flip(1))
                h = torch.cat([h, back.flip(1)], dim=-1)
        h = self.drop(self.norm(h))
        return x + h if self.residual else h


class RecurrentNet(BaseModel):
    """Rekurrentes Netz. Eingabe ``[B, T, F]`` (continuous) oder ``[B, T]`` (tokens).

    Optional ``lengths`` (``[B]``) für Sequenzen unterschiedlicher Länge
    (rechts gepaddet) – dann werden Padding-Schritte korrekt ignoriert.
    """

    def __init__(self, config: RecurrentConfig):
        super().__init__(config)
        config.validate()
        c = config
        if c.input_type == "tokens":
            self.embed: nn.Module = nn.Embedding(c.vocab_size, c.embedding_dim)
            dim = c.embedding_dim
        else:
            self.embed = nn.Identity()
            dim = c.input_dim
        self.in_drop = nn.Dropout(c.embedding_dropout)
        n = len(c.hidden_sizes)
        layers = []
        for i, hidden in enumerate(c.hidden_sizes):
            block = _RecurrentBlock(dim, hidden, resolve_layer_config(c.layer, c.layer_overrides, i, n))
            layers.append(block)
            dim = block.out_dim
        self.layers = nn.ModuleList(layers)
        self.attn_pool = nn.Linear(dim, 1) if c.pooling == "attention" else None
        head: list[nn.Module] = []
        dims = [dim] + list(c.head_hidden_dims)
        for i in range(len(c.head_hidden_dims)):
            head += [nn.Linear(dims[i], dims[i + 1]), c.head_activation.build()]
        head.append(nn.Linear(dims[-1], c.output_dim))
        self.head = nn.Sequential(*head)
        self.finish()

    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        h = self.in_drop(self.embed(x))
        for layer in self.layers:
            h = layer(h, lengths)
        p = self.config.pooling
        mask = None
        if lengths is not None:
            mask = torch.arange(h.shape[1], device=h.device).unsqueeze(0) < lengths.to(h.device).unsqueeze(1)
        if p == "none":
            return self.head(h)
        if p == "last":
            if lengths is None:
                pooled = h[:, -1]
            else:
                pooled = h[torch.arange(h.shape[0], device=h.device), (lengths.to(h.device) - 1).clamp_min(0)]
        elif p == "mean":
            pooled = h.mean(1) if mask is None else (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        elif p == "max":
            pooled = (h if mask is None else h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)).amax(1)
        else:
            scores = self.attn_pool(h).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            pooled = (scores.softmax(-1).unsqueeze(-1) * h).sum(1)
        return self.head(pooled)

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        if self.config.input_type == "tokens":
            return torch.randint(0, self.config.vocab_size, (batch_size, 12))
        return torch.randn(batch_size, 12, self.config.input_dim)


MODELS.register("rnn", RecurrentNet, config=RecurrentConfig, aliases=("lstm", "gru", "recurrent"),
                tags=("zeitreihen", "nlp"), tooltip=T(
    "Rekurrentes Netz (LSTM/GRU/RNN/minGRU), jede Schicht einzeln konfigurierbar.",
    "Natürliche Verarbeitung von Strömen mit Gedächtnis, geringer Speicherbedarf bei Inferenz.",
    "Zeitreihen, Sensorik, Steuerung, kleine Text-Modelle.",
))
