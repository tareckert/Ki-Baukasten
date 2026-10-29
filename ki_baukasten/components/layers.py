"""
ki_baukasten.components.layers
==============================

Universelle Schichten, Zusammenführungen und Tensor-Funktionen – das Material
für **frei definierte Netze** (``models.graph``) und für zusammengesetzte
Modelle.

Drei Registries
---------------
``layer``     – einzelne Schichten. Fabrik: ``factory(args, in_shape)``
                ``in_shape`` ist die Eingabeform **ohne** Batch-Dimension,
                z. B. ``(3, 32, 32)`` für Bilder oder ``(128, 64)`` für
                Sequenzen. Dadurch muss man Eingangsgrößen nie von Hand
                angeben – sie werden beim Aufbau automatisch ermittelt.
``merge``     – Zusammenführung mehrerer Eingänge (concat, add, gated ...).
                Fabrik: ``factory(args, in_shapes)``
``function``  – parameterfreie Tensor-Funktionen (softmax, transpose, mean ...),
                als Schicht nutzbar über ``layer: function``.
                Fabrik: ``factory(args) -> callable(tensor) -> tensor``

Weitere Schichten (``mlp``, ``transformer_block``, ``ssm_block`` ...) werden
von den jeweiligen Modell-Modulen in die ``layer``-Registry eingetragen.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, ComponentParam, Level, NestedParam, Param, T, create_registry
from .attention import AttentionConfig, MultiHeadAttention
from .normalization import build_norm

LAYERS = create_registry(
    "layer",
    "Schichten",
    T(
        "Einzelne Netzschichten für frei definierte Architekturen.",
        "Aus ihnen lässt sich jede beliebige Netzstruktur als Graph zusammenstecken.",
        "Modell 'graph' (freies Netz), eigene Experimente.",
    ),
    context_doc="factory(args, in_shape) – in_shape ohne Batch-Dimension",
)
MERGES = create_registry(
    "merge",
    "Zusammenführungen",
    T(
        "Verfahren, um mehrere Tensoren zu einem zu vereinen.",
        "Ermöglichen Verzweigungen, Residualverbindungen und Multi-Input-Netze.",
        "Graph-Modelle, Fusion mehrerer Modalitäten.",
    ),
    context_doc="factory(args, in_shapes) – Liste von Formen ohne Batch-Dimension",
)
FUNCTIONS = create_registry(
    "function",
    "Tensor-Funktionen",
    T(
        "Parameterfreie Tensor-Operationen.",
        "Kleine Umformungen ohne eigene Gewichte (z. B. Softmax, Mittelwert, Transponieren).",
        "Graph-Modelle, Nachbearbeitung von Ausgaben.",
    ),
    context_doc="factory(args) -> callable(x)",
)

_DIMS_CHOICES = {1: "1D (Signale, Sequenzen) [B, C, L]", 2: "2D (Bilder) [B, C, H, W]", 3: "3D (Volumen, Video) [B, C, D, H, W]"}


def _conv_cls(dims: int, transpose: bool = False) -> type[nn.Module]:
    if transpose:
        return {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[dims]
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]


# ===========================================================================
# Hilfsmodule
# ===========================================================================

class FunctionModule(nn.Module):
    """Wickelt eine Tensor-Funktion in ein ``nn.Module``."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], name: str):
        super().__init__()
        self.fn = fn
        self.fn_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)

    def extra_repr(self) -> str:
        return self.fn_name


class Reshape(nn.Module):
    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class Permute(nn.Module):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        self.dims = (0,) + tuple(d + 1 for d in dims)

    def forward(self, x):
        return x.permute(*self.dims)


class GlobalPool(nn.Module):
    """Globales Pooling über alle Raum-/Zeitachsen ab Achse 2 (Kanal-zuerst)."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        flat = x.flatten(2)
        if self.kind == "avg":
            return flat.mean(-1)
        if self.kind == "max":
            return flat.amax(-1)
        return torch.cat([flat.mean(-1), flat.amax(-1)], dim=1)  # avgmax


class DropPath(nn.Module):
    """Stochastic Depth: ganzen Pfad pro Beispiel zufällig weglassen."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class GaussianNoise(nn.Module):
    """Addiert im Training Gauß-Rauschen (Regularisierung / Robustheit)."""

    def __init__(self, std: float, relative: bool):
        super().__init__()
        self.std, self.relative = std, relative

    def forward(self, x):
        if not self.training or self.std == 0:
            return x
        scale = self.std * (x.detach().std() if self.relative else 1.0)
        return x + torch.randn_like(x) * scale


class LearnableScale(nn.Module):
    """Multipliziert mit einem lernbaren Skalar oder Vektor (z. B. LayerScale)."""

    def __init__(self, shape: tuple[int, ...], init: float, channel_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.full(shape, float(init)))
        self.channel_dim = channel_dim

    def forward(self, x):
        w = self.weight
        if self.channel_dim == 1 and x.dim() > 2 and w.numel() > 1:
            w = w.view(1, -1, *([1] * (x.dim() - 2)))
        return x * w


class SqueezeExcitation(nn.Module):
    """Kanal-Attention: globale Info → Gewichte pro Kanal (Hu et al. 2018)."""

    def __init__(self, channels: int, reduction: float, activation: nn.Module, gate: nn.Module):
        super().__init__()
        hidden = max(1, int(channels / reduction))
        self.fc1 = nn.Linear(channels, hidden)
        self.act = activation
        self.fc2 = nn.Linear(hidden, channels)
        self.gate = gate

    def forward(self, x):
        s = x.flatten(2).mean(-1)
        w = self.gate(self.fc2(self.act(self.fc1(s))))
        return x * w.view(*w.shape, *([1] * (x.dim() - 2)))


class RecurrentLayer(nn.Module):
    """LSTM/GRU/RNN, das nur Tensoren zurückgibt (Sequenz oder letzter Zustand)."""

    def __init__(self, cell: str, input_size: int, a: "RNNLayerConfig"):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell]
        self.rnn = cls(input_size, a.hidden_size, num_layers=a.num_layers, batch_first=True,
                       bidirectional=a.bidirectional, dropout=a.dropout if a.num_layers > 1 else 0.0)
        self.output = a.output

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.output == "last":
            return out[:, -1]
        if self.output == "mean":
            return out.mean(1)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, cfg: AttentionConfig, causal: bool):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, cfg)
        self.causal = causal

    def forward(self, x):
        return self.attn(x, causal=self.causal)


# ===========================================================================
# Konfigurationen der Schichten
# ===========================================================================

class LinearLayerConfig(BaseConfig):
    out_features = Param(64, min=1, tooltip=T(
        "Anzahl Ausgabe-Features.", "Breite der Schicht.", "Jede lineare Schicht."))
    bias = Param(True, tooltip=T("Bias-Vektor addieren.", "Verschiebt die Ausgabe; fast immer sinnvoll.", "Standard."))


class ConvLayerConfig(BaseConfig):
    dims = Param(2, choices=_DIMS_CHOICES, tooltip=T(
        "Räumliche Dimensionalität der Faltung.", "1D für Signale, 2D für Bilder, 3D für Volumen/Video.", "Datenart."))
    out_channels = Param(32, min=1, tooltip=T(
        "Anzahl Filter (Ausgabekanäle).", "Mehr Filter = mehr erkannte Muster, mehr Rechenaufwand.", "Jede Faltung."))
    kernel_size = Param(3, min=1, tooltip=T(
        "Größe des Filterfensters.", "Größer = größeres rezeptives Feld pro Schicht.", "3 ist Standard; 7 im Stem."))
    stride = Param(1, min=1, tooltip=T(
        "Schrittweite des Filters.", "2 halbiert die Auflösung (Downsampling ohne Pooling).", "Downsampling."))
    padding = Param("same", type=(int, str), tooltip=T(
        "Randauffüllung: Zahl, 'same' (Größe behalten) oder 'valid' (keine).",
        "Steuert, ob die Ausgabe schrumpft.",
        "'same' bei stride 1; bei stride > 1 wird 'same' automatisch zu kernel//2.",
    ))
    dilation = Param(1, min=1, tooltip=T(
        "Abstand zwischen Filterelementen (atrous convolution).",
        "Vergrößert das rezeptive Feld ohne zusätzliche Parameter.",
        "Segmentierung, Audio (WaveNet).",
    ), level=Level.ADVANCED)
    groups = Param(1, min=1, tooltip=T(
        "Gruppierte Faltung (Kanäle in Gruppen getrennt verarbeitet).",
        "Spart Parameter; groups = Kanäle → depthwise.",
        "ResNeXt, MobileNet.",
        "Ein- und Ausgabekanäle müssen durch groups teilbar sein.",
    ), level=Level.ADVANCED)
    bias = Param(True, tooltip=T("Bias addieren.", "Bei nachfolgender BatchNorm überflüssig.", "Aus vor BatchNorm."))
    padding_mode = Param("zeros", choices={
        "zeros": "mit Nullen auffüllen", "reflect": "spiegeln (ohne Randwert)",
        "replicate": "Randwert wiederholen", "circular": "periodisch umlaufend",
    }, tooltip=T("Art der Randauffüllung.", "Beeinflusst Artefakte an Bildrändern.",
                 "reflect bei Bildgenerierung, circular bei periodischen Daten."), level=Level.EXPERT)


def _padding(a: Any) -> Any:
    if a.padding == "same" and a.stride > 1:
        return (a.kernel_size // 2) * a.dilation
    return a.padding


class ConvTransposeLayerConfig(BaseConfig):
    dims = Param(2, choices=_DIMS_CHOICES, tooltip=T("Dimensionalität.", "Wie bei Faltung.", "Datenart."))
    out_channels = Param(32, min=1, tooltip=T("Ausgabekanäle.", "Anzahl Filter.", "Decoder/Generatoren."))
    kernel_size = Param(4, min=1, tooltip=T("Filtergröße.", "4 mit stride 2 verdoppelt sauber die Auflösung.", "Generatoren."))
    stride = Param(2, min=1, tooltip=T("Upsampling-Faktor.", "2 verdoppelt die Auflösung.", "Upsampling."))
    padding = Param(1, min=0, tooltip=T("Randauffüllung.", "Bestimmt die exakte Ausgabegröße.", "Mit k=4,s=2: 1."))
    output_padding = Param(0, min=0, tooltip=T("Zusätzliche Ausgabegröße.", "Behebt Mehrdeutigkeiten.", "Selten."),
                           level=Level.EXPERT)
    bias = Param(True, tooltip=T("Bias addieren.", "Wie bei Faltung.", "Standard."))


class DepthwiseSeparableConfig(BaseConfig):
    dims = Param(2, choices=_DIMS_CHOICES, tooltip=T("Dimensionalität.", "Wie bei Faltung.", "Datenart."))
    out_channels = Param(32, min=1, tooltip=T("Ausgabekanäle.", "Anzahl Filter der Punkt-Faltung.", "Effiziente CNNs."))
    kernel_size = Param(3, min=1, tooltip=T("Filtergröße der Tiefen-Faltung.", "Rezeptives Feld.", "3 oder 5."))
    stride = Param(1, min=1, tooltip=T("Schrittweite.", "Downsampling.", "1 oder 2."))
    bias = Param(False, tooltip=T("Bias.", "Vor Normalisierung überflüssig.", "Meist aus."))


class PoolConfig(BaseConfig):
    dims = Param(2, choices=_DIMS_CHOICES, tooltip=T("Dimensionalität.", "Wie bei Faltung.", "Datenart."))
    kind = Param("max", choices={"max": "Maximum (betont starke Merkmale)", "avg": "Mittelwert (glättet)",
                                 "lp": "Lp-Norm (zwischen avg und max)"}, tooltip=T(
        "Pooling-Art.", "Max für Detektion markanter Merkmale, Avg für glatte Zusammenfassung.", "Downsampling."))
    kernel_size = Param(2, min=1, tooltip=T("Fenstergröße.", "Größe des zusammengefassten Bereichs.", "2 oder 3."))
    stride = Param(None, type=int, min=1, tooltip=T("Schrittweite (None = kernel_size).", "Downsampling-Faktor.", "Meist None."))
    padding = Param(0, min=0, tooltip=T("Randauffüllung.", "Für überlappendes Pooling.", "Selten."), level=Level.ADVANCED)
    norm_type = Param(2.0, min=1.0, tooltip=T("p der Lp-Norm (nur kind=lp).", "p→∞ entspricht Max-Pooling.", "Selten."),
                      level=Level.EXPERT)


class AdaptivePoolConfig(BaseConfig):
    dims = Param(2, choices=_DIMS_CHOICES, tooltip=T("Dimensionalität.", "Wie bei Faltung.", "Datenart."))
    kind = Param("avg", choices=["avg", "max"], tooltip=T("Pooling-Art.", "Siehe Pool.", "Köpfe von CNNs."))
    output_size = Param(1, min=1, tooltip=T(
        "Zielgröße pro Raumachse.", "Macht die Ausgabe unabhängig von der Eingabegröße.", "1 = globales Pooling."))


class GlobalPoolConfig(BaseConfig):
    kind = Param("avg", choices={"avg": "Mittelwert", "max": "Maximum", "avgmax": "beides konkateniert (2·C)"}, tooltip=T(
        "Art des globalen Poolings (über alle Raum-/Zeitachsen).", "Verdichtet Feature-Maps zu einem Vektor.",
        "Übergang Faltungsteil → Klassifikationskopf."))


class FlattenConfig(BaseConfig):
    start_dim = Param(1, min=1, tooltip=T("Ab dieser Achse abflachen (1 = alles außer Batch).",
                                          "Macht aus Feature-Maps einen Vektor.", "Vor linearen Schichten."))


class ReshapeConfig(BaseConfig):
    shape = Param([-1], item_type=int, tooltip=T(
        "Neue Form ohne Batch-Achse (eine Achse darf -1 sein).", "Formt Tensoren um.", "Graph-Modelle."))


class PermuteConfig(BaseConfig):
    dims = Param([1, 0], item_type=int, min=0, tooltip=T(
        "Neue Achsenreihenfolge ohne Batch-Achse (0 = erste Nicht-Batch-Achse).",
        "Vertauscht Achsen, z. B. [T, C] → [C, T].", "Wechsel zwischen Sequenz- und Faltungslayout."))


class DropoutConfig(BaseConfig):
    p = Param(0.1, min=0.0, max=1.0, tooltip=T(
        "Wahrscheinlichkeit, ein Element (bzw. Kanal/Pfad) auf 0 zu setzen.",
        "Klassische Regularisierung gegen Überanpassung.",
        "Kleine Datensätze, große Modelle.",
    ))
    kind = Param("standard", choices={
        "standard": "einzelne Elemente",
        "channel": "ganze Kanäle (Dropout2d/3d; für Faltungs-Feature-Maps)",
        "alpha": "AlphaDropout (erhält Selbstnormalisierung von SELU-Netzen)",
        "path": "Stochastic Depth: ganzer Pfad pro Beispiel",
    }, tooltip=T("Dropout-Variante.", "Bestimmt, was weggelassen wird.", "Je nach Datenart."))


class NormLayerConfig(BaseConfig):
    norm = ComponentParam("normalization", "layernorm", tooltip=T(
        "Normalisierungsverfahren.", "Siehe Registry 'normalization'.", "Überall."))
    layout = Param("feature", type=(int, str), choices={
        "feature": "Features in der letzten Achse ([B, C] oder [B, T, C])",
        1: "Kanäle zuerst, 1D", 2: "Kanäle zuerst, 2D", 3: "Kanäle zuerst, 3D",
    }, tooltip=T("Datenlayout.", "Bestimmt, welche Achse normalisiert wird.", "Sequenzen: feature; CNN: 2."))


class ActivationLayerConfig(BaseConfig):
    activation = ComponentParam("activation", "relu", tooltip=T(
        "Aktivierungsfunktion.", "Siehe Registry 'activation'.", "Überall."))


class EmbeddingLayerConfig(BaseConfig):
    num_embeddings = Param(1000, min=1, tooltip=T(
        "Größe des Vokabulars / Anzahl Kategorien.", "Für jede ID wird ein Vektor gelernt.", "Text, Kategorien."))
    embedding_dim = Param(64, min=1, tooltip=T("Länge jedes Vektors.", "Kapazität der Repräsentation.", "Text: 64–4096."))
    padding_idx = Param(None, type=int, tooltip=T(
        "ID, deren Vektor 0 bleibt und nicht trainiert wird.", "Für Padding-Token.", "Text mit Padding."),
        level=Level.ADVANCED)
    max_norm = Param(None, type=float, min=0.0, tooltip=T(
        "Vektoren mit größerer Norm werden beim Nachschlagen gekürzt.", "Begrenzt Embedding-Größen.", "Selten."),
        level=Level.EXPERT)


class RNNLayerConfig(BaseConfig):
    hidden_size = Param(64, min=1, tooltip=T("Größe des versteckten Zustands.", "Gedächtniskapazität.", "64–1024."))
    num_layers = Param(1, min=1, tooltip=T("Gestapelte Schichten.", "Mehr Tiefe = abstraktere Muster.", "1–4."))
    bidirectional = Param(False, tooltip=T(
        "Zusätzlich rückwärts lesen.", "Kontext von beiden Seiten; nicht für Generierung.", "Klassifikation/Tagging."))
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T("Dropout zwischen Schichten.", "Regularisierung.", "Bei num_layers > 1."))
    output = Param("sequence", choices={"sequence": "alle Zeitschritte [B, T, H]", "last": "letzter Schritt [B, H]",
                                        "mean": "Mittelwert über Zeit [B, H]"}, tooltip=T(
        "Was ausgegeben wird.", "Sequenz für Weiterverarbeitung, last/mean für Klassifikation.", "Je nach Aufgabe."))


class AttentionLayerConfig(BaseConfig):
    attention = NestedParam(AttentionConfig)
    causal = Param(False, tooltip=T("Kausale Maske.", "Nur Vergangenheit sichtbar.", "Generierung."))


class SELayerConfig(BaseConfig):
    reduction = Param(4.0, min=1.0, tooltip=T(
        "Verkleinerungsfaktor der inneren Schicht.", "Kleiner = mehr Kapazität, mehr Parameter.", "4–16."))
    activation = ComponentParam("activation", "relu", tooltip=T("Innere Aktivierung.", "Nichtlinearität.", "relu/silu."))
    gate = ComponentParam("activation", "sigmoid", tooltip=T("Gate-Aktivierung.", "Erzeugt Kanalgewichte in (0,1).",
                                                             "sigmoid oder hardsigmoid."), level=Level.ADVANCED)


class UpsampleConfig(BaseConfig):
    scale_factor = Param(2.0, min=0.0, tooltip=T("Vergrößerungsfaktor.", "Erhöht die Auflösung.", "Decoder, U-Nets."))
    mode = Param("nearest", choices=["nearest", "linear", "bilinear", "bicubic", "trilinear"], tooltip=T(
        "Interpolationsverfahren.", "nearest = blockig/schnell; bilinear = glatt.", "Bilder: bilinear oder nearest."))


class PixelShuffleConfig(BaseConfig):
    upscale_factor = Param(2, min=1, tooltip=T(
        "Faktor (Kanäle → Auflösung).", "Artefaktarmes Upsampling (Sub-Pixel-Faltung).", "Super-Resolution."))


class NoiseConfig(BaseConfig):
    std = Param(0.1, min=0.0, tooltip=T("Standardabweichung des Rauschens.", "Stärke der Störung.", "Robustheit."))
    relative = Param(False, tooltip=T("Relativ zur Streuung der Eingabe.", "Skaleninvariant.", "Variable Skalen."),
                     level=Level.ADVANCED)


class ScaleConfig(BaseConfig):
    init = Param(1.0, tooltip=T("Startwert.", "LayerScale nutzt kleine Werte (1e-5..1e-1).", "Tiefe Netze."))
    per_channel = Param(True, tooltip=T("Ein Wert pro Kanal statt ein Skalar.", "Feinere Kontrolle.", "LayerScale."))
    layout = Param("feature", type=(int, str), choices=["feature", 1, 2, 3], tooltip=T(
        "Datenlayout (welche Achse 'Kanal' ist).", "Siehe Normalisierung.", "CNN: 2."))


class FunctionLayerConfig(BaseConfig):
    function = ComponentParam("function", "identity", tooltip=T(
        "Anzuwendende Tensor-Funktion.", "Parameterfreie Umformung.", "Graph-Modelle."))


class PositionalLayerConfig(BaseConfig):
    positional = ComponentParam("positional", "sinusoidal", tooltip=T(
        "Positionskodierung, die auf [B, T, D] addiert wird.", "Gibt Sequenzen Positionsinfo.", "Graph-Transformer."))
    max_seq_len = Param(2048, min=1, tooltip=T("Maximale Länge.", "Tabellengröße.", "Länge der Daten."))


# ===========================================================================
# Registrierung: Schichten
# ===========================================================================

R = LAYERS.register


@R("linear", config=LinearLayerConfig, aliases=("dense", "fc"), tooltip=T(
    "Voll verbundene Schicht y = xWᵀ + b (wirkt auf die letzte Achse).",
    "Grundbaustein fast jedes Netzes.", "Überall."))
def _linear(a, in_shape):
    return nn.Linear(in_shape[-1], a.out_features, bias=a.bias)


@R("conv", config=ConvLayerConfig, aliases=("convolution",), tooltip=T(
    "Faltung (1D/2D/3D, Kanäle in Achse 1).",
    "Lokale, verschiebungsinvariante Mustererkennung mit geteilten Gewichten.", "Bilder, Audio, Sensorsignale."))
def _conv(a, in_shape):
    return _conv_cls(a.dims)(in_shape[0], a.out_channels, a.kernel_size, stride=a.stride, padding=_padding(a),
                             dilation=a.dilation, groups=a.groups, bias=a.bias, padding_mode=a.padding_mode)


@R("conv_transpose", config=ConvTransposeLayerConfig, aliases=("deconv",), tooltip=T(
    "Transponierte Faltung (lernbares Upsampling).", "Vergrößert Feature-Maps.", "Decoder, GAN-Generatoren."))
def _convt(a, in_shape):
    return _conv_cls(a.dims, True)(in_shape[0], a.out_channels, a.kernel_size, stride=a.stride, padding=a.padding,
                                   output_padding=a.output_padding, bias=a.bias)


@R("depthwise_separable_conv", config=DepthwiseSeparableConfig, aliases=("separable_conv",), level=Level.ADVANCED,
   tooltip=T("Tiefen- + Punktfaltung.", "~8–9× weniger Rechenaufwand als normale 3×3-Faltung.", "MobileNet, Edge."))
def _dwsep(a, in_shape):
    cls = _conv_cls(a.dims)
    c = in_shape[0]
    return nn.Sequential(cls(c, c, a.kernel_size, stride=a.stride, padding=a.kernel_size // 2, groups=c, bias=a.bias),
                         cls(c, a.out_channels, 1, bias=a.bias))


@R("pool", config=PoolConfig, tooltip=T(
    "Lokales Pooling (max/avg/lp).", "Reduziert Auflösung, macht robuster gegen kleine Verschiebungen.", "CNNs."))
def _pool(a, in_shape):
    if a.kind == "lp":
        cls = {1: nn.LPPool1d, 2: nn.LPPool2d, 3: nn.LPPool3d}[a.dims]
        return cls(a.norm_type, a.kernel_size, stride=a.stride)
    table = {"max": (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d), "avg": (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)}
    return table[a.kind][a.dims - 1](a.kernel_size, stride=a.stride, padding=a.padding)


@R("adaptive_pool", config=AdaptivePoolConfig, tooltip=T(
    "Pooling auf feste Zielgröße.", "Eingabegrößen-unabhängige Köpfe.", "CNN-Köpfe."))
def _apool(a, in_shape):
    table = {"max": (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d),
             "avg": (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)}
    return table[a.kind][a.dims - 1](a.output_size)


R("global_pool", lambda a, in_shape: GlobalPool(a.kind), config=GlobalPoolConfig, tooltip=T(
    "Globales Pooling zu [B, C].", "Ein Vektor pro Beispiel.", "Vor dem Klassifikationskopf."))
R("flatten", lambda a, in_shape: nn.Flatten(a.start_dim), config=FlattenConfig, tooltip=T(
    "Abflachen zu einem Vektor.", "Übergang zu linearen Schichten.", "CNN → MLP."))
R("reshape", lambda a, in_shape: Reshape(a.shape), config=ReshapeConfig, tooltip=T(
    "Umformen.", "Beliebige neue Form.", "Graph-Modelle."))
R("permute", lambda a, in_shape: Permute(a.dims), config=PermuteConfig, tooltip=T(
    "Achsen vertauschen.", "Layout-Wechsel.", "Sequenz ↔ Faltung."))


@R("dropout", config=DropoutConfig, tooltip=T(
    "Dropout-Regularisierung (verschiedene Varianten).", "Verhindert Co-Adaption von Neuronen.", "Überall."))
def _dropout(a, in_shape):
    if a.kind == "channel":
        return nn.Dropout3d(a.p) if len(in_shape) == 4 else nn.Dropout2d(a.p) if len(in_shape) == 3 else nn.Dropout1d(a.p)
    if a.kind == "alpha":
        return nn.AlphaDropout(a.p)
    if a.kind == "path":
        return DropPath(a.p)
    return nn.Dropout(a.p)


@R("normalization", config=NormLayerConfig, aliases=("norm",), tooltip=T(
    "Normalisierungsschicht.", "Stabilisiert Aktivierungen.", "Überall."))
def _norm(a, in_shape):
    feats = in_shape[-1] if a.layout == "feature" else in_shape[0]
    return build_norm(a.norm, feats, a.layout)


R("activation", lambda a, in_shape: a.activation.build(), config=ActivationLayerConfig, aliases=("act",), tooltip=T(
    "Aktivierungsfunktion als eigene Schicht.", "Nichtlinearität.", "Überall."))


@R("embedding", config=EmbeddingLayerConfig, tooltip=T(
    "Nachschlagetabelle ID → Vektor (Eingabe: Ganzzahlen).", "Wandelt Tokens/Kategorien in Vektoren.", "Text, Kategorien."))
def _embedding(a, in_shape):
    return nn.Embedding(a.num_embeddings, a.embedding_dim, padding_idx=a.padding_idx, max_norm=a.max_norm)


for _cell in ("lstm", "gru", "rnn"):
    R(_cell, (lambda c: (lambda a, in_shape: RecurrentLayer(c, in_shape[-1], a)))(_cell), config=RNNLayerConfig,
      tooltip=T(f"Rekurrente Schicht ({_cell.upper()}) auf [B, T, F].",
                "Verarbeitet Sequenzen Schritt für Schritt mit Gedächtnis.", "Zeitreihen, Text, Steuerung."))


R("self_attention", lambda a, in_shape: SelfAttentionLayer(in_shape[-1], a.attention, a.causal),
  config=AttentionLayerConfig, aliases=("attention",), tooltip=T(
      "Self-Attention auf [B, T, D].", "Globale Interaktion aller Positionen.", "Graph-Transformer, Hybride."))


@R("squeeze_excitation", config=SELayerConfig, aliases=("se",), level=Level.ADVANCED, tooltip=T(
    "Squeeze-and-Excitation (Kanal-Attention).", "Gewichtet Kanäle nach globalem Kontext.", "EfficientNet, ResNet-SE."))
def _se(a, in_shape):
    return SqueezeExcitation(in_shape[0], a.reduction, a.activation.build(), a.gate.build())


R("upsample", lambda a, in_shape: nn.Upsample(scale_factor=a.scale_factor, mode=a.mode), config=UpsampleConfig,
  tooltip=T("Interpolations-Upsampling.", "Parameterfreie Vergrößerung.", "Decoder."))
R("pixel_shuffle", lambda a, in_shape: nn.PixelShuffle(a.upscale_factor), config=PixelShuffleConfig,
  level=Level.ADVANCED, tooltip=T("Sub-Pixel-Upsampling.", "Kanäle → Auflösung.", "Super-Resolution."))
R("gaussian_noise", lambda a, in_shape: GaussianNoise(a.std, a.relative), config=NoiseConfig, aliases=("noise",),
  level=Level.ADVANCED, tooltip=T("Rauschen im Training.", "Regularisierung, Robustheit.", "Kleine Daten, Sim2Real."))


@R("scale", config=ScaleConfig, aliases=("layer_scale",), level=Level.ADVANCED, tooltip=T(
    "Lernbare Skalierung (Skalar oder pro Kanal).", "Z. B. LayerScale für tiefe Transformer.", "CaiT, ConvNeXt."))
def _scale(a, in_shape):
    if not a.per_channel:
        return LearnableScale((1,), a.init, -1)
    ch = in_shape[-1] if a.layout == "feature" else in_shape[0]
    return LearnableScale((ch,), a.init, -1 if a.layout == "feature" else 1)


R("function", lambda a, in_shape: FunctionModule(a.function.build(), a.function.name), config=FunctionLayerConfig,
  aliases=("fn",), tooltip=T("Parameterfreie Tensor-Funktion als Schicht.", "Kleine Umformungen.", "Graph-Modelle."))


class _AddPositional(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        return self.enc.embed(x, torch.arange(x.shape[1], device=x.device))


R("positional_encoding", lambda a, in_shape: _AddPositional(a.positional.build(in_shape[-1], in_shape[-1], 1, a.max_seq_len)),
  config=PositionalLayerConfig, level=Level.ADVANCED, tooltip=T(
      "Addiert eine absolute Positionskodierung.", "Positionsinfo für Sequenzen.", "Graph-Transformer."))
R("identity", lambda a, in_shape: nn.Identity(), aliases=("none",), tooltip=T(
    "Reicht die Eingabe unverändert durch.", "Platzhalter, Verzweigungspunkte.", "Graph-Modelle."))


# ===========================================================================
# Registrierung: Zusammenführungen
# ===========================================================================

class ConcatConfig(BaseConfig):
    dim = Param(-1, tooltip=T("Achse, entlang der verkettet wird (-1 = Features, 1 = Kanäle bei CNNs).",
                              "Bestimmt, wie Informationen nebeneinander gelegt werden.", "Fusion, DenseNet, U-Net."))


class WeightedSumConfig(BaseConfig):
    softmax = Param(True, tooltip=T("Gewichte per Softmax normieren.", "Gewichte summieren zu 1.", "Standard."))


class _Merge(nn.Module):
    def __init__(self, fn: Callable[[list[torch.Tensor]], torch.Tensor], name: str):
        super().__init__()
        self.fn, self.merge_name = fn, name

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.fn(list(xs))

    def extra_repr(self) -> str:
        return self.merge_name


class WeightedSum(nn.Module):
    def __init__(self, n: int, softmax: bool):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n))
        self.softmax = softmax

    def forward(self, *xs):
        w = self.logits.softmax(0) if self.softmax else self.logits + 1.0 / len(xs)
        return sum(wi * x for wi, x in zip(w, xs))


class GatedMerge(nn.Module):
    """Lernbares, elementweises Gate zwischen zwei Eingängen: g·a + (1-g)·b."""

    def __init__(self, features: int):
        super().__init__()
        self.gate = nn.Linear(2 * features, features)

    def forward(self, a, b):
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return g * a + (1 - g) * b


M = MERGES.register
M("concat", lambda a, shapes: _Merge(lambda xs: torch.cat(xs, dim=a.dim), "concat"), config=ConcatConfig,
  aliases=("cat",), tooltip=T("Verketten entlang einer Achse.", "Behält alle Informationen getrennt bei.", "Fusion."))
M("add", lambda a, shapes: _Merge(lambda xs: torch.stack(xs).sum(0), "add"), aliases=("sum", "residual"), tooltip=T(
    "Elementweise Summe.", "Residualverbindungen; Formen müssen gleich sein.", "ResNet-artige Skips."))
M("mul", lambda a, shapes: _Merge(lambda xs: torch.stack(xs).prod(0), "mul"), aliases=("product",), tooltip=T(
    "Elementweises Produkt.", "Multiplikative Interaktion / Gating.", "Attention-artige Gates."))
M("mean", lambda a, shapes: _Merge(lambda xs: torch.stack(xs).mean(0), "mean"), aliases=("average",), tooltip=T(
    "Elementweiser Mittelwert.", "Skalenneutral bei vielen Eingängen.", "Ensembles."))
M("max", lambda a, shapes: _Merge(lambda xs: torch.stack(xs).amax(0), "max"), tooltip=T(
    "Elementweises Maximum.", "Stärkstes Signal gewinnt.", "Robuste Fusion."))
M("stack", lambda a, shapes: _Merge(lambda xs: torch.stack(xs, dim=a.dim), "stack"), config=ConcatConfig, tooltip=T(
    "Stapeln entlang neuer Achse.", "Behält Eingänge als eigene Achse.", "Spätere Attention über Quellen."))
M("weighted_sum", lambda a, shapes: WeightedSum(len(shapes), a.softmax), config=WeightedSumConfig,
  level=Level.ADVANCED, tooltip=T("Lernbar gewichtete Summe.", "Netz lernt die Wichtigkeit der Eingänge.", "Fusion."))
M("gated", lambda a, shapes: GatedMerge(shapes[0][-1]), level=Level.ADVANCED, tooltip=T(
    "Lernbares Gate zwischen genau zwei Eingängen.", "Elementweise weiche Auswahl.", "Highway-artige Verbindungen."))
M("first", lambda a, shapes: _Merge(lambda xs: xs[0], "first"), level=Level.EXPERT, tooltip=T(
    "Nur den ersten Eingang verwenden.", "Für Ablationen (Zweig abschalten ohne Umbau).", "Experimente."))


# ===========================================================================
# Registrierung: Tensor-Funktionen
# ===========================================================================

class DimConfig(BaseConfig):
    dim = Param(-1, tooltip=T("Achse der Operation.", "Bestimmt, worüber gerechnet wird.", "-1 = letzte Achse."))


class TransposeConfig(BaseConfig):
    dim0 = Param(1, tooltip=T("Erste Achse.", "Wird mit dim1 getauscht.", "1 und 2 bei [B, T, C]."))
    dim1 = Param(2, tooltip=T("Zweite Achse.", "Wird mit dim0 getauscht.", "1 und 2 bei [B, T, C]."))


class ClampConfig(BaseConfig):
    min = Param(None, type=float, tooltip=T("Untergrenze.", "Kleinere Werte werden angehoben.", "Begrenzung."))
    max = Param(None, type=float, tooltip=T("Obergrenze.", "Größere Werte werden gekappt.", "Begrenzung."))


class ScalarConfig(BaseConfig):
    value = Param(1.0, tooltip=T("Faktor.", "Multipliziert alle Elemente.", "Skalierung."))


Fn = FUNCTIONS.register
Fn("identity", lambda a: (lambda x: x), tooltip=T("Unverändert.", "Platzhalter.", "Überall."))
Fn("softmax", lambda a: (lambda x: x.softmax(a.dim)), config=DimConfig, tooltip=T(
    "Softmax entlang einer Achse.", "Wahrscheinlichkeitsverteilung.", "Ausgaben, Gewichte."))
Fn("log_softmax", lambda a: (lambda x: x.log_softmax(a.dim)), config=DimConfig, tooltip=T(
    "Log-Softmax.", "Stabile Log-Wahrscheinlichkeiten.", "NLL-Verluste."))
Fn("l2_normalize", lambda a: (lambda x: F.normalize(x, dim=a.dim)), config=DimConfig, aliases=("normalize",), tooltip=T(
    "Auf Länge 1 normieren.", "Für Kosinus-Ähnlichkeiten.", "Embeddings, kontrastives Lernen."))
Fn("mean", lambda a: (lambda x: x.mean(a.dim)), config=DimConfig, tooltip=T(
    "Mittelwert über eine Achse (entfernt sie).", "Pooling über Zeit/Tokens.", "Sequenz → Vektor."))
Fn("sum", lambda a: (lambda x: x.sum(a.dim)), config=DimConfig, tooltip=T("Summe über eine Achse.", "Pooling.", "Mengen."))
Fn("max", lambda a: (lambda x: x.amax(a.dim)), config=DimConfig, tooltip=T("Maximum über eine Achse.", "Max-Pooling.", "Mengen."))
Fn("last", lambda a: (lambda x: x.select(a.dim if a.dim >= 0 else x.dim() + a.dim, -1)), config=DimConfig, tooltip=T(
    "Letztes Element entlang einer Achse.", "Letzter Zeitschritt.", "Sequenz → Vektor."))
Fn("first", lambda a: (lambda x: x.select(a.dim if a.dim >= 0 else x.dim() + a.dim, 0)), config=DimConfig, tooltip=T(
    "Erstes Element entlang einer Achse.", "CLS-Token.", "Sequenz → Vektor."))
Fn("transpose", lambda a: (lambda x: x.transpose(a.dim0, a.dim1)), config=TransposeConfig, tooltip=T(
    "Zwei Achsen tauschen.", "Layout-Wechsel.", "[B, T, C] ↔ [B, C, T]."))
Fn("abs", lambda a: torch.abs, tooltip=T("Betrag.", "Vorzeichen entfernen.", "Spezialfälle."))
Fn("square", lambda a: torch.square, tooltip=T("Quadrat.", "Energie/Leistung.", "Signale."))
Fn("exp", lambda a: torch.exp, level=Level.ADVANCED, tooltip=T("Exponentialfunktion.", "Positive Werte.", "Raten."))
Fn("log1p", lambda a: torch.log1p, level=Level.ADVANCED, tooltip=T("log(1+x).", "Stauchung großer Werte.", "Zähldaten."))
Fn("clamp", lambda a: (lambda x: x if a.min is None and a.max is None else x.clamp(a.min, a.max)), config=ClampConfig, tooltip=T(
    "Werte begrenzen.", "Harte Grenzen.", "Stabilisierung."))
Fn("scale", lambda a: (lambda x: x * a.value), config=ScalarConfig, tooltip=T(
    "Mit Konstante multiplizieren.", "Feste Skalierung.", "Temperatur, Normierung."))
Fn("detach", lambda a: (lambda x: x.detach()), level=Level.EXPERT, tooltip=T(
    "Gradientenfluss stoppen.", "Zweig wird nicht durch diesen Pfad trainiert.", "Stop-Gradient-Tricks (BYOL, SimSiam)."))
Fn("sign_ste", lambda a: (lambda x: x + (torch.sign(x) - x).detach()), level=Level.EXPERIMENTAL, tooltip=T(
    "Vorzeichen mit Straight-Through-Gradient.", "Binäre Aktivierungen trainierbar machen.", "Binäre Netze, Hashing."))
