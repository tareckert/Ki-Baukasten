"""
ki_baukasten.components.initializers
====================================

Initialisierer – die Startwerte der Gewichte.

Warum ist Initialisierung so wichtig?
-------------------------------------
Ein Netz beginnt mit zufälligen Gewichten. Sind diese zu groß, explodieren
Aktivierungen und Gradienten; sind sie zu klein, verschwindet das Signal.
Gute Schemata (Xavier/Glorot, Kaiming/He, LeCun ...) wählen die Streuung so,
dass die Varianz der Signale über die Schichten ungefähr erhalten bleibt –
abhängig von der Aktivierungsfunktion und der Anzahl Ein-/Ausgänge (Fan).

Registry ``initializer``
------------------------
Fabrik-Signatur: ``factory(args, tensor, **kontext) -> tensor``

* ``tensor`` wird **in-place** überschrieben (unter ``torch.no_grad``).
* Kontext (optional): ``num_layers`` (für tiefenskalierte Schemata).

Die Initialisierer werden genutzt von

* ``InitConfig`` jedes Modells (Standard pro Modultyp: Linear, Conv, Embedding ...),
* der Tensor-Chirurgie (gezielte Initialisierung einzelner Tensoren oder
  sogar einzelner Ausschnitte davon, per Regex auf Parameternamen).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..core import BaseConfig, Level, Param, T, create_registry

INITIALIZERS = create_registry(
    "initializer",
    "Initialisierer",
    T(
        "Verfahren zum Setzen der Startwerte von Gewichten.",
        "Entscheidet mit, ob ein tiefes Netz überhaupt trainierbar ist.",
        "Jedes Modell; gezielt in der Tensor-Chirurgie.",
    ),
    context_doc="factory(args, tensor, num_layers=None) – überschreibt tensor in-place",
)

_NONLINEARITIES = ["linear", "conv1d", "conv2d", "conv3d", "sigmoid", "tanh", "relu", "leaky_relu", "selu"]


def fans(t: torch.Tensor) -> tuple[int, int]:
    """(fan_in, fan_out) – robust auch für 0-/1-dimensionale Tensoren (Biases)."""
    if t.dim() < 2:
        n = max(1, t.numel())
        return n, n
    receptive = t[0][0].numel() if t.dim() > 2 else 1
    return t.size(1) * receptive, t.size(0) * receptive


# ===========================================================================
# Argument-Konfigurationen
# ===========================================================================

class ConstantConfig(BaseConfig):
    value = Param(0.0, tooltip=T("Wert, auf den alle Elemente gesetzt werden.",
                                 "Deterministischer Start.", "Biases, Gates (z. B. Forget-Gate-Bias = 1)."))


class NormalConfig(BaseConfig):
    mean = Param(0.0, tooltip=T("Mittelwert der Normalverteilung.", "Verschiebt alle Startwerte.", "Fast immer 0."))
    std = Param(0.02, min=0.0, tooltip=T(
        "Standardabweichung.",
        "Bestimmt die Startgröße der Gewichte.",
        "Transformer: 0.02 (GPT/BERT-Konvention).",
    ))


class UniformConfig(BaseConfig):
    a = Param(-0.05, tooltip=T("Untere Grenze.", "Startwerte gleichverteilt in [a, b].", "Einfache Netze."))
    b = Param(0.05, tooltip=T("Obere Grenze.", "Startwerte gleichverteilt in [a, b].", "Einfache Netze."))


class TruncNormalConfig(BaseConfig):
    mean = Param(0.0, tooltip=T("Mittelwert.", "Zentrum der Verteilung.", "Meist 0."))
    std = Param(0.02, min=0.0, tooltip=T("Standardabweichung.", "Startgröße der Gewichte.", "ViT/BERT: 0.02."))
    cutoff_std = Param(2.0, min=0.0, tooltip=T(
        "Abschneiden bei ±cutoff_std Standardabweichungen.",
        "Verhindert seltene, extrem große Startwerte.",
        "Vision Transformer, große Sprachmodelle.",
    ), level=Level.ADVANCED)


class GainConfig(BaseConfig):
    gain = Param(1.0, min=0.0, tooltip=T(
        "Skalierungsfaktor der Streuung.",
        "Kompensiert die Varianzänderung durch die Aktivierung.",
        "Wird ignoriert, wenn 'nonlinearity' gesetzt ist (dann automatisch berechnet).",
    ))
    nonlinearity = Param(None, type=str, choices=_NONLINEARITIES, tooltip=T(
        "Aktivierung, für die der Gain automatisch berechnet wird.",
        "z. B. relu → √2, tanh → 5/3.",
        "Wenn die nachfolgende Aktivierung bekannt ist.",
    ), level=Level.ADVANCED)

    def resolved_gain(self) -> float:
        return nn.init.calculate_gain(self.nonlinearity) if self.nonlinearity else self.gain


class KaimingConfig(BaseConfig):
    a = Param(0.0, tooltip=T(
        "Negative Steigung der nachfolgenden (Leaky-)ReLU.",
        "Passt die Varianz für Leaky-ReLU an.",
        "Nur bei leaky_relu relevant.",
    ), level=Level.ADVANCED)
    mode = Param("fan_in", choices={
        "fan_in": "erhält die Varianz im Vorwärtsdurchlauf",
        "fan_out": "erhält die Varianz der Gradienten im Rückwärtsdurchlauf",
    }, tooltip=T("Welche Varianz erhalten bleibt.", "fan_in ist Standard; fan_out bei ResNets üblich.", "CNNs."))
    nonlinearity = Param("relu", choices=_NONLINEARITIES, tooltip=T(
        "Nachfolgende Aktivierung.", "Bestimmt den Gain.", "relu oder leaky_relu."))


class SparseConfig(BaseConfig):
    sparsity = Param(0.9, min=0.0, max=1.0, tooltip=T(
        "Anteil der Nullen pro Spalte.", "Sparse Start – manche Netze lernen damit schneller.", "Echo-State-Netze."))
    std = Param(0.01, min=0.0, tooltip=T("Streuung der Nicht-Null-Werte.", "Startgröße.", "Klein halten."))


class ScaledNormalConfig(BaseConfig):
    std = Param(0.02, min=0.0, tooltip=T("Basis-Standardabweichung.", "Vor der Tiefenskalierung.", "0.02."))
    num_layers = Param(None, type=int, min=1, tooltip=T(
        "Anzahl Schichten N; std wird durch √(2N) geteilt.",
        "Verhindert, dass sich Residual-Beiträge über viele Schichten aufsummieren und explodieren.",
        "Ausgangsprojektionen in Residual-Blöcken (GPT-2-Schema).",
        "None = automatisch vom Modell (Anzahl Blöcke).",
    ))


class SirenConfig(BaseConfig):
    w0 = Param(30.0, min=1e-6, tooltip=T("Frequenzfaktor der Sinus-Aktivierung.", "Muss zur Aktivierung passen.", "SIREN."))
    is_first = Param(False, tooltip=T(
        "Erste Schicht eines SIREN-Netzes.", "Erste Schicht nutzt eine andere Grenze (1/fan_in).", "SIREN."))


class OrthogonalConfig(BaseConfig):
    gain = Param(1.0, min=0.0, tooltip=T("Skalierung der orthogonalen Matrix.", "Normerhaltung.", "RNNs, tiefe Netze."))


class SmallInitConfig(BaseConfig):
    factor = Param(0.4, min=0.0, tooltip=T(
        "Varianz = factor / d (Standard 2/5 = 0.4).",
        "Kleine Init stabilisiert Pre-Norm-Transformer ohne Warmup.",
        "'Transformers without Tears'-Schema.",
    ))


# ===========================================================================
# Registrierung
# ===========================================================================

R = INITIALIZERS.register


def _ng(fn):
    """Dekorator: führt die Initialisierung ohne Gradientenaufzeichnung aus."""

    def wrapper(a, t, **ctx):
        with torch.no_grad():
            fn(a, t, **ctx)
        return t

    wrapper.__name__ = fn.__name__
    return wrapper


@R("keep", aliases=("default", "pytorch"), tooltip=T(
    "Nichts ändern – PyTorch-Standardinitialisierung behalten.",
    "PyTorch initialisiert Linear/Conv bereits sinnvoll (Kaiming-uniform mit a=√5).",
    "Wenn keine besondere Initialisierung nötig ist.",
))
def _keep(a, t, **ctx):
    return t


R("zeros", _ng(lambda a, t, **c: t.zero_()), tooltip=T(
    "Alles auf 0.", "Neutraler Start.", "Biases; Ausgangsprojektionen für 'Start als Identität' (Zero-Init-Residual)."))
R("ones", _ng(lambda a, t, **c: t.fill_(1.0)), tooltip=T("Alles auf 1.", "Neutrale Skala.", "Norm-Gewichte."))
R("constant", _ng(lambda a, t, **c: t.fill_(a.value)), config=ConstantConfig, tooltip=T(
    "Alles auf einen festen Wert.", "Gezielte Startwerte.", "Gate-Biases, Experimente."))
R("normal", _ng(lambda a, t, **c: t.normal_(a.mean, a.std)), config=NormalConfig, tooltip=T(
    "Normalverteilt N(mean, std²).", "Einfach und kontrollierbar.", "Transformer (std=0.02), Embeddings."))
R("uniform", _ng(lambda a, t, **c: t.uniform_(a.a, a.b)), config=UniformConfig, tooltip=T(
    "Gleichverteilt in [a, b].", "Begrenzte Startwerte.", "Einfache Netze."))
R("trunc_normal", _ng(lambda a, t, **c: nn.init.trunc_normal_(
    t, a.mean, a.std, a.mean - a.cutoff_std * a.std, a.mean + a.cutoff_std * a.std)),
  config=TruncNormalConfig, tooltip=T(
      "Abgeschnittene Normalverteilung.", "Keine Ausreißer-Startwerte.", "ViT, BERT, große LLMs."))
R("xavier_uniform", _ng(lambda a, t, **c: nn.init.xavier_uniform_(t, a.resolved_gain()) if t.dim() >= 2
                        else t.uniform_(-a.resolved_gain() * math.sqrt(3.0 / fans(t)[0]),
                                        a.resolved_gain() * math.sqrt(3.0 / fans(t)[0]))),
  config=GainConfig, aliases=("glorot_uniform",), tooltip=T(
      "Xavier/Glorot gleichverteilt: Varianz 2/(fan_in+fan_out).",
      "Erhält Varianz in beide Richtungen für symmetrische Aktivierungen (tanh, sigmoid, linear).",
      "Transformer-Projektionen, tanh-Netze.",
  ))
R("xavier_normal", _ng(lambda a, t, **c: nn.init.xavier_normal_(t, a.resolved_gain()) if t.dim() >= 2
                       else t.normal_(0, a.resolved_gain() / math.sqrt(fans(t)[0]))),
  config=GainConfig, aliases=("glorot_normal",), tooltip=T(
      "Xavier/Glorot normalverteilt.", "Wie xavier_uniform, aber Normalverteilung.", "Wie xavier_uniform."))
R("kaiming_uniform", _ng(lambda a, t, **c: nn.init.kaiming_uniform_(t, a.a, a.mode, a.nonlinearity) if t.dim() >= 2
                         else t.uniform_(-math.sqrt(6.0 / fans(t)[0]), math.sqrt(6.0 / fans(t)[0]))),
  config=KaimingConfig, aliases=("he_uniform",), tooltip=T(
      "Kaiming/He gleichverteilt: Varianz 2/fan.",
      "Erhält Varianz bei ReLU-Netzen (ReLU halbiert die Varianz).",
      "CNNs und MLPs mit ReLU.",
  ))
R("kaiming_normal", _ng(lambda a, t, **c: nn.init.kaiming_normal_(t, a.a, a.mode, a.nonlinearity) if t.dim() >= 2
                        else t.normal_(0, math.sqrt(2.0 / fans(t)[0]))),
  config=KaimingConfig, aliases=("he_normal",), tooltip=T(
      "Kaiming/He normalverteilt.", "Wie kaiming_uniform.", "ResNets (oft mode=fan_out)."))
R("lecun_normal", _ng(lambda a, t, **c: nn.init.trunc_normal_(
    t, 0.0, 1.0 / math.sqrt(fans(t)[0]), -2.0 / math.sqrt(fans(t)[0]), 2.0 / math.sqrt(fans(t)[0]))), tooltip=T(
    "LeCun normal: Varianz 1/fan_in.", "Pflicht für selbstnormalisierende SELU-Netze.", "SELU-MLPs."))
R("lecun_uniform", _ng(lambda a, t, **c: t.uniform_(-math.sqrt(3.0 / fans(t)[0]), math.sqrt(3.0 / fans(t)[0]))),
  level=Level.ADVANCED, tooltip=T("LeCun gleichverteilt.", "Varianz 1/fan_in.", "Wie lecun_normal."))
R("orthogonal", _ng(lambda a, t, **c: nn.init.orthogonal_(t, a.gain) if t.dim() >= 2 else t.normal_(0, 1e-2)),
  config=OrthogonalConfig, tooltip=T(
      "(Halb-)orthogonale Matrix.",
      "Erhält Normen exakt – verhindert explodierende/verschwindende Signale über viele Zeitschritte.",
      "Rekurrente Gewichte (RNN/LSTM), RL-Policies (PPO-Konvention: gain √2 bzw. 0.01).",
  ))
R("eye", _ng(lambda a, t, **c: nn.init.eye_(t) if t.dim() == 2 else t.fill_(1.0)), aliases=("identity",),
  level=Level.ADVANCED, tooltip=T(
      "Einheitsmatrix.", "Schicht startet als Identität.", "Residual-artige Starts, Adapter."))
R("dirac", _ng(lambda a, t, **c: nn.init.dirac_(t) if t.dim() in (3, 4, 5) else t.zero_()), level=Level.EXPERT,
  tooltip=T("Dirac-Delta für Faltungen.", "Faltung startet als Identität.", "Sehr tiefe CNNs ohne Residuals."))
R("sparse", _ng(lambda a, t, **c: nn.init.sparse_(t, a.sparsity, a.std) if t.dim() == 2 else t.normal_(0, a.std)),
  config=SparseConfig, level=Level.EXPERT, tooltip=T(
      "Dünn besetzte Startmatrix.", "Viele Nullen, wenige Verbindungen.", "Reservoir-Computing, Experimente."))
R("small_init", _ng(lambda a, t, **c: t.normal_(0.0, math.sqrt(a.factor / fans(t)[0]))), config=SmallInitConfig,
  level=Level.ADVANCED, tooltip=T(
      "Kleine Normal-Init (Varianz 2/(5d)).", "Stabil ohne Warmup.", "Pre-Norm-Transformer."))


@R("scaled_normal", config=ScaledNormalConfig, level=Level.ADVANCED, tooltip=T(
    "Normalverteilung mit tiefenabhängiger Skalierung std/√(2N).",
    "Hält die Varianz des Residual-Stroms in tiefen Netzen konstant.",
    "Ausgangsprojektionen von Attention/FFN in GPT-artigen Modellen.",
))
def _scaled_normal(a, t, num_layers=None, **ctx):
    n = a.num_layers or num_layers or 1
    with torch.no_grad():
        t.normal_(0.0, a.std / math.sqrt(2 * n))
    return t


@R("siren", config=SirenConfig, level=Level.ADVANCED, tooltip=T(
    "SIREN-Initialisierung für Sinus-Netze.",
    "Ohne sie lernen Sinus-Netze kaum.",
    "Zusammen mit Aktivierung 'sine'.",
))
def _siren(a, t, **ctx):
    fan_in = fans(t)[0]
    bound = 1.0 / fan_in if a.is_first else math.sqrt(6.0 / fan_in) / a.w0
    with torch.no_grad():
        t.uniform_(-bound, bound)
    return t


def apply_initializer(spec: object, tensor: torch.Tensor, **ctx) -> torch.Tensor:
    """Wendet einen Initialisierer (Name/dict/Spec) auf ``tensor`` an."""
    return INITIALIZERS.build(spec, tensor, **ctx)
