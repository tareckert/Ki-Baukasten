"""
ki_baukasten.components.activations
===================================

Aktivierungsfunktionen – die Nichtlinearitäten zwischen den Schichten.

Warum ist die Wahl wichtig?
---------------------------
Ohne Nichtlinearität wäre ein noch so tiefes Netz nur eine einzige lineare
Abbildung. Die Aktivierung entscheidet mit über

* **Gradientenfluss** (ReLU: tote Neuronen möglich; GELU/SiLU: glatt),
* **Ausdrucksstärke** (z. B. periodische Funktionen für Signale: Sinus/Snake),
* **Rechenkosten** (ReLU billig, Mish/GELU etwas teurer),
* **Stabilität** (SELU ist selbstnormalisierend, aber nur mit passender Init).

Registry ``activation``
-----------------------
Fabrik-Signatur: ``factory(args) -> nn.Module`` (kein weiterer Kontext).
Alle Einträge sind über ``ComponentParam("activation", ...)`` überall im
Baukasten wählbar – in MLPs, CNNs, Transformer-FFNs, Gates, Köpfen usw.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..core import BaseConfig, Level, Param, T, create_registry

ACTIVATIONS = create_registry(
    "activation",
    "Aktivierungsfunktionen",
    T(
        "Nichtlineare Funktionen, die auf jedes Element angewendet werden.",
        "Machen Netze erst ausdrucksstark; beeinflussen Gradientenfluss und Stabilität.",
        "Überall zwischen linearen Schichten / Faltungen.",
    ),
    context_doc="factory(args) – kein zusätzlicher Kontext",
)


# ===========================================================================
# Argument-Konfigurationen
# ===========================================================================

class InplaceConfig(BaseConfig):
    __title__ = "In-place-Option"
    inplace = Param(False, level=Level.EXPERT, tooltip=T(
        "Ergebnis direkt in den Eingabespeicher schreiben.",
        "Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.",
        "Speicheroptimierung bei sehr großen Aktivierungen.",
        "Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward.",
    ))


class LeakyReLUConfig(InplaceConfig):
    negative_slope = Param(0.01, tooltip=T(
        "Steigung für negative Eingaben.",
        "Verhindert 'tote' Neuronen, weil auch negative Werte einen kleinen Gradienten liefern.",
        "Tiefe Netze, GAN-Diskriminatoren (dort oft 0.2).",
        "0.01 Standard, 0.1–0.3 bei GANs. 0 entspricht ReLU.",
    ))


class PReLUConfig(BaseConfig):
    num_parameters = Param(1, min=1, tooltip=T(
        "Anzahl lernbarer Steigungen (1 = eine für alle, sonst eine pro Kanal).",
        "Lässt das Netz die negative Steigung selbst lernen.",
        "CNNs (pro Kanal), wenn Leaky-ReLU-Steigung unklar ist.",
        "Bei pro-Kanal-Variante muss num_parameters = Kanalanzahl sein.",
    ))
    init = Param(0.25, tooltip=T(
        "Startwert der lernbaren negativen Steigung.",
        "Bestimmt das Verhalten zu Trainingsbeginn.",
        "Feintuning des Startverhaltens.",
    ), level=Level.ADVANCED)


class RReLUConfig(InplaceConfig):
    lower = Param(1 / 8, tooltip=T("Untere Grenze der zufälligen negativen Steigung.",
                                   "Zufall im Training wirkt regularisierend.", "Kleine Datensätze."))
    upper = Param(1 / 3, tooltip=T("Obere Grenze der zufälligen negativen Steigung.",
                                   "Im Eval-Modus wird der Mittelwert verwendet.", "Kleine Datensätze."))


class ELUConfig(InplaceConfig):
    alpha = Param(1.0, tooltip=T(
        "Sättigungswert für negative Eingaben (-alpha).",
        "Glatte negative Seite → Mittelwert der Aktivierungen näher an 0, schnelleres Lernen.",
        "Alternative zu ReLU in tiefen MLPs.",
    ))


class CELUConfig(InplaceConfig):
    alpha = Param(1.0, min=1e-6, tooltip=T("Alpha der stetig differenzierbaren ELU.",
                                           "Wie ELU, aber für alle alpha stetig differenzierbar.", "Wie ELU."))


class GELUConfig(BaseConfig):
    approximate = Param("none", choices={
        "none": "exakte Berechnung über die Fehlerfunktion (erf)",
        "tanh": "schnellere tanh-Näherung (wie im Original-GPT-2/BERT-Code)",
    }, tooltip=T(
        "Berechnungsart der GELU.",
        "Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.",
        "Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh).",
    ), level=Level.ADVANCED)


class SoftplusConfig(BaseConfig):
    beta = Param(1.0, min=1e-6, tooltip=T("Schärfe der Softplus-Kurve.",
                                          "Größeres beta → näher an ReLU.", "Positive Ausgaben (z. B. Varianzen)."))
    threshold = Param(20.0, tooltip=T("Ab diesem Wert (x*beta) wird linear gerechnet.",
                                      "Numerische Stabilität für große Eingaben.", "Selten zu ändern."),
                      level=Level.EXPERT)


class HardtanhConfig(InplaceConfig):
    min_val = Param(-1.0, tooltip=T("Untere Grenze.", "Begrenzt Ausgaben hart.", "Quantisierungsfreundliche Netze."))
    max_val = Param(1.0, tooltip=T("Obere Grenze.", "Begrenzt Ausgaben hart.", "Quantisierungsfreundliche Netze."))


class ShrinkConfig(BaseConfig):
    lambd = Param(0.5, min=0.0, tooltip=T(
        "Schwellwert, unterhalb dessen (Betrag) auf 0 gesetzt wird.",
        "Erzeugt Sparsity (viele exakte Nullen).",
        "Sparse Coding, Entrauschen.",
    ))


class SwishConfig(BaseConfig):
    beta = Param(1.0, tooltip=T(
        "Steilheit: f(x) = x · sigmoid(beta · x).",
        "beta=1 entspricht SiLU; groß → ReLU-ähnlich; klein → fast linear.",
        "Experimente mit glatten ReLU-Varianten.",
    ))
    learnable = Param(False, tooltip=T(
        "beta als lernbaren Parameter behandeln.",
        "Das Netz passt die Form der Aktivierung selbst an.",
        "Architektursuche, Forschung.",
    ), level=Level.EXPERIMENTAL)


class SineConfig(BaseConfig):
    w0 = Param(1.0, tooltip=T(
        "Frequenzfaktor: f(x) = sin(w0 · x).",
        "Periodische Aktivierungen (SIREN) können feine Details/hohe Frequenzen darstellen.",
        "Implizite neuronale Repräsentationen (Bilder/Signale als Funktion), physikalische Felder.",
        "SIREN nutzt w0=30 in der ersten Schicht und spezielle Initialisierung (siehe Initialisierer 'siren').",
    ))


class SnakeConfig(BaseConfig):
    alpha = Param(1.0, min=1e-6, tooltip=T(
        "Frequenz: f(x) = x + sin²(alpha·x)/alpha.",
        "Kombiniert monotonen Trend mit Periodizität – gut für periodische Signale.",
        "Audio-Synthese (BigVGAN), Zeitreihen mit Saisonalität.",
    ))
    learnable = Param(True, tooltip=T("alpha lernbar machen.", "Netz findet passende Frequenzen selbst.",
                                      "Standard bei Snake."), level=Level.ADVANCED)
    num_channels = Param(1, min=1, tooltip=T(
        "Anzahl unabhängiger alpha-Werte (pro Kanal, letzte Dimension).",
        "Jeder Kanal kann eine eigene Frequenz lernen.",
        "Mehrkanalige Signale.",
        "Muss zur Größe der letzten Dimension passen oder 1 sein.",
    ), level=Level.ADVANCED)


class GaussianConfig(BaseConfig):
    sigma = Param(1.0, min=1e-6, tooltip=T(
        "Breite: f(x) = exp(-x² / (2σ²)).",
        "Lokale, glockenförmige Antwort (wie RBF-Netze).",
        "Radiale-Basis-Netze, Funktionsapproximation.",
    ))


class StarReLUConfig(BaseConfig):
    scale = Param(0.8944, tooltip=T("Skalierung s in s·relu(x)² + b.", "Normiert die Varianz der Ausgabe.",
                                    "MetaFormer-Architekturen."))
    bias = Param(-0.4472, tooltip=T("Verschiebung b in s·relu(x)² + b.", "Zentriert die Ausgabe.",
                                    "MetaFormer-Architekturen."))
    learnable = Param(True, tooltip=T("scale/bias lernbar.", "Mehr Flexibilität bei 2 Parametern.", "Standard."))


class SoftcapConfig(BaseConfig):
    cap = Param(30.0, min=1e-6, tooltip=T(
        "Grenze: f(x) = cap · tanh(x / cap).",
        "Begrenzt Werte weich, ohne Gradienten ganz abzuschneiden.",
        "Logit-Begrenzung (Gemma 2), Stabilisierung extremer Aktivierungen.",
    ))


class PolynomialConfig(BaseConfig):
    degree = Param(3, min=1, max=9, tooltip=T(
        "Grad des Polynoms f(x) = Σ a_k · x^k.",
        "Das Netz lernt die Form der Aktivierung selbst.",
        "Forschung zu lernbaren Aktivierungen.",
        "Hohe Grade sind numerisch instabil – Eingaben sollten normalisiert sein.",
    ))
    init = Param("identity", choices={"identity": "Start als f(x)=x", "relu_like": "Näherung an ReLU um 0"},
                 tooltip=T("Startform des Polynoms.", "Guter Start erleichtert das Training.", "Selten zu ändern."))


class ThresholdConfig(InplaceConfig):
    threshold = Param(0.0, tooltip=T("Schwelle.", "Werte ≤ Schwelle werden ersetzt.", "Spezialfälle."))
    value = Param(0.0, tooltip=T("Ersatzwert unterhalb der Schwelle.", "Bestimmt die Ausgabe dort.", "Spezialfälle."))


# ===========================================================================
# Eigene Module (nicht in PyTorch enthalten)
# ===========================================================================

class SquaredReLU(nn.Module):
    """relu(x)² – aus "Primer" (So et al. 2021); stark in Transformer-FFNs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


class Swish(nn.Module):
    """x · sigmoid(beta · x) mit optional lernbarem beta."""

    def __init__(self, beta: float = 1.0, learnable: bool = False):
        super().__init__()
        t = torch.tensor(float(beta))
        if learnable:
            self.beta = nn.Parameter(t)
        else:
            self.register_buffer("beta", t, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class QuickGELU(nn.Module):
    """x · sigmoid(1.702 · x) – schnelle GELU-Näherung (CLIP)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Sine(nn.Module):
    """sin(w0 · x) – SIREN-Aktivierung."""

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Snake(nn.Module):
    """x + sin²(αx)/α mit (optional lernbarem, kanalweisem) α."""

    def __init__(self, alpha: float = 1.0, learnable: bool = True, num_channels: int = 1):
        super().__init__()
        t = torch.full((num_channels,), float(alpha))
        if learnable:
            self.alpha = nn.Parameter(t)
        else:
            self.register_buffer("alpha", t, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.clamp_min(1e-6)
        return x + torch.sin(a * x).square() / a


class Gaussian(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x.square() / (2 * self.sigma ** 2))


class StarReLU(nn.Module):
    """s · relu(x)² + b (Yu et al. 2022, MetaFormer Baselines)."""

    def __init__(self, scale: float, bias: float, learnable: bool):
        super().__init__()
        s, b = torch.tensor(scale), torch.tensor(bias)
        if learnable:
            self.scale, self.bias = nn.Parameter(s), nn.Parameter(b)
        else:
            self.register_buffer("scale", s, persistent=False)
            self.register_buffer("bias", b, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.relu(x).square() + self.bias


class Softcap(nn.Module):
    def __init__(self, cap: float):
        super().__init__()
        self.cap = cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cap * torch.tanh(x / self.cap)


class PolynomialActivation(nn.Module):
    """Lernbares Polynom Σ a_k x^k (experimentell)."""

    def __init__(self, degree: int, init: str):
        super().__init__()
        coeffs = torch.zeros(degree + 1)
        if init == "identity":
            coeffs[1] = 1.0
        else:  # relu_like: 0.25 + 0.5x + 0.25x² (Näherung um 0)
            coeffs[0], coeffs[1] = 0.0, 0.5
            if degree >= 2:
                coeffs[2] = 0.25
        self.coeffs = nn.Parameter(coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Horner-Schema: numerisch stabiler und schneller als Potenzen.
        out = torch.zeros_like(x) + self.coeffs[-1]
        for c in reversed(self.coeffs[:-1]):
            out = out * x + c
        return out


# ===========================================================================
# Registrierung
# ===========================================================================

R = ACTIVATIONS.register

R("relu", lambda a: nn.ReLU(inplace=a.inplace), config=InplaceConfig, tags=("klassisch",), tooltip=T(
    "Rectified Linear Unit: max(0, x).",
    "Extrem billig, kein Sättigen für x>0 → schnelles Training.",
    "Standard für CNNs und einfache MLPs.",
    "Kann 'tote' Neuronen erzeugen (immer 0) – dann Leaky-ReLU/GELU probieren.",
))
R("leaky_relu", lambda a: nn.LeakyReLU(a.negative_slope, inplace=a.inplace), config=LeakyReLUConfig,
  aliases=("lrelu",), tooltip=T(
      "ReLU mit kleiner Steigung für negative Werte.",
      "Kein Absterben von Neuronen.",
      "GAN-Diskriminatoren, tiefe CNNs.",
  ))
R("prelu", lambda a: nn.PReLU(a.num_parameters, a.init), config=PReLUConfig, level=Level.ADVANCED, tooltip=T(
    "Parametrische ReLU mit lernbarer negativer Steigung.",
    "Netz lernt optimale Steigung selbst.",
    "Bildverarbeitung (z. B. Super-Resolution).",
))
R("rrelu", lambda a: nn.RReLU(a.lower, a.upper, inplace=a.inplace), config=RReLUConfig, level=Level.ADVANCED,
  tooltip=T("Randomisierte Leaky-ReLU.", "Zufällige Steigung im Training regularisiert.", "Kleine Datensätze."))
R("elu", lambda a: nn.ELU(a.alpha, inplace=a.inplace), config=ELUConfig, tooltip=T(
    "Exponential Linear Unit.", "Glatte negative Seite, Aktivierungsmittel nahe 0.", "Tiefe MLPs."))
R("celu", lambda a: nn.CELU(a.alpha, inplace=a.inplace), config=CELUConfig, level=Level.ADVANCED,
  tooltip=T("Stetig differenzierbare ELU.", "Glatter als ELU.", "Wie ELU."))
R("selu", lambda a: nn.SELU(inplace=a.inplace), config=InplaceConfig, tooltip=T(
    "Scaled ELU – selbstnormalisierend.",
    "Hält Mittelwert/Varianz der Aktivierungen stabil, ganz ohne Normalisierungsschichten.",
    "Tiefe MLPs ohne BatchNorm ('Self-Normalizing Networks').",
    "Nur mit LeCun-Normal-Initialisierung und AlphaDropout korrekt wirksam.",
))
R("gelu", lambda a: nn.GELU(approximate=a.approximate), config=GELUConfig, tags=("transformer",), tooltip=T(
    "Gaussian Error Linear Unit: x · Φ(x).",
    "Glatt, nicht-monoton; empirisch sehr gut in Transformern.",
    "BERT, GPT, ViT – Standard in Transformer-FFNs.",
))
R("quick_gelu", lambda a: QuickGELU(), level=Level.ADVANCED, tooltip=T(
    "Schnelle GELU-Näherung x·sigmoid(1.702x).", "Billiger als exakte GELU.", "CLIP-kompatible Modelle."))
R("silu", lambda a: nn.SiLU(), aliases=("swish1",), tags=("transformer",), tooltip=T(
    "Sigmoid Linear Unit: x · sigmoid(x) (auch Swish).",
    "Glatt, selbst-gated; sehr gut in modernen Netzen.",
    "LLaMA (als SwiGLU-Gate), EfficientNet, Diffusionsmodelle.",
))
R("swish", lambda a: Swish(a.beta, a.learnable), config=SwishConfig, level=Level.ADVANCED, tooltip=T(
    "Swish mit einstellbarem/lernbarem beta.", "Form zwischen linear und ReLU frei wählbar.", "Experimente."))
R("mish", lambda a: nn.Mish(), tooltip=T(
    "Mish: x · tanh(softplus(x)).", "Sehr glatt, oft leicht besser als ReLU in CNNs.", "Objekterkennung (YOLOv4)."))
R("tanh", lambda a: nn.Tanh(), tags=("klassisch",), tooltip=T(
    "Tangens hyperbolicus, Ausgabe in (-1, 1).",
    "Nullzentriert; sättigt aber für große |x| (verschwindende Gradienten).",
    "RNNs, Ausgaben im Bereich [-1, 1] (z. B. Bild-Generatoren, Aktionen).",
))
R("sigmoid", lambda a: nn.Sigmoid(), tags=("klassisch",), tooltip=T(
    "Logistische Funktion, Ausgabe in (0, 1).",
    "Interpretierbar als Wahrscheinlichkeit/Gate.",
    "Gates (LSTM/GLU), binäre Ausgaben.",
    "Als Zwischenaktivierung in tiefen Netzen meist schlecht (Sättigung).",
))
R("softplus", lambda a: nn.Softplus(a.beta, a.threshold), config=SoftplusConfig, tooltip=T(
    "Glatte ReLU: log(1 + e^x).", "Immer positiv und differenzierbar.", "Varianzen, Raten, Intensitäten."))
R("softsign", lambda a: nn.Softsign(), level=Level.ADVANCED, tooltip=T(
    "x / (1 + |x|).", "Wie tanh, aber langsamer sättigend.", "Alternative zu tanh."))
R("hardtanh", lambda a: nn.Hardtanh(a.min_val, a.max_val, inplace=a.inplace), config=HardtanhConfig,
  level=Level.ADVANCED, tooltip=T("Stückweise lineares tanh.", "Billig, begrenzt.", "Mobile/Quantisierung."))
R("relu6", lambda a: nn.ReLU6(inplace=a.inplace), config=InplaceConfig, tooltip=T(
    "ReLU begrenzt auf [0, 6].", "Robust bei niedriger Präzision.", "MobileNet, Quantisierung."))
R("hardswish", lambda a: nn.Hardswish(inplace=a.inplace), config=InplaceConfig, tooltip=T(
    "Stückweise Näherung von Swish.", "Swish-Qualität zu ReLU-Kosten.", "MobileNetV3, Edge-Geräte."))
R("hardsigmoid", lambda a: nn.Hardsigmoid(inplace=a.inplace), config=InplaceConfig, level=Level.ADVANCED,
  tooltip=T("Stückweise lineare Sigmoid.", "Billiges Gate.", "Squeeze-Excitation in MobileNets."))
R("logsigmoid", lambda a: nn.LogSigmoid(), level=Level.ADVANCED, tooltip=T(
    "log(sigmoid(x)).", "Numerisch stabil für Log-Wahrscheinlichkeiten.", "Spezielle Verlustterme."))
R("tanhshrink", lambda a: nn.Tanhshrink(), level=Level.EXPERT, tooltip=T(
    "x - tanh(x).", "Unterdrückt kleine Werte weich.", "Signalverarbeitung."))
R("softshrink", lambda a: nn.Softshrink(a.lambd), config=ShrinkConfig, level=Level.EXPERT, tooltip=T(
    "Weiches Schrumpfen um lambda.", "Sparsity mit stetiger Funktion.", "Sparse Coding (ISTA)."))
R("hardshrink", lambda a: nn.Hardshrink(a.lambd), config=ShrinkConfig, level=Level.EXPERT, tooltip=T(
    "Hartes Nullsetzen kleiner Werte.", "Exakte Sparsity.", "Entrauschen."))
R("threshold", lambda a: nn.Threshold(a.threshold, a.value, inplace=a.inplace), config=ThresholdConfig,
  level=Level.EXPERT, tooltip=T("Allgemeine Schwellwertfunktion.", "Frei definierbares Abschneiden.", "Spezialfälle."))
R("identity", lambda a: nn.Identity(), aliases=("linear", "none", "keine"), tooltip=T(
    "Keine Aktivierung (f(x) = x).",
    "Lässt Werte unverändert – z. B. für Ausgabeschichten bei Regression.",
    "Ausgabeschichten, lineare Modelle, Ablationsstudien.",
))
R("squared_relu", lambda a: SquaredReLU(), aliases=("relu2",), level=Level.ADVANCED, tags=("transformer",), tooltip=T(
    "Quadrierte ReLU: relu(x)².",
    "In der 'Primer'-Suche als starke FFN-Aktivierung gefunden.",
    "Transformer-FFNs als Alternative zu GELU.",
))
R("star_relu", lambda a: StarReLU(a.scale, a.bias, a.learnable), config=StarReLUConfig,
  level=Level.EXPERIMENTAL, tooltip=T("StarReLU: s·relu(x)²+b.", "Varianz-normierte quadrierte ReLU.", "MetaFormer."))
R("sine", lambda a: Sine(a.w0), config=SineConfig, aliases=("sin", "siren"), level=Level.ADVANCED, tooltip=T(
    "Sinus-Aktivierung (SIREN).",
    "Stellt hohe Frequenzen dar, Ableitungen sind wieder glatt.",
    "Implizite Repräsentationen (NeRF-artig), PDE-Lösungen.",
))
R("snake", lambda a: Snake(a.alpha, a.learnable, a.num_channels), config=SnakeConfig, level=Level.ADVANCED,
  tooltip=T("Snake: x + sin²(αx)/α.", "Periodisch + monoton.", "Audio, periodische Zeitreihen."))
R("gaussian", lambda a: Gaussian(a.sigma), config=GaussianConfig, level=Level.EXPERIMENTAL, tooltip=T(
    "Gauß-Glocke als Aktivierung.", "Lokale Antworten wie RBF-Netze.", "Funktionsapproximation."))
R("softcap", lambda a: Softcap(a.cap), config=SoftcapConfig, level=Level.ADVANCED, tooltip=T(
    "Weiche Begrenzung cap·tanh(x/cap).", "Verhindert Ausreißer-Werte.", "Logits, Attention-Scores."))
R("polynomial", lambda a: PolynomialActivation(a.degree, a.init), config=PolynomialConfig,
  level=Level.EXPERIMENTAL, tooltip=T(
      "Lernbares Polynom als Aktivierung.", "Form der Nichtlinearität wird gelernt.", "Forschung."))


def build_activation(spec: object) -> nn.Module:
    """Baut eine Aktivierung aus Name, dict oder ``ComponentSpec``."""
    return ACTIVATIONS.build(spec)

