"""
ki_baukasten.models.generative
==============================

Generative Modelle: **GAN** und **Diffusionsmodell** (DDPM/DDIM).

GAN (Generative Adversarial Network)
------------------------------------
Zwei Netze spielen gegeneinander: der **Generator** erzeugt Daten aus
Zufallsrauschen, der **Diskriminator** versucht echte von erzeugten zu
unterscheiden. Beide sind hier MLPs mit allen MLP-Einstellungen – oder über
``generator_model``/``discriminator_model`` **beliebige** Modelle aus der
Registry (Kombinationen!). Die Verlustart (Standard, Wasserstein, Hinge ...)
wählt man im Trainings-Paradigma ``gan``.

Diffusionsmodell
----------------
Lernt, schrittweise Rauschen aus Daten zu entfernen. Training: Daten
verrauschen (Vorwärtsprozess) und das Netz das Rauschen (bzw. x₀ oder v)
vorhersagen lassen. Generierung: aus reinem Rauschen Schritt für Schritt
entrauschen (DDPM: stochastisch, viele Schritte; DDIM: deterministisch,
wenige Schritte). Mit Klassen-Konditionierung und Classifier-Free Guidance.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..components.normalization import build_norm
from ..core import ComponentParam, Level, NestedParam, Param, T
from .base import MODELS, BaseModel, ModelConfig
from .mlp import MLP, MLPConfig


# ===========================================================================
# GAN
# ===========================================================================

def _default_generator() -> MLPConfig:
    return MLPConfig(hidden_dims=[128, 128], layer={"activation": "leaky_relu", "norm": "batchnorm"})


def _default_discriminator() -> MLPConfig:
    return MLPConfig(hidden_dims=[128, 128], layer={"activation": {"name": "leaky_relu", "negative_slope": 0.2},
                                                     "parametrization": "spectral_norm"})


class GANConfig(ModelConfig):
    __title__ = "GAN"
    __tooltip__ = T(
        "Generative Adversarial Network aus Generator und Diskriminator.",
        "Erzeugt sehr scharfe, realistische Samples; Training ist ein Gleichgewichtsspiel.",
        "Datensynthese, Bildgenerierung, Datenaugmentation, Simulation.",
    )
    data_dim = Param(2, min=1, tooltip=T(
        "Dimension der (abgeflachten) Daten.", "Ausgabegröße des Generators / Eingabe des Diskriminators.",
        "2 für 2D-Spielzeugdaten, 784 für 28×28-Bilder."))
    latent_dim = Param(16, min=1, tooltip=T(
        "Größe des Rauschvektors z.", "Mehr Dimensionen = mehr Variationsmöglichkeiten.", "16–512."))
    noise_distribution = Param("normal", choices={"normal": "z ~ N(0, I)", "uniform": "z ~ U(-1, 1)"}, tooltip=T(
        "Verteilung des Rauschens.", "Form des Latentraums.", "normal ist Standard."), level=Level.ADVANCED)
    num_classes = Param(0, min=0, tooltip=T(
        "Anzahl Klassen für bedingte Generierung (0 = unbedingt).",
        "Generator und Diskriminator erhalten das Label → gezielte Erzeugung einer Klasse.",
        "Conditional GAN (cGAN).",
    ))
    label_embedding_dim = Param(8, min=1, tooltip=T(
        "Größe des Label-Embeddings.", "Repräsentation der Klasse.", "8–64."), level=Level.ADVANCED)
    generator = NestedParam(MLPConfig, default_factory=_default_generator, tooltip=T(
        "MLP-Generator (input_dim/output_dim werden automatisch gesetzt).", "Erzeugt Daten aus Rauschen.", "Standard."))
    discriminator = NestedParam(MLPConfig, default_factory=_default_discriminator, tooltip=T(
        "MLP-Diskriminator (input_dim/output_dim werden automatisch gesetzt).",
        "Unterscheidet echt/erzeugt. Spectral-Norm stabilisiert.", "Standard."))
    generator_model = ComponentParam("model", None, level=Level.EXPERT, tooltip=T(
        "Beliebiges anderes Modell als Generator (überschreibt 'generator').",
        "Kombinationen, z. B. Transformer-Generator; muss latent_dim(+Label) → data_dim abbilden.",
        "Experimente.",
    ))
    discriminator_model = ComponentParam("model", None, level=Level.EXPERT, tooltip=T(
        "Beliebiges anderes Modell als Diskriminator (überschreibt 'discriminator').",
        "Z. B. CNN-Diskriminator für Bilder; muss Daten(+Label) → 1 Wert abbilden.", "Experimente."))
    output_activation = ComponentParam("activation", "identity", tooltip=T(
        "Aktivierung der Generator-Ausgabe.", "tanh für Daten in [-1, 1] (normalisierte Bilder).", "Bilder: tanh."))


class GAN(BaseModel):
    """GAN gemäß ``GANConfig``. ``model(z, labels)`` erzeugt Daten."""

    def __init__(self, config: GANConfig):
        super().__init__(config)
        c = config
        cond = c.label_embedding_dim if c.num_classes else 0
        if c.generator_model is not None:
            self.generator: nn.Module = c.generator_model.build()
        else:
            g = c.generator.replace(input_dim=c.latent_dim + cond, output_dim=c.data_dim)
            self.generator = MLP(g)
        if c.discriminator_model is not None:
            self.discriminator: nn.Module = c.discriminator_model.build()
        else:
            dcfg = c.discriminator.replace(input_dim=c.data_dim + cond, output_dim=1)
            self.discriminator = MLP(dcfg)
        self.g_label = nn.Embedding(c.num_classes, cond) if c.num_classes else None
        self.d_label = nn.Embedding(c.num_classes, cond) if c.num_classes else None
        self.out_act = c.output_activation.build()
        self.finish()

    def sample_noise(self, n: int, device: Any = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        if self.config.noise_distribution == "uniform":
            return torch.rand(n, self.config.latent_dim, device=device) * 2 - 1
        return torch.randn(n, self.config.latent_dim, device=device)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.g_label is not None:
            if labels is None:
                raise ValueError("Bedingtes GAN braucht labels.")
            z = torch.cat([z, self.g_label(labels)], dim=-1)
        return self.out_act(self.generator(z))

    def discriminate(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Diskriminator-Logit pro Beispiel ``[B]``."""
        is_mlp = isinstance(self.discriminator, MLP)
        h = x.flatten(1) if is_mlp else x  # eigene Modelle (z. B. CNN) bekommen die Originalform
        if self.d_label is not None:
            if not is_mlp:
                raise ValueError("Bedingte Diskriminierung (num_classes > 0) wird nur mit dem MLP-Diskriminator unterstützt.")
            h = torch.cat([h, self.d_label(labels)], dim=-1)
        return self.discriminator(h).reshape(x.shape[0], -1).mean(-1)

    @torch.no_grad()
    def generate(self, n: int, labels: torch.Tensor | None = None) -> torch.Tensor:
        was = self.training
        self.eval()
        out = self(self.sample_noise(n), labels)
        self.train(was)
        return out

    def example_input(self, batch_size: int = 2) -> Any:
        """Eingabe des Generators (Rauschen, ggf. mit Labels)."""
        z = torch.randn(batch_size, self.config.latent_dim)
        if self.config.num_classes:
            return {"z": z, "labels": torch.randint(0, self.config.num_classes, (batch_size,))}
        return z


MODELS.register("gan", GAN, config=GANConfig, tags=("generativ",), tooltip=T(
    "Generative Adversarial Network.", "Scharfe, realistische Samples.", "Datensynthese, Augmentation."))


# ===========================================================================
# Diffusion
# ===========================================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidale Einbettung von Zeitschritten (wie Positionskodierung)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, dim % 2))


def make_beta_schedule(kind: str, steps: int, start: float, end: float) -> torch.Tensor:
    """Rausch-Fahrplan β_t."""
    if kind == "linear":
        return torch.linspace(start, end, steps)
    if kind == "quadratic":
        return torch.linspace(start ** 0.5, end ** 0.5, steps) ** 2
    if kind == "sigmoid":
        return torch.sigmoid(torch.linspace(-6, 6, steps)) * (end - start) + start
    # cosine (Nichol & Dhariwal 2021)
    s = 0.008
    x = torch.linspace(0, steps, steps + 1)
    ac = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    return (1 - ac[1:] / ac[:-1]).clamp(1e-8, 0.999)


class DiffusionConfig(ModelConfig):
    __title__ = "Diffusionsmodell"
    __tooltip__ = T(
        "Denoising-Diffusionsmodell (DDPM/DDIM) mit MLP-Denoiser.",
        "Stabil zu trainieren, sehr hohe Sample-Qualität und -Vielfalt.",
        "Datengenerierung (2D-Verteilungen, kleine Bilder, Tabellen, Trajektorien).",
    )
    data_shape = Param([2], item_type=int, min=1, tooltip=T(
        "Form eines Datenpunkts ohne Batch (z. B. [2] oder [1, 28, 28]).",
        "Der Denoiser arbeitet auf der abgeflachten Form.", "Muss zu den Daten passen."))
    hidden_dim = Param(128, min=1, tooltip=T("Breite des Denoisers.", "Kapazität.", "128–1024."))
    num_blocks = Param(3, min=1, tooltip=T("Residual-Blöcke im Denoiser.", "Tiefe.", "2–8."))
    activation = ComponentParam("activation", "silu", tooltip=T("Aktivierung.", "Nichtlinearität.", "silu (Standard)."))
    norm = ComponentParam("normalization", "layernorm", tooltip=T("Normalisierung.", "Stabilität.", "layernorm."))
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T("Dropout.", "Regularisierung.", "Kleine Daten: 0.1."),
                    group="Regularisierung")
    time_embedding_dim = Param(64, min=2, tooltip=T(
        "Größe der Zeitschritt-Einbettung.", "Wie gut das Netz Rauschstufen unterscheiden kann.", "64–256."),
        level=Level.ADVANCED)
    num_classes = Param(0, min=0, tooltip=T(
        "Klassen für bedingte Generierung (0 = unbedingt).", "Gezielte Erzeugung.", "Conditional Diffusion."))
    class_dropout = Param(0.1, min=0.0, max=1.0, tooltip=T(
        "Wahrscheinlichkeit, im Training das Label wegzulassen.",
        "Ermöglicht Classifier-Free Guidance (gleiches Netz bedingt + unbedingt).",
        "Bedingte Modelle: 0.1–0.2.",
    ), level=Level.ADVANCED)
    timesteps = Param(1000, min=1, tooltip=T(
        "Anzahl Diffusionsschritte T.", "Feinere Zerlegung des Entrauschens.", "DDPM: 1000; Spielzeug: 100–200."))
    beta_schedule = Param("cosine", choices={
        "linear": "linear (DDPM-Original)", "cosine": "Kosinus (bessere Qualität bei wenig Schritten)",
        "quadratic": "quadratisch", "sigmoid": "sigmoid (gut für hohe Auflösungen)",
    }, tooltip=T("Rausch-Fahrplan.", "Wie schnell Information zerstört wird.", "cosine ist robust."))
    beta_start = Param(1e-4, min=0.0, tooltip=T("β am Anfang.", "Anfangs-Rauschstärke.", "1e-4."), level=Level.EXPERT)
    beta_end = Param(0.02, min=0.0, max=1.0, tooltip=T("β am Ende.", "End-Rauschstärke.", "0.02."), level=Level.EXPERT)
    prediction = Param("eps", choices={
        "eps": "Rauschen ε vorhersagen (DDPM-Standard)",
        "x0": "saubere Daten x₀ direkt vorhersagen",
        "v": "v = α·ε − σ·x₀ (stabil über alle Rauschstufen, Stable Diffusion 2)",
    }, tooltip=T("Vorhersageziel des Netzes.", "Beeinflusst Stabilität und Qualität.", "eps oder v."))
    sampler = Param("ddim", choices={"ddpm": "stochastisch, alle Schritte", "ddim": "deterministisch, wenige Schritte"},
                    tooltip=T("Sampling-Verfahren.", "DDIM ist viel schneller.", "Schnelle Generierung: ddim."))
    sample_steps = Param(50, min=1, tooltip=T(
        "Schritte beim Sampling (nur ddim; ddpm nutzt alle).", "Weniger = schneller, gröber.", "20–100."))
    ddim_eta = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Zufallsanteil bei DDIM (0 = deterministisch, 1 ≈ DDPM).", "Vielfalt vs. Reproduzierbarkeit.", "0."),
        level=Level.ADVANCED)
    guidance_scale = Param(1.0, min=0.0, tooltip=T(
        "Classifier-Free-Guidance-Stärke (1 = aus).", "Höher = klassentreuer, weniger vielfältig.", "2–7."),
        level=Level.ADVANCED)
    clip_sample = Param(None, type=float, min=0.0, tooltip=T(
        "Vorhergesagtes x₀ auf [-c, c] begrenzen.", "Verhindert Ausreißer bei Daten in bekanntem Bereich.",
        "Normalisierte Bilder: 1.0."), level=Level.ADVANCED)


class _ResBlock(nn.Module):
    def __init__(self, dim: int, temb: int, cfg: DiffusionConfig):
        super().__init__()
        self.norm = build_norm(cfg.norm, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.t = nn.Linear(temb, dim)
        self.act = cfg.activation.build()
        self.drop = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight)  # Block startet als Identität
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, temb):
        x = self.act(self.fc1(self.norm(h)) + self.t(temb))
        return h + self.fc2(self.drop(x))


class DiffusionModel(BaseModel):
    """Diffusionsmodell. ``model(x_t, t, labels)`` → Vorhersage (eps/x0/v)."""

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        c = config
        self.flat = int(math.prod(c.data_shape))
        betas = make_beta_schedule(c.beta_schedule, c.timesteps, c.beta_start, c.beta_end)
        alphas_cum = torch.cumprod(1 - betas, 0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cum, persistent=False)
        self.inp = nn.Linear(self.flat, c.hidden_dim)
        self.t_mlp = nn.Sequential(nn.Linear(c.time_embedding_dim, c.hidden_dim), c.activation.build(),
                                   nn.Linear(c.hidden_dim, c.hidden_dim))
        self.label = nn.Embedding(c.num_classes + 1, c.hidden_dim) if c.num_classes else None  # +1 = "kein Label"
        self.blocks = nn.ModuleList([_ResBlock(c.hidden_dim, c.hidden_dim, c) for _ in range(c.num_blocks)])
        self.out_norm = build_norm(c.norm, c.hidden_dim)
        self.out = nn.Linear(c.hidden_dim, self.flat)
        self.finish()

    def depth(self) -> int:
        return len(self.blocks)

    def checkpointable_blocks(self) -> list[nn.Module]:
        return list(self.blocks)

    # ------------------------------------------------------------ Netz
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        c = self.config
        h = self.inp(x_t.reshape(x_t.shape[0], -1))
        temb = self.t_mlp(timestep_embedding(t, c.time_embedding_dim))
        if self.label is not None:
            lab = labels if labels is not None else torch.full_like(t, c.num_classes)
            if self.training and c.class_dropout > 0:
                drop = torch.rand(lab.shape, device=lab.device) < c.class_dropout
                lab = torch.where(drop, torch.full_like(lab, c.num_classes), lab)
            temb = temb + self.label(lab)
        for block in self.blocks:
            h = block(h, temb)
        return self.out(self.out_norm(h)).reshape(x_t.shape)

    # ------------------------------------------------------------ Diffusionsmathematik
    def _coef(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ac = self.alphas_cumprod[t].view(-1, *([1] * (x.dim() - 1)))
        return ac.sqrt(), (1 - ac).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Vorwärtsprozess: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε."""
        a, s = self._coef(t, x0)
        return a * x0 + s * noise

    def target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Trainingsziel passend zu ``prediction``."""
        p = self.config.prediction
        if p == "eps":
            return noise
        if p == "x0":
            return x0
        a, s = self._coef(t, x0)
        return a * noise - s * x0

    def predict_x0_eps(self, x_t, t, labels=None) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        out = self(x_t, t, labels)
        if self.label is not None and c.guidance_scale != 1.0 and labels is not None:
            uncond = self(x_t, t, torch.full_like(labels, c.num_classes))
            out = uncond + c.guidance_scale * (out - uncond)
        a, s = self._coef(t, x_t)
        if c.prediction == "eps":
            eps = out
            x0 = (x_t - s * eps) / a
        elif c.prediction == "x0":
            x0 = out
            eps = (x_t - a * x0) / s
        else:
            x0 = a * x_t - s * out
            eps = s * x_t + a * out
        if c.clip_sample is not None:
            x0 = x0.clamp(-c.clip_sample, c.clip_sample)
            eps = (x_t - a * x0) / s
        return x0, eps

    @torch.no_grad()
    def sample(self, n: int, labels: torch.Tensor | None = None, steps: int | None = None,
               sampler: str | None = None) -> torch.Tensor:
        """Erzeugt ``n`` neue Datenpunkte."""
        c = self.config
        was = self.training
        self.eval()
        device = self.alphas_cumprod.device
        x = torch.randn(n, *c.data_shape, device=device)
        sampler = sampler or c.sampler
        if sampler == "ddpm":
            for i in reversed(range(c.timesteps)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                x0, eps = self.predict_x0_eps(x, t, labels)
                ac = self.alphas_cumprod[i]
                ac_prev = self.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=device)
                beta = self.betas[i]
                mean = (ac_prev.sqrt() * beta / (1 - ac)) * x0 + ((1 - beta).sqrt() * (1 - ac_prev) / (1 - ac)) * x
                if i > 0:
                    var = beta * (1 - ac_prev) / (1 - ac)
                    x = mean + var.sqrt() * torch.randn_like(x)
                else:
                    x = mean
        else:
            k = steps or c.sample_steps
            ts = torch.linspace(c.timesteps - 1, 0, min(k, c.timesteps), device=device).long()
            for j, i in enumerate(ts):
                t = torch.full((n,), int(i), device=device, dtype=torch.long)
                x0, eps = self.predict_x0_eps(x, t, labels)
                ac_prev = self.alphas_cumprod[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0, device=device)
                ac = self.alphas_cumprod[i]
                sigma = c.ddim_eta * ((1 - ac_prev) / (1 - ac) * (1 - ac / ac_prev)).clamp_min(0).sqrt()
                dir_xt = (1 - ac_prev - sigma ** 2).clamp_min(0).sqrt() * eps
                x = ac_prev.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x)
        self.train(was)
        return x

    def example_input(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        return {"x_t": torch.randn(batch_size, *self.config.data_shape),
                "t": torch.randint(0, self.config.timesteps, (batch_size,))}


MODELS.register("diffusion", DiffusionModel, config=DiffusionConfig, aliases=("ddpm", "ddim"), tags=("generativ",),
                tooltip=T(
    "Denoising-Diffusionsmodell (DDPM/DDIM).", "Stabil, hohe Qualität und Vielfalt.", "Datengenerierung."))
