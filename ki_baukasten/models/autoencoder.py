"""
ki_baukasten.models.autoencoder
===============================

Autoencoder-Familie: AE, VAE (inkl. β-VAE), Denoising-AE, Sparse-AE, VQ-VAE.

Prinzip
-------
::

    x ──► Encoder ──► z (Latentraum, klein) ──► Decoder ──► x̂ ≈ x

Durch den Engpass muss das Modell die *wesentliche* Struktur der Daten
lernen. Anwendungen: Kompression, Anomalieerkennung (hoher
Rekonstruktionsfehler = ungewöhnlich), Entrauschen, Merkmalslernen,
Generierung (VAE/VQ-VAE).

Ausgabe
-------
``forward`` liefert ein ``dict``:

* ``reconstruction`` – x̂
* ``latent``         – z
* ``mu``/``logvar``  – nur VAE
* ``kl``             – nur VAE (KL-Divergenz pro Beispiel, gemittelt)

Regularisierungen, die nicht vom Trainingsschritt abhängen (Sparsity,
VQ-Commitment), legt das Modell als ``aux_loss`` ab; der KL-Term wird vom
Paradigma ``autoencoder`` gewichtet (inkl. KL-Annealing).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..components.normalization import build_norm
from ..core import ComponentParam, Level, Param, T
from .base import MODELS, BaseModel, ModelConfig


class AutoencoderConfig(ModelConfig):
    __title__ = "Autoencoder"
    __tooltip__ = T(
        "Autoencoder: komprimiert Daten in einen kleinen Latentraum und rekonstruiert sie.",
        "Lernt die wesentliche Struktur der Daten ohne Labels.",
        "Anomalieerkennung, Kompression, Entrauschen, Generierung (VAE), Vortraining.",
    )
    variant = Param("ae", choices={
        "ae": "klassischer Autoencoder",
        "vae": "Variational Autoencoder (probabilistischer Latentraum, generativ)",
        "denoising": "lernt aus verrauschter Eingabe das Original zu rekonstruieren",
        "sparse": "erzwingt wenige aktive Latent-Einheiten",
        "vq": "Vector-Quantized VAE: diskreter Latentraum aus einem gelernten Codebuch",
    }, tooltip=T("Autoencoder-Variante.", "Bestimmt Latentraum-Eigenschaften und Regularisierung.",
                 "Anomalien: ae/denoising; Generierung: vae/vq."))
    backbone = Param("mlp", choices={"mlp": "voll verbunden (Vektoren)", "conv": "Faltungen (Bilder/Signale, 2D)"},
                     tooltip=T("Netztyp von Encoder/Decoder.", "conv nutzt räumliche Struktur.", "Bilder: conv."))
    input_dim = Param(32, min=1, tooltip=T("Eingabebreite (mlp).", "Größe der Vektoren.", "Anzahl Features."))
    in_channels = Param(1, min=1, tooltip=T("Kanäle (conv).", "Graustufen 1, RGB 3.", "Bilder."))
    image_size = Param([28, 28], item_type=int, min=1, length=2, tooltip=T(
        "Bildgröße [H, W] (conv).", "Muss durch 2^(Anzahl Kanäle-Stufen) teilbar sein.", "Bilder."))
    hidden_dims = Param([128, 64], item_type=int, min=1, tooltip=T(
        "Breiten (mlp) bzw. Kanalzahlen (conv) des Encoders; der Decoder spiegelt sie.",
        "Kapazität von Encoder/Decoder.", "mlp: [256, 128]; conv: [32, 64].",
    ))
    decoder_dims = Param(None, type=list, item_type=int, min=1, tooltip=T(
        "Eigene Decoder-Breiten (None = gespiegelter Encoder).", "Asymmetrische Autoencoder.", "Experimente."),
        level=Level.ADVANCED)
    latent_dim = Param(8, min=1, tooltip=T(
        "Größe des Latentraums (Engpass).",
        "Kleiner = stärkere Kompression, aber mehr Rekonstruktionsfehler.",
        "2–3 zur Visualisierung, 8–128 für Merkmale.",
    ))
    activation = ComponentParam("activation", "relu", tooltip=T("Aktivierung.", "Nichtlinearität.", "relu/silu/gelu."))
    norm = ComponentParam("normalization", None, tooltip=T("Normalisierung.", "Stabilität.", "Tiefe AEs."),
                          level=Level.ADVANCED)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T("Dropout.", "Regularisierung.", "Kleine Daten."), group="Regularisierung")
    output_activation = ComponentParam("activation", "identity", tooltip=T(
        "Aktivierung der Rekonstruktion.",
        "sigmoid für Daten in [0, 1] (Bilder), tanh für [-1, 1], identity für unbeschränkte Werte.",
        "Passend zum Wertebereich der Daten wählen.",
    ))
    # --- Varianten-spezifisch ---
    beta = Param(1.0, min=0.0, tooltip=T(
        "Gewicht der KL-Divergenz (β-VAE, nur vae).",
        "β > 1 erzwingt entwirrte (disentangled) Latentfaktoren, β < 1 bessere Rekonstruktion.",
        "Disentanglement: 4–10; scharfe Rekonstruktion: 0.1–1.",
    ), group="VAE")
    free_bits = Param(0.0, min=0.0, tooltip=T(
        "Mindest-KL pro Latentdimension ('free bits'); darunter keine Strafe.",
        "Verhindert 'Posterior Collapse' (Latentraum wird ignoriert).",
        "VAEs mit starken Decodern (0.1–0.5).",
    ), level=Level.EXPERT, group="VAE")
    corruption = Param("gaussian", choices={
        "gaussian": "additives Gauß-Rauschen", "masking": "zufällige Features auf 0",
        "salt_pepper": "zufällige Features auf Min/Max",
    }, tooltip=T("Art der Störung (nur denoising).", "Was das Modell 'reparieren' lernt.", "Daten-abhängig."),
        group="Denoising")
    corruption_level = Param(0.2, min=0.0, tooltip=T(
        "Stärke der Störung (Std bzw. Anteil).", "Stärker = robustere Merkmale, schwierigeres Training.", "0.1–0.5."),
        group="Denoising")
    sparsity_type = Param("l1", choices={"l1": "L1 auf Latent-Aktivierungen", "kl": "KL zu Ziel-Aktivierungsrate (sigmoid)"},
                          tooltip=T("Sparsity-Strafe (nur sparse).", "Art der Erzwingung.", "l1 ist einfach."), group="Sparse")
    sparsity_weight = Param(1e-3, min=0.0, tooltip=T(
        "Gewicht der Sparsity-Strafe.", "Stärke der Erzwingung.", "1e-4–1e-2."), group="Sparse")
    sparsity_target = Param(0.05, min=1e-4, max=0.9999, tooltip=T(
        "Ziel-Aktivierungsrate (kl).", "Gewünschter Anteil aktiver Einheiten.", "0.01–0.1."), group="Sparse",
        level=Level.ADVANCED)
    codebook_size = Param(64, min=2, tooltip=T(
        "Anzahl Codebuch-Vektoren (nur vq).", "Größe des diskreten Vokabulars.", "64–1024."), group="VQ")
    commitment_cost = Param(0.25, min=0.0, tooltip=T(
        "Gewicht des Commitment-Verlusts (nur vq).", "Bindet Encoder-Ausgaben an das Codebuch.", "VQ-VAE: 0.25."),
        group="VQ")

    def check(self) -> list[str]:
        problems = []
        if self.backbone == "conv":
            f = 2 ** len(self.hidden_dims)
            if any(s % f for s in self.image_size):
                problems.append(f"image_size muss durch {f} (2^Anzahl Stufen) teilbar sein.")
        return problems


def _mlp(dims: list[int], cfg: AutoencoderConfig, last_plain: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or not last_plain:
            if cfg.norm is not None:
                layers.append(build_norm(cfg.norm, dims[i + 1]))
            layers += [cfg.activation.build(), nn.Dropout(cfg.dropout)]
    return nn.Sequential(*layers)


class Autoencoder(BaseModel):
    """Autoencoder gemäß ``AutoencoderConfig``."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__(config)
        config.validate()
        c = config
        z_out = c.latent_dim * (2 if c.variant == "vae" else 1)
        dec = list(c.decoder_dims) if c.decoder_dims else list(reversed(c.hidden_dims))
        if c.backbone == "mlp":
            self.encoder = _mlp([c.input_dim] + list(c.hidden_dims), c, last_plain=False)
            self.to_latent = nn.Linear(c.hidden_dims[-1] if c.hidden_dims else c.input_dim, z_out)
            self.from_latent = nn.Sequential(nn.Linear(c.latent_dim, dec[0] if dec else c.input_dim), c.activation.build())
            self.decoder = _mlp(dec + [c.input_dim], c, last_plain=True) if dec else nn.Identity()
        else:
            chans = list(c.hidden_dims)
            enc: list[nn.Module] = []
            cin = c.in_channels
            for ch in chans:
                enc += [nn.Conv2d(cin, ch, 4, 2, 1), build_norm(c.norm, ch, 2) if c.norm else nn.Identity(), c.activation.build()]
                cin = ch
            self.encoder = nn.Sequential(*enc)
            self.spatial = [s // 2 ** len(chans) for s in c.image_size]
            flat = chans[-1] * self.spatial[0] * self.spatial[1]
            self.to_latent = nn.Sequential(nn.Flatten(1), nn.Linear(flat, z_out))
            self.from_latent = nn.Sequential(nn.Linear(c.latent_dim, flat), c.activation.build())
            dchans = (dec if c.decoder_dims else list(reversed(chans)))
            decl: list[nn.Module] = []
            for i, ch in enumerate(dchans):
                nxt = dchans[i + 1] if i + 1 < len(dchans) else c.in_channels
                decl.append(nn.ConvTranspose2d(ch, nxt, 4, 2, 1))
                if i + 1 < len(dchans):
                    decl += [build_norm(c.norm, nxt, 2) if c.norm else nn.Identity(), c.activation.build()]
            self.decoder = nn.Sequential(*decl)
            self._dec_first = dchans[0]
        self.out_act = c.output_activation.build()
        if c.variant == "vq":
            self.codebook = nn.Embedding(c.codebook_size, c.latent_dim)
            nn.init.uniform_(self.codebook.weight, -1.0 / c.codebook_size, 1.0 / c.codebook_size)
        self.aux_loss: torch.Tensor | None = None
        self.finish()

    # ------------------------------------------------------------ Teile
    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Störung für den Denoising-AE (nur im Training aktiv)."""
        c = self.config
        if not self.training or c.variant != "denoising":
            return x
        if c.corruption == "gaussian":
            return x + torch.randn_like(x) * c.corruption_level
        mask = torch.rand_like(x) < c.corruption_level
        if c.corruption == "masking":
            return x.masked_fill(mask, 0.0)
        extreme = torch.where(torch.rand_like(x) < 0.5, x.min().expand_as(x), x.max().expand_as(x))
        return torch.where(mask, extreme, x)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.to_latent(self.encoder(x))
        c = self.config
        if c.variant == "vae":
            mu, logvar = h.chunk(2, dim=-1)
            logvar = logvar.clamp(-20, 10)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            return {"latent": z, "mu": mu, "logvar": logvar}
        if c.variant == "vq":
            dist = torch.cdist(h, self.codebook.weight)
            idx = dist.argmin(-1)
            q = self.codebook(idx)
            self.aux_loss = F.mse_loss(q, h.detach()) + c.commitment_cost * F.mse_loss(h, q.detach())
            z = h + (q - h).detach()  # Straight-Through-Schätzer
            return {"latent": z, "codes": idx}
        return {"latent": h}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        if self.config.backbone == "conv":
            h = h.view(z.shape[0], self._dec_first, *self.spatial)
        return self.out_act(self.decoder(h))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.config
        out = self.encode(self.corrupt(x))
        z = out["latent"]
        if c.variant == "sparse":
            if c.sparsity_type == "l1":
                self.aux_loss = c.sparsity_weight * z.abs().mean()
            else:
                rho_hat = torch.sigmoid(z).mean(0).clamp(1e-6, 1 - 1e-6)
                rho = c.sparsity_target
                kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
                self.aux_loss = c.sparsity_weight * kl.sum()
        if c.variant == "vae":
            kl_dim = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())  # [B, D]
            if c.free_bits > 0:
                kl_dim = kl_dim.mean(0).clamp_min(c.free_bits)
                out["kl"] = kl_dim.sum()
            else:
                out["kl"] = kl_dim.sum(-1).mean()
        out["reconstruction"] = self.decode(z)
        return out

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """Neue Daten erzeugen (VAE: aus N(0, I); VQ: zufällige Codes)."""
        c = self.config
        device = next(self.parameters()).device
        if c.variant == "vq":
            z = self.codebook(torch.randint(0, c.codebook_size, (n,), device=device))
        else:
            z = torch.randn(n, c.latent_dim, device=device)
        return self.decode(z)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Rekonstruktionsfehler pro Beispiel (hoch = ungewöhnlich)."""
        was = self.training
        self.eval()
        with torch.no_grad():
            rec = self(x)["reconstruction"]
        self.train(was)
        return (rec - x).pow(2).flatten(1).mean(1)

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        c = self.config
        if c.backbone == "conv":
            return torch.rand(batch_size, c.in_channels, *c.image_size)
        return torch.randn(batch_size, c.input_dim)


MODELS.register("autoencoder", Autoencoder, config=AutoencoderConfig, aliases=("ae", "vae"), tags=("unsupervised",),
                tooltip=T(
    "Autoencoder-Familie (AE, VAE, Denoising, Sparse, VQ-VAE).",
    "Unüberwachtes Lernen kompakter Repräsentationen.",
    "Anomalieerkennung, Kompression, Generierung, Vortraining.",
))
