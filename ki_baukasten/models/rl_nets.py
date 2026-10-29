"""
ki_baukasten.models.rl_nets
===========================

Netze für Reinforcement Learning: **Actor-Critic** (für REINFORCE, A2C, PPO)
und **Q-Netz** (für DQN-Varianten).

Actor-Critic
------------
* **Actor** (Policy): gibt eine Wahrscheinlichkeitsverteilung über Aktionen aus
  – kategorisch (diskrete Aktionen) oder Gauß (kontinuierliche Aktionen).
* **Critic** (Value): schätzt, wie gut der aktuelle Zustand ist (V(s)).
* Optional geteilter Rumpf (``shared_backbone``) – spart Parameter, kann aber
  zu Zielkonflikten zwischen Policy- und Value-Gradienten führen.

Q-Netz
------
Schätzt für jede diskrete Aktion den erwarteten Gesamtertrag Q(s, a).
Varianten: **Dueling** (V und Vorteil A getrennt) und **Noisy Nets**
(parametrisches Rauschen statt ε-greedy zur Erkundung).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

from ..components.normalization import build_norm
from ..core import ComponentParam, Level, Param, T
from .base import MODELS, BaseModel, ModelConfig


class NoisyLinear(nn.Module):
    """Lineare Schicht mit faktorisiertem, lernbarem Rauschen (Fortunato et al. 2018)."""

    def __init__(self, din: int, dout: int, sigma0: float = 0.5):
        super().__init__()
        self.din, self.dout = din, dout
        bound = 1 / math.sqrt(din)
        self.w_mu = nn.Parameter(torch.empty(dout, din).uniform_(-bound, bound))
        self.w_sigma = nn.Parameter(torch.full((dout, din), sigma0 / math.sqrt(din)))
        self.b_mu = nn.Parameter(torch.empty(dout).uniform_(-bound, bound))
        self.b_sigma = nn.Parameter(torch.full((dout,), sigma0 / math.sqrt(din)))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.linear(x, self.w_mu, self.b_mu)
        e_in = self._f(torch.randn(self.din, device=x.device))
        e_out = self._f(torch.randn(self.dout, device=x.device))
        w = self.w_mu + self.w_sigma * torch.outer(e_out, e_in)
        b = self.b_mu + self.b_sigma * e_out
        return F.linear(x, w, b)


def _trunk(din: int, dims: list[int], act: ComponentParam, norm, noisy: bool = False, sigma0: float = 0.5) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    for h in dims:
        layers.append(NoisyLinear(din, h, sigma0) if noisy else nn.Linear(din, h))
        if norm is not None:
            layers.append(build_norm(norm, h))
        layers.append(act.build())
        din = h
    return nn.Sequential(*layers), din


def _ortho(module: nn.Module, gain: float) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)


class ActorCriticConfig(ModelConfig):
    __title__ = "Actor-Critic"
    __tooltip__ = T(
        "Policy-Netz (Actor) plus Wertschätzer (Critic) für Policy-Gradient-Verfahren.",
        "Actor wählt Aktionen, Critic reduziert die Varianz des Lernsignals.",
        "PPO, A2C, REINFORCE mit Baseline – Robotik, Spiele, Steuerung.",
    )
    obs_dim = Param(4, min=1, tooltip=T("Größe der Beobachtung.", "Eingabebreite.", "Zustandsvektor der Umgebung."))
    action_type = Param("discrete", choices={"discrete": "endlich viele Aktionen", "continuous": "reellwertige Aktionen"},
                        tooltip=T("Art des Aktionsraums.", "Bestimmt die Verteilung (kategorisch vs. Gauß).", "Je nach Umgebung."))
    action_dim = Param(2, min=1, tooltip=T(
        "Anzahl Aktionen (diskret) bzw. Aktionsdimensionen (kontinuierlich).", "Ausgabegröße der Policy.", "Umgebung."))
    hidden_dims = Param([64, 64], item_type=int, min=1, tooltip=T(
        "Breiten des Rumpfs.", "Kapazität.", "PPO-Standard: [64, 64]."))
    activation = ComponentParam("activation", "tanh", tooltip=T(
        "Aktivierung.", "tanh ist in PPO-Implementierungen üblich und stabil.", "tanh oder relu."))
    norm = ComponentParam("normalization", None, tooltip=T("Normalisierung.", "Stabilität.", "Selten nötig."),
                          level=Level.ADVANCED)
    shared_backbone = Param(False, tooltip=T(
        "Actor und Critic teilen den Rumpf.", "Weniger Parameter, aber konkurrierende Gradienten.",
        "Bild-Beobachtungen: an; Vektor: aus."))
    value_hidden_dims = Param(None, type=list, item_type=int, min=1, tooltip=T(
        "Eigene Breiten des Critic-Rumpfs (None = wie hidden_dims).", "Asymmetrische Kapazität.", "Experimente."),
        level=Level.ADVANCED)
    log_std_init = Param(0.0, tooltip=T(
        "Start-Log-Standardabweichung der Gauß-Policy (kontinuierlich).",
        "Bestimmt die anfängliche Erkundung (0 → σ = 1).", "-0.5 bis 0."), level=Level.ADVANCED)
    state_dependent_std = Param(False, tooltip=T(
        "Standardabweichung vom Zustand abhängig machen.", "Situationsabhängige Erkundung (SAC-artig).",
        "Komplexe kontinuierliche Steuerung."), level=Level.EXPERT)
    squash = Param(False, tooltip=T(
        "Aktionen mit tanh auf [-1, 1] begrenzen.", "Für Umgebungen mit begrenzten Aktionen.", "Robotik."),
        level=Level.ADVANCED)
    orthogonal_init = Param(True, tooltip=T(
        "Orthogonale Initialisierung (Rumpf √2, Policy 0.01, Value 1).",
        "Bewährte PPO-Praxis – deutlich stabilerer Start.", "PPO/A2C."), level=Level.ADVANCED)


class ActorCritic(BaseModel):
    """Actor-Critic. ``model(obs)`` → (Verteilungsparameter, Wert)."""

    def __init__(self, config: ActorCriticConfig):
        super().__init__(config)
        c = config
        self.actor_trunk, feat = _trunk(c.obs_dim, list(c.hidden_dims), c.activation, c.norm)
        if c.shared_backbone:
            self.critic_trunk, vfeat = self.actor_trunk, feat
        else:
            self.critic_trunk, vfeat = _trunk(c.obs_dim, list(c.value_hidden_dims or c.hidden_dims), c.activation, c.norm)
        out = c.action_dim * (2 if (c.action_type == "continuous" and c.state_dependent_std) else 1)
        self.policy_head = nn.Linear(feat, out)
        self.value_head = nn.Linear(vfeat, 1)
        if c.action_type == "continuous" and not c.state_dependent_std:
            self.log_std = nn.Parameter(torch.full((c.action_dim,), c.log_std_init))
        self.finish()
        if c.orthogonal_init:
            _ortho(self.actor_trunk, math.sqrt(2))
            if not c.shared_backbone:
                _ortho(self.critic_trunk, math.sqrt(2))
            _ortho(self.policy_head, 0.01)
            _ortho(self.value_head, 1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.policy_head(self.actor_trunk(obs))
        v = self.value_head(self.critic_trunk(obs)).squeeze(-1)
        return a, v

    def distribution(self, obs: torch.Tensor):
        """Aktionsverteilung (``torch.distributions``) und Wert."""
        params, value = self(obs)
        c = self.config
        if c.action_type == "discrete":
            return Categorical(logits=params), value
        if c.state_dependent_std:
            mean, log_std = params.chunk(2, dim=-1)
        else:
            mean, log_std = params, self.log_std.expand_as(params)
        return Independent(Normal(mean, log_std.clamp(-20, 2).exp()), 1), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist, _ = self.distribution(obs)
        if deterministic:
            action = dist.probs.argmax(-1) if self.config.action_type == "discrete" else dist.mean
        else:
            action = dist.sample()
        return torch.tanh(action) if (self.config.squash and self.config.action_type == "continuous") else action

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        return torch.randn(batch_size, self.config.obs_dim)


class QNetworkConfig(ModelConfig):
    __title__ = "Q-Netz"
    __tooltip__ = T(
        "Netz, das für jede diskrete Aktion den erwarteten Ertrag Q(s, a) schätzt.",
        "Grundlage wertbasierter Verfahren (DQN) – sehr dateneffizient durch Replay.",
        "Diskrete Steuerung, Spiele, Empfehlungen.",
    )
    obs_dim = Param(4, min=1, tooltip=T("Größe der Beobachtung.", "Eingabebreite.", "Umgebung."))
    num_actions = Param(2, min=1, tooltip=T("Anzahl diskreter Aktionen.", "Ausgabegröße.", "Umgebung."))
    hidden_dims = Param([128, 128], item_type=int, min=1, tooltip=T("Breiten.", "Kapazität.", "[128, 128]."))
    activation = ComponentParam("activation", "relu", tooltip=T("Aktivierung.", "Nichtlinearität.", "relu."))
    norm = ComponentParam("normalization", None, tooltip=T("Normalisierung.", "Stabilität.", "Selten."),
                          level=Level.ADVANCED)
    dueling = Param(True, tooltip=T(
        "Dueling-Architektur: Q = V(s) + A(s, a) − mean(A).",
        "Lernt den Zustandswert unabhängig von Aktionen – schneller bei vielen ähnlichen Aktionen.",
        "Standard in modernen DQNs (Rainbow)."))
    noisy = Param(False, tooltip=T(
        "Noisy Nets: lernbares Rauschen in den Gewichten statt ε-greedy.",
        "Zustandsabhängige, gelernte Erkundung.", "Rainbow-DQN; dann epsilon auf 0 setzen."), level=Level.ADVANCED)
    noisy_sigma0 = Param(0.5, min=0.0, tooltip=T(
        "Startstärke des Noisy-Net-Rauschens.", "Mehr = mehr Erkundung.", "0.5."), level=Level.EXPERT)


class QNetwork(BaseModel):
    """Q-Netz. ``model(obs)`` → Q-Werte ``[B, num_actions]``."""

    def __init__(self, config: QNetworkConfig):
        super().__init__(config)
        c = config
        self.trunk, feat = _trunk(c.obs_dim, list(c.hidden_dims), c.activation, c.norm, c.noisy, c.noisy_sigma0)
        lin = (lambda i, o: NoisyLinear(i, o, c.noisy_sigma0)) if c.noisy else nn.Linear
        self.adv = lin(feat, c.num_actions)
        self.val = lin(feat, 1) if c.dueling else None
        self.finish()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        a = self.adv(h)
        if self.val is None:
            return a
        return self.val(h) + a - a.mean(-1, keepdim=True)

    def example_input(self, batch_size: int = 2) -> torch.Tensor:
        return torch.randn(batch_size, self.config.obs_dim)


MODELS.register("actor_critic", ActorCritic, config=ActorCriticConfig, aliases=("policy", "ppo_net"), tags=("rl",),
                tooltip=T("Actor-Critic-Netz (Policy + Value).", "Für Policy-Gradient-Verfahren.", "PPO, A2C, REINFORCE."))
MODELS.register("q_network", QNetwork, config=QNetworkConfig, aliases=("dqn_net",), tags=("rl",),
                tooltip=T("Q-Netz (optional Dueling/Noisy).", "Für wertbasierte Verfahren.", "DQN und Varianten."))
