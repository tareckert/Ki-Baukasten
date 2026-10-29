"""
ki_baukasten.training.rl.environments
=====================================

Eingebaute Umgebungen (ohne Zusatzpakete) und ein Adapter für Gymnasium.

Schnittstelle (gym-ähnlich)
---------------------------
* ``reset(seed=None) -> obs``  (1D-``torch.Tensor``, float32)
* ``step(action) -> (obs, reward, terminated, truncated, info)``
* Eigenschaften: ``obs_dim``, ``action_type`` ("discrete"/"continuous"),
  ``action_dim``, ``action_low``/``action_high`` (kontinuierlich)

**Wichtig für frei gestaltbare Belohnungen:** Jede Umgebung legt im ``info``-
dict zusätzliche Größen ab (Abstand zum Ziel, Winkel, Ereignisse wie
``reached_goal``/``hit_wall`` ...). Belohnungsterme (``rewards.py``) können
darauf zugreifen – so lässt sich die Belohnung beliebig umgestalten, ohne
die Umgebung zu ändern.

Registry ``environment``
------------------------
Fabrik-Signatur: ``factory(args) -> Env``.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch

from ...core import BaseConfig, Level, Param, T, create_registry

ENVIRONMENTS = create_registry(
    "environment",
    "RL-Umgebungen",
    T(
        "Welten, in denen ein Agent handelt und Rückmeldungen erhält.",
        "Liefern Beobachtungen, Belohnungen und Zusatzinformationen für Belohnungsterme.",
        "Reinforcement Learning.",
    ),
    context_doc="factory(args) -> Env",
)


class Env:
    """Basisklasse aller Umgebungen."""

    obs_dim: int = 1
    action_type: str = "discrete"
    action_dim: int = 1
    action_low: float = -1.0
    action_high: float = 1.0
    max_steps: int = 200

    def reset(self, seed: int | None = None) -> torch.Tensor:
        raise NotImplementedError

    def step(self, action: Any) -> tuple[torch.Tensor, float, bool, bool, dict]:
        raise NotImplementedError


# ===========================================================================
# GridWorld
# ===========================================================================

class GridWorldConfig(BaseConfig):
    size = Param(5, min=2, max=64, tooltip=T("Kantenlänge des Gitters.", "Größer = schwierigeres Erkunden.", "5–10."))
    wall_density = Param(0.1, min=0.0, max=0.6, tooltip=T(
        "Anteil zufälliger Wandfelder.", "Hindernisse erhöhen die Planungsschwierigkeit.", "0–0.3."))
    slip_prob = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Wahrscheinlichkeit, dass statt der gewählten eine zufällige Aktion ausgeführt wird.",
        "Stochastische Dynamik – der Agent muss robust planen.", "0–0.2."))
    goal_reward = Param(1.0, tooltip=T("Umgebungsbelohnung am Ziel.", "Extrinsisches Ziel-Signal.", "1."))
    obs_mode = Param("coords", choices={"coords": "normierte Koordinaten (Agent, Ziel)", "onehot": "One-Hot der Position"},
                     tooltip=T("Beobachtungsform.", "onehot ist tabellarisch, coords generalisiert.", "coords."))
    max_steps = Param(None, type=int, min=1, tooltip=T(
        "Maximale Schritte pro Episode (None = 4·size²).", "Begrenzt Irrfahrten.", "None."))
    layout_seed = Param(0, tooltip=T("Seed des Wandlayouts.", "Gleiche Welt bei gleichem Seed.", "Reproduzierbarkeit."),
                        level=Level.ADVANCED)


class GridWorld(Env):
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # hoch, runter, links, rechts

    def __init__(self, a: GridWorldConfig):
        self.a = a
        n = a.size
        rng = random.Random(a.layout_seed)
        self.goal = (n - 1, n - 1)
        self.start = (0, 0)
        self.walls = {(i, j) for i in range(n) for j in range(n)
                      if rng.random() < a.wall_density and (i, j) not in (self.start, self.goal)}
        self.obs_dim = n * n if a.obs_mode == "onehot" else 4
        self.action_type, self.action_dim = "discrete", 4
        self.max_steps = a.max_steps or 4 * n * n
        self.rng = random.Random()
        self.pos, self.t = self.start, 0

    def _obs(self) -> torch.Tensor:
        n = self.a.size
        if self.a.obs_mode == "onehot":
            o = torch.zeros(n * n)
            o[self.pos[0] * n + self.pos[1]] = 1.0
            return o
        return torch.tensor([self.pos[0], self.pos[1], self.goal[0], self.goal[1]], dtype=torch.float32) / (n - 1)

    def _dist(self) -> int:
        return abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.pos, self.t = self.start, 0
        return self._obs()

    def step(self, action):
        a = int(action)
        if self.rng.random() < self.a.slip_prob:
            a = self.rng.randrange(4)
        di, dj = self.MOVES[a]
        n = self.a.size
        nxt = (self.pos[0] + di, self.pos[1] + dj)
        hit = not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or nxt in self.walls
        old_dist = self._dist()
        if not hit:
            self.pos = nxt
        self.t += 1
        reached = self.pos == self.goal
        reward = self.a.goal_reward if reached else 0.0
        info = {"distance": float(self._dist()), "distance_delta": float(old_dist - self._dist()),
                "reached_goal": reached, "hit_wall": hit, "position": self.pos}
        return self._obs(), reward, reached, self.t >= self.max_steps, info


# ===========================================================================
# CartPole (klassische Physik, Barto et al. 1983)
# ===========================================================================

class CartPoleConfig(BaseConfig):
    max_steps = Param(200, min=1, tooltip=T("Episodenlänge (Erfolg = so lange balancieren).", "Schwierigkeit.", "200/500."))
    gravity = Param(9.8, min=0.0, tooltip=T("Schwerkraft.", "Physik-Parameter.", "9.8."), level=Level.ADVANCED)
    pole_length = Param(0.5, min=0.05, tooltip=T("Halbe Stablänge.", "Kürzer = schwieriger.", "0.5."), level=Level.ADVANCED)
    force = Param(10.0, min=0.1, tooltip=T("Stoßkraft je Aktion.", "Steuerstärke.", "10."), level=Level.ADVANCED)
    angle_limit_deg = Param(12.0, min=1.0, max=90.0, tooltip=T("Abbruchwinkel in Grad.", "Toleranz.", "12."),
                            level=Level.ADVANCED)
    noise = Param(0.0, min=0.0, tooltip=T("Rauschen auf die Kraft.", "Stochastische Dynamik.", "0–2."), level=Level.ADVANCED)


class CartPole(Env):
    def __init__(self, a: CartPoleConfig):
        self.a = a
        self.obs_dim, self.action_type, self.action_dim = 4, "discrete", 2
        self.max_steps = a.max_steps
        self.rng = random.Random()
        self.state = [0.0] * 4
        self.t = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.state = [self.rng.uniform(-0.05, 0.05) for _ in range(4)]
        self.t = 0
        return torch.tensor(self.state, dtype=torch.float32)

    def step(self, action):
        a = self.a
        x, xd, th, thd = self.state
        f = a.force if int(action) == 1 else -a.force
        if a.noise:
            f += self.rng.gauss(0, a.noise)
        mc, mp, l, dt = 1.0, 0.1, a.pole_length, 0.02
        total = mc + mp
        tmp = (f + mp * l * thd ** 2 * math.sin(th)) / total
        thacc = (a.gravity * math.sin(th) - math.cos(th) * tmp) / (l * (4 / 3 - mp * math.cos(th) ** 2 / total))
        xacc = tmp - mp * l * thacc * math.cos(th) / total
        x, xd = x + dt * xd, xd + dt * xacc
        th, thd = th + dt * thd, thd + dt * thacc
        self.state = [x, xd, th, thd]
        self.t += 1
        fail = abs(x) > 2.4 or abs(th) > math.radians(a.angle_limit_deg)
        info = {"angle": th, "position": x, "angular_velocity": thd, "failed": fail,
                "upright": 1.0 - abs(th) / math.radians(a.angle_limit_deg)}
        return torch.tensor(self.state, dtype=torch.float32), 1.0, fail, self.t >= self.max_steps, info


# ===========================================================================
# Pendulum (kontinuierliche Aktionen)
# ===========================================================================

class PendulumConfig(BaseConfig):
    max_steps = Param(200, min=1, tooltip=T("Episodenlänge.", "Dauer.", "200."))
    max_torque = Param(2.0, min=0.1, tooltip=T("Maximales Drehmoment.", "Aktionsbereich.", "2."), level=Level.ADVANCED)
    gravity = Param(10.0, min=0.0, tooltip=T("Schwerkraft.", "Physik.", "10."), level=Level.ADVANCED)


class Pendulum(Env):
    def __init__(self, a: PendulumConfig):
        self.a = a
        self.obs_dim, self.action_type, self.action_dim = 3, "continuous", 1
        self.action_low, self.action_high = -a.max_torque, a.max_torque
        self.max_steps = a.max_steps
        self.rng = random.Random()
        self.th, self.thd, self.t = 0.0, 0.0, 0

    def _obs(self):
        return torch.tensor([math.cos(self.th), math.sin(self.th), self.thd], dtype=torch.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.th, self.thd, self.t = self.rng.uniform(-math.pi, math.pi), self.rng.uniform(-1, 1), 0
        return self._obs()

    def step(self, action):
        u = float(torch.as_tensor(action).reshape(-1)[0].clamp(-self.a.max_torque, self.a.max_torque))
        norm_th = ((self.th + math.pi) % (2 * math.pi)) - math.pi
        cost = norm_th ** 2 + 0.1 * self.thd ** 2 + 0.001 * u ** 2
        self.thd = max(-8.0, min(8.0, self.thd + (3 * self.a.gravity / 2 * math.sin(self.th) + 3.0 * u) * 0.05))
        self.th += self.thd * 0.05
        self.t += 1
        info = {"angle": norm_th, "torque": u, "upright": math.cos(norm_th)}
        return self._obs(), -cost, False, self.t >= self.max_steps, info


# ===========================================================================
# Mehrarmiger Bandit
# ===========================================================================

class BanditConfig(BaseConfig):
    arms = Param(10, min=2, tooltip=T("Anzahl Arme (Aktionen).", "Mehr = schwierigere Erkundung.", "10."))
    reward_std = Param(1.0, min=0.0, tooltip=T("Rauschen der Belohnungen.", "Unsicherheit.", "1."))
    episode_length = Param(100, min=1, tooltip=T("Ziehungen pro Episode.", "Horizont.", "100."))
    means_seed = Param(0, tooltip=T("Seed der Arm-Mittelwerte.", "Reproduzierbare Aufgabe.", "0."), level=Level.ADVANCED)


class Bandit(Env):
    def __init__(self, a: BanditConfig):
        self.a = a
        g = random.Random(a.means_seed)
        self.means = [g.gauss(0, 1) for _ in range(a.arms)]
        self.best = max(range(a.arms), key=lambda i: self.means[i])
        self.obs_dim, self.action_type, self.action_dim = 1, "discrete", a.arms
        self.max_steps = a.episode_length
        self.rng, self.t = random.Random(), 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.t = 0
        return torch.ones(1)

    def step(self, action):
        a = int(action)
        r = self.rng.gauss(self.means[a], self.a.reward_std)
        self.t += 1
        info = {"optimal": a == self.best, "regret": self.means[self.best] - self.means[a]}
        return torch.ones(1), r, False, self.t >= self.max_steps, info


# ===========================================================================
# Gymnasium-Adapter
# ===========================================================================

class GymnasiumConfig(BaseConfig):
    env_id = Param("CartPole-v1", tooltip=T(
        "Gymnasium-Umgebungs-ID.", "Zugriff auf hunderte Standard-Umgebungen.", "Benötigt 'pip install gymnasium'."))
    kwargs = Param({}, tooltip=T("Zusätzliche Argumente für gymnasium.make.", "Umgebungsoptionen.", "Je nach Umgebung."),
                   level=Level.ADVANCED)


class GymnasiumEnv(Env):
    def __init__(self, a: GymnasiumConfig):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("Für 'gymnasium' bitte installieren: pip install gymnasium") from exc
        self.env = gym.make(a.env_id, **a.kwargs)
        space = self.env.observation_space
        self.obs_dim = int(torch.tensor(space.shape).prod()) if space.shape else 1
        act = self.env.action_space
        if hasattr(act, "n"):
            self.action_type, self.action_dim = "discrete", int(act.n)
        else:
            self.action_type, self.action_dim = "continuous", int(torch.tensor(act.shape).prod())
            self.action_low, self.action_high = float(act.low.min()), float(act.high.max())
        self.max_steps = getattr(self.env.spec, "max_episode_steps", None) or 1000

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return torch.as_tensor(obs, dtype=torch.float32).flatten()

    def step(self, action):
        if self.action_type == "discrete":
            action = int(action)
        else:
            action = torch.as_tensor(action).float().numpy()
        obs, r, term, trunc, info = self.env.step(action)
        return torch.as_tensor(obs, dtype=torch.float32).flatten(), float(r), bool(term), bool(trunc), dict(info)


R = ENVIRONMENTS.register
R("gridworld", lambda a: GridWorld(a), config=GridWorldConfig, aliases=("gitterwelt",), tooltip=T(
    "Gitterwelt: Agent sucht den Weg zum Ziel (Wände, Rutschgefahr).", "Ideal zum Experimentieren mit Belohnungsformung.",
    "Planung, Erkundung, Reward-Shaping-Experimente."))
R("cartpole", lambda a: CartPole(a), config=CartPoleConfig, tooltip=T(
    "Stab auf Wagen balancieren (klassische Regelungsaufgabe).", "Schneller, verlässlicher RL-Test.", "Diskrete Steuerung."))
R("pendulum", lambda a: Pendulum(a), config=PendulumConfig, tooltip=T(
    "Pendel aufschwingen und halten (kontinuierliches Drehmoment).", "Test für kontinuierliche Aktionen.",
    "PPO mit Gauß-Policy."))
R("bandit", lambda a: Bandit(a), config=BanditConfig, tooltip=T(
    "Mehrarmiger Bandit (Erkundung vs. Ausnutzung).", "Einfachste RL-Aufgabe.", "Erkundungsstrategien testen."))
R("gymnasium", lambda a: GymnasiumEnv(a), config=GymnasiumConfig, aliases=("gym",), level=Level.ADVANCED, tooltip=T(
    "Beliebige Gymnasium-Umgebung.", "Zugang zu Standard-Benchmarks.", "Atari, MuJoCo, Box2D (Zusatzpakete)."))


class VecEnv:
    """Mehrere Umgebungen parallel (sequenziell gerechnet) mit automatischem Reset.

    Liefert gestapelte Tensoren und verfolgt Episodenerträge/-längen.
    """

    def __init__(self, envs: list[Env], seed: int | None = None):
        self.envs = envs
        e = envs[0]
        self.obs_dim, self.action_type, self.action_dim = e.obs_dim, e.action_type, e.action_dim
        self.action_low, self.action_high = e.action_low, e.action_high
        self.num = len(envs)
        base = seed if seed is not None else random.randrange(1 << 30)
        self.obs = torch.stack([env.reset(seed=base + i) for i, env in enumerate(envs)])
        self.ep_return = [0.0] * self.num
        self.ep_len = [0] * self.num
        self.finished: list[tuple[float, int]] = []

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        obs, rews, terms, truncs, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            a = actions[i]
            if self.action_type == "continuous":
                a = a.clamp(self.action_low, self.action_high)
            o, r, term, trunc, info = env.step(a)
            self.ep_return[i] += r
            self.ep_len[i] += 1
            if term or trunc:
                info = dict(info, final_obs=o)
                self.finished.append((self.ep_return[i], self.ep_len[i]))
                self.ep_return[i], self.ep_len[i] = 0.0, 0
                o = env.reset()
            obs.append(o)
            rews.append(r)
            terms.append(term)
            truncs.append(trunc)
            infos.append(info)
        self.obs = torch.stack(obs)
        return (self.obs, torch.tensor(rews, dtype=torch.float32), torch.tensor(terms), torch.tensor(truncs), infos)

    def pop_finished(self) -> list[tuple[float, int]]:
        out, self.finished = self.finished, []
        return out
