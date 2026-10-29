"""
ki_baukasten.training.rl
========================

Reinforcement Learning: Umgebungen, frei kombinierbare Belohnungen,
Algorithmen (PPO, A2C, REINFORCE, DQN) – eingebunden als Trainings-Paradigma
``reinforcement``. Damit gelten alle allgemeinen Trainer-Einstellungen
(Optimierer, Zeitpläne, Clipping, Callbacks, Checkpoints ...) auch für RL.

Beispiel-Konfiguration (Ausschnitt)::

    "training": {
      "paradigm": {"name": "reinforcement", "args": {
        "algorithm": {"name": "ppo", "args": {"n_steps": 256, "entropy_coef": 0.01}},
        "environment": {"name": "gridworld", "args": {"size": 6, "slip_prob": 0.1}},
        "rewards": [
          {"name": "env"},
          {"name": "constant",  "args": {"value": -0.01}},
          {"name": "potential", "args": {"key": "distance", "weight": 0.5}},
          {"name": "event",     "args": {"key": "hit_wall", "value": -0.1, "kind": "strafe"}},
          {"name": "count",     "args": {"weight": 0.05, "anneal": "linear_decay"}}
        ]
      }}
    }
"""

from ...core import BaseConfig, ComponentListParam, ComponentParam, Level, Param, T
from ..paradigms import PARADIGMS, Paradigm
from .algorithms import RL_ALGORITHMS, RLContext
from .environments import ENVIRONMENTS, Env, VecEnv
from .rewards import REWARDS, RewardSystem, RewardTerm, RewardTermConfig, RunningMeanStd, Transition


class RLConfig(BaseConfig):
    __title__ = "Reinforcement Learning"
    __tooltip__ = T(
        "Einstellungen des Reinforcement Learnings: Algorithmus, Umgebung, Belohnungen.",
        "Der Agent lernt durch Ausprobieren, welches Verhalten viel Belohnung bringt.",
        "Steuerung, Robotik, Spiele, Optimierungsprobleme, Empfehlungssysteme.",
    )
    algorithm = ComponentParam("rl_algorithm", "ppo", tooltip=T(
        "RL-Algorithmus.", "PPO (robust), A2C (schnell), REINFORCE (einfach), DQN (dateneffizient, diskret).",
        "Standard: ppo; diskret mit wenig Daten: dqn."))
    environment = ComponentParam("environment", "cartpole", tooltip=T(
        "Umgebung, in der der Agent lernt.", "Definiert Aufgabe, Beobachtungen und Aktionen.", "Je nach Problem."))
    num_envs = Param(4, min=1, tooltip=T(
        "Parallele Umgebungen.", "Mehr = vielfältigere, dekorrelierte Erfahrungen pro Update.", "4–16."))
    total_timesteps = Param(20_000, min=1, tooltip=T(
        "Gesamtzahl Umgebungsschritte (über alle Umgebungen).", "Trainingsbudget.", "CartPole: 2·10⁴–10⁵."))
    gamma = Param(0.99, min=0.0, max=1.0, tooltip=T(
        "Diskontfaktor: wie viel zukünftige Belohnungen zählen.",
        "Nahe 1 = weitsichtig (lange Horizonte), kleiner = kurzsichtig, stabiler.", "0.95–0.999."))
    rewards = ComponentListParam("reward", default=["env"], min_items=1, tooltip=T(
        "Belohnungsterme (gewichtete Summe) – frei kombinierbar.",
        "Die Belohnung bestimmt das gelernte Verhalten.",
        "Basis 'env' + Formung/Strafen/Neugier nach Bedarf.",
    ))
    reward_normalization = Param("none", choices={
        "none": "keine", "running": "durch laufende Std der Belohnung teilen",
        "return": "durch laufende Std des diskontierten Ertrags teilen (wie VecNormalize)",
    }, tooltip=T("Belohnungsnormalisierung.", "Stabile Skala trotz unterschiedlich großer Belohnungen.", "PPO: return."),
        level=Level.ADVANCED)
    reward_clip = Param(None, type=float, min=0.0, tooltip=T(
        "Gesamtbelohnung auf [-c, c] begrenzen.", "Schutz vor Ausreißern (Atari: 1).", "Optional."), level=Level.ADVANCED)
    observation_normalization = Param(False, tooltip=T(
        "Beobachtungen laufend standardisieren.", "Wichtig bei unterschiedlich skalierten Sensoren.", "Robotik: an."),
        level=Level.ADVANCED)
    eval_episodes = Param(5, min=1, tooltip=T("Episoden je Auswertung.", "Genauigkeit der Bewertung.", "5–20."))
    eval_deterministic = Param(True, tooltip=T(
        "Bei der Auswertung immer die wahrscheinlichste Aktion wählen (sonst wie im Training ziehen).",
        "Deterministisch zeigt das 'beste' Verhalten – kann aber in Schleifen hängen bleiben (z. B. an Wänden).",
        "Gitterwelten mit Wänden: aus; Regelungsaufgaben: an."), level=Level.ADVANCED)
    eval_every_env_steps = Param(None, type=int, min=1, tooltip=T(
        "Auswertung alle n Umgebungsschritte (None = nur am Ende).", "Lernkurve / Early Stopping (monitor val_return).",
        "z. B. 5000."))
    log_every_env_steps = Param(2000, min=1, tooltip=T(
        "Protokoll alle n Umgebungsschritte.", "Übersicht.", "1000–10000."))


class Reinforcement(Paradigm):
    """Paradigma 'reinforcement': übergibt die Kontrolle an den gewählten RL-Algorithmus."""

    def run(self, trainer):
        return self.cfg.algorithm.build().run(trainer, self.cfg)


PARADIGMS.register("reinforcement", lambda a: Reinforcement(a), config=RLConfig, aliases=("rl",), tags=("rl",),
                   tooltip=T("Reinforcement Learning (PPO, A2C, REINFORCE, DQN) mit frei kombinierbaren Belohnungen.",
                             "Lernen durch Interaktion statt aus festen Daten.", "Steuerung, Spiele, Robotik."))

__all__ = [
    "RLConfig", "Reinforcement", "RL_ALGORITHMS", "RLContext", "ENVIRONMENTS", "Env", "VecEnv",
    "REWARDS", "RewardSystem", "RewardTerm", "RewardTermConfig", "RunningMeanStd", "Transition",
]
