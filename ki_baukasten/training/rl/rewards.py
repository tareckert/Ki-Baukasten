"""
ki_baukasten.training.rl.rewards
================================

**Belohnungssystem** – jede Belohnung, jede Art, jede Gewichtung einstellbar.

Prinzip
-------
Die Gesamtbelohnung eines Schritts ist eine **gewichtete Summe frei
kombinierbarer Belohnungsterme**::

    r_t = Σ_k  w_k(t) · clip(term_k(Übergang))      → (optional) normalisiert → (optional) begrenzt

Jeder Term hat eine **Art** (nur zur Einordnung und für die Protokollierung):

=============  =========================================================
Art            Bedeutung
=============  =========================================================
extrinsisch    Belohnung der Umgebung selbst (Aufgabenziel)
intrinsisch    vom Agenten selbst erzeugt (Neugier, Neuheit)
formend        Shaping: Hinweise auf dem Weg zum Ziel (Potentiale, Abstände)
strafe         unerwünschtes Verhalten (Zeit, Energie, Ruckeln, Kollision)
bonus          erwünschte Ereignisse (Ziel erreicht, Meilensteine)
=============  =========================================================

Jeder Term kann zeitlich gesteuert werden: aktiv nur in einem Schrittfenster
(``start_step``/``end_step``) und mit **Gewichtsverlauf** (``anneal``: konstant,
linear ab- oder ansteigend) – z. B. Neugier am Anfang stark, später aus.

Registry ``reward``
-------------------
Fabrik-Signatur: ``factory(args) -> RewardTerm``. Eigene Terme: von
``RewardTerm`` ableiten, ``__call__(tr) -> Tensor[N]`` implementieren, registrieren.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...core import BaseConfig, Level, Param, T, create_registry

REWARDS = create_registry(
    "reward",
    "Belohnungsterme",
    T(
        "Bausteine der Belohnungsfunktion im Reinforcement Learning.",
        "Die Belohnung definiert, was der Agent lernen soll – kleinste Änderungen ändern das Verhalten.",
        "Jedes RL-Training; frei kombinierbar und gewichtbar.",
    ),
    context_doc="factory(args) -> RewardTerm mit __call__(transition) -> Tensor[N]",
)


@dataclass
class Transition:
    """Ein (gebatchter) Umgebungsschritt für ``N`` parallele Umgebungen."""

    obs: torch.Tensor
    action: torch.Tensor
    next_obs: torch.Tensor
    env_reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    infos: list[dict]
    prev_action: torch.Tensor | None
    step: int
    gamma: float

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


# ===========================================================================
# Basis
# ===========================================================================

class RewardTermConfig(BaseConfig):
    __title__ = "Belohnungsterm"
    weight = Param(1.0, tooltip=T(
        "Gewicht des Terms in der Gesamtbelohnung.", "Balanciert konkurrierende Ziele.",
        "Hauptziel 1; Formung/Strafen meist 0.001–0.1.", "Negative Gewichte kehren den Term um."))
    kind = Param("extrinsisch", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art der Belohnung (Einordnung + getrennte Protokollierung).", "Übersicht, welcher Anteil wofür sorgt.",
        "Analyse von Reward-Hacking."))
    label = Param(None, type=str, tooltip=T("Anzeigename im Protokoll (None = Komponentenname).", "Übersicht.", "Optional."),
                  level=Level.ADVANCED)
    clip_min = Param(None, type=float, tooltip=T("Untergrenze je Schritt.", "Begrenzt Ausreißer.", "Optional."),
                     level=Level.ADVANCED)
    clip_max = Param(None, type=float, tooltip=T("Obergrenze je Schritt.", "Begrenzt Ausreißer.", "Optional."),
                     level=Level.ADVANCED)
    start_step = Param(0, min=0, tooltip=T(
        "Erst ab diesem Umgebungsschritt aktiv.", "Belohnungs-Curriculum.", "Späte Zusatzziele."), level=Level.ADVANCED)
    end_step = Param(None, type=int, min=0, tooltip=T(
        "Nur bis zu diesem Schritt aktiv.", "Hilfsbelohnungen später abschalten (Shaping-Gerüst).", "Optional."),
        level=Level.ADVANCED)
    anneal = Param("constant", choices={"constant": "gleichbleibend", "linear_decay": "linear auf 0 fallend",
                                        "linear_increase": "linear von 0 steigend"}, tooltip=T(
        "Gewichtsverlauf über die Zeit.", "Z. B. Neugier anfangs stark, später schwach.", "intrinsisch: linear_decay."),
        level=Level.ADVANCED)
    anneal_steps = Param(10000, min=1, tooltip=T("Dauer des Gewichtsverlaufs (Schritte).", "Tempo.", "Trainingslänge."),
                         level=Level.ADVANCED)

    def factor(self, step: int) -> float:
        if step < self.start_step or (self.end_step is not None and step > self.end_step):
            return 0.0
        p = min(1.0, (step - self.start_step) / self.anneal_steps)
        if self.anneal == "linear_decay":
            return 1.0 - p
        if self.anneal == "linear_increase":
            return p
        return 1.0


class RewardTerm(nn.Module):
    """Basisklasse. ``nn.Module``, damit lernende Terme (RND) Parameter haben dürfen."""

    def __init__(self, cfg: RewardTermConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, tr: Transition) -> torch.Tensor:  # pragma: no cover - abstrakt
        raise NotImplementedError

    def learn(self, obs: torch.Tensor) -> float | None:
        """Optionales Lernen (z. B. RND-Prädiktor) auf gesammelten Beobachtungen."""
        return None


def _info(tr: Transition, key: str, default: float = 0.0) -> torch.Tensor:
    return torch.tensor([float(i.get(key, default)) for i in tr.infos], dtype=torch.float32)


# ===========================================================================
# Terme
# ===========================================================================

class EnvReward(RewardTerm):
    def forward(self, tr):
        return tr.env_reward.float()


class ConstantConfig(RewardTermConfig):
    kind = Param("strafe", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art.", "Einordnung.", "strafe (Zeitstrafe) oder bonus (Überlebensbonus)."))
    value = Param(-0.01, tooltip=T(
        "Wert je Schritt (negativ = Zeitstrafe, positiv = Überlebensbonus).",
        "Zeitstrafe → schnelle Lösungen; Bonus → möglichst lange 'überleben'.", "-0.01 bis 0.01."))


class Constant(RewardTerm):
    def forward(self, tr):
        return torch.full((tr.env_reward.shape[0],), self.cfg.value)


class ActionPenaltyConfig(RewardTermConfig):
    kind = Param("strafe", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art.", "Einordnung.", "strafe."))
    norm = Param("l2", choices={"l1": "Σ|a|", "l2": "Σa² (Energie)"}, tooltip=T(
        "Maß der Aktionsgröße (kontinuierlich).", "Bestraft Kraft-/Energieaufwand.", "Robotik: l2."))
    costs = Param([], item_type=float, tooltip=T(
        "Kosten je diskreter Aktion (leer = keine).", "Manche Aktionen sind teurer (z. B. 'springen').", "Spiele."))


class ActionPenalty(RewardTerm):
    def forward(self, tr):
        a = tr.action
        if a.is_floating_point():
            a = a.reshape(a.shape[0], -1)
            return -(a.abs().sum(-1) if self.cfg.norm == "l1" else a.square().sum(-1))
        if not self.cfg.costs:
            return torch.zeros(a.shape[0])
        costs = torch.tensor(self.cfg.costs)
        return -costs[a.long().clamp(0, len(costs) - 1)]


class ActionChangePenalty(RewardTerm):
    def forward(self, tr):
        if tr.prev_action is None:
            return torch.zeros(tr.action.shape[0])
        if tr.action.is_floating_point():
            return -(tr.action - tr.prev_action).reshape(tr.action.shape[0], -1).square().sum(-1)
        return -(tr.action != tr.prev_action).float()


class InfoKeyConfig(RewardTermConfig):
    key = Param("distance", tooltip=T(
        "Schlüssel im info-dict der Umgebung (z. B. distance, angle, upright).",
        "Beliebige von der Umgebung gemeldete Größe wird zur Belohnung.", "Je nach Umgebung."))
    transform = Param("identity", choices={"identity": "x", "neg": "-x", "abs": "|x|", "neg_abs": "-|x|",
                                           "square": "x²", "neg_square": "-x²"}, tooltip=T(
        "Umformung des Werts.", "Z. B. -|Winkel| belohnt aufrechte Haltung.", "Je nach Ziel."))


def _transform(x: torch.Tensor, kind: str) -> torch.Tensor:
    return {"identity": x, "neg": -x, "abs": x.abs(), "neg_abs": -x.abs(), "square": x.square(),
            "neg_square": -x.square()}[kind]


class InfoValue(RewardTerm):
    def forward(self, tr):
        return _transform(_info(tr, self.cfg.key), self.cfg.transform)


class PotentialConfig(InfoKeyConfig):
    kind = Param("formend", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art.", "Einordnung.", "formend."))
    transform = Param("neg", choices={"identity": "x", "neg": "-x", "abs": "|x|", "neg_abs": "-|x|",
                                      "square": "x²", "neg_square": "-x²"}, tooltip=T(
        "Potential Φ = transform(info[key]); Standard: -Abstand (näher = besser).", "Richtung der Formung.", "neg."))


class Potential(RewardTerm):
    """Potentialbasierte Formung r = γ·Φ(s') − Φ(s) (Ng et al. 1999) – ändert die optimale Policy nicht.

    Φ wird aus dem info-dict gelesen. Am Episodenanfang ist Φ(s) noch unbekannt
    (das info-dict kommt erst mit dem ersten Schritt) → dort ist die Formung 0.
    Terminale Zustände haben Φ = 0.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prev: torch.Tensor | None = None  # Φ(s) je Umgebung; NaN = unbekannt

    def forward(self, tr):
        phi_next = _transform(_info(tr, self.cfg.key), self.cfg.transform)
        if self.prev is None or self.prev.shape != phi_next.shape:
            self.prev = torch.full_like(phi_next, float("nan"))
        target = tr.gamma * phi_next * (~tr.terminated).float()
        r = torch.where(torch.isnan(self.prev), torch.zeros_like(phi_next), target - self.prev)
        self.prev = torch.where(tr.done, torch.full_like(phi_next, float("nan")), phi_next)
        return r


class EventConfig(RewardTermConfig):
    kind = Param("bonus", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art.", "Einordnung.", "bonus oder strafe."))
    key = Param("reached_goal", tooltip=T(
        "Ereignis-Schlüssel im info-dict (z. B. reached_goal, hit_wall, failed).",
        "Belohnt/bestraft das Eintreten eines Ereignisses.", "Je nach Umgebung."))
    value = Param(1.0, tooltip=T("Belohnung beim Ereignis (negativ = Strafe).", "Stärke.", "±0.1 bis ±10."))


class Event(RewardTerm):
    def forward(self, tr):
        return _info(tr, self.cfg.key).ne(0).float() * self.cfg.value


class TerminalConfig(RewardTermConfig):
    kind = Param("strafe", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art.", "Einordnung.", "strafe (Absturz) oder bonus."))
    on = Param("terminated", choices={"terminated": "echtes Episodenende (Erfolg/Misserfolg)",
                                      "truncated": "Abbruch durch Zeitlimit", "any": "beides"}, tooltip=T(
        "Wann der Term greift.", "Unterscheidet Scheitern von Zeitablauf.", "terminated."))
    value = Param(-1.0, tooltip=T("Wert am Episodenende.", "Z. B. Strafe fürs Umfallen.", "-1 bis -10."))


class Terminal(RewardTerm):
    def forward(self, tr):
        mask = {"terminated": tr.terminated, "truncated": tr.truncated, "any": tr.done}[self.cfg.on]
        return mask.float() * self.cfg.value


class RNDConfig(RewardTermConfig):
    kind = Param("intrinsisch", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art.", "Einordnung.", "intrinsisch."))
    weight = Param(0.1, tooltip=T("Gewicht der Neugier.", "Stärke der Erkundung.", "0.01–0.5."))
    hidden_dim = Param(64, min=1, tooltip=T("Breite der RND-Netze.", "Kapazität.", "64–256."), level=Level.ADVANCED)
    feature_dim = Param(32, min=1, tooltip=T("Ausgabegröße der RND-Netze.", "Merkmalsraum.", "32–128."),
                        level=Level.ADVANCED)
    lr = Param(1e-3, min=0.0, tooltip=T("Lernrate des Prädiktors.", "Wie schnell Neues 'bekannt' wird.", "1e-4–1e-3."),
                level=Level.ADVANCED)
    normalize = Param(True, tooltip=T("Neugier durch laufende Standardabweichung teilen.", "Stabile Skala.", "an."),
                      level=Level.ADVANCED)


class RND(RewardTerm):
    """Random Network Distillation (Burda et al. 2018): Neugier = Vorhersagefehler auf neuen Zuständen."""

    def __init__(self, cfg: RNDConfig, obs_dim: int):
        super().__init__(cfg)

        def net():
            return nn.Sequential(nn.Linear(obs_dim, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, cfg.feature_dim))

        self.target, self.predictor = net(), net()
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=cfg.lr)
        self.stats = RunningMeanStd(())

    def forward(self, tr):
        with torch.no_grad():
            err = (self.predictor(tr.next_obs) - self.target(tr.next_obs)).square().mean(-1)
        if self.cfg.normalize:
            self.stats.update(err)
            err = err / (self.stats.var.sqrt() + 1e-8)
        return err

    def learn(self, obs: torch.Tensor) -> float:
        loss = (self.predictor(obs) - self.target(obs)).square().mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


class CountConfig(RewardTermConfig):
    kind = Param("intrinsisch", choices=["extrinsisch", "intrinsisch", "formend", "strafe", "bonus"], tooltip=T(
        "Art.", "Einordnung.", "intrinsisch."))
    weight = Param(0.1, tooltip=T("Gewicht des Neuheitsbonus.", "Stärke der Erkundung.", "0.01–1."))
    resolution = Param(0.1, min=1e-6, tooltip=T(
        "Rasterweite zum Zählen von Zuständen.", "Gröber = mehr Zustände gelten als gleich.", "Je nach Skala."),
        level=Level.ADVANCED)


class CountBonus(RewardTerm):
    """Zählbasierte Erkundung: Bonus β/√N(s) für selten besuchte (gerasterte) Zustände."""

    def __init__(self, cfg: CountConfig):
        super().__init__(cfg)
        self.counts: dict[tuple, int] = defaultdict(int)

    def forward(self, tr):
        out = []
        for o in tr.next_obs:
            key = tuple((o / self.cfg.resolution).round().long().tolist())
            self.counts[key] += 1
            out.append(1.0 / math.sqrt(self.counts[key]))
        return torch.tensor(out)


R = REWARDS.register
R("env", lambda a, obs_dim=1: EnvReward(a), config=RewardTermConfig, aliases=("umgebung", "extrinsic"), tooltip=T(
    "Belohnung der Umgebung (die eigentliche Aufgabe).", "Basis fast jeder Belohnungsfunktion.", "Immer sinnvoll."))
R("constant", lambda a, obs_dim=1: Constant(a), config=ConstantConfig, aliases=("time_penalty", "zeitstrafe"),
  tooltip=T("Konstanter Wert je Schritt (Zeitstrafe/Überlebensbonus).", "Steuert Eile vs. Ausdauer.",
            "Labyrinthe (Strafe), Balancieren (Bonus)."))
R("action_penalty", lambda a, obs_dim=1: ActionPenalty(a), config=ActionPenaltyConfig, aliases=("energie",), tooltip=T(
    "Strafe für Aktionsgröße/-kosten.", "Energiesparendes, sanftes Verhalten.", "Robotik, Steuerung."))
R("action_change", lambda a, obs_dim=1: ActionChangePenalty(a), config=RewardTermConfig, aliases=("smoothness",),
  level=Level.ADVANCED, tooltip=T("Strafe für Aktionswechsel (Ruckeln).", "Glattere Steuerung.", "Roboter, Fahrzeuge."))
R("info", lambda a, obs_dim=1: InfoValue(a), config=InfoKeyConfig, aliases=("info_value",), tooltip=T(
    "Beliebige Umgebungsgröße (info-dict) als Belohnung.", "Maximale Freiheit ohne eigenen Code.",
    "Abstand, Winkel, Geschwindigkeit ..."))
R("potential", lambda a, obs_dim=1: Potential(a), config=PotentialConfig, aliases=("shaping",), tooltip=T(
    "Potentialbasierte Formung γΦ(s')−Φ(s).", "Hilft beim Lernen, ohne das optimale Verhalten zu verfälschen.",
    "Spärliche Belohnungen (Labyrinthe, Robotik)."))
R("event", lambda a, obs_dim=1: Event(a), config=EventConfig, aliases=("ereignis",), tooltip=T(
    "Bonus/Strafe bei einem Ereignis.", "Belohnt Meilensteine, bestraft Kollisionen.", "Ziel erreicht, Wand berührt."))
R("terminal", lambda a, obs_dim=1: Terminal(a), config=TerminalConfig, tooltip=T(
    "Wert am Episodenende.", "Unterscheidet Scheitern von Zeitablauf.", "Absturzstrafe."))
R("rnd", lambda a, obs_dim=1: RND(a, obs_dim), config=RNDConfig, aliases=("curiosity", "neugier"), level=Level.ADVANCED,
  tooltip=T("Neugier per Random Network Distillation.", "Intrinsische Erkundung bei spärlichen Belohnungen.",
            "Schwierige Erkundungsprobleme (Montezuma-artig)."))
R("count", lambda a, obs_dim=1: CountBonus(a), config=CountConfig, aliases=("novelty", "neuheit"), level=Level.ADVANCED,
  tooltip=T("Zählbasierter Neuheitsbonus.", "Einfache, wirksame Erkundung in kleinen Zustandsräumen.", "Gitterwelten."))


# ===========================================================================
# Normalisierung & Gesamtsystem
# ===========================================================================

class RunningMeanStd:
    """Laufender Mittelwert/Varianz (Chan et al.) – für Beobachtungs-/Belohnungsnormalisierung."""

    def __init__(self, shape: tuple[int, ...]):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1, *self.mean.shape)
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta.square() * self.count * n / tot) / tot
        self.count = tot

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + 1e-8)).clamp(-clip, clip)


class RewardSystem:
    """Kombiniert alle Terme, normalisiert, begrenzt und protokolliert je Term."""

    def __init__(self, specs: list[Any], obs_dim: int, num_envs: int, gamma: float, normalization: str,
                 clip: float | None):
        self.terms: list[tuple[str, RewardTerm]] = []
        names: dict[str, int] = defaultdict(int)
        for spec in specs:
            term = spec.build(obs_dim=obs_dim)
            base = spec.args.label or spec.name
            names[base] += 1
            self.terms.append((base if names[base] == 1 else f"{base}_{names[base]}", term))
        self.gamma, self.normalization, self.clip = gamma, normalization, clip
        self.rms = RunningMeanStd(())
        self.ret = torch.zeros(num_envs)
        self.episode = defaultdict(lambda: torch.zeros(num_envs))
        self.finished: dict[str, list[float]] = defaultdict(list)

    def __call__(self, tr: Transition) -> torch.Tensor:
        total = torch.zeros(tr.env_reward.shape[0])
        for name, term in self.terms:
            c = term.cfg
            f = c.factor(tr.step)
            if f == 0.0 and not isinstance(term, Potential):
                continue
            r = term(tr).float()
            if c.clip_min is not None or c.clip_max is not None:
                r = r.clamp(c.clip_min, c.clip_max)
            r = r * c.weight * f
            self.episode[name] += r
            total = total + r
        raw_total = total.clone()
        self.episode["gesamt"] += raw_total
        done = tr.done
        if done.any():
            for name, acc in self.episode.items():
                self.finished[name] += acc[done].tolist()
                acc[done] = 0.0
        if self.normalization == "return":
            self.ret = self.ret * self.gamma + total
            self.rms.update(self.ret)
            self.ret[done] = 0.0
            total = total / (self.rms.var.sqrt() + 1e-8)
        elif self.normalization == "running":
            self.rms.update(total)
            total = total / (self.rms.var.sqrt() + 1e-8)
        if self.clip is not None:
            total = total.clamp(-self.clip, self.clip)
        return total

    def learn(self, obs: torch.Tensor) -> dict[str, float]:
        out = {}
        for name, term in self.terms:
            v = term.learn(obs)
            if v is not None:
                out[f"{name}_loss"] = v
        return out

    def pop_breakdown(self) -> dict[str, float]:
        """Mittlerer Episodenertrag je Term seit dem letzten Aufruf."""
        out = {f"ertrag_{k}": sum(v) / len(v) for k, v in self.finished.items() if v}
        self.finished = defaultdict(list)
        return out
