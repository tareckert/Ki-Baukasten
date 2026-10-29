"""
ki_baukasten.training.rl.algorithms
===================================

RL-Algorithmen: **PPO**, **A2C**, **REINFORCE** (on-policy, Actor-Critic-Netz)
und **DQN** (off-policy, Q-Netz) mit Double-DQN, n-Schritt-Erträgen,
priorisiertem Replay und weichen/harten Ziel-Updates.

Alle Algorithmen nutzen den allgemeinen ``Trainer`` für die Optimierung –
damit gelten auch hier sämtliche Trainer-Einstellungen: Optimierer,
Lernraten-Zeitpläne, Gradient-Clipping, Akkumulation, EMA, Callbacks,
Checkpoints, Tensor-Chirurgie, Präzision ...

Registry ``rl_algorithm``
-------------------------
Fabrik-Signatur: ``factory(args) -> RLAlgorithm`` mit ``run(trainer, rl_cfg)``.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict, deque
from typing import Any

import torch
import torch.nn.functional as F

from ...core import BaseConfig, Level, Param, T, TrainingError, create_registry
from .environments import VecEnv
from .rewards import RewardSystem, RunningMeanStd, Transition

RL_ALGORITHMS = create_registry(
    "rl_algorithm",
    "RL-Algorithmen",
    T(
        "Lernverfahren für Agenten in Umgebungen.",
        "Bestimmen Stabilität, Dateneffizienz und geeignete Aktionsräume.",
        "Reinforcement Learning.",
    ),
    context_doc="factory(args) -> RLAlgorithm mit run(trainer, rl_cfg)",
)


# ===========================================================================
# Gemeinsame Laufzeit
# ===========================================================================

class RLContext:
    """Umgebungen, Belohnungssystem, Normalisierung, Logging – geteilt von allen Algorithmen."""

    def __init__(self, trainer: Any, rl: Any):
        self.trainer, self.rl = trainer, rl
        seed = trainer.cfg.seed
        self.env = VecEnv([rl.environment.build() for _ in range(rl.num_envs)], seed=seed)
        self.eval_env = rl.environment.build()
        self.rewards = RewardSystem(rl.rewards, self.env.obs_dim, rl.num_envs, rl.gamma, rl.reward_normalization,
                                    rl.reward_clip)
        self.obs_rms = RunningMeanStd((self.env.obs_dim,)) if rl.observation_normalization else None
        self.env_steps = 0
        self.prev_action: torch.Tensor | None = None
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.start = time.time()
        self.next_log = rl.log_every_env_steps
        self.next_eval = rl.eval_every_env_steps or 0
        model_cfg = getattr(trainer.model, "config", None)
        want = getattr(model_cfg, "obs_dim", None)
        if want is not None and want != self.env.obs_dim:
            raise TrainingError(f"Modell erwartet obs_dim={want}, Umgebung liefert {self.env.obs_dim}. "
                                f"Tipp: model.args.obs_dim anpassen (die Baukasten-Fassade macht das automatisch).")

    def norm(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.obs_rms is None:
            return obs
        if update:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def step(self, actions: torch.Tensor, obs: torch.Tensor) -> tuple:
        """Umgebungsschritt + Belohnungssystem. Liefert (next_obs, reward, terminated, truncated, infos, final_obs)."""
        next_obs, env_r, term, trunc, infos = self.env.step(actions.cpu())
        final = next_obs.clone()
        for i, info in enumerate(infos):
            if "final_obs" in info:
                final[i] = info["final_obs"]
        tr = Transition(obs, actions.cpu(), final, env_r, term, trunc, infos, self.prev_action, self.env_steps,
                        self.rl.gamma)
        reward = self.rewards(tr)
        self.prev_action = actions.cpu().clone()
        self.env_steps += self.env.num
        return next_obs, reward, term, trunc, infos, final

    def record(self, **values: float) -> None:
        for k, v in values.items():
            if v is not None and math.isfinite(float(v)):
                self.metrics[k].append(float(v))

    def maybe_log(self) -> None:
        t = self.trainer
        finished = self.env.pop_finished()
        for ret, length in finished:
            self.metrics["episoden_ertrag"].append(ret)
            self.metrics["episoden_laenge"].append(length)
        for k, v in self.rewards.pop_breakdown().items():
            self.metrics[k].append(v)
        if self.env_steps >= self.next_log:
            self.next_log += self.rl.log_every_env_steps
            avg = {k: sum(v) / len(v) for k, v in self.metrics.items() if v}
            lr = next(iter(t.optimizers.values())).param_groups[0]["lr"]
            avg["lr"] = lr
            t.log(kind="fortschritt", text=f"  Umgebungsschritte {self.env_steps:>8}/{self.rl.total_timesteps}  " +
                  "  ".join(f"{k}={v:.4g}" for k, v in avg.items()) + f"  ({time.time() - self.start:.0f} s)")
            t.history.append({"env_steps": self.env_steps, "step": t.global_step, **avg})
            t.logger.metrics(t.global_step, {"env_steps": self.env_steps, **avg}, "train")
            for cb in t.callbacks:
                cb.on_step_end(t, avg)
            self.metrics.clear()
        if self.rl.eval_every_env_steps and self.env_steps >= self.next_eval:
            self.next_eval += self.rl.eval_every_env_steps
            self.evaluate()

    @torch.no_grad()
    def act_eval(self, obs: torch.Tensor) -> torch.Tensor:
        model = self.trainer.model
        o = self.norm(obs.unsqueeze(0).to(self.trainer.device), update=False)
        det = self.rl.eval_deterministic
        if hasattr(model, "act"):
            return model.act(o, deterministic=det)[0].cpu()
        q = model(o)
        if not det and torch.rand(()) < 0.05:  # ε-greedy-Auswertung (ε = 0.05)
            return torch.randint(0, q.shape[-1], ())
        return q.argmax(-1)[0].cpu()

    def evaluate(self) -> dict[str, float]:
        """Deterministische Auswertung über ``eval_episodes`` Episoden (Umgebungsbelohnung)."""
        t = self.trainer
        model = t.model
        was = model.training
        model.eval()
        returns = []
        for ep in range(self.rl.eval_episodes):
            obs = self.eval_env.reset(seed=10_000 + ep)
            total, done, steps = 0.0, False, 0
            while not done and steps < 100_000:
                obs, r, term, trunc, _ = self.eval_env.step(self.act_eval(obs))
                total += r
                done = term or trunc
                steps += 1
            returns.append(total)
        model.train(was)
        metrics = {"val_return": sum(returns) / len(returns), "val_return_min": min(returns),
                   "val_return_max": max(returns)}
        t.log("  Auswertung: " + "  ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
        t.history.append({"env_steps": self.env_steps, "step": t.global_step, **metrics})
        for cb in t.callbacks:
            cb.on_eval_end(t, metrics)
        return metrics


class RLAlgorithm:
    needs = "actor_critic"

    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg

    def run(self, trainer: Any, rl: Any) -> list[dict]:
        raise NotImplementedError


# ===========================================================================
# On-Policy: PPO / A2C / REINFORCE
# ===========================================================================

class OnPolicyConfig(BaseConfig):
    n_steps = Param(128, min=1, tooltip=T(
        "Schritte pro Umgebung je Rollout.", "Länger = bessere Schätzungen, seltenere Updates.", "PPO: 128–2048; A2C: 5."))
    value_coef = Param(0.5, min=0.0, tooltip=T(
        "Gewicht des Value-Verlusts.", "Balanciert Kritiker- gegen Policy-Lernen.", "0.5."))
    entropy_coef = Param(0.01, min=0.0, tooltip=T(
        "Entropie-Bonus.", "Fördert Erkundung, verhindert vorzeitig deterministische Policies.", "0–0.05."))
    gae_lambda = Param(0.95, min=0.0, max=1.0, tooltip=T(
        "λ der Generalized Advantage Estimation.", "Bias-Varianz-Abwägung (1 = Monte-Carlo, 0 = TD).", "0.9–0.99."))
    normalize_advantages = Param(True, tooltip=T(
        "Vorteile pro Batch standardisieren.", "Stabilere Gradienten.", "PPO: an."), level=Level.ADVANCED)


class PPOConfig(OnPolicyConfig):
    n_epochs = Param(4, min=1, tooltip=T(
        "Durchläufe über jeden Rollout.", "Mehr = dateneffizienter, aber Gefahr zu großer Policy-Sprünge.", "3–10."))
    minibatch_size = Param(64, min=1, tooltip=T("Minibatch-Größe.", "Gradienten-Rauschen vs. Geschwindigkeit.", "64–512."))
    clip_range = Param(0.2, min=0.0, tooltip=T(
        "PPO-Clipping ε des Wahrscheinlichkeitsverhältnisses.", "Begrenzt Policy-Änderungen pro Update – Kern von PPO.",
        "0.1–0.3."))
    clip_value = Param(None, type=float, min=0.0, tooltip=T(
        "Clipping der Value-Updates (None = aus).", "Stabilisiert den Kritiker.", "0.2 oder None."), level=Level.ADVANCED)
    target_kl = Param(None, type=float, min=0.0, tooltip=T(
        "Abbruch der Epochen, wenn die KL-Divergenz diesen Wert überschreitet.", "Schutz vor Policy-Kollaps.", "0.01–0.05."),
        level=Level.ADVANCED)


class A2CConfig(OnPolicyConfig):
    n_steps = Param(5, min=1, tooltip=T("Schritte pro Umgebung je Update.", "Kurze Rollouts, häufige Updates.", "5–20."))
    gae_lambda = Param(1.0, min=0.0, max=1.0, tooltip=T("λ der GAE.", "1 = n-Schritt-Ertrag.", "1.0."))
    normalize_advantages = Param(False, tooltip=T("Vorteile standardisieren.", "Optional.", "aus."), level=Level.ADVANCED)


class ReinforceConfig(OnPolicyConfig):
    n_steps = Param(200, min=1, tooltip=T("Schritte pro Umgebung je Update.", "Lange Rollouts (Monte-Carlo).", "≥ Episodenlänge."))
    gae_lambda = Param(1.0, min=0.0, max=1.0, tooltip=T("λ (REINFORCE: 1 = reine Monte-Carlo-Erträge).", "MC.", "1.0."),
                       level=Level.EXPERT)
    entropy_coef = Param(0.0, min=0.0, tooltip=T("Entropie-Bonus.", "Erkundung.", "0–0.01."))
    baseline = Param("value", choices={"none": "ohne Baseline (hohe Varianz)", "mean": "Mittelwert der Erträge",
                                       "value": "gelernter Kritiker V(s)"}, tooltip=T(
        "Baseline zur Varianzreduktion.", "Ohne Baseline ist REINFORCE sehr verrauscht.", "value."))


class OnPolicy(RLAlgorithm):
    kind = "ppo"

    def run(self, trainer, rl):
        c = self.cfg
        model = trainer.model
        if not hasattr(model, "distribution"):
            raise TrainingError(f"{type(self).__name__} braucht ein Actor-Critic-Modell (model: actor_critic).")
        ctx = RLContext(trainer, rl)
        n_env, T_ = rl.num_envs, c.n_steps
        n_updates = max(1, math.ceil(rl.total_timesteps / (T_ * n_env)))
        per_update = c.n_epochs * math.ceil(T_ * n_env / c.minibatch_size) if self.kind == "ppo" else 1
        trainer.total_steps = n_updates * per_update
        trainer._build_schedulers(trainer.total_steps)
        trainer.log(f"▶ RL ({self.kind}): {rl.total_timesteps} Umgebungsschritte, {n_env} Umgebungen, "
                    f"{n_updates} Updates, Umgebung '{rl.environment.name}'")
        dev = trainer.device
        obs = ctx.norm(ctx.env.obs)
        for _ in range(n_updates):
            if trainer.should_stop:
                break
            buf = defaultdict(list)
            model.eval()
            with torch.no_grad():
                for _t in range(T_):
                    dist, value = model.distribution(obs.to(dev))
                    action = dist.sample()
                    env_action = torch.tanh(action) * ctx.env.action_high if (
                        ctx.env.action_type == "continuous" and getattr(model.config, "squash", False)) else action
                    nxt, reward, term, trunc, infos, final = ctx.step(env_action, obs)
                    nxt_n = ctx.norm(nxt)
                    boot = torch.zeros(n_env)
                    if trunc.any():  # Zeitlimit: mit V(letzte Beobachtung) bootstrappen
                        _, v_final = model.distribution(ctx.norm(final, update=False).to(dev))
                        boot = torch.where(trunc & ~term, v_final.cpu(), boot)
                    buf["obs"].append(obs)
                    buf["action"].append(action.cpu())
                    buf["logp"].append(dist.log_prob(action).cpu())
                    buf["value"].append(value.cpu())
                    buf["reward"].append(reward)
                    buf["done"].append((term | trunc).float())
                    buf["boot"].append(boot)
                    obs = nxt_n
                    ctx.maybe_log()
                _, last_v = model.distribution(obs.to(dev))
            model.train()
            S = {k: torch.stack(v) for k, v in buf.items()}  # [T, N, ...]
            adv, ret = self._gae(S, last_v.cpu(), rl.gamma, c.gae_lambda)
            learn = ctx.rewards.learn(S["obs"].reshape(-1, S["obs"].shape[-1]))
            ctx.record(**learn)
            self._update(trainer, ctx, S, adv, ret)
        ctx.maybe_log()
        final_metrics = ctx.evaluate()
        for cb in trainer.callbacks:
            cb.on_train_end(trainer)
        trainer.log(f"■ RL beendet: {ctx.env_steps} Umgebungsschritte, val_return={final_metrics['val_return']:.4g}")
        return trainer.history

    @staticmethod
    def _gae(S, last_v, gamma, lam):
        T_ = S["reward"].shape[0]
        adv = torch.zeros_like(S["reward"])
        running = torch.zeros_like(last_v)
        for t in reversed(range(T_)):
            nonterm = 1.0 - S["done"][t]
            next_v = last_v if t == T_ - 1 else S["value"][t + 1]
            next_v = next_v * nonterm + S["boot"][t]
            delta = S["reward"][t] + gamma * next_v - S["value"][t]
            running = delta + gamma * lam * nonterm * running
            adv[t] = running
        return adv, adv + S["value"]

    def _losses(self, model, obs, actions, old_logp, adv, ret, old_v):
        c = self.cfg
        dist, v = model.distribution(obs)
        logp = dist.log_prob(actions)
        ent = dist.entropy().mean()
        if self.kind == "ppo":
            ratio = torch.exp(logp - old_logp)
            pi_loss = -torch.min(ratio * adv, ratio.clamp(1 - c.clip_range, 1 + c.clip_range) * adv).mean()
            if c.clip_value is not None:
                v_clip = old_v + (v - old_v).clamp(-c.clip_value, c.clip_value)
                v_loss = torch.max((v - ret).square(), (v_clip - ret).square()).mean()
            else:
                v_loss = (v - ret).square().mean()
            kl = float((old_logp - logp).mean().detach())
        else:
            pi_loss = -(logp * adv).mean()
            v_loss = (v - ret).square().mean()
            kl = None
        use_v = self.kind != "reinforce" or c.baseline == "value"
        loss = pi_loss + (c.value_coef * v_loss if use_v else 0.0) - c.entropy_coef * ent
        return loss, float(pi_loss.detach()), float(v_loss.detach()), float(ent.detach()), kl

    def _update(self, trainer, ctx, S, adv, ret):
        c = self.cfg
        dev = trainer.device
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        obs, act, logp, val = flat(S["obs"]), flat(S["action"]), flat(S["logp"]), flat(S["value"])
        adv, ret = flat(adv), flat(ret)
        if self.kind == "reinforce":
            if c.baseline == "none":
                adv = ret.clone()
            elif c.baseline == "mean":
                adv = ret - ret.mean()
        if c.normalize_advantages and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs, act, logp, val, adv, ret = (x.to(dev) for x in (obs, act, logp, val, adv, ret))
        if self.kind != "ppo":
            loss, pl, vl, ent, _ = self._losses(trainer.model, obs, act, logp, adv, ret, val)
            trainer.optimize(loss)
            ctx.record(policy_loss=pl, value_loss=vl, entropy=ent)
            return
        n = obs.shape[0]
        for _ in range(c.n_epochs):
            perm = torch.randperm(n, device=dev)
            stop = False
            for i in range(0, n, c.minibatch_size):
                idx = perm[i: i + c.minibatch_size]
                loss, pl, vl, ent, kl = self._losses(trainer.model, obs[idx], act[idx], logp[idx], adv[idx], ret[idx],
                                                     val[idx])
                trainer.optimize(loss)
                ctx.record(policy_loss=pl, value_loss=vl, entropy=ent, approx_kl=kl)
                if c.target_kl is not None and kl is not None and kl > 1.5 * c.target_kl:
                    stop = True
                    break
            if stop:
                break
        trainer.flush()


class PPO(OnPolicy):
    kind = "ppo"


class A2C(OnPolicy):
    kind = "a2c"


class Reinforce(OnPolicy):
    kind = "reinforce"


# ===========================================================================
# Off-Policy: DQN
# ===========================================================================

class DQNConfig(BaseConfig):
    buffer_size = Param(50_000, min=1, tooltip=T(
        "Kapazität des Replay-Speichers.", "Mehr = vielfältigere, ältere Erfahrungen.", "10⁴–10⁶."))
    learning_starts = Param(1000, min=0, tooltip=T(
        "Zufällige Schritte bevor gelernt wird.", "Füllt den Speicher mit vielfältigen Daten.", "1000–50000."))
    batch_size = Param(64, min=1, tooltip=T("Batchgröße der Updates.", "Gradientenqualität.", "32–256."))
    train_freq = Param(4, min=1, tooltip=T("Update alle n Umgebungsschritte (je Umgebung).", "Daten/Update-Verhältnis.", "1–4."))
    gradient_steps = Param(1, min=1, tooltip=T("Updates je Trainingszeitpunkt.", "Mehr = dateneffizienter.", "1."))
    target_update_interval = Param(500, min=1, tooltip=T(
        "Ziel-Netz-Update alle n Gradientenschritte.", "Stabilisiert die Bootstrapping-Ziele.", "500–10000."))
    tau = Param(1.0, min=0.0, max=1.0, tooltip=T(
        "Weiches Update θ' ← τθ + (1-τ)θ' (1 = hartes Kopieren).", "Glattere Ziele.", "1.0 oder 0.005 (jeden Schritt)."),
        level=Level.ADVANCED)
    double = Param(True, tooltip=T(
        "Double-DQN: Aktion mit Online-Netz wählen, mit Ziel-Netz bewerten.", "Reduziert Überschätzung von Q-Werten.",
        "Fast immer an."))
    n_step = Param(1, min=1, tooltip=T(
        "n-Schritt-Erträge.", "Schnellere Belohnungsausbreitung, etwas mehr Varianz.", "1–5 (Rainbow: 3)."),
        level=Level.ADVANCED)
    epsilon_start = Param(1.0, min=0.0, max=1.0, tooltip=T("Start-ε (Zufallsaktionen).", "Anfangs viel erkunden.", "1.0."))
    epsilon_end = Param(0.05, min=0.0, max=1.0, tooltip=T("End-ε.", "Rest-Erkundung.", "0.01–0.05."))
    epsilon_fraction = Param(0.3, min=0.0, max=1.0, tooltip=T(
        "Anteil des Trainings, über den ε linear sinkt.", "Dauer der Erkundungsphase.", "0.1–0.5."))
    loss = Param("huber", choices=["huber", "mse"], tooltip=T(
        "TD-Verlust.", "Huber ist robust gegen große TD-Fehler.", "huber."))
    prioritized = Param(False, tooltip=T(
        "Priorisiertes Replay (große TD-Fehler öfter ziehen).", "Lernt schneller aus überraschenden Erfahrungen.",
        "Rainbow-DQN."), level=Level.ADVANCED)
    alpha = Param(0.6, min=0.0, max=1.0, tooltip=T("Stärke der Priorisierung.", "0 = gleichverteilt.", "0.6."),
                  level=Level.EXPERT)
    beta_start = Param(0.4, min=0.0, max=1.0, tooltip=T(
        "Start der Importance-Sampling-Korrektur (steigt auf 1).", "Korrigiert den Priorisierungs-Bias.", "0.4."),
        level=Level.EXPERT)


class ReplayBuffer:
    """Ring-Speicher mit optionaler Priorisierung."""

    def __init__(self, capacity: int, obs_dim: int, prioritized: bool, alpha: float):
        self.capacity, self.prioritized, self.alpha = capacity, prioritized, alpha
        self.obs = torch.zeros(capacity, obs_dim)
        self.next_obs = torch.zeros(capacity, obs_dim)
        self.action = torch.zeros(capacity, dtype=torch.long)
        self.reward = torch.zeros(capacity)
        self.done = torch.zeros(capacity)
        self.discount = torch.zeros(capacity)
        self.prio = torch.zeros(capacity)
        self.pos, self.size = 0, 0

    def add(self, o, a, r, o2, d, disc):
        i = self.pos
        self.obs[i], self.action[i], self.reward[i], self.next_obs[i], self.done[i], self.discount[i] = o, a, r, o2, d, disc
        self.prio[i] = self.prio[: self.size].max() if self.size else 1.0
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, beta: float):
        if self.prioritized:
            p = self.prio[: self.size].clamp_min(1e-6) ** self.alpha
            p = p / p.sum()
            idx = torch.multinomial(p, n, replacement=True)
            w = (self.size * p[idx]) ** (-beta)
            w = w / w.max()
        else:
            idx = torch.randint(0, self.size, (n,))
            w = torch.ones(n)
        return idx, (self.obs[idx], self.action[idx], self.reward[idx], self.next_obs[idx], self.done[idx],
                     self.discount[idx], w)

    def update_priorities(self, idx: torch.Tensor, td: torch.Tensor) -> None:
        self.prio[idx] = td.detach().abs().cpu() + 1e-6


class DQN(RLAlgorithm):
    needs = "q_network"

    def run(self, trainer, rl):
        c = self.cfg
        model = trainer.model
        ctx = RLContext(trainer, rl)
        if ctx.env.action_type != "discrete":
            raise TrainingError("DQN funktioniert nur mit diskreten Aktionen.")
        dev = trainer.device
        target = copy.deepcopy(model).eval()
        target.requires_grad_(False)
        buf = ReplayBuffer(c.buffer_size, ctx.env.obs_dim, c.prioritized, c.alpha)
        n_env = rl.num_envs
        iters = max(1, math.ceil(rl.total_timesteps / n_env))
        trainer.total_steps = max(1, (iters - c.learning_starts // n_env) // c.train_freq * c.gradient_steps)
        trainer._build_schedulers(trainer.total_steps)
        trainer.log(f"▶ RL (dqn): {rl.total_timesteps} Umgebungsschritte, {n_env} Umgebungen, "
                    f"Umgebung '{rl.environment.name}'")
        pending = [deque() for _ in range(n_env)]  # n-Schritt-Puffer je Umgebung
        obs = ctx.norm(ctx.env.obs)
        grad_steps = 0
        for it in range(iters):
            if trainer.should_stop:
                break
            frac = min(1.0, ctx.env_steps / max(1, c.epsilon_fraction * rl.total_timesteps))
            eps = c.epsilon_start + (c.epsilon_end - c.epsilon_start) * frac
            with torch.no_grad():
                greedy = model(obs.to(dev)).argmax(-1).cpu()
            rand = torch.randint(0, ctx.env.action_dim, (n_env,))
            actions = torch.where(torch.rand(n_env) < eps, rand, greedy)
            nxt, reward, term, trunc, infos, final = ctx.step(actions, obs)
            nxt_n = ctx.norm(nxt)
            final_n = ctx.norm(final, update=False)
            for i in range(n_env):
                pending[i].append((obs[i], actions[i], float(reward[i])))
                done_i = bool(term[i] or trunc[i])
                while pending[i] and (len(pending[i]) >= c.n_step or done_i):
                    R, disc = 0.0, 1.0
                    for (_, _, r) in pending[i]:
                        R += disc * r
                        disc *= rl.gamma
                    o0, a0, _ = pending[i][0]
                    buf.add(o0, a0, R, final_n[i], float(term[i]) if done_i else 0.0, disc)
                    pending[i].popleft()
                    if not done_i:
                        break
                if done_i:
                    pending[i].clear()
            obs = nxt_n
            ctx.record(epsilon=eps)
            if buf.size >= max(c.batch_size, c.learning_starts) and it % c.train_freq == 0:
                beta = c.beta_start + (1 - c.beta_start) * min(1.0, ctx.env_steps / rl.total_timesteps)
                for _ in range(c.gradient_steps):
                    idx, (o, a, r, o2, d, disc, w) = buf.sample(c.batch_size, beta)
                    o, a, r, o2, d, disc, w = (x.to(dev) for x in (o, a, r, o2, d, disc, w))
                    with torch.no_grad():
                        if c.double:
                            a2 = model(o2).argmax(-1, keepdim=True)
                            q2 = target(o2).gather(1, a2).squeeze(1)
                        else:
                            q2 = target(o2).max(-1).values
                        y = r + disc * (1 - d) * q2
                    with trainer.autocast():
                        q = model(o).gather(1, a.unsqueeze(1)).squeeze(1).float()
                        td = y - q
                        per = F.smooth_l1_loss(q, y, reduction="none") if c.loss == "huber" else td.square()
                        loss = (per * w).mean()
                    trainer.optimize(loss)
                    if c.prioritized:
                        buf.update_priorities(idx.cpu(), td)
                    grad_steps += 1
                    ctx.record(td_loss=float(loss.detach()), q_mean=float(q.detach().mean()))
                    if c.tau < 1.0:
                        with torch.no_grad():
                            for pt, p in zip(target.parameters(), model.parameters()):
                                pt.lerp_(p, c.tau)
                    elif grad_steps % c.target_update_interval == 0:
                        target.load_state_dict(model.state_dict())
            ctx.maybe_log()
        final_metrics = ctx.evaluate()
        for cb in trainer.callbacks:
            cb.on_train_end(trainer)
        trainer.log(f"■ RL beendet: {ctx.env_steps} Umgebungsschritte, val_return={final_metrics['val_return']:.4g}")
        return trainer.history


R = RL_ALGORITHMS.register
R("ppo", lambda a: PPO(a), config=PPOConfig, tooltip=T(
    "Proximal Policy Optimization.", "Robust, vielseitig, Standard für viele RL-Aufgaben.",
    "Diskrete & kontinuierliche Steuerung, Robotik, Spiele, RLHF."))
R("a2c", lambda a: A2C(a), config=A2CConfig, tooltip=T(
    "Advantage Actor-Critic.", "Einfach und schnell, weniger stabil als PPO.", "Schnelle Experimente."))
R("reinforce", lambda a: Reinforce(a), config=ReinforceConfig, tooltip=T(
    "REINFORCE (Monte-Carlo-Policy-Gradient).", "Der einfachste Policy-Gradient – lehrreich, aber verrauscht.",
    "Lehre, Bandits, kurze Episoden."))
R("dqn", lambda a: DQN(a), config=DQNConfig, tooltip=T(
    "Deep Q-Network (+ Double, n-Schritt, Priorisierung).", "Sehr dateneffizient durch Replay.",
    "Diskrete Aktionen (Spiele, Gitterwelten)."))
