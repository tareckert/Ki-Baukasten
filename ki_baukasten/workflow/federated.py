"""
ki_baukasten.workflow.federated
===============================

**Föderiertes Lernen** (FedAvg / FedProx) und **Model Soups** – viele Modelle, ein Ergebnis.

Föderiert::

    für jede Runde:
        ausgewählte Teilnehmer: Kopie des globalen Modells → lokal trainieren (eigene Daten)
        globales Modell = nach Datenmenge gewichteter Mittelwert der lokalen Modelle

Die Daten der Teilnehmer werden hier aus dem Projekt-Datensatz simuliert (gleich verteilt
oder ungleich per Dirichlet-Verteilung über die Klassen). In echten Einsätzen laufen die
lokalen Schritte auf den Geräten; nur Gewichte werden übertragen. Kombinierbar mit DP-SGD
(``training.privacy``) für zusätzlichen Schutz.

Model Soup: mehrere Läufe vom **selben Startpunkt** mit verschiedenen Seeds; die Gewichte werden
gemittelt (``uniform``) oder nur übernommen, wenn sie die Validierung verbessern (``greedy``).
"""

from __future__ import annotations

import copy
import random
from typing import Any

import torch
from torch import nn

from ..core import ConfigError


def average_state_dicts(states: list[dict[str, torch.Tensor]], weights: list[float] | None = None
                        ) -> dict[str, torch.Tensor]:
    """Gewichteter Mittelwert von state_dicts (Ganzzahl-Puffer werden vom ersten übernommen)."""
    if not states:
        raise ConfigError("Keine Modelle zum Mitteln.")
    weights = weights or [1.0] * len(states)
    total = float(sum(weights))
    out = {}
    for key, ref in states[0].items():
        if not ref.is_floating_point():
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for st, w in zip(states, weights):
            acc += st[key].float() * (w / total)
        out[key] = acc.to(ref.dtype)
    return out


def split_clients(data: dict[str, Any], n_clients: int, mode: str = "iid", alpha: float = 0.5,
                  seed: int = 0) -> list[dict[str, Any]]:
    """Teilt einen dict-Datensatz auf Teilnehmer auf (``iid`` oder ``nicht_iid`` per Dirichlet)."""
    n = next(v for v in data.values() if isinstance(v, torch.Tensor)).shape[0]
    g = torch.Generator().manual_seed(seed)
    targets = data.get("targets")
    if mode == "nicht_iid" and isinstance(targets, torch.Tensor) and not targets.is_floating_point() \
            and targets.dim() == 1:
        parts: list[list[int]] = [[] for _ in range(n_clients)]
        for c in targets.unique().tolist():
            idx = (targets == c).nonzero().flatten()
            idx = idx[torch.randperm(len(idx), generator=g)]
            share = torch.distributions.Dirichlet(torch.full((n_clients,), alpha)).sample()
            cuts = (share.cumsum(0) * len(idx)).long().tolist()
            start = 0
            for k, end in enumerate(cuts):
                parts[k].extend(idx[start:end].tolist())
                start = end
        indices = [torch.tensor(sorted(p) or [int(torch.randint(0, n, (1,), generator=g))]) for p in parts]
    else:
        perm = torch.randperm(n, generator=g)
        indices = [perm[i::n_clients] for i in range(n_clients)]
    return [{k: (v[idx] if isinstance(v, torch.Tensor) and v.shape[:1] == (n,) else v) for k, v in data.items()}
            for idx in indices]


def federated_training(project: Any, cfg: Any = None) -> list[dict[str, Any]]:
    """FedAvg/FedProx mit simulierten Teilnehmern; das Projekt-Modell wird zum globalen Modell."""
    from ..training import Trainer

    f = cfg if cfg is not None else project.config.workflow.federated
    bundle = project.data()
    if not isinstance(bundle.train, dict):
        raise ConfigError("Föderiertes Lernen braucht Daten als dict von Tensoren.")
    train, val = bundle.train, bundle.val
    if val is None and project.config.training.validation_split > 0:
        # Ohne eigene Validierungsdaten: einen Teil zurückhalten, um das globale Modell zu messen.
        n = next(v for v in train.values() if isinstance(v, torch.Tensor)).shape[0]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(project.config.training.seed or 0))
        n_val = max(1, int(n * project.config.training.validation_split))
        pick = (lambda idx: {k: v[idx] if isinstance(v, torch.Tensor) and v.shape[:1] == (n,) else v
                             for k, v in train.items()})
        train, val = pick(perm[n_val:]), pick(perm[:n_val])
    clients = split_clients(train, f.clients, f.split, f.dirichlet_alpha, project.config.training.seed or 0)
    sizes = [next(v for v in c.values() if isinstance(v, torch.Tensor)).shape[0] for c in clients]
    global_model: nn.Module = project._ensure_model()
    rng = random.Random(project.config.training.seed or 0)
    t = project.config.training.copy()
    t.max_steps = f.local_steps
    t.verbose = False
    t.validation_split = 0.0
    t.callbacks = []
    t.logging.backends = []
    history = []
    for rnd in range(f.rounds):
        chosen = sorted(rng.sample(range(f.clients), max(1, round(f.client_fraction * f.clients))))
        states, weights, losses = [], [], []
        for c in chosen:
            local = copy.deepcopy(global_model)
            tr = Trainer(local, t, project.config.memory)
            if f.algorithm == "fedprox" and f.prox_mu > 0:
                anchors = [p.detach().clone() for p in tr.model.parameters()]
                params = list(tr.model.parameters())
                tr.penalties.append(lambda params=params, anchors=anchors: 0.5 * f.prox_mu * sum(
                    (p - a).square().sum() for p, a in zip(params, anchors) if p.requires_grad))
            hist = tr.fit(clients[c])
            states.append({k: v.detach() for k, v in tr.module.state_dict().items()})
            weights.append(float(sizes[c]))
            if hist:
                losses.append(hist[-1].get("loss", float("nan")))
        global_model.load_state_dict(average_state_dicts(states, weights))
        row: dict[str, Any] = {"runde": rnd + 1, "teilnehmer": chosen,
                               "lokaler_verlust": sum(losses) / max(1, len(losses))}
        if val is not None:
            evaluator = Trainer(global_model, t, project.config.memory)
            row.update(evaluator.evaluate(val))
        history.append(row)
        if project._verbose():
            print(f"  Föderiert Runde {rnd + 1}/{f.rounds}: {len(chosen)} Teilnehmer, " +
                  "  ".join(f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)), flush=True)
    project.model = global_model
    project.trainer = None
    project.history = project.history + history
    return history


def model_soup(project: Any, runs: int = 4, mode: str = "greedy") -> dict[str, Any]:
    """Mehrere Läufe vom selben Startpunkt trainieren und zu einer 'Suppe' mitteln."""
    from ..project import Project

    base = project._ensure_model()
    start = copy.deepcopy(base.state_dict())
    results = []
    seed0 = project.config.training.seed or 0
    for i in range(runs):
        cfg = project.config.copy()
        cfg.training.seed = seed0 + i
        cfg.deployment.show_after_training = False
        cfg.training.verbose = False
        cfg.workflow.soup_runs = 0
        q = Project(cfg)
        q._bundle = project._bundle
        q.model = copy.deepcopy(base)
        q.model.load_state_dict(start)
        q.train()
        score = q.evaluate().get("val_loss", float("inf")) if q.trainer is not None else float("inf")
        results.append((score, {k: v.detach().clone() for k, v in q.model.state_dict().items()}, q))
        if project._verbose():
            print(f"  Soup-Lauf {i + 1}/{runs}: val_loss={score:.4g}", flush=True)
    results.sort(key=lambda r: r[0])
    if mode == "uniform":
        chosen = [r[1] for r in results]
    else:
        chosen = [results[0][1]]
        best = results[0][0]
        for score, state, q in results[1:]:
            trial = average_state_dicts(chosen + [state])
            q.model.load_state_dict(trial)
            new = q.evaluate().get("val_loss", float("inf"))
            if new <= best:
                chosen.append(state)
                best = new
    soup = average_state_dicts(chosen)
    base.load_state_dict(soup)
    project.model = base
    project.trainer = results[0][2].trainer
    project.trainer.model = base
    final = project.evaluate()
    if project._verbose():
        print(f"  Model Soup aus {len(chosen)} von {runs} Läufen: " +
              "  ".join(f"{k}={v:.4g}" for k, v in final.items()), flush=True)
    return {"laeufe": [r[0] for r in results], "zutaten": len(chosen), "ergebnis": final}
