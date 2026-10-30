"""
ki_baukasten.workflow.search
============================

**Automatische Suche** nach guten Einstellungen – von Zufall bis Schwarmintelligenz.

=============  =====================================================================================
Verfahren      Idee
=============  =====================================================================================
``random``     zufällige Punkte im Suchraum (Bergstra & Bengio 2012: oft besser als Gitter)
``grid``       alle Kombinationen weniger Werte je Einstellung
``evolution``  genetischer Algorithmus: die Besten überleben, werden gekreuzt und mutiert
``pso``        Partikelschwarm: jedes Partikel fliegt zu seinem und dem besten Punkt des Schwarms
``pbt``        Population Based Training: alle trainieren gleichzeitig; Schlechte kopieren Gewichte und
               Einstellungen der Guten und verändern sie leicht – die Einstellungen ändern sich *während*
               des Trainings (nur Trainings-Einstellungen, keine Architektur)
=============  =====================================================================================

Jede Dimension wird intern auf ``[0, 1]`` abgebildet (Zahlenbereiche linear oder logarithmisch,
Auswahllisten über den Index) – so arbeiten Evolution und Schwarm mit beliebigen Einstellungen.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..core import ConfigError, format_number


# ===========================================================================
# Suchraum
# ===========================================================================

@dataclass
class Dim:
    path: str
    choices: list | None = None
    low: float = 0.0
    high: float = 1.0
    log: bool = False
    integer: bool = False

    def value(self, u: float) -> Any:
        u = min(1.0, max(0.0, u))
        if self.choices is not None:
            return copy.deepcopy(self.choices[min(len(self.choices) - 1, int(u * len(self.choices)))])
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return int(round(v)) if self.integer else float(f"{v:.4g}")

    def unit(self, value: Any) -> float:
        """Umkehrung von ``value`` (für Startpunkte aus der aktuellen Konfiguration)."""
        if self.choices is not None:
            idx = self.choices.index(value) if value in self.choices else 0
            return (idx + 0.5) / len(self.choices)
        v = float(value)
        if self.log:
            return (math.log(max(v, 1e-30)) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (v - self.low) / (self.high - self.low) if self.high > self.low else 0.5

    def grid(self, n: int = 3) -> list[Any]:
        if self.choices is not None:
            return [copy.deepcopy(c) for c in self.choices]
        return list(dict.fromkeys(self.value(i / (n - 1)) for i in range(n)))


def parse_space(space: dict[str, Any]) -> list[Dim]:
    dims = []
    for path, spec in space.items():
        if isinstance(spec, (list, tuple)):
            dims.append(Dim(path, choices=list(spec)))
        elif isinstance(spec, dict) and "min" in spec and "max" in spec:
            low, high = float(spec["min"]), float(spec["max"])
            if spec.get("log") and low <= 0:
                raise ConfigError(f"Suchraum '{path}': log-Bereich braucht min > 0.", "workflow.search.space")
            dims.append(Dim(path, low=low, high=high, log=bool(spec.get("log")),
                            integer=spec.get("type") == "int"))
        else:
            raise ConfigError(f"Suchraum '{path}': erwartet Liste von Werten oder {{'min', 'max'}}.",
                              "workflow.search.space")
    return dims


def default_space(project: Any) -> dict[str, Any]:
    """Sinnvoller Standard-Suchraum: Lernrate, Weight-Decay, Dropout, Batchgröße (sofern vorhanden)."""
    cfg = project.config
    space: dict[str, Any] = {}
    try:
        lr = float(cfg.get("training.optimizer.lr"))
        space["training.optimizer.lr"] = {"min": lr / 10, "max": min(1.0, lr * 10), "log": True}
    except Exception:  # noqa: BLE001
        pass
    if cfg.find("training.optimizer.weight_decay"):
        space["training.optimizer.weight_decay"] = [0.0, 0.01, 0.1]
    drop = [p for p in cfg.find("model.**.dropout")]
    if drop:
        space[drop[0]] = [0.0, 0.1, 0.2]
    bs = cfg.training.batch_size
    space["training.batch_size"] = sorted({max(1, bs // 2), bs, bs * 2})
    return space


# ===========================================================================
# Bewertung eines Versuchs
# ===========================================================================

@dataclass
class Trial:
    params: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    generation: int = 0


@dataclass
class SearchResult:
    method: str
    metric: str
    mode: str
    trials: list[Trial] = field(default_factory=list)
    pbt_best: Any = field(default=None, repr=False)     # PBT: behaltenes Mitglied (Projekt + Einstellungen)

    @property
    def best(self) -> Trial:
        return min(self.trials, key=lambda t: t.score)

    @property
    def best_params(self) -> dict[str, Any]:
        return self.best.params

    def best_value(self) -> float:
        b = self.best.score
        return -b if self.mode == "max" else b

    def __str__(self) -> str:
        rows = sorted(self.trials, key=lambda t: t.score)
        lines = [f"Suche '{self.method}': {len(self.trials)} Versuche, Ziel {self.metric} ({self.mode})",
                 f"  Bestes Ergebnis: {self.metric} = {format_number(self.best_value(), 4)}"]
        for t in rows[:8]:
            val = -t.score if self.mode == "max" else t.score
            shown = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in t.params.items())
            lines.append(f"    {format_number(val, 4):>10}  (Gen. {t.generation})  {shown}")
        lines.append("  Übernehmen: projekt.update(ergebnis.best_params) (bei apply_best automatisch)")
        return "\n".join(lines)


def _score(metrics: dict[str, float], metric: str, mode: str, history: list[dict]) -> float:
    value = metrics.get(metric)
    if value is None:
        for row in reversed(history):
            if metric in row:
                value = row[metric]
                break
            if "loss" in row and metric.endswith("loss"):
                value = row["loss"]
                break
    if value is None or not math.isfinite(float(value)):
        return math.inf
    return -float(value) if mode == "max" else float(value)


def _trial_project(project: Any, params: dict[str, Any], steps: int, seed: int) -> Any:
    from ..project import Project

    cfg = project.config.copy()
    for path, value in params.items():
        cfg.set(path, value)
    cfg.training.max_steps = steps
    cfg.training.verbose = False
    cfg.training.seed = seed
    cfg.deployment.show_after_training = False
    cfg.workflow.search.method = "none"
    q = Project(cfg)
    q._bundle = project._bundle
    return q


def evaluate_params(project: Any, params: dict[str, Any], steps: int, metric: str, mode: str,
                    seed: int = 0) -> Trial:
    q = _trial_project(project, params, steps, seed)
    try:
        hist = q.train()
        metrics = q.evaluate() if q.trainer is not None else {}
    except Exception as exc:  # noqa: BLE001 - ungültige Kombination = schlechtester Wert
        return Trial(params, math.inf, {"fehler": str(exc)[:200]})  # type: ignore[dict-item]
    return Trial(params, _score(metrics, metric, mode, hist), metrics)


# ===========================================================================
# Verfahren
# ===========================================================================

def run_search(project: Any, cfg: Any = None, progress: bool = True) -> SearchResult:
    """Führt die in ``cfg`` (``SearchConfig``, Standard: ``workflow.search``) beschriebene Suche aus."""
    s = cfg if cfg is not None else project.config.workflow.search
    if s.mode == "auto":
        from ..training.metrics import direction

        s = s.copy()
        s.mode = direction(s.metric)
    method = s.method if s.method != "none" else "random"
    dims = parse_space(s.space or default_space(project))
    if not dims:
        raise ConfigError("Leerer Suchraum.", "workflow.search.space")
    rng = random.Random(s.seed)
    result = SearchResult(method, s.metric, s.mode)
    project.data()                       # Daten einmal laden, alle Versuche teilen sie

    def say(text: str) -> None:
        if progress and project._verbose():
            print(text, flush=True)

    def evaluate(units: list[float], generation: int) -> Trial:
        params = {d.path: d.value(u) for d, u in zip(dims, units)}
        t = evaluate_params(project, params, s.steps_per_trial, s.metric, s.mode, s.seed + len(result.trials))
        t.generation = generation
        result.trials.append(t)
        val = -t.score if s.mode == "max" else t.score
        say(f"  Versuch {len(result.trials):>3} (Gen. {generation}): {s.metric}={val:.4g}  " +
            ", ".join(f"{k.split('.')[-1]}={v}" for k, v in params.items()))
        return t

    start = [d.unit(_current(project, d.path)) for d in dims]
    if method == "random":
        evaluate(start, 0)
        for _ in range(s.trials - 1):
            evaluate([rng.random() for _ in dims], 0)
    elif method == "grid":
        grids = [d.grid() for d in dims]
        for combo in itertools.islice(itertools.product(*grids), s.trials):
            evaluate([d.unit(v) for d, v in zip(dims, combo)], 0)
    elif method == "evolution":
        _evolution(dims, s, rng, evaluate, start)
    elif method == "pso":
        _pso(dims, s, rng, evaluate, start)
    elif method == "pbt":
        _pbt(project, dims, s, rng, result, say)
    else:
        raise ConfigError(f"Unbekanntes Suchverfahren '{method}'.", "workflow.search.method")
    best_pbt = result.pbt_best
    if s.apply_best and best_pbt is not None:
        # PBT hat schon trainiert: Modell behalten (ein update() würde es verwerfen) und genau seine
        # Einstellungen übernehmen – nicht die eines früheren Durchgangs.
        trained = best_pbt["project"]
        project.config.update(best_pbt["params"])
        project.model, project.trainer = trained.model, trained.trainer
        project.history = list(trained.history)
        if hasattr(project, "_structure_signature"):
            project._built_from = project._structure_signature()
        say(f"  ✓ Bestes PBT-Modell samt Einstellungen übernommen: {best_pbt['params']}")
    elif s.apply_best and result.trials and math.isfinite(result.best.score):
        project.update(result.best_params)
        say(f"  ✓ Beste Einstellungen übernommen: {result.best_params}")
    return result


def _current(project: Any, path: str) -> Any:
    try:
        return project.config.get(path)
    except Exception:  # noqa: BLE001
        return None


def _evolution(dims, s, rng, evaluate, start) -> None:
    """Genetischer Algorithmus mit Elitismus, Turnier-Selektion, gleichmäßiger Kreuzung und Gauß-Mutation."""
    pop = [start] + [[rng.random() for _ in dims] for _ in range(s.population - 1)]
    scored = [(evaluate(ind, 0).score, ind) for ind in pop]
    for gen in range(1, s.generations):
        scored.sort(key=lambda x: x[0])
        nxt = [scored[0][1]]                                      # Elitismus: Bester überlebt unverändert

        def pick():
            a, b = rng.sample(scored, 2) if len(scored) > 1 else (scored[0], scored[0])
            return a[1] if a[0] <= b[0] else b[1]
        while len(nxt) < s.population:
            mother, father = pick(), pick()
            child = [m if rng.random() < 0.5 else f for m, f in zip(mother, father)]
            child = [min(1.0, max(0.0, c + rng.gauss(0, 0.15))) if rng.random() < 0.4 else c for c in child]
            nxt.append(child)
        scored = [scored[0]] + [(evaluate(ind, gen).score, ind) for ind in nxt[1:]]


def _pso(dims, s, rng, evaluate, start) -> None:
    """Partikelschwarm-Optimierung (Trägheit 0.7, kognitiv 1.5, sozial 1.5) im Einheitswürfel."""
    n = len(dims)
    pos = [start] + [[rng.random() for _ in range(n)] for _ in range(s.population - 1)]
    vel = [[rng.uniform(-0.2, 0.2) for _ in range(n)] for _ in pos]
    scores = [evaluate(p, 0).score for p in pos]
    best_pos = [list(p) for p in pos]
    best_score = list(scores)
    g = min(range(len(pos)), key=lambda i: best_score[i])
    for gen in range(1, s.generations):
        for i in range(len(pos)):
            for d in range(n):
                r1, r2 = rng.random(), rng.random()
                vel[i][d] = (0.7 * vel[i][d] + 1.5 * r1 * (best_pos[i][d] - pos[i][d])
                             + 1.5 * r2 * (best_pos[g][d] - pos[i][d]))
                vel[i][d] = max(-0.5, min(0.5, vel[i][d]))
                pos[i][d] = min(1.0, max(0.0, pos[i][d] + vel[i][d]))
            sc = evaluate(pos[i], gen).score
            if sc < best_score[i]:
                best_score[i], best_pos[i] = sc, list(pos[i])
        g = min(range(len(pos)), key=lambda i: best_score[i])


def _pbt(project, dims, s, rng, result: SearchResult, say) -> None:
    """Population Based Training: Exploit (Gewichte der Besten kopieren) + Explore (Einstellungen variieren)."""
    tdims = [d for d in dims if d.path.startswith("training.") and d.path != "training.batch_size"]
    if not tdims:
        raise ConfigError("PBT verändert Einstellungen während des Trainings – der Suchraum braucht mindestens eine "
                          "'training.*'-Einstellung (z. B. training.optimizer.lr).", "workflow.search.space")
    members = []
    for i in range(s.population):
        params = {d.path: d.value(rng.random() if i else d.unit(_current(project, d.path))) for d in tdims}
        q = _trial_project(project, params, s.steps_per_trial, s.seed + i)
        q.build()
        members.append({"project": q, "params": params, "score": math.inf})
    for gen in range(s.generations):
        for m in members:
            q = m["project"]
            hist = q.train()
            metrics = q.evaluate()
            m["score"] = _score(metrics, s.metric, s.mode, hist)
            result.trials.append(Trial(dict(m["params"]), m["score"], metrics, gen))
        members.sort(key=lambda m: m["score"])
        best = members[0]
        say(f"  PBT Runde {gen + 1}/{s.generations}: bestes {s.metric}="
            f"{(-best['score'] if s.mode == 'max' else best['score']):.4g}  {best['params']}")
        if gen == s.generations - 1:
            break
        k = max(1, len(members) // 4)
        for loser, winner in zip(members[-k:], members[:k]):
            loser["project"].model.load_state_dict(winner["project"].model.state_dict())   # Exploit
            params = {}
            for d in tdims:                                                                # Explore
                u = d.unit(winner["params"][d.path]) + rng.choice([-0.1, 0.1])
                params[d.path] = d.value(u)
            loser["params"] = params
            for path, value in params.items():
                loser["project"].config.set(path, value)      # direkt: Modell bleibt erhalten
    result.pbt_best = members[0]          # Modell + genau die Einstellungen, mit denen es trainiert wurde
