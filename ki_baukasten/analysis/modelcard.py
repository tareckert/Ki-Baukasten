"""
ki_baukasten.analysis.modelcard
===============================

**Modellkarte & Umgebung** – Dokumentation, die zu jeder fertigen KI gehört (Mitchell et al. 2019,
Hugging-Face-Model-Cards, Dokumentationspflichten z. B. im EU AI Act):

* Zweck, Modell, Daten, Training, aktive Techniken
* Ergebnisse auf Validierung **und Test**, Kalibrierung, Fairness, Robustheit
* Datenprobleme, Grenzen und Hinweise zur Nutzung
* Umgebung (Versionen, Hardware) und Konfigurations-Prüfsumme für Reproduzierbarkeit

``Project.model_card()`` erzeugt sie als Markdown; ``Project.save`` legt ``MODELLKARTE.md`` und
``umgebung.json`` automatisch mit ab (``deployment.model_card``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from typing import Any

import torch


def environment_info() -> dict[str, Any]:
    """Versionen und Hardware – für Reproduzierbarkeit."""
    from .. import __version__

    info: dict[str, Any] = {
        "zeit_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ki_baukasten": __version__, "python": sys.version.split()[0], "torch": torch.__version__,
        "plattform": platform.platform(), "prozessor": platform.machine(), "cpu_kerne": os.cpu_count(),
        "cuda": torch.cuda.is_available(), "mps": bool(getattr(torch.backends, "mps", None)
                                                      and torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    optional = {}
    for name in ("numpy", "PIL", "onnx", "safetensors", "tensorboard", "wandb", "mlflow", "torchao"):
        try:
            mod = __import__(name)
            optional[name] = getattr(mod, "__version__", "vorhanden")
        except Exception:  # noqa: BLE001 - nur Bestandsaufnahme
            continue
    info["zusatzpakete"] = optional
    return info


def config_hash(cfg: dict[str, Any]) -> str:
    """Kurze Prüfsumme der Konfiguration (gleiche Prüfsumme = gleiche Einstellungen)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _table(rows: dict[str, Any]) -> list[str]:
    if not rows:
        return ["(keine)"]
    return ["| Kennzahl | Wert |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in rows.items()]


def _len(data: Any) -> int | None:
    try:
        return len(data) if not isinstance(data, dict) else len(next(iter(data.values())))
    except Exception:  # noqa: BLE001
        return None


def model_card(project: Any) -> str:
    """Modellkarte (Markdown) für ein Projekt – mit allem, was bisher gemessen wurde."""
    cfg = project.config
    g = cfg.goals
    t = cfg.training
    lines = [f"# Modellkarte: {cfg.name}", ""]
    if cfg.description:
        lines += [cfg.description, ""]

    lines += ["## Zweck", ""]
    from .goals import PRIORITIES, TASKS

    lines.append(f"- Aufgabe: {TASKS.get(g.task, g.task)}")
    if g.description:
        lines.append(f"- Beschreibung: {g.description}")
    if g.priorities:
        lines.append("- Wichtig: " + ", ".join(PRIORITIES.get(p, p) for p in g.priorities))
    if g.target or cfg.deployment.main_target:
        lines.append(f"- Zielgerät: {g.target or cfg.deployment.main_target}")
    lines.append("")

    lines += ["## Modell", ""]
    lines.append(f"- Architektur: `{cfg.model.name}` (Anwendungsgebiet {cfg.domain})")
    model = project.model
    if model is not None:
        n = sum(p.numel() for p in model.parameters())
        size = sum(p.numel() * p.element_size() for p in model.parameters()) + \
            sum(b.numel() * b.element_size() for b in model.buffers())
        lines.append(f"- Parameter: {n:,}".replace(",", ".") + f" ({size / 2**20:.2f} MB)")
    lines.append("")

    lines += ["## Daten", ""]
    lines.append(f"- Datensatz: `{cfg.data.name}`" + (f" ({cfg.data.args.path})" if getattr(cfg.data.args, "path", None)
                                                      else ""))
    info = project._info() if hasattr(project, "_info") else {}
    bundle = project._bundle
    if bundle is not None:
        split = project._test_split
        n_train = _len(split[1]) if split is not None else _len(bundle.train)
        n_val = _len(project.trainer.val_data) if project.trainer is not None and project.trainer.val_data is not None \
            else _len(bundle.val)
        if n_train:
            lines.append(f"- Training: {n_train} Beispiele" + (f", Validierung: {n_val}" if n_val else ""))
    if project.test_data is not None:
        lines.append(f"- Testmenge (nie im Training): {_len(project.test_data)} Beispiele")
    if info.get("label_names"):
        lines.append("- Klassen: " + ", ".join(map(str, info["label_names"][:30])))
    if info.get("features"):
        lines.append(f"- Merkmale: {len(info['features'])} (" + ", ".join(info["features"][:12]) +
                     (" …" if len(info["features"]) > 12 else "") + ")")
    report = project.last_data_report
    if report is not None:
        probs = report.problems("warnung")
        lines.append(f"- Datenprüfung: {'keine ernsten Probleme' if not probs else f'{len(probs)} Problem(e)'}")
        lines += [f"  - {f.text}" for f in probs]
    lines.append("")

    lines += ["## Training", ""]
    lines.append(f"- Paradigma: `{t.paradigm.name}`, Optimierer: `{t.optimizer.name}` "
                 f"(lr {getattr(t.optimizer.args, 'lr', '?')}), Zeitplan: `{t.scheduler.name}`")
    lines.append(f"- Umfang: {t.max_steps or t.epochs} {'Schritte' if t.max_steps else 'Epochen'}, Batchgröße "
                 f"{t.batch_size}, Seed {t.seed}")
    try:
        active = [s.technique.title for s in project.techniques(status="an")]
    except Exception:  # noqa: BLE001
        active = []
    if active:
        lines.append("- Aktive Techniken: " + ", ".join(active))
    lines.append("")

    lines += ["## Ergebnisse", ""]
    val = {}
    for h in reversed(project.history):
        if any(str(k).startswith("val_") for k in h):
            val = {k: v for k, v in h.items() if str(k).startswith("val_")}
            break
    lines += ["### Validierung", ""] + _table(val) + [""]
    if project.last_test:
        lines += ["### Test (nie im Training benutzt)", ""] + _table(project.last_test) + [""]
    else:
        lines += ["Kein Test-Ergebnis – `training.test_split` einschalten und `project.test()` aufrufen.", ""]
    if project.last_calibration:
        lines += ["### Kalibrierung", ""] + _table(project.last_calibration) + [""]
    if project.last_fairness is not None:
        f = project.last_fairness
        lines += ["### Fairness", "", f"- Disparate Impact {f.disparate_impact:.3f}, Equalized-Odds-Differenz "
                  f"{f.equalized_odds_difference:.3f} ({'unauffällig' if f.fair else 'prüfen'})", ""]
    if project.last_robustness is not None:
        r = project.last_robustness
        worst = min((v for _, _, v in r.rows), default=r.clean)
        lines += ["### Robustheit", "", f"- {r.metric}: sauber {r.clean:.4g}, schlechtester Fall unter Störung "
                  f"{worst:.4g}", ""]

    ic = cfg.inference
    lines += ["## Nutzung", ""]
    if ic.temperature != 1.0 or ic.conformal_q is not None or ic.threshold is not None:
        lines.append(f"- Vorhersage-Einstellungen: Temperatur {ic.temperature:.3g}"
                     + (f", Schwelle {ic.threshold}" if ic.threshold is not None else "")
                     + (f", Conformal α={ic.conformal_alpha}" if ic.conformal_q is not None else "")
                     + (f", Enthaltung unter {ic.abstain_below}" if ic.abstain_below is not None else ""))
    lines += ["```python", "from ki_baukasten import Project", f"p = Project.load('{cfg.name}')",
              "p.predict_raw(daten)            # Rohdaten (Zeilen, Texte, Bilder, WAV) → lesbare Vorhersagen",
              "p.predict_with_uncertainty(x)   # mit Wahrscheinlichkeit, Unsicherheit, Conformal-Menge", "```",
              "", "Server: `python -m ki_baukasten bereitstellen <ordner>` (HTTP, JSON).", ""]

    lines += ["## Grenzen und Hinweise", ""]
    lines.append("- Das Modell ist nur für Daten verlässlich, die den Trainingsdaten ähneln – im Einsatz regelmäßig "
                 "`project.drift_report(neue_daten)` prüfen.")
    if not project.last_test:
        lines.append("- Ohne Testmenge ist die Qualität nur auf der (leicht geschönten) Validierung gemessen.")
    if not project.last_calibration and t.paradigm.name == "supervised":
        lines.append("- Wahrscheinlichkeiten sind nicht kalibriert (`project.calibrate()`).")
    goal = project.goal_report
    if goal is not None and goal.current_verdict:
        lines.append(f"- Abgleich mit den Zielen: {goal.current_verdict}")
    lines.append("- Bei Entscheidungen über Menschen: Fairness prüfen (`project.fairness_report(gruppen)`) und "
                 "menschliche Kontrolle vorsehen.")
    lines.append("")

    env = environment_info()
    lines += ["## Umgebung und Reproduzierbarkeit", "",
              f"- Erstellt: {env['zeit_utc']} mit ki_baukasten {env['ki_baukasten']}, Python {env['python']}, "
              f"PyTorch {env['torch']}",
              f"- Hardware: {env['plattform']}, {env['cpu_kerne']} CPU-Kerne"
              + (f", GPUs: {', '.join(env.get('gpus', []))}" if env.get("gpus") else ""),
              f"- Konfigurations-Prüfsumme: `{config_hash(cfg.to_dict())}` (projekt.json)", ""]
    return "\n".join(lines)
