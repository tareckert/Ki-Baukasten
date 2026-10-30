"""
ki_baukasten.analysis.diagnosis
===============================

**Diagnose nach dem Training**: liest aus Lernkurven, Kennzahlen und Basis-Vergleich, *was* das Modell
bremst – und schlägt *konkrete, direkt anwendbare* Maßnahmen vor.

==========================  =====================================================  ==================================
Befund                      Erkennung                                              Maßnahmen (Auswahl)
==========================  =====================================================  ==================================
instabil                    übersprungene NaN-Schritte, Verlust-Spitzen            Clipping, Warmup, Lernrate ÷ 2
lernt kaum                  Trainingsverlust sinkt um < 3 %                        LR-Finder, mehr Schritte
zu kurz trainiert           Validierung verbessert sich bis zum letzten Messpunkt  Schritte × 2
Überanpassung               Validierung wird nach ihrem Bestwert wieder schlechter Dropout, Weight Decay, Early
                            oder Trainingsverlust ≪ Validierungsverlust            Stopping, Augmentierung, Label
                                                                                   Smoothing, EMA
kaum besser als die Basis   Genauigkeit/R² kaum über "häufigste Klasse/Mittelwert" LR-Finder, länger, Datenprüfung
Klassen unausgeglichen      balancierte Genauigkeit ≪ Genauigkeit                  Klassengewichte, Ausgleich
schlecht kalibriert         ECE > 0,08 ohne Temperature Scaling                    Kalibrierung
keine Testmenge             keine zurückgehaltenen Daten                           Testmenge
Validierung zu optimistisch Test deutlich schlechter als Validierung               Kreuzvalidierung, mehr Daten
==========================  =====================================================  ==================================

``Project.diagnose()`` liefert den Bericht, ``Project.apply_diagnosis()`` wendet die Maßnahmen an und
``Project.improve(rounds)`` wiederholt *diagnostizieren → anwenden → neu trainieren* und **behält nur, was die
Validierung wirklich verbessert** (sonst zurück zum besseren Stand). Die Testmenge wird dabei nie benutzt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    text: str
    technique: str | None = None                 # Technik einschalten …
    changes: dict[str, Any] | None = None        # … oder Einstellungen setzen

    def __str__(self) -> str:
        how = f"Technik '{self.technique}'" if self.technique else ", ".join(f"{k} = {v!r}" for k, v in
                                                                            (self.changes or {}).items())
        return f"{self.text} ({how})"


@dataclass
class Finding:
    key: str
    title: str
    severity: str                                # hoch | mittel | hinweis
    evidence: str
    actions: list[Action] = field(default_factory=list)

    def __str__(self) -> str:
        mark = {"hoch": "✗", "mittel": "⚠", "hinweis": "ℹ"}.get(self.severity, "•")
        lines = [f"{mark} {self.title}: {self.evidence}"]
        lines += [f"    → {a}" for a in self.actions]
        return "\n".join(lines)


@dataclass
class DiagnosisReport:
    findings: list[Finding]
    monitor: str
    score: float | None

    @property
    def ok(self) -> bool:
        return not any(f.severity in ("hoch", "mittel") for f in self.findings)

    def actions(self, include_hints: bool = True) -> list[Action]:
        seen: set[str] = set()
        out = []
        for f in self.findings:
            if f.severity == "hinweis" and not include_hints:
                continue
            for a in f.actions:
                key = a.technique or repr(sorted((a.changes or {}).items()))
                if key not in seen:
                    seen.add(key)
                    out.append(a)
        return out

    def __str__(self) -> str:
        head = f"Diagnose ({self.monitor}" + (f" = {self.score:.4g}" if self.score is not None else "") + ")"
        if not self.findings:
            return head + ": keine Auffälligkeiten – das Training sieht gesund aus."
        body = "\n".join(str(f) for f in self.findings)
        tail = "\nAnwenden: projekt.apply_diagnosis() · automatisch verbessern: projekt.improve()"
        return f"{head}:\n{body}{tail}"


# ---------------------------------------------------------------------------- Hilfen
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def monitor_of(project: Any) -> str:
    """Beobachtete Metrik: die des Early Stopping, sonst ``val_loss``."""
    for cb in project.config.training.callbacks or []:
        if getattr(cb, "name", "") == "early_stopping":
            return getattr(cb.args, "monitor", "val_loss") or "val_loss"
    return "val_loss"


def _better(a: float, b: float, mode: str) -> bool:
    return a < b if mode == "min" else a > b


def _reference_lr(project: Any) -> float | None:
    """Lernrate der Ausgangsvorlage (gleicher Optimierer) – bewährter Wert."""
    ref = getattr(project, "_reference", None) or {}
    opt = ref.get("training", {}).get("optimizer", {})
    if opt.get("name") != project.config.training.optimizer.name:
        return None
    lr = (opt.get("args") or {}).get("lr")
    return float(lr) if isinstance(lr, (int, float)) and lr > 0 else None


def _technique_states(project: Any) -> dict[str, str]:
    return {r.technique.key: r.status for r in project.techniques()}


# ---------------------------------------------------------------------------- Diagnose
def diagnose(project: Any) -> DiagnosisReport:
    from ..training.metrics import direction

    cfg = project.config
    hist = [h for h in project.history if isinstance(h, dict)]
    train = [h for h in hist if "loss" in h and not any(str(k).startswith("val_") for k in h)]
    evals = [h for h in hist if "val_loss" in h]
    monitor = monitor_of(project)
    if evals and monitor not in evals[-1]:
        monitor = "val_loss"
    mode = direction(monitor)
    states = _technique_states(project)
    trainer = project.trainer
    findings: list[Finding] = []

    def tech(key: str, text: str) -> list[Action]:
        return [Action(text, technique=key)] if states.get(key) in ("aus", "teilweise") else []

    t = cfg.training
    steps_now = t.max_steps

    def longer(text: str = "Doppelt so lange trainieren") -> list[Action]:
        if steps_now is not None:
            return [Action(text, changes={"training.max_steps": int(steps_now) * 2})]
        return [Action(text, changes={"training.epochs": int(t.epochs) * 2})]

    lr = getattr(t.optimizer.args, "lr", None)

    def lr_actions() -> list[Action]:
        """Lernrate weit unter dem bewährten Wert der Vorlage → diesen wiederherstellen, sonst LR-Finder."""
        ref = _reference_lr(project)
        if ref and _finite(lr) and lr < ref / 3:
            return [Action(f"Lernrate der Vorlage wiederherstellen ({lr:.3g} ist {ref / lr:.0f}× kleiner)",
                           changes={"training.optimizer.lr": ref})]
        return tech("auto_lernrate", "Lernrate per LR-Finder bestimmen")

    # 1) Instabil
    losses = [float(h["loss"]) for h in train if _finite(h.get("loss"))]
    skipped = getattr(trainer, "skipped_steps", 0) if trainer is not None else 0
    spikes = getattr(trainer, "loss_spikes", 0) if trainer is not None else 0
    jumps = sum(1 for a, b in zip(losses, losses[1:]) if a > 0 and b > 3 * a)
    bad_val = any(not _finite(h.get("val_loss")) for h in evals)
    if skipped or spikes or jumps or bad_val:
        why = []
        if skipped:
            why.append(f"{skipped} Schritt(e) wegen NaN/Inf übersprungen")
        if spikes:
            why.append(f"{spikes} Verlust-Spitze(n)")
        if jumps:
            why.append(f"Verlust sprang {jumps}× auf mehr als das Dreifache")
        if bad_val:
            why.append("Validierungsverlust NaN/Inf")
        acts = tech("gradient_clipping", "Gradienten begrenzen") + tech("warmup", "Lernrate langsam hochfahren")
        if _finite(lr) and lr:
            acts.append(Action("Lernrate halbieren", changes={"training.optimizer.lr": float(f"{lr / 2:.3g}")}))
        findings.append(Finding("instabil", "Training instabil", "hoch", "; ".join(why), acts))

    # 2) Lernt kaum
    if len(losses) >= 4 and cfg.training.paradigm.name not in ("gan",):
        k = max(1, len(losses) // 5)
        first, last = _mean(losses[:k]), _mean(losses[-k:])
        if first > 0 and last >= 0.97 * first:
            acts = lr_actions() + longer()
            findings.append(Finding("lernt_kaum", "Modell lernt kaum", "hoch",
                                    f"Trainingsverlust {first:.4g} → {last:.4g} (Rückgang < 3 %)", acts))

    # 3) Zu kurz: Validierung verbessert sich bis zuletzt oder der Trainingsverlust fällt am Ende noch deutlich
    values = [float(h[monitor]) for h in evals if _finite(h.get(monitor))]
    stopped_early = bool(trainer is not None and getattr(trainer, "should_stop", False))
    still = None
    if len(losses) >= 10:
        k = max(2, len(losses) // 5)
        prev, end = _mean(losses[-2 * k:-k]), _mean(losses[-k:])
        if prev > 0 and end < 0.97 * prev:
            still = f"Trainingsverlust fällt am Ende noch ({prev:.4g} → {end:.4g})"
    rising = None
    if len(values) >= 3:
        best_i = min(range(len(values)), key=lambda i: values[i] if mode == "min" else -values[i])
        third = max(1, len(values) // 3)
        ref, final = values[-1 - third], values[-1]
        gain = (ref - final) / max(abs(ref), 1e-12) if mode == "min" else (final - ref) / max(abs(ref), 1e-12)
        if best_i == len(values) - 1 and gain > 0.01:
            rising = f"{monitor} verbessert sich bis zuletzt ({ref:.4g} → {final:.4g})"
    adversarial = cfg.training.paradigm.name in ("gan",)      # Verluste schwingen absichtlich – keine Kurven-Befunde
    if (still or rising) and not stopped_early and not adversarial and not any(f.key == "lernt_kaum"
                                                                                 for f in findings):
        findings.append(Finding("zu_kurz", "Training zu kurz", "mittel", "; ".join(x for x in (rising, still) if x),
                                longer("Länger trainieren – das Modell lernt noch")))

    # 4) Überanpassung: Validierungsverlust steigt nach seinem Bestwert, während der Trainingsverlust weiter fällt
    #    (der Verlust ist viel weniger verrauscht als z. B. die Genauigkeit)
    val_losses = [float(h["val_loss"]) for h in evals if _finite(h.get("val_loss"))]
    over = None
    if len(val_losses) >= 3:
        bi = min(range(len(val_losses)), key=val_losses.__getitem__)
        best_v, final_v = val_losses[bi], val_losses[-1]
        at_best = [float(h["loss"]) for h in train if _finite(h.get("loss")) and h.get("step", 0) <= evals[bi].get(
            "step", 0)]
        train_fell = bool(at_best) and losses and _mean(losses[-max(1, len(losses) // 5):]) < 0.97 * _mean(
            at_best[-max(1, len(at_best) // 5):])
        if bi <= 0.7 * (len(val_losses) - 1) and final_v > best_v * 1.05 + 1e-12 and train_fell:
            over = (f"Validierungsverlust bestens {best_v:.4g} bei Messung {bi + 1}/{len(val_losses)}, zuletzt "
                    f"{final_v:.4g} – der Trainingsverlust fällt weiter", "hoch")
    if over is None and losses and val_losses:
        tr = _mean(losses[-max(1, len(losses) // 5):])
        if val_losses[-1] > 0.2 and tr < 0.4 * val_losses[-1]:
            over = (f"Trainingsverlust {tr:.4g} ≪ Validierungsverlust {val_losses[-1]:.4g}", "mittel")
    if over is not None and not adversarial:
        acts = (tech("early_stopping", "Beim besten Stand anhalten und ihn behalten")
                + tech("dropout", "Dropout") + tech("weight_decay", "Gewichtsverfall")
                + tech("bild_augmentierung", "Bild-Augmentierung") + tech("label_smoothing", "Label Smoothing")
                + tech("mixup", "Mixup") + tech("ema", "Gewichte mitteln (EMA)"))
        findings.append(Finding("ueberanpassung", "Überanpassung", over[1], over[0], acts[:5]))

    # 5) Kaum besser als die Basis
    base = getattr(project, "last_baseline", None) or {}
    if not base and cfg.training.paradigm.name == "supervised" and project.model is not None:
        try:
            base = project.baseline()
        except Exception:  # noqa: BLE001 - kein Basis-Vergleich möglich (z. B. generatives Modell)
            base = {}
    if "modell_accuracy" in base and "basis_accuracy" in base:
        m, b = base["modell_accuracy"], base["basis_accuracy"]
        if _finite(m) and _finite(b) and m < b + 0.05:
            findings.append(Finding("basis", "Kaum besser als die Basis", "hoch",
                                    f"Genauigkeit {m:.3f} vs. häufigste Klasse {b:.3f}",
                                    lr_actions() + longer()
                                    + tech("datenpruefung", "Daten prüfen (Labels, Lecks, Ungleichgewicht)")))
    elif "modell_r2" in base:
        r2 = base["modell_r2"]
        if _finite(r2) and r2 < 0.2:
            findings.append(Finding("basis", "Kaum besser als die Basis", "hoch",
                                    f"R² = {r2:.3f} (0 = nur der Mittelwert)",
                                    lr_actions() + longer()
                                    + tech("datenpruefung", "Daten prüfen")))

    last = evals[-1] if evals else {}
    # 6) Klassen unausgeglichen
    acc, bal = last.get("val_accuracy"), last.get("val_balanced_accuracy")
    if _finite(acc) and _finite(bal) and bal < acc - 0.1:
        findings.append(Finding("ungleich", "Seltene Klassen werden übersehen", "mittel",
                                f"balancierte Genauigkeit {bal:.3f} ≪ Genauigkeit {acc:.3f}",
                                tech("klassengewichte", "Seltene Klassen stärker gewichten")
                                + tech("klassen_ausgleich", "Seltene Klassen öfter zeigen")))
    # 7) Kalibrierung
    ece = last.get("val_ece")
    if _finite(ece) and ece > 0.08 and cfg.inference.temperature == 1.0:
        findings.append(Finding("kalibrierung", "Wahrscheinlichkeiten schlecht kalibriert", "mittel",
                                f"ECE = {ece:.3f} (gut: < 0,05)", tech("kalibrierung", "Temperature Scaling")))
    # 8) Keine Testmenge
    supervised = cfg.training.paradigm.name == "supervised" and cfg.data.name != "none"
    if supervised and project.test_data is None:
        findings.append(Finding("testmenge", "Keine Testmenge", "hinweis",
                                "die Qualität ist nur auf Validierungsdaten gemessen (durch Auswahl leicht geschönt)",
                                tech("testmenge", "Testmenge zurückhalten")))
    # 9) Validierung zu optimistisch
    test = getattr(project, "last_test", None) or {}
    tkey = "test_" + monitor[4:] if monitor.startswith("val_") else None
    if tkey and tkey in test and _finite(test[tkey]) and values:
        v, te = values[-1], float(test[tkey])
        gap = (te - v) / max(abs(v), 1e-12) if mode == "min" else v - te
        if gap > (0.15 if mode == "min" else 0.05):
            findings.append(Finding("optimistisch", "Validierung zu optimistisch", "mittel",
                                    f"{monitor} {v:.4g}, aber {tkey} {te:.4g}",
                                    tech("kreuzvalidierung", "Mit Kreuzvalidierung absichern")))

    score = values[-1] if values else None
    order = {"hoch": 0, "mittel": 1, "hinweis": 2}
    findings.sort(key=lambda f: order.get(f.severity, 3))
    return DiagnosisReport(findings, monitor, score)


def apply(project: Any, report: DiagnosisReport | None = None, max_actions: int = 4,
          include_hints: bool = True) -> list[str]:
    """Wendet die Maßnahmen an (zuerst die der schwersten Befunde) → Beschreibungen der angewandten.

    ``include_hints=False``: nur Maßnahmen echter Befunde (``improve`` – Hinweise wie 'Testmenge zurückhalten'
    ändern die Datenaufteilung und machen Runden unvergleichbar)."""
    report = report or diagnose(project)
    done: list[str] = []
    changes: dict[str, Any] = {}
    actions = report.actions(include_hints)
    fixed_lr = any(a.changes and "training.optimizer.lr" in a.changes for a in actions)
    for action in actions:
        if len(done) >= max_actions:
            break
        if action.technique == "auto_lernrate" and fixed_lr:
            continue                      # der LR-Finder würde die gesetzte Lernrate sofort wieder überschreiben
        if action.technique:
            try:
                if project.technique(action.technique, "an"):
                    done.append(str(action))
            except Exception:  # noqa: BLE001 - passt doch nicht zu diesem Projekt
                continue
        elif action.changes:
            fresh = {k: v for k, v in action.changes.items() if k not in changes}
            if fresh:
                changes.update(fresh)
                done.append(str(action))
    if changes:
        project.update(changes)
    return done
