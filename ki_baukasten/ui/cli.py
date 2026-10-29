"""
ki_baukasten.ui.cli
===================

Kommandozeile: ``python -m ki_baukasten <befehl> …`` (oder ``ki-baukasten …``).

Befehle
-------
==============  =================================================================
Befehl          Zweck
==============  =================================================================
ziele           Ziel-Assistent: Zweck & Prioritäten → empfohlene/abgeratene Modelle, Techniken; neues Projekt
presets         Vorlagen nach Anwendungsgebiet auflisten
modelle         alle Modelle mit Kurzbeschreibung
komponenten     alle Registries bzw. Einträge einer Registry (``komponenten activation``)
zeige           Einstellungsbaum einer Vorlage/Datei (mit Tooltips, nach Stufe filterbar)
erklaere        ausführliche Erklärung einer Einstellung oder Komponente
vorlage         Vorlage als JSON/YAML-Datei schreiben (zum Bearbeiten)
pruefen         Konfigurationsdatei prüfen
bauen           Modell bauen und Übersicht + Speicherschätzung zeigen
speicher        Speicherschätzung mit Empfehlungen
tensoren        Parameternamen/-formen (für Tensor-Regeln) + Regel-Vorschau
aufbau          komplette Aufbau-Darstellung (Text, Markdown, Mermaid, JSON)
optimieren      Ratgeber: Speicher-/Tempo-/Stabilitätsvorschläge (optional übernehmen)
suche           Einstellungen per Volltext (Pfade + Tooltips) finden
schalter        jede Einstellung An / Aus / Eigen: Übersicht, umschalten, speichern
techniken       Katalog aller Techniken (Stabilität … Schwarm, Evolution) mit Status, an/aus
vergleiche      zwei Vorlagen/Dateien vergleichen
exportieren     Modell exportieren (torch.export, TorchScript, ONNX, state_dict)
importieren     fremdes Modell laden (pt2, TorchScript, state_dict, safetensors, HF) → Projekt
komprimieren    Modell eindampfen (Pruning, Low-Rank, Quantisierung, Heilen) mit Bericht
zielsysteme     Geräteprofile + Übersicht, ob/wie das Modell darauf passt
reduzieren      Empfehlung für ein Zielsystem anzeigen und anwenden (+ Export)
notfall         passt trotzdem nicht? stückweise ausführen (SSD/RAM-Auslagerung, Aufteilen, LISA …)
schueler        Lehrer-Schüler: kleineres Modell per Distillation trainieren
probelauf       Forward/Backward mit allen Wegen, Statistiken, Warnungen (+ Testlauf)
moeglichkeiten  Katalog aller Möglichkeiten (Speicher, Tempo, Kompression, Training …)
trainieren      trainieren (und speichern)
ablauf          ganzer Ablauf: Suche → Training (normal/föderiert/Model Soup) → Prüfung → Kompression → Export
automl          automatische Suche: Zufall, Gitter, Evolution, Partikelschwarm, PBT
verteilt        verteilt trainieren: mehrere GPUs/Rechner/Prozesse (DDP, ZeRO-1, FSDP über torchrun)
generieren      gespeichertes Projekt laden und Text/Samples erzeugen (Instruction-Modelle: Frage → Antwort)
perfektion      Anforderungs-Katalog: was braucht diese KI, was ist an/aus/offen/fehlt (+ alles Empfohlene an)
datenpruefung   Daten prüfen (Lecks, Duplikate, Klassen, NaN, Skalen …) mit Abhilfen
bewerten        gespeichertes Projekt bewerten: Validierung, Test, Basis, Fehleranalyse, Kalibrierung
modellkarte     Modellkarte (Markdown) eines gespeicherten Projekts anzeigen/schreiben
drift           neue Daten (CSV/Tensor-Datei) mit den Trainingsdaten vergleichen (PSI, KS, Vorhersage-Drift)
bereitstellen   gespeichertes Projekt als HTTP-Dienst starten (JSON: /vorhersage, /frage, /generieren)
doku            vollständige Parameter-Referenz + Anforderungs-Katalog als Markdown exportieren
gui             grafische Oberfläche starten
==============  =================================================================

Quelle ``QUELLE`` ist entweder ein Preset-Name oder eine Konfigurationsdatei.
Mit ``--setze pfad=wert`` (mehrfach) lassen sich Einstellungen überschreiben,
z. B. ``--setze model.d_model=256 --setze training.optimizer.lr=1e-3``.
Platzhalter wirken auf alle Treffer: ``--setze "dropout=0.1"`` (jede Einstellung
namens dropout) oder ``--setze "model.**.num_heads=8"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_value(text: str) -> Any:
    """'256' → 256, '[1,2]' → Liste, 'true' → True, 'null' → None, sonst Text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        low = text.strip().lower()
        if low in ("none", "null", "keine"):
            return None
        if low in ("true", "ja", "wahr"):
            return True
        if low in ("false", "nein", "falsch"):
            return False
        return text


def _project(source: str, sets: list[str] | None = None):
    from ..project import Project

    p = Project.from_file(source) if Path(source).exists() else Project.from_preset(source)
    for item in sets or []:
        if "=" not in item:
            raise SystemExit(f"--setze erwartet pfad=wert, erhalten: {item!r}")
        path, value = item.split("=", 1)
        path = path.strip()
        if "." not in path and "*" not in path:
            try:
                p.config.get(path)
            except Exception:  # kein Pfad auf oberster Ebene → als Name überall suchen
                path = "**." + path
        raw = value.strip()
        parsed = _parse_value(raw)
        try:
            p.set(path, parsed)
        except Exception:
            if parsed == raw:
                raise
            p.set(path, raw)  # z. B. 'none' als Auswahlwert statt None
    return p


def cmd_ziele(a: argparse.Namespace) -> None:
    from ..analysis.goals import PRIORITIES, TASKS, evaluate_goals

    if a.liste:
        print("Aufgaben:\n" + "\n".join(f"  {k:<24} {v}" for k, v in TASKS.items()))
        print("Prioritäten:\n" + "\n".join(f"  {k:<24} {v}" for k, v in PRIORITIES.items()))
        return
    goals: dict[str, Any] = {}
    if a.quelle:
        p = _project(a.quelle, a.setze)
        goals = p.config.goals.to_dict()
    for key, value in (("task", a.aufgabe), ("description", a.text), ("data_size", a.daten), ("labels", a.labels),
                       ("sequence_length", a.sequenzen), ("target", a.ziel), ("training_hardware", a.hardware)):
        if value is not None:
            goals[key] = value
    if a.prioritaeten:
        goals["priorities"] = [x.strip() for x in a.prioritaeten.split(",") if x.strip()]
    if a.vortrainiert:
        goals["pretrained"] = True
    if a.quelle:
        p.config.goals.update(goals)
        print(p.goal_check())
        target_project = p
    else:
        print(evaluate_goals(goals))
        target_project = None
    if a.erstellen:
        from ..project import Project

        q = Project.from_goals(goals, choice=a.wahl)
        q.config.save(a.erstellen, only_changed=True)
        print(f"\nNeues Projekt aus Vorlage '{q.config.name}' gespeichert: {a.erstellen}")
    elif target_project is not None and a.speichern:
        target_project.config.save(a.speichern, only_changed=True)
        print(f"Ziele gespeichert: {a.speichern}")


def cmd_presets(a: argparse.Namespace) -> None:
    from ..core import registry
    from ..presets import DOMAINS, presets_by_domain

    reg = registry("preset")
    for domain, names in presets_by_domain().items():
        if a.gebiet and domain != a.gebiet:
            continue
        print(f"\n■ {domain} – {DOMAINS.get(domain, '')}")
        for n in names:
            print(f"   {n:<30} {reg.get(n).tooltip.what}")
    print("\nNutzung: python -m ki_baukasten zeige <preset>   |   trainieren <preset>")


def cmd_modelle(a: argparse.Namespace) -> None:
    from ..core import registry

    print(registry("model").describe(a.stufe, width=140))


def cmd_komponenten(a: argparse.Namespace) -> None:
    from ..core import all_registries, registry

    if a.registry:
        print(registry(a.registry).describe(a.stufe, width=140))
        return
    for reg in all_registries().values():
        print(f"{reg.kind:<18} {len(reg):>3} Einträge  – {reg.title}: {reg.tooltip.short()}")
    print("\nDetails: python -m ki_baukasten komponenten <registry>   (z. B. activation, optimizer, reward)")


def cmd_zeige(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    print(p.config.describe(max_level=a.stufe, tooltips=not a.ohne_tooltips, only_changed=a.geaendert))


def cmd_erklaere(a: argparse.Namespace) -> None:
    target = a.pfad
    if ":" in target:  # Komponente: registry:name
        from ..core import registry

        kind, name = target.split(":", 1)
        comp = registry(kind).get(name)
        print(f"Komponente '{comp.name}' (Registry '{kind}', Stufe {comp.level.label})")
        if comp.aliases:
            print(f"Aliase: {', '.join(comp.aliases)}")
        print(comp.tooltip.format(indent="  "))
        args = comp.config_cls()
        if type(args)._params:
            print("\nArgumente:")
            for ref in args.walk():
                pad = "  " * (ref.depth + 1)
                print(f"{pad}• {ref.name} = {ref.value!r}  [{ref.param.type_label()}]  {ref.param.tooltip.short()}")
        return
    p = _project(a.quelle, a.setze)
    print(p.explain(target))


def cmd_vorlage(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    out = Path(a.ausgabe or f"{p.config.name}.json")
    p.config.save(out, only_changed=a.nur_geaendert)
    print(f"Konfiguration geschrieben: {out}")


def cmd_pruefen(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    problems = p.config.problems()
    if problems:
        print("✗ Probleme gefunden:")
        for pr in problems:
            print(f"  - {pr}")
        sys.exit(1)
    print("✓ Konfiguration ist gültig.")


def cmd_bauen(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    p.build()
    print(p.summary(max_depth=a.tiefe))
    print()
    print(p.memory_report())


def cmd_speicher(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    if a.profil:
        p.apply_memory_profile(a.profil)
    p.build()
    print(p.memory_report(batch_size=a.batch))


def cmd_tensoren(a: argparse.Namespace) -> None:
    from ..core import format_number

    p = _project(a.quelle, a.setze)
    cfg_rules = list(p.config.tensor_rules)
    p.config.tensor_rules = []
    model = p.build()
    print(f"{'Parameter':<64} {'Form':<20} {'Anzahl':>10}")
    for name, shape, n, trainable in model.parameter_table():
        if a.filter and a.filter not in name:
            continue
        print(f"{name:<64} {str(shape):<20} {format_number(n):>10}" + ("" if trainable else "  (eingefroren)"))
    if cfg_rules:
        from ..tensor import TensorSurgeon

        print("\nVorschau der Tensor-Regeln:")
        print(TensorSurgeon(model, cfg_rules).preview())


def cmd_aufbau(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    text = p.architecture(a.format, a.tiefe, a.ausgabe)
    if a.ausgabe:
        print(f"Aufbau geschrieben: {a.ausgabe}")
    else:
        print(text)


def cmd_optimieren(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    tips = p.advise(a.batch)
    if not tips:
        print("✓ Keine Vorschläge – die Konfiguration sieht gut aus.")
        return
    for t in tips:
        print(t, end="\n\n")
    if a.anwenden is not None:
        keys = None if a.anwenden in ("alle", "all", "") else [k.strip() for k in a.anwenden.split(",")]
        done = p.apply_advice(keys, a.kategorie)
        print("Übernommen:", ", ".join(d.key for d in done) or "(nichts)")
        out = Path(a.ausgabe or f"{p.config.name}_optimiert.json")
        p.config.save(out, only_changed=True)
        print(f"Optimierte Konfiguration gespeichert: {out}")


def cmd_suche(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    hits = p.search(a.text, limit=a.anzahl)
    if not hits:
        print(f"Nichts gefunden für '{a.text}'.")
    for path, what in hits:
        print(f"{path:<58} {what}")


def cmd_schalter(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    for state, patterns in (("aus", a.aus), ("an", a.an), ("standard", a.standard)):
        for pattern in patterns:
            if "." not in pattern and "*" not in pattern:
                try:
                    p.config.get(pattern)
                except Exception:
                    pattern = "**." + pattern
            changed = p.switch(pattern, state)
            print(f"{state:>8}: {pattern} → {len(changed)} geändert" + (f" ({', '.join(changed[:6])}"
                                                                      f"{' …' if len(changed) > 6 else ''})" if changed else ""))
    if a.uebersicht or not (a.muster or a.zustand):
        print(p.switch_overview(a.stufe))
    if a.muster or a.zustand:
        pattern = a.muster
        if pattern and "." not in pattern and "*" not in pattern:
            pattern = "**." + pattern
        infos = p.switches(pattern, a.zustand, a.stufe, a.nur_schaltbar)
        for info in infos:
            print(f"  {info}")
        print(f"{len(infos)} Einstellungen.  Umschalten: --an/--aus/--standard MUSTER  oder  --setze pfad=aus")
    if a.speichern:
        p.config.save(a.speichern, only_changed=True)
        print(f"Gespeichert: {a.speichern}")


def cmd_techniken(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    for key in a.an:
        print(f"an:  {key} → {', '.join(p.enable(key)) or 'war schon an'}")
    for key in a.aus:
        print(f"aus: {key} → {', '.join(p.disable(key)) or 'war schon aus'}")
    if a.kategorie:
        from ..techniques import CATEGORIES

        if a.kategorie not in CATEGORIES:
            raise SystemExit(f"Unbekannte Kategorie '{a.kategorie}'. Verfügbar: {', '.join(CATEGORIES)}")
        for row in p.techniques(a.kategorie, max_level=a.stufe):
            t = row.technique
            print(f"  {row}\n      {t.tooltip.what} {t.tooltip.why}" + (f"  [{t.hint}]" if t.hint else ""))
    else:
        print(p.technique_overview(a.nur_aktive, a.stufe))
    if a.speichern:
        p.config.save(a.speichern, only_changed=True)
        print(f"Gespeichert: {a.speichern}")


def cmd_vergleiche(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    diff = p.diff(a.andere)
    if not diff:
        print("Keine Unterschiede.")
    print(f"{'Pfad':<58} {a.quelle[:24]:<26} {a.andere[:24]}")
    for path, mine, theirs in diff:
        print(f"{path:<58} {str(mine)[:25]:<26} {str(theirs)[:40]}")


def cmd_exportieren(a: argparse.Namespace) -> None:
    from ..project import Project

    p = Project.load(a.quelle) if Path(a.quelle).is_dir() else _project(a.quelle, a.setze)
    target = p.export(a.ausgabe, a.format, merge_lora=not a.ohne_merge)
    print(f"Exportiert: {target}")


def _load_or_build(source: str, sets: list[str] | None, train: bool = False):
    from ..project import Project

    if Path(source).is_dir():
        p = Project.load(source)
        for item in sets or []:
            path, value = item.split("=", 1)
            p.set(path.strip(), _parse_value(value.strip()))
        return p
    p = _project(source, sets)
    if train:
        p.train()
    else:
        p.build()
    return p


def cmd_importieren(a: argparse.Namespace) -> None:
    from ..project import Project

    shape = [int(v) for v in a.form.split(",")]
    extra = {"input_shape": shape, "input_dtype": a.typ, "allow_pickle": a.pickle}
    if a.architektur:
        extra["architecture"] = _parse_value(a.architektur) if a.architektur.strip().startswith("{") else a.architektur
    p = Project.import_model(a.datei, a.format, data=a.daten or "none", name=a.name, **extra)
    print(p.architecture("text", max_depth=a.tiefe))
    if a.speichern:
        print(f"Projekt gespeichert: {p.save(a.speichern)}")


def cmd_komprimieren(a: argparse.Namespace) -> None:
    p = _load_or_build(a.quelle, a.setze, train=a.trainieren)
    steps = json.loads(a.schritte) if a.schritte else None
    p.config.training.verbose = True
    p.compress(steps)
    if a.speichern:
        p.config.training.verbose = False
        print(f"Gespeichert: {p.save(a.speichern)}")


def cmd_zielsysteme(a: argparse.Namespace) -> None:
    from ..analysis.targets import TARGETS

    if not a.quelle:
        for k, t in TARGETS.items():
            print(f"{k:<18} {t.title:<46} Speicher {t.memory_mb:>8g} MB  ~{t.gflops:g} GFLOP/s  "
                  f"Formate {', '.join(t.dtypes)}  Export: {t.export}")
        return
    p = _load_or_build(a.quelle, a.setze)
    print(p.deployment_overview(list(TARGETS)))


def cmd_reduzieren(a: argparse.Namespace) -> None:
    p = _load_or_build(a.quelle, a.setze, train=a.trainieren)
    print(p.recommend_for(a.ziel))
    if a.anwenden:
        p.config.training.verbose = True
        reduced = p.reduce_for(a.ziel, export=a.export)
        if a.speichern:
            reduced.config.training.verbose = False
            print(f"Gespeichert: {reduced.save(a.speichern)}")
        if a.export:
            print(f"Exportiert nach: {a.export}.*")


def cmd_notfall(a: argparse.Namespace) -> None:
    from ..analysis.fallback import options_text

    p = _project(a.quelle, a.setze)
    opts = p.fallback_options(a.ziel, a.budget_mb, a.training)
    budget = p._budget_mb(a.ziel, a.budget_mb)
    print(options_text(opts, budget, limit=None if a.alle else 8))
    if a.anwenden:
        chosen = p.apply_fallback(a.anwenden, a.ziel, a.budget_mb, a.training)
        print(f"\nEingeschaltet: {chosen.title}" + (f"\nNutzung: {chosen.usage}" if chosen.usage else ""))
        if a.speichern:
            p.config.save(a.speichern, only_changed=True)
            print(f"Konfiguration gespeichert: {a.speichern}")


def cmd_schueler(a: argparse.Namespace) -> None:
    p = _load_or_build(a.quelle, a.setze, train=a.trainieren)
    student = p.distill(width=a.breite, depth=a.tiefe, steps=a.schritte, feature_weight=a.features)
    n_t = sum(t.numel() for t in p.model.parameters())
    n_s = sum(t.numel() for t in student.model.parameters())
    print(f"Lehrer {n_t:,} → Schüler {n_s:,} Parameter ({100 * n_s / max(1, n_t):.1f} %)")
    if a.speichern:
        print(f"Schüler gespeichert: {student.save(a.speichern)}")


def cmd_probelauf(a: argparse.Namespace) -> None:
    from ..analysis.probe import probe, evaluate_model, evaluation_text

    p = _load_or_build(a.quelle, a.setze)
    model = p.model
    try:
        bundle = p.data()
        data = bundle.val if bundle.val is not None else bundle.train
        x = data["inputs"][: a.batch]
        y = data["targets"][: a.batch] if "targets" in data else None
    except Exception:  # noqa: BLE001
        data, x, y = None, model.example_input(a.batch), None
    print(probe(model, x, y, backward=not a.ohne_backward, max_depth=8).to_text(a.tiefe))
    if a.test and isinstance(data, dict):
        print()
        print(evaluation_text(evaluate_model(model, data)))


def cmd_moeglichkeiten(a: argparse.Namespace) -> None:
    from .docs import capabilities_markdown, export_capabilities

    if a.ausgabe:
        print(f"Katalog geschrieben: {export_capabilities(a.ausgabe)}")
    else:
        print(capabilities_markdown())


def cmd_trainieren(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    if a.speicher:
        p.apply_memory_profile(a.speicher)
    if a.epochen:
        p.set("training.epochs", a.epochen)
    p.train()
    if p.config.data.name == "anomaly":
        print("Anomalie-Erkennung:", p.anomaly_report())
    from ..training.distributed import is_main_process

    target = p.save(a.speichern) if a.speichern != "-" else None
    if target and is_main_process():
        print(f"Gespeichert in: {target}")


def cmd_ablauf(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    report = p.run()
    print("Ablauf fertig: " + ", ".join(report))
    if a.speichern:
        print(f"Gespeichert in: {p.save(a.speichern)}")


def cmd_automl(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    space = json.loads(a.raum) if a.raum else None
    result = p.tune(a.methode, space=space, trials=a.versuche, steps_per_trial=a.schritte,
                    population=a.population, generations=a.generationen, metric=a.metrik, mode=a.richtung)
    print(result)
    if a.speichern:
        p.config.save(a.speichern, only_changed=True)
        print(f"Konfiguration mit den besten Einstellungen: {a.speichern}")


def cmd_verteilt(a: argparse.Namespace) -> None:
    import shlex
    import subprocess

    from ..training.distributed import launch_command

    extra = [x for item in a.setze for x in ("--setze", item)]
    if a.speichern:
        extra += ["--speichern", a.speichern]
    cmd = launch_command(a.quelle, a.prozesse, a.strategie, a.knoten, a.knoten_rang, a.master, extra)
    print("Start:", " ".join(shlex.quote(c) for c in cmd))
    if a.nur_anzeigen:
        return
    sys.exit(subprocess.call(cmd))


def cmd_generieren(a: argparse.Namespace) -> None:
    from ..project import Project

    p = Project.load(a.ordner)
    model = p.model
    if p._info().get("prompt_template"):
        print(p.ask(a.prompt, max_new_tokens=a.laenge))
    elif hasattr(model, "generate") and hasattr(model, "lm_head"):
        print(p.generate(a.prompt, max_new_tokens=a.laenge, temperature=a.temperatur, top_k=a.top_k))
    else:
        samples = p.sample(a.anzahl)
        print(samples[: min(10, a.anzahl)])


def cmd_doku(a: argparse.Namespace) -> None:
    from ..requirements import requirements_markdown
    from .docs import export_markdown

    path = export_markdown(a.ausgabe)
    print(f"Parameter-Referenz geschrieben: {path}")
    req = Path(path).parent / "ANFORDERUNGEN.md"
    req.write_text(requirements_markdown(), encoding="utf-8")
    print(f"Anforderungs-Katalog geschrieben: {req}")


def _open(source: str, sets: list[str] | None = None):
    """Gespeicherter Projektordner (mit Gewichten) oder Preset/Datei – ohne zu bauen."""
    from ..project import Project

    if Path(source).is_dir():
        return Project.load(source)
    return _project(source, sets)


def cmd_perfektion(a: argparse.Namespace) -> None:
    if a.katalog:
        from ..requirements import requirements_markdown

        text = requirements_markdown()
        if a.ausgabe:
            Path(a.ausgabe).write_text(text, encoding="utf-8")
            print(f"Katalog geschrieben: {a.ausgabe}")
        else:
            print(text)
        return
    p = _open(a.quelle, a.setze)
    print(p.perfection_check(a.stufe))
    if a.anwenden:
        done = p.apply_perfection(a.stufe, include_heavy=a.aufwendig)
        print(f"\nEingeschaltet: {', '.join(done) or 'nichts (alles Schaltbare ist schon an)'}")
        print(f"Neue Punktzahl: {p.perfection_check(a.stufe).score:.0f} / 100")
    if a.speichern:
        p.config.save(a.speichern, only_changed=True)
        print(f"Gespeichert: {a.speichern}")


def cmd_datenpruefung(a: argparse.Namespace) -> None:
    p = _open(a.quelle, a.setze)
    report = p.data_check()
    print(report)
    if a.anwenden:
        print("\nÜbernommen: " + (", ".join(p.apply_data_fixes(report)) or "nichts"))
    if a.speichern:
        p.config.save(a.speichern, only_changed=True)
        print(f"Gespeichert: {a.speichern}")


def cmd_bewerten(a: argparse.Namespace) -> None:
    p = _open(a.quelle, a.setze)
    if p.trainer is None and not Path(a.quelle).is_dir():
        p.train()
    data = None
    if Path(a.quelle).is_dir():                     # geladenes Projekt: auf seinen Daten bewerten
        bundle = p.data()
        data = bundle.val if bundle.val is not None else bundle.train
    fmt = (lambda d: "  ".join(f"{k}={v:.4g}" for k, v in d.items() if isinstance(v, (int, float))))
    print("Validierung: " + fmt(p.evaluate(data)))
    if p.test_data is not None:
        print("Test:        " + fmt(p.test()))
    if p.config.training.paradigm.name == "supervised":
        print("Basis:       " + fmt(p.baseline(data)))
        ea = p.error_analysis(data, top=a.top)
        if "konfusionsmatrix" in ea:
            print(f"Fehlerquote: {ea['fehlerquote']:.3f}")
            print("Konfusionsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
            for row in ea["konfusionsmatrix"]:
                print("  " + " ".join(f"{v:>5}" for v in row))
            for w in ea["sicherste_fehler"]:
                print(f"  #{w['index']}: richtig {w['richtig']}, vorhergesagt {w['vorhergesagt']} "
                      f"({w['sicherheit']:.0%} sicher)")
        else:
            print(f"Mittlerer Fehler: {ea['mittlerer_fehler']:.4g}; größte: "
                  + ", ".join(f"#{e['index']} ({e['fehler']:.3g})" for e in ea["groesste_fehler"]))
        if a.kalibrieren:
            print("Kalibrierung: " + fmt(p.calibrate(data)))


def cmd_modellkarte(a: argparse.Namespace) -> None:
    p = _open(a.quelle)
    text = p.model_card(a.ausgabe)
    print(text if not a.ausgabe else f"Modellkarte geschrieben: {a.ausgabe}")


def cmd_drift(a: argparse.Namespace) -> None:
    import csv as _csv

    import torch

    from ..data.artifacts import prepare_raw

    p = _open(a.quelle)
    if a.neu.endswith(".pt"):
        new = torch.load(a.neu, weights_only=True)
    else:
        with open(a.neu, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f, delimiter=a.trenner))
        new = {"inputs": prepare_raw(p._info(), rows)}
    print(p.drift_report(new, with_model=not a.ohne_modell))


def cmd_bereitstellen(a: argparse.Namespace) -> None:
    p = _open(a.quelle)
    p.serve(host=a.host, port=a.port, api_key=a.schluessel)


def cmd_gui(a: argparse.Namespace) -> None:
    from .gui import main as gui_main

    gui_main(a.quelle)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ki_baukasten", description="KI-Baukasten – KIs bis ins Detail bauen.")
    sub = ap.add_subparsers(dest="befehl", required=True)

    def with_source(sp: argparse.ArgumentParser, optional: bool = False) -> None:
        if optional:
            sp.add_argument("quelle", nargs="?", default="tabellen_klassifikation", help="Preset-Name oder Datei")
        else:
            sp.add_argument("quelle", help="Preset-Name oder Konfigurationsdatei (.json/.yaml)")
        sp.add_argument("--setze", action="append", default=[], metavar="PFAD=WERT", help="Einstellung überschreiben")

    sp = sub.add_parser("ziele", help="Ziel-Assistent: Modelle nach Zweck und Prioritäten")
    sp.add_argument("quelle", nargs="?", help="vorhandenes Projekt prüfen (Preset oder Datei, optional)")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--aufgabe", help="z. B. text_erzeugen, bild_klassifizieren, zeitreihe_prognose (--liste)")
    sp.add_argument("--text", help="freie Beschreibung (Stichwörter wie Handy, Echtzeit, Datenschutz werden erkannt)")
    sp.add_argument("--prioritaeten", help="Komma-Liste, z. B. genauigkeit,wenig_speicher,datenschutz")
    sp.add_argument("--daten", choices=["winzig", "klein", "mittel", "gross"])
    sp.add_argument("--labels", choices=["vorhanden", "wenige", "keine"])
    sp.add_argument("--sequenzen", choices=["keine", "kurz", "mittel", "lang", "sehr_lang"])
    sp.add_argument("--ziel", help="Zielgerät (smartphone, raspberry_pi, laptop_cpu, gaming_gpu …)")
    sp.add_argument("--hardware", choices=["cpu", "laptop_gpu", "gaming_gpu", "apple_silicon", "rechenzentrum_gpu",
                                           "cluster"])
    sp.add_argument("--vortrainiert", action="store_true", help="vortrainiertes Modell anpassen (Feintuning)")
    sp.add_argument("--erstellen", metavar="DATEI", help="neues Projekt aus der besten Empfehlung speichern")
    sp.add_argument("--wahl", type=int, default=0, help="Index der Empfehlung für --erstellen (0 = beste)")
    sp.add_argument("--speichern", help="geprüftes Projekt mit neuen Zielen speichern")
    sp.add_argument("--liste", action="store_true", help="alle Aufgaben und Prioritäten zeigen")
    sp.set_defaults(fn=cmd_ziele)

    sp = sub.add_parser("presets", help="Vorlagen auflisten")
    sp.add_argument("--gebiet", help="nur dieses Anwendungsgebiet")
    sp.set_defaults(fn=cmd_presets)

    sp = sub.add_parser("modelle", help="Modelle auflisten")
    sp.add_argument("--stufe", default=None)
    sp.set_defaults(fn=cmd_modelle)

    sp = sub.add_parser("komponenten", help="Registries/Komponenten auflisten")
    sp.add_argument("registry", nargs="?")
    sp.add_argument("--stufe", default=None)
    sp.set_defaults(fn=cmd_komponenten)

    sp = sub.add_parser("zeige", help="Einstellungsbaum anzeigen")
    with_source(sp)
    sp.add_argument("--stufe", default="basis", help="basis | fortgeschritten | experte | experimentell")
    sp.add_argument("--geaendert", action="store_true", help="nur vom Standard abweichende Werte")
    sp.add_argument("--ohne-tooltips", action="store_true")
    sp.set_defaults(fn=cmd_zeige)

    sp = sub.add_parser("erklaere", help="Einstellung oder Komponente erklären")
    sp.add_argument("pfad", help="Einstellungspfad (model.d_model) oder registry:name (activation:gelu)")
    sp.add_argument("--quelle", default="tabellen_klassifikation", help="Preset oder Datei als Kontext")
    sp.add_argument("--setze", action="append", default=[])
    sp.set_defaults(fn=cmd_erklaere)

    sp = sub.add_parser("vorlage", help="Konfigurationsdatei schreiben")
    with_source(sp)
    sp.add_argument("-o", "--ausgabe")
    sp.add_argument("--nur-geaendert", action="store_true", help="nur Abweichungen vom Standard speichern")
    sp.set_defaults(fn=cmd_vorlage)

    sp = sub.add_parser("pruefen", help="Konfiguration prüfen")
    with_source(sp)
    sp.set_defaults(fn=cmd_pruefen)

    sp = sub.add_parser("bauen", help="Modell bauen + Übersicht")
    with_source(sp)
    sp.add_argument("--tiefe", type=int, default=2)
    sp.set_defaults(fn=cmd_bauen)

    sp = sub.add_parser("speicher", help="Speicherschätzung")
    with_source(sp)
    sp.add_argument("--batch", type=int)
    sp.add_argument("--profil", help="Speicherprofil vorher anwenden (sparsam, extrem_sparsam, schnell)")
    sp.set_defaults(fn=cmd_speicher)

    sp = sub.add_parser("tensoren", help="Parameter auflisten / Tensor-Regeln vorschauen")
    with_source(sp)
    sp.add_argument("--filter", help="nur Namen, die diesen Text enthalten")
    sp.set_defaults(fn=cmd_tensoren)

    sp = sub.add_parser("aufbau", help="komplette Aufbau-Darstellung des Modells")
    with_source(sp)
    sp.add_argument("--format", default="text", choices=["text", "markdown", "mermaid", "json"])
    sp.add_argument("--tiefe", type=int, default=3)
    sp.add_argument("-o", "--ausgabe", help="in Datei schreiben (z. B. aufbau.md)")
    sp.set_defaults(fn=cmd_aufbau)

    sp = sub.add_parser("optimieren", help="Optimierungs- und Speichervorschläge")
    with_source(sp)
    sp.add_argument("--batch", type=int)
    sp.add_argument("--anwenden", nargs="?", const="alle", default=None,
                    help="Vorschläge übernehmen: ohne Wert = alle, sonst Kennungen (adamw8bit,gqa)")
    sp.add_argument("--kategorie", help="nur diese Kategorie übernehmen (speicher, tempo, stabilitaet, struktur)")
    sp.add_argument("-o", "--ausgabe", help="Zieldatei der optimierten Konfiguration")
    sp.set_defaults(fn=cmd_optimieren)

    sp = sub.add_parser("suche", help="Einstellungen per Volltext finden")
    sp.add_argument("text")
    sp.add_argument("--quelle", default="tabellen_klassifikation")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--anzahl", type=int, default=40)
    sp.set_defaults(fn=cmd_suche)

    sp = sub.add_parser("schalter", help="Einstellungen An/Aus/Eigen anzeigen und umschalten")
    with_source(sp)
    sp.add_argument("muster", nargs="?", help="nur passende Einstellungen (z. B. dropout, 'training.**')")
    sp.add_argument("--zustand", choices=["an", "aus", "eigen"], help="nur Einstellungen in diesem Zustand")
    sp.add_argument("--stufe", default="fortgeschritten", help="basis | fortgeschritten | experte | experimentell")
    sp.add_argument("--nur-schaltbar", action="store_true", help="Pflicht-Einstellungen ausblenden")
    sp.add_argument("--an", action="append", default=[], metavar="MUSTER", help="einschalten (mehrfach möglich)")
    sp.add_argument("--aus", action="append", default=[], metavar="MUSTER", help="ausschalten (mehrfach möglich)")
    sp.add_argument("--standard", action="append", default=[], metavar="MUSTER", help="auf Standard zurücksetzen")
    sp.add_argument("--uebersicht", action="store_true", help="kompakte Übersicht zeigen")
    sp.add_argument("--speichern", help="Ergebnis als Konfigurationsdatei speichern")
    sp.set_defaults(fn=cmd_schalter)

    sp = sub.add_parser("techniken", help="Techniken-Katalog mit Status, an/aus")
    sp.add_argument("quelle", nargs="?", default="tabellen_klassifikation")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--kategorie", help="nur diese Kategorie (stabilitaet, generalisierung, tempo, grosse_modelle, "
                                        "cluster, automl, schwarm, evolution, daten, datenschutz, ueberwachung, "
                                        "kompression, experimentell)")
    sp.add_argument("--nur-aktive", action="store_true")
    sp.add_argument("--stufe", default=None)
    sp.add_argument("--an", action="append", default=[], metavar="KENNUNG")
    sp.add_argument("--aus", action="append", default=[], metavar="KENNUNG")
    sp.add_argument("--speichern", help="Konfiguration speichern")
    sp.set_defaults(fn=cmd_techniken)

    sp = sub.add_parser("vergleiche", help="zwei Vorlagen/Dateien vergleichen")
    with_source(sp)
    sp.add_argument("andere", help="zweite Vorlage oder Datei")
    sp.set_defaults(fn=cmd_vergleiche)

    sp = sub.add_parser("exportieren", help="Modell exportieren")
    sp.add_argument("quelle", help="gespeicherter Projektordner, Preset oder Datei")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--format", default="export", choices=["export", "torchscript", "onnx", "state_dict"])
    sp.add_argument("-o", "--ausgabe", default="export/modell")
    sp.add_argument("--ohne-merge", action="store_true", help="LoRA-Adapter nicht einrechnen")
    sp.set_defaults(fn=cmd_exportieren)

    sp = sub.add_parser("importieren", help="fremdes Modell importieren")
    sp.add_argument("datei", help="Modelldatei/-ordner (.pt2, .pt, .ts, .safetensors, Hugging-Face-Ordner/-Name)")
    sp.add_argument("--format", default="auto",
                    choices=["auto", "export", "torchscript", "module", "state_dict", "safetensors", "huggingface"])
    sp.add_argument("--form", default="1,16", help="Form einer Beispiel-Eingabe, z. B. 1,16 oder 1,3,224,224")
    sp.add_argument("--typ", default="long", choices=["long", "float32"], help="Datentyp der Beispiel-Eingabe")
    sp.add_argument("--architektur", help="Baukasten-Modell für reine Gewichte (Name oder JSON)")
    sp.add_argument("--daten", help="Datensatz für Nachtraining (Name), Standard: keiner")
    sp.add_argument("--name")
    sp.add_argument("--pickle", action="store_true", help="Pickle erlauben (nur vertrauenswürdige Dateien!)")
    sp.add_argument("--tiefe", type=int, default=2)
    sp.add_argument("--speichern", help="als Baukasten-Projekt speichern")
    sp.set_defaults(fn=cmd_importieren)

    sp = sub.add_parser("komprimieren", help="Modell eindampfen")
    sp.add_argument("quelle", help="Projektordner, Preset oder Datei")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--schritte", help='JSON-Liste, z. B. \'[{"name":"prune_ffn","args":{"amount":0.4}},"heal","quantize"]\'')
    sp.add_argument("--trainieren", action="store_true", help="Preset/Datei vorher trainieren")
    sp.add_argument("--speichern", help="komprimiertes Projekt speichern")
    sp.set_defaults(fn=cmd_komprimieren)

    sp = sub.add_parser("zielsysteme", help="Geräteprofile bzw. Übersicht für ein Modell")
    sp.add_argument("quelle", nargs="?", help="Projektordner, Preset oder Datei (leer = nur Profile)")
    sp.add_argument("--setze", action="append", default=[])
    sp.set_defaults(fn=cmd_zielsysteme)

    sp = sub.add_parser("reduzieren", help="Empfehlung für ein Zielsystem (und anwenden)")
    sp.add_argument("quelle")
    sp.add_argument("--ziel", default="smartphone")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--trainieren", action="store_true", help="Preset/Datei vorher trainieren")
    sp.add_argument("--anwenden", action="store_true", help="Empfehlung ausführen")
    sp.add_argument("--export", help="danach im passenden Format exportieren (Pfad ohne Endung)")
    sp.add_argument("--speichern", help="reduziertes Projekt speichern")
    sp.set_defaults(fn=cmd_reduzieren)

    sp = sub.add_parser("notfall", help="Notfall-Ausführung, wenn das Modell nicht passt")
    with_source(sp)
    sp.add_argument("--ziel", help="Zielsystem (Budget aus dem Profil), z. B. laptop_gpu")
    sp.add_argument("--budget-mb", type=float, help="verfügbarer Gerätespeicher in MB")
    sp.add_argument("--training", action="store_true", help="Optionen fürs Training statt für die Inferenz")
    sp.add_argument("--anwenden", metavar="KENNUNG", help="Option einschalten ('auto' = schnellste passende)")
    sp.add_argument("--alle", action="store_true", help="alle Optionen zeigen")
    sp.add_argument("--speichern", help="Konfiguration nach dem Anwenden speichern")
    sp.set_defaults(fn=cmd_notfall)

    sp = sub.add_parser("schueler", help="kleineres Schüler-Modell per Distillation")
    sp.add_argument("quelle")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--breite", type=float, default=0.5)
    sp.add_argument("--tiefe", type=float, default=0.5)
    sp.add_argument("--schritte", type=int)
    sp.add_argument("--features", type=float, default=0.0, help="Gewicht der Feature-Distillation")
    sp.add_argument("--trainieren", action="store_true", help="Lehrer (Preset/Datei) vorher trainieren")
    sp.add_argument("--speichern")
    sp.set_defaults(fn=cmd_schueler)

    sp = sub.add_parser("probelauf", help="alle Wege durchs Netz mit Statistiken")
    sp.add_argument("quelle")
    sp.add_argument("--setze", action="append", default=[])
    sp.add_argument("--batch", type=int, default=2)
    sp.add_argument("--tiefe", type=int, default=3)
    sp.add_argument("--ohne-backward", action="store_true")
    sp.add_argument("--test", action="store_true", help="zusätzlich Testlauf auf Validierungsdaten")
    sp.set_defaults(fn=cmd_probelauf)

    sp = sub.add_parser("moeglichkeiten", help="Katalog aller Möglichkeiten")
    sp.add_argument("-o", "--ausgabe", help="als Markdown-Datei schreiben (z. B. docs/MOEGLICHKEITEN.md)")
    sp.set_defaults(fn=cmd_moeglichkeiten)

    sp = sub.add_parser("trainieren", help="trainieren")
    with_source(sp)
    sp.add_argument("--speicher", help="Speicherprofil: standard, sparsam, extrem_sparsam, schnell")
    sp.add_argument("--epochen", type=int)
    sp.add_argument("--speichern", default=None, help="Zielordner (Standard: ausgaben/<name>; '-' = nicht speichern)")
    sp.set_defaults(fn=cmd_trainieren)

    sp = sub.add_parser("ablauf", help="ganzen Ablauf ausführen (workflow)")
    with_source(sp)
    sp.add_argument("--speichern", help="Zielordner für das fertige Projekt")
    sp.set_defaults(fn=cmd_ablauf)

    sp = sub.add_parser("automl", help="automatische Hyperparameter-Suche")
    with_source(sp)
    sp.add_argument("--methode", default="random", choices=["random", "grid", "evolution", "pso", "pbt"])
    sp.add_argument("--raum", help="Suchraum als JSON, z. B. '{\"training.optimizer.lr\": {\"min\": 1e-4, "
                                   "\"max\": 1e-2, \"log\": true}}' (leer = automatisch)")
    sp.add_argument("--versuche", type=int, default=12)
    sp.add_argument("--schritte", type=int, default=60, help="Trainingsschritte pro Versuch")
    sp.add_argument("--population", type=int, default=6)
    sp.add_argument("--generationen", type=int, default=3)
    sp.add_argument("--metrik", default="val_loss")
    sp.add_argument("--richtung", default="auto", choices=["auto", "min", "max"])
    sp.add_argument("--speichern", help="Konfiguration mit den besten Einstellungen speichern")
    sp.set_defaults(fn=cmd_automl)

    sp = sub.add_parser("verteilt", help="verteilt trainieren (mehrere GPUs/Rechner)")
    with_source(sp)
    sp.add_argument("--prozesse", type=int, default=2, help="Prozesse pro Rechner (meist = Anzahl GPUs)")
    sp.add_argument("--strategie", default="ddp", choices=["ddp", "zero1", "fsdp"])
    sp.add_argument("--knoten", type=int, default=1, help="Anzahl Rechner")
    sp.add_argument("--knoten-rang", type=int, default=0, help="Nummer dieses Rechners (0 … knoten-1)")
    sp.add_argument("--master", help="Adresse:Port von Rechner 0 (bei mehreren Rechnern)")
    sp.add_argument("--speichern", help="Zielordner für das trainierte Projekt")
    sp.add_argument("--nur-anzeigen", action="store_true", help="Befehl nur anzeigen, nicht starten")
    sp.set_defaults(fn=cmd_verteilt)

    sp = sub.add_parser("generieren", help="aus gespeichertem Projekt generieren")
    sp.add_argument("ordner")
    sp.add_argument("--prompt", default="Der ")
    sp.add_argument("--laenge", type=int, default=100)
    sp.add_argument("--temperatur", type=float, default=0.8)
    sp.add_argument("--top-k", type=int, default=None)
    sp.add_argument("--anzahl", type=int, default=16)
    sp.set_defaults(fn=cmd_generieren)

    sp = sub.add_parser("perfektion", help="Anforderungs-Katalog: Perfektions-Check eines Projekts")
    sp.add_argument("quelle", nargs="?", default="tabellen_klassifikation",
                    help="Preset, Konfigurationsdatei oder gespeicherter Projektordner")
    sp.add_argument("--setze", action="append", default=[], metavar="PFAD=WERT")
    sp.add_argument("--stufe", default="optional", choices=["pflicht", "empfohlen", "optional"],
                    help="ab welcher Wichtigkeit anzeigen/einschalten")
    sp.add_argument("--anwenden", action="store_true", help="alle schaltbaren offenen Anforderungen einschalten")
    sp.add_argument("--aufwendig", action="store_true", help="auch aufwendige Techniken (DP-SGD, föderiert …)")
    sp.add_argument("--katalog", action="store_true", help="ganzen Katalog als Markdown ausgeben")
    sp.add_argument("-o", "--ausgabe", help="Katalog in diese Datei schreiben")
    sp.add_argument("--speichern", help="geänderte Konfiguration speichern")
    sp.set_defaults(fn=cmd_perfektion)

    sp = sub.add_parser("datenpruefung", help="Daten prüfen (Lecks, Duplikate, Klassen …)")
    with_source(sp)
    sp.add_argument("--anwenden", action="store_true", help="Vorschläge übernehmen")
    sp.add_argument("--speichern", help="geänderte Konfiguration speichern")
    sp.set_defaults(fn=cmd_datenpruefung)

    sp = sub.add_parser("bewerten", help="Validierung, Test, Basis, Fehleranalyse, Kalibrierung")
    with_source(sp)
    sp.add_argument("--top", type=int, default=5, help="so viele sicherste Fehler zeigen")
    sp.add_argument("--kalibrieren", action="store_true", help="zusätzlich Temperature Scaling bestimmen")
    sp.set_defaults(fn=cmd_bewerten)

    sp = sub.add_parser("modellkarte", help="Modellkarte anzeigen/schreiben")
    sp.add_argument("quelle", help="gespeicherter Projektordner (oder Preset)")
    sp.add_argument("-o", "--ausgabe", help="als Datei schreiben")
    sp.set_defaults(fn=cmd_modellkarte)

    sp = sub.add_parser("drift", help="neue Daten mit den Trainingsdaten vergleichen")
    sp.add_argument("quelle", help="gespeicherter Projektordner (oder Preset)")
    sp.add_argument("neu", help="neue Daten: CSV (gleiche Spalten wie im Training) oder .pt-Datei mit Tensoren")
    sp.add_argument("--trenner", default=",", help="CSV-Trennzeichen")
    sp.add_argument("--ohne-modell", action="store_true", help="keine Vorhersage-Drift berechnen")
    sp.set_defaults(fn=cmd_drift)

    sp = sub.add_parser("bereitstellen", help="als HTTP-Dienst starten")
    sp.add_argument("quelle", help="gespeicherter Projektordner (oder Preset)")
    sp.add_argument("--host", default=None, help="Adresse (Standard 127.0.0.1 = nur lokal)")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--schluessel", default=None, help="API-Schlüssel (Kopfzeile X-API-Key)")
    sp.set_defaults(fn=cmd_bereitstellen)

    sp = sub.add_parser("doku", help="Parameter-Referenz exportieren")
    sp.add_argument("-o", "--ausgabe", default="docs/PARAMETER_REFERENZ.md")
    sp.set_defaults(fn=cmd_doku)

    sp = sub.add_parser("gui", help="grafische Oberfläche")
    sp.add_argument("quelle", nargs="?", default=None)
    sp.set_defaults(fn=cmd_gui)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
    except Exception as exc:  # verständliche Fehlermeldung statt Traceback
        from ..core import KIBaukastenError

        if isinstance(exc, (KIBaukastenError, KeyError, ValueError)) and not getattr(args, "debug", False):
            print(f"Fehler: {exc}", file=sys.stderr)
            sys.exit(2)
        raise
