"""
ki_baukasten.ui.cli
===================

Kommandozeile: ``python -m ki_baukasten <befehl> …`` (oder ``ki-baukasten …``).

Befehle
-------
==============  =================================================================
Befehl          Zweck
==============  =================================================================
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
trainieren      trainieren (und speichern)
generieren      gespeichertes Projekt laden und Text/Samples erzeugen
doku            vollständige Parameter-Referenz als Markdown exportieren
gui             grafische Oberfläche starten
==============  =================================================================

Quelle ``QUELLE`` ist entweder ein Preset-Name oder eine Konfigurationsdatei.
Mit ``--setze pfad=wert`` (mehrfach) lassen sich Einstellungen überschreiben,
z. B. ``--setze model.d_model=256 --setze training.optimizer.lr=1e-3``.
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
        p.set(path.strip(), _parse_value(value.strip()))
    return p


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


def cmd_trainieren(a: argparse.Namespace) -> None:
    p = _project(a.quelle, a.setze)
    if a.speicher:
        p.apply_memory_profile(a.speicher)
    if a.epochen:
        p.set("training.epochs", a.epochen)
    p.train()
    if p.config.data.name == "anomaly":
        print("Anomalie-Erkennung:", p.anomaly_report())
    target = p.save(a.speichern) if a.speichern != "-" else None
    if target:
        print(f"Gespeichert in: {target}")


def cmd_generieren(a: argparse.Namespace) -> None:
    from ..project import Project

    p = Project.load(a.ordner)
    model = p.model
    if hasattr(model, "generate") and hasattr(model, "lm_head"):
        print(p.generate(a.prompt, max_new_tokens=a.laenge, temperature=a.temperatur, top_k=a.top_k))
    else:
        samples = p.sample(a.anzahl)
        print(samples[: min(10, a.anzahl)])


def cmd_doku(a: argparse.Namespace) -> None:
    from .docs import export_markdown

    path = export_markdown(a.ausgabe)
    print(f"Parameter-Referenz geschrieben: {path}")


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

    sp = sub.add_parser("trainieren", help="trainieren")
    with_source(sp)
    sp.add_argument("--speicher", help="Speicherprofil: standard, sparsam, extrem_sparsam, schnell")
    sp.add_argument("--epochen", type=int)
    sp.add_argument("--speichern", default=None, help="Zielordner (Standard: ausgaben/<name>; '-' = nicht speichern)")
    sp.set_defaults(fn=cmd_trainieren)

    sp = sub.add_parser("generieren", help="aus gespeichertem Projekt generieren")
    sp.add_argument("ordner")
    sp.add_argument("--prompt", default="Der ")
    sp.add_argument("--laenge", type=int, default=100)
    sp.add_argument("--temperatur", type=float, default=0.8)
    sp.add_argument("--top-k", type=int, default=None)
    sp.add_argument("--anzahl", type=int, default=16)
    sp.set_defaults(fn=cmd_generieren)

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
