"""
ki_baukasten.ui.docs
====================

Erzeugt die **vollständige Parameter-Referenz** als Markdown – direkt aus den
Tooltips im Code. Dadurch ist die Dokumentation immer aktuell: wer eine
Einstellung hinzufügt, dokumentiert sie automatisch mit.

Aufruf: ``python -m ki_baukasten doku -o docs/PARAMETER_REFERENZ.md``
"""

from __future__ import annotations

from pathlib import Path

from ..core import BaseConfig, Level, all_registries


def _esc(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _tooltip_cell(tip) -> str:
    parts = [f"**Was:** {tip.what}"]
    if tip.why:
        parts.append(f"**Nutzen:** {tip.why}")
    if tip.use_for:
        parts.append(f"**Wofür:** {tip.use_for}")
    if tip.tip:
        parts.append(f"**Tipp:** {tip.tip}")
    return _esc("<br>".join(parts))


def config_table(cfg: BaseConfig, prefix: str = "") -> list[str]:
    """Tabelle aller Einstellungen eines Konfigurationsobjekts (rekursiv)."""
    lines = ["| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |", "|---|---|---|---|---|---|"]
    for ref in cfg.walk(prefix=prefix):
        p = ref.param
        kind = getattr(p, "kind", "value")
        if kind in ("nested",):
            default = f"⟶ {type(ref.value).__name__}" if ref.value is not None else "None"
        elif kind == "component":
            default = f"`{ref.value.name}`" if ref.value is not None else "None"
        elif kind in ("component_list", "config_list"):
            default = f"{len(ref.value)} Elemente"
        else:
            default = f"`{_esc(repr(ref.value))[:60]}`"
        rules = _esc(p.constraint_label())
        if len(rules) > 160:  # lange Auswahllisten an einer Optionsgrenze kürzen
            rules = rules[:160].rsplit(" \\| ", 1)[0] + " \\| …"
        lines.append(f"| `{ref.path}` | {_esc(p.type_label())} | {default} | {rules} | "
                     f"{ref.level.label} | {_tooltip_cell(p.tooltip)} |")
    return lines


def export_markdown(path: str | Path = "docs/PARAMETER_REFERENZ.md") -> Path:
    from ..project import ProjectConfig

    out: list[str] = [
        "# KI-Baukasten – Parameter-Referenz",
        "",
        "Automatisch erzeugt aus den Tooltips im Quellcode (`python -m ki_baukasten doku`).",
        "Stufen: **Basis** → **Fortgeschritten** → **Experte** → **Experimentell** (nur Orientierung – alles ist änderbar).",
        "",
        "## Inhalt",
        "",
        "1. [Projekt-Konfiguration](#projekt-konfiguration)",
        "2. [Registries und Komponenten](#registries-und-komponenten)",
        "",
        "## Projekt-Konfiguration",
        "",
        "Standardwerte eines leeren Projekts (`ProjectConfig()`). Modell- und Komponenten-Argumente",
        "hängen von der gewählten Komponente ab – siehe Abschnitt *Registries und Komponenten*.",
        "",
    ]
    out += config_table(ProjectConfig())
    out += ["", "## Registries und Komponenten", ""]
    regs = all_registries()
    for kind, reg in regs.items():
        out.append(f"- [{reg.title} (`{kind}`)](#registry-{kind.replace('_', '-')})")
    for kind, reg in regs.items():
        out += ["", f"### Registry `{kind}`", "", f"<a id=\"registry-{kind.replace('_', '-')}\"></a>", "",
                f"**{reg.title}** – {reg.tooltip.what} {reg.tooltip.why}", ""]
        if reg.context_doc:
            out += [f"Fabrik-Signatur: `{reg.context_doc}`", ""]
        for comp in reg.components(Level.EXPERIMENTAL):
            alias = f" (Aliase: {', '.join(f'`{x}`' for x in comp.aliases)})" if comp.aliases else ""
            out += [f"#### `{kind}:{comp.name}`{alias} — Stufe {comp.level.label}", "",
                    _esc(f"**Was:** {comp.tooltip.what}"), ""]
            if comp.tooltip.why:
                out += [_esc(f"**Nutzen:** {comp.tooltip.why}"), ""]
            if comp.tooltip.use_for:
                out += [_esc(f"**Wofür:** {comp.tooltip.use_for}"), ""]
            if comp.tooltip.tip:
                out += [_esc(f"**Tipp:** {comp.tooltip.tip}"), ""]
            try:
                args = comp.config_cls()
            except Exception:
                args = None
            if args is not None and type(args)._params:
                out += config_table(args)
                out.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# ===========================================================================
# Möglichkeiten-Katalog
# ===========================================================================

_CURATED: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    ("Anpassen – jede Schicht, jedes Modul, jeder Tensor",
     "Was sich wie fein verändern lässt (jeweils für alle, nur bestimmte oder genau einen Treffer).", [
         ("Basis-Block", "`model.block.…`", "Bauplan aller Schichten (Mixer, FFN, Norm, Dropout …)"),
         ("Schichtweise Abweichungen", "`model.layer_overrides`", "Selektoren `0`, `-1`, `::2`, `2-5`, `gerade` …"),
         ("Übernahme zwischen Schichten", "`@von`, `@nur`, `@ausser`", "Konfiguration einer anderen Schicht ganz/teilweise"),
         ("Modul-Chirurgie", "`module_rules`", "ersetzen, entfernen, LoRA/Quantisierung einzeln, Präzision, Hooks"),
         ("Tensor-Chirurgie", "`tensor_rules`", "einfrieren, Pruning, Kopieren, eigene Lernraten, Beschränkungen"),
         ("Geltungsbereich", "`layers`, `match`, `scope`, `pick`", "alle · nur_auswahl · nur_einer · alle_ausser"),
         ("Platzhalter-Pfade", "`Project.set_all`, `set('model.**.x', …)`", "eine Einstellung überall setzen"),
         ("Suchen / Rückgängig / Vergleich", "`search`, `undo`, `redo`, `diff`", "Handling großer Konfigurationen"),
     ]),
    ("Speicher im Training reduzieren",
     "Wo der Speicher hingeht und welche Weiche hilft.", [
         ("Gemischte Präzision", "`memory.mixed_precision = bf16`", "halbiert Aktivierungen, schneller auf GPUs"),
         ("Reines bf16", "`memory.param_dtype = bfloat16`", "halbiert Gewichte und Gradienten"),
         ("Gradient-Checkpointing", "`memory.gradient_checkpointing`", "spart 60–80 % Aktivierungen (+30 % Zeit)"),
         ("Aktivierungs-Offload", "`memory.activation_offload`", "Aktivierungen im CPU-RAM"),
         ("Blöcke trainierbar im RAM", "`memory.offload.mode = stream`", "Gewichte, Gradienten, Optimierer im RAM"),
         ("8-Bit-Optimierer", "`training.optimizer = adamw8bit`", "¼ der Adam-Zustände"),
         ("LoRA / QLoRA", "`memory.lora`, `memory.quantization = nf4`", "nur kleine Adapter trainieren"),
         ("Akkumulation", "`training.gradient_accumulation_steps`", "große effektive Batches mit wenig Speicher"),
         ("Schneller SSM-Scan", "`scan_backward = fused`", "Mamba-Aktivierungen ÷ d_state"),
         ("Speicherprofile", "`Project.apply_memory_profile('extrem_sparsam')`", "alles auf einmal"),
         ("Schätzung", "`Project.memory_report()`", "Gewichte, Gradienten, Optimierer, gemessene Aktivierungen"),
     ]),
    ("Generierung & Inferenz",
     "Schneller und sparsamer Vorhersagen.", [
         ("KV-/Zustands-Cache", "`generate(..., use_cache=True)`", "nur neue Tokens rechnen (alle kausalen Mixer)"),
         ("Cache-Format", "`cache={'dtype': 'int8'}`", "¼ Cache-Speicher; Fenster + Attention-Sinks"),
         ("GQA / MLA-lite", "`num_kv_heads`, `kv_compression_dim`", "kleinerer KV-Cache schon beim Bau"),
         ("Kontextverlängerung", "`positional.scaling = yarn`", "längere Texte als im Training"),
         ("Layer-Swapping", "`memory.offload.mode = cpu | disk`", "Modelle größer als der Gerätespeicher"),
     ]),
    ("Lehrer-Schüler & Weiterlernen",
     "Wissen übertragen und neue Daten lernen, ohne Altes zu vergessen.", [
         ("Schüler trainieren", "`Project.distill(width, depth)`", "automatisch verkleinertes Modell lernt vom Lehrer"),
         ("Distillation", "`training.paradigm = distillation`", "KL, reverse KL, JS, MSE; Feature-Distillation"),
         ("Lehrer-Quellen", "`teacher_project`, `teacher_export`, `teacher`", "gespeichertes Projekt, .pt2, Modell"),
         ("Weiterlernen", "`Project.continue_training(method=…)`", "lwf · ewc · replay · none"),
     ]),
    ("Import & Export",
     "Vorhandene Modelle weiterverwenden und Ergebnisse ausliefern.", [
         ("Import", "`Project.import_model(pfad)`", ".pt2, TorchScript, state_dict, safetensors, Hugging Face"),
         ("Gewichte abbilden", "`key_map`, `architecture`", "fremde Schlüsselnamen auf Baukasten-Modelle"),
         ("Export", "`Project.export(pfad, fmt)`", "torch.export (dynamische Formen), TorchScript, ONNX, state_dict"),
         ("Speichern/Laden", "`save`, `load`", "inkl. Kompression, Quantisierung, LoRA – exakt wiederhergestellt"),
     ]),
    ("Einblick & Analyse",
     "Verstehen, was die KI tut.", [
         ("Aufbau", "`Project.architecture(fmt)`", "Text, Markdown + Mermaid, JSON; Anpassungspfade je Schicht"),
         ("Probelauf", "`analysis.probe.probe`, GUI „Probelauf & Gewichte“", "alle Wege, Statistiken, Gradienten, Warnungen"),
         ("Gewichte & Attention", "`weight_stats`, `heatmap`, `attention_maps`", "Verteilungen, Matrizen, Attention-Karten"),
         ("Testlauf", "`analysis.probe.evaluate_model`", "Metriken, Konfusionsmatrix, Beispiele"),
         ("Ratgeber", "`Project.advise()` / `apply_advice`", "Speicher, Tempo, Stabilität, Struktur"),
         ("Messung", "`analysis.benchmark`", "Größe, MACs, Latenz"),
     ]),
]


def capabilities_markdown() -> str:
    """Katalog aller Möglichkeiten (kuratierte Übersicht + automatisch aus den Registries)."""
    from ..analysis.targets import TARGETS
    from .cli import build_parser

    regs = all_registries()
    total = sum(len(r) for r in regs.values())
    out = ["# KI-Baukasten – alle Möglichkeiten", "",
           f"Automatisch erzeugt (`python -m ki_baukasten moeglichkeiten -o docs/MOEGLICHKEITEN.md`). "
           f"{total} Komponenten in {len(regs)} Registries – jede überall per Name einsetzbar. "
           "Einzelheiten zu jeder Einstellung: [PARAMETER_REFERENZ.md](PARAMETER_REFERENZ.md), "
           "innerer Aufbau: [ARCHITEKTUR.md](ARCHITEKTUR.md).", ""]
    for title, intro, rows in _CURATED:
        out += [f"## {title}", "", intro, "", "| Möglichkeit | Wo | Wirkung |", "|---|---|---|"]
        out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
        out.append("")
    out += ["## Modelle eindampfen (Kompression)", "",
            "`Project.compress([...])` bzw. `compression.steps`; Reihenfolge: Struktur → Heilen → Low-Rank/Sparsity → "
            "Quantisierung. Jeder Schritt wird gemessen (Parameter, Größe, MACs, Latenz, Verlust).", "",
            "| Schritt | Wirkung | Wofür |", "|---|---|---|"]
    for name in regs["compress_step"].names():
        t = regs["compress_step"].get(name).tooltip
        out.append(f"| `{name}` | {_esc(t.what)} {_esc(t.why)} | {_esc(t.use_for)} |")
    out += ["", "## Zielsysteme", "",
            "`Project.recommend_for(ziel)` zeigt das Rezept, `reduce_for(ziel)` wendet es an; nach `train()` erscheint "
            "automatisch eine Übersicht (`deployment.targets`).", "",
            "| Schlüssel | Gerät | Speicher | Leistung (Batch 1) | Formate | Export |", "|---|---|---:|---:|---|---|"]
    for k, t in TARGETS.items():
        out.append(f"| `{k}` | {t.title} | {t.memory_mb:g} MB | ~{t.gflops:g} GFLOP/s | {', '.join(t.dtypes)} | "
                   f"{t.export} |")
    out += ["", "## Bausteine (Registries)", ""]
    for kind, reg in regs.items():
        names = ", ".join(f"`{n}`" for n in reg.names())
        out.append(f"- **{reg.title}** (`{kind}`, {len(reg)}): {names}")
    out += ["", "## Kommandozeile", "", "| Befehl | Zweck |", "|---|---|"]
    parser = build_parser()
    for action in parser._subparsers._group_actions:  # noqa: SLF001 - argparse bietet keine öffentliche Liste
        for choice in action._choices_actions:  # noqa: SLF001
            out.append(f"| `{choice.dest}` | {_esc(choice.help or '')} |")
    out += ["", "Grafische Oberfläche: `python -m ki_baukasten gui` – Einstellungsbaum mit Tooltips, Prüfen, Bauen, "
            "Speicher, Trainieren, Aufbau, Optimieren, Probelauf & Gewichte, Reduzieren/Export, Modell importieren, "
            "Rückgängig/Wiederholen.", ""]
    return "\n".join(out)


def export_capabilities(path: str | Path = "docs/MOEGLICHKEITEN.md") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(capabilities_markdown(), encoding="utf-8")
    return path
