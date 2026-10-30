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
    out += _overview_sections()
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
    out += ["", "Grafische Oberfläche: `python -m ki_baukasten gui` – Menü (Datei · Assistent · Einstellungen · "
            "Techniken · Ausführen · Analyse · Qualität · Einsatz · Hilfe; „Datei → Automatisch aus Daten…“), "
            "Ziel-Assistent, Techniken-Fenster mit "
            "Reitern, Perfektions-Check, Datenprüfung, Test, Basis, Fehleranalyse, Kalibrierung, Modellkarte, "
            "HTTP-Dienst, "
            "Einstellungsbaum mit Tooltips und An/Aus/Eigen-Schaltern je Einstellung, Ansichtsfilter, Probelauf & "
            "Gewichte, Reduzieren/Export mit Notfall-Ausführung, Modell importieren, Rückgängig/Wiederholen.", ""]
    return "\n".join(out)


def _overview_sections() -> list[str]:
    """Bedienkonzept, Ziel-Assistent, Techniken-Katalog, Notfall-Ausführung, Logger, Cluster, Ablauf."""
    from ..analysis.goals import PRIORITIES, TASKS
    from ..techniques import CATALOG, CATEGORIES

    out = ["## Bedienung in drei Ebenen", "",
           "So bleibt es trotz über tausend Einstellungen übersichtlich:", "",
           "| Ebene | Wofür | Wo |", "|---|---|---|",
           "| 1. Ziele | Aufgabe, wichtige Punkte, Zielgerät angeben → Modell wird empfohlen, Unpassendes abgeraten | "
           "`Project.from_goals(...)`, CLI `ziele`, GUI „Neue KI nach Zielen“ |",
           "| 2. Techniken | bewährte Verfahren mit einem Häkchen an/aus (Status ● ○ ◐ –) | "
           "`Project.enable/disable`, CLI `techniken`, GUI „Techniken“ |",
           "| 3. Einstellungen | jede einzelne Einstellung: An · Aus · Eigen, mit Tooltip | "
           "`Project.switch/set`, CLI `schalter`, GUI-Formular + Ansichtsfilter |", "",
           "Jede Einstellung hat einen Zustand **an** (empfohlener Wert), **aus** (z. B. Dropout 0, Komponente "
           "`none`) oder **eigen** (eigener Wert); Pflicht-Einstellungen (z. B. `d_model`) kennen nur an/eigen. "
           "`p.set('**.dropout', 'aus')`, `p.switch('training.regularization', 'aus')`, `p.switch_overview()`.", "",
           "## Ziel-Assistent", "",
           "`Project.from_goals(task=…, priorities=[…], data_size=…, target=…, description='…')` bewertet alle "
           "Vorlagen mit Begründung (✓ Stärken, ✗ Schwächen), rät von unpassenden ab, schlägt Techniken vor und "
           "speichert die Ziele im Projekt (`goals`) – später änderbar: `set_goals(...)`, `goal_check()`.", "",
           "| Aufgabe | Beschreibung |", "|---|---|"]
    out += [f"| `{k}` | {v} |" for k, v in TASKS.items()]
    out += ["", "Prioritäten: " + ", ".join(f"`{k}` ({v})" for k, v in PRIORITIES.items()) + ".", "",
            "## Techniken-Katalog", "",
            f"{len(CATALOG)} Techniken, die Communitys, Forschung und Firmen nutzen – jede an/abwählbar "
            "(`p.enable('qlora')`, `p.disable('qlora')` stellt den vorherigen Zustand exakt wieder her).", ""]
    for cat, (title, desc) in CATEGORIES.items():
        out += [f"### {title}", "", desc, "", "| Kennung | Technik | Wer nutzt es | Hinweis |", "|---|---|---|---|"]
        out += [f"| `{t.key}` | {_esc(t.title)} | {_esc(t.used_by)} | {_esc(t.hint)} |" for t in CATALOG
                if t.category == cat]
        out.append("")
    out += ["## Notfall-Ausführung (wenn es trotz Verkleinern nicht passt)", "",
            "`Project.fallback_options(ziel | budget_mb, training)`, `apply_fallback('auto' | kennung)`, "
            "`reduce_for(ziel, fallback='auto')`, CLI `notfall`, GUI „Reduzieren / Export → Notfall-Ausführung“.", "",
            "| Kennung | Wie | Modus |", "|---|---|---|",
            "| `micro_batch` | Eingaben stückweise (`predict(chunk_size=…)`, `memory.oom_strategy`) | Inferenz |",
            "| `split` | hintere Blöcke dauerhaft auf der CPU (`offload.mode=split`, device_map) | Inferenz/Training |",
            "| `ram_offload` | Blöcke im RAM, einzeln aufs Gerät (`offload.mode=cpu`) | Inferenz |",
            "| `disk_offload` / `layerweise_stapel` | Schicht für Schicht von der SSD (Memory-Map) | Inferenz |",
            "| `quantisieren_nf4` | 4-Bit-Gewichte ohne Training | Inferenz |",
            "| `sparsam` | Checkpointing + Mikro-Batches + Akkumulation + bf16 | Training |",
            "| `optimierer_8bit` | 8-Bit-AdamW | Training |",
            "| `lisa` | nur wenige, wechselnde Blöcke trainieren | Training |",
            "| `lora` / `qlora` | nur Adapter trainieren (Basis ggf. 4 Bit) | Training |",
            "| `stream` | Gewichte/Gradienten/Optimierer im RAM, Blöcke gestreamt | Training |",
            "| `stream_disk_lora` | eingefrorene Basis auf der SSD, Adapter im RAM | Training |",
            "| `cpu` | auf der CPU rechnen | beides |",
            "| `cluster` | FSDP über mehrere Geräte | Training |", "",
            "Zusätzlich: Batch automatisch teilen bei Speichermangel (`memory.oom_strategy='batch_teilen'`), "
            "Chunked Prefill für lange Prompts (`generate(..., prefill_chunk=512)`), Laden per Memory-Map "
            "(`Project.load(ordner, mmap=True)`).", "",
            "## Logger", "",
            "`training.logging`: Stufe `still` · `minimal` · `normal` · `ausfuehrlich` · `debug`; Ziele frei kombinierbar "
            "(`console`, `datei` mit Rotation, `jsonl`, `csv`, `python` (Standard-logging), `tensorboard`, `wandb`, "
            "`mlflow`), eigene Stufe je Ziel, Intervalle, Durchsatz/Speicher/Lernrate/Gradientennorm, Statistiken je "
            "Tensor. Eigene Logger: `projekt.add_logger(objekt_oder_logging.Logger)` oder Registry `log_backend`.", "",
            "## Cluster & verteiltes Training", "",
            "`training.distributed.strategy`: `ddp` (Daten parallel), `zero1` (Optimierer verteilt), `fsdp` (alles "
            "verteilt). Start: `python -m ki_baukasten verteilt projekt.json --prozesse 4 --strategie fsdp` (mehrere "
            "Rechner: `--knoten`, `--knoten-rang`, `--master`). Ausgaben und Speichern nur auf Prozess 0.", "",
            "## Ablauf: Suche, Schwarm, Evolution", "",
            "`Project.run()` führt `workflow` aus: Datenprüfung → Suche (`random`, `grid`, `evolution`, `pso`, `pbt`) "
            "→ Training (normal, föderiert FedAvg/FedProx, Model Soup) → Test → Basis-Vergleich → Kalibrierung → "
            "Robustheit/Fairness → Kreuzvalidierung → Kompression → Export → Speichern. Einzeln: `tune(...)`, "
            "`federated_train(...)`, `model_soup(...)`, `cross_validate(k)`, `find_lr()`. Gradientenfrei: Paradigma "
            "`evolution_strategies`.", ""]
    out += _quality_sections()
    return out


def _quality_sections() -> list[str]:
    """Anforderungs-Katalog, Daten, Bewertung, Vertrauen, Einsatz."""
    from ..requirements import CATALOG, PHASES, coverage

    cov = coverage()
    out = ["## Anforderungs-Katalog & Perfektions-Check", "",
           f"{len(CATALOG)} Anforderungen in {len(PHASES)} Phasen ({', '.join(PHASES.values())}), je Anwendung; "
           f"{cov['vorhanden']} davon sind im Baukasten umsetzbar. `project.perfection_check()` zeigt für das eigene "
           "Projekt aktiv · eingebaut · aus · offen · fehlt mit Punktzahl, `apply_perfection()` schaltet alles "
           "Empfohlene ein, CLI `perfektion`, GUI „Qualität“. Vollständig: [ANFORDERUNGEN.md](ANFORDERUNGEN.md).", "",
           "## Daten, Bewertung, Vertrauen, Einsatz", "",
           "| Bereich | Möglichkeiten | Wo |", "|---|---|---|",
           "| Eigene Daten | CSV (Kategorien, fehlende Werte + Indikator, Skalierung, Ziel-Skalierung), Texte mit Labels "
           "(Tokenizer zeichen/wort/bpe), Bildordner, Segmentierungs-Masken, Audio (WAV → Log-Mel), Anweisungen (SFT) | "
           "`data = csv`, `text_classification`, `image_folder`, `segmentation`, `audio`, `instructions` |",
           "| Aufteilung | zufällig, stratifiziert, zeitlich; Testmenge, die beim Neutrainieren fest bleibt | "
           "`training.validation_strategy`, `training.test_split`, `Project.test()` |",
           "| Datenprüfung & Drift | Lecks, Duplikate, Überlappung, Klassen, NaN, Skalen, abgeschnittene Texte; PSI/KS "
           "je Merkmal, Vorhersage-Drift | `data_check()`, `apply_data_fixes()`, `drift_report()` |",
           "| Metriken | 27 Metriken (Registry `metric`), automatisch je Aufgabe, über alle Beispiele, Richtung für "
           "Early Stopping/Suche automatisch | `training.metrics`, `training.auto_metrics` |",
           "| Bewertung | Basis-Vergleich, Fehleranalyse (Konfusionsmatrix, sicherste Fehler), Klassengewichte | "
           "`baseline()`, `error_analysis()`, `training.class_weights` |",
           "| Vertrauen | Temperature Scaling, Schwelle, Enthaltung, Conformal Prediction, MC-Dropout, TTA | "
           "`inference.*`, `calibrate()`, `predict_with_uncertainty()` |",
           "| Erklärbarkeit & Fairness | Gradient × Eingabe, Integrated Gradients, Occlusion (auch Tokens), "
           "Permutations-Wichtigkeit; Disparate Impact, Equalized Odds | `explain_prediction()`, "
           "`feature_importance()`, `fairness_report()` |",
           "| Robustheit | FGSM/PGD-Training, Robustheitsbericht (Rauschen/Angriffe) | `training.adversarial`, "
           "`robustness_report()` |",
           "| Wenige Labels | Active Learning (Unsicherheit, Marge, Entropie, Vielfalt), Pseudo-Labeling | "
           "`select_for_labeling()`, `pseudo_label()` |",
           "| Domänen | U-Net-Segmentierung, Audio-Klassifikation, Instruction-Tuning (Verlust nur auf Antworten), "
           "Beam Search, n-Gramm-Sperre | `model = unet`, `ask()`, `generation.*` |",
           "| Einsatz | Modellkarte, Umgebung, gespeicherte Vorverarbeitung, Rohdaten-Vorhersage, HTTP-Dienst, "
           "Fortsetzen vom Checkpoint | `model_card()`, `save/load`, `predict_raw()`, `serve()`, "
           "`training.resume_from` |", "",
           "## Leistung, Hardware & schlanker Export", "",
           "| Bereich | Möglichkeiten | Wo |", "|---|---|---|",
           "| Ein Aufruf | Datenart erkennen → Vorlage → Empfehlungen → Training → Test → Basis | `Project.auto(datei)`, "
           "CLI `auto` |",
           "| Hardware-Automatik | bf16/fp16/aus, TF32, cuDNN-Benchmark, channels_last, gepinnter Speicher je Gerät | "
           "`memory.* = auto`, `hardware_report()`, CLI `hardware` |",
           "| Tempo | fusionierter Optimierer, `TensorBatchLoader`, eine Synchronisation je Schritt, GQA ohne KV-Kopien | "
           "automatisch |",
           "| Automatisch einstellen | Lernrate (LR-Finder), Batchgröße passend zum GPU-Speicher | `training.auto_lr`, "
           "`training.auto_batch_size` |",
           "| Schlanker Export | Hüllen entfernen, EMA, LoRA exakt, BatchNorm falten, Ungenutztes entfernen, Präzision, "
           "Prüfung, kompakte projekt.json | `deployment.export.*`, `export(...)`, `last_export` |", "",
           "## Diagnose & automatisch verbessern", "",
           "| Bereich | Möglichkeiten | Wo |", "|---|---|---|",
           "| Diagnose | instabil, lernt kaum, zu kurz, Überanpassung, kaum besser als Basis, Klassen-Ungleichgewicht, "
           "Kalibrierung, keine Testmenge, Validierung zu optimistisch – je mit Maßnahmen | `diagnose()`, "
           "`apply_diagnosis()`, CLI `diagnose`, GUI „Qualität“ |",
           "| Verbessern | Diagnose → Maßnahmen → neu trainieren, nur Verbesserungen bleiben | `improve(rounds)`, "
           "`Project.auto(improve=…)`, CLI `auto --verbessern` |",
           "| Wirkungsprüfung | geänderte, aber wirkungslose Einstellungen mit Grund und Abhilfe; Technik-Konflikte | "
           "`inactive_settings()`, Meldung vor `train()` |", ""]
    return out


def export_capabilities(path: str | Path = "docs/MOEGLICHKEITEN.md") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(capabilities_markdown(), encoding="utf-8")
    return path
