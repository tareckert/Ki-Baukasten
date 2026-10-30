"""
ki_baukasten.requirements
=========================

**Anforderungs-Katalog & Perfektions-Check** – was eine KI je nach Anwendung braucht, um wirklich
gut (und verlässlich) zu werden, und ob das im Projekt umgesetzt ist.

Jede Anforderung gehört zu einer **Phase** (Daten → Modell → Training → Bewertung → Vertrauen →
Einsatz) und gilt für alle oder bestimmte **Anwendungen** (Tabellen, Bilder, Sprache, Zeitreihen,
Audio, RL …). Für ein Projekt ergibt sich je Anforderung ein Status:

==========  ==================================================================================
Status      Bedeutung
==========  ==================================================================================
aktiv       eingeschaltet bzw. durchgeführt
eingebaut   passiert automatisch (nichts zu tun)
aus         vorhanden, aber ausgeschaltet → mit ``apply_perfection()`` bzw. ``enable(…)`` an
offen       ein Schritt, der noch nicht ausgeführt wurde (z. B. ``project.test()``)
fehlt       (noch) nicht im Baukasten – mit ehrlichem Hinweis auf Alternativen
==========  ==================================================================================

Nutzung::

    report = project.perfection_check()     # Bewertung mit Punktzahl und nächsten Schritten
    print(report)
    project.apply_perfection()              # alle empfohlenen, schaltbaren Anforderungen einschalten
    print(requirements_markdown())          # ganzer Katalog (docs/ANFORDERUNGEN.md)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

PHASES: dict[str, str] = {
    "daten": "Daten", "modell": "Modell", "training": "Training", "bewertung": "Bewertung",
    "vertrauen": "Vertrauen & Sicherheit", "einsatz": "Einsatz & Betrieb",
}
IMPORTANCE = {"pflicht": 3, "empfohlen": 2, "optional": 1}
DOMAINS: dict[str, str] = {
    "alle": "alle Anwendungen", "datensatz": "mit Datensatz (nicht RL)", "wenige_labels": "wenige Labels",
    "tabellen": "Tabellen", "bild": "Bilder", "segmentierung": "Segmentierung",
    "sprache": "Sprache/Text", "generierung": "Textgenerierung", "zeitreihen": "Zeitreihen", "audio": "Audio",
    "klassifikation": "Klassifikation", "regression": "Regression/Prognose", "rl": "Reinforcement Learning",
    "generativ": "Generative Modelle", "graph": "Graphen", "anomalie": "Anomalien", "menschen": "Entscheidungen "
    "über Menschen", "detektion": "Objekterkennung", "empfehlung": "Empfehlungssysteme",
}
MARK = {"aktiv": "✓", "eingebaut": "✓", "aus": "○", "offen": "◇", "fehlt": "✗"}


@dataclass(frozen=True)
class Requirement:
    key: str
    title: str
    phase: str
    domains: tuple[str, ...]
    importance: str
    why: str
    how: str                                                    # Umsetzung im Baukasten
    techniques: tuple[str, ...] = ()                            # schaltbar über diese Techniken
    active: Callable[[Any], bool | None] | None = None          # eigener Status-Test (Projekt → an/aus/None)
    builtin: bool = False                                       # passiert automatisch
    step: str = ""                                              # Methode, die ausgeführt werden muss
    available: Callable[[], bool] | None = None                 # gibt es das im Baukasten?
    gap: str = ""                                               # ehrlicher Hinweis, falls es fehlt


@dataclass
class Check:
    requirement: Requirement
    status: str
    hint: str = ""


@dataclass
class PerfectionReport:
    project: str
    domains: list[str]
    checks: list[Check] = field(default_factory=list)

    @property
    def score(self) -> float:
        """0–100: gewichteter Anteil erfüllter Anforderungen (Pflicht 3×, empfohlen 2×, optional 1×)."""
        total = done = 0.0
        for c in self.checks:
            w = IMPORTANCE[c.requirement.importance]
            total += w
            done += w if c.status in ("aktiv", "eingebaut") else 0.0
        return 100.0 * done / total if total else 100.0

    def open(self, min_importance: str = "empfohlen") -> list[Check]:
        limit = IMPORTANCE[min_importance]
        return [c for c in self.checks if c.status in ("aus", "offen") and IMPORTANCE[c.requirement.importance]
                >= limit]

    def missing(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fehlt"]

    def switchable(self, min_importance: str = "empfohlen") -> list[str]:
        """Techniken, die die offenen Anforderungen erfüllen würden (je Anforderung die erste passende)."""
        keys = []
        for c in self.open(min_importance):
            if c.status == "aus" and c.hint.startswith("enable:"):
                key = c.hint.split(":", 1)[1]
                if key not in keys:
                    keys.append(key)
        return keys

    def __str__(self) -> str:
        n = {s: sum(c.status == s for c in self.checks) for s in MARK}
        head = (f"Perfektions-Check '{self.project}' ({', '.join(DOMAINS.get(d, d) for d in self.domains)}): "
                f"{self.score:.0f} / 100 Punkte")
        lines = [head, "=" * len(head),
                 f"✓ {n['aktiv'] + n['eingebaut']} erfüllt · ○ {n['aus']} ausgeschaltet · ◇ {n['offen']} offen · "
                 f"✗ {n['fehlt']} nicht im Baukasten"]
        for phase, title in PHASES.items():
            rows = [c for c in self.checks if c.requirement.phase == phase]
            if not rows:
                continue
            lines.append(f"\n■ {title}")
            for c in rows:
                r = c.requirement
                imp = {"pflicht": "Pflicht", "empfohlen": "empfohlen", "optional": "optional"}[r.importance]
                lines.append(f"  {MARK[c.status]} {r.title}  [{imp}]")
                if c.status in ("aus", "offen", "fehlt"):
                    hint = c.hint
                    if hint.startswith("enable:"):
                        hint = f"einschalten: project.enable('{hint.split(':', 1)[1]}')"
                    lines.append(f"      → {hint}")
        todo = self.switchable()
        if todo:
            lines.append(f"\nAlle empfohlenen einschalten: project.apply_perfection()  ({', '.join(todo)})")
        steps = [c.hint for c in self.open() if c.status == "offen"]
        if steps:
            lines.append("Noch auszuführen: " + " · ".join(dict.fromkeys(steps)))
        return "\n".join(lines)


# ===========================================================================
# Hilfen für Status-Tests
# ===========================================================================

def _cfg(p: Any) -> Any:
    return p.config


def _data(p: Any) -> str:
    return p.config.data.name


def _args(p: Any) -> Any:
    return p.config.data.args


def _has(kind: str, name: str) -> Callable[[], bool]:
    def check() -> bool:
        from .core import registry

        return name in registry(kind)
    return check


def _attr(path: str) -> Callable[[], bool]:
    def check() -> bool:
        import importlib

        mod, _, name = path.rpartition(".")
        try:
            return hasattr(importlib.import_module(mod), name)
        except ImportError:
            return False
    return check


def _project_has(name: str) -> Callable[[], bool]:
    def check() -> bool:
        from .project import Project

        return hasattr(Project, name)
    return check


def _never() -> bool:
    return False


R = Requirement
_CATALOG: list[Requirement] = [
    # ------------------------------------------------------------------ Daten
    R("datenqualitaet", "Datenqualität geprüft (Lecks, Duplikate, NaN, Klassen, Skalen)", "daten", ("datensatz",), "pflicht",
      "Die meisten KI-Fehler sind Datenfehler.", "workflow.check_data bzw. project.data_check()",
      techniques=("datenpruefung",), active=lambda p: p.config.workflow.check_data or p.last_data_report is not None,
      available=_project_has("data_check")),
    R("ehrliche_aufteilung", "Ehrliche Aufteilung mit Testmenge", "daten", ("datensatz",), "pflicht",
      "Nur nie benutzte Daten zeigen die echte Qualität.", "training.test_split + project.test()",
      techniques=("testmenge",)),
    R("aufteilung_klassen", "Stratifizierte Aufteilung", "daten", ("klassifikation",), "empfohlen",
      "Seltene Klassen sollen in allen Teilmengen vorkommen.", "training.validation_strategy = 'stratifiziert'",
      techniques=("stratifiziert",)),
    R("aufteilung_zeit", "Zeitliche Aufteilung (keine Zukunft im Training)", "daten", ("zeitreihen",), "pflicht",
      "Zufällige Aufteilung verrät bei Zeitreihen die Zukunft.", "training.validation_strategy = 'zeitlich'",
      techniques=("zeitliche_aufteilung",)),
    R("leckfreie_vorverarbeitung", "Vorverarbeitung nur auf Trainingsdaten gelernt", "daten", ("tabellen", "sprache",
                                                                                               "bild", "audio"),
      "pflicht", "Skalen/Vokabular aus Validierungsdaten wären ein Datenleck.",
      "csv/text_classification/image_folder/audio lernen Skalen, Kategorien und Tokenizer nur auf Trainingszeilen",
      builtin=True),
    R("fehlende_werte", "Fehlende Werte behandeln (+ Indikator)", "daten", ("tabellen",), "pflicht",
      "Lücken dürfen das Training nicht zerstören und sind oft selbst informativ.",
      "data = 'csv': missing (mittelwert/median/null/zeile_entfernen), missing_indicator", builtin=True,
      active=lambda p: _data(p) == "csv" or None),
    R("kategorien", "Kategorie-Spalten kodieren", "daten", ("tabellen",), "pflicht",
      "Texte wie 'Berlin' müssen zu Zahlen werden, ohne falsche Reihenfolge.", "data = 'csv': encoding onehot/ordinal",
      builtin=True, active=lambda p: _data(p) == "csv" or None),
    R("skalierung", "Merkmale skalieren", "daten", ("tabellen",), "empfohlen",
      "Unterschiedliche Wertebereiche bremsen das Lernen.", "data = 'csv': scaling standard/robust/minmax",
      builtin=True, active=lambda p: (getattr(_args(p), "scaling", "keine") != "keine") if _data(p) == "csv" else None),
    R("unausgewogen", "Unausgewogene Klassen ausgleichen", "daten", ("klassifikation",), "empfohlen",
      "Sonst ignoriert das Modell seltene Klassen.", "training.class_weights oder training.balance_classes",
      techniques=("klassengewichte", "klassen_ausgleich")),
    R("augmentierung", "Daten-Augmentierung", "daten", ("bild", "sprache", "audio"), "empfohlen",
      "Mehr Vielfalt aus denselben Daten, weniger Auswendiglernen.", "training.augmentation",
      techniques=("bild_augmentierung", "random_erasing", "token_rauschen", "cutmix")),
    R("tokenizer", "Kompakter Tokenizer (BPE)", "daten", ("sprache",), "empfohlen",
      "Kürzere Folgen, keine unbekannten Wörter.", "data.tokenizer = 'bpe'", builtin=True,
      techniques=("bpe_tokenizer",)),
    R("eigene_daten", "Eigene Daten ohne Code laden", "daten", ("datensatz",), "pflicht",
      "Echte Projekte brauchen echte Daten.", "Datensätze csv, image_folder, text_classification, segmentation, "
      "audio, instructions, tensor_file", builtin=True, available=_has("dataset", "csv")),
    R("wenige_labels", "Mit wenigen Labels umgehen", "daten", ("wenige_labels",), "empfohlen",
      "Beschriften ist teuer.", "select_for_labeling (Active Learning), pseudo_label, selbstüberwachtes Vortraining",
      step="project.select_for_labeling(pool) / project.pseudo_label(unbeschriftet)",
      available=_project_has("pseudo_label")),

    # ------------------------------------------------------------------ Modell
    R("modellwahl", "Passendes Modell zur Aufgabe gewählt", "modell", ("alle",), "pflicht",
      "Die Architektur bestimmt, was überhaupt lernbar ist.", "Ziel-Assistent: project.set_goals(…) / goal_check()",
      step="project.set_goals(task=…) und project.goal_check()",
      active=lambda p: p.config.goals.task != "unbekannt" or None),
    R("regularisierung", "Regularisierung (Dropout, Weight Decay)", "modell", ("datensatz",), "empfohlen",
      "Gegen Auswendiglernen.", "model.**.dropout, training.optimizer.weight_decay",
      techniques=("dropout", "weight_decay", "drop_path")),
    R("vortrainiert", "Vortrainiertes Modell nutzen können", "modell", ("sprache", "bild"), "optional",
      "Vortraining spart Daten und Rechenzeit.", "Project.import_model + LoRA/QLoRA/schichtweise Lernraten",
      builtin=True, available=_project_has("import_model")),
    R("segmentierungsmodell", "Segmentierungsmodell (U-Net)", "modell", ("segmentierung",), "pflicht",
      "Eine Klasse je Pixel braucht Encoder-Decoder mit Skip-Verbindungen.", "model = 'unet'",
      active=lambda p: p.config.model.name in ("unet", "cnn"), available=_has("model", "unet")),
    R("lange_kontexte", "Lange Kontexte effizient", "modell", ("sprache", "zeitreihen"), "optional",
      "Attention wächst quadratisch.", "Mamba/SSM, Hybrid, RoPE, KV-Cache, Chunked Prefill", builtin=True,
      available=_has("model", "ssm")),
    R("objekterkennung", "Objekterkennung (Bounding Boxes)", "modell", ("detektion",), "pflicht",
      "Objekte finden und einrahmen.", "–", available=_never,
      gap="Kein Detektor (YOLO/DETR) enthalten – Alternativen: Segmentierung mit U-Net oder Export + spezialisiertes "
          "Framework."),
    R("empfehlungssystem", "Empfehlungssystem (Nutzer × Artikel)", "modell", ("empfehlung",), "pflicht",
      "Personalisierte Vorschläge.", "–", available=_never,
      gap="Keine Matrixfaktorisierung/Two-Tower-Vorlage – Näherung: Tabellenmodell auf Nutzer-/Artikel-Merkmalen "
          "oder kontrastives Lernen."),

    # ------------------------------------------------------------------ Training
    R("stabilitaet", "Stabiles Training (Clipping, NaN-Schutz)", "training", ("alle",), "pflicht",
      "Ein Ausreißer darf nicht Stunden Training zerstören.", "training.gradient_clipping, training.nan_policy",
      techniques=("gradient_clipping", "nan_schutz", "agc")),
    R("lernrate", "Lernraten-Zeitplan mit Warmup", "training", ("datensatz",), "empfohlen",
      "Die Lernrate ist die wichtigste Einstellung.", "training.scheduler (warmup_cosine, wsd …), find_lr()",
      techniques=("warmup", "wsd")),
    R("early_stopping", "Early Stopping / beste Gewichte behalten", "training", ("datensatz",), "empfohlen",
      "Stoppt, bevor das Auswendiglernen beginnt.", "callback early_stopping (monitor val_loss/val_f1 …)",
      techniques=("early_stopping",)),
    R("reproduzierbar", "Reproduzierbar (Seed)", "training", ("alle",), "pflicht",
      "Ergebnisse müssen wiederholbar und vergleichbar sein.", "training.seed", techniques=("seed",)),
    R("checkpoints", "Checkpoints und Fortsetzen", "training", ("alle",), "empfohlen",
      "Abbrüche kosten sonst die ganze Arbeit.", "callback checkpoint, training.resume_from = 'auto'",
      techniques=("checkpoints", "fortsetzen")),
    R("hardware", "Hardware ausgenutzt (Präzision, TF32, fusionierter Optimierer …)", "training", ("alle",),
      "pflicht", "Gleiche Hardware, gleiches Tempo wie von Hand optimierte Modelle.",
      "memory.mixed_precision/matmul_precision = 'auto', fusionierter Optimierer, SDPA, schneller Tensor-Lader",
      active=lambda p: p.config.memory.mixed_precision != "none" or p.config.memory.matmul_precision != "highest"
      or None, builtin=True),
    R("lernrate_bestimmt", "Lernrate bestimmt (LR-Finder)", "training", ("datensatz",), "optional",
      "Eine passende Lernrate spart viele Versuche.", "training.auto_lr bzw. project.find_lr()",
      techniques=("auto_lernrate",)),
    R("hyperparameter", "Hyperparameter-Suche", "training", ("alle",), "optional",
      "Gute Einstellungen automatisch finden.", "workflow.search (random/grid/evolution/pso/pbt), project.tune()",
      techniques=("zufallssuche", "partikelschwarm", "genetisch", "pbt", "gittersuche")),
    R("protokoll", "Verlauf protokollieren", "training", ("alle",), "empfohlen",
      "Nachvollziehen, was passiert ist.", "training.logging (CSV/JSONL/TensorBoard/W&B/MLflow)",
      techniques=("log_datei", "log_jsonl", "tensorboard", "wandb", "mlflow")),
    R("speicher", "Große Modelle auf kleiner Hardware", "training", ("alle",), "optional",
      "Hardware ist oft der Engpass.", "Checkpointing, 8-Bit-Optimierer, LoRA/QLoRA, Streaming, Notfall-Ausführung",
      builtin=True, available=_project_has("fallback_options")),
    R("weiterlernen", "Weiterlernen ohne Vergessen", "training", ("alle",), "optional",
      "Neue Daten kommen – altes Wissen soll bleiben.", "project.continue_training(method='lwf'|'ewc'|'replay')",
      builtin=True, available=_project_has("continue_training")),
    R("instruction_tuning", "Instruction-Tuning (Verlust nur auf Antworten)", "training", ("generierung",), "empfohlen",
      "Ein Assistent soll antworten, nicht Fragen nachplappern.", "data = 'instructions' (loss_mask), project.ask()",
      active=lambda p: _data(p) == "instructions" or None, available=_has("dataset", "instructions")),
    R("alignment", "Präferenz-Lernen (DPO/Reward-Modell)", "training", ("generierung",), "optional",
      "Antworten an menschliche Vorlieben anpassen.", "paradigm 'dpo' / 'preference'", builtin=True,
      available=_has("paradigm", "dpo")),
    R("rl_belohnung", "Belohnungen formen und kombinieren", "training", ("rl",), "pflicht",
      "Die Belohnung definiert, was der Agent lernt.", "training.paradigm.rewards (Liste gewichteter Terme)",
      builtin=True, available=_has("paradigm", "reinforcement")),
    R("rl_auswertung", "Auswertung über mehrere Episoden (deterministisch)", "bewertung", ("rl",), "pflicht",
      "Eine Episode ist Zufall – erst der Mittelwert vieler zeigt die Qualität.",
      "training.paradigm.eval_episodes, eval_deterministic", builtin=True),

    # ------------------------------------------------------------------ Bewertung
    R("metriken", "Passende Metriken (nicht nur Verlust)", "bewertung", ("datensatz",), "pflicht",
      "Genauigkeit lügt bei seltenen Klassen; Prognosen brauchen Fehler in echten Einheiten.",
      "training.metrics / training.auto_metrics", techniques=("metriken_auto",)),
    R("testbewertung", "Endbewertung auf der Testmenge", "bewertung", ("datensatz",), "pflicht",
      "Der ehrliche Abschluss.", "project.test() (bzw. workflow.run_test)", step="project.test()",
      active=lambda p: bool(p.last_test) or None),
    R("baseline", "Vergleich mit einfacher Basis", "bewertung", ("klassifikation", "regression"), "empfohlen",
      "Schlägt das Modell nicht einmal 'häufigste Klasse'/'Mittelwert', hat es nichts gelernt.",
      "project.baseline()", step="project.baseline()", active=lambda p: bool(p.last_baseline) or None,
      available=_project_has("baseline")),
    R("fehleranalyse", "Fehleranalyse (Konfusionsmatrix, sicherste Fehler)", "bewertung", ("klassifikation",
                                                                                          "regression"), "optional",
      "Zeigt, *wo* das Modell scheitert – oft Datenfehler.", "project.error_analysis()",
      step="project.error_analysis()", available=_project_has("error_analysis")),
    R("kreuzvalidierung", "Kreuzvalidierung bei wenig Daten", "bewertung", ("tabellen",), "optional",
      "Eine einzelne Aufteilung ist bei wenig Daten Zufall.", "workflow.cross_validation / cross_validate()",
      techniques=("kreuzvalidierung",)),
    R("segmentierungsmetriken", "IoU/Dice für Segmentierung", "bewertung", ("segmentierung",), "pflicht",
      "Pixel-Genauigkeit wird vom Hintergrund dominiert.", "training.metrics ['iou', 'dice'] (auto)",
      techniques=("metriken_auto",)),

    # ------------------------------------------------------------------ Vertrauen & Sicherheit
    R("kalibrierung", "Kalibrierte Wahrscheinlichkeiten", "vertrauen", ("klassifikation",), "empfohlen",
      "'90 % sicher' soll in 90 % der Fälle stimmen.", "inference.auto_calibrate / project.calibrate()",
      techniques=("kalibrierung",), active=lambda p: bool(p.last_calibration) or None),
    R("unsicherheit", "Unsicherheit angeben", "vertrauen", ("klassifikation", "regression"), "empfohlen",
      "Im Einsatz zählt auch, wie sicher die Antwort ist.", "Conformal Prediction, MC-Dropout, TTA, Enthaltung",
      techniques=("conformal", "mc_dropout", "tta", "enthaltung")),
    R("erklaerbarkeit", "Entscheidungen erklärbar", "vertrauen", ("tabellen", "bild", "sprache"), "empfohlen",
      "Vertrauen, Fehlersuche, Vorgaben (z. B. EU AI Act).", "project.explain_prediction / feature_importance",
      step="project.explain_prediction(x) / project.feature_importance()",
      available=_project_has("explain_prediction")),
    R("robustheit", "Robust gegen Störungen", "vertrauen", ("tabellen", "bild", "audio"), "optional",
      "Kleine Änderungen dürfen das Ergebnis nicht kippen.", "training.adversarial, workflow.robustness_check",
      techniques=("robustheitspruefung", "adversarial_training", "fgsm_training", "eingaberauschen"),
      active=lambda p: True if p.last_robustness is not None else None),
    R("fairness", "Fairness geprüft", "vertrauen", ("menschen",), "pflicht",
      "Entscheidungen über Menschen dürfen Gruppen nicht benachteiligen.", "project.fairness_report(gruppen)",
      step="project.fairness_report('spalte')",
      active=lambda p: (p.last_fairness is not None or bool(p.config.workflow.fairness_column)) or None),
    R("datenschutz", "Datenschutz (DP-SGD / föderiert)", "vertrauen", ("menschen",), "optional",
      "Einzelne Personen dürfen nicht rekonstruierbar sein.", "training.privacy, workflow.federated",
      techniques=("dp_sgd", "foederiert")),
    R("generierung_qualitaet", "Saubere Generierung (Beam Search, Wiederholungssperre)", "vertrauen",
      ("generierung",), "optional", "Kleine Sprachmodelle wiederholen sich gern.", "project.generation",
      techniques=("keine_wiederholungen", "beam_search")),

    # ------------------------------------------------------------------ Einsatz & Betrieb
    R("vorverarbeitung_gespeichert", "Vorverarbeitung mit dem Modell gespeichert", "einsatz", ("datensatz",), "pflicht",
      "Im Einsatz müssen Rohdaten genau wie im Training aufbereitet werden.",
      "save() → vorverarbeitung.json; load() + predict_raw()", builtin=True, available=_project_has("predict_raw")),
    R("schlanker_export", "Export ohne Ballast (geprüft)", "einsatz", ("alle",), "pflicht",
      "Ein Einsatz-Modell soll nur enthalten, was die Vorhersage braucht.",
      "deployment.export: Trainings-Hüllen weg, EMA, LoRA einrechnen, BatchNorm falten, Ungenutztes entfernen, "
      "Ausgabe gegen das Original geprüft",
      active=lambda p: (p.config.deployment.export.strip_unused and p.config.deployment.export.verify) or None,
      builtin=True),
    R("modellkarte", "Modellkarte und Umgebung dokumentiert", "einsatz", ("alle",), "empfohlen",
      "Zweck, Daten, Ergebnisse, Grenzen, Versionen – nachvollziehbar.", "deployment.model_card",
      techniques=("modellkarte",)),
    R("zielgeraet", "Für das Zielgerät verkleinert", "einsatz", ("alle",), "optional",
      "Das Modell muss dort laufen, wo es gebraucht wird.", "deployment.main_target, recommend_for/reduce_for",
      techniques=("komprimieren_nach_training",),
      active=lambda p: True if p.config.deployment.main_target else None),
    R("export", "Export in ein Standardformat", "einsatz", ("alle",), "optional",
      "Einsatz ohne Python/PyTorch (ONNX, TorchScript …).", "workflow.export_format / project.export()",
      techniques=("export_nach_training",)),
    R("bereitstellung", "Als Dienst bereitstellen (HTTP)", "einsatz", ("alle",), "optional",
      "Apps und andere Programme nutzen die KI.", "project.serve() / python -m ki_baukasten bereitstellen",
      builtin=True, available=_attr("ki_baukasten.ui.serve.serve")),
    R("drift", "Drift im Einsatz überwachen", "einsatz", ("datensatz",), "empfohlen",
      "Ändern sich die Daten, wird die KI unbemerkt schlechter.", "project.drift_report(neue_daten)",
      step="project.drift_report(neue_daten) (regelmäßig im Einsatz)", available=_project_has("drift_report")),
    R("versionierung", "Modell-Versionen nachvollziehbar", "einsatz", ("alle",), "optional",
      "Welches Modell lief wann mit welchen Daten?", "Konfigurations-Prüfsumme + Modellkarte; MLflow-Logger",
      builtin=True, available=_attr("ki_baukasten.analysis.modelcard.config_hash"),
      gap="Keine eigene Modell-Registry – Prüfsumme/Modellkarte oder MLflow verwenden."),
]

CATALOG: list[Requirement] = _CATALOG
BY_KEY = {r.key: r for r in CATALOG}


# ===========================================================================
# Anwendungen eines Projekts
# ===========================================================================

def project_domains(project: Any) -> list[str]:
    """Für welche Anwendungen gelten die Anforderungen? (aus Vorlage, Daten, Paradigma, Zielen)"""
    cfg = project.config
    d = {"alle"}
    data, model, paradigm = cfg.data.name, cfg.model.name, cfg.training.paradigm.name
    if data != "none" and paradigm != "reinforcement":
        d.add("datensatz")
    if cfg.goals.labels in ("wenige", "keine"):
        d.add("wenige_labels")
    info_task = ""
    try:
        info_task = str(project._info().get("task", "")) if data != "none" else ""
    except Exception:  # noqa: BLE001 - Daten evtl. nicht ladbar
        pass
    domain = cfg.domain
    if paradigm == "reinforcement":
        d.add("rl")
    elif paradigm in ("causal_lm", "dpo", "masked_lm", "seq2seq"):
        d.update({"sprache", "generierung"} if paradigm != "masked_lm" else {"sprache"})
    elif paradigm in ("gan", "diffusion", "autoencoder"):
        d.add("anomalie" if data == "anomaly" else "generativ")
    if paradigm == "supervised":
        if info_task == "segmentation" or data == "segmentation" or model == "unet":
            d.update({"bild", "segmentierung"})
        elif info_task == "regression" or data in ("regression", "time_series"):
            d.add("regression")
        else:
            d.add("klassifikation")
    if domain in ("tabellen", "bild", "sprache", "zeitreihen", "audio", "graph") or data in ("csv",):
        d.add("tabellen" if data == "csv" else domain)
    if data in ("text_classification", "sequence_classification", "instructions", "text"):
        d.add("sprache")
    if data in ("image_folder", "shapes"):
        d.add("bild")
    if data == "time_series" or cfg.goals.task == "zeitreihe_prognose":
        d.add("zeitreihen")
    g = cfg.goals
    text = (g.description or "").lower()
    if "datenschutz" in g.priorities or any(w in text for w in ("kunde", "patient", "bewerb", "kredit", "person",
                                                              "mensch", "mitarbeit")):
        d.add("menschen")
    if any(w in text for w in ("objekterkennung", "bounding", "objekte finden", "detektion")):
        d.add("detektion")
    if any(w in text for w in ("empfehl", "recommend")):
        d.add("empfehlung")
    d.discard("allgemein")
    return sorted(d, key=list(DOMAINS).index)


# ===========================================================================
# Prüfung
# ===========================================================================

def _status(project: Any, r: Requirement, states: dict[str, str]) -> Check:
    if r.available is not None and not r.available():
        return Check(r, "fehlt", r.gap or "Noch nicht im Baukasten.")
    custom = r.active(project) if r.active is not None else None
    if custom is True:
        return Check(r, "aktiv")
    tech = [states.get(k, "nicht_anwendbar") for k in r.techniques]
    if any(s == "an" for s in tech):
        return Check(r, "aktiv")
    if r.techniques and custom is None and not r.step and not r.builtin and \
            all(s == "nicht_anwendbar" for s in tech):
        return Check(r, "nicht_anwendbar")           # passt nicht zu diesem Modell/diesen Daten
    usable = [k for k, s in zip(r.techniques, tech) if s in ("aus", "teilweise")]
    if usable:
        return Check(r, "aus", f"enable:{usable[0]}")
    if custom is False:
        return Check(r, "aus", r.how)
    if r.step:
        return Check(r, "offen", r.step)
    if r.builtin:
        return Check(r, "eingebaut")
    return Check(r, "offen", r.how)


def perfection_check(project: Any, min_importance: str = "optional") -> PerfectionReport:
    """Bewertet ein Projekt gegen alle Anforderungen seiner Anwendungen."""
    domains = project_domains(project)
    states = {s.technique.key: s.status for s in project.techniques()}
    limit = IMPORTANCE[min_importance]
    report = PerfectionReport(project.config.name, domains)
    for r in CATALOG:
        if IMPORTANCE[r.importance] < limit or not set(r.domains) & set(domains):
            continue
        check = _status(project, r, states)
        if check.status != "nicht_anwendbar":
            report.checks.append(check)
    return report


def coverage() -> dict[str, Any]:
    """Abdeckung des **Baukastens** (unabhängig vom Projekt): welche Anforderungen sind umsetzbar?"""
    from .techniques import BY_KEY as TECH

    rows = []
    for r in CATALOG:
        ok = (r.available() if r.available is not None else True) and all(k in TECH for k in r.techniques)
        rows.append((r, ok))
    return {"vorhanden": sum(ok for _, ok in rows), "gesamt": len(rows), "zeilen": rows}


def requirements_markdown() -> str:
    """Der ganze Katalog als Markdown (docs/ANFORDERUNGEN.md)."""
    cov = coverage()
    lines = ["# Anforderungen an eine perfekte KI – und wie der Baukasten sie erfüllt", "",
             "Automatisch erzeugt aus `ki_baukasten/requirements.py` "
             "(`python -m ki_baukasten doku` bzw. `perfektion --katalog`).", "",
             f"**Abdeckung: {cov['vorhanden']} von {cov['gesamt']} Anforderungen sind im Baukasten umsetzbar.** "
             "Für ein konkretes Projekt prüft `project.perfection_check()`, was davon eingeschaltet ist; "
             "`project.apply_perfection()` schaltet alle empfohlenen, schaltbaren Punkte ein.", "",
             "Legende: **Pflicht** – ohne das ist das Ergebnis unzuverlässig · **empfohlen** – Stand der Praxis · "
             "**optional** – je nach Lage. Anwendungen: " + ", ".join(f"`{k}` = {v}" for k, v in DOMAINS.items()) + ".",
             ""]
    for phase, title in PHASES.items():
        lines += [f"## {title}", "", "| | Anforderung | gilt für | Stufe | Warum | Umsetzung im Baukasten | "
                  "an/abwählbar |", "|---|---|---|---|---|---|---|"]
        for r, ok in cov["zeilen"]:
            if r.phase != phase:
                continue
            how = r.how if ok else f"**fehlt:** {r.gap}"
            switch = ", ".join(f"`{k}`" for k in r.techniques) or ("automatisch" if r.builtin else
                                                                   (f"`{r.step.split('(')[0]}`" if r.step else "–"))
            lines.append(f"| {'✓' if ok else '✗'} | {r.title} | {', '.join(r.domains)} | {r.importance} | {r.why} | "
                         f"{how} | {switch} |")
        lines.append("")
    gaps = [r for r, ok in cov["zeilen"] if not ok]
    if gaps:
        lines += ["## Ehrliche Lücken", ""] + [f"- **{r.title}**: {r.gap}" for r in gaps] + [""]
    return "\n".join(lines)


def apply(project: Any, min_importance: str = "empfohlen", include_heavy: bool = False) -> list[str]:
    """Schaltet für alle offenen Anforderungen (ab ``min_importance``) die erste passende Technik ein."""
    from .analysis.goals import HEAVY

    done = []
    for key in perfection_check(project).switchable(min_importance):
        if key in HEAVY and not include_heavy:
            continue
        try:
            if project.technique(key, "an"):
                done.append(key)
        except Exception:  # noqa: BLE001 - unpassend für dieses Projekt → überspringen
            continue
    return done
