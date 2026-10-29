"""
ki_baukasten.workflow.config
============================

Der **Ablauf** eines Projekts: was ``Project.run()`` nacheinander ausführt – jede Stufe an- und abschaltbar.

::

    Suche (Zufall · Gitter · Evolution · Partikelschwarm · PBT)
      → Training  (normal · föderiert · Model Soup aus mehreren Läufen)
      → Kreuzvalidierung (optional)
      → Kompression (optional) → Export (optional)
"""

from __future__ import annotations

from ..core import BaseConfig, Level, NestedParam, Param, T


class SearchConfig(BaseConfig):
    __title__ = "Automatische Suche (AutoML)"
    __tooltip__ = T(
        "Hyperparameter (und Architektur-Werte) automatisch suchen.",
        "Findet gute Einstellungen ohne Handarbeit – mit Zufall, Gitter, Evolution, Partikelschwarm oder "
        "Population Based Training.",
        "Neue Aufgaben, Feinabstimmung vor dem großen Lauf.",
    )
    method = Param("none", choices={
        "none": "keine Suche",
        "random": "Zufallssuche (starke, einfache Basis)",
        "grid": "Gittersuche (alle Kombinationen, wenige Werte)",
        "evolution": "genetischer Algorithmus (Selektion, Kreuzung, Mutation)",
        "pso": "Partikelschwarm-Optimierung (Schwarmintelligenz)",
        "pbt": "Population Based Training (DeepMind): Population trainiert parallel, schlechte kopieren gute",
    }, tooltip=T(
        "Suchverfahren.", "Zufall ist überraschend stark; Evolution/PSO nutzen Erfahrungen früherer Versuche; PBT "
        "passt Hyperparameter während des Trainings an.",
        "Kleines Budget: random. Mittleres: evolution/pso. Lange Läufe: pbt."))
    space = Param({}, tooltip=T(
        "Suchraum: {pfad: [werte]} oder {pfad: {'min':…, 'max':…, 'log': true, 'type': 'int'}} (leer = automatisch).",
        "Bestimmt, was variiert wird – jede Einstellung des Projekts ist möglich.",
        "z. B. {'training.optimizer.lr': {'min': 1e-4, 'max': 1e-2, 'log': true}, 'model.dropout': [0, 0.1]}."))
    trials = Param(12, min=1, tooltip=T(
        "Anzahl Versuche (random/grid) bzw. Obergrenze.", "Mehr Versuche = bessere Chancen, mehr Rechenzeit.", "10–100."))
    population = Param(6, min=2, tooltip=T(
        "Populations-/Schwarmgröße (evolution, pso, pbt).", "Vielfalt der Suche.", "4–20."))
    generations = Param(3, min=1, tooltip=T(
        "Generationen bzw. Runden (evolution, pso, pbt).", "Wie oft aus den Ergebnissen gelernt wird.", "3–10."))
    steps_per_trial = Param(60, min=1, tooltip=T(
        "Trainingsschritte pro Versuch (pbt: pro Runde).", "Kurze Versuche sparen Zeit, sind aber verrauschter.",
        "50–500."))
    metric = Param("val_loss", tooltip=T(
        "Zielgröße (aus der Validierung, sonst Trainingsverlust).", "Was optimiert wird.",
        "val_loss, val_accuracy, val_f1, val_rmse (alle Metriken aus training.metrics)."))
    mode = Param("auto", choices={"auto": "aus dem Metriknamen ableiten (Verlust/Fehler: min, Genauigkeit/F1: max)",
                                  "min": "kleiner ist besser", "max": "größer ist besser"}, tooltip=T(
        "Richtung der Verbesserung.", "Falsche Richtung = die Suche optimiert das Schlechteste.", "auto."))
    seed = Param(0, min=0, tooltip=T("Zufalls-Seed der Suche.", "Reproduzierbar.", "0."), level=Level.ADVANCED)
    apply_best = Param(True, tooltip=T(
        "Beste Einstellungen danach ins Projekt übernehmen.", "Direkt weiterarbeiten.", "an."))


class FederatedConfig(BaseConfig):
    __title__ = "Föderiertes Lernen"
    __tooltip__ = T(
        "Viele Teilnehmer trainieren lokal auf ihren eigenen Daten; nur Gewichte werden zusammengeführt.",
        "Daten verlassen nie das Gerät (Datenschutz) – wie bei Smartphone-Tastaturen (Google Gboard).",
        "Verteilte, sensible Daten: Kliniken, Geräte, Filialen. Hier simuliert durch Aufteilen der Daten.",
    )
    enabled = Param(False, tooltip=T("Föderiert trainieren.", "Simulation mehrerer Teilnehmer.", "Datenschutz."))
    clients = Param(4, min=2, tooltip=T("Anzahl Teilnehmer.", "Mehr = realistischer, langsamer.", "4–100."))
    rounds = Param(5, min=1, tooltip=T("Kommunikationsrunden.", "Wie oft zusammengeführt wird.", "5–100."))
    local_steps = Param(20, min=1, tooltip=T(
        "Lokale Trainingsschritte je Runde und Teilnehmer.", "Mehr = weniger Kommunikation, mehr Abdrift.", "10–100."))
    client_fraction = Param(1.0, min=0.01, max=1.0, tooltip=T(
        "Anteil der Teilnehmer pro Runde.", "Realistisch: nicht alle sind immer online.", "0.1–1."))
    algorithm = Param("fedavg", choices={
        "fedavg": "Federated Averaging (gewichteter Mittelwert)",
        "fedprox": "FedProx: zusätzlicher Strafterm hält lokale Modelle nahe am globalen (robuster bei ungleichen Daten)",
    }, tooltip=T("Verfahren.", "fedprox hilft bei sehr unterschiedlichen Teilnehmern.", "fedavg."))
    prox_mu = Param(0.01, min=0.0, tooltip=T("Stärke des FedProx-Terms.", "Nähe zum globalen Modell.", "0.001–0.1."))
    split = Param("iid", choices={"iid": "zufällig gleich verteilt", "nicht_iid": "ungleich (Dirichlet über Klassen)"},
                  tooltip=T("Wie die Daten auf Teilnehmer verteilt werden.", "nicht_iid ist realistischer und schwerer.",
                            "Test der Robustheit: nicht_iid."))
    dirichlet_alpha = Param(0.5, min=0.01, tooltip=T(
        "Ungleichheit bei nicht_iid (klein = sehr ungleich).", "Steuert die Datenverteilung.", "0.1–1."),
        level=Level.ADVANCED)


class WorkflowConfig(BaseConfig):
    __title__ = "Ablauf (Project.run)"
    __tooltip__ = T(
        "Was Project.run() nacheinander ausführt: Suche → Training → Kreuzvalidierung → Kompression → Export.",
        "Ein Befehl für den ganzen Weg zur fertigen KI; jede Stufe einzeln schaltbar.",
        "Reproduzierbare Pipelines, Automatisierung (python -m ki_baukasten ablauf …).",
    )
    search = NestedParam(SearchConfig, group="Suche")
    federated = NestedParam(FederatedConfig, group="Training")
    soup_runs = Param(0, min=0, tooltip=T(
        "Model Soup: so viele Läufe vom selben Startpunkt (verschiedene Seeds) trainieren und Gewichte mitteln "
        "(0 = aus).", "Mittelung mehrerer Läufe verbessert oft Genauigkeit und Robustheit ohne Mehrkosten bei der "
        "Nutzung (Wortsman et al. 2022).", "3–8."), group="Training", on=4)
    soup_mode = Param("greedy", choices={
        "uniform": "alle Läufe gleich gewichtet mitteln",
        "greedy": "nur Läufe hinzunehmen, die die Validierung verbessern",
    }, group="Training", level=Level.ADVANCED, tooltip=T("Art der Mittelung.", "greedy ist robuster.", "greedy."))
    cross_validation = Param(0, min=0, tooltip=T(
        "Zusätzlich k-fache Kreuzvalidierung als Bericht (0 = aus).", "Ehrliche Schätzung der Generalisierung, "
        "zeigt die Streuung.", "Kleine Datensätze: 5."), group="Prüfung", on=5)
    compress_after_training = Param(False, tooltip=T(
        "Nach dem Training komprimieren (compression.steps, sonst Empfehlung für deployment.main_target).",
        "Fertiges Modell direkt verkleinert.", "Einsatz auf Geräten."), group="Abschluss")
    export_format = Param("none", choices=["none", "export", "torchscript", "onnx", "state_dict"], group="Abschluss",
                          tooltip=T("Am Ende exportieren (none = nicht).", "Fertige Datei für den Einsatz.", "export."))
    export_path = Param(None, type=str, nullable=True, group="Abschluss", tooltip=T(
        "Zielpfad ohne Endung (None = output_dir/name/modell).", "Ablage.", "None."))
