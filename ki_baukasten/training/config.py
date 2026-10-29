"""
ki_baukasten.training.config
============================

``TrainingConfig`` – alle Stellräder des Trainings an einem Ort.

Gliederung
----------
* **Paradigma** – *welche Art* von Training (überwacht, Sprachmodell, GAN,
  Diffusion, Distillation, Präferenzen, Reinforcement Learning ...)
* **Umfang** – Epochen, Schritte, Batchgröße, Gradientenakkumulation
* **Optimierung** – Optimierer, Lernraten-Zeitplan, Parametergruppen
  (Weight-Decay-Ausnahmen, schichtweise Lernraten)
* **Verluste** – frei kombinierbare Liste gewichteter Verlustterme
* **Regularisierung** – L1/L2, Mixup, Rauschen, Gradientenrauschen, Flooding, SAM
* **Stabilität** – Gradient-Clipping, NaN-Behandlung, EMA der Gewichte
* **Curriculum** – Daten von leicht nach schwer
* **Ablauf** – Evaluierung, Logging, Callbacks, Seeds, DataLoader
* **Cluster** – verteiltes Training (DDP, ZeRO-1, FSDP)
"""

from __future__ import annotations

from ..core import (BaseConfig, ComponentListParam, ComponentParam, Level, NestedParam, Param, T)
from .distributed import DistributedConfig


class RegularizationConfig(BaseConfig):
    __title__ = "Regularisierung"
    __tooltip__ = T(
        "Zusätzliche Techniken gegen Überanpassung und für robustere Modelle.",
        "Ergänzen Dropout/Weight-Decay um weitere, teils experimentelle Verfahren.",
        "Kleine Datensätze, Generalisierung, Robustheit.",
    )
    l1 = Param(0.0, min=0.0, tooltip=T(
        "L1-Strafe auf alle Gewichte (λ·Σ|w|).", "Erzeugt spärliche Gewichte (viele exakt 0).",
        "Merkmalsauswahl, kompakte Modelle.", "Klein halten: 1e-6…1e-4."))
    l2 = Param(0.0, min=0.0, tooltip=T(
        "Explizite L2-Strafe im Verlust (λ·Σw²) – zusätzlich zu weight_decay des Optimierers.",
        "Klassische Ridge-Regularisierung (wirkt bei Adam anders als AdamW-Decay).", "Selten nötig."),
        level=Level.ADVANCED)
    orthogonality = Param(0.0, min=0.0, tooltip=T(
        "Strafe für Abweichung von Orthogonalität der Gewichtsmatrizen ‖WᵀW − I‖².",
        "Hält Signale normerhaltend, stabilisiert tiefe Netze und GANs.", "BigGAN, tiefe Netze."),
        level=Level.EXPERT)
    mixup_alpha = Param(0.0, min=0.0, tooltip=T(
        "Mixup: Beispiele und Ziele linear mischen (λ ~ Beta(α, α)); 0 = aus.",
        "Glattere Entscheidungsgrenzen, bessere Kalibrierung.", "Bildklassifikation: 0.2–0.4.",
        "Nur für Eingaben mit Gleitkommazahlen; Ziele werden zu weichen Labels."), level=Level.ADVANCED)
    input_noise_std = Param(0.0, min=0.0, tooltip=T(
        "Gauß-Rauschen auf Eingaben im Training.", "Robustheit gegen Messrauschen.", "Sensor-/Tabellendaten."),
        level=Level.ADVANCED)
    gradient_noise_std = Param(0.0, min=0.0, tooltip=T(
        "Rauschen auf Gradienten (σ_t = σ/(1+t)^γ).", "Hilft aus schlechten Minima (Neelakantan et al.).",
        "Sehr tiefe/schwer trainierbare Netze."), level=Level.EXPERIMENTAL)
    gradient_noise_decay = Param(0.55, min=0.0, tooltip=T(
        "Abklingexponent γ des Gradientenrauschens.", "Rauschen nimmt über die Zeit ab.", "0.55."),
        level=Level.EXPERIMENTAL)
    flooding_level = Param(None, type=float, min=0.0, tooltip=T(
        "Flooding: Verlust wird um diesen Pegel gespiegelt (|L−b|+b).",
        "Verhindert, dass der Trainingsverlust gegen 0 geht – wirkt gegen Überanpassung.",
        "Kleine Datensätze (Ishida et al. 2020).",
    ), level=Level.EXPERIMENTAL)
    sam_rho = Param(0.0, min=0.0, tooltip=T(
        "Sharpness-Aware Minimization: Radius der Nachbarschaft (0 = aus).",
        "Sucht flache Minima → bessere Generalisierung; kostet doppelte Rechenzeit.",
        "Bildklassifikation (ρ ≈ 0.05).",
        "Nicht mit gradient_accumulation_steps > 1 kombinierbar.",
    ), level=Level.EXPERIMENTAL)


class GradientClippingConfig(BaseConfig):
    __title__ = "Gradient-Clipping"
    mode = Param("norm", choices={
        "none": "kein Clipping",
        "norm": "Gesamtnorm aller Gradienten begrenzen (Standard)",
        "value": "jeden Gradienten elementweise begrenzen",
        "adaptive": "Adaptive Gradient Clipping (AGC, NFNets): relativ zur Gewichtsnorm",
    }, tooltip=T("Art des Gradient-Clippings.", "Verhindert explodierende Gradienten.",
                 "Transformer/RNNs: norm (1.0)."))
    max_norm = Param(1.0, min=0.0, tooltip=T(
        "Maximale Gradienten-Gesamtnorm (mode=norm).", "Begrenzt Schrittgrößen bei Ausreißern.", "0.5–5; LLMs 1.0."))
    clip_value = Param(1.0, min=0.0, tooltip=T(
        "Maximaler Betrag je Gradient (mode=value).", "Hartes Abschneiden.", "Selten."), level=Level.ADVANCED)
    agc_ratio = Param(0.01, min=0.0, tooltip=T(
        "Erlaubtes Verhältnis ‖g‖/‖w‖ je Einheit (mode=adaptive).", "Skaleninvariantes Clipping.", "0.01–0.16."),
        level=Level.EXPERT)
    norm_type = Param(2.0, min=0.0, tooltip=T(
        "Norm-Typ (2 = euklidisch, inf = Maximum).", "Art der Größenmessung.", "2."), level=Level.EXPERT)


class EMAConfig(BaseConfig):
    __title__ = "EMA der Gewichte"
    enabled = Param(False, tooltip=T(
        "Exponentiell gleitenden Mittelwert der Gewichte mitführen.",
        "Gemittelte Gewichte sind oft besser und stabiler als die letzten.",
        "Diffusionsmodelle (Pflicht), GANs, Bildklassifikation.",
    ))
    decay = Param(0.999, min=0.0, max=1.0, tooltip=T(
        "Abklingrate (näher an 1 = längeres Gedächtnis).", "Glättungsstärke.", "0.999–0.9999."))
    start_step = Param(0, min=0, tooltip=T(
        "Erst ab diesem Schritt mitteln.", "Frühe, schlechte Gewichte auslassen.", "Ein paar hundert Schritte."),
        level=Level.ADVANCED)
    update_every = Param(1, min=1, tooltip=T("Aktualisierung alle n Schritte.", "Spart Rechenzeit.", "1–10."),
                         level=Level.ADVANCED)
    use_for_eval = Param(True, tooltip=T(
        "EMA-Gewichte für Evaluierung verwenden.", "Bewertet das gemittelte Modell.", "Standard: an."))


class CurriculumConfig(BaseConfig):
    __title__ = "Curriculum Learning"
    strategy = Param("none", choices={
        "none": "zufällige Reihenfolge (Standard)",
        "length": "kurze Sequenzen zuerst (Schwierigkeit = Länge der Eingabe)",
        "key": "Schwierigkeit aus Datenfeld 'difficulty' (kleiner = leichter)",
    }, tooltip=T("Curriculum-Strategie.", "Leicht → schwer kann Training beschleunigen und stabilisieren.",
                 "Sprachmodelle (kurz → lang), Robotik, Mathe-Aufgaben."), level=Level.ADVANCED)
    start_fraction = Param(0.2, min=0.01, max=1.0, tooltip=T(
        "Anteil der leichtesten Daten zu Beginn.", "Startumfang des Curriculums.", "0.1–0.3."), level=Level.ADVANCED)
    full_after_fraction = Param(0.5, min=0.0, max=1.0, tooltip=T(
        "Ab diesem Trainingsfortschritt werden alle Daten genutzt.", "Länge der Curriculum-Phase.", "0.3–0.7."),
        level=Level.ADVANCED)


class TrainingConfig(BaseConfig):
    __title__ = "Training"
    __tooltip__ = T(
        "Alle Einstellungen des Trainingsablaufs.",
        "Wie, wie lange und womit das Modell lernt.",
        "Jedes Training.",
    )

    # --- Art des Trainings --------------------------------------------------
    paradigm = ComponentParam("paradigm", "supervised", group="Art", tooltip=T(
        "Trainings-Paradigma (welche Art von Lernen).",
        "Bestimmt, wie aus Daten Verluste werden (überwacht, generativ, kontrastiv, RL ...).",
        "Passend zur Aufgabe wählen; Presets setzen es automatisch.",
    ))

    # --- Umfang -------------------------------------------------------------
    epochs = Param(10, min=1, group="Umfang", tooltip=T(
        "Durchläufe durch den Datensatz.", "Mehr = mehr Lernen, aber Überanpassungsgefahr.", "5–100."))
    max_steps = Param(None, type=int, min=1, group="Umfang", tooltip=T(
        "Exakte Anzahl Optimierungsschritte (wenn gesetzt, wird 'epochs' ignoriert).",
        "Feste Rechenbudgets unabhängig von der Datensatzgröße.", "LLM-Training wird meist in Schritten bemessen."))
    batch_size = Param(32, min=1, group="Umfang", tooltip=T(
        "Beispiele pro Schritt.", "Größer = stabilere Gradienten, mehr Speicher.", "8–512."))
    eval_batch_size = Param(None, type=int, min=1, group="Umfang", level=Level.ADVANCED, tooltip=T(
        "Batchgröße für Evaluierung (None = batch_size).", "Evaluierung braucht weniger Speicher.", "2× batch_size."))
    gradient_accumulation_steps = Param(1, min=1, group="Umfang", tooltip=T(
        "Gradienten über n Mikro-Batches sammeln vor einem Schritt.",
        "Simuliert große Batches mit wenig Speicher (effektive Batchgröße = batch_size · n).",
        "Große Modelle auf kleinen GPUs.",
    ))

    # --- Optimierung ----------------------------------------------------------
    optimizer = ComponentParam("optimizer", "adamw", group="Optimierung", tooltip=T(
        "Optimierer.", "Wie Gradienten zu Updates werden.", "adamw (Standard), sgd (CNNs), lion/muon (experimentell)."))
    optimizer_overrides = Param({}, group="Optimierung", level=Level.ADVANCED, tooltip=T(
        "Abweichende Optimierer je Optimierungsgruppe des Paradigmas, z. B. {'discriminator': {'name': 'adam', 'lr': 4e-4}}.",
        "GANs brauchen oft getrennte Lernraten für Generator/Diskriminator (TTUR).", "GAN-Training."))
    scheduler = ComponentParam("scheduler", "warmup_cosine", group="Optimierung", tooltip=T(
        "Lernraten-Zeitplan.", "Warmup + Abklingen.", "warmup_cosine (Standard), constant (Debug)."))
    no_decay_patterns = Param(["bias", "norm", "embed", "pos", "log_std", "scale", "gamma"], item_type=str,
                              group="Optimierung", level=Level.ADVANCED, tooltip=T(
        "Parameter, deren Name eines dieser Teilwörter enthält, bekommen keinen Weight-Decay.",
        "Biases/Normen/Embeddings zu verkleinern schadet meist.", "Standardpraxis bei Transformern."))
    layerwise_lr_decay = Param(None, type=float, min=0.0, max=1.0, group="Optimierung", level=Level.EXPERT, tooltip=T(
        "Schichtweiser Lernratenfaktor: tiefere (frühere) Blöcke lernen langsamer (LR · d^(N−i)).",
        "Schützt vortrainiertes Wissen in frühen Schichten.", "Feintuning (BEiT/ELECTRA: 0.65–0.9)."))

    # --- Verluste -------------------------------------------------------------
    losses = ComponentListParam("loss", group="Verluste", tooltip=T(
        "Liste gewichteter Verlustterme (leer = Standard des Paradigmas).",
        "Beliebige Kombinationen mehrerer Ziele.",
        "Multi-Task, kombinierte Verluste.",
    ))

    # --- Regularisierung & Stabilität ------------------------------------------
    regularization = NestedParam(RegularizationConfig, group="Regularisierung")
    gradient_clipping = NestedParam(GradientClippingConfig, group="Stabilität")
    ema = NestedParam(EMAConfig, group="Stabilität", level=Level.ADVANCED)
    nan_policy = Param("skip", choices={
        "raise": "Training abbrechen",
        "skip": "Schritt überspringen (Gradienten verwerfen)",
        "ignore": "nichts tun (nicht empfohlen)",
    }, group="Stabilität", level=Level.ADVANCED, tooltip=T(
        "Verhalten bei NaN/Inf in Verlust oder Gradienten.", "Schützt vor Zerstörung der Gewichte.", "skip."))
    curriculum = NestedParam(CurriculumConfig, group="Daten", level=Level.ADVANCED)

    # --- Ablauf ---------------------------------------------------------------
    eval_every_steps = Param(None, type=int, min=1, group="Ablauf", tooltip=T(
        "Evaluierung alle n Schritte (None = nach jeder Epoche).", "Häufiger = bessere Kontrolle, langsamer.",
        "Lange Epochen: alle 500–2000 Schritte."))
    eval_max_batches = Param(None, type=int, min=1, group="Ablauf", level=Level.ADVANCED, tooltip=T(
        "Höchstens so viele Batches evaluieren.", "Schnellere Zwischen-Evaluierungen.", "Große Validierungsmengen."))
    log_every_steps = Param(10, min=1, group="Ablauf", tooltip=T(
        "Ausgabe des Fortschritts alle n Schritte.", "Übersicht im Terminal.", "10–100."))
    verbose = Param(True, group="Ablauf", tooltip=T("Fortschritt ausgeben.", "Konsolenausgabe.", "Standard: an."))
    callbacks = ComponentListParam("callback", group="Ablauf", tooltip=T(
        "Callbacks (Early Stopping, Checkpoints, Logger ...).", "Erweitern den Trainingsablauf.", "Nach Bedarf."))
    seed = Param(42, type=int, nullable=True, group="Ablauf", tooltip=T(
        "Zufalls-Seed (None = nicht setzen).", "Reproduzierbare Ergebnisse.", "Für Vergleiche immer setzen."))
    deterministic = Param(False, group="Ablauf", level=Level.EXPERT, tooltip=T(
        "Deterministische Algorithmen erzwingen.", "Bitgenaue Reproduzierbarkeit, oft langsamer.", "Debugging."))

    # --- Daten-Laden ------------------------------------------------------------
    validation_split = Param(0.1, min=0.0, max=0.9, group="Daten", tooltip=T(
        "Anteil der Trainingsdaten als Validierung (wenn keine eigenen Validierungsdaten übergeben werden).",
        "Misst Generalisierung während des Trainings.", "0.1–0.2."))
    shuffle = Param(True, group="Daten", tooltip=T("Daten jede Epoche mischen.", "Verhindert Reihenfolge-Effekte.", "an."))
    drop_last = Param(False, group="Daten", level=Level.ADVANCED, tooltip=T(
        "Letzten unvollständigen Batch verwerfen.", "Gleich große Batches (wichtig für BatchNorm).", "BatchNorm: an."))
    num_workers = Param(0, min=0, group="Daten", level=Level.ADVANCED, tooltip=T(
        "Hintergrundprozesse zum Datenladen.", "Schneller bei aufwendigem Laden.", "0 (einfach) bis CPU-Kerne."))
    pin_memory = Param(False, group="Daten", level=Level.EXPERT, tooltip=T(
        "Gepinnter Speicher für schnellere GPU-Übertragung.", "Schneller bei CUDA.", "GPU-Training: an."))

    # --- Cluster ----------------------------------------------------------------
    distributed = NestedParam(DistributedConfig, group="Cluster", level=Level.ADVANCED)

    def check(self) -> list[str]:
        problems = []
        if self.regularization.sam_rho > 0 and self.gradient_accumulation_steps > 1:
            problems.append("SAM (sam_rho > 0) ist nicht mit gradient_accumulation_steps > 1 kombinierbar.")
        return problems
