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
from .logger import LoggingConfig


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
    apply_at_end = Param(True, tooltip=T(
        "Am Ende des Trainings die EMA-Gewichte ins Modell übernehmen.",
        "Vorhersage, Speichern und Export nutzen dann genau das Modell, dessen Werte in der Validierung "
        "angezeigt wurden.", "an (aus: letzte Trainingsgewichte behalten)."))
    warmup = Param(True, tooltip=T(
        "Abklingrate anfangs kleiner: min(decay, (1+n)/(10+n)).",
        "Ohne Aufwärmen enthält die EMA nach kurzen Läufen fast nur die zufälligen Startgewichte.", "an."),
        level=Level.ADVANCED)


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


class StabilityConfig(BaseConfig):
    __title__ = "Stabilität: Verlust-Spitzen"
    __tooltip__ = T(
        "Schutz vor plötzlichen Verlust-Spitzen (Loss-Spikes), die große Trainingsläufe zerstören können.",
        "Ein einzelner schlechter Batch oder eine instabile Phase kann Gewichte unbrauchbar machen – der Schutz "
        "überspringt solche Schritte oder springt auf den letzten guten Stand zurück (wie bei großen LLM-Läufen).",
        "Lange/teure Läufe, hohe Lernraten, fp16, verrauschte Daten.",
    )
    spike_guard = Param(False, tooltip=T(
        "Verlust-Spitzen erkennen und abfangen.", "Verhindert, dass Ausreißer das Training zerstören.",
        "Lange Läufe: an."))
    spike_factor = Param(3.0, min=1.0, tooltip=T(
        "Spitze = Verlust größer als Faktor × gleitender Mittelwert.", "Empfindlichkeit.", "2–5."))
    spike_window = Param(0.98, min=0.0, max=0.9999, level=Level.ADVANCED, tooltip=T(
        "Glättung des gleitenden Mittels (näher an 1 = längeres Gedächtnis).", "Robuste Referenz.", "0.95–0.99."))
    spike_warmup = Param(20, min=0, level=Level.ADVANCED, tooltip=T(
        "Erst nach so vielen Schritten prüfen (Anfangsverluste schwanken stark).", "Keine Fehlalarme am Start.", "20–100."))
    spike_action = Param("skip", choices={
        "skip": "Schritt überspringen (Gradienten verwerfen)",
        "rollback": "auf den letzten guten Stand zurückspringen und die Lernrate senken",
    }, tooltip=T("Reaktion auf eine Spitze.", "rollback hilft, wenn das Modell schon 'abgerutscht' ist.", "skip."))
    rollback_every = Param(50, min=1, level=Level.ADVANCED, tooltip=T(
        "Alle n Schritte einen Sicherungsstand im RAM ablegen (rollback).", "Wie weit zurückgesprungen wird.",
        "Kostet eine Kopie von Modell + Optimierer im RAM."))
    rollback_lr_factor = Param(0.5, min=0.0, max=1.0, level=Level.ADVANCED, tooltip=T(
        "Lernrate nach einem Rücksprung mit diesem Faktor multiplizieren.", "Verhindert den gleichen Absturz.", "0.5."))


class OptimizerExtrasConfig(BaseConfig):
    __title__ = "Optimierer-Erweiterungen"
    __tooltip__ = T(
        "Zusätze, die mit jedem Optimierer kombinierbar sind (Lookahead, Gradient Centralization, Grokfast).",
        "Verbessern Stabilität bzw. Generalisierung, ohne den Optimierer zu wechseln.",
        "Experimente, schwer trainierbare Modelle.",
    )
    lookahead_k = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Lookahead: alle k Schritte die 'langsamen' Gewichte ein Stück zu den schnellen ziehen (None = aus).",
        "Glättet die Optimierung, weniger empfindlich gegenüber der Lernrate (Zhang et al. 2019).", "k = 5."),
        on=5)
    lookahead_alpha = Param(0.5, min=0.0, max=1.0, tooltip=T(
        "Lookahead-Schrittweite der langsamen Gewichte.", "0.5 = halber Weg.", "0.5."), level=Level.ADVANCED)
    gradient_centralization = Param(False, tooltip=T(
        "Gradient Centralization: Mittelwert jeder Gewichts-Gradientenzeile abziehen.",
        "Stabilisiert und beschleunigt oft das Training von CNNs/MLPs (Yong et al. 2020).", "CNNs."))
    grokfast_alpha = Param(None, type=float, min=0.0, max=0.9999, nullable=True, tooltip=T(
        "Grokfast: langsame Gradientenanteile (EMA mit dieser Glättung) verstärken (None = aus).",
        "Beschleunigt 'Grokking' – späte Generalisierung nach Auswendiglernen (Lee et al. 2024).",
        "Kleine algorithmische Datensätze; 0.98.", "Experimentell."), level=Level.EXPERIMENTAL, on=0.98)
    grokfast_lambda = Param(2.0, min=0.0, tooltip=T(
        "Verstärkung der langsamen Gradientenanteile.", "Stärke des Effekts.", "1–5."), level=Level.EXPERIMENTAL)


class AugmentationConfig(BaseConfig):
    __title__ = "Daten-Augmentierung"
    __tooltip__ = T(
        "Zufällige Veränderungen der Trainingsbeispiele (nur im Training).",
        "Das Modell sieht jedes Beispiel in vielen Varianten – weniger Auswendiglernen, robustere Vorhersagen.",
        "Bilder (Spiegeln, Ausschnitt, Ausradieren, CutMix), Texte (Token-Rauschen).",
    )
    hflip = Param(False, tooltip=T("Bilder zufällig horizontal spiegeln.", "Klassische Bild-Augmentierung.",
                                   "Fotos, Objekte (nicht bei Text/Ziffern mit Richtung)."))
    vflip = Param(False, tooltip=T("Bilder zufällig vertikal spiegeln.", "Für richtungsfreie Bilder.",
                                   "Satelliten-, Mikroskopbilder."), level=Level.ADVANCED)
    crop_padding = Param(0, min=0, tooltip=T(
        "Zufälliger Ausschnitt nach Auffüllen um so viele Pixel (0 = aus).", "Verschiebungs-Invarianz.",
        "CIFAR-Standard: 4."), on=4)
    random_erasing = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Wahrscheinlichkeit, ein zufälliges Rechteck im Bild zu löschen (Cutout).", "Robust gegen Verdeckungen.",
        "0.25."), on=0.25)
    cutmix_alpha = Param(0.0, min=0.0, tooltip=T(
        "CutMix: Bildausschnitt eines anderen Beispiels einsetzen, Ziele anteilig mischen (0 = aus).",
        "Stärker als Mixup bei Bildern; mit Mixup kombiniert wird pro Batch zufällig gewählt.", "1.0."), on=1.0)
    token_noise = Param(0.0, min=0.0, max=0.5, tooltip=T(
        "Anteil der Eingabe-Tokens, die durch zufällige Tokens ersetzt werden (nur überwachte Aufgaben).",
        "Robust gegen Tippfehler und Rauschen.", "0.05–0.1."), level=Level.ADVANCED, on=0.05)


class AdversarialConfig(BaseConfig):
    __title__ = "Adversariales Training (Robustheit)"
    __tooltip__ = T(
        "Trainiert zusätzlich auf gezielt gestörten Eingaben, die das Modell am stärksten täuschen (FGSM/PGD).",
        "Winzige, unsichtbare Änderungen können ungeschützte Netze komplett täuschen – adversariales Training macht "
        "sie robust (Madry et al. 2018). Wirkt auch als starke Regularisierung.",
        "Sicherheitskritische Bild-/Sensormodelle, Betrugserkennung, Robustheit gegen Rauschen.",
    )
    method = Param("keine", choices={
        "keine": "aus", "fgsm": "Fast Gradient Sign Method: ein Schritt (schnell)",
        "pgd": "Projected Gradient Descent: mehrere Schritte (stärker, langsamer)"}, tooltip=T(
        "Angriffsverfahren zum Erzeugen der Trainingsbeispiele.", "pgd ist der Standard für echte Robustheit.",
        "Einstieg: fgsm; ernsthaft: pgd."), on="fgsm")
    epsilon = Param(0.03, min=0.0, tooltip=T(
        "Maximale Störung je Eingabewert (L∞).", "Größer = robuster, aber oft etwas ungenauer auf sauberen Daten.",
        "Bilder in [0,1]: 8/255 ≈ 0.03; standardisierte Tabellen: 0.05–0.2."))
    steps = Param(5, min=1, tooltip=T("PGD-Schritte.", "Mehr = stärkerer Angriff.", "3–10."), level=Level.ADVANCED)
    step_size = Param(None, type=float, min=0.0, nullable=True, tooltip=T(
        "PGD-Schrittweite (None = 2.5 · epsilon / steps).", "Feinheit der Suche.", "None."), level=Level.ADVANCED)
    ratio = Param(0.5, min=0.0, max=1.0, tooltip=T(
        "Anteil des adversarialen Verlusts (Rest: saubere Beispiele).", "Ausgleich Genauigkeit ↔ Robustheit.", "0.5."))


class PrivacyConfig(BaseConfig):
    __title__ = "Datenschutz (DP-SGD)"
    __tooltip__ = T(
        "Differentiell privates Training: Gradient jedes Beispiels begrenzen und Rauschen hinzufügen.",
        "Das Modell kann sich einzelne Datensätze kaum merken – Schutz sensibler Daten (Google, Apple).",
        "Medizin-, Finanz-, Nutzerdaten.",
        "Rechnet jedes Beispiel einzeln (langsam). Ohne Privacy-Accounting (ε); für formale Garantien Opacus nutzen.",
    )
    enabled = Param(False, tooltip=T("DP-SGD einschalten.", "Schützt einzelne Trainingsbeispiele.", "Sensible Daten."))
    clip_norm = Param(1.0, min=0.0, tooltip=T(
        "Maximale Gradientennorm pro Beispiel.", "Begrenzt den Einfluss jedes Einzelnen.", "0.5–2."))
    noise_multiplier = Param(1.0, min=0.0, tooltip=T(
        "Rauschstärke relativ zu clip_norm.", "Mehr Rauschen = mehr Schutz, schlechtere Genauigkeit.", "0.5–2."))


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
    augmentation = NestedParam(AugmentationConfig, group="Regularisierung")
    adversarial = NestedParam(AdversarialConfig, group="Regularisierung", level=Level.ADVANCED)
    gradient_clipping = NestedParam(GradientClippingConfig, group="Stabilität")
    stability = NestedParam(StabilityConfig, group="Stabilität", level=Level.ADVANCED)
    optimizer_extras = NestedParam(OptimizerExtrasConfig, group="Optimierung", level=Level.ADVANCED)
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
    logging = NestedParam(LoggingConfig, group="Ablauf")
    callbacks = ComponentListParam("callback", group="Ablauf", tooltip=T(
        "Callbacks (Early Stopping, Checkpoints, Logger ...).", "Erweitern den Trainingsablauf.", "Nach Bedarf."))
    seed = Param(42, type=int, nullable=True, group="Ablauf", tooltip=T(
        "Zufalls-Seed (None = nicht setzen).", "Reproduzierbare Ergebnisse.", "Für Vergleiche immer setzen."))
    deterministic = Param(False, group="Ablauf", level=Level.EXPERT, tooltip=T(
        "Deterministische Algorithmen erzwingen.", "Bitgenaue Reproduzierbarkeit, oft langsamer.", "Debugging."))

    # --- Bewertung ----------------------------------------------------------------
    metrics = ComponentListParam("metric", group="Bewertung", tooltip=T(
        "Metriken für Validierung und Test (leer = automatische Auswahl je Aufgabe, siehe auto_metrics).",
        "Der Verlust allein sagt wenig: bei seltenen Klassen f1/balanced_accuracy, bei Prognosen mae/rmse, bei "
        "Risiken ece (Kalibrierung), bei Übersetzung bleu.",
        "Klassifikation: ['f1', 'auroc']; Regression: ['mae', 'r2']; Early Stopping mit monitor='val_f1'."))
    auto_metrics = Param(True, group="Bewertung", tooltip=T(
        "Ohne eigene Liste passende Metriken automatisch berechnen (Klassifikation: accuracy, balanced_accuracy, f1, "
        "auroc, ece; Regression: mae, rmse, r2; Sprache: perplexity, token_accuracy …).",
        "Man sieht sofort, ob das Modell wirklich gut ist.", "an (kostet nur etwas Rechenzeit in der Evaluierung)."))
    metric_max_samples = Param(5000, min=1, group="Bewertung", level=Level.ADVANCED, tooltip=T(
        "Höchstens so viele Beispiele für die Metriken sammeln.", "Begrenzt Speicher bei großen Validierungsmengen.",
        "1000–20000."))
    class_weights = Param("keine", group="Bewertung", choices={
        "keine": "alle Klassen gleich", "ausgeglichen": "Gewicht N / (K · n_Klasse) – seltene Klassen zählen mehr",
        "wurzel": "Wurzel davon – milder, oft stabiler"}, tooltip=T(
        "Klassengewichte im Standard-Klassifikationsverlust.",
        "Bei unausgewogenen Daten ignoriert das Modell sonst seltene Klassen (Alternative zu balance_classes).",
        "Betrug, Diagnosen, seltene Fehler: ausgeglichen oder wurzel."), on="ausgeglichen")

    auto_lr = Param(False, group="Optimierung", tooltip=T(
        "Vor dem ersten Training die Lernrate per LR-Finder (Bereichstest) bestimmen und übernehmen.",
        "Die Lernrate ist die wichtigste Einstellung – automatisch passend zu Modell und Daten.",
        "Neue Modelle/Daten ohne Erfahrungswerte."), on=True)
    auto_batch_size = Param(False, group="Umfang", tooltip=T(
        "Auf GPUs die größte Batchgröße wählen, die sicher in den Gerätespeicher passt (Lernrate √-skaliert).",
        "Nutzt die Hardware voll aus – gleiche Rechenzeit, mehr Daten je Schritt.", "GPU-Training; auf der CPU "
        "ohne Wirkung."), on=True)
    resume_from = Param(None, type=str, nullable=True, group="Ablauf", tooltip=T(
        "Training aus einem Checkpoint fortsetzen (Pfad zu einer .pt-Datei oder 'auto' = neuester Checkpoint im "
        "Checkpoint-Ordner).", "Nach Abbruch, Stromausfall oder Zeitlimit weitermachen – mit Optimierer-, "
        "Zeitplan- und Schrittstand.", "Lange Läufe, Cluster mit Zeitslots."), on="auto")

    # --- Daten-Laden ------------------------------------------------------------
    validation_split = Param(0.1, min=0.0, max=0.9, group="Daten", tooltip=T(
        "Anteil der Trainingsdaten als Validierung (wenn keine eigenen Validierungsdaten übergeben werden).",
        "Misst Generalisierung während des Trainings.", "0.1–0.2."))
    validation_strategy = Param("zufall", group="Daten", choices={
        "zufall": "zufällige Auswahl", "stratifiziert": "Klassenanteile in Training und Validierung gleich",
        "zeitlich": "die letzten Beispiele als Validierung (Zeitreihen: keine Zukunft im Training)"}, tooltip=T(
        "Wie Validierungs- und Testdaten abgeteilt werden.",
        "Falsche Aufteilung führt zu geschönten Ergebnissen (z. B. Zukunftsdaten im Training bei Zeitreihen).",
        "Klassifikation: stratifiziert; Zeitreihen/Prognosen: zeitlich."))
    test_split = Param(0.0, min=0.0, max=0.5, group="Daten", tooltip=T(
        "Anteil, der als Testmenge ganz zurückgehalten wird (0 = keine).",
        "Validierung steuert das Training (Early Stopping, Suche) und ist dadurch leicht geschönt – nur eine nie "
        "benutzte Testmenge zeigt die echte Qualität (Project.test()).", "0.1–0.2 bei genug Daten."), on=0.15)
    shuffle = Param(True, group="Daten", tooltip=T("Daten jede Epoche mischen.", "Verhindert Reihenfolge-Effekte.", "an."))
    drop_last = Param(False, group="Daten", level=Level.ADVANCED, tooltip=T(
        "Letzten unvollständigen Batch verwerfen.", "Gleich große Batches (wichtig für BatchNorm).", "BatchNorm: an."))
    num_workers = Param(0, min=0, group="Daten", level=Level.ADVANCED, tooltip=T(
        "Hintergrundprozesse zum Datenladen.", "Schneller bei aufwendigem Laden.", "0 (einfach) bis CPU-Kerne."))
    pin_memory = Param(None, type=bool, nullable=True, group="Daten", level=Level.EXPERT, off=False, on=True,
                       tooltip=T("Gepinnter Speicher für schnellere GPU-Übertragung (None = automatisch an auf CUDA).",
                                 "Schneller bei CUDA.", "None lassen."))
    balance_classes = Param(False, group="Daten", level=Level.ADVANCED, tooltip=T(
        "Seltene Klassen häufiger ziehen (gewichtetes Sampling nach Klassenhäufigkeit).",
        "Bei unausgewogenen Daten lernt das Modell sonst vor allem die häufigen Klassen.",
        "Betrugserkennung, Diagnosen, seltene Ereignisse."))
    privacy = NestedParam(PrivacyConfig, group="Datenschutz", level=Level.ADVANCED)

    # --- Cluster ----------------------------------------------------------------
    distributed = NestedParam(DistributedConfig, group="Cluster", level=Level.ADVANCED)

    def check(self) -> list[str]:
        problems = []
        if self.regularization.sam_rho > 0 and self.gradient_accumulation_steps > 1:
            problems.append("SAM (sam_rho > 0) ist nicht mit gradient_accumulation_steps > 1 kombinierbar.")
        if self.privacy.enabled and self.regularization.sam_rho > 0:
            problems.append("DP-SGD (privacy) ist nicht mit SAM kombinierbar.")
        return problems
