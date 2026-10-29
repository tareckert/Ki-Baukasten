# Anforderungen an eine perfekte KI – und wie der Baukasten sie erfüllt

Automatisch erzeugt aus `ki_baukasten/requirements.py` (`python -m ki_baukasten doku` bzw. `perfektion --katalog`).

**Abdeckung: 54 von 56 Anforderungen sind im Baukasten umsetzbar.** Für ein konkretes Projekt prüft `project.perfection_check()`, was davon eingeschaltet ist; `project.apply_perfection()` schaltet alle empfohlenen, schaltbaren Punkte ein.

Legende: **Pflicht** – ohne das ist das Ergebnis unzuverlässig · **empfohlen** – Stand der Praxis · **optional** – je nach Lage. Anwendungen: `alle` = alle Anwendungen, `datensatz` = mit Datensatz (nicht RL), `wenige_labels` = wenige Labels, `tabellen` = Tabellen, `bild` = Bilder, `segmentierung` = Segmentierung, `sprache` = Sprache/Text, `generierung` = Textgenerierung, `zeitreihen` = Zeitreihen, `audio` = Audio, `klassifikation` = Klassifikation, `regression` = Regression/Prognose, `rl` = Reinforcement Learning, `generativ` = Generative Modelle, `graph` = Graphen, `anomalie` = Anomalien, `menschen` = Entscheidungen über Menschen, `detektion` = Objekterkennung, `empfehlung` = Empfehlungssysteme.

## Daten

| | Anforderung | gilt für | Stufe | Warum | Umsetzung im Baukasten | an/abwählbar |
|---|---|---|---|---|---|---|
| ✓ | Datenqualität geprüft (Lecks, Duplikate, NaN, Klassen, Skalen) | datensatz | pflicht | Die meisten KI-Fehler sind Datenfehler. | workflow.check_data bzw. project.data_check() | `datenpruefung` |
| ✓ | Ehrliche Aufteilung mit Testmenge | datensatz | pflicht | Nur nie benutzte Daten zeigen die echte Qualität. | training.test_split + project.test() | `testmenge` |
| ✓ | Stratifizierte Aufteilung | klassifikation | empfohlen | Seltene Klassen sollen in allen Teilmengen vorkommen. | training.validation_strategy = 'stratifiziert' | `stratifiziert` |
| ✓ | Zeitliche Aufteilung (keine Zukunft im Training) | zeitreihen | pflicht | Zufällige Aufteilung verrät bei Zeitreihen die Zukunft. | training.validation_strategy = 'zeitlich' | `zeitliche_aufteilung` |
| ✓ | Vorverarbeitung nur auf Trainingsdaten gelernt | tabellen, sprache, bild, audio | pflicht | Skalen/Vokabular aus Validierungsdaten wären ein Datenleck. | csv/text_classification/image_folder/audio lernen Skalen, Kategorien und Tokenizer nur auf Trainingszeilen | automatisch |
| ✓ | Fehlende Werte behandeln (+ Indikator) | tabellen | pflicht | Lücken dürfen das Training nicht zerstören und sind oft selbst informativ. | data = 'csv': missing (mittelwert/median/null/zeile_entfernen), missing_indicator | automatisch |
| ✓ | Kategorie-Spalten kodieren | tabellen | pflicht | Texte wie 'Berlin' müssen zu Zahlen werden, ohne falsche Reihenfolge. | data = 'csv': encoding onehot/ordinal | automatisch |
| ✓ | Merkmale skalieren | tabellen | empfohlen | Unterschiedliche Wertebereiche bremsen das Lernen. | data = 'csv': scaling standard/robust/minmax | automatisch |
| ✓ | Unausgewogene Klassen ausgleichen | klassifikation | empfohlen | Sonst ignoriert das Modell seltene Klassen. | training.class_weights oder training.balance_classes | `klassengewichte`, `klassen_ausgleich` |
| ✓ | Daten-Augmentierung | bild, sprache, audio | empfohlen | Mehr Vielfalt aus denselben Daten, weniger Auswendiglernen. | training.augmentation | `bild_augmentierung`, `random_erasing`, `token_rauschen`, `cutmix` |
| ✓ | Kompakter Tokenizer (BPE) | sprache | empfohlen | Kürzere Folgen, keine unbekannten Wörter. | data.tokenizer = 'bpe' | `bpe_tokenizer` |
| ✓ | Eigene Daten ohne Code laden | datensatz | pflicht | Echte Projekte brauchen echte Daten. | Datensätze csv, image_folder, text_classification, segmentation, audio, instructions, tensor_file | automatisch |
| ✓ | Mit wenigen Labels umgehen | wenige_labels | empfohlen | Beschriften ist teuer. | select_for_labeling (Active Learning), pseudo_label, selbstüberwachtes Vortraining | `project.select_for_labeling` |

## Modell

| | Anforderung | gilt für | Stufe | Warum | Umsetzung im Baukasten | an/abwählbar |
|---|---|---|---|---|---|---|
| ✓ | Passendes Modell zur Aufgabe gewählt | alle | pflicht | Die Architektur bestimmt, was überhaupt lernbar ist. | Ziel-Assistent: project.set_goals(…) / goal_check() | `project.set_goals` |
| ✓ | Regularisierung (Dropout, Weight Decay) | datensatz | empfohlen | Gegen Auswendiglernen. | model.**.dropout, training.optimizer.weight_decay | `dropout`, `weight_decay`, `drop_path` |
| ✓ | Vortrainiertes Modell nutzen können | sprache, bild | optional | Vortraining spart Daten und Rechenzeit. | Project.import_model + LoRA/QLoRA/schichtweise Lernraten | automatisch |
| ✓ | Segmentierungsmodell (U-Net) | segmentierung | pflicht | Eine Klasse je Pixel braucht Encoder-Decoder mit Skip-Verbindungen. | model = 'unet' | – |
| ✓ | Lange Kontexte effizient | sprache, zeitreihen | optional | Attention wächst quadratisch. | Mamba/SSM, Hybrid, RoPE, KV-Cache, Chunked Prefill | automatisch |
| ✗ | Objekterkennung (Bounding Boxes) | detektion | pflicht | Objekte finden und einrahmen. | **fehlt:** Kein Detektor (YOLO/DETR) enthalten – Alternativen: Segmentierung mit U-Net oder Export + spezialisiertes Framework. | – |
| ✗ | Empfehlungssystem (Nutzer × Artikel) | empfehlung | pflicht | Personalisierte Vorschläge. | **fehlt:** Keine Matrixfaktorisierung/Two-Tower-Vorlage – Näherung: Tabellenmodell auf Nutzer-/Artikel-Merkmalen oder kontrastives Lernen. | – |

## Training

| | Anforderung | gilt für | Stufe | Warum | Umsetzung im Baukasten | an/abwählbar |
|---|---|---|---|---|---|---|
| ✓ | Stabiles Training (Clipping, NaN-Schutz) | alle | pflicht | Ein Ausreißer darf nicht Stunden Training zerstören. | training.gradient_clipping, training.nan_policy | `gradient_clipping`, `nan_schutz`, `agc` |
| ✓ | Lernraten-Zeitplan mit Warmup | datensatz | empfohlen | Die Lernrate ist die wichtigste Einstellung. | training.scheduler (warmup_cosine, wsd …), find_lr() | `warmup`, `wsd` |
| ✓ | Early Stopping / beste Gewichte behalten | datensatz | empfohlen | Stoppt, bevor das Auswendiglernen beginnt. | callback early_stopping (monitor val_loss/val_f1 …) | `early_stopping` |
| ✓ | Reproduzierbar (Seed) | alle | pflicht | Ergebnisse müssen wiederholbar und vergleichbar sein. | training.seed | `seed` |
| ✓ | Checkpoints und Fortsetzen | alle | empfohlen | Abbrüche kosten sonst die ganze Arbeit. | callback checkpoint, training.resume_from = 'auto' | `checkpoints`, `fortsetzen` |
| ✓ | Hardware ausgenutzt (Präzision, TF32, fusionierter Optimierer …) | alle | pflicht | Gleiche Hardware, gleiches Tempo wie von Hand optimierte Modelle. | memory.mixed_precision/matmul_precision = 'auto', fusionierter Optimierer, SDPA, schneller Tensor-Lader | automatisch |
| ✓ | Lernrate bestimmt (LR-Finder) | datensatz | optional | Eine passende Lernrate spart viele Versuche. | training.auto_lr bzw. project.find_lr() | `auto_lernrate` |
| ✓ | Hyperparameter-Suche | alle | optional | Gute Einstellungen automatisch finden. | workflow.search (random/grid/evolution/pso/pbt), project.tune() | `zufallssuche`, `partikelschwarm`, `genetisch`, `pbt`, `gittersuche` |
| ✓ | Verlauf protokollieren | alle | empfohlen | Nachvollziehen, was passiert ist. | training.logging (CSV/JSONL/TensorBoard/W&B/MLflow) | `log_datei`, `log_jsonl`, `tensorboard`, `wandb`, `mlflow` |
| ✓ | Große Modelle auf kleiner Hardware | alle | optional | Hardware ist oft der Engpass. | Checkpointing, 8-Bit-Optimierer, LoRA/QLoRA, Streaming, Notfall-Ausführung | automatisch |
| ✓ | Weiterlernen ohne Vergessen | alle | optional | Neue Daten kommen – altes Wissen soll bleiben. | project.continue_training(method='lwf'|'ewc'|'replay') | automatisch |
| ✓ | Instruction-Tuning (Verlust nur auf Antworten) | generierung | empfohlen | Ein Assistent soll antworten, nicht Fragen nachplappern. | data = 'instructions' (loss_mask), project.ask() | – |
| ✓ | Präferenz-Lernen (DPO/Reward-Modell) | generierung | optional | Antworten an menschliche Vorlieben anpassen. | paradigm 'dpo' / 'preference' | automatisch |
| ✓ | Belohnungen formen und kombinieren | rl | pflicht | Die Belohnung definiert, was der Agent lernt. | training.paradigm.rewards (Liste gewichteter Terme) | automatisch |

## Bewertung

| | Anforderung | gilt für | Stufe | Warum | Umsetzung im Baukasten | an/abwählbar |
|---|---|---|---|---|---|---|
| ✓ | Auswertung über mehrere Episoden (deterministisch) | rl | pflicht | Eine Episode ist Zufall – erst der Mittelwert vieler zeigt die Qualität. | training.paradigm.eval_episodes, eval_deterministic | automatisch |
| ✓ | Passende Metriken (nicht nur Verlust) | datensatz | pflicht | Genauigkeit lügt bei seltenen Klassen; Prognosen brauchen Fehler in echten Einheiten. | training.metrics / training.auto_metrics | `metriken_auto` |
| ✓ | Endbewertung auf der Testmenge | datensatz | pflicht | Der ehrliche Abschluss. | project.test() (bzw. workflow.run_test) | `project.test` |
| ✓ | Vergleich mit einfacher Basis | klassifikation, regression | empfohlen | Schlägt das Modell nicht einmal 'häufigste Klasse'/'Mittelwert', hat es nichts gelernt. | project.baseline() | `project.baseline` |
| ✓ | Fehleranalyse (Konfusionsmatrix, sicherste Fehler) | klassifikation, regression | optional | Zeigt, *wo* das Modell scheitert – oft Datenfehler. | project.error_analysis() | `project.error_analysis` |
| ✓ | Kreuzvalidierung bei wenig Daten | tabellen | optional | Eine einzelne Aufteilung ist bei wenig Daten Zufall. | workflow.cross_validation / cross_validate() | `kreuzvalidierung` |
| ✓ | IoU/Dice für Segmentierung | segmentierung | pflicht | Pixel-Genauigkeit wird vom Hintergrund dominiert. | training.metrics ['iou', 'dice'] (auto) | `metriken_auto` |

## Vertrauen & Sicherheit

| | Anforderung | gilt für | Stufe | Warum | Umsetzung im Baukasten | an/abwählbar |
|---|---|---|---|---|---|---|
| ✓ | Kalibrierte Wahrscheinlichkeiten | klassifikation | empfohlen | '90 % sicher' soll in 90 % der Fälle stimmen. | inference.auto_calibrate / project.calibrate() | `kalibrierung` |
| ✓ | Unsicherheit angeben | klassifikation, regression | empfohlen | Im Einsatz zählt auch, wie sicher die Antwort ist. | Conformal Prediction, MC-Dropout, TTA, Enthaltung | `conformal`, `mc_dropout`, `tta`, `enthaltung` |
| ✓ | Entscheidungen erklärbar | tabellen, bild, sprache | empfohlen | Vertrauen, Fehlersuche, Vorgaben (z. B. EU AI Act). | project.explain_prediction / feature_importance | `project.explain_prediction` |
| ✓ | Robust gegen Störungen | tabellen, bild, audio | optional | Kleine Änderungen dürfen das Ergebnis nicht kippen. | training.adversarial, workflow.robustness_check | `robustheitspruefung`, `adversarial_training`, `fgsm_training`, `eingaberauschen` |
| ✓ | Fairness geprüft | menschen | pflicht | Entscheidungen über Menschen dürfen Gruppen nicht benachteiligen. | project.fairness_report(gruppen) | `project.fairness_report` |
| ✓ | Datenschutz (DP-SGD / föderiert) | menschen | optional | Einzelne Personen dürfen nicht rekonstruierbar sein. | training.privacy, workflow.federated | `dp_sgd`, `foederiert` |
| ✓ | Saubere Generierung (Beam Search, Wiederholungssperre) | generierung | optional | Kleine Sprachmodelle wiederholen sich gern. | project.generation | `keine_wiederholungen`, `beam_search` |

## Einsatz & Betrieb

| | Anforderung | gilt für | Stufe | Warum | Umsetzung im Baukasten | an/abwählbar |
|---|---|---|---|---|---|---|
| ✓ | Vorverarbeitung mit dem Modell gespeichert | datensatz | pflicht | Im Einsatz müssen Rohdaten genau wie im Training aufbereitet werden. | save() → vorverarbeitung.json; load() + predict_raw() | automatisch |
| ✓ | Export ohne Ballast (geprüft) | alle | pflicht | Ein Einsatz-Modell soll nur enthalten, was die Vorhersage braucht. | deployment.export: Trainings-Hüllen weg, EMA, LoRA einrechnen, BatchNorm falten, Ungenutztes entfernen, Ausgabe gegen das Original geprüft | automatisch |
| ✓ | Modellkarte und Umgebung dokumentiert | alle | empfohlen | Zweck, Daten, Ergebnisse, Grenzen, Versionen – nachvollziehbar. | deployment.model_card | `modellkarte` |
| ✓ | Für das Zielgerät verkleinert | alle | optional | Das Modell muss dort laufen, wo es gebraucht wird. | deployment.main_target, recommend_for/reduce_for | `komprimieren_nach_training` |
| ✓ | Export in ein Standardformat | alle | optional | Einsatz ohne Python/PyTorch (ONNX, TorchScript …). | workflow.export_format / project.export() | `export_nach_training` |
| ✓ | Als Dienst bereitstellen (HTTP) | alle | optional | Apps und andere Programme nutzen die KI. | project.serve() / python -m ki_baukasten bereitstellen | automatisch |
| ✓ | Drift im Einsatz überwachen | datensatz | empfohlen | Ändern sich die Daten, wird die KI unbemerkt schlechter. | project.drift_report(neue_daten) | `project.drift_report` |
| ✓ | Modell-Versionen nachvollziehbar | alle | optional | Welches Modell lief wann mit welchen Daten? | Konfigurations-Prüfsumme + Modellkarte; MLflow-Logger | automatisch |

## Ehrliche Lücken

- **Objekterkennung (Bounding Boxes)**: Kein Detektor (YOLO/DETR) enthalten – Alternativen: Segmentierung mit U-Net oder Export + spezialisiertes Framework.
- **Empfehlungssystem (Nutzer × Artikel)**: Keine Matrixfaktorisierung/Two-Tower-Vorlage – Näherung: Tabellenmodell auf Nutzer-/Artikel-Merkmalen oder kontrastives Lernen.
