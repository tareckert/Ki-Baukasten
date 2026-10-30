# KI-Baukasten – alle Möglichkeiten

Automatisch erzeugt (`python -m ki_baukasten moeglichkeiten -o docs/MOEGLICHKEITEN.md`). 399 Komponenten in 27 Registries – jede überall per Name einsetzbar. Einzelheiten zu jeder Einstellung: [PARAMETER_REFERENZ.md](PARAMETER_REFERENZ.md), innerer Aufbau: [ARCHITEKTUR.md](ARCHITEKTUR.md).

## Bedienung in drei Ebenen

So bleibt es trotz über tausend Einstellungen übersichtlich:

| Ebene | Wofür | Wo |
|---|---|---|
| 1. Ziele | Aufgabe, wichtige Punkte, Zielgerät angeben → Modell wird empfohlen, Unpassendes abgeraten | `Project.from_goals(...)`, CLI `ziele`, GUI „Neue KI nach Zielen“ |
| 2. Techniken | bewährte Verfahren mit einem Häkchen an/aus (Status ● ○ ◐ –) | `Project.enable/disable`, CLI `techniken`, GUI „Techniken“ |
| 3. Einstellungen | jede einzelne Einstellung: An · Aus · Eigen, mit Tooltip | `Project.switch/set`, CLI `schalter`, GUI-Formular + Ansichtsfilter |

Jede Einstellung hat einen Zustand **an** (empfohlener Wert), **aus** (z. B. Dropout 0, Komponente `none`) oder **eigen** (eigener Wert); Pflicht-Einstellungen (z. B. `d_model`) kennen nur an/eigen. `p.set('**.dropout', 'aus')`, `p.switch('training.regularization', 'aus')`, `p.switch_overview()`.

## Ziel-Assistent

`Project.from_goals(task=…, priorities=[…], data_size=…, target=…, description='…')` bewertet alle Vorlagen mit Begründung (✓ Stärken, ✗ Schwächen), rät von unpassenden ab, schlägt Techniken vor und speichert die Ziele im Projekt (`goals`) – später änderbar: `set_goals(...)`, `goal_check()`.

| Aufgabe | Beschreibung |
|---|---|
| `unbekannt` | noch offen – aus der Beschreibung ableiten |
| `text_erzeugen` | Text erzeugen (Chatbot, Vervollständigung, Code) |
| `text_klassifizieren` | Text einordnen (Stimmung, Spam, Thema) |
| `uebersetzen` | Text umwandeln (Übersetzung, Zusammenfassung) |
| `bild_klassifizieren` | Bilder erkennen/einordnen |
| `bild_segmentieren` | Bilder pixelgenau einteilen (Segmentierung: Medizin, Satellit, Prüfung) |
| `bild_erzeugen` | Bilder/Daten erzeugen |
| `audio_klassifizieren` | Geräusche/Sprachbefehle erkennen |
| `assistent` | Anweisungen befolgen / Fragen beantworten (Assistent, Instruction-Tuning) |
| `zeitreihe_prognose` | Zeitreihen vorhersagen (Sensoren, Verkäufe, Energie) |
| `tabelle_klassifizieren` | Tabellendaten einordnen |
| `tabelle_regression` | Zahlen aus Tabellendaten vorhersagen |
| `anomalien_finden` | Auffälligkeiten erkennen (Betrug, Defekte) |
| `graph_analyse` | Graphen analysieren (Moleküle, Netzwerke) |
| `steuerung_rl` | Agenten steuern (Spiele, Robotik, Regelung) |
| `bewertung_praeferenz` | Antworten bewerten / Präferenzen lernen (RLHF) |
| `aehnlichkeit_embeddings` | Ähnlichkeit / Suche / Vortraining ohne Labels |
| `multimodal` | mehrere Datenarten kombinieren |

Prioritäten: `genauigkeit` (möglichst gute Ergebnisse), `geschwindigkeit` (schnelle Vorhersagen), `wenig_speicher` (wenig Speicher / kleines Modell), `schnelles_training` (kurze Trainingszeit), `erklaerbarkeit` (nachvollziehbare Entscheidungen), `robustheit` (stabil und robust), `datenschutz` (sensible Daten schützen), `energie` (wenig Energie / Akku), `einfachheit` (einfach zu verstehen und zu warten), `lange_kontexte` (lange Texte/Sequenzen), `echtzeit` (Echtzeit / Streaming), `weiterlernen` (später mit neuen Daten weiterlernen), `kreativitaet` (vielfältige, kreative Ausgaben).

## Techniken-Katalog

114 Techniken, die Communitys, Forschung und Firmen nutzen – jede an/abwählbar (`p.enable('qlora')`, `p.disable('qlora')` stellt den vorherigen Zustand exakt wieder her).

### Stabilität

Training läuft sicher durch: keine Explosionen, NaNs oder Abstürze.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `gradient_clipping` | Gradient-Clipping (Gesamtnorm 1.0) | GPT-3, LLaMA, fast alle großen Modelle |  |
| `agc` | Adaptives Gradient-Clipping (AGC) | NFNets (DeepMind) |  |
| `warmup` | Lernraten-Warmup | BERT, GPT, ViT |  |
| `nan_schutz` | NaN/Inf-Schutz | Standard in Trainings-Frameworks |  |
| `spike_schutz` | Schutz vor Verlust-Spitzen | große LLM-Läufe (PaLM, OPT) |  |
| `spike_rueckspung` | Verlust-Spitzen: Rücksprung + kleinere Lernrate | OPT-175B-Logbuch (Meta): manuelle Rücksprünge – hier automatisch |  |
| `ema` | EMA der Gewichte | Stable Diffusion, YOLO, timm |  |
| `qk_norm` | QK-Normalisierung in der Attention | ViT-22B, Gemma 3, OLMo 2 |  |
| `z_loss` | z-Loss für Ausgabe-Logits | PaLM, OLMo |  |
| `logit_softcap` | Logit-Soft-Capping | Gemma 2 |  |
| `plateau_lr` | Lernrate bei Stillstand senken | ReduceLROnPlateau (Keras/PyTorch) |  |

### Gegen Auswendiglernen

Besser auf neuen Daten statt nur auf den Trainingsdaten.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `dropout` | Dropout (0.1) | AlexNet, BERT, fast alle |  |
| `drop_path` | Stochastic Depth / DropPath (0.1) | ViT, ConvNeXt, Swin |  |
| `weight_decay` | Weight Decay (0.05) | AdamW-Standard (Transformer, ViT) |  |
| `label_smoothing` | Label Smoothing (0.1) | Inception, Transformer (Vaswani) |  |
| `mixup` | Mixup | timm-/ImageNet-Rezepte |  |
| `cutmix` | CutMix | timm, YOLO |  |
| `early_stopping` | Early Stopping | Keras/Lightning-Standard |  |
| `sam` | Sharpness-Aware Minimization (SAM) | Google (ViT/ResNet ohne Vortraining) |  |
| `swa` | Stochastic Weight Averaging (SWA) | PyTorch SWA, Kaggle-Lösungen |  |
| `flooding` | Flooding | Ishida et al. (2020) |  |
| `eingaberauschen` | Rauschen auf den Eingaben | klassische Regularisierung |  |
| `l1_sparsity` | L1-Regularisierung (spärliche Gewichte) | Lasso (Statistik) |  |
| `kreuzvalidierung` | k-fache Kreuzvalidierung (5) | Kaggle, Medizin-Statistik |  |

### Performance

Schneller trainieren und rechnen.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `bf16` | Gemischte Präzision bf16 | Standard für LLM-Training | GPU empfohlen |
| `fp16` | Gemischte Präzision fp16 + Loss-Scaling | NVIDIA Apex/AMP | nur GPU |
| `compile` | torch.compile | PyTorch 2 | GPU empfohlen |
| `tf32` | TF32-Matrixmultiplikation | NVIDIA-Standard | GPU |
| `channels_last` | Speicherlayout channels_last | NVIDIA, timm | GPU, CNN |
| `cudnn_benchmark` | cuDNN-Benchmark | torchvision-Referenzskripte | GPU |
| `daten_parallel_laden` | Daten parallel laden | PyTorch-DataLoader-Praxis |  |
| `fused_optimizer` | Fusionierter Optimierer (erzwingen) | PyTorch AdamW(fused=True) | Standard ist bereits automatisch |
| `gqa` | Grouped-Query-Attention | LLaMA 2/3, Mistral, Gemma |  |
| `auto_batchgroesse` | Batchgröße an den GPU-Speicher anpassen | Lightning auto_scale_batch_size |  |

### Große Modelle, kleine Hardware

Speicher sparen, auslagern, stückweise rechnen.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `checkpointing` | Gradient-Checkpointing | Standard für große Modelle (Hugging Face, Megatron) |  |
| `akkumulation` | Kleine Batches + Gradientenakkumulation | überall, wo der Speicher knapp ist |  |
| `adamw8bit` | 8-Bit-AdamW | bitsandbytes (Hugging Face) |  |
| `adafactor` | Adafactor | T5, PaLM (Google) |  |
| `lora` | LoRA-Adapter | Microsoft LoRA, Hugging Face PEFT |  |
| `qlora` | QLoRA (4-Bit-Basis + LoRA) | QLoRA (Dettmers et al. 2023) |  |
| `int8_gewichte` | Gewichte in int8 | LLM.int8 (bitsandbytes) |  |
| `aktivierungen_auslagern` | Aktivierungen in den RAM auslagern | PyTorch save_on_cpu |  |
| `stream_ram` | Blöcke trainierbar im RAM (Streaming) | Prinzip von DeepSpeed ZeRO-Offload |  |
| `stream_ssd` | Eingefrorene Basis auf der SSD (Streaming) | Prinzip von ZeRO-Infinity |  |
| `split_cpu` | Teile auf der CPU rechnen (device_map) | Hugging Face accelerate device_map |  |
| `layerweise` | Schicht für Schicht von der SSD (Inferenz) | AirLLM | beim Training trainierbarer Blöcke automatisch als SSD-Streaming |
| `lisa` | LISA: wenige Blöcke gleichzeitig trainieren | LISA (Pan et al. 2024) |  |
| `oom_schutz` | Speichermangel: Batch automatisch teilen | Lightning auto_scale (Prinzip) |  |
| `bf16_gewichte` | Gewichte komplett in bf16 | reines bf16-Training großer LLMs |  |

### Cluster & verteilt

Mehrere GPUs, Rechner oder Prozesse.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `ddp` | Data Parallel (DDP) | PyTorch-Standard | Start mit 'verteilt' |
| `zero1` | ZeRO-1 (Optimierer verteilt) | DeepSpeed ZeRO | Start mit 'verteilt' |
| `fsdp` | FSDP (alles verteilt) | Meta (LLaMA-Training) | Start mit 'verteilt' |
| `fsdp_cpu` | FSDP + CPU-Auslagerung | PyTorch FSDP |  |
| `sync_batchnorm` | Synchronisierte BatchNorm | Detectron2, MMDetection |  |

### Automatische Suche

Gute Einstellungen automatisch finden.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `zufallssuche` | Zufallssuche | Bergstra & Bengio (2012) |  |
| `gittersuche` | Gittersuche | scikit-learn GridSearch |  |
| `auto_lernrate` | Lernrate automatisch bestimmen (LR-Finder) | fastai lr_find, Lightning Tuner |  |

### Schwarmintelligenz

Viele Modelle oder Teilnehmer arbeiten zusammen.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `partikelschwarm` | Partikelschwarm-Optimierung (PSO) | Kennedy & Eberhart (1995) |  |
| `foederiert` | Föderiertes Lernen (FedAvg) | Google Gboard, Apple |  |
| `fedprox` | FedProx (robust bei ungleichen Teilnehmern) | Li et al. (2020) |  |
| `model_soup` | Model Soup (mehrere Läufe mitteln) | Google (Wortsman et al. 2022) |  |

### Evolution

Auslese, Kreuzung und Mutation als Optimierer.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `genetisch` | Genetischer Algorithmus | Neuroevolution (Uber, Google) |  |
| `pbt` | Population Based Training | DeepMind (AlphaStar), Waymo |  |
| `evolution_strategies` | Evolution Strategies (ohne Backward) | OpenAI-ES (Salimans et al. 2017) |  |

### Daten

Augmentierung, Ausgleich seltener Klassen, Curriculum.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `bild_augmentierung` | Bild-Augmentierung (Spiegeln + Ausschnitt) | CIFAR/ImageNet-Standard |  |
| `random_erasing` | Random Erasing (Cutout) | Zhong et al., timm |  |
| `token_rauschen` | Token-Rauschen | BERT-Maskierung (Prinzip) |  |
| `klassen_ausgleich` | Seltene Klassen ausgleichen | Standard bei unbalancierten Daten |  |
| `bpe_tokenizer` | BPE-Tokenizer | GPT-2, LLaMA, RoBERTa |  |
| `curriculum` | Curriculum (leicht → schwer) | Bengio et al. (2009) |  |
| `klassengewichte` | Klassengewichte im Verlust | scikit-learn class_weight='balanced' |  |

### Datenschutz

Sensible Daten schützen.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `dp_sgd` | Differential Privacy (DP-SGD) | Google, Apple, US-Zensus | langsam (jedes Beispiel einzeln); BatchNorm wird dafür durch GroupNorm ersetzt |

### Überwachung & Reproduzierbarkeit

Protokolle, Checkpoints, Seeds.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `checkpoints` | Checkpoints (bestes Modell sichern) | Standard |  |
| `seed` | Fester Zufalls-Seed | wissenschaftliche Praxis |  |
| `deterministisch` | Deterministische Algorithmen | Audit-/Medizin-Anforderungen |  |
| `log_datei` | Protokoll-Datei mit Rotation | Server-Betrieb |  |
| `log_jsonl` | Maschinenlesbares Protokoll (JSONL) | MLOps-Praxis |  |
| `tensorboard` | TensorBoard | Google/TensorFlow, PyTorch | Paket 'tensorboard' |
| `wandb` | Weights & Biases | OpenAI, viele Forschungsteams | Paket 'wandb' |
| `mlflow` | MLflow-Tracking | Databricks, Unternehmen | Paket 'mlflow' |
| `gradienten_ueberwachung` | Gradienten je Schicht überwachen | W&B watch, TensorBoard |  |
| `gewichts_ueberwachung` | Gewichte je Schicht überwachen | TensorBoard-Histogramme |  |
| `zeitlimit` | Zeitlimit | SLURM-Praxis |  |
| `modellkarte` | Modellkarte beim Speichern | Mitchell et al. (2019), Hugging Face Model Cards |  |
| `fortsetzen` | Automatisch fortsetzen (neuester Checkpoint) | Lightning/Hugging Face 'resume_from_checkpoint' |  |

### Kompression & Einsatz

Nach dem Training verkleinern und exportieren.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `komprimieren_nach_training` | Nach dem Training komprimieren | Edge-Deployment |  |
| `export_nach_training` | Nach dem Training exportieren | PyTorch 2 |  |
| `qat` | Quantization-Aware Training | Google, Qualcomm (TFLite, AIMET) |  |

### Bewertung & Vertrauen

Ehrlich messen, Wahrscheinlichkeiten kalibrieren, Unsicherheit zeigen, robust gegen Störungen.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `testmenge` | Testmenge zurückhalten (15 %) | Standard in Forschung und Industrie |  |
| `metriken_auto` | Passende Metriken automatisch | scikit-learn/torchmetrics-Praxis |  |
| `stratifiziert` | Stratifizierte Aufteilung | scikit-learn StratifiedKFold |  |
| `zeitliche_aufteilung` | Zeitliche Aufteilung (keine Zukunft im Training) | Prognose-Wettbewerbe (M4/M5) |  |
| `kalibrierung` | Wahrscheinlichkeiten kalibrieren (Temperature Scaling) | Guo et al. (2017) |  |
| `schwelle_optimieren` | Entscheidungsschwelle optimieren (F1) | Praxis in Kreditscoring/Medizin |  |
| `enthaltung` | Bei Unsicherheit enthalten ('unsicher') | Selective Prediction / Reject Option |  |
| `conformal` | Conformal Prediction (90 % garantiert) | Angelopoulos & Bates (2021) |  |
| `mc_dropout` | Unsicherheit per MC-Dropout | Gal & Ghahramani (2016) |  |
| `tta` | Test-Time Augmentation | Kaggle-Standard |  |
| `adversarial_training` | Adversariales Training (PGD) | Madry et al. (2018) |  |
| `fgsm_training` | Adversariales Training (FGSM, schnell) | Goodfellow et al. (2015), Wong et al. (2020) |  |
| `beam_search` | Beam Search bei der Generierung | Übersetzungssysteme (Marian, T5) |  |
| `keine_wiederholungen` | Keine Wiederholungs-Schleifen (n-Gramm-Sperre) | Hugging Face generate() |  |
| `datenpruefung` | Datenprüfung vor dem Training | Google 'Data Validation for ML' |  |
| `robustheitspruefung` | Robustheitsprüfung nach dem Training | Robustheits-Benchmarks (RobustBench) |  |

### Experimentell

Neue Verfahren aus der Forschung – Nutzen nicht immer gesichert.

| Kennung | Technik | Wer nutzt es | Hinweis |
|---|---|---|---|
| `lookahead` | Lookahead-Optimierer | Zhang et al. (2019) |  |
| `gradient_centralization` | Gradient Centralization | Yong et al. (2020) |  |
| `grokfast` | Grokfast (schnelleres Grokking) | Lee et al. (2024) |  |
| `lion` | Lion-Optimierer | Google (Chen et al. 2023) |  |
| `muon` | Muon-Optimierer | Moonshot AI / Kimi (2025) |  |
| `gradientenrauschen` | Gradientenrauschen | Neelakantan et al. (2015) |  |
| `orthogonalitaet` | Orthogonalitäts-Regularisierung | BigGAN (DeepMind) |  |
| `schrittweise_auftauen` | Schichten schrittweise auftauen | ULMFiT (fast.ai) |  |
| `schichtweise_lr` | Schichtweise Lernraten (LLRD) | BEiT, ELECTRA |  |
| `wsd` | Warmup-Stable-Decay-Zeitplan | MiniCPM, DeepSeek |  |

## Notfall-Ausführung (wenn es trotz Verkleinern nicht passt)

`Project.fallback_options(ziel | budget_mb, training)`, `apply_fallback('auto' | kennung)`, `reduce_for(ziel, fallback='auto')`, CLI `notfall`, GUI „Reduzieren / Export → Notfall-Ausführung“.

| Kennung | Wie | Modus |
|---|---|---|
| `micro_batch` | Eingaben stückweise (`predict(chunk_size=…)`, `memory.oom_strategy`) | Inferenz |
| `split` | hintere Blöcke dauerhaft auf der CPU (`offload.mode=split`, device_map) | Inferenz/Training |
| `ram_offload` | Blöcke im RAM, einzeln aufs Gerät (`offload.mode=cpu`) | Inferenz |
| `disk_offload` / `layerweise_stapel` | Schicht für Schicht von der SSD (Memory-Map) | Inferenz |
| `quantisieren_nf4` | 4-Bit-Gewichte ohne Training | Inferenz |
| `sparsam` | Checkpointing + Mikro-Batches + Akkumulation + bf16 | Training |
| `optimierer_8bit` | 8-Bit-AdamW | Training |
| `lisa` | nur wenige, wechselnde Blöcke trainieren | Training |
| `lora` / `qlora` | nur Adapter trainieren (Basis ggf. 4 Bit) | Training |
| `stream` | Gewichte/Gradienten/Optimierer im RAM, Blöcke gestreamt | Training |
| `stream_disk_lora` | eingefrorene Basis auf der SSD, Adapter im RAM | Training |
| `cpu` | auf der CPU rechnen | beides |
| `cluster` | FSDP über mehrere Geräte | Training |

Zusätzlich: Batch automatisch teilen bei Speichermangel (`memory.oom_strategy='batch_teilen'`), Chunked Prefill für lange Prompts (`generate(..., prefill_chunk=512)`), Laden per Memory-Map (`Project.load(ordner, mmap=True)`).

## Logger

`training.logging`: Stufe `still` · `minimal` · `normal` · `ausfuehrlich` · `debug`; Ziele frei kombinierbar (`console`, `datei` mit Rotation, `jsonl`, `csv`, `python` (Standard-logging), `tensorboard`, `wandb`, `mlflow`), eigene Stufe je Ziel, Intervalle, Durchsatz/Speicher/Lernrate/Gradientennorm, Statistiken je Tensor. Eigene Logger: `projekt.add_logger(objekt_oder_logging.Logger)` oder Registry `log_backend`.

## Cluster & verteiltes Training

`training.distributed.strategy`: `ddp` (Daten parallel), `zero1` (Optimierer verteilt), `fsdp` (alles verteilt). Start: `python -m ki_baukasten verteilt projekt.json --prozesse 4 --strategie fsdp` (mehrere Rechner: `--knoten`, `--knoten-rang`, `--master`). Ausgaben und Speichern nur auf Prozess 0.

## Ablauf: Suche, Schwarm, Evolution

`Project.run()` führt `workflow` aus: Datenprüfung → Suche (`random`, `grid`, `evolution`, `pso`, `pbt`) → Training (normal, föderiert FedAvg/FedProx, Model Soup) → Test → Basis-Vergleich → Kalibrierung → Robustheit/Fairness → Kreuzvalidierung → Kompression → Export → Speichern. Einzeln: `tune(...)`, `federated_train(...)`, `model_soup(...)`, `cross_validate(k)`, `find_lr()`. Gradientenfrei: Paradigma `evolution_strategies`.

## Anforderungs-Katalog & Perfektions-Check

56 Anforderungen in 6 Phasen (Daten, Modell, Training, Bewertung, Vertrauen & Sicherheit, Einsatz & Betrieb), je Anwendung; 54 davon sind im Baukasten umsetzbar. `project.perfection_check()` zeigt für das eigene Projekt aktiv · eingebaut · aus · offen · fehlt mit Punktzahl, `apply_perfection()` schaltet alles Empfohlene ein, CLI `perfektion`, GUI „Qualität“. Vollständig: [ANFORDERUNGEN.md](ANFORDERUNGEN.md).

## Daten, Bewertung, Vertrauen, Einsatz

| Bereich | Möglichkeiten | Wo |
|---|---|---|
| Eigene Daten | CSV (Kategorien, fehlende Werte + Indikator, Skalierung, Ziel-Skalierung), Texte mit Labels (Tokenizer zeichen/wort/bpe), Bildordner, Segmentierungs-Masken, Audio (WAV → Log-Mel), Anweisungen (SFT) | `data = csv`, `text_classification`, `image_folder`, `segmentation`, `audio`, `instructions` |
| Aufteilung | zufällig, stratifiziert, zeitlich; Testmenge, die beim Neutrainieren fest bleibt | `training.validation_strategy`, `training.test_split`, `Project.test()` |
| Datenprüfung & Drift | Lecks, Duplikate, Überlappung, Klassen, NaN, Skalen, abgeschnittene Texte; PSI/KS je Merkmal, Vorhersage-Drift | `data_check()`, `apply_data_fixes()`, `drift_report()` |
| Metriken | 27 Metriken (Registry `metric`), automatisch je Aufgabe, über alle Beispiele, Richtung für Early Stopping/Suche automatisch | `training.metrics`, `training.auto_metrics` |
| Bewertung | Basis-Vergleich, Fehleranalyse (Konfusionsmatrix, sicherste Fehler), Klassengewichte | `baseline()`, `error_analysis()`, `training.class_weights` |
| Vertrauen | Temperature Scaling, Schwelle, Enthaltung, Conformal Prediction, MC-Dropout, TTA | `inference.*`, `calibrate()`, `predict_with_uncertainty()` |
| Erklärbarkeit & Fairness | Gradient × Eingabe, Integrated Gradients, Occlusion (auch Tokens), Permutations-Wichtigkeit; Disparate Impact, Equalized Odds | `explain_prediction()`, `feature_importance()`, `fairness_report()` |
| Robustheit | FGSM/PGD-Training, Robustheitsbericht (Rauschen/Angriffe) | `training.adversarial`, `robustness_report()` |
| Wenige Labels | Active Learning (Unsicherheit, Marge, Entropie, Vielfalt), Pseudo-Labeling | `select_for_labeling()`, `pseudo_label()` |
| Domänen | U-Net-Segmentierung, Audio-Klassifikation, Instruction-Tuning (Verlust nur auf Antworten), Beam Search, n-Gramm-Sperre | `model = unet`, `ask()`, `generation.*` |
| Einsatz | Modellkarte, Umgebung, gespeicherte Vorverarbeitung, Rohdaten-Vorhersage, HTTP-Dienst, Fortsetzen vom Checkpoint | `model_card()`, `save/load`, `predict_raw()`, `serve()`, `training.resume_from` |

## Leistung, Hardware & schlanker Export

| Bereich | Möglichkeiten | Wo |
|---|---|---|
| Ein Aufruf | Datenart erkennen → Vorlage → Empfehlungen → Training → Test → Basis | `Project.auto(datei)`, CLI `auto` |
| Hardware-Automatik | bf16/fp16/aus, TF32, cuDNN-Benchmark, channels_last, gepinnter Speicher je Gerät | `memory.* = auto`, `hardware_report()`, CLI `hardware` |
| Tempo | fusionierter Optimierer, `TensorBatchLoader`, eine Synchronisation je Schritt, GQA ohne KV-Kopien | automatisch |
| Automatisch einstellen | Lernrate (LR-Finder), Batchgröße passend zum GPU-Speicher | `training.auto_lr`, `training.auto_batch_size` |
| Schlanker Export | Hüllen entfernen, EMA, LoRA exakt, BatchNorm falten, Ungenutztes entfernen, Präzision, Prüfung, kompakte projekt.json | `deployment.export.*`, `export(...)`, `last_export` |

## Diagnose & automatisch verbessern

| Bereich | Möglichkeiten | Wo |
|---|---|---|
| Diagnose | instabil, lernt kaum, zu kurz, Überanpassung, kaum besser als Basis, Klassen-Ungleichgewicht, Kalibrierung, keine Testmenge, Validierung zu optimistisch – je mit Maßnahmen | `diagnose()`, `apply_diagnosis()`, CLI `diagnose`, GUI „Qualität“ |
| Verbessern | Diagnose → Maßnahmen → neu trainieren, nur Verbesserungen bleiben | `improve(rounds)`, `Project.auto(improve=…)`, CLI `auto --verbessern` |
| Wirkungsprüfung | geänderte, aber wirkungslose Einstellungen mit Grund und Abhilfe; Technik-Konflikte | `inactive_settings()`, Meldung vor `train()` |

## Anpassen – jede Schicht, jedes Modul, jeder Tensor

Was sich wie fein verändern lässt (jeweils für alle, nur bestimmte oder genau einen Treffer).

| Möglichkeit | Wo | Wirkung |
|---|---|---|
| Basis-Block | `model.block.…` | Bauplan aller Schichten (Mixer, FFN, Norm, Dropout …) |
| Schichtweise Abweichungen | `model.layer_overrides` | Selektoren `0`, `-1`, `::2`, `2-5`, `gerade` … |
| Übernahme zwischen Schichten | `@von`, `@nur`, `@ausser` | Konfiguration einer anderen Schicht ganz/teilweise |
| Modul-Chirurgie | `module_rules` | ersetzen, entfernen, LoRA/Quantisierung einzeln, Präzision, Hooks |
| Tensor-Chirurgie | `tensor_rules` | einfrieren, Pruning, Kopieren, eigene Lernraten, Beschränkungen |
| Geltungsbereich | `layers`, `match`, `scope`, `pick` | alle · nur_auswahl · nur_einer · alle_ausser |
| Platzhalter-Pfade | `Project.set_all`, `set('model.**.x', …)` | eine Einstellung überall setzen |
| Suchen / Rückgängig / Vergleich | `search`, `undo`, `redo`, `diff` | Handling großer Konfigurationen |

## Speicher im Training reduzieren

Wo der Speicher hingeht und welche Weiche hilft.

| Möglichkeit | Wo | Wirkung |
|---|---|---|
| Gemischte Präzision | `memory.mixed_precision = bf16` | halbiert Aktivierungen, schneller auf GPUs |
| Reines bf16 | `memory.param_dtype = bfloat16` | halbiert Gewichte und Gradienten |
| Gradient-Checkpointing | `memory.gradient_checkpointing` | spart 60–80 % Aktivierungen (+30 % Zeit) |
| Aktivierungs-Offload | `memory.activation_offload` | Aktivierungen im CPU-RAM |
| Blöcke trainierbar im RAM | `memory.offload.mode = stream` | Gewichte, Gradienten, Optimierer im RAM |
| 8-Bit-Optimierer | `training.optimizer = adamw8bit` | ¼ der Adam-Zustände |
| LoRA / QLoRA | `memory.lora`, `memory.quantization = nf4` | nur kleine Adapter trainieren |
| Akkumulation | `training.gradient_accumulation_steps` | große effektive Batches mit wenig Speicher |
| Schneller SSM-Scan | `scan_backward = fused` | Mamba-Aktivierungen ÷ d_state |
| Speicherprofile | `Project.apply_memory_profile('extrem_sparsam')` | alles auf einmal |
| Schätzung | `Project.memory_report()` | Gewichte, Gradienten, Optimierer, gemessene Aktivierungen |

## Generierung & Inferenz

Schneller und sparsamer Vorhersagen.

| Möglichkeit | Wo | Wirkung |
|---|---|---|
| KV-/Zustands-Cache | `generate(..., use_cache=True)` | nur neue Tokens rechnen (alle kausalen Mixer) |
| Cache-Format | `cache={'dtype': 'int8'}` | ¼ Cache-Speicher; Fenster + Attention-Sinks |
| GQA / MLA-lite | `num_kv_heads`, `kv_compression_dim` | kleinerer KV-Cache schon beim Bau |
| Kontextverlängerung | `positional.scaling = yarn` | längere Texte als im Training |
| Layer-Swapping | `memory.offload.mode = cpu | disk` | Modelle größer als der Gerätespeicher |

## Lehrer-Schüler & Weiterlernen

Wissen übertragen und neue Daten lernen, ohne Altes zu vergessen.

| Möglichkeit | Wo | Wirkung |
|---|---|---|
| Schüler trainieren | `Project.distill(width, depth)` | automatisch verkleinertes Modell lernt vom Lehrer |
| Distillation | `training.paradigm = distillation` | KL, reverse KL, JS, MSE; Feature-Distillation |
| Lehrer-Quellen | `teacher_project`, `teacher_export`, `teacher` | gespeichertes Projekt, .pt2, Modell |
| Weiterlernen | `Project.continue_training(method=…)` | lwf · ewc · replay · none |

## Import & Export

Vorhandene Modelle weiterverwenden und Ergebnisse ausliefern.

| Möglichkeit | Wo | Wirkung |
|---|---|---|
| Import | `Project.import_model(pfad)` | .pt2, TorchScript, state_dict, safetensors, Hugging Face |
| Gewichte abbilden | `key_map`, `architecture` | fremde Schlüsselnamen auf Baukasten-Modelle |
| Export | `Project.export(pfad, fmt)` | torch.export (dynamische Formen), TorchScript, ONNX, state_dict |
| Speichern/Laden | `save`, `load` | inkl. Kompression, Quantisierung, LoRA – exakt wiederhergestellt |

## Einblick & Analyse

Verstehen, was die KI tut.

| Möglichkeit | Wo | Wirkung |
|---|---|---|
| Aufbau | `Project.architecture(fmt)` | Text, Markdown + Mermaid, JSON; Anpassungspfade je Schicht |
| Probelauf | `analysis.probe.probe`, GUI „Probelauf & Gewichte“ | alle Wege, Statistiken, Gradienten, Warnungen |
| Gewichte & Attention | `weight_stats`, `heatmap`, `attention_maps` | Verteilungen, Matrizen, Attention-Karten |
| Testlauf | `analysis.probe.evaluate_model` | Metriken, Konfusionsmatrix, Beispiele |
| Ratgeber | `Project.advise()` / `apply_advice` | Speicher, Tempo, Stabilität, Struktur |
| Messung | `analysis.benchmark` | Größe, MACs, Latenz |

## Modelle eindampfen (Kompression)

`Project.compress([...])` bzw. `compression.steps`; Reihenfolge: Struktur → Heilen → Low-Rank/Sparsity → Quantisierung. Jeder Schritt wird gemessen (Parameter, Größe, MACs, Latenz, Verlust).

| Schritt | Wirkung | Wofür |
|---|---|---|
| `prune_ffn` | FFN-Neuronen entfernen (strukturiert). Echte Ersparnis an Parametern und Rechenzeit. | Transformer/SSM/Hybrid (MLP-, GLU- und MoE-Experten-FFNs). |
| `prune_heads` | Attention-Köpfe entfernen. Weniger Rechenaufwand in der Attention, bei 'group' auch kleinerer KV-Cache. | Transformer, Hybride. |
| `drop_layers` | Ganze Blöcke entfernen. Proportional schneller (Tiefe) – oft erstaunlich verlustarm. | Tiefe Modelle. |
| `prune_mlp` | Neuronen eines MLP-Modells entfernen. Schmalere Schichten. | Modell 'mlp'. |
| `low_rank` | Lineare Schichten per SVD zerlegen. Spart Parameter, wo Matrizen 'eigentlich' niedrigrangig sind. | Große Projektionen, nach dem Training. |
| `sparsify` | Unstrukturiertes Pruning (kleinste Gewichte = 0). Hohe Kompressionsraten mit Sparse-Formaten/Zip. | Speichergröße, Vorbereitung auf Sparse-Kernel. |
| `heal` | Kurze Distillation vom unkomprimierten Original. Holt einen Großteil der durch Pruning verlorenen Qualität zurück. | Nach jedem stärkeren Strukturschritt. |
| `finetune` | Normales Nachtraining. Ohne Lehrer. | Wenn kein Original vorliegt. |
| `quantize` | Gewichte quantisieren (int8/int4/nf4/pal4). ¼ bis ⅛ des Speichers. | Als letzter Schritt. |
| `dtype` | Alle Gewichte in 16 Bit. Halber Speicher, auf GPUs/Apple oft schneller. | GPU-/Mobil-Deployment. |
| `merge_lora` | LoRA-Adapter in die Gewichte einrechnen. Keine Zusatzkosten bei der Inferenz. | Vor dem Export. |

## Zielsysteme

`Project.recommend_for(ziel)` zeigt das Rezept, `reduce_for(ziel)` wendet es an; nach `train()` erscheint automatisch eine Übersicht (`deployment.targets`).

| Schlüssel | Gerät | Speicher | Leistung (Batch 1) | Formate | Export |
|---|---|---:|---:|---|---|
| `rechenzentrum_gpu` | Rechenzentrum-GPU (A100/H100, 40–80 GB) | 60000 MB | ~150000 GFLOP/s | bfloat16, int8, int4 | torch.export → AOTInductor / TensorRT |
| `gaming_gpu` | Gaming-/Workstation-GPU (8–24 GB) | 8000 MB | ~40000 GFLOP/s | float16, bfloat16, int8, int4 | torch.export / ONNX → TensorRT |
| `laptop_gpu` | Laptop-/Einsteiger-GPU (4–6 GB) | 3000 MB | ~10000 GFLOP/s | float16, int8, int4 | torch.export / ONNX |
| `apple_silicon` | Apple Silicon (M1–M4, gemeinsamer Speicher) | 6000 MB | ~5000 GFLOP/s | float16, int8, pal4 | torch.export → CoreML / ExecuTorch (MPS) |
| `server_cpu` | Server-CPU (x86, viele Kerne) | 16000 MB | ~1000 GFLOP/s | int8, bfloat16 | ONNX (onnxruntime) / torch.export |
| `laptop_cpu` | Laptop-/Desktop-CPU | 4000 MB | ~200 GFLOP/s | int8 | ONNX (onnxruntime) / torch.export |
| `smartphone` | Smartphone (ARM, 6–12 GB RAM, Akku) | 1000 MB | ~100 GFLOP/s | int8, int4, float16 | ExecuTorch (.pte aus torch.export) / ONNX Mobile / CoreML |
| `raspberry_pi` | Raspberry Pi 4/5 (ARM-CPU, 4–8 GB) | 500 MB | ~15 GFLOP/s | int8 | ONNX (onnxruntime) / ExecuTorch |
| `mikrocontroller` | Mikrocontroller (Cortex-M, ≤ 1 MB Flash) | 0.5 MB | ~0.1 GFLOP/s | int8 | ONNX → TFLite Micro / ExecuTorch für MCUs |
| `browser` | Browser (WebGPU/WebAssembly) | 200 MB | ~50 GFLOP/s | int8, float16 | ONNX (onnxruntime-web, WebGPU) |

## Bausteine (Registries)

- **Aktivierungsfunktionen** (`activation`, 33): `relu`, `leaky_relu`, `prelu`, `rrelu`, `elu`, `celu`, `selu`, `gelu`, `quick_gelu`, `silu`, `swish`, `mish`, `tanh`, `sigmoid`, `softplus`, `softsign`, `hardtanh`, `relu6`, `hardswish`, `hardsigmoid`, `logsigmoid`, `tanhshrink`, `softshrink`, `hardshrink`, `threshold`, `identity`, `squared_relu`, `star_relu`, `sine`, `snake`, `gaussian`, `softcap`, `polynomial`
- **Normalisierungen** (`normalization`, 9): `layernorm`, `rmsnorm`, `batchnorm`, `groupnorm`, `instancenorm`, `scalenorm`, `l2norm`, `dyt`, `identity`
- **Attention-Kernel** (`attention_kernel`, 5): `softmax`, `sigmoid`, `relu`, `cosine`, `linear`
- **Feed-Forward-Netze** (`feedforward`, 5): `mlp`, `glu`, `moe`, `kan`, `none`
- **Initialisierer** (`initializer`, 20): `keep`, `zeros`, `ones`, `constant`, `normal`, `uniform`, `trunc_normal`, `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `lecun_normal`, `lecun_uniform`, `orthogonal`, `eye`, `dirac`, `sparse`, `small_init`, `scaled_normal`, `siren`
- **Schichten** (`layer`, 28): `linear`, `conv`, `conv_transpose`, `depthwise_separable_conv`, `pool`, `adaptive_pool`, `global_pool`, `flatten`, `reshape`, `permute`, `dropout`, `normalization`, `activation`, `embedding`, `lstm`, `gru`, `rnn`, `self_attention`, `squeeze_excitation`, `upsample`, `pixel_shuffle`, `gaussian_noise`, `scale`, `function`, `positional_encoding`, `identity`, `mlp`, `transformer_block`
- **Zusammenführungen** (`merge`, 9): `concat`, `add`, `mul`, `mean`, `max`, `stack`, `weighted_sum`, `gated`, `first`
- **Tensor-Funktionen** (`function`, 18): `identity`, `softmax`, `log_softmax`, `l2_normalize`, `mean`, `sum`, `max`, `last`, `first`, `transpose`, `abs`, `square`, `exp`, `log1p`, `clamp`, `scale`, `detach`, `sign_ste`
- **Sequenz-Mixer** (`sequence_mixer`, 6): `attention`, `ssm`, `rnn`, `conv`, `fourier`, `none`
- **Positionskodierungen** (`positional`, 7): `none`, `sinusoidal`, `learned`, `rope`, `alibi`, `relative_bias`, `combined`
- **Modelle** (`model`, 20): `autoencoder`, `cnn`, `ensemble`, `pipeline`, `fusion`, `mixture_of_models`, `multitask`, `extern`, `mlp`, `gan`, `diffusion`, `gnn`, `graph`, `actor_critic`, `q_network`, `rnn`, `transformer`, `ssm`, `hybrid`, `unet`
- **Callbacks** (`callback`, 9): `early_stopping`, `checkpoint`, `csv_logger`, `jsonl_logger`, `gradient_monitor`, `time_limit`, `swa`, `lisa`, `gradual_unfreeze`
- **Verlustfunktionen** (`loss`, 17): `cross_entropy`, `bce`, `mse`, `l1`, `huber`, `smooth_l1`, `log_cosh`, `quantile`, `gaussian_nll`, `poisson_nll`, `focal`, `dice`, `kl_div`, `cosine`, `info_nce`, `hinge`, `nll`
- **Metriken** (`metric`, 27): `accuracy`, `top_k_accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `mcc`, `auroc`, `log_loss`, `ece`, `brier`, `mae`, `mse`, `rmse`, `r2`, `mape`, `smape`, `max_error`, `perplexity`, `token_accuracy`, `exact_match`, `bleu`, `iou`, `dice`, `multilabel_accuracy`, `subset_accuracy`, `multilabel_f1`
- **Optimierer** (`optimizer`, 13): `sgd`, `adam`, `adamw`, `nadam`, `radam`, `adamax`, `adagrad`, `adadelta`, `rmsprop`, `adafactor`, `lion`, `adamw8bit`, `muon`
- **Trainings-Paradigmen** (`paradigm`, 13): `supervised`, `causal_lm`, `masked_lm`, `seq2seq`, `autoencoder`, `contrastive`, `distillation`, `gan`, `diffusion`, `preference`, `dpo`, `evolution_strategies`, `reinforcement`
- **Lernraten-Zeitpläne** (`scheduler`, 12): `constant`, `linear`, `warmup_cosine`, `cosine_restarts`, `polynomial`, `exponential`, `step`, `inverse_sqrt`, `one_cycle`, `wsd`, `cyclic`, `plateau`
- **Logger-Ziele** (`log_backend`, 8): `console`, `datei`, `jsonl`, `csv`, `python`, `tensorboard`, `wandb`, `mlflow`
- **RL-Umgebungen** (`environment`, 5): `gridworld`, `cartpole`, `pendulum`, `bandit`, `gymnasium`
- **Belohnungsterme** (`reward`, 10): `env`, `constant`, `action_penalty`, `action_change`, `info`, `potential`, `event`, `terminal`, `rnd`, `count`
- **RL-Algorithmen** (`rl_algorithm`, 4): `ppo`, `a2c`, `reinforce`, `dqn`
- **Modul-Operationen** (`module_op`, 25): `freeze`, `unfreeze`, `replace_activation`, `replace_norm`, `identity`, `set_attr`, `dtype`, `quantize`, `lora`, `spectral_norm`, `weight_norm`, `checkpoint`, `init`, `reset`, `copy_from`, `scale_output`, `output_noise`, `clamp_output`, `stop_gradient`, `record`, `resize_linear`, `resize_norm`, `resize_attention`, `factorize`, `drop_items`
- **Tensor-Operationen** (`tensor_op`, 22): `freeze`, `unfreeze`, `init`, `scale`, `add`, `set`, `noise`, `clamp`, `prune`, `low_rank`, `orthogonalize`, `load`, `custom`, `copy`, `tie`, `lr_scale`, `weight_decay`, `grad_scale`, `grad_clip`, `grad_noise`, `grad_mask`, `constraint`
- **Tensor-Funktionen (Chirurgie)** (`tensor_function`, 6): `identity`, `abs`, `sign`, `normalize_rows`, `flip`, `shuffle`
- **Kompressions-Schritte** (`compress_step`, 11): `prune_ffn`, `prune_heads`, `drop_layers`, `prune_mlp`, `low_rank`, `sparsify`, `heal`, `finetune`, `quantize`, `dtype`, `merge_lora`
- **Datensätze** (`dataset`, 20): `multimodal`, `classification`, `regression`, `sequence_classification`, `text`, `seq2seq_toy`, `time_series`, `shapes`, `graphs`, `distribution_2d`, `preferences`, `anomaly`, `csv`, `text_classification`, `image_folder`, `segmentation`, `audio`, `instructions`, `tensor_file`, `none`
- **Vorlagen (Presets)** (`preset`, 37): `sprachmodell_mini`, `sprachmodell_moe`, `sprachmodell_mamba`, `sprachmodell_hybrid`, `sprachmodell_speichersparend`, `sprachmodell_qlora`, `sprachmodell_ram_streaming`, `sprachmodell_sft`, `text_klassifikation`, `bert_vortraining`, `uebersetzung_seq2seq`, `bildklassifikation_cnn`, `bildklassifikation_vit`, `bildklassifikation_mobil`, `segmentierung_unet`, `audio_klassifikation`, `vae_bilder`, `zeitreihen_lstm`, `zeitreihen_transformer`, `sequenz_gedaechtnis_rnn`, `tabellen_klassifikation`, `tabellen_regression`, `gan_2d`, `diffusion_2d`, `graph_klassifikation`, `anomalie_autoencoder`, `reward_modell`, `rl_cartpole_ppo`, `rl_cartpole_dqn`, `rl_gitterwelt_belohnungen`, `rl_pendel_kontinuierlich`, `ensemble_tabellen`, `multimodal_fusion`, `mixture_of_models`, `freies_netz`, `kontrastiv_vortraining`, `forschung_experimentell`

## Kommandozeile

| Befehl | Zweck |
|---|---|
| `ziele` | Ziel-Assistent: Modelle nach Zweck und Prioritäten |
| `presets` | Vorlagen auflisten |
| `modelle` | Modelle auflisten |
| `komponenten` | Registries/Komponenten auflisten |
| `zeige` | Einstellungsbaum anzeigen |
| `erklaere` | Einstellung oder Komponente erklären |
| `vorlage` | Konfigurationsdatei schreiben |
| `pruefen` | Konfiguration prüfen |
| `bauen` | Modell bauen + Übersicht |
| `speicher` | Speicherschätzung |
| `tensoren` | Parameter auflisten / Tensor-Regeln vorschauen |
| `aufbau` | komplette Aufbau-Darstellung des Modells |
| `optimieren` | Optimierungs- und Speichervorschläge |
| `suche` | Einstellungen per Volltext finden |
| `schalter` | Einstellungen An/Aus/Eigen anzeigen und umschalten |
| `techniken` | Techniken-Katalog mit Status, an/aus |
| `vergleiche` | zwei Vorlagen/Dateien vergleichen |
| `exportieren` | Modell exportieren |
| `importieren` | fremdes Modell importieren |
| `komprimieren` | Modell eindampfen |
| `zielsysteme` | Geräteprofile bzw. Übersicht für ein Modell |
| `reduzieren` | Empfehlung für ein Zielsystem (und anwenden) |
| `notfall` | Notfall-Ausführung, wenn das Modell nicht passt |
| `schueler` | kleineres Schüler-Modell per Distillation |
| `probelauf` | alle Wege durchs Netz mit Statistiken |
| `moeglichkeiten` | Katalog aller Möglichkeiten |
| `trainieren` | trainieren |
| `ablauf` | ganzen Ablauf ausführen (workflow) |
| `automl` | automatische Hyperparameter-Suche |
| `verteilt` | verteilt trainieren (mehrere GPUs/Rechner) |
| `generieren` | aus gespeichertem Projekt generieren |
| `auto` | von den Daten zur fertigen KI in einem Schritt |
| `diagnose` | Diagnose nach dem Training mit konkreten Maßnahmen (optional anwenden) |
| `hardware` | Rechengerät und automatisch gewählte Einstellungen |
| `perfektion` | Anforderungs-Katalog: Perfektions-Check eines Projekts |
| `datenpruefung` | Daten prüfen (Lecks, Duplikate, Klassen …) |
| `bewerten` | Validierung, Test, Basis, Fehleranalyse, Kalibrierung |
| `modellkarte` | Modellkarte anzeigen/schreiben |
| `drift` | neue Daten mit den Trainingsdaten vergleichen |
| `bereitstellen` | als HTTP-Dienst starten |
| `doku` | Parameter-Referenz exportieren |
| `gui` | grafische Oberfläche |

Grafische Oberfläche: `python -m ki_baukasten gui` – Menü (Datei · Assistent · Einstellungen · Techniken · Ausführen · Analyse · Qualität · Einsatz · Hilfe; „Datei → Automatisch aus Daten…“), Ziel-Assistent, Techniken-Fenster mit Reitern, Perfektions-Check, Datenprüfung, Test, Basis, Fehleranalyse, Kalibrierung, Modellkarte, HTTP-Dienst, Einstellungsbaum mit Tooltips und An/Aus/Eigen-Schaltern je Einstellung, Ansichtsfilter, Probelauf & Gewichte, Reduzieren/Export mit Notfall-Ausführung, Modell importieren, Rückgängig/Wiederholen.
