# KI-Baukasten – Parameter-Referenz

Automatisch erzeugt aus den Tooltips im Quellcode (`python -m ki_baukasten doku`).
Stufen: **Basis** → **Fortgeschritten** → **Experte** → **Experimentell** (nur Orientierung – alles ist änderbar).

## Inhalt

1. [Projekt-Konfiguration](#projekt-konfiguration)
2. [Registries und Komponenten](#registries-und-komponenten)

## Projekt-Konfiguration

Standardwerte eines leeren Projekts (`ProjectConfig()`). Modell- und Komponenten-Argumente
hängen von der gewählten Komponente ab – siehe Abschnitt *Registries und Komponenten*.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `name` | Text | `'meine_ki'` |  | Basis | **Was:** Projektname.<br>**Nutzen:** Für Ausgabeordner und Übersicht.<br>**Wofür:** Kurz, ohne Leerzeichen. |
| `domain` | Text | `'allgemein'` |  | Basis | **Was:** Anwendungsgebiet (z. B. sprache, bild, zeitreihen, tabellen, rl …).<br>**Nutzen:** Ordnet Vorlagen ein.<br>**Wofür:** Wird von Presets gesetzt. |
| `description` | Text | `''` |  | Basis | **Was:** Freie Beschreibung.<br>**Nutzen:** Dokumentation des Experiments.<br>**Wofür:** Was wird getestet? |
| `model` | Komponente aus Registry 'model' | `mlp` | Auswahl: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Das Modell (beliebig aus der Registry 'model', auch Kombinationen).<br>**Nutzen:** Architektur der KI.<br>**Wofür:** Siehe 'python -m ki_baukasten modelle'. |
| `model.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `model.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `model.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `model.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `model.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `model.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `model.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `model.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `model.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `model.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `model.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `model.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `model.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `model.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `model.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `model.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `model.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `model.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `model.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `model.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `model.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `model.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `model.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `model.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `model.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `model.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `model.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `model.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `model.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `data` | Komponente aus Registry 'dataset' | `classification` | Auswahl: multimodal \| classification \| regression \| sequence_classification \| text \| seq2seq_toy \| time_series \| shapes \| graphs \| distribution_2d \| … | Basis | **Was:** Datenquelle (synthetisch oder eigene Datei).<br>**Nutzen:** Woraus gelernt wird.<br>**Wofür:** Siehe 'python -m ki_baukasten komponenten dataset'. |
| `data.n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `data.seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `data.kind` | Text | `'spirals'` | Auswahl: blobs \| xor \| spirals \| moons \| rings | Basis | **Was:** Form der Klassen.<br>**Nutzen:** Steuert die Schwierigkeit (blobs leicht, spirals schwer).<br>**Wofür:** spirals für Nichtlinearität. |
| `data.n_classes` | Ganzzahl | `3` | ≥ 2 | Basis | **Was:** Anzahl Klassen.<br>**Nutzen:** Ausgabegröße.<br>**Wofür:** 2–10. |
| `data.n_features` | Ganzzahl | `2` | ≥ 2 | Basis | **Was:** Merkmale (2 = sichtbar; weitere sind Rauschdimensionen).<br>**Nutzen:** Zusätzliche irrelevante Merkmale testen Robustheit.<br>**Wofür:** 2–50. |
| `data.noise` | Kommazahl | `0.05` | ≥ 0.0 | Basis | **Was:** Rauschen.<br>**Nutzen:** Überlappung der Klassen (ab ~0.1 überlappen die Spiralarme im Zentrum stark).<br>**Wofür:** 0.02–0.2. |
| `training` | Unter-Konfiguration Training | ⟶ TrainingConfig |  | Basis | **Was:** Alle Einstellungen des Trainingsablaufs.<br>**Nutzen:** Wie, wie lange und womit das Modell lernt.<br>**Wofür:** Jedes Training. |
| `training.paradigm` | Komponente aus Registry 'paradigm' | `supervised` | Auswahl: supervised \| causal_lm \| masked_lm \| seq2seq \| autoencoder \| contrastive \| distillation \| gan \| diffusion \| preference \| dpo \| reinforcement | Basis | **Was:** Trainings-Paradigma (welche Art von Lernen).<br>**Nutzen:** Bestimmt, wie aus Daten Verluste werden (überwacht, generativ, kontrastiv, RL ...).<br>**Wofür:** Passend zur Aufgabe wählen; Presets setzen es automatisch. |
| `training.paradigm.task` | Text | `'auto'` | Auswahl: auto \| classification \| regression \| multilabel | Basis | **Was:** Art der überwachten Aufgabe (nur relevant, wenn keine eigenen Verluste gesetzt sind).<br>**Nutzen:** Bestimmt Standardverlust und Metriken.<br>**Wofür:** Meist 'auto'. |
| `training.epochs` | Ganzzahl | `10` | ≥ 1 | Basis | **Was:** Durchläufe durch den Datensatz.<br>**Nutzen:** Mehr = mehr Lernen, aber Überanpassungsgefahr.<br>**Wofür:** 5–100. |
| `training.max_steps` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Exakte Anzahl Optimierungsschritte (wenn gesetzt, wird 'epochs' ignoriert).<br>**Nutzen:** Feste Rechenbudgets unabhängig von der Datensatzgröße.<br>**Wofür:** LLM-Training wird meist in Schritten bemessen. |
| `training.batch_size` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Beispiele pro Schritt.<br>**Nutzen:** Größer = stabilere Gradienten, mehr Speicher.<br>**Wofür:** 8–512. |
| `training.eval_batch_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Batchgröße für Evaluierung (None = batch_size).<br>**Nutzen:** Evaluierung braucht weniger Speicher.<br>**Wofür:** 2× batch_size. |
| `training.gradient_accumulation_steps` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Gradienten über n Mikro-Batches sammeln vor einem Schritt.<br>**Nutzen:** Simuliert große Batches mit wenig Speicher (effektive Batchgröße = batch_size · n).<br>**Wofür:** Große Modelle auf kleinen GPUs. |
| `training.optimizer` | Komponente aus Registry 'optimizer' | `adamw` | Auswahl: sgd \| adam \| adamw \| nadam \| radam \| adamax \| adagrad \| adadelta \| rmsprop \| adafactor \| lion \| adamw8bit \| muon | Basis | **Was:** Optimierer.<br>**Nutzen:** Wie Gradienten zu Updates werden.<br>**Wofür:** adamw (Standard), sgd (CNNs), lion/muon (experimentell). |
| `training.optimizer.lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate – Schrittweite der Gewichtsänderungen.<br>**Nutzen:** Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.<br>**Wofür:** Jedes Training.<br>**Tipp:** Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5. |
| `training.optimizer.weight_decay` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Entkoppelter Gewichtsverfall (AdamW).<br>**Nutzen:** Korrekte Regularisierung für adaptive Optimierer – der Grund, AdamW statt Adam zu nehmen.<br>**Wofür:** 0.01…0.1 (LLMs 0.1). |
| `training.optimizer.betas` | Liste[Kommazahl] | `[0.9, 0.999]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).<br>**Nutzen:** β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.<br>**Wofür:** LLMs: [0.9, 0.95]; Standard: [0.9, 0.999]. |
| `training.optimizer.eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch ~0; größer = vorsichtigere Updates.<br>**Wofür:** fp16: 1e-6…1e-5. |
| `training.optimizer.amsgrad` | Ja/Nein | `False` |  | Experte | **Was:** AMSGrad-Variante (monotones 2. Moment).<br>**Nutzen:** Theoretische Konvergenzgarantie, praktisch selten besser.<br>**Wofür:** Experimente. |
| `training.optimizer.fused` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).<br>**Nutzen:** Schneller auf GPUs.<br>**Wofür:** Geschwindigkeit auf CUDA. |
| `training.optimizer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichende Optimierer je Optimierungsgruppe des Paradigmas, z. B. {'discriminator': {'name': 'adam', 'lr': 4e-4}}.<br>**Nutzen:** GANs brauchen oft getrennte Lernraten für Generator/Diskriminator (TTUR).<br>**Wofür:** GAN-Training. |
| `training.scheduler` | Komponente aus Registry 'scheduler' | `warmup_cosine` | Auswahl: constant \| linear \| warmup_cosine \| cosine_restarts \| polynomial \| exponential \| step \| inverse_sqrt \| one_cycle \| wsd \| cyclic \| plateau | Basis | **Was:** Lernraten-Zeitplan.<br>**Nutzen:** Warmup + Abklingen.<br>**Wofür:** warmup_cosine (Standard), constant (Debug). |
| `training.scheduler.warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `training.scheduler.warmup_ratio` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Skaliert automatisch mit der Trainingslänge.<br>**Wofür:** 0.01–0.1. |
| `training.scheduler.warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `training.scheduler.warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `training.scheduler.min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `training.no_decay_patterns` | Liste[Text] | `['bias', 'norm', 'embed', 'pos', 'log_std', 'scale', 'gamma'` |  | Fortgeschritten | **Was:** Parameter, deren Name eines dieser Teilwörter enthält, bekommen keinen Weight-Decay.<br>**Nutzen:** Biases/Normen/Embeddings zu verkleinern schadet meist.<br>**Wofür:** Standardpraxis bei Transformern. |
| `training.layerwise_lr_decay` | Kommazahl (oder None) | `None` | Bereich 0.0 … 1.0 | Experte | **Was:** Schichtweiser Lernratenfaktor: tiefere (frühere) Blöcke lernen langsamer (LR · d^(N−i)).<br>**Nutzen:** Schützt vortrainiertes Wissen in frühen Schichten.<br>**Wofür:** Feintuning (BEiT/ELECTRA: 0.65–0.9). |
| `training.losses` | Liste von Komponenten aus Registry 'loss' | 0 Elemente | Elemente: cross_entropy \| bce \| mse \| l1 \| huber \| smooth_l1 \| log_cosh \| quantile \| gaussian_nll \| poisson_nll \| focal \| dice \| kl_div \| … | Basis | **Was:** Liste gewichteter Verlustterme (leer = Standard des Paradigmas).<br>**Nutzen:** Beliebige Kombinationen mehrerer Ziele.<br>**Wofür:** Multi-Task, kombinierte Verluste. |
| `training.regularization` | Unter-Konfiguration Regularisierung | ⟶ RegularizationConfig |  | Basis | **Was:** Zusätzliche Techniken gegen Überanpassung und für robustere Modelle.<br>**Nutzen:** Ergänzen Dropout/Weight-Decay um weitere, teils experimentelle Verfahren.<br>**Wofür:** Kleine Datensätze, Generalisierung, Robustheit. |
| `training.regularization.l1` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** L1-Strafe auf alle Gewichte (λ·Σ\|w\|).<br>**Nutzen:** Erzeugt spärliche Gewichte (viele exakt 0).<br>**Wofür:** Merkmalsauswahl, kompakte Modelle.<br>**Tipp:** Klein halten: 1e-6…1e-4. |
| `training.regularization.l2` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** Explizite L2-Strafe im Verlust (λ·Σw²) – zusätzlich zu weight_decay des Optimierers.<br>**Nutzen:** Klassische Ridge-Regularisierung (wirkt bei Adam anders als AdamW-Decay).<br>**Wofür:** Selten nötig. |
| `training.regularization.orthogonality` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Strafe für Abweichung von Orthogonalität der Gewichtsmatrizen ‖WᵀW − I‖².<br>**Nutzen:** Hält Signale normerhaltend, stabilisiert tiefe Netze und GANs.<br>**Wofür:** BigGAN, tiefe Netze. |
| `training.regularization.mixup_alpha` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** Mixup: Beispiele und Ziele linear mischen (λ ~ Beta(α, α)); 0 = aus.<br>**Nutzen:** Glattere Entscheidungsgrenzen, bessere Kalibrierung.<br>**Wofür:** Bildklassifikation: 0.2–0.4.<br>**Tipp:** Nur für Eingaben mit Gleitkommazahlen; Ziele werden zu weichen Labels. |
| `training.regularization.input_noise_std` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** Gauß-Rauschen auf Eingaben im Training.<br>**Nutzen:** Robustheit gegen Messrauschen.<br>**Wofür:** Sensor-/Tabellendaten. |
| `training.regularization.gradient_noise_std` | Kommazahl | `0.0` | ≥ 0.0 | Experimentell | **Was:** Rauschen auf Gradienten (σ_t = σ/(1+t)^γ).<br>**Nutzen:** Hilft aus schlechten Minima (Neelakantan et al.).<br>**Wofür:** Sehr tiefe/schwer trainierbare Netze. |
| `training.regularization.gradient_noise_decay` | Kommazahl | `0.55` | ≥ 0.0 | Experimentell | **Was:** Abklingexponent γ des Gradientenrauschens.<br>**Nutzen:** Rauschen nimmt über die Zeit ab.<br>**Wofür:** 0.55. |
| `training.regularization.flooding_level` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experimentell | **Was:** Flooding: Verlust wird um diesen Pegel gespiegelt (\|L−b\|+b).<br>**Nutzen:** Verhindert, dass der Trainingsverlust gegen 0 geht – wirkt gegen Überanpassung.<br>**Wofür:** Kleine Datensätze (Ishida et al. 2020). |
| `training.regularization.sam_rho` | Kommazahl | `0.0` | ≥ 0.0 | Experimentell | **Was:** Sharpness-Aware Minimization: Radius der Nachbarschaft (0 = aus).<br>**Nutzen:** Sucht flache Minima → bessere Generalisierung; kostet doppelte Rechenzeit.<br>**Wofür:** Bildklassifikation (ρ ≈ 0.05).<br>**Tipp:** Nicht mit gradient_accumulation_steps > 1 kombinierbar. |
| `training.gradient_clipping` | Unter-Konfiguration Gradient-Clipping | ⟶ GradientClippingConfig |  | Basis | **Was:** Unter-Konfiguration 'Gradient-Clipping'. |
| `training.gradient_clipping.mode` | Text | `'norm'` | Auswahl: none \| norm \| value \| adaptive | Basis | **Was:** Art des Gradient-Clippings.<br>**Nutzen:** Verhindert explodierende Gradienten.<br>**Wofür:** Transformer/RNNs: norm (1.0). |
| `training.gradient_clipping.max_norm` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Maximale Gradienten-Gesamtnorm (mode=norm).<br>**Nutzen:** Begrenzt Schrittgrößen bei Ausreißern.<br>**Wofür:** 0.5–5; LLMs 1.0. |
| `training.gradient_clipping.clip_value` | Kommazahl | `1.0` | ≥ 0.0 | Fortgeschritten | **Was:** Maximaler Betrag je Gradient (mode=value).<br>**Nutzen:** Hartes Abschneiden.<br>**Wofür:** Selten. |
| `training.gradient_clipping.agc_ratio` | Kommazahl | `0.01` | ≥ 0.0 | Experte | **Was:** Erlaubtes Verhältnis ‖g‖/‖w‖ je Einheit (mode=adaptive).<br>**Nutzen:** Skaleninvariantes Clipping.<br>**Wofür:** 0.01–0.16. |
| `training.gradient_clipping.norm_type` | Kommazahl | `2.0` | ≥ 0.0 | Experte | **Was:** Norm-Typ (2 = euklidisch, inf = Maximum).<br>**Nutzen:** Art der Größenmessung.<br>**Wofür:** 2. |
| `training.ema` | Unter-Konfiguration EMA der Gewichte | ⟶ EMAConfig |  | Fortgeschritten | **Was:** Unter-Konfiguration 'EMA der Gewichte'. |
| `training.ema.enabled` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Exponentiell gleitenden Mittelwert der Gewichte mitführen.<br>**Nutzen:** Gemittelte Gewichte sind oft besser und stabiler als die letzten.<br>**Wofür:** Diffusionsmodelle (Pflicht), GANs, Bildklassifikation. |
| `training.ema.decay` | Kommazahl | `0.999` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Abklingrate (näher an 1 = längeres Gedächtnis).<br>**Nutzen:** Glättungsstärke.<br>**Wofür:** 0.999–0.9999. |
| `training.ema.start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Schritt mitteln.<br>**Nutzen:** Frühe, schlechte Gewichte auslassen.<br>**Wofür:** Ein paar hundert Schritte. |
| `training.ema.update_every` | Ganzzahl | `1` | ≥ 1 | Fortgeschritten | **Was:** Aktualisierung alle n Schritte.<br>**Nutzen:** Spart Rechenzeit.<br>**Wofür:** 1–10. |
| `training.ema.use_for_eval` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** EMA-Gewichte für Evaluierung verwenden.<br>**Nutzen:** Bewertet das gemittelte Modell.<br>**Wofür:** Standard: an. |
| `training.nan_policy` | Text | `'skip'` | Auswahl: raise \| skip \| ignore | Fortgeschritten | **Was:** Verhalten bei NaN/Inf in Verlust oder Gradienten.<br>**Nutzen:** Schützt vor Zerstörung der Gewichte.<br>**Wofür:** skip. |
| `training.curriculum` | Unter-Konfiguration Curriculum Learning | ⟶ CurriculumConfig |  | Fortgeschritten | **Was:** Unter-Konfiguration 'Curriculum Learning'. |
| `training.curriculum.strategy` | Text | `'none'` | Auswahl: none \| length \| key | Fortgeschritten | **Was:** Curriculum-Strategie.<br>**Nutzen:** Leicht → schwer kann Training beschleunigen und stabilisieren.<br>**Wofür:** Sprachmodelle (kurz → lang), Robotik, Mathe-Aufgaben. |
| `training.curriculum.start_fraction` | Kommazahl | `0.2` | Bereich 0.01 … 1.0 | Fortgeschritten | **Was:** Anteil der leichtesten Daten zu Beginn.<br>**Nutzen:** Startumfang des Curriculums.<br>**Wofür:** 0.1–0.3. |
| `training.curriculum.full_after_fraction` | Kommazahl | `0.5` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Ab diesem Trainingsfortschritt werden alle Daten genutzt.<br>**Nutzen:** Länge der Curriculum-Phase.<br>**Wofür:** 0.3–0.7. |
| `training.eval_every_steps` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Evaluierung alle n Schritte (None = nach jeder Epoche).<br>**Nutzen:** Häufiger = bessere Kontrolle, langsamer.<br>**Wofür:** Lange Epochen: alle 500–2000 Schritte. |
| `training.eval_max_batches` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Höchstens so viele Batches evaluieren.<br>**Nutzen:** Schnellere Zwischen-Evaluierungen.<br>**Wofür:** Große Validierungsmengen. |
| `training.log_every_steps` | Ganzzahl | `10` | ≥ 1 | Basis | **Was:** Ausgabe des Fortschritts alle n Schritte.<br>**Nutzen:** Übersicht im Terminal.<br>**Wofür:** 10–100. |
| `training.verbose` | Ja/Nein | `True` |  | Basis | **Was:** Fortschritt ausgeben.<br>**Nutzen:** Konsolenausgabe.<br>**Wofür:** Standard: an. |
| `training.callbacks` | Liste von Komponenten aus Registry 'callback' | 0 Elemente | Elemente: early_stopping \| checkpoint \| csv_logger \| jsonl_logger \| gradient_monitor \| time_limit \| gradual_unfreeze | Basis | **Was:** Callbacks (Early Stopping, Checkpoints, Logger ...).<br>**Nutzen:** Erweitern den Trainingsablauf.<br>**Wofür:** Nach Bedarf. |
| `training.seed` | Ganzzahl (oder None) | `42` |  | Basis | **Was:** Zufalls-Seed (None = nicht setzen).<br>**Nutzen:** Reproduzierbare Ergebnisse.<br>**Wofür:** Für Vergleiche immer setzen. |
| `training.deterministic` | Ja/Nein | `False` |  | Experte | **Was:** Deterministische Algorithmen erzwingen.<br>**Nutzen:** Bitgenaue Reproduzierbarkeit, oft langsamer.<br>**Wofür:** Debugging. |
| `training.validation_split` | Kommazahl | `0.1` | Bereich 0.0 … 0.9 | Basis | **Was:** Anteil der Trainingsdaten als Validierung (wenn keine eigenen Validierungsdaten übergeben werden).<br>**Nutzen:** Misst Generalisierung während des Trainings.<br>**Wofür:** 0.1–0.2. |
| `training.shuffle` | Ja/Nein | `True` |  | Basis | **Was:** Daten jede Epoche mischen.<br>**Nutzen:** Verhindert Reihenfolge-Effekte.<br>**Wofür:** an. |
| `training.drop_last` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzten unvollständigen Batch verwerfen.<br>**Nutzen:** Gleich große Batches (wichtig für BatchNorm).<br>**Wofür:** BatchNorm: an. |
| `training.num_workers` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Hintergrundprozesse zum Datenladen.<br>**Nutzen:** Schneller bei aufwendigem Laden.<br>**Wofür:** 0 (einfach) bis CPU-Kerne. |
| `training.pin_memory` | Ja/Nein | `False` |  | Experte | **Was:** Gepinnter Speicher für schnellere GPU-Übertragung.<br>**Nutzen:** Schneller bei CUDA.<br>**Wofür:** GPU-Training: an. |
| `memory` | Unter-Konfiguration Speicher & Präzision | ⟶ MemoryConfig |  | Basis | **Was:** Einstellungen für Speicherbedarf, numerische Präzision und Geschwindigkeit.<br>**Nutzen:** Entscheiden, ob ein Modell überhaupt in den Speicher passt und wie schnell es läuft.<br>**Wofür:** Große Modelle, kleine GPUs, CPU-Training, Inferenz-Optimierung. |
| `memory.device` | Text | `'auto'` |  | Basis | **Was:** Rechengerät: 'auto', 'cpu', 'cuda', 'cuda:1', 'mps' (Apple).<br>**Nutzen:** GPUs sind für Training meist 10–100× schneller.<br>**Wofür:** 'auto' wählt die beste verfügbare Option. |
| `memory.param_dtype` | Text | `'float32'` | Auswahl: float32 \| bfloat16 \| float16 \| float64 | Basis | **Was:** Datentyp der Gewichte.<br>**Nutzen:** 16 Bit halbiert den Speicher für Gewichte und Gradienten.<br>**Wofür:** Reines bf16-Training großer Modelle; sonst float32 + mixed_precision. |
| `memory.mixed_precision` | Text | `'none'` | Auswahl: none \| bf16 \| fp16 | Basis | **Was:** Gemischte Präzision (autocast).<br>**Nutzen:** Halbiert Aktivierungsspeicher und beschleunigt auf Tensor-Core-GPUs deutlich.<br>**Wofür:** GPU-Training: bf16 (Ampere+), fp16 (ältere GPUs). |
| `memory.grad_scaler` | Text | `'auto'` | Auswahl: auto \| on \| off | Experte | **Was:** Dynamisches Loss-Scaling.<br>**Nutzen:** Verhindert Unterlauf kleiner fp16-Gradienten.<br>**Wofür:** auto. |
| `memory.gradient_checkpointing` | Ja/Nein | `False` |  | Basis | **Was:** Aktivierungen der Blöcke nicht speichern, sondern im Backward neu berechnen.<br>**Nutzen:** Spart bei tiefen Netzen oft 60–80 % Aktivierungsspeicher – kostet ~30 % mehr Rechenzeit.<br>**Wofür:** Große Modelle / lange Sequenzen, wenn der Speicher knapp ist. |
| `memory.checkpoint_layers` | Text | `'*'` |  | Fortgeschritten | **Was:** Welche Blöcke neu berechnet werden (Selektor wie bei layer_overrides).<br>**Nutzen:** Feiner Kompromiss Speicher vs. Zeit (z. B. nur jeden zweiten Block).<br>**Wofür:** '*', '::2', '0:8'. |
| `memory.activation_offload` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Für den Backward gespeicherte Aktivierungen in den CPU-RAM auslagern.<br>**Nutzen:** Spart GPU-Speicher ohne Neuberechnung, kostet Übertragungszeit.<br>**Wofür:** Lange Sequenzen auf kleiner GPU. |
| `memory.offload` | Unter-Konfiguration Offloading / Layer-Swapping | ⟶ OffloadConfig |  | Fortgeschritten | **Was:** Auslagern von Modellblöcken auf CPU-RAM oder Festplatte (Swapping).<br>**Nutzen:** Ermöglicht Modelle, die größer als der Speicher des Rechengeräts sind.<br>**Wofür:** Inferenz sehr großer Modelle auf kleiner GPU / wenig RAM. |
| `memory.offload.mode` | Text | `'none'` | Auswahl: none \| cpu \| disk \| stream | Fortgeschritten | **Was:** Auslagerungsmodus.<br>**Nutzen:** Tauscht Geschwindigkeit gegen Speicher: disk spart am meisten, ist aber am langsamsten; stream erlaubt Training von Modellen, deren Gewichte + Optimierer nicht auf die GPU passen.<br>**Wofür:** Inferenz großer Modelle: cpu/disk. Training großer Modelle auf kleiner GPU: stream.<br>**Tipp:** stream rechnet jeden Block im Backward neu (wie Checkpointing) und überträgt die Gewichte zweimal pro Schritt – typischerweise 1.3–3× langsamer. |
| `memory.offload.targets` | Text | `'auto'` |  | Fortgeschritten | **Was:** Welche Module ausgelagert werden: 'auto' = Hauptblöcke des Modells, sonst Regex auf Modulnamen.<br>**Nutzen:** Kleine Teile (Embeddings, Kopf) bleiben dauerhaft auf dem Gerät.<br>**Wofür:** Meist 'auto'; z. B. 'blocks\.(1[0-9]\|2[0-9])' für nur hintere Blöcke. |
| `memory.offload.layers` | Text | `'*'` |  | Fortgeschritten | **Was:** Welche der automatisch gefundenen Blöcke ausgelagert werden (Selektor wie bei layer_overrides).<br>**Nutzen:** Feiner Kompromiss: nur so viele Blöcke auslagern, wie nötig – der Rest rechnet ohne Übertragung.<br>**Wofür:** '*' = alle, '0:12' = die ersten 12, '::2' = jeder zweite. Nur mit targets='auto'. |
| `memory.offload.prefetch` | Ja/Nein | `False` |  | Experimentell | **Was:** Nächsten Block auf einem eigenen CUDA-Stream vorladen, während der aktuelle rechnet (nur stream + CUDA).<br>**Nutzen:** Versteckt Übertragungszeit hinter Rechenzeit.<br>**Wofür:** Experimentell; bei Problemen ausschalten. |
| `memory.offload.directory` | Text | `'offload_cache'` |  | Fortgeschritten | **Was:** Ordner für ausgelagerte Gewichte (disk).<br>**Nutzen:** Schnelle SSD empfohlen.<br>**Wofür:** Lokale SSD. |
| `memory.offload.pin_memory` | Ja/Nein | `True` |  | Experte | **Was:** CPU-Kopien in gepinntem Speicher halten (cpu-Modus, nur CUDA).<br>**Nutzen:** Deutlich schnellere Übertragung CPU → GPU.<br>**Wofür:** CUDA. |
| `memory.offload.keep_resident` | Ganzzahl | `1` | ≥ 0 | Fortgeschritten | **Was:** Wie viele zuletzt genutzte Blöcke auf dem Gerät bleiben dürfen.<br>**Nutzen:** Mehr = weniger Nachladen, mehr Speicher.<br>**Wofür:** 1–2. |
| `memory.quantization` | Text | `'none'` | Auswahl: none \| int8 \| int4 \| nf4 \| fake_quant \| torchao_int8 \| torchao_int4 \| dynamic_int8 | Fortgeschritten | **Was:** Quantisierung der Gewichte linearer Schichten.<br>**Nutzen:** int8 braucht ¼, int4/nf4 ⅛ des float32-Gewichtsspeichers – Inferenz großer Modelle und QLoRA-Feintuning; fake_quant bereitet ein Modell auf echte Quantisierung vor.<br>**Wofür:** Inferenz/Edge: int8; Feintuning großer Modelle: nf4 + memory.lora (QLoRA).<br>**Tipp:** int8/int4/nf4 sind eine eigene Implementierung in reinem PyTorch – unabhängig von torch.ao. |
| `memory.quant_bits` | Ganzzahl | `8` | Bereich 2 … 16 | Fortgeschritten | **Was:** Bitbreite für fake_quant.<br>**Nutzen:** Weniger Bits = kleiner, ungenauer.<br>**Wofür:** 8 (sicher), 4 (aggressiv). |
| `memory.quant_group_size` | Ganzzahl (oder None) | `128` | ≥ 8 | Fortgeschritten | **Was:** Gruppengröße: so viele Gewichte einer Zeile teilen sich eine Skala (None = eine Skala pro Ausgabekanal).<br>**Nutzen:** Kleinere Gruppen = genauer (wichtig bei 4 Bit), minimal mehr Speicher für Skalen.<br>**Wofür:** int8: None oder 128; int4/nf4: 32–128. |
| `memory.quant_skip` | Text (oder None) | `'lm_head'` |  | Fortgeschritten | **Was:** Regex: Module, die NICHT quantisiert werden.<br>**Nutzen:** Ausgabeschicht und empfindliche Teile in voller Genauigkeit halten.<br>**Wofür:** 'lm_head', 'lm_head\|router'. |
| `memory.quant_only` | Text (oder None) | `None` |  | Experte | **Was:** Regex: nur diese Module quantisieren (None = alle linearen Schichten).<br>**Nutzen:** Gezielt nur große Schichten (z. B. FFN) komprimieren.<br>**Wofür:** 'ffn', 'blocks\.(1[0-9])'. |
| `memory.quant_compute_dtype` | Text | `'auto'` | Auswahl: auto \| float32 \| bfloat16 \| float16 | Experte | **Was:** Datentyp, in den die Gewichte zum Rechnen entpackt werden.<br>**Nutzen:** Geschwindigkeit vs. Genauigkeit.<br>**Wofür:** auto. |
| `memory.quant_targets` | Text | `'linear'` | Auswahl: linear \| linear_conv | Experte | **Was:** Welche Schichten fake_quant betrifft.<br>**Nutzen:** Faltungen sind empfindlicher.<br>**Wofür:** linear. |
| `memory.lora` | Unter-Konfiguration LoRA / QLoRA (Adapter-Feintuning) | ⟶ LoRAConfig |  | Fortgeschritten | **Was:** Kleine, trainierbare Niedrig-Rang-Adapter auf linearen Schichten; die Basisgewichte bleiben eingefroren.<br>**Nutzen:** Feintuning mit einem Bruchteil des Speichers (Gradienten + Optimierer nur für die Adapter).<br>**Wofür:** Anpassen großer vortrainierter Modelle; mit memory.quantization = 'nf4'/'int8' → QLoRA. |
| `memory.lora.enabled` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** LoRA-Adapter einbauen.<br>**Nutzen:** Aktiviert Adapter-Feintuning.<br>**Wofür:** Feintuning großer Modelle. |
| `memory.lora.rank` | Ganzzahl | `8` | ≥ 1 | Fortgeschritten | **Was:** Rang r der Korrektur.<br>**Nutzen:** Größer = ausdrucksstärker, mehr Parameter.<br>**Wofür:** 4–64; Standard 8–16. |
| `memory.lora.alpha` | Kommazahl | `16.0` | ≥ 0.0 | Fortgeschritten | **Was:** Skalierung α (effektiver Faktor α/r).<br>**Nutzen:** Entkoppelt Lernrate vom Rang.<br>**Wofür:** Meist 2·rank. |
| `memory.lora.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout auf die Adapter-Eingabe.<br>**Nutzen:** Regularisierung bei kleinen Datensätzen.<br>**Wofür:** 0.05–0.1. |
| `memory.lora.targets` | Text | `'auto'` |  | Fortgeschritten | **Was:** Welche Schichten Adapter bekommen: 'auto' = alle linearen Schichten in den Blöcken, sonst Regex auf Modulnamen.<br>**Nutzen:** Weniger Ziele = weniger Parameter; Attention-Projektionen sind die wichtigsten.<br>**Wofür:** 'auto', 'q_proj\|v_proj' (Original-LoRA), '.*proj' (alle Projektionen). |
| `memory.lora.layers` | Text | `'*'` |  | Fortgeschritten | **Was:** Nur in diesen Blöcken (Selektor wie bei layer_overrides; nur mit targets='auto').<br>**Nutzen:** Z. B. nur die oberen Schichten anpassen.<br>**Wofür:** '*', '-4:' (letzte vier Blöcke). |
| `memory.lora.exclude` | Text (oder None) | `'lm_head'` |  | Fortgeschritten | **Was:** Regex: diese Module nie mit Adaptern versehen.<br>**Nutzen:** Ausgabeschicht bleibt meist unverändert.<br>**Wofür:** lm_head. |
| `memory.lora.freeze_base` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Alle Nicht-Adapter-Parameter einfrieren.<br>**Nutzen:** Der eigentliche Speichervorteil von LoRA.<br>**Wofür:** an. |
| `memory.lora.train_also` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Regex: diese Parameter trotz freeze_base trainieren (z. B. Normen, Ausgabekopf).<br>**Nutzen:** Kleine Zusatzteile trainieren oft spürbar besser mit.<br>**Wofür:** 'norm', 'lm_head\|head'. |
| `memory.lora.init` | Text | `'kaiming'` | Auswahl: kaiming \| gaussian \| pissa | Experte | **Was:** Initialisierung der Adapter.<br>**Nutzen:** Bestimmt den Startpunkt.<br>**Wofür:** kaiming; pissa für schnelleres Lernen. |
| `memory.lora.use_dora` | Ja/Nein | `False` |  | Experimentell | **Was:** DoRA: zusätzlicher lernbarer Betrag pro Ausgabekanal.<br>**Nutzen:** Oft näher an vollem Feintuning, etwas teurer.<br>**Wofür:** Wenn LoRA zu schwach ist. |
| `memory.channels_last` | Ja/Nein | `False` |  | Experte | **Was:** Speicherlayout NHWC für Faltungsnetze.<br>**Nutzen:** Schneller auf Tensor-Core-GPUs.<br>**Wofür:** CNN-Training auf GPU. |
| `memory.compile` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Modell mit torch.compile übersetzen.<br>**Nutzen:** Oft 1.3–2× schneller nach Aufwärmzeit.<br>**Wofür:** Lange Trainings auf GPU. |
| `memory.compile_mode` | Text | `'default'` | Auswahl: default \| reduce-overhead \| max-autotune | Experte | **Was:** Modus von torch.compile.<br>**Nutzen:** Mehr Optimierung = längeres Kompilieren.<br>**Wofür:** default. |
| `memory.matmul_precision` | Text | `'highest'` | Auswahl: highest \| high \| medium | Experte | **Was:** Genauigkeit von fp32-Matrixmultiplikationen.<br>**Nutzen:** Geschwindigkeit vs. Genauigkeit.<br>**Wofür:** Training auf A100/H100: high. |
| `memory.cpu_threads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl CPU-Threads für PyTorch (None = automatisch).<br>**Nutzen:** Begrenzen auf geteilten Rechnern.<br>**Wofür:** Kernanzahl. |
| `memory.max_memory_fraction` | Kommazahl (oder None) | `None` | Bereich 0.05 … 1.0 | Experte | **Was:** Maximaler Anteil des GPU-Speichers für diesen Prozess.<br>**Nutzen:** Mehrere Prozesse auf einer GPU.<br>**Wofür:** 0.5. |
| `memory.empty_cache_every_n_steps` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** GPU-Cache alle n Schritte leeren.<br>**Nutzen:** Gegen Fragmentierung bei variablen Formen.<br>**Wofür:** Selten nötig. |
| `memory.memory_budget_gb` | Kommazahl (oder None) | `None` | ≥ 0.0 | Basis | **Was:** Verfügbarer Speicher in GB für die Speicherschätzung (None = automatisch erkennen).<br>**Nutzen:** Die Schätzung prüft, ob das Training hineinpasst, und schlägt Einstellungen vor.<br>**Wofür:** Planung vor dem Training großer Modelle. |
| `module_rules` | Liste von Modul-Regel | 0 Elemente |  | Fortgeschritten | **Was:** Regeln für Eingriffe in ganze Schichten (ersetzen, entfernen, LoRA/Quantisierung einzeln, Hooks …).<br>**Nutzen:** Jede Schicht jeder Art anpassen – für alle, nur bestimmte oder genau eine (scope).<br>**Wofür:** Ablationen, gezieltes Feintuning, Präzision pro Schicht, Analyse. Wirken vor den Tensor-Regeln. |
| `tensor_rules` | Liste von Tensor-Regel | 0 Elemente |  | Fortgeschritten | **Was:** Regeln für Eingriffe in einzelne Tensoren (Einfrieren, Pruning, Kopieren, eigene LR …).<br>**Nutzen:** Feinste Kontrolle bis auf einzelne Gewichte – für alle, nur bestimmte oder genau einen (scope).<br>**Wofür:** Transfer-Learning, Kompression, Experimente. |
| `auto_fit_dims` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Modellgrößen (Ein-/Ausgabe, Vokabular, Beobachtungen …) automatisch an Daten/Umgebung anpassen.<br>**Nutzen:** Vorlagen passen dadurch ohne Handarbeit zu jedem Datensatz.<br>**Wofür:** Aus, wenn exakte Werte erzwungen werden sollen. |
| `output_dir` | Text | `'ausgaben'` |  | Fortgeschritten | **Was:** Basisordner für Ausgaben (Checkpoints, Protokolle, gespeicherte Projekte).<br>**Nutzen:** Ordnung.<br>**Wofür:** ausgaben. |

## Registries und Komponenten

- [Aktivierungsfunktionen (`activation`)](#registry-activation)
- [Normalisierungen (`normalization`)](#registry-normalization)
- [Attention-Kernel (`attention_kernel`)](#registry-attention-kernel)
- [Feed-Forward-Netze (`feedforward`)](#registry-feedforward)
- [Initialisierer (`initializer`)](#registry-initializer)
- [Schichten (`layer`)](#registry-layer)
- [Zusammenführungen (`merge`)](#registry-merge)
- [Tensor-Funktionen (`function`)](#registry-function)
- [Sequenz-Mixer (`sequence_mixer`)](#registry-sequence-mixer)
- [Positionskodierungen (`positional`)](#registry-positional)
- [Modelle (`model`)](#registry-model)
- [Callbacks (`callback`)](#registry-callback)
- [Verlustfunktionen (`loss`)](#registry-loss)
- [Optimierer (`optimizer`)](#registry-optimizer)
- [Trainings-Paradigmen (`paradigm`)](#registry-paradigm)
- [Lernraten-Zeitpläne (`scheduler`)](#registry-scheduler)
- [RL-Umgebungen (`environment`)](#registry-environment)
- [Belohnungsterme (`reward`)](#registry-reward)
- [RL-Algorithmen (`rl_algorithm`)](#registry-rl-algorithm)
- [Modul-Operationen (`module_op`)](#registry-module-op)
- [Tensor-Operationen (`tensor_op`)](#registry-tensor-op)
- [Tensor-Funktionen (Chirurgie) (`tensor_function`)](#registry-tensor-function)
- [Datensätze (`dataset`)](#registry-dataset)
- [Vorlagen (Presets) (`preset`)](#registry-preset)

### Registry `activation`

<a id="registry-activation"></a>

**Aktivierungsfunktionen** – Nichtlineare Funktionen, die auf jedes Element angewendet werden. Machen Netze erst ausdrucksstark; beeinflussen Gradientenfluss und Stabilität.

Fabrik-Signatur: `factory(args) – kein zusätzlicher Kontext`

#### `activation:relu` — Stufe Basis

**Was:** Rectified Linear Unit: max(0, x).

**Nutzen:** Extrem billig, kein Sättigen für x>0 → schnelles Training.

**Wofür:** Standard für CNNs und einfache MLPs.

**Tipp:** Kann 'tote' Neuronen erzeugen (immer 0) – dann Leaky-ReLU/GELU probieren.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |

#### `activation:leaky_relu` (Aliase: `lrelu`) — Stufe Basis

**Was:** ReLU mit kleiner Steigung für negative Werte.

**Nutzen:** Kein Absterben von Neuronen.

**Wofür:** GAN-Diskriminatoren, tiefe CNNs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `negative_slope` | Kommazahl | `0.01` |  | Basis | **Was:** Steigung für negative Eingaben.<br>**Nutzen:** Verhindert 'tote' Neuronen, weil auch negative Werte einen kleinen Gradienten liefern.<br>**Wofür:** Tiefe Netze, GAN-Diskriminatoren (dort oft 0.2).<br>**Tipp:** 0.01 Standard, 0.1–0.3 bei GANs. 0 entspricht ReLU. |

#### `activation:prelu` — Stufe Fortgeschritten

**Was:** Parametrische ReLU mit lernbarer negativer Steigung.

**Nutzen:** Netz lernt optimale Steigung selbst.

**Wofür:** Bildverarbeitung (z. B. Super-Resolution).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `num_parameters` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl lernbarer Steigungen (1 = eine für alle, sonst eine pro Kanal).<br>**Nutzen:** Lässt das Netz die negative Steigung selbst lernen.<br>**Wofür:** CNNs (pro Kanal), wenn Leaky-ReLU-Steigung unklar ist.<br>**Tipp:** Bei pro-Kanal-Variante muss num_parameters = Kanalanzahl sein. |
| `init` | Kommazahl | `0.25` |  | Fortgeschritten | **Was:** Startwert der lernbaren negativen Steigung.<br>**Nutzen:** Bestimmt das Verhalten zu Trainingsbeginn.<br>**Wofür:** Feintuning des Startverhaltens. |

#### `activation:rrelu` — Stufe Fortgeschritten

**Was:** Randomisierte Leaky-ReLU.

**Nutzen:** Zufällige Steigung im Training regularisiert.

**Wofür:** Kleine Datensätze.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `lower` | Kommazahl | `0.125` |  | Basis | **Was:** Untere Grenze der zufälligen negativen Steigung.<br>**Nutzen:** Zufall im Training wirkt regularisierend.<br>**Wofür:** Kleine Datensätze. |
| `upper` | Kommazahl | `0.3333333333333333` |  | Basis | **Was:** Obere Grenze der zufälligen negativen Steigung.<br>**Nutzen:** Im Eval-Modus wird der Mittelwert verwendet.<br>**Wofür:** Kleine Datensätze. |

#### `activation:elu` — Stufe Basis

**Was:** Exponential Linear Unit.

**Nutzen:** Glatte negative Seite, Aktivierungsmittel nahe 0.

**Wofür:** Tiefe MLPs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `alpha` | Kommazahl | `1.0` |  | Basis | **Was:** Sättigungswert für negative Eingaben (-alpha).<br>**Nutzen:** Glatte negative Seite → Mittelwert der Aktivierungen näher an 0, schnelleres Lernen.<br>**Wofür:** Alternative zu ReLU in tiefen MLPs. |

#### `activation:celu` — Stufe Fortgeschritten

**Was:** Stetig differenzierbare ELU.

**Nutzen:** Glatter als ELU.

**Wofür:** Wie ELU.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `alpha` | Kommazahl | `1.0` | ≥ 1e-06 | Basis | **Was:** Alpha der stetig differenzierbaren ELU.<br>**Nutzen:** Wie ELU, aber für alle alpha stetig differenzierbar.<br>**Wofür:** Wie ELU. |

#### `activation:selu` — Stufe Basis

**Was:** Scaled ELU – selbstnormalisierend.

**Nutzen:** Hält Mittelwert/Varianz der Aktivierungen stabil, ganz ohne Normalisierungsschichten.

**Wofür:** Tiefe MLPs ohne BatchNorm ('Self-Normalizing Networks').

**Tipp:** Nur mit LeCun-Normal-Initialisierung und AlphaDropout korrekt wirksam.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |

#### `activation:gelu` — Stufe Basis

**Was:** Gaussian Error Linear Unit: x · Φ(x).

**Nutzen:** Glatt, nicht-monoton; empirisch sehr gut in Transformern.

**Wofür:** BERT, GPT, ViT – Standard in Transformer-FFNs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |

#### `activation:quick_gelu` — Stufe Fortgeschritten

**Was:** Schnelle GELU-Näherung x·sigmoid(1.702x).

**Nutzen:** Billiger als exakte GELU.

**Wofür:** CLIP-kompatible Modelle.

#### `activation:silu` (Aliase: `swish1`) — Stufe Basis

**Was:** Sigmoid Linear Unit: x · sigmoid(x) (auch Swish).

**Nutzen:** Glatt, selbst-gated; sehr gut in modernen Netzen.

**Wofür:** LLaMA (als SwiGLU-Gate), EfficientNet, Diffusionsmodelle.

#### `activation:swish` — Stufe Fortgeschritten

**Was:** Swish mit einstellbarem/lernbarem beta.

**Nutzen:** Form zwischen linear und ReLU frei wählbar.

**Wofür:** Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `beta` | Kommazahl | `1.0` |  | Basis | **Was:** Steilheit: f(x) = x · sigmoid(beta · x).<br>**Nutzen:** beta=1 entspricht SiLU; groß → ReLU-ähnlich; klein → fast linear.<br>**Wofür:** Experimente mit glatten ReLU-Varianten. |
| `learnable` | Ja/Nein | `False` |  | Experimentell | **Was:** beta als lernbaren Parameter behandeln.<br>**Nutzen:** Das Netz passt die Form der Aktivierung selbst an.<br>**Wofür:** Architektursuche, Forschung. |

#### `activation:mish` — Stufe Basis

**Was:** Mish: x · tanh(softplus(x)).

**Nutzen:** Sehr glatt, oft leicht besser als ReLU in CNNs.

**Wofür:** Objekterkennung (YOLOv4).

#### `activation:tanh` — Stufe Basis

**Was:** Tangens hyperbolicus, Ausgabe in (-1, 1).

**Nutzen:** Nullzentriert; sättigt aber für große \|x\| (verschwindende Gradienten).

**Wofür:** RNNs, Ausgaben im Bereich [-1, 1] (z. B. Bild-Generatoren, Aktionen).

#### `activation:sigmoid` — Stufe Basis

**Was:** Logistische Funktion, Ausgabe in (0, 1).

**Nutzen:** Interpretierbar als Wahrscheinlichkeit/Gate.

**Wofür:** Gates (LSTM/GLU), binäre Ausgaben.

**Tipp:** Als Zwischenaktivierung in tiefen Netzen meist schlecht (Sättigung).

#### `activation:softplus` — Stufe Basis

**Was:** Glatte ReLU: log(1 + e^x).

**Nutzen:** Immer positiv und differenzierbar.

**Wofür:** Varianzen, Raten, Intensitäten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `beta` | Kommazahl | `1.0` | ≥ 1e-06 | Basis | **Was:** Schärfe der Softplus-Kurve.<br>**Nutzen:** Größeres beta → näher an ReLU.<br>**Wofür:** Positive Ausgaben (z. B. Varianzen). |
| `threshold` | Kommazahl | `20.0` |  | Experte | **Was:** Ab diesem Wert (x*beta) wird linear gerechnet.<br>**Nutzen:** Numerische Stabilität für große Eingaben.<br>**Wofür:** Selten zu ändern. |

#### `activation:softsign` — Stufe Fortgeschritten

**Was:** x / (1 + \|x\|).

**Nutzen:** Wie tanh, aber langsamer sättigend.

**Wofür:** Alternative zu tanh.

#### `activation:hardtanh` — Stufe Fortgeschritten

**Was:** Stückweise lineares tanh.

**Nutzen:** Billig, begrenzt.

**Wofür:** Mobile/Quantisierung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `min_val` | Kommazahl | `-1.0` |  | Basis | **Was:** Untere Grenze.<br>**Nutzen:** Begrenzt Ausgaben hart.<br>**Wofür:** Quantisierungsfreundliche Netze. |
| `max_val` | Kommazahl | `1.0` |  | Basis | **Was:** Obere Grenze.<br>**Nutzen:** Begrenzt Ausgaben hart.<br>**Wofür:** Quantisierungsfreundliche Netze. |

#### `activation:relu6` — Stufe Basis

**Was:** ReLU begrenzt auf [0, 6].

**Nutzen:** Robust bei niedriger Präzision.

**Wofür:** MobileNet, Quantisierung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |

#### `activation:hardswish` — Stufe Basis

**Was:** Stückweise Näherung von Swish.

**Nutzen:** Swish-Qualität zu ReLU-Kosten.

**Wofür:** MobileNetV3, Edge-Geräte.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |

#### `activation:hardsigmoid` — Stufe Fortgeschritten

**Was:** Stückweise lineare Sigmoid.

**Nutzen:** Billiges Gate.

**Wofür:** Squeeze-Excitation in MobileNets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |

#### `activation:logsigmoid` — Stufe Fortgeschritten

**Was:** log(sigmoid(x)).

**Nutzen:** Numerisch stabil für Log-Wahrscheinlichkeiten.

**Wofür:** Spezielle Verlustterme.

#### `activation:tanhshrink` — Stufe Experte

**Was:** x - tanh(x).

**Nutzen:** Unterdrückt kleine Werte weich.

**Wofür:** Signalverarbeitung.

#### `activation:softshrink` — Stufe Experte

**Was:** Weiches Schrumpfen um lambda.

**Nutzen:** Sparsity mit stetiger Funktion.

**Wofür:** Sparse Coding (ISTA).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lambd` | Kommazahl | `0.5` | ≥ 0.0 | Basis | **Was:** Schwellwert, unterhalb dessen (Betrag) auf 0 gesetzt wird.<br>**Nutzen:** Erzeugt Sparsity (viele exakte Nullen).<br>**Wofür:** Sparse Coding, Entrauschen. |

#### `activation:hardshrink` — Stufe Experte

**Was:** Hartes Nullsetzen kleiner Werte.

**Nutzen:** Exakte Sparsity.

**Wofür:** Entrauschen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lambd` | Kommazahl | `0.5` | ≥ 0.0 | Basis | **Was:** Schwellwert, unterhalb dessen (Betrag) auf 0 gesetzt wird.<br>**Nutzen:** Erzeugt Sparsity (viele exakte Nullen).<br>**Wofür:** Sparse Coding, Entrauschen. |

#### `activation:threshold` — Stufe Experte

**Was:** Allgemeine Schwellwertfunktion.

**Nutzen:** Frei definierbares Abschneiden.

**Wofür:** Spezialfälle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `threshold` | Kommazahl | `0.0` |  | Basis | **Was:** Schwelle.<br>**Nutzen:** Werte ≤ Schwelle werden ersetzt.<br>**Wofür:** Spezialfälle. |
| `value` | Kommazahl | `0.0` |  | Basis | **Was:** Ersatzwert unterhalb der Schwelle.<br>**Nutzen:** Bestimmt die Ausgabe dort.<br>**Wofür:** Spezialfälle. |

#### `activation:identity` (Aliase: `linear`, `none`, `keine`) — Stufe Basis

**Was:** Keine Aktivierung (f(x) = x).

**Nutzen:** Lässt Werte unverändert – z. B. für Ausgabeschichten bei Regression.

**Wofür:** Ausgabeschichten, lineare Modelle, Ablationsstudien.

#### `activation:squared_relu` (Aliase: `relu2`) — Stufe Fortgeschritten

**Was:** Quadrierte ReLU: relu(x)².

**Nutzen:** In der 'Primer'-Suche als starke FFN-Aktivierung gefunden.

**Wofür:** Transformer-FFNs als Alternative zu GELU.

#### `activation:star_relu` — Stufe Experimentell

**Was:** StarReLU: s·relu(x)²+b.

**Nutzen:** Varianz-normierte quadrierte ReLU.

**Wofür:** MetaFormer.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `scale` | Kommazahl | `0.8944` |  | Basis | **Was:** Skalierung s in s·relu(x)² + b.<br>**Nutzen:** Normiert die Varianz der Ausgabe.<br>**Wofür:** MetaFormer-Architekturen. |
| `bias` | Kommazahl | `-0.4472` |  | Basis | **Was:** Verschiebung b in s·relu(x)² + b.<br>**Nutzen:** Zentriert die Ausgabe.<br>**Wofür:** MetaFormer-Architekturen. |
| `learnable` | Ja/Nein | `True` |  | Basis | **Was:** scale/bias lernbar.<br>**Nutzen:** Mehr Flexibilität bei 2 Parametern.<br>**Wofür:** Standard. |

#### `activation:sine` (Aliase: `sin`, `siren`) — Stufe Fortgeschritten

**Was:** Sinus-Aktivierung (SIREN).

**Nutzen:** Stellt hohe Frequenzen dar, Ableitungen sind wieder glatt.

**Wofür:** Implizite Repräsentationen (NeRF-artig), PDE-Lösungen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `w0` | Kommazahl | `1.0` |  | Basis | **Was:** Frequenzfaktor: f(x) = sin(w0 · x).<br>**Nutzen:** Periodische Aktivierungen (SIREN) können feine Details/hohe Frequenzen darstellen.<br>**Wofür:** Implizite neuronale Repräsentationen (Bilder/Signale als Funktion), physikalische Felder.<br>**Tipp:** SIREN nutzt w0=30 in der ersten Schicht und spezielle Initialisierung (siehe Initialisierer 'siren'). |

#### `activation:snake` — Stufe Fortgeschritten

**Was:** Snake: x + sin²(αx)/α.

**Nutzen:** Periodisch + monoton.

**Wofür:** Audio, periodische Zeitreihen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `alpha` | Kommazahl | `1.0` | ≥ 1e-06 | Basis | **Was:** Frequenz: f(x) = x + sin²(alpha·x)/alpha.<br>**Nutzen:** Kombiniert monotonen Trend mit Periodizität – gut für periodische Signale.<br>**Wofür:** Audio-Synthese (BigVGAN), Zeitreihen mit Saisonalität. |
| `learnable` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** alpha lernbar machen.<br>**Nutzen:** Netz findet passende Frequenzen selbst.<br>**Wofür:** Standard bei Snake. |
| `num_channels` | Ganzzahl | `1` | ≥ 1 | Fortgeschritten | **Was:** Anzahl unabhängiger alpha-Werte (pro Kanal, letzte Dimension).<br>**Nutzen:** Jeder Kanal kann eine eigene Frequenz lernen.<br>**Wofür:** Mehrkanalige Signale.<br>**Tipp:** Muss zur Größe der letzten Dimension passen oder 1 sein. |

#### `activation:gaussian` — Stufe Experimentell

**Was:** Gauß-Glocke als Aktivierung.

**Nutzen:** Lokale Antworten wie RBF-Netze.

**Wofür:** Funktionsapproximation.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `sigma` | Kommazahl | `1.0` | ≥ 1e-06 | Basis | **Was:** Breite: f(x) = exp(-x² / (2σ²)).<br>**Nutzen:** Lokale, glockenförmige Antwort (wie RBF-Netze).<br>**Wofür:** Radiale-Basis-Netze, Funktionsapproximation. |

#### `activation:softcap` — Stufe Fortgeschritten

**Was:** Weiche Begrenzung cap·tanh(x/cap).

**Nutzen:** Verhindert Ausreißer-Werte.

**Wofür:** Logits, Attention-Scores.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `cap` | Kommazahl | `30.0` | ≥ 1e-06 | Basis | **Was:** Grenze: f(x) = cap · tanh(x / cap).<br>**Nutzen:** Begrenzt Werte weich, ohne Gradienten ganz abzuschneiden.<br>**Wofür:** Logit-Begrenzung (Gemma 2), Stabilisierung extremer Aktivierungen. |

#### `activation:polynomial` — Stufe Experimentell

**Was:** Lernbares Polynom als Aktivierung.

**Nutzen:** Form der Nichtlinearität wird gelernt.

**Wofür:** Forschung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `degree` | Ganzzahl | `3` | Bereich 1 … 9 | Basis | **Was:** Grad des Polynoms f(x) = Σ a_k · x^k.<br>**Nutzen:** Das Netz lernt die Form der Aktivierung selbst.<br>**Wofür:** Forschung zu lernbaren Aktivierungen.<br>**Tipp:** Hohe Grade sind numerisch instabil – Eingaben sollten normalisiert sein. |
| `init` | Text | `'identity'` | Auswahl: identity \| relu_like | Basis | **Was:** Startform des Polynoms.<br>**Nutzen:** Guter Start erleichtert das Training.<br>**Wofür:** Selten zu ändern. |


### Registry `normalization`

<a id="registry-normalization"></a>

**Normalisierungen** – Schichten, die Aktivierungen auf eine stabile Skala bringen. Ermöglichen tiefere Netze, höhere Lernraten und stabileres Training.

Fabrik-Signatur: `factory(args, num_features, dims='feature'|1|2|3)`

#### `normalization:layernorm` (Aliase: `ln`, `layer_norm`) — Stufe Basis

**Was:** Layer Normalization: Mittelwert 0, Varianz 1 über die Features jedes Elements.

**Nutzen:** Unabhängig von der Batchgröße, stabil für Sequenzen.

**Wofür:** Standard in Transformern (BERT, GPT-2) und RNNs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `elementwise_affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala (γ) und Verschiebung (β) pro Feature.<br>**Nutzen:** Gibt dem Netz die Möglichkeit, die Normalisierung teilweise rückgängig zu machen.<br>**Wofür:** Fast immer an; aus für parameterfreie Normierung. |
| `bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Verschiebung β verwenden.<br>**Nutzen:** Ohne Bias etwas weniger Parameter; manche LLMs verzichten darauf.<br>**Wofür:** Moderne Transformer (oft ohne Bias). |

#### `normalization:rmsnorm` (Aliase: `rms`) — Stufe Basis

**Was:** Root-Mean-Square-Normalisierung (nur Skalierung, kein Zentrieren).

**Nutzen:** Billiger als LayerNorm bei praktisch gleicher Qualität.

**Wofür:** Moderne LLMs (LLaMA, Mistral, Gemma).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `learnable` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |

#### `normalization:batchnorm` (Aliase: `bn`, `batch_norm`) — Stufe Basis

**Was:** Batch Normalization: normalisiert jeden Kanal über den Batch.

**Nutzen:** Beschleunigt CNN-Training stark, wirkt leicht regularisierend.

**Wofür:** CNNs mit Batchgröße ≥ 16.

**Tipp:** Bei kleinen Batches oder Sequenzmodellen besser GroupNorm/LayerNorm.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `momentum` | Kommazahl (oder None) | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Gewicht neuer Batches im laufenden Mittelwert/Varianz.<br>**Nutzen:** Klein = glattere, trägere Statistik; None = kumulativer Durchschnitt.<br>**Wofür:** Kleine/rauschige Batches → kleineres Momentum. |
| `affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard: an. |
| `track_running_stats` | Ja/Nein | `True` |  | Experte | **Was:** Laufende Statistik für die Inferenz mitführen.<br>**Nutzen:** Aus → auch bei Inferenz Batch-Statistik (Ergebnis hängt dann vom Batch ab!).<br>**Wofür:** Spezialfälle (z. B. Test-Time-Adaption). |

#### `normalization:groupnorm` (Aliase: `gn`, `group_norm`) — Stufe Basis

**Was:** Group Normalization: normalisiert Kanalgruppen pro Element.

**Nutzen:** Batchgrößen-unabhängig, gut für CNNs mit kleinen Batches.

**Wofür:** Detektion, Segmentierung, Diffusions-U-Nets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `num_groups` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Anzahl Kanalgruppen.<br>**Nutzen:** 1 Gruppe = LayerNorm über Kanäle+Raum; Gruppen = Kanäle → InstanceNorm.<br>**Wofür:** CNNs mit kleinen Batches (Detektion, Segmentierung, Diffusions-U-Nets).<br>**Tipp:** Kanalanzahl muss durch num_groups teilbar sein – sonst wird automatisch der größte Teiler genommen. |
| `eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard. |

#### `normalization:instancenorm` (Aliase: `in`, `instance_norm`) — Stufe Fortgeschritten

**Was:** Instance Normalization: normalisiert jeden Kanal jedes Elements über den Raum.

**Nutzen:** Entfernt Kontrast-/Stil-Informationen pro Bild.

**Wofür:** Stil-Transfer, Bild-Generatoren.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `affine` | Ja/Nein | `False` |  | Basis | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Stil-Transfer: oft an. |
| `track_running_stats` | Ja/Nein | `False` |  | Experte | **Was:** Laufende Statistik mitführen.<br>**Nutzen:** Normalerweise aus bei InstanceNorm.<br>**Wofür:** Selten. |

#### `normalization:scalenorm` — Stufe Experte

**Was:** ScaleNorm: g · x / \|\|x\|\| mit einem einzigen lernbaren Skalar.

**Nutzen:** Extrem wenige Parameter, schnelleres Konvergieren in kleinen Transformern.

**Wofür:** Maschinelle Übersetzung mit wenig Daten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |

#### `normalization:l2norm` — Stufe Experimentell

**Was:** Projektion auf die Einheitskugel (Länge 1), ohne Parameter.

**Nutzen:** Alle Repräsentationen haben dieselbe Norm – Basis von 'normalisierten' Transformern (nGPT).

**Wofür:** Forschung, Embedding-Räume mit Kosinus-Ähnlichkeit.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |

#### `normalization:dyt` (Aliase: `dynamic_tanh`) — Stufe Experimentell

**Was:** Dynamic Tanh: γ·tanh(α·x)+β als Ersatz für Normalisierung.

**Nutzen:** Keine Statistik-Berechnung nötig; laut Paper gleichwertig zu LayerNorm in Transformern.

**Wofür:** Experimente mit normalisierungsfreien Transformern.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `alpha_init` | Kommazahl | `0.5` |  | Basis | **Was:** Startwert von α in tanh(α·x).<br>**Nutzen:** Bestimmt, ab welcher Größe Werte gesättigt werden.<br>**Wofür:** Normalisierungsfreie Transformer.<br>**Tipp:** Paper empfiehlt 0.5 (größere Modelle evtl. kleiner). |
| `affine` | Ja/Nein | `True` |  | Basis | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard. |

#### `normalization:identity` (Aliase: `none`, `keine`) — Stufe Basis

**Was:** Keine Normalisierung.

**Nutzen:** Spart Rechenzeit; nötig für Ablationen oder selbstnormalisierende Netze (SELU).

**Wofür:** Flache Netze, SELU-Netze, Experimente.


### Registry `attention_kernel`

<a id="registry-attention-kernel"></a>

**Attention-Kernel** – Rechenvorschrift, die aus Query/Key-Ähnlichkeiten Gewichte über die Values macht. Bestimmt Ausdrucksstärke, Kosten (quadratisch vs. linear) und Stabilität.

Fabrik-Signatur: `factory(args) -> Modul mit forward(q, k, v, masks, bias, scale, dropout_p)`

#### `attention_kernel:softmax` (Aliase: `standard`, `dot_product`) — Stufe Basis

**Was:** Standard-Softmax-Attention (scaled dot product).

**Nutzen:** Bewährt, ausdrucksstark; Kosten quadratisch in der Sequenzlänge.

**Wofür:** Fast alle Transformer.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Basis | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |

#### `attention_kernel:sigmoid` — Stufe Experimentell

**Was:** Sigmoid-Attention: jedes Gewicht unabhängig in (0, 1).

**Nutzen:** Keine Konkurrenz zwischen Keys – kann mehrere Stellen gleichzeitig voll beachten.

**Wofür:** Forschung; laut Apple-Paper 2024 gleichwertig zu Softmax mit passendem Bias.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `bias` | Kommazahl (oder None) | `None` |  | Experte | **Was:** Konstanter Bias b in sigmoid(qk + b). None = -log(Sequenzlänge).<br>**Nutzen:** Hält die Summe der Gewichte anfangs nahe 1 – wichtig für stabiles Training.<br>**Wofür:** Sigmoid-Attention (Ramapuram et al. 2024). |

#### `attention_kernel:relu` — Stufe Experimentell

**Was:** ReLU-Attention, normiert durch die Sequenzlänge.

**Nutzen:** Erzeugt spärliche Gewichte, keine Exponentialfunktion.

**Wofür:** Forschung, hardware-effiziente Varianten.

#### `attention_kernel:cosine` — Stufe Fortgeschritten

**Was:** Kosinus-Ähnlichkeit mit lernbarer Temperatur.

**Nutzen:** Begrenzte Logits → stabil auch in sehr großen Modellen.

**Wofür:** Vision-Transformer (Swin V2), instabile Trainings.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init_scale` | Kommazahl | `10.0` | ≥ 0.001 | Basis | **Was:** Startwert der lernbaren Temperatur τ (Logits = τ · cos(q, k)).<br>**Nutzen:** Kosinus-Ähnlichkeit liegt in [-1, 1]; τ macht die Softmax ausreichend scharf.<br>**Wofür:** Swin-Transformer-V2-artige Stabilisierung großer Modelle. |
| `learnable` | Ja/Nein | `True` |  | Basis | **Was:** τ lernbar machen.<br>**Nutzen:** Netz wählt die Schärfe selbst.<br>**Wofür:** Standard. |

#### `attention_kernel:linear` — Stufe Fortgeschritten

**Was:** Lineare Attention (Kernel-Näherung).

**Nutzen:** Kosten wachsen nur linear mit der Sequenzlänge.

**Wofür:** Sehr lange Sequenzen, Streaming.

**Tipp:** Unterstützt kein Sliding Window und keinen Positions-Bias (RoPE geht).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `feature_map` | Text | `'elu'` | Auswahl: elu \| relu \| exp | Basis | **Was:** Merkmalsabbildung φ für die Kernel-Näherung.<br>**Nutzen:** Bestimmt, wie gut die Softmax-Attention angenähert wird.<br>**Wofür:** Lange Sequenzen mit linearen Kosten. |
| `eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch 0.<br>**Wofür:** Selten zu ändern. |


### Registry `feedforward`

<a id="registry-feedforward"></a>

**Feed-Forward-Netze** – Positionsweise Teilnetze in Transformer-Blöcken. Tragen den Großteil der Kapazität eines Transformers.

Fabrik-Signatur: `factory(args, d_model)`

#### `feedforward:mlp` — Stufe Basis

**Was:** Klassisches FFN: Linear → Aktivierung → Linear.

**Nutzen:** Einfach, bewährt.

**Wofür:** GPT-2, BERT, ViT.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `hidden_mult` | Kommazahl | `4.0` | ≥ 0.01 | Basis | **Was:** Breite der versteckten Schicht relativ zu d_model.<br>**Nutzen:** Mehr Breite = mehr Kapazität und Parameter (FFN ≈ 2·mult·d² Parameter).<br>**Wofür:** Klassisch 4 (GPT/BERT). |
| `hidden_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Exakte Breite der versteckten Schicht (überschreibt hidden_mult).<br>**Nutzen:** Für exakte Nachbauten bekannter Architekturen.<br>**Wofür:** Gewichte-Import, exakte Parameterzahlen. |
| `multiple_of` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Versteckte Breite auf ein Vielfaches davon aufrunden.<br>**Nutzen:** Vielfache von 64/256 rechnen auf GPUs schneller (Tensor Cores).<br>**Wofür:** Geschwindigkeitsoptimierung (LLaMA: 256). |
| `activation` | Komponente aus Registry 'activation' | `gelu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung zwischen den beiden linearen Schichten.<br>**Nutzen:** Bestimmt die Nichtlinearität des FFN.<br>**Wofür:** gelu (Standard), relu, squared_relu ... |
| `activation.approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |
| `bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in den linearen Schichten.<br>**Nutzen:** Klassisch an; moderne LLMs oft aus.<br>**Wofür:** Kompatibilität. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach der Aktivierung (innerhalb des FFN).<br>**Nutzen:** Regularisiert die versteckte Schicht.<br>**Wofür:** Kleine Datensätze. |

#### `feedforward:glu` (Aliase: `swiglu`, `gated`) — Stufe Basis

**Was:** Gated Linear Unit FFN (SwiGLU mit silu).

**Nutzen:** Gating verbessert Qualität messbar bei gleicher Parameterzahl.

**Wofür:** LLaMA, Mistral, PaLM.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `hidden_mult` | Kommazahl | `2.6666666666666665` | ≥ 0.01 | Basis | **Was:** Breite relativ zu d_model (GLU hat 3 Matrizen → Standard 8/3 hält die Parameter wie MLP mit 4).<br>**Nutzen:** Mehr Breite = mehr Kapazität.<br>**Wofür:** LLaMA: 8/3 (aufgerundet auf multiple_of). |
| `hidden_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Exakte Breite der versteckten Schicht (überschreibt hidden_mult).<br>**Nutzen:** Für exakte Nachbauten bekannter Architekturen.<br>**Wofür:** Gewichte-Import, exakte Parameterzahlen. |
| `multiple_of` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Versteckte Breite auf ein Vielfaches davon aufrunden.<br>**Nutzen:** Vielfache von 64/256 rechnen auf GPUs schneller (Tensor Cores).<br>**Wofür:** Geschwindigkeitsoptimierung (LLaMA: 256). |
| `activation` | Komponente aus Registry 'activation' | `silu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Gate-Aktivierung: silu → SwiGLU, gelu → GeGLU, sigmoid → GLU, relu → ReGLU, identity → Bilinear.<br>**Nutzen:** Das Gate entscheidet elementweise, welche Information durchgelassen wird.<br>**Wofür:** SwiGLU ist Standard moderner LLMs. |
| `bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den linearen Schichten.<br>**Nutzen:** Moderne LLMs verzichten darauf.<br>**Wofür:** Kompatibilität. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach der Aktivierung (innerhalb des FFN).<br>**Nutzen:** Regularisiert die versteckte Schicht.<br>**Wofür:** Kleine Datensätze. |

#### `feedforward:moe` (Aliase: `mixture_of_experts`) — Stufe Fortgeschritten

**Was:** Mixture-of-Experts: Router wählt pro Token die besten k Experten.

**Nutzen:** Sehr viele Parameter bei geringen Rechenkosten pro Token.

**Wofür:** Große Sprachmodelle (Mixtral, DeepSeek, Switch).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `num_experts` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Anzahl Experten.<br>**Nutzen:** Mehr Experten = viel mehr Parameter bei (fast) gleichen Rechenkosten pro Token.<br>**Wofür:** Große Modelle mit begrenztem Rechenbudget (Mixtral: 8, DeepSeek: 64+). |
| `top_k` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Wie viele Experten pro Token aktiv sind.<br>**Nutzen:** Mehr = bessere Qualität, aber höhere Kosten. 1 = Switch-Transformer.<br>**Wofür:** Mixtral: 2.<br>**Tipp:** Muss ≤ num_experts sein. |
| `expert` | Komponente aus Registry 'feedforward' | `glu` | Auswahl: mlp \| glu \| moe \| kan \| none | Basis | **Was:** Bauart jedes Experten (beliebiges FFN, sogar ein weiteres MoE).<br>**Nutzen:** Experten können MLP, GLU oder exotische Varianten sein.<br>**Wofür:** Standard: GLU (SwiGLU). |
| `expert.hidden_mult` | Kommazahl | `2.6666666666666665` | ≥ 0.01 | Basis | **Was:** Breite relativ zu d_model (GLU hat 3 Matrizen → Standard 8/3 hält die Parameter wie MLP mit 4).<br>**Nutzen:** Mehr Breite = mehr Kapazität.<br>**Wofür:** LLaMA: 8/3 (aufgerundet auf multiple_of). |
| `expert.hidden_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Exakte Breite der versteckten Schicht (überschreibt hidden_mult).<br>**Nutzen:** Für exakte Nachbauten bekannter Architekturen.<br>**Wofür:** Gewichte-Import, exakte Parameterzahlen. |
| `expert.multiple_of` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Versteckte Breite auf ein Vielfaches davon aufrunden.<br>**Nutzen:** Vielfache von 64/256 rechnen auf GPUs schneller (Tensor Cores).<br>**Wofür:** Geschwindigkeitsoptimierung (LLaMA: 256). |
| `expert.activation` | Komponente aus Registry 'activation' | `silu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Gate-Aktivierung: silu → SwiGLU, gelu → GeGLU, sigmoid → GLU, relu → ReGLU, identity → Bilinear.<br>**Nutzen:** Das Gate entscheidet elementweise, welche Information durchgelassen wird.<br>**Wofür:** SwiGLU ist Standard moderner LLMs. |
| `expert.bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den linearen Schichten.<br>**Nutzen:** Moderne LLMs verzichten darauf.<br>**Wofür:** Kompatibilität. |
| `expert.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach der Aktivierung (innerhalb des FFN).<br>**Nutzen:** Regularisiert die versteckte Schicht.<br>**Wofür:** Kleine Datensätze. |
| `num_shared_experts` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Zusätzliche Experten, die *jedes* Token immer durchläuft.<br>**Nutzen:** Fangen gemeinsames Wissen ab, Routing-Experten können sich spezialisieren.<br>**Wofür:** DeepSeek-MoE-Stil. |
| `normalize_topk` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Gewichte der gewählten Experten auf Summe 1 normieren.<br>**Nutzen:** Hält die Ausgabegröße unabhängig von der Router-Sicherheit.<br>**Wofür:** Mixtral: an. |
| `router_noise_std` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Rauschen auf die Router-Logits im Training (Noisy Top-k Gating).<br>**Nutzen:** Fördert Erkundung, verhindert Kollaps auf wenige Experten.<br>**Wofür:** Training großer MoEs. |
| `router_bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias im Router.<br>**Nutzen:** Erlaubt globale Vorlieben für Experten.<br>**Wofür:** Selten. |
| `capacity_factor` | Kommazahl (oder None) | `None` | ≥ 0.1 | Experte | **Was:** Maximale Tokens pro Experte = factor · (Tokens·top_k / Experten). None = unbegrenzt.<br>**Nutzen:** Begrenzt Lastspitzen (überzählige Tokens werden für diesen Experten verworfen).<br>**Wofür:** Verteiltes Training mit festen Puffergrößen. |
| `aux_loss_coef` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Gewicht des Lastverteilungs-Verlusts (Switch-Transformer).<br>**Nutzen:** Bestraft ungleichmäßige Expertennutzung – sonst lernt oft nur ein Experte.<br>**Wofür:** Jedes MoE-Training. 0 = aus. |
| `z_loss_coef` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Gewicht des Router-z-Loss (bestraft große Router-Logits).<br>**Nutzen:** Stabilisiert das Routing numerisch (ST-MoE).<br>**Wofür:** Große MoEs mit Instabilitäten (typisch 1e-3). |
| `expert_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Wahrscheinlichkeit, im Training eine Experten-Zuweisung zu verwerfen.<br>**Nutzen:** Regularisiert, erzwingt Redundanz zwischen Experten.<br>**Wofür:** Experimente. |

#### `feedforward:kan` (Aliase: `kolmogorov_arnold`) — Stufe Experimentell

**Was:** Kolmogorov-Arnold-FFN (lernbare Kantenfunktionen, FastKAN-Variante).

**Nutzen:** Andere induktive Annahme als MLPs: glatte, interpretierbare 1D-Funktionen pro Verbindung.

**Wofür:** Forschung (Liu et al. 2024), wissenschaftliche Regression.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `hidden_mult` | Kommazahl | `2.0` | ≥ 0.01 | Basis | **Was:** Breite der versteckten KAN-Schicht relativ zu d_model.<br>**Nutzen:** KAN-Schichten haben grid_size-mal mehr Parameter als lineare – daher kleiner als bei MLP.<br>**Wofür:** 1–2. |
| `grid_size` | Ganzzahl | `8` | ≥ 2 | Basis | **Was:** Anzahl Basisfunktionen pro Eingabe.<br>**Nutzen:** Feinere Funktionen, mehr Parameter (∝ grid_size).<br>**Wofür:** 5–10. |
| `grid_range` | Liste[Kommazahl] | `[-2.0, 2.0]` | Länge 2 | Experte | **Was:** Wertebereich der Basisfunktions-Zentren (Eingaben werden vorher normalisiert).<br>**Nutzen:** Sollte die typischen Werte abdecken.<br>**Wofür:** [-2, 2]. |
| `base_activation` | Komponente aus Registry 'activation' (oder None) | `silu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Zusätzlicher Basis-Pfad: lineare Schicht auf act(x) (wie im KAN-Original).<br>**Nutzen:** Stabilisiert das Training, gibt einen glatten Grundverlauf.<br>**Wofür:** silu; None = aus. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout zwischen den KAN-Schichten.<br>**Nutzen:** Regularisierung.<br>**Wofür:** 0–0.1. |

#### `feedforward:none` (Aliase: `keins`, `identity`) — Stufe Experimentell

**Was:** Kein FFN (Block besteht nur aus Attention).

**Nutzen:** Ablationsstudien, extrem leichte Modelle.

**Wofür:** Forschung.


### Registry `initializer`

<a id="registry-initializer"></a>

**Initialisierer** – Verfahren zum Setzen der Startwerte von Gewichten. Entscheidet mit, ob ein tiefes Netz überhaupt trainierbar ist.

Fabrik-Signatur: `factory(args, tensor, num_layers=None) – überschreibt tensor in-place`

#### `initializer:keep` (Aliase: `default`, `pytorch`) — Stufe Basis

**Was:** Nichts ändern – PyTorch-Standardinitialisierung behalten.

**Nutzen:** PyTorch initialisiert Linear/Conv bereits sinnvoll (Kaiming-uniform mit a=√5).

**Wofür:** Wenn keine besondere Initialisierung nötig ist.

#### `initializer:zeros` — Stufe Basis

**Was:** Alles auf 0.

**Nutzen:** Neutraler Start.

**Wofür:** Biases; Ausgangsprojektionen für 'Start als Identität' (Zero-Init-Residual).

#### `initializer:ones` — Stufe Basis

**Was:** Alles auf 1.

**Nutzen:** Neutrale Skala.

**Wofür:** Norm-Gewichte.

#### `initializer:constant` — Stufe Basis

**Was:** Alles auf einen festen Wert.

**Nutzen:** Gezielte Startwerte.

**Wofür:** Gate-Biases, Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `value` | Kommazahl | `0.0` |  | Basis | **Was:** Wert, auf den alle Elemente gesetzt werden.<br>**Nutzen:** Deterministischer Start.<br>**Wofür:** Biases, Gates (z. B. Forget-Gate-Bias = 1). |

#### `initializer:normal` — Stufe Basis

**Was:** Normalverteilt N(mean, std²).

**Nutzen:** Einfach und kontrollierbar.

**Wofür:** Transformer (std=0.02), Embeddings.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `mean` | Kommazahl | `0.0` |  | Basis | **Was:** Mittelwert der Normalverteilung.<br>**Nutzen:** Verschiebt alle Startwerte.<br>**Wofür:** Fast immer 0. |
| `std` | Kommazahl | `0.02` | ≥ 0.0 | Basis | **Was:** Standardabweichung.<br>**Nutzen:** Bestimmt die Startgröße der Gewichte.<br>**Wofür:** Transformer: 0.02 (GPT/BERT-Konvention). |

#### `initializer:uniform` — Stufe Basis

**Was:** Gleichverteilt in [a, b].

**Nutzen:** Begrenzte Startwerte.

**Wofür:** Einfache Netze.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `a` | Kommazahl | `-0.05` |  | Basis | **Was:** Untere Grenze.<br>**Nutzen:** Startwerte gleichverteilt in [a, b].<br>**Wofür:** Einfache Netze. |
| `b` | Kommazahl | `0.05` |  | Basis | **Was:** Obere Grenze.<br>**Nutzen:** Startwerte gleichverteilt in [a, b].<br>**Wofür:** Einfache Netze. |

#### `initializer:trunc_normal` — Stufe Basis

**Was:** Abgeschnittene Normalverteilung.

**Nutzen:** Keine Ausreißer-Startwerte.

**Wofür:** ViT, BERT, große LLMs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `mean` | Kommazahl | `0.0` |  | Basis | **Was:** Mittelwert.<br>**Nutzen:** Zentrum der Verteilung.<br>**Wofür:** Meist 0. |
| `std` | Kommazahl | `0.02` | ≥ 0.0 | Basis | **Was:** Standardabweichung.<br>**Nutzen:** Startgröße der Gewichte.<br>**Wofür:** ViT/BERT: 0.02. |
| `cutoff_std` | Kommazahl | `2.0` | ≥ 0.0 | Fortgeschritten | **Was:** Abschneiden bei ±cutoff_std Standardabweichungen.<br>**Nutzen:** Verhindert seltene, extrem große Startwerte.<br>**Wofür:** Vision Transformer, große Sprachmodelle. |

#### `initializer:xavier_uniform` (Aliase: `glorot_uniform`) — Stufe Basis

**Was:** Xavier/Glorot gleichverteilt: Varianz 2/(fan_in+fan_out).

**Nutzen:** Erhält Varianz in beide Richtungen für symmetrische Aktivierungen (tanh, sigmoid, linear).

**Wofür:** Transformer-Projektionen, tanh-Netze.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `gain` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Skalierungsfaktor der Streuung.<br>**Nutzen:** Kompensiert die Varianzänderung durch die Aktivierung.<br>**Wofür:** Wird ignoriert, wenn 'nonlinearity' gesetzt ist (dann automatisch berechnet). |
| `nonlinearity` | Text (oder None) | `None` | Auswahl: linear \| conv1d \| conv2d \| conv3d \| sigmoid \| tanh \| relu \| leaky_relu \| selu | Fortgeschritten | **Was:** Aktivierung, für die der Gain automatisch berechnet wird.<br>**Nutzen:** z. B. relu → √2, tanh → 5/3.<br>**Wofür:** Wenn die nachfolgende Aktivierung bekannt ist. |

#### `initializer:xavier_normal` (Aliase: `glorot_normal`) — Stufe Basis

**Was:** Xavier/Glorot normalverteilt.

**Nutzen:** Wie xavier_uniform, aber Normalverteilung.

**Wofür:** Wie xavier_uniform.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `gain` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Skalierungsfaktor der Streuung.<br>**Nutzen:** Kompensiert die Varianzänderung durch die Aktivierung.<br>**Wofür:** Wird ignoriert, wenn 'nonlinearity' gesetzt ist (dann automatisch berechnet). |
| `nonlinearity` | Text (oder None) | `None` | Auswahl: linear \| conv1d \| conv2d \| conv3d \| sigmoid \| tanh \| relu \| leaky_relu \| selu | Fortgeschritten | **Was:** Aktivierung, für die der Gain automatisch berechnet wird.<br>**Nutzen:** z. B. relu → √2, tanh → 5/3.<br>**Wofür:** Wenn die nachfolgende Aktivierung bekannt ist. |

#### `initializer:kaiming_uniform` (Aliase: `he_uniform`) — Stufe Basis

**Was:** Kaiming/He gleichverteilt: Varianz 2/fan.

**Nutzen:** Erhält Varianz bei ReLU-Netzen (ReLU halbiert die Varianz).

**Wofür:** CNNs und MLPs mit ReLU.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `a` | Kommazahl | `0.0` |  | Fortgeschritten | **Was:** Negative Steigung der nachfolgenden (Leaky-)ReLU.<br>**Nutzen:** Passt die Varianz für Leaky-ReLU an.<br>**Wofür:** Nur bei leaky_relu relevant. |
| `mode` | Text | `'fan_in'` | Auswahl: fan_in \| fan_out | Basis | **Was:** Welche Varianz erhalten bleibt.<br>**Nutzen:** fan_in ist Standard; fan_out bei ResNets üblich.<br>**Wofür:** CNNs. |
| `nonlinearity` | Text | `'relu'` | Auswahl: linear \| conv1d \| conv2d \| conv3d \| sigmoid \| tanh \| relu \| leaky_relu \| selu | Basis | **Was:** Nachfolgende Aktivierung.<br>**Nutzen:** Bestimmt den Gain.<br>**Wofür:** relu oder leaky_relu. |

#### `initializer:kaiming_normal` (Aliase: `he_normal`) — Stufe Basis

**Was:** Kaiming/He normalverteilt.

**Nutzen:** Wie kaiming_uniform.

**Wofür:** ResNets (oft mode=fan_out).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `a` | Kommazahl | `0.0` |  | Fortgeschritten | **Was:** Negative Steigung der nachfolgenden (Leaky-)ReLU.<br>**Nutzen:** Passt die Varianz für Leaky-ReLU an.<br>**Wofür:** Nur bei leaky_relu relevant. |
| `mode` | Text | `'fan_in'` | Auswahl: fan_in \| fan_out | Basis | **Was:** Welche Varianz erhalten bleibt.<br>**Nutzen:** fan_in ist Standard; fan_out bei ResNets üblich.<br>**Wofür:** CNNs. |
| `nonlinearity` | Text | `'relu'` | Auswahl: linear \| conv1d \| conv2d \| conv3d \| sigmoid \| tanh \| relu \| leaky_relu \| selu | Basis | **Was:** Nachfolgende Aktivierung.<br>**Nutzen:** Bestimmt den Gain.<br>**Wofür:** relu oder leaky_relu. |

#### `initializer:lecun_normal` — Stufe Basis

**Was:** LeCun normal: Varianz 1/fan_in.

**Nutzen:** Pflicht für selbstnormalisierende SELU-Netze.

**Wofür:** SELU-MLPs.

#### `initializer:lecun_uniform` — Stufe Fortgeschritten

**Was:** LeCun gleichverteilt.

**Nutzen:** Varianz 1/fan_in.

**Wofür:** Wie lecun_normal.

#### `initializer:orthogonal` — Stufe Basis

**Was:** (Halb-)orthogonale Matrix.

**Nutzen:** Erhält Normen exakt – verhindert explodierende/verschwindende Signale über viele Zeitschritte.

**Wofür:** Rekurrente Gewichte (RNN/LSTM), RL-Policies (PPO-Konvention: gain √2 bzw. 0.01).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `gain` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Skalierung der orthogonalen Matrix.<br>**Nutzen:** Normerhaltung.<br>**Wofür:** RNNs, tiefe Netze. |

#### `initializer:eye` (Aliase: `identity`) — Stufe Fortgeschritten

**Was:** Einheitsmatrix.

**Nutzen:** Schicht startet als Identität.

**Wofür:** Residual-artige Starts, Adapter.

#### `initializer:dirac` — Stufe Experte

**Was:** Dirac-Delta für Faltungen.

**Nutzen:** Faltung startet als Identität.

**Wofür:** Sehr tiefe CNNs ohne Residuals.

#### `initializer:sparse` — Stufe Experte

**Was:** Dünn besetzte Startmatrix.

**Nutzen:** Viele Nullen, wenige Verbindungen.

**Wofür:** Reservoir-Computing, Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `sparsity` | Kommazahl | `0.9` | Bereich 0.0 … 1.0 | Basis | **Was:** Anteil der Nullen pro Spalte.<br>**Nutzen:** Sparse Start – manche Netze lernen damit schneller.<br>**Wofür:** Echo-State-Netze. |
| `std` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Streuung der Nicht-Null-Werte.<br>**Nutzen:** Startgröße.<br>**Wofür:** Klein halten. |

#### `initializer:small_init` — Stufe Fortgeschritten

**Was:** Kleine Normal-Init (Varianz 2/(5d)).

**Nutzen:** Stabil ohne Warmup.

**Wofür:** Pre-Norm-Transformer.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `factor` | Kommazahl | `0.4` | ≥ 0.0 | Basis | **Was:** Varianz = factor / d (Standard 2/5 = 0.4).<br>**Nutzen:** Kleine Init stabilisiert Pre-Norm-Transformer ohne Warmup.<br>**Wofür:** 'Transformers without Tears'-Schema. |

#### `initializer:scaled_normal` — Stufe Fortgeschritten

**Was:** Normalverteilung mit tiefenabhängiger Skalierung std/√(2N).

**Nutzen:** Hält die Varianz des Residual-Stroms in tiefen Netzen konstant.

**Wofür:** Ausgangsprojektionen von Attention/FFN in GPT-artigen Modellen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `std` | Kommazahl | `0.02` | ≥ 0.0 | Basis | **Was:** Basis-Standardabweichung.<br>**Nutzen:** Vor der Tiefenskalierung.<br>**Wofür:** 0.02. |
| `num_layers` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Anzahl Schichten N; std wird durch √(2N) geteilt.<br>**Nutzen:** Verhindert, dass sich Residual-Beiträge über viele Schichten aufsummieren und explodieren.<br>**Wofür:** Ausgangsprojektionen in Residual-Blöcken (GPT-2-Schema).<br>**Tipp:** None = automatisch vom Modell (Anzahl Blöcke). |

#### `initializer:siren` — Stufe Fortgeschritten

**Was:** SIREN-Initialisierung für Sinus-Netze.

**Nutzen:** Ohne sie lernen Sinus-Netze kaum.

**Wofür:** Zusammen mit Aktivierung 'sine'.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `w0` | Kommazahl | `30.0` | ≥ 1e-06 | Basis | **Was:** Frequenzfaktor der Sinus-Aktivierung.<br>**Nutzen:** Muss zur Aktivierung passen.<br>**Wofür:** SIREN. |
| `is_first` | Ja/Nein | `False` |  | Basis | **Was:** Erste Schicht eines SIREN-Netzes.<br>**Nutzen:** Erste Schicht nutzt eine andere Grenze (1/fan_in).<br>**Wofür:** SIREN. |


### Registry `layer`

<a id="registry-layer"></a>

**Schichten** – Einzelne Netzschichten für frei definierte Architekturen. Aus ihnen lässt sich jede beliebige Netzstruktur als Graph zusammenstecken.

Fabrik-Signatur: `factory(args, in_shape) – in_shape ohne Batch-Dimension`

#### `layer:linear` (Aliase: `dense`, `fc`) — Stufe Basis

**Was:** Voll verbundene Schicht y = xWᵀ + b (wirkt auf die letzte Achse).

**Nutzen:** Grundbaustein fast jedes Netzes.

**Wofür:** Überall.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `out_features` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Anzahl Ausgabe-Features.<br>**Nutzen:** Breite der Schicht.<br>**Wofür:** Jede lineare Schicht. |
| `bias` | Ja/Nein | `True` |  | Basis | **Was:** Bias-Vektor addieren.<br>**Nutzen:** Verschiebt die Ausgabe; fast immer sinnvoll.<br>**Wofür:** Standard. |

#### `layer:conv` (Aliase: `convolution`) — Stufe Basis

**Was:** Faltung (1D/2D/3D, Kanäle in Achse 1).

**Nutzen:** Lokale, verschiebungsinvariante Mustererkennung mit geteilten Gewichten.

**Wofür:** Bilder, Audio, Sensorsignale.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dims` | Ganzzahl | `2` | Auswahl: 1 \| 2 \| 3 | Basis | **Was:** Räumliche Dimensionalität der Faltung.<br>**Nutzen:** 1D für Signale, 2D für Bilder, 3D für Volumen/Video.<br>**Wofür:** Datenart. |
| `out_channels` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Anzahl Filter (Ausgabekanäle).<br>**Nutzen:** Mehr Filter = mehr erkannte Muster, mehr Rechenaufwand.<br>**Wofür:** Jede Faltung. |
| `kernel_size` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Größe des Filterfensters.<br>**Nutzen:** Größer = größeres rezeptives Feld pro Schicht.<br>**Wofür:** 3 ist Standard; 7 im Stem. |
| `stride` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Schrittweite des Filters.<br>**Nutzen:** 2 halbiert die Auflösung (Downsampling ohne Pooling).<br>**Wofür:** Downsampling. |
| `padding` | Ganzzahl oder Text | `'same'` |  | Basis | **Was:** Randauffüllung: Zahl, 'same' (Größe behalten) oder 'valid' (keine).<br>**Nutzen:** Steuert, ob die Ausgabe schrumpft.<br>**Wofür:** 'same' bei stride 1; bei stride > 1 wird 'same' automatisch zu kernel//2. |
| `dilation` | Ganzzahl | `1` | ≥ 1 | Fortgeschritten | **Was:** Abstand zwischen Filterelementen (atrous convolution).<br>**Nutzen:** Vergrößert das rezeptive Feld ohne zusätzliche Parameter.<br>**Wofür:** Segmentierung, Audio (WaveNet). |
| `groups` | Ganzzahl | `1` | ≥ 1 | Fortgeschritten | **Was:** Gruppierte Faltung (Kanäle in Gruppen getrennt verarbeitet).<br>**Nutzen:** Spart Parameter; groups = Kanäle → depthwise.<br>**Wofür:** ResNeXt, MobileNet.<br>**Tipp:** Ein- und Ausgabekanäle müssen durch groups teilbar sein. |
| `bias` | Ja/Nein | `True` |  | Basis | **Was:** Bias addieren.<br>**Nutzen:** Bei nachfolgender BatchNorm überflüssig.<br>**Wofür:** Aus vor BatchNorm. |
| `padding_mode` | Text | `'zeros'` | Auswahl: zeros \| reflect \| replicate \| circular | Experte | **Was:** Art der Randauffüllung.<br>**Nutzen:** Beeinflusst Artefakte an Bildrändern.<br>**Wofür:** reflect bei Bildgenerierung, circular bei periodischen Daten. |

#### `layer:conv_transpose` (Aliase: `deconv`) — Stufe Basis

**Was:** Transponierte Faltung (lernbares Upsampling).

**Nutzen:** Vergrößert Feature-Maps.

**Wofür:** Decoder, GAN-Generatoren.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dims` | Ganzzahl | `2` | Auswahl: 1 \| 2 \| 3 | Basis | **Was:** Dimensionalität.<br>**Nutzen:** Wie bei Faltung.<br>**Wofür:** Datenart. |
| `out_channels` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Ausgabekanäle.<br>**Nutzen:** Anzahl Filter.<br>**Wofür:** Decoder/Generatoren. |
| `kernel_size` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Filtergröße.<br>**Nutzen:** 4 mit stride 2 verdoppelt sauber die Auflösung.<br>**Wofür:** Generatoren. |
| `stride` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Upsampling-Faktor.<br>**Nutzen:** 2 verdoppelt die Auflösung.<br>**Wofür:** Upsampling. |
| `padding` | Ganzzahl | `1` | ≥ 0 | Basis | **Was:** Randauffüllung.<br>**Nutzen:** Bestimmt die exakte Ausgabegröße.<br>**Wofür:** Mit k=4,s=2: 1. |
| `output_padding` | Ganzzahl | `0` | ≥ 0 | Experte | **Was:** Zusätzliche Ausgabegröße.<br>**Nutzen:** Behebt Mehrdeutigkeiten.<br>**Wofür:** Selten. |
| `bias` | Ja/Nein | `True` |  | Basis | **Was:** Bias addieren.<br>**Nutzen:** Wie bei Faltung.<br>**Wofür:** Standard. |

#### `layer:depthwise_separable_conv` (Aliase: `separable_conv`) — Stufe Fortgeschritten

**Was:** Tiefen- + Punktfaltung.

**Nutzen:** ~8–9× weniger Rechenaufwand als normale 3×3-Faltung.

**Wofür:** MobileNet, Edge.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dims` | Ganzzahl | `2` | Auswahl: 1 \| 2 \| 3 | Basis | **Was:** Dimensionalität.<br>**Nutzen:** Wie bei Faltung.<br>**Wofür:** Datenart. |
| `out_channels` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Ausgabekanäle.<br>**Nutzen:** Anzahl Filter der Punkt-Faltung.<br>**Wofür:** Effiziente CNNs. |
| `kernel_size` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Filtergröße der Tiefen-Faltung.<br>**Nutzen:** Rezeptives Feld.<br>**Wofür:** 3 oder 5. |
| `stride` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Schrittweite.<br>**Nutzen:** Downsampling.<br>**Wofür:** 1 oder 2. |
| `bias` | Ja/Nein | `False` |  | Basis | **Was:** Bias.<br>**Nutzen:** Vor Normalisierung überflüssig.<br>**Wofür:** Meist aus. |

#### `layer:pool` — Stufe Basis

**Was:** Lokales Pooling (max/avg/lp).

**Nutzen:** Reduziert Auflösung, macht robuster gegen kleine Verschiebungen.

**Wofür:** CNNs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dims` | Ganzzahl | `2` | Auswahl: 1 \| 2 \| 3 | Basis | **Was:** Dimensionalität.<br>**Nutzen:** Wie bei Faltung.<br>**Wofür:** Datenart. |
| `kind` | Text | `'max'` | Auswahl: max \| avg \| lp | Basis | **Was:** Pooling-Art.<br>**Nutzen:** Max für Detektion markanter Merkmale, Avg für glatte Zusammenfassung.<br>**Wofür:** Downsampling. |
| `kernel_size` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Fenstergröße.<br>**Nutzen:** Größe des zusammengefassten Bereichs.<br>**Wofür:** 2 oder 3. |
| `stride` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Schrittweite (None = kernel_size).<br>**Nutzen:** Downsampling-Faktor.<br>**Wofür:** Meist None. |
| `padding` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Randauffüllung.<br>**Nutzen:** Für überlappendes Pooling.<br>**Wofür:** Selten. |
| `norm_type` | Kommazahl | `2.0` | ≥ 1.0 | Experte | **Was:** p der Lp-Norm (nur kind=lp).<br>**Nutzen:** p→∞ entspricht Max-Pooling.<br>**Wofür:** Selten. |

#### `layer:adaptive_pool` — Stufe Basis

**Was:** Pooling auf feste Zielgröße.

**Nutzen:** Eingabegrößen-unabhängige Köpfe.

**Wofür:** CNN-Köpfe.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dims` | Ganzzahl | `2` | Auswahl: 1 \| 2 \| 3 | Basis | **Was:** Dimensionalität.<br>**Nutzen:** Wie bei Faltung.<br>**Wofür:** Datenart. |
| `kind` | Text | `'avg'` | Auswahl: avg \| max | Basis | **Was:** Pooling-Art.<br>**Nutzen:** Siehe Pool.<br>**Wofür:** Köpfe von CNNs. |
| `output_size` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Zielgröße pro Raumachse.<br>**Nutzen:** Macht die Ausgabe unabhängig von der Eingabegröße.<br>**Wofür:** 1 = globales Pooling. |

#### `layer:global_pool` — Stufe Basis

**Was:** Globales Pooling zu [B, C].

**Nutzen:** Ein Vektor pro Beispiel.

**Wofür:** Vor dem Klassifikationskopf.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `kind` | Text | `'avg'` | Auswahl: avg \| max \| avgmax | Basis | **Was:** Art des globalen Poolings (über alle Raum-/Zeitachsen).<br>**Nutzen:** Verdichtet Feature-Maps zu einem Vektor.<br>**Wofür:** Übergang Faltungsteil → Klassifikationskopf. |

#### `layer:flatten` — Stufe Basis

**Was:** Abflachen zu einem Vektor.

**Nutzen:** Übergang zu linearen Schichten.

**Wofür:** CNN → MLP.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `start_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Ab dieser Achse abflachen (1 = alles außer Batch).<br>**Nutzen:** Macht aus Feature-Maps einen Vektor.<br>**Wofür:** Vor linearen Schichten. |

#### `layer:reshape` — Stufe Basis

**Was:** Umformen.

**Nutzen:** Beliebige neue Form.

**Wofür:** Graph-Modelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `shape` | Liste[Ganzzahl] | `[-1]` |  | Basis | **Was:** Neue Form ohne Batch-Achse (eine Achse darf -1 sein).<br>**Nutzen:** Formt Tensoren um.<br>**Wofür:** Graph-Modelle. |

#### `layer:permute` — Stufe Basis

**Was:** Achsen vertauschen.

**Nutzen:** Layout-Wechsel.

**Wofür:** Sequenz ↔ Faltung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dims` | Liste[Ganzzahl] | `[1, 0]` | ≥ 0 | Basis | **Was:** Neue Achsenreihenfolge ohne Batch-Achse (0 = erste Nicht-Batch-Achse).<br>**Nutzen:** Vertauscht Achsen, z. B. [T, C] → [C, T].<br>**Wofür:** Wechsel zwischen Sequenz- und Faltungslayout. |

#### `layer:dropout` — Stufe Basis

**Was:** Dropout-Regularisierung (verschiedene Varianten).

**Nutzen:** Verhindert Co-Adaption von Neuronen.

**Wofür:** Überall.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `p` | Kommazahl | `0.1` | Bereich 0.0 … 1.0 | Basis | **Was:** Wahrscheinlichkeit, ein Element (bzw. Kanal/Pfad) auf 0 zu setzen.<br>**Nutzen:** Klassische Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze, große Modelle. |
| `kind` | Text | `'standard'` | Auswahl: standard \| channel \| alpha \| path | Basis | **Was:** Dropout-Variante.<br>**Nutzen:** Bestimmt, was weggelassen wird.<br>**Wofür:** Je nach Datenart. |

#### `layer:normalization` (Aliase: `norm`) — Stufe Basis

**Was:** Normalisierungsschicht.

**Nutzen:** Stabilisiert Aktivierungen.

**Wofür:** Überall.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `norm` | Komponente aus Registry 'normalization' | `layernorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierungsverfahren.<br>**Nutzen:** Siehe Registry 'normalization'.<br>**Wofür:** Überall. |
| `norm.eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `norm.elementwise_affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala (γ) und Verschiebung (β) pro Feature.<br>**Nutzen:** Gibt dem Netz die Möglichkeit, die Normalisierung teilweise rückgängig zu machen.<br>**Wofür:** Fast immer an; aus für parameterfreie Normierung. |
| `norm.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Verschiebung β verwenden.<br>**Nutzen:** Ohne Bias etwas weniger Parameter; manche LLMs verzichten darauf.<br>**Wofür:** Moderne Transformer (oft ohne Bias). |
| `layout` | Ganzzahl oder Text | `'feature'` | Auswahl: feature \| 1 \| 2 \| 3 | Basis | **Was:** Datenlayout.<br>**Nutzen:** Bestimmt, welche Achse normalisiert wird.<br>**Wofür:** Sequenzen: feature; CNN: 2. |

#### `layer:activation` (Aliase: `act`) — Stufe Basis

**Was:** Aktivierungsfunktion als eigene Schicht.

**Nutzen:** Nichtlinearität.

**Wofür:** Überall.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion.<br>**Nutzen:** Siehe Registry 'activation'.<br>**Wofür:** Überall. |
| `activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |

#### `layer:embedding` — Stufe Basis

**Was:** Nachschlagetabelle ID → Vektor (Eingabe: Ganzzahlen).

**Nutzen:** Wandelt Tokens/Kategorien in Vektoren.

**Wofür:** Text, Kategorien.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `num_embeddings` | Ganzzahl | `1000` | ≥ 1 | Basis | **Was:** Größe des Vokabulars / Anzahl Kategorien.<br>**Nutzen:** Für jede ID wird ein Vektor gelernt.<br>**Wofür:** Text, Kategorien. |
| `embedding_dim` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Länge jedes Vektors.<br>**Nutzen:** Kapazität der Repräsentation.<br>**Wofür:** Text: 64–4096. |
| `padding_idx` | Ganzzahl (oder None) | `None` |  | Fortgeschritten | **Was:** ID, deren Vektor 0 bleibt und nicht trainiert wird.<br>**Nutzen:** Für Padding-Token.<br>**Wofür:** Text mit Padding. |
| `max_norm` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Vektoren mit größerer Norm werden beim Nachschlagen gekürzt.<br>**Nutzen:** Begrenzt Embedding-Größen.<br>**Wofür:** Selten. |

#### `layer:lstm` — Stufe Basis

**Was:** Rekurrente Schicht (LSTM) auf [B, T, F].

**Nutzen:** Verarbeitet Sequenzen Schritt für Schritt mit Gedächtnis.

**Wofür:** Zeitreihen, Text, Steuerung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `hidden_size` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Größe des versteckten Zustands.<br>**Nutzen:** Gedächtniskapazität.<br>**Wofür:** 64–1024. |
| `num_layers` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Gestapelte Schichten.<br>**Nutzen:** Mehr Tiefe = abstraktere Muster.<br>**Wofür:** 1–4. |
| `bidirectional` | Ja/Nein | `False` |  | Basis | **Was:** Zusätzlich rückwärts lesen.<br>**Nutzen:** Kontext von beiden Seiten; nicht für Generierung.<br>**Wofür:** Klassifikation/Tagging. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout zwischen Schichten.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Bei num_layers > 1. |
| `output` | Text | `'sequence'` | Auswahl: sequence \| last \| mean | Basis | **Was:** Was ausgegeben wird.<br>**Nutzen:** Sequenz für Weiterverarbeitung, last/mean für Klassifikation.<br>**Wofür:** Je nach Aufgabe. |

#### `layer:gru` — Stufe Basis

**Was:** Rekurrente Schicht (GRU) auf [B, T, F].

**Nutzen:** Verarbeitet Sequenzen Schritt für Schritt mit Gedächtnis.

**Wofür:** Zeitreihen, Text, Steuerung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `hidden_size` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Größe des versteckten Zustands.<br>**Nutzen:** Gedächtniskapazität.<br>**Wofür:** 64–1024. |
| `num_layers` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Gestapelte Schichten.<br>**Nutzen:** Mehr Tiefe = abstraktere Muster.<br>**Wofür:** 1–4. |
| `bidirectional` | Ja/Nein | `False` |  | Basis | **Was:** Zusätzlich rückwärts lesen.<br>**Nutzen:** Kontext von beiden Seiten; nicht für Generierung.<br>**Wofür:** Klassifikation/Tagging. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout zwischen Schichten.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Bei num_layers > 1. |
| `output` | Text | `'sequence'` | Auswahl: sequence \| last \| mean | Basis | **Was:** Was ausgegeben wird.<br>**Nutzen:** Sequenz für Weiterverarbeitung, last/mean für Klassifikation.<br>**Wofür:** Je nach Aufgabe. |

#### `layer:rnn` — Stufe Basis

**Was:** Rekurrente Schicht (RNN) auf [B, T, F].

**Nutzen:** Verarbeitet Sequenzen Schritt für Schritt mit Gedächtnis.

**Wofür:** Zeitreihen, Text, Steuerung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `hidden_size` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Größe des versteckten Zustands.<br>**Nutzen:** Gedächtniskapazität.<br>**Wofür:** 64–1024. |
| `num_layers` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Gestapelte Schichten.<br>**Nutzen:** Mehr Tiefe = abstraktere Muster.<br>**Wofür:** 1–4. |
| `bidirectional` | Ja/Nein | `False` |  | Basis | **Was:** Zusätzlich rückwärts lesen.<br>**Nutzen:** Kontext von beiden Seiten; nicht für Generierung.<br>**Wofür:** Klassifikation/Tagging. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout zwischen Schichten.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Bei num_layers > 1. |
| `output` | Text | `'sequence'` | Auswahl: sequence \| last \| mean | Basis | **Was:** Was ausgegeben wird.<br>**Nutzen:** Sequenz für Weiterverarbeitung, last/mean für Klassifikation.<br>**Wofür:** Je nach Aufgabe. |

#### `layer:self_attention` (Aliase: `attention`) — Stufe Basis

**Was:** Self-Attention auf [B, T, D].

**Nutzen:** Globale Interaktion aller Positionen.

**Wofür:** Graph-Transformer, Hybride.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `attention` | Unter-Konfiguration Attention | ⟶ AttentionConfig |  | Basis | **Was:** Einstellungen einer Multi-Head-Attention-Schicht.<br>**Nutzen:** Attention lässt jedes Element gezielt Informationen von anderen Elementen sammeln.<br>**Wofür:** Transformer-Blöcke (Self- und Cross-Attention). |
| `attention.num_heads` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `attention.num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `attention.head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `attention.kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Basis | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `attention.kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Basis | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `attention.kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `attention.kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `attention.kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `attention.kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `attention.causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `attention.window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `attention.scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `attention.qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `attention.qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `attention.out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `attention.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `attention.head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `attention.output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `attention.kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `attention.kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `attention.kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `attention.kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `attention.kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `attention.kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `causal` | Ja/Nein | `False` |  | Basis | **Was:** Kausale Maske.<br>**Nutzen:** Nur Vergangenheit sichtbar.<br>**Wofür:** Generierung. |

#### `layer:squeeze_excitation` (Aliase: `se`) — Stufe Fortgeschritten

**Was:** Squeeze-and-Excitation (Kanal-Attention).

**Nutzen:** Gewichtet Kanäle nach globalem Kontext.

**Wofür:** EfficientNet, ResNet-SE.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `reduction` | Kommazahl | `4.0` | ≥ 1.0 | Basis | **Was:** Verkleinerungsfaktor der inneren Schicht.<br>**Nutzen:** Kleiner = mehr Kapazität, mehr Parameter.<br>**Wofür:** 4–16. |
| `activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Innere Aktivierung.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/silu. |
| `activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `gate` | Komponente aus Registry 'activation' | `sigmoid` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Gate-Aktivierung.<br>**Nutzen:** Erzeugt Kanalgewichte in (0,1).<br>**Wofür:** sigmoid oder hardsigmoid. |

#### `layer:upsample` — Stufe Basis

**Was:** Interpolations-Upsampling.

**Nutzen:** Parameterfreie Vergrößerung.

**Wofür:** Decoder.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `scale_factor` | Kommazahl | `2.0` | ≥ 0.0 | Basis | **Was:** Vergrößerungsfaktor.<br>**Nutzen:** Erhöht die Auflösung.<br>**Wofür:** Decoder, U-Nets. |
| `mode` | Text | `'nearest'` | Auswahl: nearest \| linear \| bilinear \| bicubic \| trilinear | Basis | **Was:** Interpolationsverfahren.<br>**Nutzen:** nearest = blockig/schnell; bilinear = glatt.<br>**Wofür:** Bilder: bilinear oder nearest. |

#### `layer:pixel_shuffle` — Stufe Fortgeschritten

**Was:** Sub-Pixel-Upsampling.

**Nutzen:** Kanäle → Auflösung.

**Wofür:** Super-Resolution.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `upscale_factor` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Faktor (Kanäle → Auflösung).<br>**Nutzen:** Artefaktarmes Upsampling (Sub-Pixel-Faltung).<br>**Wofür:** Super-Resolution. |

#### `layer:gaussian_noise` (Aliase: `noise`) — Stufe Fortgeschritten

**Was:** Rauschen im Training.

**Nutzen:** Regularisierung, Robustheit.

**Wofür:** Kleine Daten, Sim2Real.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `std` | Kommazahl | `0.1` | ≥ 0.0 | Basis | **Was:** Standardabweichung des Rauschens.<br>**Nutzen:** Stärke der Störung.<br>**Wofür:** Robustheit. |
| `relative` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Relativ zur Streuung der Eingabe.<br>**Nutzen:** Skaleninvariant.<br>**Wofür:** Variable Skalen. |

#### `layer:scale` (Aliase: `layer_scale`) — Stufe Fortgeschritten

**Was:** Lernbare Skalierung (Skalar oder pro Kanal).

**Nutzen:** Z. B. LayerScale für tiefe Transformer.

**Wofür:** CaiT, ConvNeXt.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Kommazahl | `1.0` |  | Basis | **Was:** Startwert.<br>**Nutzen:** LayerScale nutzt kleine Werte (1e-5..1e-1).<br>**Wofür:** Tiefe Netze. |
| `per_channel` | Ja/Nein | `True` |  | Basis | **Was:** Ein Wert pro Kanal statt ein Skalar.<br>**Nutzen:** Feinere Kontrolle.<br>**Wofür:** LayerScale. |
| `layout` | Ganzzahl oder Text | `'feature'` | Auswahl: feature \| 1 \| 2 \| 3 | Basis | **Was:** Datenlayout (welche Achse 'Kanal' ist).<br>**Nutzen:** Siehe Normalisierung.<br>**Wofür:** CNN: 2. |

#### `layer:function` (Aliase: `fn`) — Stufe Basis

**Was:** Parameterfreie Tensor-Funktion als Schicht.

**Nutzen:** Kleine Umformungen.

**Wofür:** Graph-Modelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `function` | Komponente aus Registry 'function' | `identity` | Auswahl: identity \| softmax \| log_softmax \| l2_normalize \| mean \| sum \| max \| last \| first \| transpose \| abs \| square \| exp \| log1p \| … | Basis | **Was:** Anzuwendende Tensor-Funktion.<br>**Nutzen:** Parameterfreie Umformung.<br>**Wofür:** Graph-Modelle. |

#### `layer:positional_encoding` — Stufe Fortgeschritten

**Was:** Addiert eine absolute Positionskodierung.

**Nutzen:** Positionsinfo für Sequenzen.

**Wofür:** Graph-Transformer.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `positional` | Komponente aus Registry 'positional' | `sinusoidal` | Auswahl: none \| sinusoidal \| learned \| rope \| alibi \| relative_bias \| combined | Basis | **Was:** Positionskodierung, die auf [B, T, D] addiert wird.<br>**Nutzen:** Gibt Sequenzen Positionsinfo.<br>**Wofür:** Graph-Transformer. |
| `positional.max_len` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Maximale Sequenzlänge dieser Kodierung.<br>**Nutzen:** Bestimmt die Größe der Tabelle; längere Sequenzen sind danach unmöglich.<br>**Wofür:** Nur setzen, wenn abweichend von max_seq_len des Modells (None = übernehmen). |
| `positional.base` | Kommazahl | `10000.0` | ≥ 1.0 | Experte | **Was:** Basis der Wellenlängen (geometrische Reihe bis base·2π).<br>**Nutzen:** Größere Basis = langsamere Frequenzen = längere unterscheidbare Abstände.<br>**Wofür:** Sehr lange Sequenzen. |
| `positional.scale` | Kommazahl | `1.0` |  | Experte | **Was:** Faktor, mit dem die Kodierung addiert wird.<br>**Nutzen:** Balanciert Positions- gegen Inhaltsinformation.<br>**Wofür:** Experimente. |
| `max_seq_len` | Ganzzahl | `2048` | ≥ 1 | Basis | **Was:** Maximale Länge.<br>**Nutzen:** Tabellengröße.<br>**Wofür:** Länge der Daten. |

#### `layer:identity` (Aliase: `none`) — Stufe Basis

**Was:** Reicht die Eingabe unverändert durch.

**Nutzen:** Platzhalter, Verzweigungspunkte.

**Wofür:** Graph-Modelle.

#### `layer:mlp` — Stufe Basis

**Was:** Komplettes MLP als eine Schicht.

**Nutzen:** Kompakter Teilblock in freien Netzen.

**Wofür:** Graph-Modelle, Köpfe.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `input_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Eingabebreite (None = automatisch aus der Eingabe).<br>**Nutzen:** Wird im Graph automatisch bestimmt.<br>**Wofür:** Graph-Modelle. |
| `output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |

#### `layer:transformer_block` — Stufe Basis

**Was:** Vollständiger Transformer-Block auf [B, T, D].

**Nutzen:** Baustein für eigene Transformer-Hybride.

**Wofür:** Graph-Modelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `mixer` | Komponente aus Registry 'sequence_mixer' | `attention` | Auswahl: attention \| ssm \| rnn \| conv \| fourier \| none | Basis | **Was:** Mixer: tauscht Information zwischen Positionen aus (Attention, SSM, RNN, Faltung ...).<br>**Nutzen:** Bestimmt Kosten und Reichweite; pro Schicht wählbar → Hybride.<br>**Wofür:** Standard: attention. Lange Sequenzen: ssm. Hybride über layer_overrides.<br>**Tipp:** Bei 'attention' liegen die Attention-Einstellungen direkt unter mixer.* (z. B. mixer.num_heads). |
| `mixer.num_heads` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `mixer.num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `mixer.head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `mixer.kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Basis | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `mixer.kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Basis | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `mixer.kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `mixer.kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `mixer.kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `mixer.kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `mixer.causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `mixer.window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `mixer.scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `mixer.qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `mixer.qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `mixer.out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `mixer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `mixer.head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `mixer.output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `mixer.kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `mixer.kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `mixer.kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `mixer.kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `mixer.kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `mixer.kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `cross_attention` | Unter-Konfiguration Attention | ⟶ AttentionConfig |  | Fortgeschritten | **Was:** Cross-Attention der Decoder-Blöcke (nur encoder_decoder).<br>**Nutzen:** Lässt den Decoder auf die Encoder-Ausgabe schauen.<br>**Wofür:** Übersetzung, Zusammenfassung, Seq2Seq. |
| `cross_attention.num_heads` | Ganzzahl | `8` | ≥ 1 | Fortgeschritten | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `cross_attention.num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `cross_attention.head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `cross_attention.kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Fortgeschritten | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `cross_attention.kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Fortgeschritten | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `cross_attention.kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `cross_attention.kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `cross_attention.kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `cross_attention.kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `cross_attention.causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `cross_attention.window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `cross_attention.scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `cross_attention.qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `cross_attention.qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `cross_attention.out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `cross_attention.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `cross_attention.head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `cross_attention.output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `cross_attention.kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `cross_attention.kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `cross_attention.kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `cross_attention.kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `cross_attention.kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `cross_attention.kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `ffn` | Komponente aus Registry 'feedforward' | `glu` | Auswahl: mlp \| glu \| moe \| kan \| none | Basis | **Was:** Feed-Forward-Teil (mlp, glu/SwiGLU, moe, none).<br>**Nutzen:** Positionsweise Verarbeitung – hier steckt die meiste Kapazität.<br>**Wofür:** glu (SwiGLU) ist moderner Standard; moe für sehr große Kapazität. |
| `ffn.hidden_mult` | Kommazahl | `2.6666666666666665` | ≥ 0.01 | Basis | **Was:** Breite relativ zu d_model (GLU hat 3 Matrizen → Standard 8/3 hält die Parameter wie MLP mit 4).<br>**Nutzen:** Mehr Breite = mehr Kapazität.<br>**Wofür:** LLaMA: 8/3 (aufgerundet auf multiple_of). |
| `ffn.hidden_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Exakte Breite der versteckten Schicht (überschreibt hidden_mult).<br>**Nutzen:** Für exakte Nachbauten bekannter Architekturen.<br>**Wofür:** Gewichte-Import, exakte Parameterzahlen. |
| `ffn.multiple_of` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Versteckte Breite auf ein Vielfaches davon aufrunden.<br>**Nutzen:** Vielfache von 64/256 rechnen auf GPUs schneller (Tensor Cores).<br>**Wofür:** Geschwindigkeitsoptimierung (LLaMA: 256). |
| `ffn.activation` | Komponente aus Registry 'activation' | `silu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Gate-Aktivierung: silu → SwiGLU, gelu → GeGLU, sigmoid → GLU, relu → ReGLU, identity → Bilinear.<br>**Nutzen:** Das Gate entscheidet elementweise, welche Information durchgelassen wird.<br>**Wofür:** SwiGLU ist Standard moderner LLMs. |
| `ffn.bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den linearen Schichten.<br>**Nutzen:** Moderne LLMs verzichten darauf.<br>**Wofür:** Kompatibilität. |
| `ffn.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach der Aktivierung (innerhalb des FFN).<br>**Nutzen:** Regularisiert die versteckte Schicht.<br>**Wofür:** Kleine Datensätze. |
| `norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung in den Residual-Zweigen.<br>**Nutzen:** Stabilisiert das Training.<br>**Wofür:** rmsnorm (modern), layernorm (klassisch). |
| `norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `norm.learnable` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `norm_position` | Text | `'pre'` | Auswahl: pre \| post \| sandwich \| none | Fortgeschritten | **Was:** Position der Normalisierung relativ zur Residualverbindung.<br>**Nutzen:** Großer Einfluss auf Trainingsstabilität und Endqualität.<br>**Wofür:** pre für tiefe Modelle; post für kleine Modelle mit Warmup. |
| `parallel` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Mixer und FFN parallel statt nacheinander (x + Mixer(n(x)) + FFN(n(x))).<br>**Nutzen:** ~15 % schneller bei großen Modellen, minimal andere Qualität.<br>**Wofür:** GPT-J, PaLM. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Ausgabe jedes Zweigs (Residual-Dropout).<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Datensätze: 0.1; große LLMs: 0. |
| `drop_path` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Stochastic Depth: Wahrscheinlichkeit, einen ganzen Zweig auszulassen.<br>**Nutzen:** Starke Regularisierung, besonders für Vision-Transformer.<br>**Wofür:** ViT/DeiT: 0.1–0.3 (mit linearem Anstieg über die Tiefe). |
| `layer_scale_init` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** LayerScale: lernbarer Faktor pro Kanal auf jedem Zweig (Startwert; None = aus).<br>**Nutzen:** Kleiner Start (1e-5…1e-1) lässt tiefe Netze zunächst wie flache trainieren.<br>**Wofür:** Sehr tiefe Transformer (CaiT, DINOv2). |
| `branch_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf jeder Zweig-Ausgabe.<br>**Nutzen:** Dämpft Residual-Beiträge in tiefen Netzen.<br>**Wofür:** Experimente; z. B. 1/√(2N). |
| `skip_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf dem Skip-Pfad (DeepNet: α > 1).<br>**Nutzen:** Stärkerer Skip stabilisiert Post-LN-Netze mit hunderten Schichten.<br>**Wofür:** DeepNorm-Experimente. |
| `causal` | Ja/Nein | `False` |  | Basis | **Was:** Kausale Maske.<br>**Nutzen:** Nur Vergangenheit sichtbar.<br>**Wofür:** Generative Graphen. |
| `positional` | Komponente aus Registry 'positional' | `none` | Auswahl: none \| sinusoidal \| learned \| rope \| alibi \| relative_bias \| combined | Basis | **Was:** Positionskodierung innerhalb des Blocks (rope/alibi wirken hier).<br>**Nutzen:** Positionsinfo.<br>**Wofür:** Graph-Transformer. |


### Registry `merge`

<a id="registry-merge"></a>

**Zusammenführungen** – Verfahren, um mehrere Tensoren zu einem zu vereinen. Ermöglichen Verzweigungen, Residualverbindungen und Multi-Input-Netze.

Fabrik-Signatur: `factory(args, in_shapes) – Liste von Formen ohne Batch-Dimension`

#### `merge:concat` (Aliase: `cat`) — Stufe Basis

**Was:** Verketten entlang einer Achse.

**Nutzen:** Behält alle Informationen getrennt bei.

**Wofür:** Fusion.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse, entlang der verkettet wird (-1 = Features, 1 = Kanäle bei CNNs).<br>**Nutzen:** Bestimmt, wie Informationen nebeneinander gelegt werden.<br>**Wofür:** Fusion, DenseNet, U-Net. |

#### `merge:add` (Aliase: `sum`, `residual`) — Stufe Basis

**Was:** Elementweise Summe.

**Nutzen:** Residualverbindungen; Formen müssen gleich sein.

**Wofür:** ResNet-artige Skips.

#### `merge:mul` (Aliase: `product`) — Stufe Basis

**Was:** Elementweises Produkt.

**Nutzen:** Multiplikative Interaktion / Gating.

**Wofür:** Attention-artige Gates.

#### `merge:mean` (Aliase: `average`) — Stufe Basis

**Was:** Elementweiser Mittelwert.

**Nutzen:** Skalenneutral bei vielen Eingängen.

**Wofür:** Ensembles.

#### `merge:max` — Stufe Basis

**Was:** Elementweises Maximum.

**Nutzen:** Stärkstes Signal gewinnt.

**Wofür:** Robuste Fusion.

#### `merge:stack` — Stufe Basis

**Was:** Stapeln entlang neuer Achse.

**Nutzen:** Behält Eingänge als eigene Achse.

**Wofür:** Spätere Attention über Quellen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse, entlang der verkettet wird (-1 = Features, 1 = Kanäle bei CNNs).<br>**Nutzen:** Bestimmt, wie Informationen nebeneinander gelegt werden.<br>**Wofür:** Fusion, DenseNet, U-Net. |

#### `merge:weighted_sum` — Stufe Fortgeschritten

**Was:** Lernbar gewichtete Summe.

**Nutzen:** Netz lernt die Wichtigkeit der Eingänge.

**Wofür:** Fusion.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `softmax` | Ja/Nein | `True` |  | Basis | **Was:** Gewichte per Softmax normieren.<br>**Nutzen:** Gewichte summieren zu 1.<br>**Wofür:** Standard. |

#### `merge:gated` — Stufe Fortgeschritten

**Was:** Lernbares Gate zwischen genau zwei Eingängen.

**Nutzen:** Elementweise weiche Auswahl.

**Wofür:** Highway-artige Verbindungen.

#### `merge:first` — Stufe Experte

**Was:** Nur den ersten Eingang verwenden.

**Nutzen:** Für Ablationen (Zweig abschalten ohne Umbau).

**Wofür:** Experimente.


### Registry `function`

<a id="registry-function"></a>

**Tensor-Funktionen** – Parameterfreie Tensor-Operationen. Kleine Umformungen ohne eigene Gewichte (z. B. Softmax, Mittelwert, Transponieren).

Fabrik-Signatur: `factory(args) -> callable(x)`

#### `function:identity` — Stufe Basis

**Was:** Unverändert.

**Nutzen:** Platzhalter.

**Wofür:** Überall.

#### `function:softmax` — Stufe Basis

**Was:** Softmax entlang einer Achse.

**Nutzen:** Wahrscheinlichkeitsverteilung.

**Wofür:** Ausgaben, Gewichte.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:log_softmax` — Stufe Basis

**Was:** Log-Softmax.

**Nutzen:** Stabile Log-Wahrscheinlichkeiten.

**Wofür:** NLL-Verluste.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:l2_normalize` (Aliase: `normalize`) — Stufe Basis

**Was:** Auf Länge 1 normieren.

**Nutzen:** Für Kosinus-Ähnlichkeiten.

**Wofür:** Embeddings, kontrastives Lernen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:mean` — Stufe Basis

**Was:** Mittelwert über eine Achse (entfernt sie).

**Nutzen:** Pooling über Zeit/Tokens.

**Wofür:** Sequenz → Vektor.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:sum` — Stufe Basis

**Was:** Summe über eine Achse.

**Nutzen:** Pooling.

**Wofür:** Mengen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:max` — Stufe Basis

**Was:** Maximum über eine Achse.

**Nutzen:** Max-Pooling.

**Wofür:** Mengen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:last` — Stufe Basis

**Was:** Letztes Element entlang einer Achse.

**Nutzen:** Letzter Zeitschritt.

**Wofür:** Sequenz → Vektor.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:first` — Stufe Basis

**Was:** Erstes Element entlang einer Achse.

**Nutzen:** CLS-Token.

**Wofür:** Sequenz → Vektor.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse der Operation.<br>**Nutzen:** Bestimmt, worüber gerechnet wird.<br>**Wofür:** -1 = letzte Achse. |

#### `function:transpose` — Stufe Basis

**Was:** Zwei Achsen tauschen.

**Nutzen:** Layout-Wechsel.

**Wofür:** [B, T, C] ↔ [B, C, T].

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dim0` | Ganzzahl | `1` |  | Basis | **Was:** Erste Achse.<br>**Nutzen:** Wird mit dim1 getauscht.<br>**Wofür:** 1 und 2 bei [B, T, C]. |
| `dim1` | Ganzzahl | `2` |  | Basis | **Was:** Zweite Achse.<br>**Nutzen:** Wird mit dim0 getauscht.<br>**Wofür:** 1 und 2 bei [B, T, C]. |

#### `function:abs` — Stufe Basis

**Was:** Betrag.

**Nutzen:** Vorzeichen entfernen.

**Wofür:** Spezialfälle.

#### `function:square` — Stufe Basis

**Was:** Quadrat.

**Nutzen:** Energie/Leistung.

**Wofür:** Signale.

#### `function:exp` — Stufe Fortgeschritten

**Was:** Exponentialfunktion.

**Nutzen:** Positive Werte.

**Wofür:** Raten.

#### `function:log1p` — Stufe Fortgeschritten

**Was:** log(1+x).

**Nutzen:** Stauchung großer Werte.

**Wofür:** Zähldaten.

#### `function:clamp` — Stufe Basis

**Was:** Werte begrenzen.

**Nutzen:** Harte Grenzen.

**Wofür:** Stabilisierung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `min` | Kommazahl (oder None) | `None` |  | Basis | **Was:** Untergrenze.<br>**Nutzen:** Kleinere Werte werden angehoben.<br>**Wofür:** Begrenzung. |
| `max` | Kommazahl (oder None) | `None` |  | Basis | **Was:** Obergrenze.<br>**Nutzen:** Größere Werte werden gekappt.<br>**Wofür:** Begrenzung. |

#### `function:scale` — Stufe Basis

**Was:** Mit Konstante multiplizieren.

**Nutzen:** Feste Skalierung.

**Wofür:** Temperatur, Normierung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `value` | Kommazahl | `1.0` |  | Basis | **Was:** Faktor.<br>**Nutzen:** Multipliziert alle Elemente.<br>**Wofür:** Skalierung. |

#### `function:detach` — Stufe Experte

**Was:** Gradientenfluss stoppen.

**Nutzen:** Zweig wird nicht durch diesen Pfad trainiert.

**Wofür:** Stop-Gradient-Tricks (BYOL, SimSiam).

#### `function:sign_ste` — Stufe Experimentell

**Was:** Vorzeichen mit Straight-Through-Gradient.

**Nutzen:** Binäre Aktivierungen trainierbar machen.

**Wofür:** Binäre Netze, Hashing.


### Registry `sequence_mixer`

<a id="registry-sequence-mixer"></a>

**Sequenz-Mixer** – Teil eines Blocks, der Informationen zwischen Sequenzpositionen austauscht. Bestimmt Kosten (quadratisch vs. linear), Reichweite und induktive Annahmen.

Fabrik-Signatur: `factory(args, d_model) -> forward(x, key_padding_mask, positional, positions, causal)`

#### `sequence_mixer:attention` (Aliase: `self_attention`) — Stufe Basis

**Was:** Multi-Head-Self-Attention.

**Nutzen:** Globale, inhaltsabhängige Interaktion aller Positionen.

**Wofür:** Standard.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `num_heads` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Basis | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Basis | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |

#### `sequence_mixer:ssm` (Aliase: `mamba`) — Stufe Fortgeschritten

**Was:** Selektives Zustandsraummodell (Mamba-artig).

**Nutzen:** Lineare Kosten in der Sequenzlänge.

**Wofür:** Lange Sequenzen, Hybride (Jamba).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `d_state` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Größe des inneren Zustands pro Kanal (N).<br>**Nutzen:** Mehr Zustand = mehr Gedächtniskapazität, mehr Rechenaufwand.<br>**Wofür:** Mamba: 16. |
| `expand` | Kommazahl | `2.0` | ≥ 0.25 | Basis | **Was:** Aufweitung der inneren Breite (E = expand · d_model).<br>**Nutzen:** Kapazität des Blocks.<br>**Wofür:** Mamba: 2. |
| `d_conv` | Ganzzahl | `4` | ≥ 1 | Fortgeschritten | **Was:** Kernelgröße der kurzen kausalen Faltung vor dem Scan.<br>**Nutzen:** Erfasst lokale Muster, bevor der Zustand aktualisiert wird.<br>**Wofür:** Mamba: 4. |
| `dt_rank` | Ganzzahl oder Text | `'auto'` |  | Experte | **Was:** Rang der Δ-Projektion ('auto' = ceil(d_model/16)).<br>**Nutzen:** Steuert, wie fein die Schrittweite Δ von der Eingabe abhängen kann.<br>**Wofür:** Selten zu ändern. |
| `dt_min` | Kommazahl | `0.001` | ≥ 1e-06 | Experte | **Was:** Kleinste initiale Schrittweite Δ.<br>**Nutzen:** Kleine Δ = langes Gedächtnis.<br>**Wofür:** Mamba: 0.001. |
| `dt_max` | Kommazahl | `0.1` | ≥ 1e-06 | Experte | **Was:** Größte initiale Schrittweite Δ.<br>**Nutzen:** Große Δ = Fokus auf aktuelle Eingabe.<br>**Wofür:** Mamba: 0.1. |
| `a_init` | Text | `'s4d_real'` | Auswahl: s4d_real \| s4d_lin \| random | Experte | **Was:** Initialisierung der Zustandsmatrix A.<br>**Nutzen:** Bestimmt Start-Gedächtnis-Zeitskalen.<br>**Wofür:** Forschung. |
| `bidirectional` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Zusätzlicher Rückwärts-Scan (wird bei kausalen Modellen ignoriert).<br>**Nutzen:** Kontext von beiden Seiten für Encoder-Aufgaben.<br>**Wofür:** Klassifikation, Tagging. |
| `conv_bias` | Ja/Nein | `True` |  | Experte | **Was:** Bias der kurzen Faltung.<br>**Nutzen:** Verschiebung.<br>**Wofür:** Standard. |
| `bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias der Ein-/Ausgangsprojektion.<br>**Nutzen:** Verschiebung.<br>**Wofür:** Mamba: aus. |
| `scan_mode` | Text | `'auto'` | Auswahl: auto \| parallel \| chunked \| sequential \| mamba_ssm | Basis | **Was:** Algorithmus für den Zustands-Scan h_t = exp(Δ·A)·h_{t-1} + Δ·B·x.<br>**Nutzen:** Der parallele/chunked Scan braucht nur log₂(T) Runden statt T Schleifenschritte – Training oft 10–40× schneller als die Python-Schleife.<br>**Wofür:** auto ist fast immer richtig (CPU: fusionierte Schleife, GPU: chunked); mamba_ssm auf NVIDIA-GPUs mit installiertem Paket. |
| `scan_chunk_size` | Ganzzahl | `32` | ≥ 1 | Fortgeschritten | **Was:** Blocklänge des chunked-Scans.<br>**Nutzen:** Größer = weniger sequenzielle Überträge, aber mehr Zwischenspeicher pro Runde.<br>**Wofür:** CPU: 16–32; GPU: 64–256. |
| `scan_backward` | Text | `'fused'` | Auswahl: fused \| scan \| autograd | Experte | **Was:** Wie der Gradient des Scans berechnet wird.<br>**Nutzen:** 'fused' senkt den Aktivierungsspeicher des Scans etwa um den Faktor d_state – wie der offizielle Mamba-Kernel, aber in reinem PyTorch (CPU, GPU, MPS).<br>**Wofür:** Lange Sequenzen / große Batches: fused. Kurze Sequenzen mit viel Speicher: scan. |

#### `sequence_mixer:rnn` — Stufe Fortgeschritten

**Was:** Rekurrente Schicht als Mixer.

**Nutzen:** Starke Reihenfolge-Annahme, Streaming-fähig.

**Wofür:** Hybride, kleine Daten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `cell` | Text | `'gru'` | Auswahl: gru \| lstm \| rnn \| mingru | Basis | **Was:** Rekurrente Zelle.<br>**Nutzen:** GRU: schlank; LSTM: mehr Gedächtnissteuerung; minGRU: parallel trainierbar.<br>**Wofür:** Hybride, Streaming. |
| `bidirectional` | Ja/Nein | `True` |  | Basis | **Was:** Beide Richtungen (nur ohne Kausalität).<br>**Nutzen:** Kontext von beiden Seiten.<br>**Wofür:** Encoder. |

#### `sequence_mixer:conv` — Stufe Fortgeschritten

**Was:** Tiefenfaltung über die Zeit.

**Nutzen:** Billig, lokal, kausal möglich.

**Wofür:** Audio, lokale Muster, Hybride.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `kernel_size` | Ganzzahl | `7` | ≥ 1 | Basis | **Was:** Fenstergröße der Zeitfaltung.<br>**Nutzen:** Reichweite pro Schicht.<br>**Wofür:** Lokale Muster: 3–15. |
| `activation` | Ja/Nein | `True` |  | Basis | **Was:** SiLU nach der Faltung.<br>**Nutzen:** Nichtlinearität im Mixer.<br>**Wofür:** Standard. |

#### `sequence_mixer:fourier` (Aliase: `fnet`) — Stufe Experimentell

**Was:** Fourier-Transformation als Mixer (FNet).

**Nutzen:** Parameterfrei, sehr schnell; etwas schwächer als Attention.

**Wofür:** Schnelle Encoder, Experimente.

#### `sequence_mixer:none` (Aliase: `keiner`) — Stufe Experimentell

**Was:** Kein Mixer – Positionen bleiben getrennt.

**Nutzen:** Ablationen; reine positionsweise Netze.

**Wofür:** Forschung.


### Registry `positional`

<a id="registry-positional"></a>

**Positionskodierungen** – Verfahren, mit denen ein Modell die Position von Elementen in einer Sequenz kennt. Ohne sie ist Attention reihenfolgeblind.

Fabrik-Signatur: `factory(args, d_model, head_dim, num_heads, max_seq_len)`

#### `positional:none` (Aliase: `nope`, `keine`) — Stufe Basis

**Was:** Keine Positionskodierung (NoPE).

**Nutzen:** Kausale Decoder können Positionen teils implizit aus der Maske lernen; generalisiert oft gut auf Länge.

**Wofür:** Forschung, Längen-Generalisierung, Mengen-Daten (Reihenfolge egal).

#### `positional:sinusoidal` (Aliase: `sin`) — Stufe Basis

**Was:** Feste Sinus/Kosinus-Kodierung (Original-Transformer).

**Nutzen:** Keine Parameter, deterministisch.

**Wofür:** Klassische Encoder-Decoder, Zeitreihen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `max_len` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Maximale Sequenzlänge dieser Kodierung.<br>**Nutzen:** Bestimmt die Größe der Tabelle; längere Sequenzen sind danach unmöglich.<br>**Wofür:** Nur setzen, wenn abweichend von max_seq_len des Modells (None = übernehmen). |
| `base` | Kommazahl | `10000.0` | ≥ 1.0 | Experte | **Was:** Basis der Wellenlängen (geometrische Reihe bis base·2π).<br>**Nutzen:** Größere Basis = langsamere Frequenzen = längere unterscheidbare Abstände.<br>**Wofür:** Sehr lange Sequenzen. |
| `scale` | Kommazahl | `1.0` |  | Experte | **Was:** Faktor, mit dem die Kodierung addiert wird.<br>**Nutzen:** Balanciert Positions- gegen Inhaltsinformation.<br>**Wofür:** Experimente. |

#### `positional:learned` (Aliase: `absolute`) — Stufe Basis

**Was:** Gelernte absolute Positions-Embeddings.

**Nutzen:** Flexibel, aber keine Extrapolation über max_seq_len hinaus.

**Wofür:** GPT-2, BERT, ViT.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `max_len` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Maximale Sequenzlänge dieser Kodierung.<br>**Nutzen:** Bestimmt die Größe der Tabelle; längere Sequenzen sind danach unmöglich.<br>**Wofür:** Nur setzen, wenn abweichend von max_seq_len des Modells (None = übernehmen). |
| `init_std` | Kommazahl | `0.02` | ≥ 0.0 | Experte | **Was:** Startstreuung der gelernten Positionsvektoren.<br>**Nutzen:** Kleiner Start stört Token-Embeddings wenig.<br>**Wofür:** GPT-2/BERT: 0.02. |

#### `positional:rope` (Aliase: `rotary`) — Stufe Basis

**Was:** Rotary Position Embedding (RoPE).

**Nutzen:** Relative Positionen, gute Qualität, Kontext erweiterbar.

**Wofür:** Standard moderner LLMs (LLaMA, Mistral, Qwen).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `theta` | Kommazahl | `10000.0` | ≥ 1.0 | Basis | **Was:** Basisfrequenz θ der Rotationen.<br>**Nutzen:** Größeres θ → längere Kontexte unterscheidbar (LLaMA-3: 500000).<br>**Wofür:** Lange Kontexte → θ erhöhen. |
| `fraction` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil der Kopf-Dimensionen, die rotiert werden ('partial rotary').<br>**Nutzen:** Rest bleibt positionsfrei – manche Modelle (GPT-NeoX: 0.25) profitieren davon.<br>**Wofür:** Feintuning von Positions- vs. Inhaltsanteil. |
| `scaling` | Text | `'none'` | Auswahl: none \| linear \| ntk \| yarn | Fortgeschritten | **Was:** Verfahren zur Kontextverlängerung.<br>**Nutzen:** Erlaubt Sequenzen länger als im Training.<br>**Wofür:** Kontext-Erweiterung vortrainierter Modelle. |
| `factor` | Kommazahl | `1.0` | ≥ 1.0 | Fortgeschritten | **Was:** Erweiterungsfaktor des Kontexts (z. B. 4 = vierfache Länge).<br>**Nutzen:** Stärke der Skalierung.<br>**Wofür:** Nur mit scaling ≠ none. |
| `original_context` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** Kontextlänge, mit der ursprünglich trainiert wurde (nur YaRN; None = max_seq_len / factor).<br>**Nutzen:** YaRN entscheidet anhand dieser Länge, welche Frequenzen interpoliert werden.<br>**Wofür:** Vortrainiertes Modell mit 4096 Tokens auf 16k erweitern: 4096, factor 4. |
| `yarn_beta_fast` | Kommazahl | `32.0` | ≥ 0.0 | Experte | **Was:** YaRN β_fast: ab so vielen Umdrehungen im Originalkontext bleibt eine Frequenz unverändert.<br>**Nutzen:** Schützt die lokale Positionsauflösung.<br>**Wofür:** 32 (Paper). |
| `yarn_beta_slow` | Kommazahl | `1.0` | ≥ 0.0 | Experte | **Was:** YaRN β_slow: unter so vielen Umdrehungen wird voll interpoliert.<br>**Nutzen:** Niedrige Frequenzen.<br>**Wofür:** 1 (Paper). |
| `interleaved` | Ja/Nein | `False` |  | Experte | **Was:** Paare benachbarter Dimensionen rotieren (GPT-J) statt Hälften (GPT-NeoX/LLaMA).<br>**Nutzen:** Nur für Kompatibilität mit fremden Gewichten relevant.<br>**Wofür:** Gewichte-Import. |

#### `positional:alibi` — Stufe Basis

**Was:** ALiBi: linearer Distanz-Strafterm pro Kopf.

**Nutzen:** Keine Parameter, extrapoliert gut auf längere Sequenzen.

**Wofür:** Modelle, die länger generieren sollen als trainiert (BLOOM, MPT).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `learnable` | Ja/Nein | `False` |  | Experimentell | **Was:** Steigungen lernbar machen.<br>**Nutzen:** Netz wählt selbst, wie stark Distanz bestraft wird.<br>**Wofür:** Experimente. |
| `slope_scale` | Kommazahl | `1.0` | ≥ 0.0 | Experte | **Was:** Faktor auf alle Steigungen.<br>**Nutzen:** Größer = stärkerer Fokus auf nahe Elemente.<br>**Wofür:** Experimente. |

#### `positional:relative_bias` (Aliase: `t5`) — Stufe Fortgeschritten

**Was:** Gelernter relativer Positions-Bias (T5).

**Nutzen:** Lernt, wie wichtig Distanzen sind; gebucketted für lange Distanzen.

**Wofür:** Encoder-Decoder (T5), Encoder.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `num_buckets` | Ganzzahl | `32` | ≥ 4 | Basis | **Was:** Anzahl Distanz-Klassen.<br>**Nutzen:** Mehr Buckets = feinere Unterscheidung von Distanzen.<br>**Wofür:** T5: 32. |
| `max_distance` | Ganzzahl | `128` | ≥ 2 | Basis | **Was:** Ab dieser Distanz teilen sich alle Abstände einen Bucket.<br>**Nutzen:** Begrenzt Parameter, generalisiert auf Länge.<br>**Wofür:** T5: 128. |
| `bidirectional` | Ja/Nein | `True` |  | Basis | **Was:** Vor/zurück unterscheiden (nur ohne Kausalmaske).<br>**Nutzen:** Encoder profitieren von Richtungsinfo.<br>**Wofür:** Encoder. |

#### `positional:combined` — Stufe Experimentell

**Was:** Kombination mehrerer Positionskodierungen.

**Nutzen:** Maximale Freiheit: z. B. absolute + rotierende Kodierung gleichzeitig.

**Wofür:** Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `parts` | Liste von Komponenten aus Registry 'positional' | 2 Elemente | Elemente: none \| sinusoidal \| learned \| rope \| alibi \| relative_bias \| combined | Basis | **Was:** Liste der zu kombinierenden Positionskodierungen.<br>**Nutzen:** Experimentell: verschiedene Positionssignale gleichzeitig.<br>**Wofür:** Forschung / Ablationen. |
| `parts.0` | beliebig | `positional:learned()` |  | Basis | **Was:** Gelernte absolute Positions-Embeddings.<br>**Nutzen:** Flexibel, aber keine Extrapolation über max_seq_len hinaus.<br>**Wofür:** GPT-2, BERT, ViT. |
| `parts.0.max_len` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Maximale Sequenzlänge dieser Kodierung.<br>**Nutzen:** Bestimmt die Größe der Tabelle; längere Sequenzen sind danach unmöglich.<br>**Wofür:** Nur setzen, wenn abweichend von max_seq_len des Modells (None = übernehmen). |
| `parts.0.init_std` | Kommazahl | `0.02` | ≥ 0.0 | Experte | **Was:** Startstreuung der gelernten Positionsvektoren.<br>**Nutzen:** Kleiner Start stört Token-Embeddings wenig.<br>**Wofür:** GPT-2/BERT: 0.02. |
| `parts.1` | beliebig | `positional:rope()` |  | Basis | **Was:** Rotary Position Embedding (RoPE).<br>**Nutzen:** Relative Positionen, gute Qualität, Kontext erweiterbar.<br>**Wofür:** Standard moderner LLMs (LLaMA, Mistral, Qwen). |
| `parts.1.theta` | Kommazahl | `10000.0` | ≥ 1.0 | Basis | **Was:** Basisfrequenz θ der Rotationen.<br>**Nutzen:** Größeres θ → längere Kontexte unterscheidbar (LLaMA-3: 500000).<br>**Wofür:** Lange Kontexte → θ erhöhen. |
| `parts.1.fraction` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil der Kopf-Dimensionen, die rotiert werden ('partial rotary').<br>**Nutzen:** Rest bleibt positionsfrei – manche Modelle (GPT-NeoX: 0.25) profitieren davon.<br>**Wofür:** Feintuning von Positions- vs. Inhaltsanteil. |
| `parts.1.scaling` | Text | `'none'` | Auswahl: none \| linear \| ntk \| yarn | Fortgeschritten | **Was:** Verfahren zur Kontextverlängerung.<br>**Nutzen:** Erlaubt Sequenzen länger als im Training.<br>**Wofür:** Kontext-Erweiterung vortrainierter Modelle. |
| `parts.1.factor` | Kommazahl | `1.0` | ≥ 1.0 | Fortgeschritten | **Was:** Erweiterungsfaktor des Kontexts (z. B. 4 = vierfache Länge).<br>**Nutzen:** Stärke der Skalierung.<br>**Wofür:** Nur mit scaling ≠ none. |
| `parts.1.original_context` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** Kontextlänge, mit der ursprünglich trainiert wurde (nur YaRN; None = max_seq_len / factor).<br>**Nutzen:** YaRN entscheidet anhand dieser Länge, welche Frequenzen interpoliert werden.<br>**Wofür:** Vortrainiertes Modell mit 4096 Tokens auf 16k erweitern: 4096, factor 4. |
| `parts.1.yarn_beta_fast` | Kommazahl | `32.0` | ≥ 0.0 | Experte | **Was:** YaRN β_fast: ab so vielen Umdrehungen im Originalkontext bleibt eine Frequenz unverändert.<br>**Nutzen:** Schützt die lokale Positionsauflösung.<br>**Wofür:** 32 (Paper). |
| `parts.1.yarn_beta_slow` | Kommazahl | `1.0` | ≥ 0.0 | Experte | **Was:** YaRN β_slow: unter so vielen Umdrehungen wird voll interpoliert.<br>**Nutzen:** Niedrige Frequenzen.<br>**Wofür:** 1 (Paper). |
| `parts.1.interleaved` | Ja/Nein | `False` |  | Experte | **Was:** Paare benachbarter Dimensionen rotieren (GPT-J) statt Hälften (GPT-NeoX/LLaMA).<br>**Nutzen:** Nur für Kompatibilität mit fremden Gewichten relevant.<br>**Wofür:** Gewichte-Import. |


### Registry `model`

<a id="registry-model"></a>

**Modelle** – Vollständige, vorgefertigte Modellarchitekturen. Jede ist bis ins Detail konfigurierbar und mit allen anderen kombinierbar.

Fabrik-Signatur: `factory(args) -> BaseModel`

#### `model:autoencoder` (Aliase: `ae`, `vae`) — Stufe Basis

**Was:** Autoencoder-Familie (AE, VAE, Denoising, Sparse, VQ-VAE).

**Nutzen:** Unüberwachtes Lernen kompakter Repräsentationen.

**Wofür:** Anomalieerkennung, Kompression, Generierung, Vortraining.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `variant` | Text | `'ae'` | Auswahl: ae \| vae \| denoising \| sparse \| vq | Basis | **Was:** Autoencoder-Variante.<br>**Nutzen:** Bestimmt Latentraum-Eigenschaften und Regularisierung.<br>**Wofür:** Anomalien: ae/denoising; Generierung: vae/vq. |
| `backbone` | Text | `'mlp'` | Auswahl: mlp \| conv | Basis | **Was:** Netztyp von Encoder/Decoder.<br>**Nutzen:** conv nutzt räumliche Struktur.<br>**Wofür:** Bilder: conv. |
| `input_dim` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Eingabebreite (mlp).<br>**Nutzen:** Größe der Vektoren.<br>**Wofür:** Anzahl Features. |
| `in_channels` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Kanäle (conv).<br>**Nutzen:** Graustufen 1, RGB 3.<br>**Wofür:** Bilder. |
| `image_size` | Liste[Ganzzahl] | `[28, 28]` | ≥ 1 · Länge 2 | Basis | **Was:** Bildgröße [H, W] (conv).<br>**Nutzen:** Muss durch 2^(Anzahl Kanäle-Stufen) teilbar sein.<br>**Wofür:** Bilder. |
| `hidden_dims` | Liste[Ganzzahl] | `[128, 64]` | ≥ 1 | Basis | **Was:** Breiten (mlp) bzw. Kanalzahlen (conv) des Encoders; der Decoder spiegelt sie.<br>**Nutzen:** Kapazität von Encoder/Decoder.<br>**Wofür:** mlp: [256, 128]; conv: [32, 64]. |
| `decoder_dims` | Liste[Ganzzahl] (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Eigene Decoder-Breiten (None = gespiegelter Encoder).<br>**Nutzen:** Asymmetrische Autoencoder.<br>**Wofür:** Experimente. |
| `latent_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Größe des Latentraums (Engpass).<br>**Nutzen:** Kleiner = stärkere Kompression, aber mehr Rekonstruktionsfehler.<br>**Wofür:** 2–3 zur Visualisierung, 8–128 für Merkmale. |
| `activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/silu/gelu. |
| `activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung.<br>**Nutzen:** Stabilität.<br>**Wofür:** Tiefe AEs. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Daten. |
| `output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Rekonstruktion.<br>**Nutzen:** sigmoid für Daten in [0, 1] (Bilder), tanh für [-1, 1], identity für unbeschränkte Werte.<br>**Wofür:** Passend zum Wertebereich der Daten wählen. |
| `beta` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Gewicht der KL-Divergenz (β-VAE, nur vae).<br>**Nutzen:** β > 1 erzwingt entwirrte (disentangled) Latentfaktoren, β < 1 bessere Rekonstruktion.<br>**Wofür:** Disentanglement: 4–10; scharfe Rekonstruktion: 0.1–1. |
| `free_bits` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Mindest-KL pro Latentdimension ('free bits'); darunter keine Strafe.<br>**Nutzen:** Verhindert 'Posterior Collapse' (Latentraum wird ignoriert).<br>**Wofür:** VAEs mit starken Decodern (0.1–0.5). |
| `corruption` | Text | `'gaussian'` | Auswahl: gaussian \| masking \| salt_pepper | Basis | **Was:** Art der Störung (nur denoising).<br>**Nutzen:** Was das Modell 'reparieren' lernt.<br>**Wofür:** Daten-abhängig. |
| `corruption_level` | Kommazahl | `0.2` | ≥ 0.0 | Basis | **Was:** Stärke der Störung (Std bzw. Anteil).<br>**Nutzen:** Stärker = robustere Merkmale, schwierigeres Training.<br>**Wofür:** 0.1–0.5. |
| `sparsity_type` | Text | `'l1'` | Auswahl: l1 \| kl | Basis | **Was:** Sparsity-Strafe (nur sparse).<br>**Nutzen:** Art der Erzwingung.<br>**Wofür:** l1 ist einfach. |
| `sparsity_weight` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Gewicht der Sparsity-Strafe.<br>**Nutzen:** Stärke der Erzwingung.<br>**Wofür:** 1e-4–1e-2. |
| `sparsity_target` | Kommazahl | `0.05` | Bereich 0.0001 … 0.9999 | Fortgeschritten | **Was:** Ziel-Aktivierungsrate (kl).<br>**Nutzen:** Gewünschter Anteil aktiver Einheiten.<br>**Wofür:** 0.01–0.1. |
| `codebook_size` | Ganzzahl | `64` | ≥ 2 | Basis | **Was:** Anzahl Codebuch-Vektoren (nur vq).<br>**Nutzen:** Größe des diskreten Vokabulars.<br>**Wofür:** 64–1024. |
| `commitment_cost` | Kommazahl | `0.25` | ≥ 0.0 | Basis | **Was:** Gewicht des Commitment-Verlusts (nur vq).<br>**Nutzen:** Bindet Encoder-Ausgaben an das Codebuch.<br>**Wofür:** VQ-VAE: 0.25. |

#### `model:cnn` (Aliase: `convnet`, `faltungsnetz`) — Stufe Basis

**Was:** Faltungsnetz (1D/2D/3D) mit frei konfigurierbaren Stufen und Blocktypen.

**Nutzen:** Effizient und dateneffizient für räumlich strukturierte Daten.

**Wofür:** Bilder, Audio, Sensorsignale, Volumendaten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `dims` | Ganzzahl | `2` | Auswahl: 1 \| 2 \| 3 | Basis | **Was:** Räumliche Dimensionalität.<br>**Nutzen:** Bestimmt Faltungstyp und Eingabeform [B, C, *raum].<br>**Wofür:** Bilder: 2. |
| `in_channels` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Eingabekanäle.<br>**Nutzen:** RGB = 3, Graustufen = 1, Mehrkanal-Sensoren = n.<br>**Wofür:** Muss zu den Daten passen. |
| `input_size` | Liste[Ganzzahl] | `[32, 32]` | ≥ 1 | Basis | **Was:** Räumliche Eingabegröße (für example_input und 'flatten'-Köpfe).<br>**Nutzen:** Wird für Dummy-Eingaben/Speicherschätzung und den Flatten-Kopf benötigt.<br>**Wofür:** Länge = dims, z. B. [32, 32] oder [16000] für 1D. |
| `stem` | Unter-Konfiguration Stem | ⟶ StemConfig |  | Basis | **Was:** Unter-Konfiguration 'Stem'. |
| `stem.channels` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Kanäle des Stems.<br>**Nutzen:** Breite der ersten Merkmalsschicht.<br>**Wofür:** 32–64. |
| `stem.kernel_size` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Kernelgröße.<br>**Nutzen:** Groß (7) für hochauflösende Bilder.<br>**Wofür:** 3 klein, 7 ImageNet. |
| `stem.stride` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Schrittweite.<br>**Nutzen:** Frühes Downsampling spart Rechenzeit.<br>**Wofür:** 1 kleine, 2 große Bilder. |
| `stem.pool` | Text | `'none'` | Auswahl: none \| max | Basis | **Was:** Max-Pooling nach dem Stem.<br>**Nutzen:** Weiteres Downsampling.<br>**Wofür:** ResNet auf ImageNet: max. |
| `stem.norm` | Komponente aus Registry 'normalization' | `batchnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung im Stem.<br>**Nutzen:** Stabilität.<br>**Wofür:** CNNs. |
| `stem.norm.eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `stem.norm.momentum` | Kommazahl (oder None) | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Gewicht neuer Batches im laufenden Mittelwert/Varianz.<br>**Nutzen:** Klein = glattere, trägere Statistik; None = kumulativer Durchschnitt.<br>**Wofür:** Kleine/rauschige Batches → kleineres Momentum. |
| `stem.norm.affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard: an. |
| `stem.norm.track_running_stats` | Ja/Nein | `True` |  | Experte | **Was:** Laufende Statistik für die Inferenz mitführen.<br>**Nutzen:** Aus → auch bei Inferenz Batch-Statistik (Ergebnis hängt dann vom Batch ab!).<br>**Wofür:** Spezialfälle (z. B. Test-Time-Adaption). |
| `stem.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung im Stem.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** CNNs. |
| `stem.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `stem.patchify` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Stem als nicht-überlappende Patch-Faltung (kernel = stride).<br>**Nutzen:** ConvNeXt/ViT-artig: grobe Patches statt Pixel.<br>**Wofür:** ConvNeXt (4×4-Patches). |
| `stages` | Liste von CNN-Stufe | 3 Elemente | ≥ 1 Elemente | Basis | **Was:** Liste der Stufen (jede mit eigenen Einstellungen).<br>**Nutzen:** Frei kombinierbar: z. B. erst residual, dann inverted_residual mit SE.<br>**Wofür:** Tiefe und Breite des Netzes. |
| `stages.0` | beliebig | `StageConfig(channels=32, num_blocks=1, stride=1)` |  | Basis | **Was:** Element vom Typ 'CNN-Stufe'. |
| `stages.0.block_type` | Text | `'residual'` | Auswahl: plain \| residual \| bottleneck \| inverted_residual \| convnext \| depthwise_separable | Basis | **Was:** Bauart der Blöcke.<br>**Nutzen:** Bestimmt Tiefe-vs-Effizienz-Abwägung und Gradientenfluss.<br>**Wofür:** residual: robuster Standard; inverted_residual: effizient; convnext: modern & stark. |
| `stages.0.kernel_size` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Kernelgröße der (Haupt-)Faltung.<br>**Nutzen:** Größer = mehr Kontext pro Schicht.<br>**Wofür:** 3 Standard; ConvNeXt: 7. |
| `stages.0.expansion` | Kommazahl | `4.0` | ≥ 0.1 | Fortgeschritten | **Was:** Expansionsfaktor (bottleneck: Reduktion; inverted_residual/convnext: Aufweitung).<br>**Nutzen:** Steuert Breite der inneren Schicht.<br>**Wofür:** bottleneck 4, MobileNetV2 6, ConvNeXt 4. |
| `stages.0.groups` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Gruppen der Hauptfaltung (ResNeXt-Kardinalität).<br>**Nutzen:** Mehr Gruppen = effizienter.<br>**Wofür:** ResNeXt: 32. |
| `stages.0.dilation` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Dilatation der Hauptfaltung.<br>**Nutzen:** Größeres rezeptives Feld ohne Downsampling.<br>**Wofür:** Segmentierung. |
| `stages.0.norm` | Komponente aus Registry 'normalization' | `batchnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung in den Blöcken.<br>**Nutzen:** BatchNorm Standard; GroupNorm bei kleinen Batches.<br>**Wofür:** CNNs. |
| `stages.0.norm.eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `stages.0.norm.momentum` | Kommazahl (oder None) | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Gewicht neuer Batches im laufenden Mittelwert/Varianz.<br>**Nutzen:** Klein = glattere, trägere Statistik; None = kumulativer Durchschnitt.<br>**Wofür:** Kleine/rauschige Batches → kleineres Momentum. |
| `stages.0.norm.affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard: an. |
| `stages.0.norm.track_running_stats` | Ja/Nein | `True` |  | Experte | **Was:** Laufende Statistik für die Inferenz mitführen.<br>**Nutzen:** Aus → auch bei Inferenz Batch-Statistik (Ergebnis hängt dann vom Batch ab!).<br>**Wofür:** Spezialfälle (z. B. Test-Time-Adaption). |
| `stages.0.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung in den Blöcken.<br>**Nutzen:** relu klassisch; silu (EfficientNet); gelu (ConvNeXt).<br>**Wofür:** CNNs. |
| `stages.0.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `stages.0.se_ratio` | Kommazahl (oder None) | `None` | ≥ 1.0 | Fortgeschritten | **Was:** Squeeze-Excitation-Reduktion (None = aus, z. B. 4 oder 16).<br>**Nutzen:** Kanal-Attention verbessert Genauigkeit bei geringen Kosten.<br>**Wofür:** EfficientNet, SE-ResNet. |
| `stages.0.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Kanal-Dropout im Block.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Datensätze. |
| `stages.0.drop_path` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Stochastic Depth: Wahrscheinlichkeit, den Residual-Zweig auszulassen.<br>**Nutzen:** Starke Regularisierung tiefer Residual-Netze.<br>**Wofür:** Tiefe ResNets/ConvNeXt (0.1–0.4). |
| `stages.0.layer_scale_init` | Kommazahl (oder None) | `1e-06` | ≥ 0.0 | Experte | **Was:** Startwert der LayerScale (nur convnext; None = aus).<br>**Nutzen:** Kleiner Start macht tiefe Netze stabil.<br>**Wofür:** ConvNeXt: 1e-6. |
| `stages.0.bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias in den Faltungen.<br>**Nutzen:** Vor Normalisierung überflüssig.<br>**Wofür:** Ohne Normalisierung: an. |
| `stages.0.padding_mode` | Text | `'zeros'` | Auswahl: zeros \| reflect \| replicate \| circular | Experte | **Was:** Randauffüllung.<br>**Nutzen:** Beeinflusst Randartefakte.<br>**Wofür:** circular für periodische Daten. |
| `stages.0.channels` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Ausgabekanäle der Stufe.<br>**Nutzen:** Mehr Kanäle = mehr Merkmale.<br>**Wofür:** Typisch verdoppelnd: 64, 128, 256, 512. |
| `stages.0.num_blocks` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Blöcke der Stufe.<br>**Nutzen:** Tiefe der Stufe.<br>**Wofür:** ResNet-18: 2,2,2,2; ResNet-50: 3,4,6,3. |
| `stages.0.stride` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Schrittweite des ersten Blocks (Downsampling).<br>**Nutzen:** 2 halbiert die Auflösung.<br>**Wofür:** Erste Stufe oft 1. |
| `stages.0.pool_after` | Text | `'none'` | Auswahl: none \| max \| avg | Fortgeschritten | **Was:** Zusätzliches Pooling nach der Stufe.<br>**Nutzen:** Extra-Downsampling.<br>**Wofür:** Plain-Netze (VGG). |
| `stages.1` | beliebig | `StageConfig()` |  | Basis | **Was:** Element vom Typ 'CNN-Stufe'. |
| `stages.1.block_type` | Text | `'residual'` | Auswahl: plain \| residual \| bottleneck \| inverted_residual \| convnext \| depthwise_separable | Basis | **Was:** Bauart der Blöcke.<br>**Nutzen:** Bestimmt Tiefe-vs-Effizienz-Abwägung und Gradientenfluss.<br>**Wofür:** residual: robuster Standard; inverted_residual: effizient; convnext: modern & stark. |
| `stages.1.kernel_size` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Kernelgröße der (Haupt-)Faltung.<br>**Nutzen:** Größer = mehr Kontext pro Schicht.<br>**Wofür:** 3 Standard; ConvNeXt: 7. |
| `stages.1.expansion` | Kommazahl | `4.0` | ≥ 0.1 | Fortgeschritten | **Was:** Expansionsfaktor (bottleneck: Reduktion; inverted_residual/convnext: Aufweitung).<br>**Nutzen:** Steuert Breite der inneren Schicht.<br>**Wofür:** bottleneck 4, MobileNetV2 6, ConvNeXt 4. |
| `stages.1.groups` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Gruppen der Hauptfaltung (ResNeXt-Kardinalität).<br>**Nutzen:** Mehr Gruppen = effizienter.<br>**Wofür:** ResNeXt: 32. |
| `stages.1.dilation` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Dilatation der Hauptfaltung.<br>**Nutzen:** Größeres rezeptives Feld ohne Downsampling.<br>**Wofür:** Segmentierung. |
| `stages.1.norm` | Komponente aus Registry 'normalization' | `batchnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung in den Blöcken.<br>**Nutzen:** BatchNorm Standard; GroupNorm bei kleinen Batches.<br>**Wofür:** CNNs. |
| `stages.1.norm.eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `stages.1.norm.momentum` | Kommazahl (oder None) | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Gewicht neuer Batches im laufenden Mittelwert/Varianz.<br>**Nutzen:** Klein = glattere, trägere Statistik; None = kumulativer Durchschnitt.<br>**Wofür:** Kleine/rauschige Batches → kleineres Momentum. |
| `stages.1.norm.affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard: an. |
| `stages.1.norm.track_running_stats` | Ja/Nein | `True` |  | Experte | **Was:** Laufende Statistik für die Inferenz mitführen.<br>**Nutzen:** Aus → auch bei Inferenz Batch-Statistik (Ergebnis hängt dann vom Batch ab!).<br>**Wofür:** Spezialfälle (z. B. Test-Time-Adaption). |
| `stages.1.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung in den Blöcken.<br>**Nutzen:** relu klassisch; silu (EfficientNet); gelu (ConvNeXt).<br>**Wofür:** CNNs. |
| `stages.1.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `stages.1.se_ratio` | Kommazahl (oder None) | `None` | ≥ 1.0 | Fortgeschritten | **Was:** Squeeze-Excitation-Reduktion (None = aus, z. B. 4 oder 16).<br>**Nutzen:** Kanal-Attention verbessert Genauigkeit bei geringen Kosten.<br>**Wofür:** EfficientNet, SE-ResNet. |
| `stages.1.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Kanal-Dropout im Block.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Datensätze. |
| `stages.1.drop_path` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Stochastic Depth: Wahrscheinlichkeit, den Residual-Zweig auszulassen.<br>**Nutzen:** Starke Regularisierung tiefer Residual-Netze.<br>**Wofür:** Tiefe ResNets/ConvNeXt (0.1–0.4). |
| `stages.1.layer_scale_init` | Kommazahl (oder None) | `1e-06` | ≥ 0.0 | Experte | **Was:** Startwert der LayerScale (nur convnext; None = aus).<br>**Nutzen:** Kleiner Start macht tiefe Netze stabil.<br>**Wofür:** ConvNeXt: 1e-6. |
| `stages.1.bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias in den Faltungen.<br>**Nutzen:** Vor Normalisierung überflüssig.<br>**Wofür:** Ohne Normalisierung: an. |
| `stages.1.padding_mode` | Text | `'zeros'` | Auswahl: zeros \| reflect \| replicate \| circular | Experte | **Was:** Randauffüllung.<br>**Nutzen:** Beeinflusst Randartefakte.<br>**Wofür:** circular für periodische Daten. |
| `stages.1.channels` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Ausgabekanäle der Stufe.<br>**Nutzen:** Mehr Kanäle = mehr Merkmale.<br>**Wofür:** Typisch verdoppelnd: 64, 128, 256, 512. |
| `stages.1.num_blocks` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Anzahl Blöcke der Stufe.<br>**Nutzen:** Tiefe der Stufe.<br>**Wofür:** ResNet-18: 2,2,2,2; ResNet-50: 3,4,6,3. |
| `stages.1.stride` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Schrittweite des ersten Blocks (Downsampling).<br>**Nutzen:** 2 halbiert die Auflösung.<br>**Wofür:** Erste Stufe oft 1. |
| `stages.1.pool_after` | Text | `'none'` | Auswahl: none \| max \| avg | Fortgeschritten | **Was:** Zusätzliches Pooling nach der Stufe.<br>**Nutzen:** Extra-Downsampling.<br>**Wofür:** Plain-Netze (VGG). |
| `stages.2` | beliebig | `StageConfig(channels=128)` |  | Basis | **Was:** Element vom Typ 'CNN-Stufe'. |
| `stages.2.block_type` | Text | `'residual'` | Auswahl: plain \| residual \| bottleneck \| inverted_residual \| convnext \| depthwise_separable | Basis | **Was:** Bauart der Blöcke.<br>**Nutzen:** Bestimmt Tiefe-vs-Effizienz-Abwägung und Gradientenfluss.<br>**Wofür:** residual: robuster Standard; inverted_residual: effizient; convnext: modern & stark. |
| `stages.2.kernel_size` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Kernelgröße der (Haupt-)Faltung.<br>**Nutzen:** Größer = mehr Kontext pro Schicht.<br>**Wofür:** 3 Standard; ConvNeXt: 7. |
| `stages.2.expansion` | Kommazahl | `4.0` | ≥ 0.1 | Fortgeschritten | **Was:** Expansionsfaktor (bottleneck: Reduktion; inverted_residual/convnext: Aufweitung).<br>**Nutzen:** Steuert Breite der inneren Schicht.<br>**Wofür:** bottleneck 4, MobileNetV2 6, ConvNeXt 4. |
| `stages.2.groups` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Gruppen der Hauptfaltung (ResNeXt-Kardinalität).<br>**Nutzen:** Mehr Gruppen = effizienter.<br>**Wofür:** ResNeXt: 32. |
| `stages.2.dilation` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Dilatation der Hauptfaltung.<br>**Nutzen:** Größeres rezeptives Feld ohne Downsampling.<br>**Wofür:** Segmentierung. |
| `stages.2.norm` | Komponente aus Registry 'normalization' | `batchnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung in den Blöcken.<br>**Nutzen:** BatchNorm Standard; GroupNorm bei kleinen Batches.<br>**Wofür:** CNNs. |
| `stages.2.norm.eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `stages.2.norm.momentum` | Kommazahl (oder None) | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Gewicht neuer Batches im laufenden Mittelwert/Varianz.<br>**Nutzen:** Klein = glattere, trägere Statistik; None = kumulativer Durchschnitt.<br>**Wofür:** Kleine/rauschige Batches → kleineres Momentum. |
| `stages.2.norm.affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard: an. |
| `stages.2.norm.track_running_stats` | Ja/Nein | `True` |  | Experte | **Was:** Laufende Statistik für die Inferenz mitführen.<br>**Nutzen:** Aus → auch bei Inferenz Batch-Statistik (Ergebnis hängt dann vom Batch ab!).<br>**Wofür:** Spezialfälle (z. B. Test-Time-Adaption). |
| `stages.2.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung in den Blöcken.<br>**Nutzen:** relu klassisch; silu (EfficientNet); gelu (ConvNeXt).<br>**Wofür:** CNNs. |
| `stages.2.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `stages.2.se_ratio` | Kommazahl (oder None) | `None` | ≥ 1.0 | Fortgeschritten | **Was:** Squeeze-Excitation-Reduktion (None = aus, z. B. 4 oder 16).<br>**Nutzen:** Kanal-Attention verbessert Genauigkeit bei geringen Kosten.<br>**Wofür:** EfficientNet, SE-ResNet. |
| `stages.2.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Kanal-Dropout im Block.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Datensätze. |
| `stages.2.drop_path` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Stochastic Depth: Wahrscheinlichkeit, den Residual-Zweig auszulassen.<br>**Nutzen:** Starke Regularisierung tiefer Residual-Netze.<br>**Wofür:** Tiefe ResNets/ConvNeXt (0.1–0.4). |
| `stages.2.layer_scale_init` | Kommazahl (oder None) | `1e-06` | ≥ 0.0 | Experte | **Was:** Startwert der LayerScale (nur convnext; None = aus).<br>**Nutzen:** Kleiner Start macht tiefe Netze stabil.<br>**Wofür:** ConvNeXt: 1e-6. |
| `stages.2.bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias in den Faltungen.<br>**Nutzen:** Vor Normalisierung überflüssig.<br>**Wofür:** Ohne Normalisierung: an. |
| `stages.2.padding_mode` | Text | `'zeros'` | Auswahl: zeros \| reflect \| replicate \| circular | Experte | **Was:** Randauffüllung.<br>**Nutzen:** Beeinflusst Randartefakte.<br>**Wofür:** circular für periodische Daten. |
| `stages.2.channels` | Ganzzahl | `128` | ≥ 1 | Basis | **Was:** Ausgabekanäle der Stufe.<br>**Nutzen:** Mehr Kanäle = mehr Merkmale.<br>**Wofür:** Typisch verdoppelnd: 64, 128, 256, 512. |
| `stages.2.num_blocks` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Anzahl Blöcke der Stufe.<br>**Nutzen:** Tiefe der Stufe.<br>**Wofür:** ResNet-18: 2,2,2,2; ResNet-50: 3,4,6,3. |
| `stages.2.stride` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Schrittweite des ersten Blocks (Downsampling).<br>**Nutzen:** 2 halbiert die Auflösung.<br>**Wofür:** Erste Stufe oft 1. |
| `stages.2.pool_after` | Text | `'none'` | Auswahl: none \| max \| avg | Fortgeschritten | **Was:** Zusätzliches Pooling nach der Stufe.<br>**Nutzen:** Extra-Downsampling.<br>**Wofür:** Plain-Netze (VGG). |
| `block_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Blöcke (global durchnummeriert über alle Stufen).<br>**Nutzen:** Jeder einzelne Block kann anders sein.<br>**Wofür:** Feinstes Tuning, z. B. {'-1': {'se_ratio': 4}}. |
| `drop_path_schedule` | Text | `'linear'` | Auswahl: constant \| linear | Experte | **Was:** Verteilung der Stochastic-Depth-Raten.<br>**Nutzen:** Tiefe Blöcke häufiger auslassen ist üblich.<br>**Wofür:** Bei drop_path > 0. |
| `task` | Text | `'classification'` | Auswahl: classification \| dense | Basis | **Was:** Art der Ausgabe.<br>**Nutzen:** Ein Wert pro Bild vs. ein Wert pro Pixel.<br>**Wofür:** Klassifikation vs. Segmentierung. |
| `global_pool` | Text | `'avg'` | Auswahl: avg \| max \| avgmax \| flatten | Basis | **Was:** Verdichtung der Feature-Maps vor dem Kopf.<br>**Nutzen:** avg ist robust; flatten behält Ortsinformation (braucht feste input_size).<br>**Wofür:** Klassifikation. |
| `head_hidden_dims` | Liste[Ganzzahl] | `[]` | ≥ 1 | Basis | **Was:** Versteckte Schichten des Klassifikationskopfs.<br>**Nutzen:** Mehr Kapazität im Kopf.<br>**Wofür:** Meist leer (linearer Kopf). |
| `head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout vor dem Kopf.<br>**Nutzen:** Regularisierung.<br>**Wofür:** EfficientNet: 0.2–0.5. |
| `head_activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Aktivierung im Kopf-MLP.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** Nur mit head_hidden_dims. |
| `head_activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `output_dim` | Ganzzahl | `10` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben/Klassen.<br>**Nutzen:** Form der Vorhersage.<br>**Wofür:** Klassenzahl. |
| `upsample_dense` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Dichte Ausgabe auf Eingabeauflösung hochskalieren.<br>**Nutzen:** Pixelgenaue Masken.<br>**Wofür:** Segmentierung. |

#### `model:ensemble` — Stufe Basis

**Was:** Ensemble mehrerer Modelle.

**Nutzen:** Genauer, robuster, mit Unsicherheitsmaß.

**Wofür:** Kombination beliebiger Modelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `members` | Liste von Komponenten aus Registry 'model' | 2 Elemente | Elemente: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Mitglieder des Ensembles (beliebige Modelle mit gleicher Ein-/Ausgabeform).<br>**Nutzen:** Vielfalt der Mitglieder (Architektur, Init, Daten) macht Ensembles stark.<br>**Wofür:** 2–10 Mitglieder. |
| `members.0` | beliebig | `model:mlp()` |  | Basis | **Was:** Mehrschichtiges Perzeptron.<br>**Nutzen:** Einfach, schnell, universell.<br>**Wofür:** Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe. |
| `members.0.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `members.0.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `members.0.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `members.0.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `members.0.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `members.0.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `members.0.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `members.0.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `members.0.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `members.0.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `members.0.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `members.0.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `members.0.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `members.0.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `members.0.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `members.0.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `members.0.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `members.0.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `members.0.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `members.0.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `members.0.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `members.0.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `members.0.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `members.0.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `members.0.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `members.0.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `members.0.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `members.0.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `members.0.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `members.1` | beliebig | `model:mlp(hidden_dims=[64])` |  | Basis | **Was:** Mehrschichtiges Perzeptron.<br>**Nutzen:** Einfach, schnell, universell.<br>**Wofür:** Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe. |
| `members.1.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `members.1.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `members.1.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `members.1.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `members.1.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `members.1.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `members.1.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `members.1.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `members.1.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `members.1.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `members.1.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `members.1.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `members.1.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `members.1.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `members.1.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `members.1.hidden_dims` | Liste[Ganzzahl] | `[64]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `members.1.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `members.1.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `members.1.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `members.1.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `members.1.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `members.1.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `members.1.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `members.1.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `members.1.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `members.1.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `members.1.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `members.1.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `members.1.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `combine` | Text | `'mean'` | Auswahl: mean \| weighted \| median \| max \| vote \| stacking \| softmax_mean | Basis | **Was:** Kombinationsregel.<br>**Nutzen:** Bestimmt, wie Einzelvorhersagen verschmelzen.<br>**Wofür:** Regression: mean/median; Klassifikation: softmax_mean/vote. |
| `weights` | Liste[Kommazahl] | `[]` | ≥ 0.0 | Basis | **Was:** Gewichte für 'weighted' (leer = gleich).<br>**Nutzen:** Bessere Modelle stärker gewichten.<br>**Wofür:** Länge = Anzahl Mitglieder. |
| `learnable_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Gewichte für 'weighted' mittrainieren (Softmax-normiert).<br>**Nutzen:** Datengetriebene Gewichtung.<br>**Wofür:** Experimente. |
| `member_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 0.95 | Experimentell | **Was:** Im Training zufällig Mitglieder weglassen.<br>**Nutzen:** Erzwingt, dass jedes Mitglied alleine gut ist.<br>**Wofür:** Experimente. |
| `output_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** Ausgabebreite (nur für 'stacking'; None = automatisch).<br>**Nutzen:** Größe der Stacking-Schicht.<br>**Wofür:** Selten nötig. |

#### `model:pipeline` — Stufe Basis

**Was:** Modelle hintereinander geschaltet.

**Nutzen:** Mehrstufige Systeme.

**Wofür:** Extraktor + Kopf.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `stages` | Liste von Komponenten aus Registry 'model' | 2 Elemente | Elemente: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Stufen in Reihenfolge.<br>**Nutzen:** Jede Stufe darf ein beliebiges Modell sein.<br>**Wofür:** 2–5 Stufen. |
| `stages.0` | beliebig | `model:mlp(output_dim=32)` |  | Basis | **Was:** Mehrschichtiges Perzeptron.<br>**Nutzen:** Einfach, schnell, universell.<br>**Wofür:** Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe. |
| `stages.0.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `stages.0.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `stages.0.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `stages.0.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `stages.0.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `stages.0.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `stages.0.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `stages.0.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `stages.0.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `stages.0.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `stages.0.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `stages.0.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `stages.0.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `stages.0.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `stages.0.output_dim` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `stages.0.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `stages.0.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `stages.0.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `stages.0.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `stages.0.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `stages.0.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `stages.0.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `stages.0.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `stages.0.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `stages.0.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `stages.0.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `stages.0.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `stages.0.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `stages.0.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `stages.1` | beliebig | `model:mlp(input_dim=32)` |  | Basis | **Was:** Mehrschichtiges Perzeptron.<br>**Nutzen:** Einfach, schnell, universell.<br>**Wofür:** Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe. |
| `stages.1.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `stages.1.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `stages.1.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `stages.1.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `stages.1.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `stages.1.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `stages.1.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `stages.1.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `stages.1.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `stages.1.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `stages.1.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `stages.1.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `stages.1.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `stages.1.input_dim` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `stages.1.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `stages.1.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `stages.1.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `stages.1.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `stages.1.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `stages.1.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `stages.1.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `stages.1.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `stages.1.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `stages.1.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `stages.1.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `stages.1.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `stages.1.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `stages.1.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `stages.1.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `adapters` | Ja/Nein | `True` |  | Basis | **Was:** Automatisch lineare Adapter einfügen, wenn Ausgabe- und Eingabebreite nicht passen.<br>**Nutzen:** Verbindet Modelle unterschiedlicher Breite ohne Handarbeit.<br>**Wofür:** Fast immer an. |
| `adapter_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Aktivierung nach Adaptern.<br>**Nutzen:** Zusätzliche Nichtlinearität zwischen Stufen.<br>**Wofür:** identity oder gelu. |
| `detach_between` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Gradientenfluss zwischen Stufen kappen.<br>**Nutzen:** Frühere Stufen werden dann nicht mittrainiert (eingefroren).<br>**Wofür:** Vortrainierte Merkmalsextraktoren. |

#### `model:fusion` (Aliase: `multimodal`) — Stufe Basis

**Was:** Multimodale Fusion mehrerer Zweige.

**Nutzen:** Verbindet Datenarten.

**Wofür:** Bild + Text + Tabelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `branches` | Liste von Fusionszweig | 2 Elemente | ≥ 1 Elemente | Basis | **Was:** Zweige (je eine Modalität).<br>**Nutzen:** Jeder Zweig ein eigenes Modell.<br>**Wofür:** 2–4 Zweige. |
| `branches.0` | beliebig | `FusionBranchConfig(name='a', input_key='a', model={'name': '` |  | Basis | **Was:** Element vom Typ 'Fusionszweig'. |
| `branches.0.name` | Text | `'a'` |  | Basis | **Was:** Name des Zweigs.<br>**Nutzen:** Zur Übersicht.<br>**Wofür:** z. B. 'bild', 'text'. |
| `branches.0.input_key` | Text | `'a'` |  | Basis | **Was:** Schlüssel der Eingabe im Eingabe-dict.<br>**Nutzen:** Welche Daten dieser Zweig bekommt.<br>**Wofür:** z. B. 'bild'. |
| `branches.0.model` | Komponente aus Registry 'model' | `mlp` | Auswahl: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Modell des Zweigs (Ausgabe = Merkmalsvektor).<br>**Nutzen:** Jede Modalität mit passender Architektur (CNN für Bilder, Transformer für Text ...).<br>**Wofür:** output_dim des Modells = Merkmalsgröße. |
| `branches.0.model.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `branches.0.model.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `branches.0.model.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `branches.0.model.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `branches.0.model.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `branches.0.model.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `branches.0.model.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `branches.0.model.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `branches.0.model.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `branches.0.model.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `branches.0.model.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `branches.0.model.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `branches.0.model.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `branches.0.model.input_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `branches.0.model.output_dim` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `branches.0.model.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `branches.0.model.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `branches.0.model.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `branches.0.model.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `branches.0.model.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `branches.0.model.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `branches.0.model.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `branches.0.model.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `branches.0.model.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `branches.0.model.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `branches.0.model.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `branches.0.model.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `branches.0.model.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `branches.0.model.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `branches.0.freeze` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Zweig einfrieren (nicht trainieren).<br>**Nutzen:** Für vortrainierte Zweige.<br>**Wofür:** Transfer. |
| `branches.1` | beliebig | `FusionBranchConfig(name='b', input_key='b', model={'name': '` |  | Basis | **Was:** Element vom Typ 'Fusionszweig'. |
| `branches.1.name` | Text | `'b'` |  | Basis | **Was:** Name des Zweigs.<br>**Nutzen:** Zur Übersicht.<br>**Wofür:** z. B. 'bild', 'text'. |
| `branches.1.input_key` | Text | `'b'` |  | Basis | **Was:** Schlüssel der Eingabe im Eingabe-dict.<br>**Nutzen:** Welche Daten dieser Zweig bekommt.<br>**Wofür:** z. B. 'bild'. |
| `branches.1.model` | Komponente aus Registry 'model' | `mlp` | Auswahl: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Modell des Zweigs (Ausgabe = Merkmalsvektor).<br>**Nutzen:** Jede Modalität mit passender Architektur (CNN für Bilder, Transformer für Text ...).<br>**Wofür:** output_dim des Modells = Merkmalsgröße. |
| `branches.1.model.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `branches.1.model.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `branches.1.model.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `branches.1.model.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `branches.1.model.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `branches.1.model.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `branches.1.model.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `branches.1.model.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `branches.1.model.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `branches.1.model.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `branches.1.model.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `branches.1.model.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `branches.1.model.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `branches.1.model.input_dim` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `branches.1.model.output_dim` | Ganzzahl | `32` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `branches.1.model.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `branches.1.model.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `branches.1.model.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `branches.1.model.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `branches.1.model.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `branches.1.model.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `branches.1.model.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `branches.1.model.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `branches.1.model.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `branches.1.model.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `branches.1.model.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `branches.1.model.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `branches.1.model.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `branches.1.model.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `branches.1.freeze` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Zweig einfrieren (nicht trainieren).<br>**Nutzen:** Für vortrainierte Zweige.<br>**Wofür:** Transfer. |
| `fusion` | Text | `'concat'` | Auswahl: concat \| sum \| gated \| attention | Basis | **Was:** Fusionsverfahren.<br>**Nutzen:** Wie Modalitäten interagieren.<br>**Wofür:** concat einfach; attention ausdrucksstark. |
| `fusion_dim` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Gemeinsame Breite (sum/gated/attention).<br>**Nutzen:** Größe des gemeinsamen Raums.<br>**Wofür:** 64–512. |
| `attention_heads` | Ganzzahl | `4` | ≥ 1 | Fortgeschritten | **Was:** Köpfe (attention).<br>**Nutzen:** Interaktionsmuster.<br>**Wofür:** 4. |
| `head_hidden_dims` | Liste[Ganzzahl] | `[64]` | ≥ 1 | Basis | **Was:** Versteckte Schichten des gemeinsamen Kopfs.<br>**Nutzen:** Verarbeitung nach der Fusion.<br>**Wofür:** [64]. |
| `activation` | Komponente aus Registry 'activation' | `gelu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung im Kopf.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** gelu. |
| `activation.approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout im Kopf.<br>**Nutzen:** Regularisierung.<br>**Wofür:** 0.1. |
| `modality_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 0.95 | Experimentell | **Was:** Im Training ganze Modalitäten zufällig auf 0 setzen.<br>**Nutzen:** Robustheit gegen fehlende Modalitäten zur Laufzeit.<br>**Wofür:** Multimodale Systeme mit Ausfällen. |
| `output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Ausgabegröße.<br>**Nutzen:** Klassen/Zielwerte.<br>**Wofür:** Je nach Aufgabe. |

#### `model:mixture_of_models` (Aliase: `mom`) — Stufe Fortgeschritten

**Was:** Router über ganze Modelle.

**Nutzen:** Spezialisierte Experten.

**Wofür:** Heterogene Daten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `experts` | Liste von Komponenten aus Registry 'model' | 2 Elemente | Elemente: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Experten-Modelle (gleiche Ein-/Ausgabeform).<br>**Nutzen:** Beliebige Architekturen.<br>**Wofür:** 2–8. |
| `experts.0` | beliebig | `model:mlp()` |  | Basis | **Was:** Mehrschichtiges Perzeptron.<br>**Nutzen:** Einfach, schnell, universell.<br>**Wofür:** Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe. |
| `experts.0.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `experts.0.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `experts.0.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `experts.0.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `experts.0.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `experts.0.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `experts.0.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `experts.0.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `experts.0.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `experts.0.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `experts.0.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `experts.0.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `experts.0.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `experts.0.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `experts.0.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `experts.0.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `experts.0.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `experts.0.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `experts.0.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `experts.0.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `experts.0.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `experts.0.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `experts.0.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `experts.0.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `experts.0.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `experts.0.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `experts.0.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `experts.0.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `experts.0.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `experts.1` | beliebig | `model:mlp(hidden_dims=[32, 32, 32])` |  | Basis | **Was:** Mehrschichtiges Perzeptron.<br>**Nutzen:** Einfach, schnell, universell.<br>**Wofür:** Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe. |
| `experts.1.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `experts.1.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `experts.1.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `experts.1.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `experts.1.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `experts.1.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `experts.1.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `experts.1.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `experts.1.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `experts.1.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `experts.1.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `experts.1.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `experts.1.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `experts.1.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `experts.1.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `experts.1.hidden_dims` | Liste[Ganzzahl] | `[32, 32, 32]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `experts.1.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `experts.1.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `experts.1.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `experts.1.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `experts.1.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `experts.1.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `experts.1.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `experts.1.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `experts.1.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `experts.1.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `experts.1.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `experts.1.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `experts.1.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `router_hidden_dims` | Liste[Ganzzahl] | `[32]` | ≥ 1 | Basis | **Was:** Versteckte Schichten des Routers.<br>**Nutzen:** Kapazität der Zuteilungsentscheidung.<br>**Wofür:** [32]. |
| `top_k` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Nur die k besten Experten nutzen (None = alle gewichtet).<br>**Nutzen:** Sparsere, spezialisiertere Zuteilung.<br>**Wofür:** 1–2. |
| `router_noise_std` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Rauschen auf Router-Logits im Training.<br>**Nutzen:** Erkundung, verhindert Kollaps.<br>**Wofür:** 0.1. |
| `balance_loss_coef` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Gewicht des Lastverteilungs-Verlusts.<br>**Nutzen:** Verhindert, dass nur ein Experte genutzt wird.<br>**Wofür:** 0.01. |

#### `model:multitask` (Aliase: `multi_task`) — Stufe Basis

**Was:** Gemeinsamer Rumpf, mehrere Köpfe.

**Nutzen:** Multi-Task-Learning.

**Wofür:** Mehrere Ziele gleichzeitig.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `backbone` | Komponente aus Registry 'model' | `mlp` | Auswahl: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Gemeinsamer Rumpf (Ausgabe = gemeinsame Merkmale).<br>**Nutzen:** Lernt die geteilte Repräsentation.<br>**Wofür:** Beliebiges Modell. |
| `backbone.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `backbone.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `backbone.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `backbone.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `backbone.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `backbone.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `backbone.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `backbone.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `backbone.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `backbone.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `backbone.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `backbone.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `backbone.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `backbone.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `backbone.output_dim` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `backbone.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `backbone.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `backbone.layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `backbone.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `backbone.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `backbone.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `backbone.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `backbone.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `backbone.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `backbone.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `backbone.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `backbone.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `backbone.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `backbone.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `heads` | Liste von Aufgaben-Kopf | 2 Elemente | ≥ 1 Elemente | Basis | **Was:** Köpfe (je Aufgabe einer).<br>**Nutzen:** Aufgabenspezifische Ausgaben.<br>**Wofür:** 2–5. |
| `heads.0` | beliebig | `TaskHeadConfig(name='klasse', output_dim=3)` |  | Basis | **Was:** Element vom Typ 'Aufgaben-Kopf'. |
| `heads.0.name` | Text | `'klasse'` |  | Basis | **Was:** Name der Aufgabe (Schlüssel der Ausgabe).<br>**Nutzen:** Für Verluste/Targets.<br>**Wofür:** z. B. 'klasse'. |
| `heads.0.output_dim` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Ausgabegröße dieser Aufgabe.<br>**Nutzen:** Klassen/Zielwerte.<br>**Wofür:** Je Aufgabe. |
| `heads.0.hidden_dims` | Liste[Ganzzahl] | `[]` | ≥ 1 | Basis | **Was:** Versteckte Schichten des Kopfs.<br>**Nutzen:** Kapazität.<br>**Wofür:** Meist leer. |
| `heads.0.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu. |
| `heads.0.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `heads.1` | beliebig | `TaskHeadConfig(name='wert')` |  | Basis | **Was:** Element vom Typ 'Aufgaben-Kopf'. |
| `heads.1.name` | Text | `'wert'` |  | Basis | **Was:** Name der Aufgabe (Schlüssel der Ausgabe).<br>**Nutzen:** Für Verluste/Targets.<br>**Wofür:** z. B. 'klasse'. |
| `heads.1.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Ausgabegröße dieser Aufgabe.<br>**Nutzen:** Klassen/Zielwerte.<br>**Wofür:** Je Aufgabe. |
| `heads.1.hidden_dims` | Liste[Ganzzahl] | `[]` | ≥ 1 | Basis | **Was:** Versteckte Schichten des Kopfs.<br>**Nutzen:** Kapazität.<br>**Wofür:** Meist leer. |
| `heads.1.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu. |
| `heads.1.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `backbone_activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Aktivierung zwischen Rumpf und Köpfen.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu oder identity. |
| `backbone_activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |

#### `model:mlp` (Aliase: `perzeptron`, `dense_net`) — Stufe Basis

**Was:** Mehrschichtiges Perzeptron.

**Nutzen:** Einfach, schnell, universell.

**Wofür:** Tabellendaten, Regression, Klassifikation, RL-Policies, Köpfe.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |

#### `model:gan` — Stufe Basis

**Was:** Generative Adversarial Network.

**Nutzen:** Scharfe, realistische Samples.

**Wofür:** Datensynthese, Augmentation.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `data_dim` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Dimension der (abgeflachten) Daten.<br>**Nutzen:** Ausgabegröße des Generators / Eingabe des Diskriminators.<br>**Wofür:** 2 für 2D-Spielzeugdaten, 784 für 28×28-Bilder. |
| `latent_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Größe des Rauschvektors z.<br>**Nutzen:** Mehr Dimensionen = mehr Variationsmöglichkeiten.<br>**Wofür:** 16–512. |
| `noise_distribution` | Text | `'normal'` | Auswahl: normal \| uniform | Fortgeschritten | **Was:** Verteilung des Rauschens.<br>**Nutzen:** Form des Latentraums.<br>**Wofür:** normal ist Standard. |
| `num_classes` | Ganzzahl | `0` | ≥ 0 | Basis | **Was:** Anzahl Klassen für bedingte Generierung (0 = unbedingt).<br>**Nutzen:** Generator und Diskriminator erhalten das Label → gezielte Erzeugung einer Klasse.<br>**Wofür:** Conditional GAN (cGAN). |
| `label_embedding_dim` | Ganzzahl | `8` | ≥ 1 | Fortgeschritten | **Was:** Größe des Label-Embeddings.<br>**Nutzen:** Repräsentation der Klasse.<br>**Wofür:** 8–64. |
| `generator` | Unter-Konfiguration MLP | ⟶ MLPConfig |  | Basis | **Was:** MLP-Generator (input_dim/output_dim werden automatisch gesetzt).<br>**Nutzen:** Erzeugt Daten aus Rauschen.<br>**Wofür:** Standard. |
| `generator.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `generator.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `generator.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `generator.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `generator.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `generator.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `generator.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `generator.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `generator.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `generator.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `generator.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `generator.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `generator.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `generator.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `generator.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `generator.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `generator.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `generator.layer.activation` | Komponente aus Registry 'activation' | `leaky_relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `generator.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `generator.layer.activation.negative_slope` | Kommazahl | `0.01` |  | Basis | **Was:** Steigung für negative Eingaben.<br>**Nutzen:** Verhindert 'tote' Neuronen, weil auch negative Werte einen kleinen Gradienten liefern.<br>**Wofür:** Tiefe Netze, GAN-Diskriminatoren (dort oft 0.2).<br>**Tipp:** 0.01 Standard, 0.1–0.3 bei GANs. 0 entspricht ReLU. |
| `generator.layer.norm` | Komponente aus Registry 'normalization' (oder None) | `batchnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `generator.layer.norm.eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `generator.layer.norm.momentum` | Kommazahl (oder None) | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Gewicht neuer Batches im laufenden Mittelwert/Varianz.<br>**Nutzen:** Klein = glattere, trägere Statistik; None = kumulativer Durchschnitt.<br>**Wofür:** Kleine/rauschige Batches → kleineres Momentum. |
| `generator.layer.norm.affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala/Verschiebung.<br>**Nutzen:** Wie γ/β.<br>**Wofür:** Standard: an. |
| `generator.layer.norm.track_running_stats` | Ja/Nein | `True` |  | Experte | **Was:** Laufende Statistik für die Inferenz mitführen.<br>**Nutzen:** Aus → auch bei Inferenz Batch-Statistik (Ergebnis hängt dann vom Batch ab!).<br>**Wofür:** Spezialfälle (z. B. Test-Time-Adaption). |
| `generator.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `generator.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `generator.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `generator.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `generator.layer.parametrization` | Text | `'none'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `generator.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `generator.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `generator.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `generator.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `discriminator` | Unter-Konfiguration MLP | ⟶ MLPConfig |  | Basis | **Was:** MLP-Diskriminator (input_dim/output_dim werden automatisch gesetzt).<br>**Nutzen:** Unterscheidet echt/erzeugt. Spectral-Norm stabilisiert.<br>**Wofür:** Standard. |
| `discriminator.init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `discriminator.init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `discriminator.init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `discriminator.init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `discriminator.init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `discriminator.init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `discriminator.init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `discriminator.init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `discriminator.init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `discriminator.init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `discriminator.init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `discriminator.init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `discriminator.extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `discriminator.input_dim` | Ganzzahl | `16` | ≥ 1 | Basis | **Was:** Anzahl Eingabe-Features.<br>**Nutzen:** Muss zur Datenbreite passen.<br>**Wofür:** Spaltenzahl der Tabelle / Größe des Vektors. |
| `discriminator.output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgaben (Klassen bei Klassifikation, Zielwerte bei Regression).<br>**Nutzen:** Bestimmt die Form der Vorhersage.<br>**Wofür:** Klassifikation: Klassenzahl; Regression: Zahl der Zielgrößen. |
| `discriminator.hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten der versteckten Schichten (Liste; Länge = Tiefe).<br>**Nutzen:** Breite = Kapazität pro Schicht, Länge = Abstraktionstiefe.<br>**Wofür:** Tabellendaten: 2–4 Schichten à 64–512.<br>**Tipp:** Leere Liste = lineares Modell (logistische/lineare Regression). |
| `discriminator.layer` | Unter-Konfiguration MLP-Schicht | ⟶ MLPLayerConfig |  | Basis | **Was:** Bauplan einer versteckten MLP-Schicht.<br>**Nutzen:** Gilt für alle Schichten; Abweichungen über layer_overrides.<br>**Wofür:** Feintuning jeder einzelnen Schicht. |
| `discriminator.layer.activation` | Komponente aus Registry 'activation' | `leaky_relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion der Schicht.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/gelu/silu sind gute Standards. |
| `discriminator.layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `discriminator.layer.activation.negative_slope` | Kommazahl | `0.2` |  | Basis | **Was:** Steigung für negative Eingaben.<br>**Nutzen:** Verhindert 'tote' Neuronen, weil auch negative Werte einen kleinen Gradienten liefern.<br>**Wofür:** Tiefe Netze, GAN-Diskriminatoren (dort oft 0.2).<br>**Tipp:** 0.01 Standard, 0.1–0.3 bei GANs. 0 entspricht ReLU. |
| `discriminator.layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung (None = keine).<br>**Nutzen:** Stabilisiert tiefe MLPs; batchnorm bei großen Batches, layernorm sonst.<br>**Wofür:** Tiefe MLPs (> 4 Schichten). |
| `discriminator.layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout-Rate.<br>**Nutzen:** Regularisierung gegen Überanpassung.<br>**Wofür:** Kleine Datensätze: 0.1–0.5. |
| `discriminator.layer.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias in der linearen Schicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Vor BatchNorm überflüssig. |
| `discriminator.layer.order` | Liste[Text] | `['linear', 'norm', 'activation', 'dropout']` |  | Experte | **Was:** Reihenfolge der Operationen innerhalb der Schicht.<br>**Nutzen:** Unterschiedliche Reihenfolgen (Norm vor/nach Aktivierung, Pre-Norm) beeinflussen Stabilität und Leistung.<br>**Wofür:** Experimente; z. B. ['norm', 'linear', 'activation'] für Pre-Norm-MLPs.<br>**Tipp:** Elemente: linear, norm, activation, dropout (jeweils höchstens sinnvoll einmal). |
| `discriminator.layer.residual` | Text | `'none'` | Auswahl: none \| identity \| projection | Fortgeschritten | **Was:** Residual-/Skip-Verbindung um die Schicht.<br>**Nutzen:** Erleichtert den Gradientenfluss in tiefen MLPs erheblich.<br>**Wofür:** MLPs mit vielen Schichten. |
| `discriminator.layer.parametrization` | Text | `'spectral_norm'` | Auswahl: none \| weight_norm \| spectral_norm | Experte | **Was:** Umparametrisierung der Gewichtsmatrix.<br>**Nutzen:** Spectral Norm stabilisiert GAN-Diskriminatoren; Weight Norm beschleunigt manche Trainings.<br>**Wofür:** GANs, Lipschitz-beschränkte Netze, RL-Kritiker. |
| `discriminator.layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede Schicht kann eigene Aktivierung, Normalisierung, Dropout ... haben.<br>**Wofür:** Feinstes Tuning, Experimente. Selektoren: '0', '-1', '1:3', 'gerade', '*' ...<br>**Tipp:** Beispiel: {'-1': {'dropout': 0.5}, '0': {'activation': 'tanh'}} |
| `discriminator.input_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout direkt auf die Eingabe.<br>**Nutzen:** Robustheit gegen fehlende/verrauschte Features.<br>**Wofür:** Tabellendaten, Denoising. |
| `discriminator.output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Ausgabeschicht.<br>**Nutzen:** identity für Logits/Regression; tanh/sigmoid für begrenzte Ausgaben.<br>**Wofür:** Klassifikation: identity (Softmax steckt im Verlust). Generatoren: tanh. |
| `discriminator.output_bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Verschiebung der Ausgabe.<br>**Wofür:** Standard. |
| `generator_model` | Komponente aus Registry 'model' (oder None) | None | Auswahl: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Experte | **Was:** Beliebiges anderes Modell als Generator (überschreibt 'generator').<br>**Nutzen:** Kombinationen, z. B. Transformer-Generator; muss latent_dim(+Label) → data_dim abbilden.<br>**Wofür:** Experimente. |
| `discriminator_model` | Komponente aus Registry 'model' (oder None) | None | Auswahl: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Experte | **Was:** Beliebiges anderes Modell als Diskriminator (überschreibt 'discriminator').<br>**Nutzen:** Z. B. CNN-Diskriminator für Bilder; muss Daten(+Label) → 1 Wert abbilden.<br>**Wofür:** Experimente. |
| `output_activation` | Komponente aus Registry 'activation' | `identity` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung der Generator-Ausgabe.<br>**Nutzen:** tanh für Daten in [-1, 1] (normalisierte Bilder).<br>**Wofür:** Bilder: tanh. |

#### `model:diffusion` (Aliase: `ddpm`, `ddim`) — Stufe Basis

**Was:** Denoising-Diffusionsmodell (DDPM/DDIM).

**Nutzen:** Stabil, hohe Qualität und Vielfalt.

**Wofür:** Datengenerierung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `data_shape` | Liste[Ganzzahl] | `[2]` | ≥ 1 | Basis | **Was:** Form eines Datenpunkts ohne Batch (z. B. [2] oder [1, 28, 28]).<br>**Nutzen:** Der Denoiser arbeitet auf der abgeflachten Form.<br>**Wofür:** Muss zu den Daten passen. |
| `hidden_dim` | Ganzzahl | `128` | ≥ 1 | Basis | **Was:** Breite des Denoisers.<br>**Nutzen:** Kapazität.<br>**Wofür:** 128–1024. |
| `num_blocks` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Residual-Blöcke im Denoiser.<br>**Nutzen:** Tiefe.<br>**Wofür:** 2–8. |
| `activation` | Komponente aus Registry 'activation' | `silu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** silu (Standard). |
| `norm` | Komponente aus Registry 'normalization' | `layernorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung.<br>**Nutzen:** Stabilität.<br>**Wofür:** layernorm. |
| `norm.eps` | Kommazahl | `1e-05` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `norm.elementwise_affine` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala (γ) und Verschiebung (β) pro Feature.<br>**Nutzen:** Gibt dem Netz die Möglichkeit, die Normalisierung teilweise rückgängig zu machen.<br>**Wofür:** Fast immer an; aus für parameterfreie Normierung. |
| `norm.bias` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Verschiebung β verwenden.<br>**Nutzen:** Ohne Bias etwas weniger Parameter; manche LLMs verzichten darauf.<br>**Wofür:** Moderne Transformer (oft ohne Bias). |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Daten: 0.1. |
| `time_embedding_dim` | Ganzzahl | `64` | ≥ 2 | Fortgeschritten | **Was:** Größe der Zeitschritt-Einbettung.<br>**Nutzen:** Wie gut das Netz Rauschstufen unterscheiden kann.<br>**Wofür:** 64–256. |
| `num_classes` | Ganzzahl | `0` | ≥ 0 | Basis | **Was:** Klassen für bedingte Generierung (0 = unbedingt).<br>**Nutzen:** Gezielte Erzeugung.<br>**Wofür:** Conditional Diffusion. |
| `class_dropout` | Kommazahl | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Wahrscheinlichkeit, im Training das Label wegzulassen.<br>**Nutzen:** Ermöglicht Classifier-Free Guidance (gleiches Netz bedingt + unbedingt).<br>**Wofür:** Bedingte Modelle: 0.1–0.2. |
| `timesteps` | Ganzzahl | `1000` | ≥ 1 | Basis | **Was:** Anzahl Diffusionsschritte T.<br>**Nutzen:** Feinere Zerlegung des Entrauschens.<br>**Wofür:** DDPM: 1000; Spielzeug: 100–200. |
| `beta_schedule` | Text | `'cosine'` | Auswahl: linear \| cosine \| quadratic \| sigmoid | Basis | **Was:** Rausch-Fahrplan.<br>**Nutzen:** Wie schnell Information zerstört wird.<br>**Wofür:** cosine ist robust. |
| `beta_start` | Kommazahl | `0.0001` | ≥ 0.0 | Experte | **Was:** β am Anfang.<br>**Nutzen:** Anfangs-Rauschstärke.<br>**Wofür:** 1e-4. |
| `beta_end` | Kommazahl | `0.02` | Bereich 0.0 … 1.0 | Experte | **Was:** β am Ende.<br>**Nutzen:** End-Rauschstärke.<br>**Wofür:** 0.02. |
| `prediction` | Text | `'eps'` | Auswahl: eps \| x0 \| v | Basis | **Was:** Vorhersageziel des Netzes.<br>**Nutzen:** Beeinflusst Stabilität und Qualität.<br>**Wofür:** eps oder v. |
| `sampler` | Text | `'ddim'` | Auswahl: ddpm \| ddim | Basis | **Was:** Sampling-Verfahren.<br>**Nutzen:** DDIM ist viel schneller.<br>**Wofür:** Schnelle Generierung: ddim. |
| `sample_steps` | Ganzzahl | `50` | ≥ 1 | Basis | **Was:** Schritte beim Sampling (nur ddim; ddpm nutzt alle).<br>**Nutzen:** Weniger = schneller, gröber.<br>**Wofür:** 20–100. |
| `ddim_eta` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Zufallsanteil bei DDIM (0 = deterministisch, 1 ≈ DDPM).<br>**Nutzen:** Vielfalt vs. Reproduzierbarkeit.<br>**Wofür:** 0. |
| `guidance_scale` | Kommazahl | `1.0` | ≥ 0.0 | Fortgeschritten | **Was:** Classifier-Free-Guidance-Stärke (1 = aus).<br>**Nutzen:** Höher = klassentreuer, weniger vielfältig.<br>**Wofür:** 2–7. |
| `clip_sample` | Kommazahl (oder None) | `None` | ≥ 0.0 | Fortgeschritten | **Was:** Vorhergesagtes x₀ auf [-c, c] begrenzen.<br>**Nutzen:** Verhindert Ausreißer bei Daten in bekanntem Bereich.<br>**Wofür:** Normalisierte Bilder: 1.0. |

#### `model:gnn` (Aliase: `graph_nn`) — Stufe Basis

**Was:** Graph-Neuronales Netz (GCN, GraphSAGE, GIN, GAT).

**Nutzen:** Lernt auf Beziehungsstrukturen.

**Wofür:** Moleküle, Netzwerke, Empfehlungen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `input_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Merkmale pro Knoten.<br>**Nutzen:** Eingabebreite.<br>**Wofür:** Je nach Daten. |
| `hidden_dims` | Liste[Ganzzahl] | `[64, 64]` | ≥ 1 | Basis | **Was:** Breiten der GNN-Schichten (Länge = Anzahl Nachrichtenrunden).<br>**Nutzen:** Jede Schicht erweitert die 'Sichtweite' um einen Kantenschritt.<br>**Wofür:** 2–4 Schichten (sonst Over-Smoothing). |
| `layer` | Unter-Konfiguration GNN-Schicht | ⟶ GNNLayerConfig |  | Basis | **Was:** Unter-Konfiguration 'GNN-Schicht'. |
| `layer.conv` | Text | `'gcn'` | Auswahl: gcn \| sage \| gin \| gat | Basis | **Was:** Art der Graph-Faltung.<br>**Nutzen:** Bestimmt, wie Nachbarinformation kombiniert wird.<br>**Wofür:** gcn: Standard; gin: Graph-Klassifikation; gat: heterogene Nachbarschaften. |
| `layer.aggregation` | Text | `'mean'` | Auswahl: sum \| mean \| max | Fortgeschritten | **Was:** Aggregation der Nachbarn (sage/gin).<br>**Nutzen:** sum unterscheidet Knotengrade, mean ist robuster.<br>**Wofür:** gin: sum. |
| `layer.heads` | Ganzzahl | `4` | ≥ 1 | Fortgeschritten | **Was:** Attention-Köpfe (gat).<br>**Nutzen:** Mehrere Sichtweisen.<br>**Wofür:** 4–8. |
| `layer.gat_concat` | Ja/Nein | `True` |  | Experte | **Was:** GAT-Köpfe konkatenieren (sonst mitteln).<br>**Nutzen:** Konkatenation = mehr Kapazität.<br>**Wofür:** Letzte Schicht: mitteln. |
| `layer.self_loops` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Jeder Knoten ist auch sein eigener Nachbar (gcn/gat).<br>**Nutzen:** Eigene Merkmale fließen mit ein.<br>**Wofür:** Fast immer an. |
| `layer.activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu/elu. |
| `layer.activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung.<br>**Nutzen:** Stabilität tiefer GNNs.<br>**Wofür:** layernorm. |
| `layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout.<br>**Nutzen:** Regularisierung.<br>**Wofür:** 0.1–0.5. |
| `layer.residual` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Residualverbindung (bei gleicher Breite).<br>**Nutzen:** Gegen Over-Smoothing in tiefen GNNs.<br>**Wofür:** > 3 Schichten. |
| `layer.learn_eps` | Ja/Nein | `True` |  | Experte | **Was:** ε in GIN lernbar.<br>**Nutzen:** Flexibles Gewicht des eigenen Knotens.<br>**Wofür:** gin. |
| `layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten.<br>**Nutzen:** Z. B. GAT in der ersten, GIN in späteren Schichten.<br>**Wofür:** Experimente. |
| `output_dim` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Ausgabegröße.<br>**Nutzen:** Klassen oder Zielwerte.<br>**Wofür:** Je nach Aufgabe. |
| `readout` | Text | `'mean'` | Auswahl: none \| mean \| sum \| max | Basis | **Was:** Verdichtung auf Graph-Ebene.<br>**Nutzen:** Graph- statt Knotenvorhersagen.<br>**Wofür:** Moleküle: mean/sum. |
| `jumping_knowledge` | Text | `'last'` | Auswahl: last \| concat \| max | Fortgeschritten | **Was:** Kombination der Schicht-Repräsentationen.<br>**Nutzen:** Nutzt mehrere Reichweiten gleichzeitig.<br>**Wofür:** Tiefe GNNs. |
| `edge_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil zufällig entfernter Kanten im Training (DropEdge).<br>**Nutzen:** Regularisierung, gegen Over-Smoothing.<br>**Wofür:** 0.1–0.3. |

#### `model:graph` (Aliase: `frei`, `custom`, `dag`) — Stufe Basis

**Was:** Freies Netz aus beliebigen Schichten (gerichteter Graph).

**Nutzen:** Jede denkbare Architektur ohne eigenen Code.

**Wofür:** Prototypen, Forschung, Multi-Input/-Output.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `inputs` | Liste von Eingang | 1 Elemente | ≥ 1 Elemente | Basis | **Was:** Eingänge des Netzes.<br>**Nutzen:** Mehrere Eingänge für multimodale Daten möglich.<br>**Wofür:** Mindestens einer. |
| `inputs.0` | beliebig | `GraphInputConfig()` |  | Basis | **Was:** Element vom Typ 'Eingang'. |
| `inputs.0.name` | Text | `'x'` |  | Basis | **Was:** Name des Eingangs.<br>**Nutzen:** Wird von Knoten in 'inputs' referenziert.<br>**Wofür:** z. B. 'bild', 'text'. |
| `inputs.0.shape` | Liste[Ganzzahl] | `[16]` | ≥ 1 | Basis | **Was:** Form eines Beispiels OHNE Batch-Achse.<br>**Nutzen:** Für den automatischen Probelauf.<br>**Wofür:** [16], [3, 32, 32], [128] ... |
| `inputs.0.dtype` | Text | `'float'` | Auswahl: float \| long | Basis | **Was:** Datentyp des Eingangs.<br>**Nutzen:** long für Embeddings.<br>**Wofür:** Text-IDs: long. |
| `inputs.0.max_value` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Obergrenze (exklusiv) für Dummy-Ganzzahlen im Probelauf (nur long).<br>**Nutzen:** Muss ≤ Vokabulargröße folgender Embeddings sein.<br>**Wofür:** Selten zu ändern. |
| `nodes` | Liste von Knoten | 3 Elemente | ≥ 1 Elemente | Basis | **Was:** Knoten in Ausführungsreihenfolge.<br>**Nutzen:** Die eigentliche Architektur.<br>**Wofür:** Beliebig viele. |
| `nodes.0` | beliebig | `GraphNodeConfig(name='dense1')` |  | Basis | **Was:** Ein Knoten des freien Netzes (eine Schicht mit Eingängen).<br>**Nutzen:** Grundbaustein frei definierter Architekturen.<br>**Wofür:** Modell 'graph'. |
| `nodes.0.name` | Text | `'dense1'` |  | Basis | **Was:** Eindeutiger Name.<br>**Nutzen:** Für Verweise anderer Knoten.<br>**Wofür:** Sprechende Namen helfen. |
| `nodes.0.layer` | Komponente aus Registry 'layer' | `linear` | Auswahl: linear \| conv \| conv_transpose \| depthwise_separable_conv \| pool \| adaptive_pool \| global_pool \| flatten \| reshape \| permute \| dropout \| … | Basis | **Was:** Schicht aus der Registry 'layer'.<br>**Nutzen:** Was dieser Knoten berechnet.<br>**Wofür:** Beliebig. |
| `nodes.0.layer.out_features` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Anzahl Ausgabe-Features.<br>**Nutzen:** Breite der Schicht.<br>**Wofür:** Jede lineare Schicht. |
| `nodes.0.layer.bias` | Ja/Nein | `True` |  | Basis | **Was:** Bias-Vektor addieren.<br>**Nutzen:** Verschiebt die Ausgabe; fast immer sinnvoll.<br>**Wofür:** Standard. |
| `nodes.0.inputs` | Liste[Text] | `[]` |  | Basis | **Was:** Namen der Eingänge/Knoten, die hier hineinfließen (leer = vorheriger Knoten).<br>**Nutzen:** Definiert die Kanten des Graphen.<br>**Wofür:** Verzweigungen, Skip-Verbindungen. |
| `nodes.0.merge` | Komponente aus Registry 'merge' | `concat` | Auswahl: concat \| add \| mul \| mean \| max \| stack \| weighted_sum \| gated \| first | Basis | **Was:** Zusammenführung bei mehreren Eingängen.<br>**Nutzen:** concat, add (Residual), mul, gated ...<br>**Wofür:** Nur bei > 1 Eingang. |
| `nodes.0.merge.dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse, entlang der verkettet wird (-1 = Features, 1 = Kanäle bei CNNs).<br>**Nutzen:** Bestimmt, wie Informationen nebeneinander gelegt werden.<br>**Wofür:** Fusion, DenseNet, U-Net. |
| `nodes.0.repeat` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Schicht n-mal hintereinander (je eigene Gewichte).<br>**Nutzen:** Kompakt für wiederholte Blöcke.<br>**Wofür:** 6× transformer_block. |
| `nodes.0.share_weights_with` | Text (oder None) | `None` |  | Experte | **Was:** Gewichte mit einem früheren Knoten teilen (gleiche Schicht, gleiche Parameter).<br>**Nutzen:** Siamesische Netze, Gewichtsbindung.<br>**Wofür:** Muss formkompatibel sein. |
| `nodes.1` | beliebig | `GraphNodeConfig(name='act1', layer={'name': 'activation', 'a` |  | Basis | **Was:** Ein Knoten des freien Netzes (eine Schicht mit Eingängen).<br>**Nutzen:** Grundbaustein frei definierter Architekturen.<br>**Wofür:** Modell 'graph'. |
| `nodes.1.name` | Text | `'act1'` |  | Basis | **Was:** Eindeutiger Name.<br>**Nutzen:** Für Verweise anderer Knoten.<br>**Wofür:** Sprechende Namen helfen. |
| `nodes.1.layer` | Komponente aus Registry 'layer' | `activation` | Auswahl: linear \| conv \| conv_transpose \| depthwise_separable_conv \| pool \| adaptive_pool \| global_pool \| flatten \| reshape \| permute \| dropout \| … | Basis | **Was:** Schicht aus der Registry 'layer'.<br>**Nutzen:** Was dieser Knoten berechnet.<br>**Wofür:** Beliebig. |
| `nodes.1.layer.activation` | Komponente aus Registry 'activation' | `gelu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierungsfunktion.<br>**Nutzen:** Siehe Registry 'activation'.<br>**Wofür:** Überall. |
| `nodes.1.layer.activation.approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |
| `nodes.1.inputs` | Liste[Text] | `[]` |  | Basis | **Was:** Namen der Eingänge/Knoten, die hier hineinfließen (leer = vorheriger Knoten).<br>**Nutzen:** Definiert die Kanten des Graphen.<br>**Wofür:** Verzweigungen, Skip-Verbindungen. |
| `nodes.1.merge` | Komponente aus Registry 'merge' | `concat` | Auswahl: concat \| add \| mul \| mean \| max \| stack \| weighted_sum \| gated \| first | Basis | **Was:** Zusammenführung bei mehreren Eingängen.<br>**Nutzen:** concat, add (Residual), mul, gated ...<br>**Wofür:** Nur bei > 1 Eingang. |
| `nodes.1.merge.dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse, entlang der verkettet wird (-1 = Features, 1 = Kanäle bei CNNs).<br>**Nutzen:** Bestimmt, wie Informationen nebeneinander gelegt werden.<br>**Wofür:** Fusion, DenseNet, U-Net. |
| `nodes.1.repeat` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Schicht n-mal hintereinander (je eigene Gewichte).<br>**Nutzen:** Kompakt für wiederholte Blöcke.<br>**Wofür:** 6× transformer_block. |
| `nodes.1.share_weights_with` | Text (oder None) | `None` |  | Experte | **Was:** Gewichte mit einem früheren Knoten teilen (gleiche Schicht, gleiche Parameter).<br>**Nutzen:** Siamesische Netze, Gewichtsbindung.<br>**Wofür:** Muss formkompatibel sein. |
| `nodes.2` | beliebig | `GraphNodeConfig(name='out', layer={'name': 'linear', 'args':` |  | Basis | **Was:** Ein Knoten des freien Netzes (eine Schicht mit Eingängen).<br>**Nutzen:** Grundbaustein frei definierter Architekturen.<br>**Wofür:** Modell 'graph'. |
| `nodes.2.name` | Text | `'out'` |  | Basis | **Was:** Eindeutiger Name.<br>**Nutzen:** Für Verweise anderer Knoten.<br>**Wofür:** Sprechende Namen helfen. |
| `nodes.2.layer` | Komponente aus Registry 'layer' | `linear` | Auswahl: linear \| conv \| conv_transpose \| depthwise_separable_conv \| pool \| adaptive_pool \| global_pool \| flatten \| reshape \| permute \| dropout \| … | Basis | **Was:** Schicht aus der Registry 'layer'.<br>**Nutzen:** Was dieser Knoten berechnet.<br>**Wofür:** Beliebig. |
| `nodes.2.layer.out_features` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Anzahl Ausgabe-Features.<br>**Nutzen:** Breite der Schicht.<br>**Wofür:** Jede lineare Schicht. |
| `nodes.2.layer.bias` | Ja/Nein | `True` |  | Basis | **Was:** Bias-Vektor addieren.<br>**Nutzen:** Verschiebt die Ausgabe; fast immer sinnvoll.<br>**Wofür:** Standard. |
| `nodes.2.inputs` | Liste[Text] | `[]` |  | Basis | **Was:** Namen der Eingänge/Knoten, die hier hineinfließen (leer = vorheriger Knoten).<br>**Nutzen:** Definiert die Kanten des Graphen.<br>**Wofür:** Verzweigungen, Skip-Verbindungen. |
| `nodes.2.merge` | Komponente aus Registry 'merge' | `concat` | Auswahl: concat \| add \| mul \| mean \| max \| stack \| weighted_sum \| gated \| first | Basis | **Was:** Zusammenführung bei mehreren Eingängen.<br>**Nutzen:** concat, add (Residual), mul, gated ...<br>**Wofür:** Nur bei > 1 Eingang. |
| `nodes.2.merge.dim` | Ganzzahl | `-1` |  | Basis | **Was:** Achse, entlang der verkettet wird (-1 = Features, 1 = Kanäle bei CNNs).<br>**Nutzen:** Bestimmt, wie Informationen nebeneinander gelegt werden.<br>**Wofür:** Fusion, DenseNet, U-Net. |
| `nodes.2.repeat` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Schicht n-mal hintereinander (je eigene Gewichte).<br>**Nutzen:** Kompakt für wiederholte Blöcke.<br>**Wofür:** 6× transformer_block. |
| `nodes.2.share_weights_with` | Text (oder None) | `None` |  | Experte | **Was:** Gewichte mit einem früheren Knoten teilen (gleiche Schicht, gleiche Parameter).<br>**Nutzen:** Siamesische Netze, Gewichtsbindung.<br>**Wofür:** Muss formkompatibel sein. |
| `outputs` | Liste[Text] | `[]` |  | Basis | **Was:** Ausgabeknoten (leer = letzter Knoten; mehrere → dict).<br>**Nutzen:** Multi-Output-Netze.<br>**Wofür:** Multi-Task. |

#### `model:actor_critic` (Aliase: `policy`, `ppo_net`) — Stufe Basis

**Was:** Actor-Critic-Netz (Policy + Value).

**Nutzen:** Für Policy-Gradient-Verfahren.

**Wofür:** PPO, A2C, REINFORCE.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `obs_dim` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Größe der Beobachtung.<br>**Nutzen:** Eingabebreite.<br>**Wofür:** Zustandsvektor der Umgebung. |
| `action_type` | Text | `'discrete'` | Auswahl: discrete \| continuous | Basis | **Was:** Art des Aktionsraums.<br>**Nutzen:** Bestimmt die Verteilung (kategorisch vs. Gauß).<br>**Wofür:** Je nach Umgebung. |
| `action_dim` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Anzahl Aktionen (diskret) bzw. Aktionsdimensionen (kontinuierlich).<br>**Nutzen:** Ausgabegröße der Policy.<br>**Wofür:** Umgebung. |
| `hidden_dims` | Liste[Ganzzahl] | `[64, 64]` | ≥ 1 | Basis | **Was:** Breiten des Rumpfs.<br>**Nutzen:** Kapazität.<br>**Wofür:** PPO-Standard: [64, 64]. |
| `activation` | Komponente aus Registry 'activation' | `tanh` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung.<br>**Nutzen:** tanh ist in PPO-Implementierungen üblich und stabil.<br>**Wofür:** tanh oder relu. |
| `norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung.<br>**Nutzen:** Stabilität.<br>**Wofür:** Selten nötig. |
| `shared_backbone` | Ja/Nein | `False` |  | Basis | **Was:** Actor und Critic teilen den Rumpf.<br>**Nutzen:** Weniger Parameter, aber konkurrierende Gradienten.<br>**Wofür:** Bild-Beobachtungen: an; Vektor: aus. |
| `value_hidden_dims` | Liste[Ganzzahl] (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Eigene Breiten des Critic-Rumpfs (None = wie hidden_dims).<br>**Nutzen:** Asymmetrische Kapazität.<br>**Wofür:** Experimente. |
| `log_std_init` | Kommazahl | `0.0` |  | Fortgeschritten | **Was:** Start-Log-Standardabweichung der Gauß-Policy (kontinuierlich).<br>**Nutzen:** Bestimmt die anfängliche Erkundung (0 → σ = 1).<br>**Wofür:** -0.5 bis 0. |
| `state_dependent_std` | Ja/Nein | `False` |  | Experte | **Was:** Standardabweichung vom Zustand abhängig machen.<br>**Nutzen:** Situationsabhängige Erkundung (SAC-artig).<br>**Wofür:** Komplexe kontinuierliche Steuerung. |
| `squash` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Aktionen mit tanh auf [-1, 1] begrenzen.<br>**Nutzen:** Für Umgebungen mit begrenzten Aktionen.<br>**Wofür:** Robotik. |
| `orthogonal_init` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Orthogonale Initialisierung (Rumpf √2, Policy 0.01, Value 1).<br>**Nutzen:** Bewährte PPO-Praxis – deutlich stabilerer Start.<br>**Wofür:** PPO/A2C. |

#### `model:q_network` (Aliase: `dqn_net`) — Stufe Basis

**Was:** Q-Netz (optional Dueling/Noisy).

**Nutzen:** Für wertbasierte Verfahren.

**Wofür:** DQN und Varianten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `obs_dim` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Größe der Beobachtung.<br>**Nutzen:** Eingabebreite.<br>**Wofür:** Umgebung. |
| `num_actions` | Ganzzahl | `2` | ≥ 1 | Basis | **Was:** Anzahl diskreter Aktionen.<br>**Nutzen:** Ausgabegröße.<br>**Wofür:** Umgebung. |
| `hidden_dims` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Breiten.<br>**Nutzen:** Kapazität.<br>**Wofür:** [128, 128]. |
| `activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Aktivierung.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** relu. |
| `activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung.<br>**Nutzen:** Stabilität.<br>**Wofür:** Selten. |
| `dueling` | Ja/Nein | `True` |  | Basis | **Was:** Dueling-Architektur: Q = V(s) + A(s, a) − mean(A).<br>**Nutzen:** Lernt den Zustandswert unabhängig von Aktionen – schneller bei vielen ähnlichen Aktionen.<br>**Wofür:** Standard in modernen DQNs (Rainbow). |
| `noisy` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Noisy Nets: lernbares Rauschen in den Gewichten statt ε-greedy.<br>**Nutzen:** Zustandsabhängige, gelernte Erkundung.<br>**Wofür:** Rainbow-DQN; dann epsilon auf 0 setzen. |
| `noisy_sigma0` | Kommazahl | `0.5` | ≥ 0.0 | Experte | **Was:** Startstärke des Noisy-Net-Rauschens.<br>**Nutzen:** Mehr = mehr Erkundung.<br>**Wofür:** 0.5. |

#### `model:rnn` (Aliase: `lstm`, `gru`, `recurrent`) — Stufe Basis

**Was:** Rekurrentes Netz (LSTM/GRU/RNN/minGRU), jede Schicht einzeln konfigurierbar.

**Nutzen:** Natürliche Verarbeitung von Strömen mit Gedächtnis, geringer Speicherbedarf bei Inferenz.

**Wofür:** Zeitreihen, Sensorik, Steuerung, kleine Text-Modelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Startwerte der Gewichte, getrennt nach Modultyp.<br>**Nutzen:** Erlaubt bewährte Schemata (GPT-2, ResNet, PPO ...) ohne jede Schicht einzeln anzufassen.<br>**Wofür:** Für einzelne Tensoren/Ausschnitte: Tensor-Chirurgie (tensor_rules) verwenden. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `input_type` | Text | `'continuous'` | Auswahl: continuous \| tokens | Basis | **Was:** Art der Eingabe.<br>**Nutzen:** Tokens bekommen ein Embedding.<br>**Wofür:** Zeitreihen: continuous. |
| `input_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Features pro Zeitschritt (continuous).<br>**Nutzen:** Eingabebreite.<br>**Wofür:** Anzahl Sensoren. |
| `vocab_size` | Ganzzahl | `1000` | ≥ 1 | Basis | **Was:** Vokabular (tokens).<br>**Nutzen:** Anzahl Symbole.<br>**Wofür:** Zeichen: ~100. |
| `embedding_dim` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Embedding-Größe (tokens).<br>**Nutzen:** Repräsentation der Symbole.<br>**Wofür:** 32–512. |
| `hidden_sizes` | Liste[Ganzzahl] | `[128, 128]` | ≥ 1 | Basis | **Was:** Zustandsgröße jeder Schicht (Länge = Anzahl Schichten).<br>**Nutzen:** Gedächtniskapazität pro Schicht.<br>**Wofür:** 1–3 Schichten à 64–512. |
| `layer` | Unter-Konfiguration Rekurrente Schicht | ⟶ RecurrentLayerConfig |  | Basis | **Was:** Unter-Konfiguration 'Rekurrente Schicht'. |
| `layer.cell` | Text | `'lstm'` | Auswahl: lstm \| gru \| rnn \| mingru | Basis | **Was:** Zellart.<br>**Nutzen:** Bestimmt Gedächtnisfähigkeit und Kosten.<br>**Wofür:** LSTM/GRU sind sichere Standards. |
| `layer.bidirectional` | Ja/Nein | `False` |  | Basis | **Was:** Zusätzlich rückwärts verarbeiten.<br>**Nutzen:** Jeder Zeitschritt kennt Vergangenheit UND Zukunft.<br>**Wofür:** Klassifikation, Tagging. Nicht für Generierung/Echtzeit! |
| `layer.nonlinearity` | Text | `'tanh'` | Auswahl: tanh \| relu | Experte | **Was:** Aktivierung des einfachen RNN.<br>**Nutzen:** relu kann explodieren, tanh sättigt.<br>**Wofür:** Nur cell='rnn'. |
| `layer.proj_size` | Ganzzahl | `0` | ≥ 0 | Experte | **Was:** Projektion des LSTM-Zustands auf kleinere Größe (0 = aus).<br>**Nutzen:** Spart Parameter in großen LSTMs (LSTMP).<br>**Wofür:** Spracherkennung. |
| `layer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Ausgabe der Schicht.<br>**Nutzen:** Regularisierung zwischen Schichten.<br>**Wofür:** 0.1–0.5. |
| `layer.norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung nach der Schicht.<br>**Nutzen:** Stabilisiert tiefe RNN-Stapel.<br>**Wofür:** layernorm bei > 2 Schichten. |
| `layer.residual` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Residualverbindung um die Schicht (bei gleicher Breite).<br>**Nutzen:** Bessere Gradienten in tiefen Stapeln (GNMT).<br>**Wofür:** ≥ 4 Schichten. |
| `layer.forget_bias` | Kommazahl (oder None) | `None` |  | Experte | **Was:** Startwert des LSTM-Forget-Gate-Bias (None = PyTorch-Standard 0).<br>**Nutzen:** 1.0 lässt das LSTM anfangs eher erinnern – klassischer Trick für lange Abhängigkeiten.<br>**Wofür:** LSTMs auf langen Sequenzen. |
| `layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Abweichungen einzelner Schichten (z. B. andere Zellart).<br>**Nutzen:** Jede Schicht individuell.<br>**Wofür:** {'0': {'bidirectional': True}, '-1': {'cell': 'gru'}} |
| `pooling` | Text | `'last'` | Auswahl: none \| last \| mean \| max \| attention | Basis | **Was:** Verdichtung über die Zeit.<br>**Nutzen:** Sequenz → Vektor für sequenzweise Vorhersagen.<br>**Wofür:** Klassifikation: last/attention; Tagging/LM: none. |
| `output_dim` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Ausgabegröße.<br>**Nutzen:** Klassen oder Zielgrößen.<br>**Wofür:** Je nach Aufgabe. |
| `head_hidden_dims` | Liste[Ganzzahl] | `[]` | ≥ 1 | Fortgeschritten | **Was:** Versteckte Schichten des Ausgabekopfs.<br>**Nutzen:** Mehr Kapazität im Kopf.<br>**Wofür:** Meist leer. |
| `head_activation` | Komponente aus Registry 'activation' | `relu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Aktivierung im Kopf.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** Mit head_hidden_dims. |
| `head_activation.inplace` | Ja/Nein | `False` |  | Experte | **Was:** Ergebnis direkt in den Eingabespeicher schreiben.<br>**Nutzen:** Spart etwas Speicher, kann aber mit Autograd kollidieren, wenn die Eingabe noch gebraucht wird.<br>**Wofür:** Speicheroptimierung bei sehr großen Aktivierungen.<br>**Tipp:** Im Zweifel aus lassen – Fehler äußern sich als RuntimeError beim Backward. |
| `embedding_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach dem Embedding / auf die Eingabe.<br>**Nutzen:** Regularisierung.<br>**Wofür:** Text: 0.1–0.3. |

#### `model:transformer` (Aliase: `gpt`, `bert`, `vit`) — Stufe Basis

**Was:** Transformer (Decoder / Encoder / Encoder-Decoder) für Text, Bilder und Zeitreihen.

**Nutzen:** Extrem vielseitig; jede Schicht einzeln konfigurierbar, Mixer austauschbar.

**Wofür:** Sprachmodelle, Klassifikation, Übersetzung, ViT, Prognosen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Initialisierung (Transformer-Standard: Embeddings ~ N(0, 0.02²), Biases 0).<br>**Nutzen:** Kleine Embeddings halten die Logits gebundener Ausgabegewichte anfangs klein → stabiler Start.<br>**Wofür:** GPT-2-Schema zusätzlich: linear_weight normal(0.02) + residual_projection scaled_normal. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `zeros` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `normal` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.embedding.mean` | Kommazahl | `0.0` |  | Fortgeschritten | **Was:** Mittelwert der Normalverteilung.<br>**Nutzen:** Verschiebt alle Startwerte.<br>**Wofür:** Fast immer 0. |
| `init.embedding.std` | Kommazahl | `0.02` | ≥ 0.0 | Fortgeschritten | **Was:** Standardabweichung.<br>**Nutzen:** Bestimmt die Startgröße der Gewichte.<br>**Wofür:** Transformer: 0.02 (GPT/BERT-Konvention). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `architecture` | Text | `'decoder_only'` | Auswahl: decoder_only \| encoder_only \| encoder_decoder | Basis | **Was:** Grundarchitektur.<br>**Nutzen:** Bestimmt Maskierung, Datenfluss und Einsatzzweck.<br>**Wofür:** Siehe Optionen. |
| `input_type` | Text | `'tokens'` | Auswahl: tokens \| continuous \| patches | Basis | **Was:** Art der Eingabedaten.<br>**Nutzen:** Bestimmt die Eingangsschicht.<br>**Wofür:** Text: tokens; Zeitreihen: continuous. |
| `vocab_size` | Ganzzahl | `1000` | ≥ 1 | Basis | **Was:** Vokabulargröße (nur tokens).<br>**Nutzen:** Anzahl unterscheidbarer Tokens.<br>**Wofür:** Zeichen-Modelle: ~100; BPE: 32k–256k. |
| `input_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Features pro Zeitschritt (nur continuous).<br>**Nutzen:** Breite der Eingabevektoren.<br>**Wofür:** Anzahl Sensoren/Variablen. |
| `image_channels` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Bildkanäle (nur patches).<br>**Nutzen:** RGB = 3.<br>**Wofür:** ViT. |
| `image_size` | Liste[Ganzzahl] | `[32, 32]` | ≥ 1 · Länge 2 | Basis | **Was:** Bildgröße [H, W] (nur patches).<br>**Nutzen:** Bestimmt Anzahl der Patches.<br>**Wofür:** ViT. |
| `patch_size` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Kantenlänge eines Patches (nur patches).<br>**Nutzen:** Kleinere Patches = mehr Tokens = feiner, aber quadratisch teurer.<br>**Wofür:** ViT-B/16: 16; kleine Bilder: 4. |
| `output_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Ausgabegröße. None = vocab_size (Sprachmodell) bzw. Pflicht bei anderen Aufgaben.<br>**Nutzen:** Klassen bei Klassifikation, Zielgrößen bei Regression.<br>**Wofür:** Klassifikation mit pooling ≠ none: Klassenzahl. |
| `d_model` | Ganzzahl | `128` | ≥ 1 | Basis | **Was:** Modellbreite (Größe der Repräsentation jedes Tokens).<br>**Nutzen:** Hauptregler der Kapazität (Parameter wachsen ~quadratisch).<br>**Wofür:** Klein 128–512, mittel 768–2048, groß 4096+. |
| `n_layers` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Anzahl Blöcke (beim encoder_decoder: Decoder-Blöcke).<br>**Nutzen:** Mehr Tiefe = abstraktere Verarbeitung.<br>**Wofür:** Klein 2–6, mittel 12–24, groß 32–80+. |
| `n_encoder_layers` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Encoder-Blöcke (nur encoder_decoder; None = n_layers).<br>**Nutzen:** Getrennte Tiefe für Lesen/Schreiben.<br>**Wofür:** T5: gleich. |
| `max_seq_len` | Ganzzahl | `256` | ≥ 1 | Basis | **Was:** Maximale Sequenzlänge.<br>**Nutzen:** Größe von Positionstabellen; längere Eingaben nicht möglich (außer RoPE/ALiBi).<br>**Wofür:** Kontextlänge der Anwendung. |
| `block` | Unter-Konfiguration Transformer-Block | ⟶ TransformerBlockConfig |  | Basis | **Was:** Bauplan eines Transformer-Blocks (Mixer + FFN mit Residualverbindungen).<br>**Nutzen:** Gilt für alle Schichten; einzelne Schichten weichen über layer_overrides ab.<br>**Wofür:** Jede Transformer-Variante. |
| `block.mixer` | Komponente aus Registry 'sequence_mixer' | `attention` | Auswahl: attention \| ssm \| rnn \| conv \| fourier \| none | Basis | **Was:** Mixer: tauscht Information zwischen Positionen aus (Attention, SSM, RNN, Faltung ...).<br>**Nutzen:** Bestimmt Kosten und Reichweite; pro Schicht wählbar → Hybride.<br>**Wofür:** Standard: attention. Lange Sequenzen: ssm. Hybride über layer_overrides.<br>**Tipp:** Bei 'attention' liegen die Attention-Einstellungen direkt unter mixer.* (z. B. mixer.num_heads). |
| `block.mixer.num_heads` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `block.mixer.num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `block.mixer.head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `block.mixer.kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Basis | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `block.mixer.kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Basis | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `block.mixer.kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `block.mixer.kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `block.mixer.kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `block.mixer.kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `block.mixer.causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `block.mixer.window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `block.mixer.scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `block.mixer.qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `block.mixer.qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `block.mixer.out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `block.mixer.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `block.mixer.head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `block.mixer.output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `block.mixer.kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `block.mixer.kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `block.mixer.kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `block.mixer.kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `block.mixer.kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `block.mixer.kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `block.cross_attention` | Unter-Konfiguration Attention | ⟶ AttentionConfig |  | Fortgeschritten | **Was:** Cross-Attention der Decoder-Blöcke (nur encoder_decoder).<br>**Nutzen:** Lässt den Decoder auf die Encoder-Ausgabe schauen.<br>**Wofür:** Übersetzung, Zusammenfassung, Seq2Seq. |
| `block.cross_attention.num_heads` | Ganzzahl | `8` | ≥ 1 | Fortgeschritten | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `block.cross_attention.num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `block.cross_attention.head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `block.cross_attention.kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Fortgeschritten | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `block.cross_attention.kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Fortgeschritten | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `block.cross_attention.kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `block.cross_attention.kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `block.cross_attention.kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `block.cross_attention.kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `block.cross_attention.causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `block.cross_attention.window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `block.cross_attention.scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `block.cross_attention.qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `block.cross_attention.qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `block.cross_attention.out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `block.cross_attention.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `block.cross_attention.head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `block.cross_attention.output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `block.cross_attention.kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `block.cross_attention.kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `block.cross_attention.kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `block.cross_attention.kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `block.cross_attention.kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `block.cross_attention.kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `block.ffn` | Komponente aus Registry 'feedforward' | `glu` | Auswahl: mlp \| glu \| moe \| kan \| none | Basis | **Was:** Feed-Forward-Teil (mlp, glu/SwiGLU, moe, none).<br>**Nutzen:** Positionsweise Verarbeitung – hier steckt die meiste Kapazität.<br>**Wofür:** glu (SwiGLU) ist moderner Standard; moe für sehr große Kapazität. |
| `block.ffn.hidden_mult` | Kommazahl | `2.6666666666666665` | ≥ 0.01 | Basis | **Was:** Breite relativ zu d_model (GLU hat 3 Matrizen → Standard 8/3 hält die Parameter wie MLP mit 4).<br>**Nutzen:** Mehr Breite = mehr Kapazität.<br>**Wofür:** LLaMA: 8/3 (aufgerundet auf multiple_of). |
| `block.ffn.hidden_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Exakte Breite der versteckten Schicht (überschreibt hidden_mult).<br>**Nutzen:** Für exakte Nachbauten bekannter Architekturen.<br>**Wofür:** Gewichte-Import, exakte Parameterzahlen. |
| `block.ffn.multiple_of` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Versteckte Breite auf ein Vielfaches davon aufrunden.<br>**Nutzen:** Vielfache von 64/256 rechnen auf GPUs schneller (Tensor Cores).<br>**Wofür:** Geschwindigkeitsoptimierung (LLaMA: 256). |
| `block.ffn.activation` | Komponente aus Registry 'activation' | `silu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Gate-Aktivierung: silu → SwiGLU, gelu → GeGLU, sigmoid → GLU, relu → ReGLU, identity → Bilinear.<br>**Nutzen:** Das Gate entscheidet elementweise, welche Information durchgelassen wird.<br>**Wofür:** SwiGLU ist Standard moderner LLMs. |
| `block.ffn.bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den linearen Schichten.<br>**Nutzen:** Moderne LLMs verzichten darauf.<br>**Wofür:** Kompatibilität. |
| `block.ffn.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach der Aktivierung (innerhalb des FFN).<br>**Nutzen:** Regularisiert die versteckte Schicht.<br>**Wofür:** Kleine Datensätze. |
| `block.norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung in den Residual-Zweigen.<br>**Nutzen:** Stabilisiert das Training.<br>**Wofür:** rmsnorm (modern), layernorm (klassisch). |
| `block.norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `block.norm.learnable` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `block.norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `block.norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `block.norm_position` | Text | `'pre'` | Auswahl: pre \| post \| sandwich \| none | Fortgeschritten | **Was:** Position der Normalisierung relativ zur Residualverbindung.<br>**Nutzen:** Großer Einfluss auf Trainingsstabilität und Endqualität.<br>**Wofür:** pre für tiefe Modelle; post für kleine Modelle mit Warmup. |
| `block.parallel` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Mixer und FFN parallel statt nacheinander (x + Mixer(n(x)) + FFN(n(x))).<br>**Nutzen:** ~15 % schneller bei großen Modellen, minimal andere Qualität.<br>**Wofür:** GPT-J, PaLM. |
| `block.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Ausgabe jedes Zweigs (Residual-Dropout).<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Datensätze: 0.1; große LLMs: 0. |
| `block.drop_path` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Stochastic Depth: Wahrscheinlichkeit, einen ganzen Zweig auszulassen.<br>**Nutzen:** Starke Regularisierung, besonders für Vision-Transformer.<br>**Wofür:** ViT/DeiT: 0.1–0.3 (mit linearem Anstieg über die Tiefe). |
| `block.layer_scale_init` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** LayerScale: lernbarer Faktor pro Kanal auf jedem Zweig (Startwert; None = aus).<br>**Nutzen:** Kleiner Start (1e-5…1e-1) lässt tiefe Netze zunächst wie flache trainieren.<br>**Wofür:** Sehr tiefe Transformer (CaiT, DINOv2). |
| `block.branch_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf jeder Zweig-Ausgabe.<br>**Nutzen:** Dämpft Residual-Beiträge in tiefen Netzen.<br>**Wofür:** Experimente; z. B. 1/√(2N). |
| `block.skip_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf dem Skip-Pfad (DeepNet: α > 1).<br>**Nutzen:** Stärkerer Skip stabilisiert Post-LN-Netze mit hunderten Schichten.<br>**Wofür:** DeepNorm-Experimente. |
| `encoder_block` | Unter-Konfiguration Transformer-Block | None |  | Fortgeschritten | **Was:** Abweichender Block-Bauplan für den Encoder (None = wie block).<br>**Nutzen:** Encoder und Decoder können völlig verschieden sein.<br>**Wofür:** encoder_decoder. |
| `layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Schichtweise Abweichungen des (Decoder-)Stapels: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede einzelne Schicht kann anders gebaut sein.<br>**Wofür:** Hybride, lokale/globale Attention im Wechsel, MoE nur in manchen Schichten.<br>**Tipp:** Beispiel: {'1::2': {'ffn': 'moe'}, '-1': {'mixer.num_heads': 16}} |
| `encoder_layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Schichtweise Abweichungen des Encoders.<br>**Nutzen:** Wie layer_overrides.<br>**Wofür:** encoder_decoder. |
| `drop_path_schedule` | Text | `'linear'` | Auswahl: constant \| linear | Experte | **Was:** Verteilung von drop_path über die Tiefe.<br>**Nutzen:** Frühe Schichten seltener auslassen.<br>**Wofür:** Bei drop_path > 0. |
| `positional` | Komponente aus Registry 'positional' | `rope` | Auswahl: none \| sinusoidal \| learned \| rope \| alibi \| relative_bias \| combined | Basis | **Was:** Positionskodierung.<br>**Nutzen:** Gibt dem Modell Reihenfolge-Information.<br>**Wofür:** rope (Text), learned (ViT), sinusoidal. |
| `positional.theta` | Kommazahl | `10000.0` | ≥ 1.0 | Basis | **Was:** Basisfrequenz θ der Rotationen.<br>**Nutzen:** Größeres θ → längere Kontexte unterscheidbar (LLaMA-3: 500000).<br>**Wofür:** Lange Kontexte → θ erhöhen. |
| `positional.fraction` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil der Kopf-Dimensionen, die rotiert werden ('partial rotary').<br>**Nutzen:** Rest bleibt positionsfrei – manche Modelle (GPT-NeoX: 0.25) profitieren davon.<br>**Wofür:** Feintuning von Positions- vs. Inhaltsanteil. |
| `positional.scaling` | Text | `'none'` | Auswahl: none \| linear \| ntk \| yarn | Fortgeschritten | **Was:** Verfahren zur Kontextverlängerung.<br>**Nutzen:** Erlaubt Sequenzen länger als im Training.<br>**Wofür:** Kontext-Erweiterung vortrainierter Modelle. |
| `positional.factor` | Kommazahl | `1.0` | ≥ 1.0 | Fortgeschritten | **Was:** Erweiterungsfaktor des Kontexts (z. B. 4 = vierfache Länge).<br>**Nutzen:** Stärke der Skalierung.<br>**Wofür:** Nur mit scaling ≠ none. |
| `positional.original_context` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** Kontextlänge, mit der ursprünglich trainiert wurde (nur YaRN; None = max_seq_len / factor).<br>**Nutzen:** YaRN entscheidet anhand dieser Länge, welche Frequenzen interpoliert werden.<br>**Wofür:** Vortrainiertes Modell mit 4096 Tokens auf 16k erweitern: 4096, factor 4. |
| `positional.yarn_beta_fast` | Kommazahl | `32.0` | ≥ 0.0 | Experte | **Was:** YaRN β_fast: ab so vielen Umdrehungen im Originalkontext bleibt eine Frequenz unverändert.<br>**Nutzen:** Schützt die lokale Positionsauflösung.<br>**Wofür:** 32 (Paper). |
| `positional.yarn_beta_slow` | Kommazahl | `1.0` | ≥ 0.0 | Experte | **Was:** YaRN β_slow: unter so vielen Umdrehungen wird voll interpoliert.<br>**Nutzen:** Niedrige Frequenzen.<br>**Wofür:** 1 (Paper). |
| `positional.interleaved` | Ja/Nein | `False` |  | Experte | **Was:** Paare benachbarter Dimensionen rotieren (GPT-J) statt Hälften (GPT-NeoX/LLaMA).<br>**Nutzen:** Nur für Kompatibilität mit fremden Gewichten relevant.<br>**Wofür:** Gewichte-Import. |
| `embedding_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach dem Embedding.<br>**Nutzen:** Regularisiert Eingaberepräsentationen.<br>**Wofür:** Kleine Datensätze. |
| `embedding_scale` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Embeddings mit √d_model multiplizieren (None = automatisch: an bei 'sinusoidal').<br>**Nutzen:** Balanciert Token- gegen Positionssignal: ohne Skalierung gehen kleine Token-Embeddings im festen Sinus-Signal (Amplitude 1) unter (Original-Transformer skaliert deshalb).<br>**Wofür:** Sinusoidale Positionen, Gemma. |
| `embedding_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung direkt nach dem Embedding.<br>**Nutzen:** Stabilisiert sehr große Vokabulare (BLOOM).<br>**Wofür:** Selten. |
| `tie_embeddings` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Ausgabe-Gewichte = Embedding-Gewichte (Weight Tying).<br>**Nutzen:** Spart vocab·d Parameter und verbessert oft kleine Sprachmodelle.<br>**Wofür:** Sprachmodelle (nur wenn output_dim = vocab_size und kein Kopf-MLP). |
| `pad_token_id` | Ganzzahl (oder None) | `None` |  | Fortgeschritten | **Was:** Padding-Token: daraus wird automatisch die Padding-Maske erzeugt.<br>**Nutzen:** Padding wird von Attention ignoriert.<br>**Wofür:** Batches mit unterschiedlich langen Texten. |
| `num_register_tokens` | Ganzzahl | `0` | ≥ 0 | Experimentell | **Was:** Zusätzliche lernbare 'Register'-Tokens am Sequenzanfang.<br>**Nutzen:** Geben dem Modell Notizplatz; reduziert Artefakte in Attention-Karten (ViT-Register).<br>**Wofür:** Vision Transformer, Forschung. |
| `final_norm` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Normalisierung nach dem letzten Block.<br>**Nutzen:** Bei Pre-Norm nötig, damit die Ausgabe skaliert ist.<br>**Wofür:** Pre-Norm: an. |
| `pooling` | Text | `'none'` | Auswahl: none \| cls \| mean \| max \| last | Basis | **Was:** Verdichtung zu einem Vektor pro Sequenz.<br>**Nutzen:** Für sequenzweise Vorhersagen.<br>**Wofür:** Klassifikation: cls/mean; Decoder: last. |
| `head_hidden_dims` | Liste[Ganzzahl] | `[]` | ≥ 1 | Fortgeschritten | **Was:** Versteckte Schichten des Ausgabekopfs.<br>**Nutzen:** Mehr Kapazität im Kopf.<br>**Wofür:** Meist leer. |
| `head_activation` | Komponente aus Registry 'activation' | `gelu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Aktivierung im Kopf-MLP.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** Nur mit head_hidden_dims. |
| `head_activation.approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |
| `head_bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Sprachmodelle meist ohne.<br>**Wofür:** Klassifikation: an. |
| `logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Ausgabe-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert überkonfidente Vorhersagen und Instabilität.<br>**Wofür:** Gemma 2: 30. |

#### `model:ssm` (Aliase: `mamba`) — Stufe Fortgeschritten

**Was:** Mamba-artiges State-Space-Modell.

**Nutzen:** Linear skalierende Sequenzverarbeitung.

**Wofür:** Lange Sequenzen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Initialisierung (Transformer-Standard: Embeddings ~ N(0, 0.02²), Biases 0).<br>**Nutzen:** Kleine Embeddings halten die Logits gebundener Ausgabegewichte anfangs klein → stabiler Start.<br>**Wofür:** GPT-2-Schema zusätzlich: linear_weight normal(0.02) + residual_projection scaled_normal. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `zeros` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `normal` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.embedding.mean` | Kommazahl | `0.0` |  | Fortgeschritten | **Was:** Mittelwert der Normalverteilung.<br>**Nutzen:** Verschiebt alle Startwerte.<br>**Wofür:** Fast immer 0. |
| `init.embedding.std` | Kommazahl | `0.02` | ≥ 0.0 | Fortgeschritten | **Was:** Standardabweichung.<br>**Nutzen:** Bestimmt die Startgröße der Gewichte.<br>**Wofür:** Transformer: 0.02 (GPT/BERT-Konvention). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `architecture` | Text | `'decoder_only'` | Auswahl: decoder_only \| encoder_only \| encoder_decoder | Basis | **Was:** Grundarchitektur.<br>**Nutzen:** Bestimmt Maskierung, Datenfluss und Einsatzzweck.<br>**Wofür:** Siehe Optionen. |
| `input_type` | Text | `'tokens'` | Auswahl: tokens \| continuous \| patches | Basis | **Was:** Art der Eingabedaten.<br>**Nutzen:** Bestimmt die Eingangsschicht.<br>**Wofür:** Text: tokens; Zeitreihen: continuous. |
| `vocab_size` | Ganzzahl | `1000` | ≥ 1 | Basis | **Was:** Vokabulargröße (nur tokens).<br>**Nutzen:** Anzahl unterscheidbarer Tokens.<br>**Wofür:** Zeichen-Modelle: ~100; BPE: 32k–256k. |
| `input_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Features pro Zeitschritt (nur continuous).<br>**Nutzen:** Breite der Eingabevektoren.<br>**Wofür:** Anzahl Sensoren/Variablen. |
| `image_channels` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Bildkanäle (nur patches).<br>**Nutzen:** RGB = 3.<br>**Wofür:** ViT. |
| `image_size` | Liste[Ganzzahl] | `[32, 32]` | ≥ 1 · Länge 2 | Basis | **Was:** Bildgröße [H, W] (nur patches).<br>**Nutzen:** Bestimmt Anzahl der Patches.<br>**Wofür:** ViT. |
| `patch_size` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Kantenlänge eines Patches (nur patches).<br>**Nutzen:** Kleinere Patches = mehr Tokens = feiner, aber quadratisch teurer.<br>**Wofür:** ViT-B/16: 16; kleine Bilder: 4. |
| `output_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Ausgabegröße. None = vocab_size (Sprachmodell) bzw. Pflicht bei anderen Aufgaben.<br>**Nutzen:** Klassen bei Klassifikation, Zielgrößen bei Regression.<br>**Wofür:** Klassifikation mit pooling ≠ none: Klassenzahl. |
| `d_model` | Ganzzahl | `128` | ≥ 1 | Basis | **Was:** Modellbreite (Größe der Repräsentation jedes Tokens).<br>**Nutzen:** Hauptregler der Kapazität (Parameter wachsen ~quadratisch).<br>**Wofür:** Klein 128–512, mittel 768–2048, groß 4096+. |
| `n_layers` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Anzahl Blöcke (beim encoder_decoder: Decoder-Blöcke).<br>**Nutzen:** Mehr Tiefe = abstraktere Verarbeitung.<br>**Wofür:** Klein 2–6, mittel 12–24, groß 32–80+. |
| `n_encoder_layers` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Encoder-Blöcke (nur encoder_decoder; None = n_layers).<br>**Nutzen:** Getrennte Tiefe für Lesen/Schreiben.<br>**Wofür:** T5: gleich. |
| `max_seq_len` | Ganzzahl | `256` | ≥ 1 | Basis | **Was:** Maximale Sequenzlänge.<br>**Nutzen:** Größe von Positionstabellen; längere Eingaben nicht möglich (außer RoPE/ALiBi).<br>**Wofür:** Kontextlänge der Anwendung. |
| `block` | Unter-Konfiguration Transformer-Block | ⟶ TransformerBlockConfig |  | Basis | **Was:** Bauplan eines Transformer-Blocks (Mixer + FFN mit Residualverbindungen).<br>**Nutzen:** Gilt für alle Schichten; einzelne Schichten weichen über layer_overrides ab.<br>**Wofür:** Jede Transformer-Variante. |
| `block.mixer` | Komponente aus Registry 'sequence_mixer' | `ssm` | Auswahl: attention \| ssm \| rnn \| conv \| fourier \| none | Basis | **Was:** Mixer: tauscht Information zwischen Positionen aus (Attention, SSM, RNN, Faltung ...).<br>**Nutzen:** Bestimmt Kosten und Reichweite; pro Schicht wählbar → Hybride.<br>**Wofür:** Standard: attention. Lange Sequenzen: ssm. Hybride über layer_overrides.<br>**Tipp:** Bei 'attention' liegen die Attention-Einstellungen direkt unter mixer.* (z. B. mixer.num_heads). |
| `block.mixer.d_state` | Ganzzahl | `16` | ≥ 1 | Fortgeschritten | **Was:** Größe des inneren Zustands pro Kanal (N).<br>**Nutzen:** Mehr Zustand = mehr Gedächtniskapazität, mehr Rechenaufwand.<br>**Wofür:** Mamba: 16. |
| `block.mixer.expand` | Kommazahl | `2.0` | ≥ 0.25 | Fortgeschritten | **Was:** Aufweitung der inneren Breite (E = expand · d_model).<br>**Nutzen:** Kapazität des Blocks.<br>**Wofür:** Mamba: 2. |
| `block.mixer.d_conv` | Ganzzahl | `4` | ≥ 1 | Fortgeschritten | **Was:** Kernelgröße der kurzen kausalen Faltung vor dem Scan.<br>**Nutzen:** Erfasst lokale Muster, bevor der Zustand aktualisiert wird.<br>**Wofür:** Mamba: 4. |
| `block.mixer.dt_rank` | Ganzzahl oder Text | `'auto'` |  | Experte | **Was:** Rang der Δ-Projektion ('auto' = ceil(d_model/16)).<br>**Nutzen:** Steuert, wie fein die Schrittweite Δ von der Eingabe abhängen kann.<br>**Wofür:** Selten zu ändern. |
| `block.mixer.dt_min` | Kommazahl | `0.001` | ≥ 1e-06 | Experte | **Was:** Kleinste initiale Schrittweite Δ.<br>**Nutzen:** Kleine Δ = langes Gedächtnis.<br>**Wofür:** Mamba: 0.001. |
| `block.mixer.dt_max` | Kommazahl | `0.1` | ≥ 1e-06 | Experte | **Was:** Größte initiale Schrittweite Δ.<br>**Nutzen:** Große Δ = Fokus auf aktuelle Eingabe.<br>**Wofür:** Mamba: 0.1. |
| `block.mixer.a_init` | Text | `'s4d_real'` | Auswahl: s4d_real \| s4d_lin \| random | Experte | **Was:** Initialisierung der Zustandsmatrix A.<br>**Nutzen:** Bestimmt Start-Gedächtnis-Zeitskalen.<br>**Wofür:** Forschung. |
| `block.mixer.bidirectional` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Zusätzlicher Rückwärts-Scan (wird bei kausalen Modellen ignoriert).<br>**Nutzen:** Kontext von beiden Seiten für Encoder-Aufgaben.<br>**Wofür:** Klassifikation, Tagging. |
| `block.mixer.conv_bias` | Ja/Nein | `True` |  | Experte | **Was:** Bias der kurzen Faltung.<br>**Nutzen:** Verschiebung.<br>**Wofür:** Standard. |
| `block.mixer.bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias der Ein-/Ausgangsprojektion.<br>**Nutzen:** Verschiebung.<br>**Wofür:** Mamba: aus. |
| `block.mixer.scan_mode` | Text | `'auto'` | Auswahl: auto \| parallel \| chunked \| sequential \| mamba_ssm | Fortgeschritten | **Was:** Algorithmus für den Zustands-Scan h_t = exp(Δ·A)·h_{t-1} + Δ·B·x.<br>**Nutzen:** Der parallele/chunked Scan braucht nur log₂(T) Runden statt T Schleifenschritte – Training oft 10–40× schneller als die Python-Schleife.<br>**Wofür:** auto ist fast immer richtig (CPU: fusionierte Schleife, GPU: chunked); mamba_ssm auf NVIDIA-GPUs mit installiertem Paket. |
| `block.mixer.scan_chunk_size` | Ganzzahl | `32` | ≥ 1 | Fortgeschritten | **Was:** Blocklänge des chunked-Scans.<br>**Nutzen:** Größer = weniger sequenzielle Überträge, aber mehr Zwischenspeicher pro Runde.<br>**Wofür:** CPU: 16–32; GPU: 64–256. |
| `block.mixer.scan_backward` | Text | `'fused'` | Auswahl: fused \| scan \| autograd | Experte | **Was:** Wie der Gradient des Scans berechnet wird.<br>**Nutzen:** 'fused' senkt den Aktivierungsspeicher des Scans etwa um den Faktor d_state – wie der offizielle Mamba-Kernel, aber in reinem PyTorch (CPU, GPU, MPS).<br>**Wofür:** Lange Sequenzen / große Batches: fused. Kurze Sequenzen mit viel Speicher: scan. |
| `block.cross_attention` | Unter-Konfiguration Attention | ⟶ AttentionConfig |  | Fortgeschritten | **Was:** Cross-Attention der Decoder-Blöcke (nur encoder_decoder).<br>**Nutzen:** Lässt den Decoder auf die Encoder-Ausgabe schauen.<br>**Wofür:** Übersetzung, Zusammenfassung, Seq2Seq. |
| `block.cross_attention.num_heads` | Ganzzahl | `8` | ≥ 1 | Fortgeschritten | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `block.cross_attention.num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `block.cross_attention.head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `block.cross_attention.kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Fortgeschritten | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `block.cross_attention.kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Fortgeschritten | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `block.cross_attention.kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `block.cross_attention.kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `block.cross_attention.kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `block.cross_attention.kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `block.cross_attention.causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `block.cross_attention.window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `block.cross_attention.scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `block.cross_attention.qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `block.cross_attention.qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `block.cross_attention.out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `block.cross_attention.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `block.cross_attention.head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `block.cross_attention.output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `block.cross_attention.kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `block.cross_attention.kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `block.cross_attention.kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `block.cross_attention.kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `block.cross_attention.kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `block.cross_attention.kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `block.ffn` | Komponente aus Registry 'feedforward' | `none` | Auswahl: mlp \| glu \| moe \| kan \| none | Basis | **Was:** Feed-Forward-Teil (mlp, glu/SwiGLU, moe, none).<br>**Nutzen:** Positionsweise Verarbeitung – hier steckt die meiste Kapazität.<br>**Wofür:** glu (SwiGLU) ist moderner Standard; moe für sehr große Kapazität. |
| `block.norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung in den Residual-Zweigen.<br>**Nutzen:** Stabilisiert das Training.<br>**Wofür:** rmsnorm (modern), layernorm (klassisch). |
| `block.norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `block.norm.learnable` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `block.norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `block.norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `block.norm_position` | Text | `'pre'` | Auswahl: pre \| post \| sandwich \| none | Fortgeschritten | **Was:** Position der Normalisierung relativ zur Residualverbindung.<br>**Nutzen:** Großer Einfluss auf Trainingsstabilität und Endqualität.<br>**Wofür:** pre für tiefe Modelle; post für kleine Modelle mit Warmup. |
| `block.parallel` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Mixer und FFN parallel statt nacheinander (x + Mixer(n(x)) + FFN(n(x))).<br>**Nutzen:** ~15 % schneller bei großen Modellen, minimal andere Qualität.<br>**Wofür:** GPT-J, PaLM. |
| `block.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Ausgabe jedes Zweigs (Residual-Dropout).<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Datensätze: 0.1; große LLMs: 0. |
| `block.drop_path` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Stochastic Depth: Wahrscheinlichkeit, einen ganzen Zweig auszulassen.<br>**Nutzen:** Starke Regularisierung, besonders für Vision-Transformer.<br>**Wofür:** ViT/DeiT: 0.1–0.3 (mit linearem Anstieg über die Tiefe). |
| `block.layer_scale_init` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** LayerScale: lernbarer Faktor pro Kanal auf jedem Zweig (Startwert; None = aus).<br>**Nutzen:** Kleiner Start (1e-5…1e-1) lässt tiefe Netze zunächst wie flache trainieren.<br>**Wofür:** Sehr tiefe Transformer (CaiT, DINOv2). |
| `block.branch_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf jeder Zweig-Ausgabe.<br>**Nutzen:** Dämpft Residual-Beiträge in tiefen Netzen.<br>**Wofür:** Experimente; z. B. 1/√(2N). |
| `block.skip_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf dem Skip-Pfad (DeepNet: α > 1).<br>**Nutzen:** Stärkerer Skip stabilisiert Post-LN-Netze mit hunderten Schichten.<br>**Wofür:** DeepNorm-Experimente. |
| `encoder_block` | Unter-Konfiguration Transformer-Block | None |  | Fortgeschritten | **Was:** Abweichender Block-Bauplan für den Encoder (None = wie block).<br>**Nutzen:** Encoder und Decoder können völlig verschieden sein.<br>**Wofür:** encoder_decoder. |
| `layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Schichtweise Abweichungen des (Decoder-)Stapels: {selektor: {einstellung: wert}}.<br>**Nutzen:** Jede einzelne Schicht kann anders gebaut sein.<br>**Wofür:** Hybride, lokale/globale Attention im Wechsel, MoE nur in manchen Schichten.<br>**Tipp:** Beispiel: {'1::2': {'ffn': 'moe'}, '-1': {'mixer.num_heads': 16}} |
| `encoder_layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Schichtweise Abweichungen des Encoders.<br>**Nutzen:** Wie layer_overrides.<br>**Wofür:** encoder_decoder. |
| `drop_path_schedule` | Text | `'linear'` | Auswahl: constant \| linear | Experte | **Was:** Verteilung von drop_path über die Tiefe.<br>**Nutzen:** Frühe Schichten seltener auslassen.<br>**Wofür:** Bei drop_path > 0. |
| `positional` | Komponente aus Registry 'positional' | `none` | Auswahl: none \| sinusoidal \| learned \| rope \| alibi \| relative_bias \| combined | Basis | **Was:** Positionskodierung (SSMs brauchen keine – Reihenfolge steckt im Scan).<br>**Nutzen:** Meist 'none'.<br>**Wofür:** SSM-Modelle. |
| `embedding_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach dem Embedding.<br>**Nutzen:** Regularisiert Eingaberepräsentationen.<br>**Wofür:** Kleine Datensätze. |
| `embedding_scale` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Embeddings mit √d_model multiplizieren (None = automatisch: an bei 'sinusoidal').<br>**Nutzen:** Balanciert Token- gegen Positionssignal: ohne Skalierung gehen kleine Token-Embeddings im festen Sinus-Signal (Amplitude 1) unter (Original-Transformer skaliert deshalb).<br>**Wofür:** Sinusoidale Positionen, Gemma. |
| `embedding_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung direkt nach dem Embedding.<br>**Nutzen:** Stabilisiert sehr große Vokabulare (BLOOM).<br>**Wofür:** Selten. |
| `tie_embeddings` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Ausgabe-Gewichte = Embedding-Gewichte (Weight Tying).<br>**Nutzen:** Spart vocab·d Parameter und verbessert oft kleine Sprachmodelle.<br>**Wofür:** Sprachmodelle (nur wenn output_dim = vocab_size und kein Kopf-MLP). |
| `pad_token_id` | Ganzzahl (oder None) | `None` |  | Fortgeschritten | **Was:** Padding-Token: daraus wird automatisch die Padding-Maske erzeugt.<br>**Nutzen:** Padding wird von Attention ignoriert.<br>**Wofür:** Batches mit unterschiedlich langen Texten. |
| `num_register_tokens` | Ganzzahl | `0` | ≥ 0 | Experimentell | **Was:** Zusätzliche lernbare 'Register'-Tokens am Sequenzanfang.<br>**Nutzen:** Geben dem Modell Notizplatz; reduziert Artefakte in Attention-Karten (ViT-Register).<br>**Wofür:** Vision Transformer, Forschung. |
| `final_norm` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Normalisierung nach dem letzten Block.<br>**Nutzen:** Bei Pre-Norm nötig, damit die Ausgabe skaliert ist.<br>**Wofür:** Pre-Norm: an. |
| `pooling` | Text | `'none'` | Auswahl: none \| cls \| mean \| max \| last | Basis | **Was:** Verdichtung zu einem Vektor pro Sequenz.<br>**Nutzen:** Für sequenzweise Vorhersagen.<br>**Wofür:** Klassifikation: cls/mean; Decoder: last. |
| `head_hidden_dims` | Liste[Ganzzahl] | `[]` | ≥ 1 | Fortgeschritten | **Was:** Versteckte Schichten des Ausgabekopfs.<br>**Nutzen:** Mehr Kapazität im Kopf.<br>**Wofür:** Meist leer. |
| `head_activation` | Komponente aus Registry 'activation' | `gelu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Aktivierung im Kopf-MLP.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** Nur mit head_hidden_dims. |
| `head_activation.approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |
| `head_bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Sprachmodelle meist ohne.<br>**Wofür:** Klassifikation: an. |
| `logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Ausgabe-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert überkonfidente Vorhersagen und Instabilität.<br>**Wofür:** Gemma 2: 30. |

#### `model:hybrid` (Aliase: `jamba`) — Stufe Fortgeschritten

**Was:** Hybrid aus SSM- und Attention-Schichten.

**Nutzen:** Effizienz + Abrufpräzision.

**Wofür:** Lange Kontexte.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `init` | Unter-Konfiguration Initialisierung | ⟶ InitConfig |  | Fortgeschritten | **Was:** Initialisierung (Transformer-Standard: Embeddings ~ N(0, 0.02²), Biases 0).<br>**Nutzen:** Kleine Embeddings halten die Logits gebundener Ausgabegewichte anfangs klein → stabiler Start.<br>**Wofür:** GPT-2-Schema zusätzlich: linear_weight normal(0.02) + residual_projection scaled_normal. |
| `init.linear_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Gewichte linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Transformer: normal(std=0.02). |
| `init.linear_bias` | Komponente aus Registry 'initializer' | `zeros` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases linearer Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Häufig 'zeros'. |
| `init.conv_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Faltungsgewichte.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** ResNet: kaiming_normal(mode=fan_out). |
| `init.conv_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Faltungs-Biases.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.embedding` | Komponente aus Registry 'initializer' | `normal` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Fortgeschritten | **Was:** Initialisierer für Embedding-Tabellen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT/BERT: normal(std=0.02). |
| `init.embedding.mean` | Kommazahl | `0.0` |  | Fortgeschritten | **Was:** Mittelwert der Normalverteilung.<br>**Nutzen:** Verschiebt alle Startwerte.<br>**Wofür:** Fast immer 0. |
| `init.embedding.std` | Kommazahl | `0.02` | ≥ 0.0 | Fortgeschritten | **Was:** Standardabweichung.<br>**Nutzen:** Bestimmt die Startgröße der Gewichte.<br>**Wofür:** Transformer: 0.02 (GPT/BERT-Konvention). |
| `init.norm_weight` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Skalen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Normalerweise 1 ('keep'). |
| `init.norm_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Verschiebungen von Normalisierungen.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.rnn_weight_ih` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Eingangsgewichte rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Oft xavier_uniform. |
| `init.rnn_weight_hh` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für rekurrente Gewichte (Zustand→Zustand).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** Klassiker: orthogonal. |
| `init.rnn_bias` | Komponente aus Registry 'initializer' | `keep` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Biases rekurrenter Schichten.<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard. |
| `init.residual_projection` | Komponente aus Registry 'initializer' (oder None) | None | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Experte | **Was:** Initialisierer für Ausgangsprojektionen von Residual-Zweigen (Attention-Out, FFN-Down).<br>**Nutzen:** Startwerte bestimmen, ob das Training stabil startet.<br>**Wofür:** Feintuning der Trainingsdynamik; 'keep' = PyTorch-Standard.<br>**Tipp:** GPT-2: scaled_normal (std/√(2N)); 'zeros' = Zweige starten als Identität. |
| `extras` | Zuordnung (dict) | `{}` |  | Experimentell | **Was:** Freies dict für eigene, experimentelle Einstellungen.<br>**Nutzen:** Wird unverändert im Modell als self.config.extras bereitgestellt – eigene Hooks/Unterklassen können es lesen.<br>**Wofür:** Prototyping eigener Ideen, ohne gleich eine neue Config-Klasse zu schreiben. |
| `architecture` | Text | `'decoder_only'` | Auswahl: decoder_only \| encoder_only \| encoder_decoder | Basis | **Was:** Grundarchitektur.<br>**Nutzen:** Bestimmt Maskierung, Datenfluss und Einsatzzweck.<br>**Wofür:** Siehe Optionen. |
| `input_type` | Text | `'tokens'` | Auswahl: tokens \| continuous \| patches | Basis | **Was:** Art der Eingabedaten.<br>**Nutzen:** Bestimmt die Eingangsschicht.<br>**Wofür:** Text: tokens; Zeitreihen: continuous. |
| `vocab_size` | Ganzzahl | `1000` | ≥ 1 | Basis | **Was:** Vokabulargröße (nur tokens).<br>**Nutzen:** Anzahl unterscheidbarer Tokens.<br>**Wofür:** Zeichen-Modelle: ~100; BPE: 32k–256k. |
| `input_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Features pro Zeitschritt (nur continuous).<br>**Nutzen:** Breite der Eingabevektoren.<br>**Wofür:** Anzahl Sensoren/Variablen. |
| `image_channels` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Bildkanäle (nur patches).<br>**Nutzen:** RGB = 3.<br>**Wofür:** ViT. |
| `image_size` | Liste[Ganzzahl] | `[32, 32]` | ≥ 1 · Länge 2 | Basis | **Was:** Bildgröße [H, W] (nur patches).<br>**Nutzen:** Bestimmt Anzahl der Patches.<br>**Wofür:** ViT. |
| `patch_size` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Kantenlänge eines Patches (nur patches).<br>**Nutzen:** Kleinere Patches = mehr Tokens = feiner, aber quadratisch teurer.<br>**Wofür:** ViT-B/16: 16; kleine Bilder: 4. |
| `output_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Ausgabegröße. None = vocab_size (Sprachmodell) bzw. Pflicht bei anderen Aufgaben.<br>**Nutzen:** Klassen bei Klassifikation, Zielgrößen bei Regression.<br>**Wofür:** Klassifikation mit pooling ≠ none: Klassenzahl. |
| `d_model` | Ganzzahl | `128` | ≥ 1 | Basis | **Was:** Modellbreite (Größe der Repräsentation jedes Tokens).<br>**Nutzen:** Hauptregler der Kapazität (Parameter wachsen ~quadratisch).<br>**Wofür:** Klein 128–512, mittel 768–2048, groß 4096+. |
| `n_layers` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Anzahl Blöcke (beim encoder_decoder: Decoder-Blöcke).<br>**Nutzen:** Mehr Tiefe = abstraktere Verarbeitung.<br>**Wofür:** Klein 2–6, mittel 12–24, groß 32–80+. |
| `n_encoder_layers` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Encoder-Blöcke (nur encoder_decoder; None = n_layers).<br>**Nutzen:** Getrennte Tiefe für Lesen/Schreiben.<br>**Wofür:** T5: gleich. |
| `max_seq_len` | Ganzzahl | `256` | ≥ 1 | Basis | **Was:** Maximale Sequenzlänge.<br>**Nutzen:** Größe von Positionstabellen; längere Eingaben nicht möglich (außer RoPE/ALiBi).<br>**Wofür:** Kontextlänge der Anwendung. |
| `block` | Unter-Konfiguration Transformer-Block | ⟶ TransformerBlockConfig |  | Basis | **Was:** Bauplan eines Transformer-Blocks (Mixer + FFN mit Residualverbindungen).<br>**Nutzen:** Gilt für alle Schichten; einzelne Schichten weichen über layer_overrides ab.<br>**Wofür:** Jede Transformer-Variante. |
| `block.mixer` | Komponente aus Registry 'sequence_mixer' | `ssm` | Auswahl: attention \| ssm \| rnn \| conv \| fourier \| none | Basis | **Was:** Mixer: tauscht Information zwischen Positionen aus (Attention, SSM, RNN, Faltung ...).<br>**Nutzen:** Bestimmt Kosten und Reichweite; pro Schicht wählbar → Hybride.<br>**Wofür:** Standard: attention. Lange Sequenzen: ssm. Hybride über layer_overrides.<br>**Tipp:** Bei 'attention' liegen die Attention-Einstellungen direkt unter mixer.* (z. B. mixer.num_heads). |
| `block.mixer.d_state` | Ganzzahl | `16` | ≥ 1 | Fortgeschritten | **Was:** Größe des inneren Zustands pro Kanal (N).<br>**Nutzen:** Mehr Zustand = mehr Gedächtniskapazität, mehr Rechenaufwand.<br>**Wofür:** Mamba: 16. |
| `block.mixer.expand` | Kommazahl | `2.0` | ≥ 0.25 | Fortgeschritten | **Was:** Aufweitung der inneren Breite (E = expand · d_model).<br>**Nutzen:** Kapazität des Blocks.<br>**Wofür:** Mamba: 2. |
| `block.mixer.d_conv` | Ganzzahl | `4` | ≥ 1 | Fortgeschritten | **Was:** Kernelgröße der kurzen kausalen Faltung vor dem Scan.<br>**Nutzen:** Erfasst lokale Muster, bevor der Zustand aktualisiert wird.<br>**Wofür:** Mamba: 4. |
| `block.mixer.dt_rank` | Ganzzahl oder Text | `'auto'` |  | Experte | **Was:** Rang der Δ-Projektion ('auto' = ceil(d_model/16)).<br>**Nutzen:** Steuert, wie fein die Schrittweite Δ von der Eingabe abhängen kann.<br>**Wofür:** Selten zu ändern. |
| `block.mixer.dt_min` | Kommazahl | `0.001` | ≥ 1e-06 | Experte | **Was:** Kleinste initiale Schrittweite Δ.<br>**Nutzen:** Kleine Δ = langes Gedächtnis.<br>**Wofür:** Mamba: 0.001. |
| `block.mixer.dt_max` | Kommazahl | `0.1` | ≥ 1e-06 | Experte | **Was:** Größte initiale Schrittweite Δ.<br>**Nutzen:** Große Δ = Fokus auf aktuelle Eingabe.<br>**Wofür:** Mamba: 0.1. |
| `block.mixer.a_init` | Text | `'s4d_real'` | Auswahl: s4d_real \| s4d_lin \| random | Experte | **Was:** Initialisierung der Zustandsmatrix A.<br>**Nutzen:** Bestimmt Start-Gedächtnis-Zeitskalen.<br>**Wofür:** Forschung. |
| `block.mixer.bidirectional` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Zusätzlicher Rückwärts-Scan (wird bei kausalen Modellen ignoriert).<br>**Nutzen:** Kontext von beiden Seiten für Encoder-Aufgaben.<br>**Wofür:** Klassifikation, Tagging. |
| `block.mixer.conv_bias` | Ja/Nein | `True` |  | Experte | **Was:** Bias der kurzen Faltung.<br>**Nutzen:** Verschiebung.<br>**Wofür:** Standard. |
| `block.mixer.bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias der Ein-/Ausgangsprojektion.<br>**Nutzen:** Verschiebung.<br>**Wofür:** Mamba: aus. |
| `block.mixer.scan_mode` | Text | `'auto'` | Auswahl: auto \| parallel \| chunked \| sequential \| mamba_ssm | Fortgeschritten | **Was:** Algorithmus für den Zustands-Scan h_t = exp(Δ·A)·h_{t-1} + Δ·B·x.<br>**Nutzen:** Der parallele/chunked Scan braucht nur log₂(T) Runden statt T Schleifenschritte – Training oft 10–40× schneller als die Python-Schleife.<br>**Wofür:** auto ist fast immer richtig (CPU: fusionierte Schleife, GPU: chunked); mamba_ssm auf NVIDIA-GPUs mit installiertem Paket. |
| `block.mixer.scan_chunk_size` | Ganzzahl | `32` | ≥ 1 | Fortgeschritten | **Was:** Blocklänge des chunked-Scans.<br>**Nutzen:** Größer = weniger sequenzielle Überträge, aber mehr Zwischenspeicher pro Runde.<br>**Wofür:** CPU: 16–32; GPU: 64–256. |
| `block.mixer.scan_backward` | Text | `'fused'` | Auswahl: fused \| scan \| autograd | Experte | **Was:** Wie der Gradient des Scans berechnet wird.<br>**Nutzen:** 'fused' senkt den Aktivierungsspeicher des Scans etwa um den Faktor d_state – wie der offizielle Mamba-Kernel, aber in reinem PyTorch (CPU, GPU, MPS).<br>**Wofür:** Lange Sequenzen / große Batches: fused. Kurze Sequenzen mit viel Speicher: scan. |
| `block.cross_attention` | Unter-Konfiguration Attention | ⟶ AttentionConfig |  | Fortgeschritten | **Was:** Cross-Attention der Decoder-Blöcke (nur encoder_decoder).<br>**Nutzen:** Lässt den Decoder auf die Encoder-Ausgabe schauen.<br>**Wofür:** Übersetzung, Zusammenfassung, Seq2Seq. |
| `block.cross_attention.num_heads` | Ganzzahl | `8` | ≥ 1 | Fortgeschritten | **Was:** Anzahl paralleler Attention-Köpfe.<br>**Nutzen:** Jeder Kopf kann andere Beziehungen lernen (Syntax, Nähe, Bezüge ...).<br>**Wofür:** Alle Attention-Modelle.<br>**Tipp:** Faustregel: head_dim (= d_model / num_heads) zwischen 32 und 128. |
| `block.cross_attention.num_kv_heads` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Anzahl Key/Value-Köpfe (Grouped-Query-Attention). None = num_heads.<br>**Nutzen:** Weniger KV-Köpfe sparen Parameter und vor allem KV-Cache-Speicher bei der Generierung.<br>**Wofür:** Große Sprachmodelle (LLaMA-2-70B: 8 KV-Köpfe bei 64 Köpfen). 1 = Multi-Query-Attention.<br>**Tipp:** num_heads muss durch num_kv_heads teilbar sein. |
| `block.cross_attention.head_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Dimension pro Kopf. None = d_model / num_heads.<br>**Nutzen:** Entkoppelt Kopfgröße von Modellbreite (z. B. mehr Köpfe ohne Breitenänderung).<br>**Wofür:** Feintuning der Kapazität. |
| `block.cross_attention.kernel` | Komponente aus Registry 'attention_kernel' | `softmax` | Auswahl: softmax \| sigmoid \| relu \| cosine \| linear | Fortgeschritten | **Was:** Attention-Kernel (wie aus Ähnlichkeiten Gewichte werden).<br>**Nutzen:** Bestimmt Kosten (quadratisch/linear) und Verhalten.<br>**Wofür:** Softmax ist Standard; andere für lange Sequenzen oder Forschung. |
| `block.cross_attention.kernel.backend` | Text | `'auto'` | Auswahl: auto \| sdpa \| manual | Fortgeschritten | **Was:** Implementierung der Softmax-Attention.<br>**Nutzen:** SDPA nutzt optimierte Kernel (Flash-Attention) – viel weniger Speicher bei langen Sequenzen.<br>**Wofür:** Speicher/Geschwindigkeit; 'manual' für Analyse/Debugging. |
| `block.cross_attention.kernel.temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Experte | **Was:** Softmax-Temperatur (Logits werden durch sie geteilt).<br>**Nutzen:** < 1 = schärfere, fokussiertere Attention; > 1 = weichere, breitere Verteilung.<br>**Wofür:** Experimente, Kalibrierung. |
| `block.cross_attention.kernel.logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Attention-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert extreme Logits und damit Instabilität in großen Modellen.<br>**Wofür:** Große Modelle, Training mit hohen Lernraten (Gemma 2: 50).<br>**Tipp:** Erzwingt den manuellen (langsameren) Pfad. |
| `block.cross_attention.kernel.softmax_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Softmax im manuellen Pfad in 32-Bit rechnen.<br>**Nutzen:** Numerisch stabil bei fp16/bf16.<br>**Wofür:** Mixed Precision. |
| `block.cross_attention.kernel.store_weights` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Letzte Attention-Gewichte im Modul speichern (Attribut last_weights).<br>**Nutzen:** Erlaubt Visualisierung/Analyse, kostet Speicher und erzwingt den manuellen Pfad.<br>**Wofür:** Interpretierbarkeit, Debugging. |
| `block.cross_attention.causal` | Ja/Nein (oder None) | `None` |  | Fortgeschritten | **Was:** Kausale Maske: jedes Element sieht nur sich selbst und frühere Elemente.<br>**Nutzen:** Pflicht für autoregressive Generierung (Sprachmodelle), sonst 'Spicken' in die Zukunft.<br>**Wofür:** None = automatisch durch die Architektur (Decoder kausal, Encoder nicht). |
| `block.cross_attention.window_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Sliding-Window-Attention: nur die letzten/nächsten N Elemente sehen.<br>**Nutzen:** Spart Rechenaufwand bei langen Sequenzen; über mehrere Schichten wächst die Reichweite trotzdem.<br>**Wofür:** Mistral (4096), lange Dokumente, lokale Muster (Audio, Zeitreihen).<br>**Tipp:** Per layer_overrides z. B. nur jede zweite Schicht lokal machen. |
| `block.cross_attention.scale` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** Skalierungsfaktor der Logits. None = 1/√head_dim.<br>**Nutzen:** Verhindert zu große Logits (zu scharfe Softmax) bei großer Kopfdimension.<br>**Wofür:** µP-Skalierung nutzt 1/head_dim; Experimente. |
| `block.cross_attention.qk_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Fortgeschritten | **Was:** Normalisierung von Queries und Keys pro Kopf (z. B. rmsnorm).<br>**Nutzen:** Begrenzt Logit-Wachstum → deutlich stabileres Training mit hohen Lernraten.<br>**Wofür:** Große Modelle (OLMo-2, Gemma 3, ViT-22B). |
| `block.cross_attention.qkv_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den Q/K/V-Projektionen.<br>**Nutzen:** Klassische Transformer (GPT-2, BERT) nutzen Bias; moderne LLMs meist nicht.<br>**Wofür:** Kompatibilität; Qwen nutzt nur QKV-Bias. |
| `block.cross_attention.out_bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in der Ausgangsprojektion.<br>**Nutzen:** Siehe qkv_bias.<br>**Wofür:** Kompatibilität. |
| `block.cross_attention.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Dropout auf die Attention-Gewichte.<br>**Nutzen:** Regularisiert, indem zufällig Verbindungen gekappt werden.<br>**Wofür:** Kleine Datensätze. Große LLMs: 0. |
| `block.cross_attention.head_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experimentell | **Was:** Ganze Köpfe im Training zufällig abschalten (DropHead).<br>**Nutzen:** Verhindert, dass sich das Modell auf wenige Köpfe verlässt.<br>**Wofür:** Regularisierung, Robustheit, Pruning-Vorbereitung. |
| `block.cross_attention.output_gate` | Ja/Nein | `False` |  | Experimentell | **Was:** Ausgabe jedes Kopfes mit sigmoid(W·x) gaten ('gated attention').<br>**Nutzen:** Reduziert 'Attention Sinks' und Aktivierungsausreißer, verbessert Stabilität.<br>**Wofür:** Neuere LLM-Forschung (Qwen3-Next). |
| `block.cross_attention.kv_compression_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** MLA-lite: Keys/Values entstehen aus einem gemeinsamen, komprimierten Latent-Vektor dieser Größe.<br>**Nutzen:** Der KV-Cache speichert nur noch den Latent-Vektor (r statt 2·H_kv·head_dim Werte pro Token) – oft 4–16× weniger Cache-Speicher; zusätzlich weniger Parameter.<br>**Wofür:** Lange Kontexte, Generierung mit großen Batches (DeepSeek-V2/V3 Multi-Head Latent Attention).<br>**Tipp:** Vereinfachte Variante: RoPE wird auf die rekonstruierten Keys angewandt (keine entkoppelten RoPE-Köpfe). |
| `block.cross_attention.kv_latent_norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung des MLA-Latent-Vektors (nur mit kv_compression_dim).<br>**Nutzen:** Stabilisiert die Rekonstruktion von Keys/Values.<br>**Wofür:** DeepSeek: RMSNorm. |
| `block.cross_attention.kv_latent_norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `block.cross_attention.kv_latent_norm.learnable` | Ja/Nein | `True` |  | Experte | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `block.cross_attention.kv_latent_norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `block.cross_attention.kv_latent_norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `block.ffn` | Komponente aus Registry 'feedforward' | `glu` | Auswahl: mlp \| glu \| moe \| kan \| none | Basis | **Was:** Feed-Forward-Teil (mlp, glu/SwiGLU, moe, none).<br>**Nutzen:** Positionsweise Verarbeitung – hier steckt die meiste Kapazität.<br>**Wofür:** glu (SwiGLU) ist moderner Standard; moe für sehr große Kapazität. |
| `block.ffn.hidden_mult` | Kommazahl | `2.6666666666666665` | ≥ 0.01 | Basis | **Was:** Breite relativ zu d_model (GLU hat 3 Matrizen → Standard 8/3 hält die Parameter wie MLP mit 4).<br>**Nutzen:** Mehr Breite = mehr Kapazität.<br>**Wofür:** LLaMA: 8/3 (aufgerundet auf multiple_of). |
| `block.ffn.hidden_dim` | Ganzzahl (oder None) | `None` | ≥ 1 | Fortgeschritten | **Was:** Exakte Breite der versteckten Schicht (überschreibt hidden_mult).<br>**Nutzen:** Für exakte Nachbauten bekannter Architekturen.<br>**Wofür:** Gewichte-Import, exakte Parameterzahlen. |
| `block.ffn.multiple_of` | Ganzzahl | `1` | ≥ 1 | Experte | **Was:** Versteckte Breite auf ein Vielfaches davon aufrunden.<br>**Nutzen:** Vielfache von 64/256 rechnen auf GPUs schneller (Tensor Cores).<br>**Wofür:** Geschwindigkeitsoptimierung (LLaMA: 256). |
| `block.ffn.activation` | Komponente aus Registry 'activation' | `silu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Gate-Aktivierung: silu → SwiGLU, gelu → GeGLU, sigmoid → GLU, relu → ReGLU, identity → Bilinear.<br>**Nutzen:** Das Gate entscheidet elementweise, welche Information durchgelassen wird.<br>**Wofür:** SwiGLU ist Standard moderner LLMs. |
| `block.ffn.bias` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Bias in den linearen Schichten.<br>**Nutzen:** Moderne LLMs verzichten darauf.<br>**Wofür:** Kompatibilität. |
| `block.ffn.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach der Aktivierung (innerhalb des FFN).<br>**Nutzen:** Regularisiert die versteckte Schicht.<br>**Wofür:** Kleine Datensätze. |
| `block.norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Normalisierung in den Residual-Zweigen.<br>**Nutzen:** Stabilisiert das Training.<br>**Wofür:** rmsnorm (modern), layernorm (klassisch). |
| `block.norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `block.norm.learnable` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `block.norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `block.norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `block.norm_position` | Text | `'pre'` | Auswahl: pre \| post \| sandwich \| none | Fortgeschritten | **Was:** Position der Normalisierung relativ zur Residualverbindung.<br>**Nutzen:** Großer Einfluss auf Trainingsstabilität und Endqualität.<br>**Wofür:** pre für tiefe Modelle; post für kleine Modelle mit Warmup. |
| `block.parallel` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Mixer und FFN parallel statt nacheinander (x + Mixer(n(x)) + FFN(n(x))).<br>**Nutzen:** ~15 % schneller bei großen Modellen, minimal andere Qualität.<br>**Wofür:** GPT-J, PaLM. |
| `block.dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout auf die Ausgabe jedes Zweigs (Residual-Dropout).<br>**Nutzen:** Regularisierung.<br>**Wofür:** Kleine Datensätze: 0.1; große LLMs: 0. |
| `block.drop_path` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Stochastic Depth: Wahrscheinlichkeit, einen ganzen Zweig auszulassen.<br>**Nutzen:** Starke Regularisierung, besonders für Vision-Transformer.<br>**Wofür:** ViT/DeiT: 0.1–0.3 (mit linearem Anstieg über die Tiefe). |
| `block.layer_scale_init` | Kommazahl (oder None) | `None` | ≥ 0.0 | Experte | **Was:** LayerScale: lernbarer Faktor pro Kanal auf jedem Zweig (Startwert; None = aus).<br>**Nutzen:** Kleiner Start (1e-5…1e-1) lässt tiefe Netze zunächst wie flache trainieren.<br>**Wofür:** Sehr tiefe Transformer (CaiT, DINOv2). |
| `block.branch_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf jeder Zweig-Ausgabe.<br>**Nutzen:** Dämpft Residual-Beiträge in tiefen Netzen.<br>**Wofür:** Experimente; z. B. 1/√(2N). |
| `block.skip_scale` | Kommazahl | `1.0` |  | Experte | **Was:** Fester Faktor auf dem Skip-Pfad (DeepNet: α > 1).<br>**Nutzen:** Stärkerer Skip stabilisiert Post-LN-Netze mit hunderten Schichten.<br>**Wofür:** DeepNorm-Experimente. |
| `encoder_block` | Unter-Konfiguration Transformer-Block | None |  | Fortgeschritten | **Was:** Abweichender Block-Bauplan für den Encoder (None = wie block).<br>**Nutzen:** Encoder und Decoder können völlig verschieden sein.<br>**Wofür:** encoder_decoder. |
| `layer_overrides` | Zuordnung (dict) | `{'1::2': {'mixer': 'attention'}}` |  | Fortgeschritten | **Was:** Schichtweise Abweichungen; Standard: jede zweite Schicht (1, 3, 5 ...) nutzt Attention.<br>**Nutzen:** Steuert das Mischungsverhältnis der Mixer.<br>**Wofür:** Jamba: 1 Attention auf 7 Mamba → {'7::8': {'mixer': 'attention'}}. |
| `encoder_layer_overrides` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Schichtweise Abweichungen des Encoders.<br>**Nutzen:** Wie layer_overrides.<br>**Wofür:** encoder_decoder. |
| `drop_path_schedule` | Text | `'linear'` | Auswahl: constant \| linear | Experte | **Was:** Verteilung von drop_path über die Tiefe.<br>**Nutzen:** Frühe Schichten seltener auslassen.<br>**Wofür:** Bei drop_path > 0. |
| `positional` | Komponente aus Registry 'positional' | `rope` | Auswahl: none \| sinusoidal \| learned \| rope \| alibi \| relative_bias \| combined | Basis | **Was:** Positionskodierung.<br>**Nutzen:** Gibt dem Modell Reihenfolge-Information.<br>**Wofür:** rope (Text), learned (ViT), sinusoidal. |
| `positional.theta` | Kommazahl | `10000.0` | ≥ 1.0 | Basis | **Was:** Basisfrequenz θ der Rotationen.<br>**Nutzen:** Größeres θ → längere Kontexte unterscheidbar (LLaMA-3: 500000).<br>**Wofür:** Lange Kontexte → θ erhöhen. |
| `positional.fraction` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil der Kopf-Dimensionen, die rotiert werden ('partial rotary').<br>**Nutzen:** Rest bleibt positionsfrei – manche Modelle (GPT-NeoX: 0.25) profitieren davon.<br>**Wofür:** Feintuning von Positions- vs. Inhaltsanteil. |
| `positional.scaling` | Text | `'none'` | Auswahl: none \| linear \| ntk \| yarn | Fortgeschritten | **Was:** Verfahren zur Kontextverlängerung.<br>**Nutzen:** Erlaubt Sequenzen länger als im Training.<br>**Wofür:** Kontext-Erweiterung vortrainierter Modelle. |
| `positional.factor` | Kommazahl | `1.0` | ≥ 1.0 | Fortgeschritten | **Was:** Erweiterungsfaktor des Kontexts (z. B. 4 = vierfache Länge).<br>**Nutzen:** Stärke der Skalierung.<br>**Wofür:** Nur mit scaling ≠ none. |
| `positional.original_context` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** Kontextlänge, mit der ursprünglich trainiert wurde (nur YaRN; None = max_seq_len / factor).<br>**Nutzen:** YaRN entscheidet anhand dieser Länge, welche Frequenzen interpoliert werden.<br>**Wofür:** Vortrainiertes Modell mit 4096 Tokens auf 16k erweitern: 4096, factor 4. |
| `positional.yarn_beta_fast` | Kommazahl | `32.0` | ≥ 0.0 | Experte | **Was:** YaRN β_fast: ab so vielen Umdrehungen im Originalkontext bleibt eine Frequenz unverändert.<br>**Nutzen:** Schützt die lokale Positionsauflösung.<br>**Wofür:** 32 (Paper). |
| `positional.yarn_beta_slow` | Kommazahl | `1.0` | ≥ 0.0 | Experte | **Was:** YaRN β_slow: unter so vielen Umdrehungen wird voll interpoliert.<br>**Nutzen:** Niedrige Frequenzen.<br>**Wofür:** 1 (Paper). |
| `positional.interleaved` | Ja/Nein | `False` |  | Experte | **Was:** Paare benachbarter Dimensionen rotieren (GPT-J) statt Hälften (GPT-NeoX/LLaMA).<br>**Nutzen:** Nur für Kompatibilität mit fremden Gewichten relevant.<br>**Wofür:** Gewichte-Import. |
| `embedding_dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout nach dem Embedding.<br>**Nutzen:** Regularisiert Eingaberepräsentationen.<br>**Wofür:** Kleine Datensätze. |
| `embedding_scale` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Embeddings mit √d_model multiplizieren (None = automatisch: an bei 'sinusoidal').<br>**Nutzen:** Balanciert Token- gegen Positionssignal: ohne Skalierung gehen kleine Token-Embeddings im festen Sinus-Signal (Amplitude 1) unter (Original-Transformer skaliert deshalb).<br>**Wofür:** Sinusoidale Positionen, Gemma. |
| `embedding_norm` | Komponente aus Registry 'normalization' (oder None) | None | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Experte | **Was:** Normalisierung direkt nach dem Embedding.<br>**Nutzen:** Stabilisiert sehr große Vokabulare (BLOOM).<br>**Wofür:** Selten. |
| `tie_embeddings` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Ausgabe-Gewichte = Embedding-Gewichte (Weight Tying).<br>**Nutzen:** Spart vocab·d Parameter und verbessert oft kleine Sprachmodelle.<br>**Wofür:** Sprachmodelle (nur wenn output_dim = vocab_size und kein Kopf-MLP). |
| `pad_token_id` | Ganzzahl (oder None) | `None` |  | Fortgeschritten | **Was:** Padding-Token: daraus wird automatisch die Padding-Maske erzeugt.<br>**Nutzen:** Padding wird von Attention ignoriert.<br>**Wofür:** Batches mit unterschiedlich langen Texten. |
| `num_register_tokens` | Ganzzahl | `0` | ≥ 0 | Experimentell | **Was:** Zusätzliche lernbare 'Register'-Tokens am Sequenzanfang.<br>**Nutzen:** Geben dem Modell Notizplatz; reduziert Artefakte in Attention-Karten (ViT-Register).<br>**Wofür:** Vision Transformer, Forschung. |
| `final_norm` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Normalisierung nach dem letzten Block.<br>**Nutzen:** Bei Pre-Norm nötig, damit die Ausgabe skaliert ist.<br>**Wofür:** Pre-Norm: an. |
| `pooling` | Text | `'none'` | Auswahl: none \| cls \| mean \| max \| last | Basis | **Was:** Verdichtung zu einem Vektor pro Sequenz.<br>**Nutzen:** Für sequenzweise Vorhersagen.<br>**Wofür:** Klassifikation: cls/mean; Decoder: last. |
| `head_hidden_dims` | Liste[Ganzzahl] | `[]` | ≥ 1 | Fortgeschritten | **Was:** Versteckte Schichten des Ausgabekopfs.<br>**Nutzen:** Mehr Kapazität im Kopf.<br>**Wofür:** Meist leer. |
| `head_activation` | Komponente aus Registry 'activation' | `gelu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Fortgeschritten | **Was:** Aktivierung im Kopf-MLP.<br>**Nutzen:** Nichtlinearität.<br>**Wofür:** Nur mit head_hidden_dims. |
| `head_activation.approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |
| `head_bias` | Ja/Nein | `False` |  | Experte | **Was:** Bias der Ausgabeschicht.<br>**Nutzen:** Sprachmodelle meist ohne.<br>**Wofür:** Klassifikation: an. |
| `logit_softcap` | Kommazahl (oder None) | `None` | ≥ 1e-06 | Experte | **Was:** Weiche Obergrenze der Ausgabe-Logits: cap·tanh(x/cap).<br>**Nutzen:** Verhindert überkonfidente Vorhersagen und Instabilität.<br>**Wofür:** Gemma 2: 30. |


### Registry `callback`

<a id="registry-callback"></a>

**Callbacks** – Erweiterungen, die sich in den Trainingsablauf einklinken. Automatisieren Abbruch, Speichern, Protokollieren, Überwachen.

Fabrik-Signatur: `factory(args) -> Callback`

#### `callback:early_stopping` — Stufe Basis

**Was:** Training beenden, wenn sich die Validierung nicht mehr verbessert.

**Nutzen:** Gegen Überanpassung, spart Zeit.

**Wofür:** Fast jedes überwachte Training.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `monitor` | Text | `'val_loss'` |  | Basis | **Was:** Beobachtete Metrik (z. B. val_loss, val_accuracy).<br>**Nutzen:** Kriterium für 'besser'.<br>**Wofür:** val_loss. |
| `mode` | Text | `'min'` | Auswahl: min \| max | Basis | **Was:** min: kleiner ist besser (Verluste); max: größer ist besser (Genauigkeit).<br>**Nutzen:** Richtung.<br>**Wofür:** Passend zur Metrik. |
| `patience` | Ganzzahl | `5` | ≥ 0 | Basis | **Was:** Evaluierungen ohne Verbesserung bis zum Abbruch.<br>**Nutzen:** Verhindert Überanpassung und spart Zeit.<br>**Wofür:** 3–10. |
| `min_delta` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** Mindestverbesserung, die als Verbesserung zählt.<br>**Nutzen:** Ignoriert Rauschen.<br>**Wofür:** 1e-4. |
| `restore_best` | Ja/Nein | `True` |  | Basis | **Was:** Am Ende die besten Gewichte wiederherstellen.<br>**Nutzen:** Man behält das beste, nicht das letzte Modell.<br>**Wofür:** an. |

#### `callback:checkpoint` — Stufe Basis

**Was:** Modell regelmäßig bzw. bei Verbesserung speichern.

**Nutzen:** Fortsetzen nach Abbrüchen, bestes Modell behalten.

**Wofür:** Längere Trainings.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `monitor` | Text | `'val_loss'` |  | Basis | **Was:** Beobachtete Metrik (z. B. val_loss, val_accuracy).<br>**Nutzen:** Kriterium für 'besser'.<br>**Wofür:** val_loss. |
| `mode` | Text | `'min'` | Auswahl: min \| max | Basis | **Was:** min: kleiner ist besser (Verluste); max: größer ist besser (Genauigkeit).<br>**Nutzen:** Richtung.<br>**Wofür:** Passend zur Metrik. |
| `directory` | Text | `'checkpoints'` |  | Basis | **Was:** Zielordner.<br>**Nutzen:** Hier landen die Checkpoints.<br>**Wofür:** Projektordner. |
| `every_n_steps` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Zusätzlich alle n Schritte speichern (None = nur bei Evaluierung).<br>**Nutzen:** Absicherung langer Läufe.<br>**Wofür:** 1000. |
| `save_best_only` | Ja/Nein | `True` |  | Basis | **Was:** Nur bei Verbesserung speichern (beste.pt).<br>**Nutzen:** Spart Speicherplatz.<br>**Wofür:** an. |
| `keep_last` | Ganzzahl | `2` | ≥ 1 | Fortgeschritten | **Was:** Wie viele periodische Checkpoints behalten.<br>**Nutzen:** Speicherplatz vs. Rückfalloptionen.<br>**Wofür:** 2–5. |

#### `callback:csv_logger` — Stufe Basis

**Was:** Metriken als CSV protokollieren.

**Nutzen:** Auswertung in Tabellenkalkulation/Pandas.

**Wofür:** Analyse.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `path` | Text | `'training_log.csv'` |  | Basis | **Was:** Dateipfad des Protokolls.<br>**Nutzen:** Hier werden Metriken angehängt.<br>**Wofür:** Projektordner. |

#### `callback:jsonl_logger` — Stufe Basis

**Was:** Metriken als JSON-Zeilen protokollieren.

**Nutzen:** Maschinenlesbar, robust.

**Wofür:** Analyse, Dashboards.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `path` | Text | `'training_log.csv'` |  | Basis | **Was:** Dateipfad des Protokolls.<br>**Nutzen:** Hier werden Metriken angehängt.<br>**Wofür:** Projektordner. |

#### `callback:gradient_monitor` — Stufe Fortgeschritten

**Was:** Gradienten-Normen je Parameter überwachen.

**Nutzen:** Findet explodierende/verschwindende Gradienten.

**Wofür:** Debugging.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `every_n_steps` | Ganzzahl | `50` | ≥ 1 | Basis | **Was:** Messintervall.<br>**Nutzen:** Wie oft Gradienten-Normen gemessen werden.<br>**Wofür:** 50. |
| `top_k` | Ganzzahl | `5` | ≥ 1 | Basis | **Was:** Anzahl Parameter mit den größten Gradienten, die ausgegeben werden.<br>**Nutzen:** Findet explodierende Schichten.<br>**Wofür:** 5. |

#### `callback:time_limit` — Stufe Basis

**Was:** Training nach fester Zeit beenden.

**Nutzen:** Zeitbudgets einhalten.

**Wofür:** Cluster, Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `max_minutes` | Kommazahl | `60.0` | ≥ 0.0 | Basis | **Was:** Maximale Trainingsdauer in Minuten.<br>**Nutzen:** Feste Zeitbudgets.<br>**Wofür:** Cluster-Zeitslots. |

#### `callback:gradual_unfreeze` — Stufe Experimentell

**Was:** Schichten schrittweise auftauen.

**Nutzen:** Schonendes Feintuning.

**Wofür:** Transfer-Learning (ULMFiT).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `patterns` | Liste[Text] | `['head', 'blocks\\.3', 'blocks\\.2', 'blocks\\.1', 'blocks\\` |  | Basis | **Was:** Regex-Gruppen in Auftau-Reihenfolge (zuerst trainierbar → zuletzt).<br>**Nutzen:** Schrittweises Auftauen schützt vortrainiertes Wissen (ULMFiT).<br>**Wofür:** Transfer-Learning. |
| `every_epochs` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Alle n Epochen eine Gruppe mehr auftauen.<br>**Nutzen:** Tempo.<br>**Wofür:** 1. |


### Registry `loss`

<a id="registry-loss"></a>

**Verlustfunktionen** – Funktionen, die messen, wie falsch eine Vorhersage ist. Das Training minimiert genau diese Größe – sie definiert, was 'gut' bedeutet.

Fabrik-Signatur: `factory(args) -> Modul mit forward(pred, target)`

#### `loss:cross_entropy` (Aliase: `ce`) — Stufe Basis

**Was:** Kreuzentropie für Klassifikation (Logits + Klassenindizes oder Wahrscheinlichkeiten).

**Nutzen:** Standardverlust für Klassifikation und Sprachmodelle.

**Wofür:** Klassifikation, LMs, Tagging.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `label_smoothing` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Label Smoothing: Zielwahrscheinlichkeit wird leicht über alle Klassen verteilt.<br>**Nutzen:** Verhindert Überkonfidenz, verbessert Kalibrierung und Generalisierung.<br>**Wofür:** Bildklassifikation (0.1), Übersetzung (0.1). |
| `ignore_index` | Ganzzahl | `-100` |  | Fortgeschritten | **Was:** Zielwert, der ignoriert wird (z. B. Padding).<br>**Nutzen:** Diese Positionen zählen nicht zum Verlust.<br>**Wofür:** Sprachmodelle, Tagging mit Padding. |
| `class_weights` | Liste[Kommazahl] | `[]` | ≥ 0.0 | Fortgeschritten | **Was:** Gewicht pro Klasse (leer = alle gleich).<br>**Nutzen:** Gleicht unbalancierte Datensätze aus.<br>**Wofür:** Seltene Klassen stärker gewichten (z. B. ∝ 1/Häufigkeit). |

#### `loss:bce` (Aliase: `binary_cross_entropy`) — Stufe Basis

**Was:** Binäre Kreuzentropie (mit Logits).

**Nutzen:** Binär- und Multi-Label-Klassifikation.

**Wofür:** Ja/Nein-Fragen, Multi-Label.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `pos_weight` | Kommazahl (oder None) | `None` | ≥ 0.0 | Basis | **Was:** Gewicht positiver Beispiele.<br>**Nutzen:** Gleicht seltene Positive aus (> 1).<br>**Wofür:** Unbalancierte Binärprobleme. |
| `label_smoothing` | Kommazahl | `0.0` | Bereich 0.0 … 0.5 | Basis | **Was:** Ziele 0/1 werden zu ε/2 bzw. 1-ε/2.<br>**Nutzen:** Weniger Überkonfidenz.<br>**Wofür:** GAN-Diskriminatoren, Multi-Label. |

#### `loss:mse` (Aliase: `l2`) — Stufe Basis

**Was:** Mittlerer quadratischer Fehler.

**Nutzen:** Bestraft große Fehler stark.

**Wofür:** Regression.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |

#### `loss:l1` (Aliase: `mae`) — Stufe Basis

**Was:** Mittlerer absoluter Fehler.

**Nutzen:** Robust gegen Ausreißer (Median-Schätzer).

**Wofür:** Regression mit Ausreißern.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |

#### `loss:huber` — Stufe Basis

**Was:** Huber-Verlust: quadratisch nah, linear fern.

**Nutzen:** Kompromiss aus MSE und L1.

**Wofür:** Robuste Regression, DQN.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `delta` | Kommazahl | `1.0` | ≥ 1e-09 | Basis | **Was:** Übergang quadratisch → linear.<br>**Nutzen:** Robust gegen Ausreißer (linear für große Fehler).<br>**Wofür:** Skala der Fehler. |

#### `loss:smooth_l1` — Stufe Fortgeschritten

**Was:** Smooth-L1.

**Nutzen:** Wie Huber (skaliert).

**Wofür:** Bounding-Box-Regression.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `delta` | Kommazahl | `1.0` | ≥ 1e-09 | Basis | **Was:** Übergang quadratisch → linear.<br>**Nutzen:** Robust gegen Ausreißer (linear für große Fehler).<br>**Wofür:** Skala der Fehler. |

#### `loss:log_cosh` — Stufe Fortgeschritten

**Was:** log(cosh(Fehler)).

**Nutzen:** Glatt wie MSE, robust wie L1.

**Wofür:** Regression.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |

#### `loss:quantile` (Aliase: `pinball`) — Stufe Basis

**Was:** Quantil-(Pinball-)Verlust.

**Nutzen:** Vorhersageintervalle.

**Wofür:** Prognosen mit Unsicherheit.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `quantiles` | Liste[Kommazahl] | `[0.1, 0.5, 0.9]` | Bereich 0.0 … 1.0 | Basis | **Was:** Vorherzusagende Quantile (letzte Ausgabeachse = Anzahl Quantile).<br>**Nutzen:** Liefert Vorhersageintervalle statt Punktschätzungen.<br>**Wofür:** Prognosen mit Unsicherheit (Nachfrage, Energie). |

#### `loss:gaussian_nll` — Stufe Fortgeschritten

**Was:** Gauß-Likelihood mit gelernter Varianz.

**Nutzen:** Modell schätzt seine eigene Unsicherheit.

**Wofür:** Probabilistische Regression.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |

#### `loss:poisson_nll` — Stufe Fortgeschritten

**Was:** Poisson-Likelihood.

**Nutzen:** Für Zähldaten.

**Wofür:** Ereignisanzahlen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `log_input` | Ja/Nein | `True` |  | Basis | **Was:** Vorhersage ist log(Rate) (empfohlen).<br>**Nutzen:** Numerisch stabil, Rate immer positiv.<br>**Wofür:** Zähldaten. |

#### `loss:focal` — Stufe Basis

**Was:** Focal Loss.

**Nutzen:** Fokus auf schwierige Beispiele.

**Wofür:** Starke Klassen-Unbalance, Detektion.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `gamma` | Kommazahl | `2.0` | ≥ 0.0 | Basis | **Was:** Fokussierungsstärke: leichte Beispiele werden heruntergewichtet.<br>**Nutzen:** Konzentriert das Training auf schwierige/seltene Fälle.<br>**Wofür:** Objekterkennung, starke Unbalance: 2. |
| `alpha` | Kommazahl (oder None) | `None` | Bereich 0.0 … 1.0 | Basis | **Was:** Zusätzliche Klassengewichtung (binär) bzw. Faktor.<br>**Nutzen:** Balanciert Klassen.<br>**Wofür:** 0.25 (RetinaNet). |
| `ignore_index` | Ganzzahl | `-100` |  | Experte | **Was:** Zu ignorierender Zielwert.<br>**Nutzen:** Padding.<br>**Wofür:** Selten. |

#### `loss:dice` — Stufe Basis

**Was:** Dice-Verlust.

**Nutzen:** Optimiert direkt die Überlappung von Masken.

**Wofür:** Segmentierung (Medizin).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `smooth` | Kommazahl | `1.0` | ≥ 0.0 | Fortgeschritten | **Was:** Glättungsterm gegen Division durch 0.<br>**Nutzen:** Stabilität bei leeren Masken.<br>**Wofür:** 1.0. |

#### `loss:kl_div` (Aliase: `kl`) — Stufe Fortgeschritten

**Was:** KL-Divergenz zwischen Verteilungen.

**Nutzen:** Angleichen an eine Zielverteilung.

**Wofür:** Distillation.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `temperature` | Kommazahl | `1.0` | ≥ 1e-06 | Basis | **Was:** Temperatur zum Weichzeichnen der Verteilungen.<br>**Nutzen:** Höher = mehr Information über Nebenklassen.<br>**Wofür:** Distillation: 2–5. |

#### `loss:cosine` — Stufe Fortgeschritten

**Was:** 1 - Kosinus-Ähnlichkeit.

**Nutzen:** Richtungsgleichheit von Vektoren.

**Wofür:** Embeddings.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |

#### `loss:info_nce` (Aliase: `nt_xent`, `contrastive`) — Stufe Fortgeschritten

**Was:** InfoNCE-Kontrastverlust.

**Nutzen:** Zieht zusammengehörige Paare zusammen, stößt andere ab.

**Wofür:** Selbstüberwachtes Lernen, CLIP.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `temperature` | Kommazahl | `0.1` | ≥ 1e-06 | Basis | **Was:** Temperatur der Kontrast-Softmax.<br>**Nutzen:** Kleiner = schärfere Trennung, schwerer zu trainieren.<br>**Wofür:** SimCLR: 0.1–0.5; CLIP: lernbar ~0.07. |
| `symmetric` | Ja/Nein | `True` |  | Basis | **Was:** Verlust in beide Richtungen (a→b und b→a).<br>**Nutzen:** Symmetrisches Lernsignal.<br>**Wofür:** CLIP-artig. |

#### `loss:hinge` — Stufe Fortgeschritten

**Was:** Hinge-Verlust (SVM-artig).

**Nutzen:** Maximiert die Trennmarge.

**Wofür:** Binäre Klassifikation.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |
| `margin` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Abstandsmarge.<br>**Nutzen:** Mindestabstand korrekt vs. falsch.<br>**Wofür:** 1.0. |

#### `loss:nll` — Stufe Fortgeschritten

**Was:** Negative Log-Likelihood (Eingabe: Log-Wahrscheinlichkeiten).

**Nutzen:** Wenn das Modell bereits log_softmax ausgibt.

**Wofür:** Spezialfälle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht dieses Terms im Gesamtverlust.<br>**Nutzen:** Balanciert mehrere Ziele gegeneinander.<br>**Wofür:** Kombinierte Verluste, Multi-Task.<br>**Tipp:** Negative Gewichte sind erlaubt (Maximierung) – mit Vorsicht. |
| `output_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel der Modellausgabe (bei dict-Ausgaben).<br>**Nutzen:** Wählt die passende Ausgabe.<br>**Wofür:** Multi-Task, Autoencoder. |
| `target_key` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Schlüssel des Ziels (bei dict-Zielen).<br>**Nutzen:** Wählt das passende Ziel.<br>**Wofür:** Multi-Task. |


### Registry `optimizer`

<a id="registry-optimizer"></a>

**Optimierer** – Verfahren, das Gradienten in Gewichtsänderungen übersetzt. Bestimmt Konvergenzgeschwindigkeit, Stabilität, Speicherbedarf und Generalisierung.

Fabrik-Signatur: `factory(args, param_groups) -> torch.optim.Optimizer`

#### `optimizer:sgd` — Stufe Basis

**Was:** Stochastischer Gradientenabstieg (mit Momentum/Nesterov).

**Nutzen:** Einfach, wenig Speicher, generalisiert oft am besten (CNNs).

**Wofür:** Bildklassifikation (ResNet), wenn genug Tuning-Zeit vorhanden.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Lernrate.<br>**Nutzen:** Schrittweite.<br>**Wofür:** SGD: 0.01–0.1. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `momentum` | Kommazahl | `0.9` | Bereich 0.0 … 1.0 | Basis | **Was:** Momentum: Anteil der letzten Bewegung, der beibehalten wird.<br>**Nutzen:** Glättet die Richtung, beschleunigt in Tälern.<br>**Wofür:** 0.9 Standard. |
| `dampening` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Dämpfung des Momentums.<br>**Nutzen:** Reduziert den Gradientenanteil im Momentum.<br>**Wofür:** Meist 0. |
| `nesterov` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Nesterov-Momentum (schaut voraus).<br>**Nutzen:** Oft etwas schnellere Konvergenz.<br>**Wofür:** Mit momentum > 0. |

#### `optimizer:adam` — Stufe Basis

**Was:** Adam: adaptive Schrittweite pro Parameter.

**Nutzen:** Robust, schnell, wenig Tuning.

**Wofür:** Allzweck; mit weight_decay besser AdamW nehmen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate – Schrittweite der Gewichtsänderungen.<br>**Nutzen:** Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.<br>**Wofür:** Jedes Training.<br>**Tipp:** Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `betas` | Liste[Kommazahl] | `[0.9, 0.999]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).<br>**Nutzen:** β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.<br>**Wofür:** LLMs: [0.9, 0.95]; Standard: [0.9, 0.999]. |
| `eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch ~0; größer = vorsichtigere Updates.<br>**Wofür:** fp16: 1e-6…1e-5. |
| `amsgrad` | Ja/Nein | `False` |  | Experte | **Was:** AMSGrad-Variante (monotones 2. Moment).<br>**Nutzen:** Theoretische Konvergenzgarantie, praktisch selten besser.<br>**Wofür:** Experimente. |
| `fused` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).<br>**Nutzen:** Schneller auf GPUs.<br>**Wofür:** Geschwindigkeit auf CUDA. |

#### `optimizer:adamw` — Stufe Basis

**Was:** AdamW: Adam mit entkoppeltem Gewichtsverfall.

**Nutzen:** De-facto-Standard für Transformer und die meisten modernen Netze.

**Wofür:** Fast alles.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate – Schrittweite der Gewichtsänderungen.<br>**Nutzen:** Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.<br>**Wofür:** Jedes Training.<br>**Tipp:** Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5. |
| `weight_decay` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Entkoppelter Gewichtsverfall (AdamW).<br>**Nutzen:** Korrekte Regularisierung für adaptive Optimierer – der Grund, AdamW statt Adam zu nehmen.<br>**Wofür:** 0.01…0.1 (LLMs 0.1). |
| `betas` | Liste[Kommazahl] | `[0.9, 0.999]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).<br>**Nutzen:** β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.<br>**Wofür:** LLMs: [0.9, 0.95]; Standard: [0.9, 0.999]. |
| `eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch ~0; größer = vorsichtigere Updates.<br>**Wofür:** fp16: 1e-6…1e-5. |
| `amsgrad` | Ja/Nein | `False` |  | Experte | **Was:** AMSGrad-Variante (monotones 2. Moment).<br>**Nutzen:** Theoretische Konvergenzgarantie, praktisch selten besser.<br>**Wofür:** Experimente. |
| `fused` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).<br>**Nutzen:** Schneller auf GPUs.<br>**Wofür:** Geschwindigkeit auf CUDA. |

#### `optimizer:nadam` — Stufe Fortgeschritten

**Was:** Adam mit Nesterov-Momentum.

**Nutzen:** Teils schneller.

**Wofür:** Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate – Schrittweite der Gewichtsänderungen.<br>**Nutzen:** Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.<br>**Wofür:** Jedes Training.<br>**Tipp:** Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `betas` | Liste[Kommazahl] | `[0.9, 0.999]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).<br>**Nutzen:** β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.<br>**Wofür:** LLMs: [0.9, 0.95]; Standard: [0.9, 0.999]. |
| `eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch ~0; größer = vorsichtigere Updates.<br>**Wofür:** fp16: 1e-6…1e-5. |
| `amsgrad` | Ja/Nein | `False` |  | Experte | **Was:** AMSGrad-Variante (monotones 2. Moment).<br>**Nutzen:** Theoretische Konvergenzgarantie, praktisch selten besser.<br>**Wofür:** Experimente. |
| `fused` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).<br>**Nutzen:** Schneller auf GPUs.<br>**Wofür:** Geschwindigkeit auf CUDA. |
| `momentum_decay` | Kommazahl | `0.004` | ≥ 0.0 | Experte | **Was:** Momentum-Zerfall von NAdam.<br>**Nutzen:** Steuert Nesterov-Anteil über die Zeit.<br>**Wofür:** Selten. |

#### `optimizer:radam` — Stufe Fortgeschritten

**Was:** Rectified Adam (Varianzkorrektur am Anfang).

**Nutzen:** Weniger Warmup nötig.

**Wofür:** Wenn Warmup stört.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate – Schrittweite der Gewichtsänderungen.<br>**Nutzen:** Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.<br>**Wofür:** Jedes Training.<br>**Tipp:** Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `betas` | Liste[Kommazahl] | `[0.9, 0.999]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).<br>**Nutzen:** β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.<br>**Wofür:** LLMs: [0.9, 0.95]; Standard: [0.9, 0.999]. |
| `eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch ~0; größer = vorsichtigere Updates.<br>**Wofür:** fp16: 1e-6…1e-5. |
| `amsgrad` | Ja/Nein | `False` |  | Experte | **Was:** AMSGrad-Variante (monotones 2. Moment).<br>**Nutzen:** Theoretische Konvergenzgarantie, praktisch selten besser.<br>**Wofür:** Experimente. |
| `fused` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).<br>**Nutzen:** Schneller auf GPUs.<br>**Wofür:** Geschwindigkeit auf CUDA. |

#### `optimizer:adamax` — Stufe Fortgeschritten

**Was:** Adam mit ∞-Norm.

**Nutzen:** Robust gegen seltene große Gradienten.

**Wofür:** Embeddings.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate – Schrittweite der Gewichtsänderungen.<br>**Nutzen:** Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.<br>**Wofür:** Jedes Training.<br>**Tipp:** Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `betas` | Liste[Kommazahl] | `[0.9, 0.999]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).<br>**Nutzen:** β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.<br>**Wofür:** LLMs: [0.9, 0.95]; Standard: [0.9, 0.999]. |
| `eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch ~0; größer = vorsichtigere Updates.<br>**Wofür:** fp16: 1e-6…1e-5. |
| `amsgrad` | Ja/Nein | `False` |  | Experte | **Was:** AMSGrad-Variante (monotones 2. Moment).<br>**Nutzen:** Theoretische Konvergenzgarantie, praktisch selten besser.<br>**Wofür:** Experimente. |
| `fused` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).<br>**Nutzen:** Schneller auf GPUs.<br>**Wofür:** Geschwindigkeit auf CUDA. |

#### `optimizer:adagrad` — Stufe Fortgeschritten

**Was:** Adagrad: Lernrate sinkt mit der Gradientensumme.

**Nutzen:** Gut für spärliche Merkmale.

**Wofür:** Empfehlung, NLP-Klassiker.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Lernrate.<br>**Nutzen:** Schrittweite.<br>**Wofür:** 0.01. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `lr_decay` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Lernratenzerfall.<br>**Nutzen:** Zusätzliches Abklingen.<br>**Wofür:** Meist 0. |
| `initial_accumulator_value` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Startwert der Gradienten-Summe.<br>**Nutzen:** Dämpft frühe Schritte.<br>**Wofür:** 0–0.1. |
| `eps` | Kommazahl | `1e-10` | ≥ 0.0 | Experte | **Was:** Stabilisator.<br>**Nutzen:** Gegen Division durch 0.<br>**Wofür:** Selten. |

#### `optimizer:adadelta` — Stufe Fortgeschritten

**Was:** Adadelta.

**Nutzen:** Kaum Lernraten-Tuning.

**Wofür:** Klassiker.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Skalierungsfaktor (Adadelta ist lernratenarm).<br>**Nutzen:** Meist 1.<br>**Wofür:** 1.0. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `rho` | Kommazahl | `0.9` | Bereich 0.0 … 1.0 | Basis | **Was:** Abklingrate.<br>**Nutzen:** Fenster der Gradientenstatistik.<br>**Wofür:** 0.9. |
| `eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Stabilisator.<br>**Nutzen:** Gegen Division durch 0.<br>**Wofür:** Selten. |

#### `optimizer:rmsprop` — Stufe Basis

**Was:** RMSprop.

**Nutzen:** Adaptive Skalierung ohne Momentum-Bias-Korrektur.

**Wofür:** RNNs, A2C/DQN-Klassiker.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate.<br>**Nutzen:** Schrittweite.<br>**Wofür:** 1e-3; RL: 7e-4. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `alpha` | Kommazahl | `0.99` | Bereich 0.0 … 1.0 | Basis | **Was:** Glättung des 2. Moments.<br>**Nutzen:** Gedächtnis der Skalierung.<br>**Wofür:** 0.99. |
| `eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator.<br>**Nutzen:** Gegen Division durch 0.<br>**Wofür:** RL (A2C): 1e-5. |
| `momentum` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Momentum.<br>**Nutzen:** Glättung der Richtung.<br>**Wofür:** 0. |
| `centered` | Ja/Nein | `False` |  | Experte | **Was:** Zentrierte Variante (normiert mit Varianz statt 2. Moment).<br>**Nutzen:** Manchmal stabiler.<br>**Wofür:** Experimente. |

#### `optimizer:adafactor` — Stufe Fortgeschritten

**Was:** Adafactor: faktorisierte 2. Momente.

**Nutzen:** Viel weniger Optimierer-Speicher als Adam.

**Wofür:** Sehr große Modelle (T5).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Lernrate.<br>**Nutzen:** Schrittweite.<br>**Wofür:** 1e-2 (relativ). |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `beta2_decay` | Kommazahl | `-0.8` | ≤ 0.0 | Experte | **Was:** Exponent des β₂-Zeitplans.<br>**Nutzen:** Steuert das Gedächtnis des 2. Moments.<br>**Wofür:** -0.8. |
| `d` | Kommazahl | `1.0` | ≥ 0.0 | Experte | **Was:** Clipping-Schwelle der Update-RMS.<br>**Nutzen:** Begrenzt Ausreißer-Updates.<br>**Wofür:** 1.0. |

#### `optimizer:lion` — Stufe Fortgeschritten

**Was:** Lion: Vorzeichen-Momentum.

**Nutzen:** Halber Optimierer-Speicher von Adam, oft gleich gut.

**Wofür:** Große Modelle, Bild-Transformer.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.0001` | ≥ 0.0 | Basis | **Was:** Lernrate (Lion braucht ~3–10× kleinere Lernraten als AdamW).<br>**Nutzen:** Schrittweite.<br>**Wofür:** 1e-4…3e-4. |
| `weight_decay` | Kommazahl | `0.1` | ≥ 0.0 | Basis | **Was:** Entkoppelter Gewichtsverfall (Lion braucht ~3–10× größeren WD als AdamW).<br>**Nutzen:** Regularisierung.<br>**Wofür:** 0.1…1. |
| `betas` | Liste[Kommazahl] | `[0.9, 0.99]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** β₁ (Interpolation für Update), β₂ (Momentum-Update).<br>**Nutzen:** Glättung.<br>**Wofür:** [0.9, 0.99]. |

#### `optimizer:adamw8bit` (Aliase: `adam8bit`) — Stufe Fortgeschritten

**Was:** AdamW mit 8-Bit-Zuständen (blockweise logarithmisch quantisiert).

**Nutzen:** Optimierer-Speicher sinkt von 8 auf ~2 Byte pro Parameter – bei großen Modellen oft der größte Posten.

**Wofür:** Große Modelle / Feintuning mit knappem Speicher; Verhalten praktisch wie AdamW.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Lernrate – Schrittweite der Gewichtsänderungen.<br>**Nutzen:** Wichtigster Hyperparameter: zu groß → Divergenz, zu klein → quälend langsam.<br>**Wofür:** Jedes Training.<br>**Tipp:** Adam/AdamW: 1e-4…1e-3 (Transformer 3e-4); SGD: 0.01…0.1; Feintuning: 1e-5. |
| `weight_decay` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Entkoppelter Gewichtsverfall (AdamW).<br>**Nutzen:** Korrekte Regularisierung für adaptive Optimierer – der Grund, AdamW statt Adam zu nehmen.<br>**Wofür:** 0.01…0.1 (LLMs 0.1). |
| `betas` | Liste[Kommazahl] | `[0.9, 0.999]` | Bereich 0.0 … 1.0 · Länge 2 | Fortgeschritten | **Was:** Abklingraten für 1. Moment (Richtung) und 2. Moment (Skalierung).<br>**Nutzen:** β₂ kleiner (0.95) stabilisiert große Modelle; β₁ steuert die Glättung.<br>**Wofür:** LLMs: [0.9, 0.95]; Standard: [0.9, 0.999]. |
| `eps` | Kommazahl | `1e-08` | ≥ 0.0 | Experte | **Was:** Stabilisator im Nenner.<br>**Nutzen:** Verhindert Division durch ~0; größer = vorsichtigere Updates.<br>**Wofür:** fp16: 1e-6…1e-5. |
| `amsgrad` | Ja/Nein | `False` |  | Experte | **Was:** AMSGrad-Variante (monotones 2. Moment).<br>**Nutzen:** Theoretische Konvergenzgarantie, praktisch selten besser.<br>**Wofür:** Experimente. |
| `fused` | Ja/Nein (oder None) | `None` |  | Experte | **Was:** Fusionierte CUDA-Implementierung (None = PyTorch entscheidet).<br>**Nutzen:** Schneller auf GPUs.<br>**Wofür:** Geschwindigkeit auf CUDA. |
| `block_size` | Ganzzahl | `256` | ≥ 16 | Experte | **Was:** Blockgröße der 8-Bit-Quantisierung der Optimierer-Zustände.<br>**Nutzen:** Jeder Block hat eine eigene Skala – kleiner = genauer, minimal mehr Speicher.<br>**Wofür:** 256. |
| `min_8bit_size` | Ganzzahl | `4096` | ≥ 0 | Experte | **Was:** Parameter mit weniger Elementen behalten 32-Bit-Zustände.<br>**Nutzen:** Kleine Tensoren (Biases, Normen) sparen kaum, sind aber empfindlich.<br>**Wofür:** 4096. |
| `dynamic_range` | Kommazahl | `14.0` | Bereich 2.0 … 40.0 | Experimentell | **Was:** Dynamikbereich der logarithmischen Quantisierung (natürlicher Log; 14 ≈ Faktor 10⁶ pro Block).<br>**Nutzen:** Größer = auch sehr kleine Werte darstellbar, aber gröbere Stufen.<br>**Wofür:** 14 (m), doppelt für v. |

#### `optimizer:muon` — Stufe Experimentell

**Was:** Muon: orthogonalisierte Momentum-Updates für Gewichtsmatrizen (+ AdamW für den Rest).

**Nutzen:** Hat in Speedrun-Experimenten Transformer schneller trainiert als AdamW.

**Wofür:** Transformer-Vortraining, Forschung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `lr` | Kommazahl | `0.02` | ≥ 0.0 | Basis | **Was:** Lernrate für Matrix-Parameter (Muon).<br>**Nutzen:** Schrittweite der orthogonalisierten Updates.<br>**Wofür:** 0.02. |
| `weight_decay` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewichtsverfall: zieht Gewichte in Richtung 0.<br>**Nutzen:** Regularisiert gegen Überanpassung, hält Gewichte klein.<br>**Wofür:** AdamW: 0.01…0.1; SGD: 1e-4…5e-4.<br>**Tipp:** Biases/Normen werden standardmäßig ausgenommen (training.no_decay_patterns). |
| `momentum` | Kommazahl | `0.95` | Bereich 0.0 … 1.0 | Basis | **Was:** Momentum.<br>**Nutzen:** Glättung.<br>**Wofür:** 0.95. |
| `nesterov` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Nesterov-Momentum.<br>**Nutzen:** Vorausschauende Richtung.<br>**Wofür:** an. |
| `ns_steps` | Ganzzahl | `5` | Bereich 1 … 20 | Experte | **Was:** Newton-Schulz-Iterationen zur Orthogonalisierung.<br>**Nutzen:** Genauigkeit vs. Rechenaufwand.<br>**Wofür:** 5. |
| `adamw_lr` | Kommazahl | `0.0003` | ≥ 0.0 | Basis | **Was:** Lernrate für Nicht-Matrix-Parameter (Biases, Normen, Embeddings), die mit AdamW laufen.<br>**Nutzen:** Muon ist nur für 2D-Gewichte gedacht.<br>**Wofür:** 3e-4. |
| `adamw_betas` | Liste[Kommazahl] | `[0.9, 0.95]` | Bereich 0.0 … 1.0 · Länge 2 | Experte | **Was:** AdamW-Betas für Nicht-Matrix-Parameter.<br>**Nutzen:** Siehe Adam.<br>**Wofür:** [0.9, 0.95]. |
| `treat_embeddings_as_matrix` | Ja/Nein | `False` |  | Experimentell | **Was:** Auch Embedding-Tabellen mit Muon optimieren.<br>**Nutzen:** Empfehlung des Originals: nein (AdamW).<br>**Wofür:** Experimente. |


### Registry `paradigm`

<a id="registry-paradigm"></a>

**Trainings-Paradigmen** – Art des Lernens (überwacht, generativ, selbstüberwacht, präferenzbasiert, RL ...). Bestimmt, wie aus Daten ein Lernsignal entsteht.

Fabrik-Signatur: `factory(args) -> Paradigm`

#### `paradigm:supervised` (Aliase: `ueberwacht`) — Stufe Basis

**Was:** Überwachtes Lernen: Eingabe → Ziel.

**Nutzen:** Der Klassiker für Klassifikation und Regression.

**Wofür:** Tabellen, Bilder, Zeitreihen, Multi-Task.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `task` | Text | `'auto'` | Auswahl: auto \| classification \| regression \| multilabel | Basis | **Was:** Art der überwachten Aufgabe (nur relevant, wenn keine eigenen Verluste gesetzt sind).<br>**Nutzen:** Bestimmt Standardverlust und Metriken.<br>**Wofür:** Meist 'auto'. |

#### `paradigm:causal_lm` (Aliase: `language_modeling`, `lm`) — Stufe Basis

**Was:** Kausales Sprachmodell-Training (nächstes Token).

**Nutzen:** Selbstüberwacht auf Rohtext.

**Wofür:** GPT-artige Modelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `pad_token_id` | Ganzzahl (oder None) | `None` |  | Basis | **Was:** Padding-Token, das im Verlust ignoriert wird.<br>**Nutzen:** Unterschiedlich lange Sequenzen.<br>**Wofür:** Wenn gepaddet wird. |
| `z_loss_coef` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** z-Loss: bestraft große log Σ exp(Logits) (PaLM).<br>**Nutzen:** Stabilisiert große Sprachmodelle numerisch.<br>**Wofür:** Große LMs: 1e-4. |
| `label_smoothing` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Label Smoothing auf die Token-Ziele.<br>**Nutzen:** Weniger Überkonfidenz.<br>**Wofür:** Selten bei LMs. |

#### `paradigm:masked_lm` (Aliase: `mlm`) — Stufe Basis

**Was:** Maskiertes Sprachmodell (BERT).

**Nutzen:** Bidirektionales Vortraining.

**Wofür:** Encoder-Modelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `mask_token_id` | Ganzzahl (oder None) | `None` |  | Basis | **Was:** ID des [MASK]-Tokens (None = letzte Vokabular-ID).<br>**Nutzen:** Ersetzt maskierte Tokens.<br>**Wofür:** Je nach Tokenizer. |
| `mask_prob` | Kommazahl | `0.15` | Bereich 0.0 … 1.0 | Basis | **Was:** Anteil maskierter Tokens.<br>**Nutzen:** Mehr = schwierigere Aufgabe, mehr Lernsignal pro Beispiel.<br>**Wofür:** BERT: 0.15. |
| `random_ratio` | Kommazahl | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil der ausgewählten Tokens, die durch zufällige Tokens ersetzt werden.<br>**Nutzen:** Robustheit.<br>**Wofür:** 0.1. |
| `keep_ratio` | Kommazahl | `0.1` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil der ausgewählten Tokens, die unverändert bleiben.<br>**Nutzen:** Verhindert Train/Test-Diskrepanz.<br>**Wofür:** 0.1. |
| `pad_token_id` | Ganzzahl (oder None) | `None` |  | Basis | **Was:** Padding-Token (wird nie maskiert).<br>**Nutzen:** Nur echte Tokens.<br>**Wofür:** Bei Padding. |
| `vocab_size` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** Vokabulargröße (None = aus der Modellausgabe).<br>**Nutzen:** Für zufällige Ersatz-Tokens.<br>**Wofür:** Meist None. |

#### `paradigm:seq2seq` — Stufe Basis

**Was:** Sequenz-zu-Sequenz mit Teacher Forcing.

**Nutzen:** Eingabe- und Ausgabesequenz verschieden.

**Wofür:** Übersetzung, Zusammenfassung, Mehrschritt-Prognosen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `bos_token_id` | Ganzzahl | `0` |  | Basis | **Was:** Start-Token, mit dem der Decoder beginnt (nur Token-Ziele).<br>**Nutzen:** Teacher Forcing: Ziele um eins verschoben.<br>**Wofür:** Je nach Tokenizer. |
| `pad_token_id` | Ganzzahl (oder None) | `None` |  | Basis | **Was:** Padding im Ziel (wird ignoriert).<br>**Nutzen:** Unterschiedliche Längen.<br>**Wofür:** Bei Padding. |
| `label_smoothing` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Label Smoothing.<br>**Nutzen:** Bessere Übersetzungen (0.1).<br>**Wofür:** Übersetzung. |

#### `paradigm:autoencoder` (Aliase: `reconstruction`) — Stufe Basis

**Was:** Rekonstruktionstraining (+ KL-Annealing).

**Nutzen:** Unüberwacht.

**Wofür:** Autoencoder, VAEs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `reconstruction_loss` | Text | `'mse'` | Auswahl: mse \| bce \| l1 | Basis | **Was:** Rekonstruktionsverlust.<br>**Nutzen:** Passend zum Datentyp.<br>**Wofür:** Bilder in [0,1]: bce; sonst mse. |
| `reduction` | Text | `'auto'` | Auswahl: auto \| mean \| sum | Fortgeschritten | **Was:** Reduktion des Rekonstruktionsverlusts.<br>**Nutzen:** Beim VAE muss die Rekonstruktion je Beispiel summiert werden, sonst erdrückt der KL-Term sie (Posterior Collapse: Latentraum wird ignoriert).<br>**Wofür:** auto. |
| `kl_schedule` | Text | `'linear'` | Auswahl: constant \| linear \| cyclical | Fortgeschritten | **Was:** KL-Annealing (nur VAE).<br>**Nutzen:** Verhindert Posterior Collapse zu Beginn.<br>**Wofür:** linear. |
| `kl_warmup_steps` | Ganzzahl | `500` | ≥ 0 | Fortgeschritten | **Was:** Schritte bis β voll erreicht ist (bzw. Zykluslänge).<br>**Nutzen:** Dauer des Annealings.<br>**Wofür:** 500–10000. |

#### `paradigm:contrastive` (Aliase: `simclr`) — Stufe Fortgeschritten

**Was:** Kontrastives selbstüberwachtes Lernen.

**Nutzen:** Repräsentationen ohne Labels.

**Wofür:** Vortraining, Embeddings.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `temperature` | Kommazahl | `0.1` | ≥ 0.0001 | Basis | **Was:** Temperatur des InfoNCE-Verlusts.<br>**Nutzen:** Schärfe der Unterscheidung.<br>**Wofür:** 0.05–0.5. |
| `noise_std` | Kommazahl | `0.1` | ≥ 0.0 | Basis | **Was:** Gauß-Rauschen als Augmentierung (Gleitkomma-Eingaben).<br>**Nutzen:** Erzeugt zwei Sichten.<br>**Wofür:** 0.05–0.2. |
| `feature_dropout` | Kommazahl | `0.1` | Bereich 0.0 … 0.9 | Basis | **Was:** Zufälliges Nullsetzen von Eingabewerten als Augmentierung.<br>**Nutzen:** Zweite Sicht-Variation.<br>**Wofür:** 0.1–0.3. |
| `token_mask_prob` | Kommazahl | `0.1` | Bereich 0.0 … 0.9 | Fortgeschritten | **Was:** Anteil zufällig ersetzter Tokens (Ganzzahl-Eingaben).<br>**Nutzen:** Augmentierung für Text.<br>**Wofür:** 0.1. |
| `mask_token_id` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Ersatz-Token für token_mask_prob.<br>**Nutzen:** Augmentierung.<br>**Wofür:** 0 oder [MASK]. |

#### `paradigm:distillation` (Aliase: `kd`) — Stufe Fortgeschritten

**Was:** Wissensdestillation von einem Lehrer.

**Nutzen:** Kleine Modelle mit großem Wissen.

**Wofür:** Kompression.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `teacher` | Komponente aus Registry 'model' (oder None) | None | Auswahl: autoencoder \| cnn \| ensemble \| pipeline \| fusion \| mixture_of_models \| multitask \| mlp \| gan \| diffusion \| gnn \| graph \| actor_critic \| … | Basis | **Was:** Lehrer-Modell (beliebiges Modell; meist größer).<br>**Nutzen:** Liefert 'weiche' Ziele mit mehr Information als Labels.<br>**Wofür:** Modellkompression. |
| `teacher_checkpoint` | Text (oder None) | `None` |  | Basis | **Was:** Pfad zu trainierten Lehrer-Gewichten (.pt).<br>**Nutzen:** Ohne Gewichte lehrt der Lehrer Zufall!<br>**Wofür:** Pflicht in der Praxis. |
| `temperature` | Kommazahl | `2.0` | ≥ 0.001 | Basis | **Was:** Temperatur zum Weichzeichnen der Lehrer-Verteilung.<br>**Nutzen:** Macht 'dunkles Wissen' über Nebenklassen sichtbar.<br>**Wofür:** 2–5. |
| `alpha` | Kommazahl | `0.5` | Bereich 0.0 … 1.0 | Basis | **Was:** Gewicht des Distillationsverlusts (1-α für echte Labels).<br>**Nutzen:** Balance Lehrer vs. Daten.<br>**Wofür:** 0.5–0.9. |
| `kd_loss` | Text | `'kl'` | Auswahl: kl \| mse | Basis | **Was:** Art des Distillationsverlusts.<br>**Nutzen:** kl für Klassifikation; mse für Regression/Logits.<br>**Wofür:** kl. |

#### `paradigm:gan` — Stufe Basis

**Was:** Adversariales Training (Generator vs. Diskriminator).

**Nutzen:** Generative Modelle.

**Wofür:** Modell 'gan'.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `loss` | Text | `'non_saturating'` | Auswahl: standard \| non_saturating \| wasserstein \| hinge \| least_squares | Basis | **Was:** GAN-Verlustart.<br>**Nutzen:** Großer Einfluss auf Stabilität und Mode-Collapse.<br>**Wofür:** non_saturating oder hinge. |
| `n_critic` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Diskriminator-Schritte pro Generator-Schritt.<br>**Nutzen:** WGAN braucht einen starken Kritiker.<br>**Wofür:** 1; WGAN: 5. |
| `gradient_penalty` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Gewicht der Gradientenstrafe (WGAN-GP).<br>**Nutzen:** Erzwingt Lipschitz-Stetigkeit des Kritikers.<br>**Wofür:** WGAN-GP: 10. |
| `r1_penalty` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** R1-Strafe (Gradient auf echten Daten).<br>**Nutzen:** Stabilisiert (StyleGAN).<br>**Wofür:** 1–10. |
| `label_smoothing` | Kommazahl | `0.0` | Bereich 0.0 … 0.5 | Fortgeschritten | **Was:** Einseitiges Label Smoothing für echte Daten.<br>**Nutzen:** Weniger überkonfidenter Diskriminator.<br>**Wofür:** 0.1. |
| `instance_noise_std` | Kommazahl | `0.0` | ≥ 0.0 | Experte | **Was:** Rauschen auf Diskriminator-Eingaben (klingt linear ab).<br>**Nutzen:** Überlappende Verteilungen, stabileres Training.<br>**Wofür:** 0.1. |
| `instance_noise_steps` | Ganzzahl | `5000` | ≥ 1 | Experte | **Was:** Schritte bis das Instanzrauschen 0 ist.<br>**Nutzen:** Abklingdauer.<br>**Wofür:** Trainingslänge/2. |

#### `paradigm:diffusion` — Stufe Basis

**Was:** Diffusions-Training (Rauschen vorhersagen).

**Nutzen:** Stabil, hohe Qualität.

**Wofür:** Modell 'diffusion'.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `loss` | Text | `'mse'` | Auswahl: mse \| l1 \| huber | Basis | **Was:** Verlust zwischen Vorhersage und Ziel.<br>**Nutzen:** mse ist Standard; huber robuster.<br>**Wofür:** mse. |
| `weighting` | Text | `'none'` | Auswahl: none \| min_snr \| snr | Fortgeschritten | **Was:** Gewichtung der Zeitschritte.<br>**Nutzen:** Balanciert leichte/schwere Rauschstufen.<br>**Wofür:** min_snr. |
| `min_snr_gamma` | Kommazahl | `5.0` | ≥ 0.0 | Experte | **Was:** γ für Min-SNR.<br>**Nutzen:** Obergrenze der Gewichte.<br>**Wofür:** 5. |

#### `paradigm:preference` (Aliase: `reward_model`, `bradley_terry`) — Stufe Fortgeschritten

**Was:** Reward-Modell aus Präferenzpaaren.

**Nutzen:** Lernt menschliche Vorlieben als Belohnung.

**Wofür:** RLHF-Pipelines.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `margin` | Kommazahl | `0.0` |  | Basis | **Was:** Mindestabstand der Belohnungen bevorzugt vs. abgelehnt.<br>**Nutzen:** Erzwingt deutlichere Trennung.<br>**Wofür:** 0–1. |
| `center_coef` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** Strafe auf (r_c + r_r)² – zentriert Belohnungen um 0.<br>**Nutzen:** Verhindert Drift der Belohnungsskala.<br>**Wofür:** 1e-3. |

#### `paradigm:dpo` — Stufe Fortgeschritten

**Was:** Direct Preference Optimization.

**Nutzen:** Präferenzlernen ohne separates Reward-Modell/RL.

**Wofür:** Ausrichtung von Sprachmodellen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `beta` | Kommazahl | `0.1` | ≥ 0.0 | Basis | **Was:** Stärke der Bindung an das Referenzmodell (β).<br>**Nutzen:** Klein = stärkere Anpassung an Präferenzen.<br>**Wofür:** 0.1–0.5. |
| `variant` | Text | `'sigmoid'` | Auswahl: sigmoid \| ipo \| hinge | Basis | **Was:** DPO-Verlustvariante.<br>**Nutzen:** Unterschiedliche Robustheit.<br>**Wofür:** sigmoid. |
| `label_smoothing` | Kommazahl | `0.0` | Bereich 0.0 … 0.5 | Fortgeschritten | **Was:** Annahme verrauschter Präferenzen (cDPO).<br>**Nutzen:** Robust gegen falsch gelabelte Paare.<br>**Wofür:** 0–0.1. |
| `reference_checkpoint` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Gewichte des Referenzmodells (None = Kopie des Start-Modells).<br>**Nutzen:** Ankerpunkt der Optimierung.<br>**Wofür:** Meist None. |

#### `paradigm:reinforcement` (Aliase: `rl`) — Stufe Basis

**Was:** Reinforcement Learning (PPO, A2C, REINFORCE, DQN) mit frei kombinierbaren Belohnungen.

**Nutzen:** Lernen durch Interaktion statt aus festen Daten.

**Wofür:** Steuerung, Spiele, Robotik.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `algorithm` | Komponente aus Registry 'rl_algorithm' | `ppo` | Auswahl: ppo \| a2c \| reinforce \| dqn | Basis | **Was:** RL-Algorithmus.<br>**Nutzen:** PPO (robust), A2C (schnell), REINFORCE (einfach), DQN (dateneffizient, diskret).<br>**Wofür:** Standard: ppo; diskret mit wenig Daten: dqn. |
| `algorithm.n_steps` | Ganzzahl | `128` | ≥ 1 | Basis | **Was:** Schritte pro Umgebung je Rollout.<br>**Nutzen:** Länger = bessere Schätzungen, seltenere Updates.<br>**Wofür:** PPO: 128–2048; A2C: 5. |
| `algorithm.value_coef` | Kommazahl | `0.5` | ≥ 0.0 | Basis | **Was:** Gewicht des Value-Verlusts.<br>**Nutzen:** Balanciert Kritiker- gegen Policy-Lernen.<br>**Wofür:** 0.5. |
| `algorithm.entropy_coef` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Entropie-Bonus.<br>**Nutzen:** Fördert Erkundung, verhindert vorzeitig deterministische Policies.<br>**Wofür:** 0–0.05. |
| `algorithm.gae_lambda` | Kommazahl | `0.95` | Bereich 0.0 … 1.0 | Basis | **Was:** λ der Generalized Advantage Estimation.<br>**Nutzen:** Bias-Varianz-Abwägung (1 = Monte-Carlo, 0 = TD).<br>**Wofür:** 0.9–0.99. |
| `algorithm.normalize_advantages` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Vorteile pro Batch standardisieren.<br>**Nutzen:** Stabilere Gradienten.<br>**Wofür:** PPO: an. |
| `algorithm.n_epochs` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Durchläufe über jeden Rollout.<br>**Nutzen:** Mehr = dateneffizienter, aber Gefahr zu großer Policy-Sprünge.<br>**Wofür:** 3–10. |
| `algorithm.minibatch_size` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Minibatch-Größe.<br>**Nutzen:** Gradienten-Rauschen vs. Geschwindigkeit.<br>**Wofür:** 64–512. |
| `algorithm.clip_range` | Kommazahl | `0.2` | ≥ 0.0 | Basis | **Was:** PPO-Clipping ε des Wahrscheinlichkeitsverhältnisses.<br>**Nutzen:** Begrenzt Policy-Änderungen pro Update – Kern von PPO.<br>**Wofür:** 0.1–0.3. |
| `algorithm.clip_value` | Kommazahl (oder None) | `None` | ≥ 0.0 | Fortgeschritten | **Was:** Clipping der Value-Updates (None = aus).<br>**Nutzen:** Stabilisiert den Kritiker.<br>**Wofür:** 0.2 oder None. |
| `algorithm.target_kl` | Kommazahl (oder None) | `None` | ≥ 0.0 | Fortgeschritten | **Was:** Abbruch der Epochen, wenn die KL-Divergenz diesen Wert überschreitet.<br>**Nutzen:** Schutz vor Policy-Kollaps.<br>**Wofür:** 0.01–0.05. |
| `environment` | Komponente aus Registry 'environment' | `cartpole` | Auswahl: gridworld \| cartpole \| pendulum \| bandit \| gymnasium | Basis | **Was:** Umgebung, in der der Agent lernt.<br>**Nutzen:** Definiert Aufgabe, Beobachtungen und Aktionen.<br>**Wofür:** Je nach Problem. |
| `environment.max_steps` | Ganzzahl | `200` | ≥ 1 | Basis | **Was:** Episodenlänge (Erfolg = so lange balancieren).<br>**Nutzen:** Schwierigkeit.<br>**Wofür:** 200/500. |
| `environment.gravity` | Kommazahl | `9.8` | ≥ 0.0 | Fortgeschritten | **Was:** Schwerkraft.<br>**Nutzen:** Physik-Parameter.<br>**Wofür:** 9.8. |
| `environment.pole_length` | Kommazahl | `0.5` | ≥ 0.05 | Fortgeschritten | **Was:** Halbe Stablänge.<br>**Nutzen:** Kürzer = schwieriger.<br>**Wofür:** 0.5. |
| `environment.force` | Kommazahl | `10.0` | ≥ 0.1 | Fortgeschritten | **Was:** Stoßkraft je Aktion.<br>**Nutzen:** Steuerstärke.<br>**Wofür:** 10. |
| `environment.angle_limit_deg` | Kommazahl | `12.0` | Bereich 1.0 … 90.0 | Fortgeschritten | **Was:** Abbruchwinkel in Grad.<br>**Nutzen:** Toleranz.<br>**Wofür:** 12. |
| `environment.noise` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** Rauschen auf die Kraft.<br>**Nutzen:** Stochastische Dynamik.<br>**Wofür:** 0–2. |
| `num_envs` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Parallele Umgebungen.<br>**Nutzen:** Mehr = vielfältigere, dekorrelierte Erfahrungen pro Update.<br>**Wofür:** 4–16. |
| `total_timesteps` | Ganzzahl | `20000` | ≥ 1 | Basis | **Was:** Gesamtzahl Umgebungsschritte (über alle Umgebungen).<br>**Nutzen:** Trainingsbudget.<br>**Wofür:** CartPole: 2·10⁴–10⁵. |
| `gamma` | Kommazahl | `0.99` | Bereich 0.0 … 1.0 | Basis | **Was:** Diskontfaktor: wie viel zukünftige Belohnungen zählen.<br>**Nutzen:** Nahe 1 = weitsichtig (lange Horizonte), kleiner = kurzsichtig, stabiler.<br>**Wofür:** 0.95–0.999. |
| `rewards` | Liste von Komponenten aus Registry 'reward' | 1 Elemente | Elemente: env \| constant \| action_penalty \| action_change \| info \| potential \| event \| terminal \| rnd \| count | Basis | **Was:** Belohnungsterme (gewichtete Summe) – frei kombinierbar.<br>**Nutzen:** Die Belohnung bestimmt das gelernte Verhalten.<br>**Wofür:** Basis 'env' + Formung/Strafen/Neugier nach Bedarf. |
| `rewards.0` | beliebig | `reward:env()` |  | Basis | **Was:** Belohnung der Umgebung (die eigentliche Aufgabe).<br>**Nutzen:** Basis fast jeder Belohnungsfunktion.<br>**Wofür:** Immer sinnvoll. |
| `rewards.0.weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `rewards.0.kind` | Text | `'extrinsisch'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art der Belohnung (Einordnung + getrennte Protokollierung).<br>**Nutzen:** Übersicht, welcher Anteil wofür sorgt.<br>**Wofür:** Analyse von Reward-Hacking. |
| `rewards.0.label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `rewards.0.clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `rewards.0.clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `rewards.0.start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `rewards.0.end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `rewards.0.anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `rewards.0.anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `reward_normalization` | Text | `'none'` | Auswahl: none \| running \| return | Fortgeschritten | **Was:** Belohnungsnormalisierung.<br>**Nutzen:** Stabile Skala trotz unterschiedlich großer Belohnungen.<br>**Wofür:** PPO: return. |
| `reward_clip` | Kommazahl (oder None) | `None` | ≥ 0.0 | Fortgeschritten | **Was:** Gesamtbelohnung auf [-c, c] begrenzen.<br>**Nutzen:** Schutz vor Ausreißern (Atari: 1).<br>**Wofür:** Optional. |
| `observation_normalization` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Beobachtungen laufend standardisieren.<br>**Nutzen:** Wichtig bei unterschiedlich skalierten Sensoren.<br>**Wofür:** Robotik: an. |
| `eval_episodes` | Ganzzahl | `5` | ≥ 1 | Basis | **Was:** Episoden je Auswertung.<br>**Nutzen:** Genauigkeit der Bewertung.<br>**Wofür:** 5–20. |
| `eval_deterministic` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Bei der Auswertung immer die wahrscheinlichste Aktion wählen (sonst wie im Training ziehen).<br>**Nutzen:** Deterministisch zeigt das 'beste' Verhalten – kann aber in Schleifen hängen bleiben (z. B. an Wänden).<br>**Wofür:** Gitterwelten mit Wänden: aus; Regelungsaufgaben: an. |
| `eval_every_env_steps` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Auswertung alle n Umgebungsschritte (None = nur am Ende).<br>**Nutzen:** Lernkurve / Early Stopping (monitor val_return).<br>**Wofür:** z. B. 5000. |
| `log_every_env_steps` | Ganzzahl | `2000` | ≥ 1 | Basis | **Was:** Protokoll alle n Umgebungsschritte.<br>**Nutzen:** Übersicht.<br>**Wofür:** 1000–10000. |


### Registry `scheduler`

<a id="registry-scheduler"></a>

**Lernraten-Zeitpläne** – Verlauf der Lernrate über das Training. Warmup verhindert frühe Instabilität, Abklingen verbessert die Endqualität.

Fabrik-Signatur: `factory(args, optimizer, total_steps)`

#### `scheduler:constant` (Aliase: `none`, `konstant`) — Stufe Basis

**Was:** Konstante Lernrate (optional mit Warmup).

**Nutzen:** Einfach, gut zum Debuggen und für kurze Trainings.

**Wofür:** Prototypen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Optional auch bei konstanter Lernrate.<br>**Wofür:** 0 oder 0.05. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |

#### `scheduler:linear` (Aliase: `warmup_linear`) — Stufe Basis

**Was:** Warmup, dann linear auf min_lr fallen.

**Nutzen:** Einfach und wirksam.

**Wofür:** BERT-Feintuning.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Skaliert automatisch mit der Trainingslänge.<br>**Wofür:** 0.01–0.1. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |

#### `scheduler:warmup_cosine` (Aliase: `cosine`) — Stufe Basis

**Was:** Warmup, dann Kosinus-Abfall.

**Nutzen:** Weiches Abklingen, Standard moderner Trainings.

**Wofür:** Transformer, ViT, LLMs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Skaliert automatisch mit der Trainingslänge.<br>**Wofür:** 0.01–0.1. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |

#### `scheduler:cosine_restarts` (Aliase: `sgdr`) — Stufe Fortgeschritten

**Was:** Kosinus mit Neustarts (SGDR).

**Nutzen:** Periodische Neustarts erkunden neue Minima.

**Wofür:** Lange Trainings.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Skaliert automatisch mit der Trainingslänge.<br>**Wofür:** 0.01–0.1. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `cycles` | Ganzzahl | `3` | ≥ 1 | Basis | **Was:** Anzahl Kosinus-Zyklen (Neustarts).<br>**Nutzen:** Neustarts können aus flachen Minima herausspringen (SGDR).<br>**Wofür:** 2–5. |
| `cycle_decay` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Faktor, mit dem jeder neue Zyklus kleiner startet.<br>**Nutzen:** Allmählich vorsichtiger werden.<br>**Wofür:** 0.5–1. |

#### `scheduler:polynomial` — Stufe Fortgeschritten

**Was:** Polynomielles Abklingen.

**Nutzen:** Flexible Kurvenform.

**Wofür:** Segmentierung (DeepLab: power 0.9).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Skaliert automatisch mit der Trainingslänge.<br>**Wofür:** 0.01–0.1. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `power` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Exponent des Abklingens (1 = linear, 2 = quadratisch).<br>**Nutzen:** Form der Kurve.<br>**Wofür:** 1–2. |

#### `scheduler:exponential` — Stufe Fortgeschritten

**Was:** Exponentielles Abklingen.

**Nutzen:** Gleichmäßig geometrisch.

**Wofür:** Klassische Netze, RL.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Skaliert automatisch mit der Trainingslänge.<br>**Wofür:** 0.01–0.1. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `decay_rate` | Kommazahl | `0.96` | Bereich 0.0 … 1.0 | Basis | **Was:** Faktor pro decay_steps Schritte.<br>**Nutzen:** Geometrisches Abklingen.<br>**Wofür:** 0.9–0.99. |
| `decay_steps` | Ganzzahl | `100` | ≥ 1 | Basis | **Was:** Schritte pro Abklingstufe.<br>**Nutzen:** Granularität.<br>**Wofür:** Je nach Länge. |

#### `scheduler:step` (Aliase: `multistep`) — Stufe Basis

**Was:** Stufenweise Reduktion an Meilensteinen.

**Nutzen:** Klassisch und interpretierbar.

**Wofür:** ResNet-Training mit SGD.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase.<br>**Nutzen:** Optional.<br>**Wofür:** 0. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `milestones` | Liste[Kommazahl] | `[0.5, 0.75]` | ≥ 0.0 | Basis | **Was:** Zeitpunkte der Reduktion (Werte < 1 = Anteil der Schritte, sonst Schrittzahl).<br>**Nutzen:** Klassisch für ResNet-Training (bei 50 % und 75 %).<br>**Wofür:** [0.5, 0.75]. |
| `gamma` | Kommazahl | `0.1` | Bereich 0.0 … 1.0 | Basis | **Was:** Faktor je Meilenstein.<br>**Nutzen:** Stärke der Stufe.<br>**Wofür:** 0.1. |

#### `scheduler:inverse_sqrt` — Stufe Fortgeschritten

**Was:** Warmup, dann ∝ 1/√Schritt.

**Nutzen:** Kein festes Trainingsende nötig.

**Wofür:** Original-Transformer, fairseq.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.1` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase.<br>**Nutzen:** Danach ∝ 1/√schritt.<br>**Wofür:** Original-Transformer. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |

#### `scheduler:one_cycle` (Aliase: `onecycle`) — Stufe Basis

**Was:** 1cycle-Politik (Smith): steigen auf Maximum, dann tief abfallen.

**Nutzen:** Sehr schnelle Konvergenz ('Super-Convergence').

**Wofür:** CNNs, kurze Trainingsbudgets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.3` | Bereich 0.0 … 1.0 | Basis | **Was:** Anteil des Anstiegs zur Maximal-Lernrate (pct_start).<br>**Nutzen:** Langer Anstieg, dann Abfall.<br>**Wofür:** 0.25–0.3. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `div_factor` | Kommazahl | `25.0` | ≥ 1.0 | Fortgeschritten | **Was:** Start-LR = max_lr / div_factor.<br>**Nutzen:** Startniveau.<br>**Wofür:** 25. |
| `final_div_factor` | Kommazahl | `10000.0` | ≥ 1.0 | Fortgeschritten | **Was:** End-LR = Start-LR / final_div_factor.<br>**Nutzen:** Endniveau.<br>**Wofür:** 1e4. |

#### `scheduler:wsd` (Aliase: `warmup_stable_decay`) — Stufe Fortgeschritten

**Was:** Warmup-Stable-Decay.

**Nutzen:** Konstante Hauptphase, kurzes Abklingen; Checkpoints vor dem Decay weitertrainierbar.

**Wofür:** LLM-Vortraining.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase als Anteil aller Schritte.<br>**Nutzen:** Skaliert automatisch mit der Trainingslänge.<br>**Wofür:** 0.01–0.1. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `decay_ratio` | Kommazahl | `0.2` | Bereich 0.0 … 1.0 | Basis | **Was:** Anteil der Schritte für die finale Abklingphase.<br>**Nutzen:** Warmup-Stable-Decay: lange konstante Phase, kurzes Abklingen am Ende – erlaubt Weitertrainieren.<br>**Wofür:** LLM-Vortraining (MiniCPM, DeepSeek): 0.1–0.2. |
| `decay_shape` | Text | `'linear'` | Auswahl: linear \| cosine \| sqrt | Fortgeschritten | **Was:** Form des Abklingens.<br>**Nutzen:** sqrt fällt am Ende steil ab.<br>**Wofür:** linear oder sqrt. |

#### `scheduler:cyclic` — Stufe Experte

**Was:** Dreieckige zyklische Lernrate.

**Nutzen:** Pendelt zwischen tiefer und hoher LR.

**Wofür:** LR-Bereichssuche, Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase.<br>**Nutzen:** Optional.<br>**Wofür:** 0. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `cycle_steps` | Ganzzahl | `200` | ≥ 2 | Basis | **Was:** Länge eines Zyklus.<br>**Nutzen:** Periode der Schwingung.<br>**Wofür:** 2–10 Epochen. |
| `low_ratio` | Kommazahl | `0.1` | Bereich 0.0 … 1.0 | Basis | **Was:** Tiefpunkt relativ zur LR.<br>**Nutzen:** Amplitude.<br>**Wofür:** 0.1. |

#### `scheduler:plateau` (Aliase: `reduce_on_plateau`) — Stufe Basis

**Was:** Reduktion bei Stillstand der Validierungsmetrik.

**Nutzen:** Adaptiv, ohne festen Plan.

**Wofür:** Wenn die Trainingslänge unklar ist.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `warmup_steps` | Ganzzahl (oder None) | `None` | ≥ 0 | Basis | **Was:** Anzahl Aufwärmschritte (überschreibt warmup_ratio).<br>**Nutzen:** Lernrate steigt langsam an – schützt vor Divergenz am Anfang.<br>**Wofür:** Transformer: 1–5 % der Schritte bzw. 500–2000 Schritte. |
| `warmup_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Aufwärmphase.<br>**Nutzen:** Optional.<br>**Wofür:** 0. |
| `warmup_type` | Text | `'linear'` | Auswahl: linear \| cosine \| exponential | Experte | **Was:** Form des Anstiegs.<br>**Nutzen:** Feinheit; linear ist Standard.<br>**Wofür:** Selten zu ändern. |
| `warmup_start_factor` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Experte | **Was:** Startfaktor der Lernrate im Warmup (0 = bei 0 beginnen).<br>**Nutzen:** Sanfter oder härterer Start.<br>**Wofür:** 0–0.1. |
| `min_lr_ratio` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Untergrenze der Lernrate relativ zur Start-Lernrate.<br>**Nutzen:** Verhindert, dass das Training am Ende 'einfriert'.<br>**Wofür:** LLMs: 0.1. |
| `factor` | Kommazahl | `0.5` | Bereich 0.0 … 1.0 | Basis | **Was:** Reduktionsfaktor bei Stillstand.<br>**Nutzen:** Wie stark die LR fällt.<br>**Wofür:** 0.1–0.5. |
| `patience` | Ganzzahl | `2` | ≥ 0 | Basis | **Was:** Evaluierungen ohne Verbesserung bis zur Reduktion.<br>**Nutzen:** Geduld.<br>**Wofür:** 2–10. |
| `mode` | Text | `'min'` | Auswahl: min \| max | Basis | **Was:** Ob die Metrik kleiner (Verlust) oder größer (Genauigkeit) besser ist.<br>**Nutzen:** Richtung.<br>**Wofür:** Verlust: min. |
| `threshold` | Kommazahl | `0.0001` | ≥ 0.0 | Experte | **Was:** Mindestverbesserung (relativ).<br>**Nutzen:** Rauschen ignorieren.<br>**Wofür:** 1e-4. |


### Registry `environment`

<a id="registry-environment"></a>

**RL-Umgebungen** – Welten, in denen ein Agent handelt und Rückmeldungen erhält. Liefern Beobachtungen, Belohnungen und Zusatzinformationen für Belohnungsterme.

Fabrik-Signatur: `factory(args) -> Env`

#### `environment:gridworld` (Aliase: `gitterwelt`) — Stufe Basis

**Was:** Gitterwelt: Agent sucht den Weg zum Ziel (Wände, Rutschgefahr).

**Nutzen:** Ideal zum Experimentieren mit Belohnungsformung.

**Wofür:** Planung, Erkundung, Reward-Shaping-Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `size` | Ganzzahl | `5` | Bereich 2 … 64 | Basis | **Was:** Kantenlänge des Gitters.<br>**Nutzen:** Größer = schwierigeres Erkunden.<br>**Wofür:** 5–10. |
| `wall_density` | Kommazahl | `0.1` | Bereich 0.0 … 0.6 | Basis | **Was:** Anteil zufälliger Wandfelder.<br>**Nutzen:** Hindernisse erhöhen die Planungsschwierigkeit.<br>**Wofür:** 0–0.3. |
| `slip_prob` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Wahrscheinlichkeit, dass statt der gewählten eine zufällige Aktion ausgeführt wird.<br>**Nutzen:** Stochastische Dynamik – der Agent muss robust planen.<br>**Wofür:** 0–0.2. |
| `goal_reward` | Kommazahl | `1.0` |  | Basis | **Was:** Umgebungsbelohnung am Ziel.<br>**Nutzen:** Extrinsisches Ziel-Signal.<br>**Wofür:** 1. |
| `obs_mode` | Text | `'coords'` | Auswahl: coords \| onehot | Basis | **Was:** Beobachtungsform.<br>**Nutzen:** onehot ist tabellarisch, coords generalisiert.<br>**Wofür:** coords. |
| `max_steps` | Ganzzahl (oder None) | `None` | ≥ 1 | Basis | **Was:** Maximale Schritte pro Episode (None = 4·size²).<br>**Nutzen:** Begrenzt Irrfahrten.<br>**Wofür:** None. |
| `layout_seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed des Wandlayouts.<br>**Nutzen:** Gleiche Welt bei gleichem Seed.<br>**Wofür:** Reproduzierbarkeit. |

#### `environment:cartpole` — Stufe Basis

**Was:** Stab auf Wagen balancieren (klassische Regelungsaufgabe).

**Nutzen:** Schneller, verlässlicher RL-Test.

**Wofür:** Diskrete Steuerung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `max_steps` | Ganzzahl | `200` | ≥ 1 | Basis | **Was:** Episodenlänge (Erfolg = so lange balancieren).<br>**Nutzen:** Schwierigkeit.<br>**Wofür:** 200/500. |
| `gravity` | Kommazahl | `9.8` | ≥ 0.0 | Fortgeschritten | **Was:** Schwerkraft.<br>**Nutzen:** Physik-Parameter.<br>**Wofür:** 9.8. |
| `pole_length` | Kommazahl | `0.5` | ≥ 0.05 | Fortgeschritten | **Was:** Halbe Stablänge.<br>**Nutzen:** Kürzer = schwieriger.<br>**Wofür:** 0.5. |
| `force` | Kommazahl | `10.0` | ≥ 0.1 | Fortgeschritten | **Was:** Stoßkraft je Aktion.<br>**Nutzen:** Steuerstärke.<br>**Wofür:** 10. |
| `angle_limit_deg` | Kommazahl | `12.0` | Bereich 1.0 … 90.0 | Fortgeschritten | **Was:** Abbruchwinkel in Grad.<br>**Nutzen:** Toleranz.<br>**Wofür:** 12. |
| `noise` | Kommazahl | `0.0` | ≥ 0.0 | Fortgeschritten | **Was:** Rauschen auf die Kraft.<br>**Nutzen:** Stochastische Dynamik.<br>**Wofür:** 0–2. |

#### `environment:pendulum` — Stufe Basis

**Was:** Pendel aufschwingen und halten (kontinuierliches Drehmoment).

**Nutzen:** Test für kontinuierliche Aktionen.

**Wofür:** PPO mit Gauß-Policy.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `max_steps` | Ganzzahl | `200` | ≥ 1 | Basis | **Was:** Episodenlänge.<br>**Nutzen:** Dauer.<br>**Wofür:** 200. |
| `max_torque` | Kommazahl | `2.0` | ≥ 0.1 | Fortgeschritten | **Was:** Maximales Drehmoment.<br>**Nutzen:** Aktionsbereich.<br>**Wofür:** 2. |
| `gravity` | Kommazahl | `10.0` | ≥ 0.0 | Fortgeschritten | **Was:** Schwerkraft.<br>**Nutzen:** Physik.<br>**Wofür:** 10. |

#### `environment:bandit` — Stufe Basis

**Was:** Mehrarmiger Bandit (Erkundung vs. Ausnutzung).

**Nutzen:** Einfachste RL-Aufgabe.

**Wofür:** Erkundungsstrategien testen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `arms` | Ganzzahl | `10` | ≥ 2 | Basis | **Was:** Anzahl Arme (Aktionen).<br>**Nutzen:** Mehr = schwierigere Erkundung.<br>**Wofür:** 10. |
| `reward_std` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Rauschen der Belohnungen.<br>**Nutzen:** Unsicherheit.<br>**Wofür:** 1. |
| `episode_length` | Ganzzahl | `100` | ≥ 1 | Basis | **Was:** Ziehungen pro Episode.<br>**Nutzen:** Horizont.<br>**Wofür:** 100. |
| `means_seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Arm-Mittelwerte.<br>**Nutzen:** Reproduzierbare Aufgabe.<br>**Wofür:** 0. |

#### `environment:gymnasium` (Aliase: `gym`) — Stufe Fortgeschritten

**Was:** Beliebige Gymnasium-Umgebung.

**Nutzen:** Zugang zu Standard-Benchmarks.

**Wofür:** Atari, MuJoCo, Box2D (Zusatzpakete).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `env_id` | Text | `'CartPole-v1'` |  | Basis | **Was:** Gymnasium-Umgebungs-ID.<br>**Nutzen:** Zugriff auf hunderte Standard-Umgebungen.<br>**Wofür:** Benötigt 'pip install gymnasium'. |
| `kwargs` | Zuordnung (dict) | `{}` |  | Fortgeschritten | **Was:** Zusätzliche Argumente für gymnasium.make.<br>**Nutzen:** Umgebungsoptionen.<br>**Wofür:** Je nach Umgebung. |


### Registry `reward`

<a id="registry-reward"></a>

**Belohnungsterme** – Bausteine der Belohnungsfunktion im Reinforcement Learning. Die Belohnung definiert, was der Agent lernen soll – kleinste Änderungen ändern das Verhalten.

Fabrik-Signatur: `factory(args) -> RewardTerm mit __call__(transition) -> Tensor[N]`

#### `reward:env` (Aliase: `umgebung`, `extrinsic`) — Stufe Basis

**Was:** Belohnung der Umgebung (die eigentliche Aufgabe).

**Nutzen:** Basis fast jeder Belohnungsfunktion.

**Wofür:** Immer sinnvoll.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'extrinsisch'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art der Belohnung (Einordnung + getrennte Protokollierung).<br>**Nutzen:** Übersicht, welcher Anteil wofür sorgt.<br>**Wofür:** Analyse von Reward-Hacking. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |

#### `reward:constant` (Aliase: `time_penalty`, `zeitstrafe`) — Stufe Basis

**Was:** Konstanter Wert je Schritt (Zeitstrafe/Überlebensbonus).

**Nutzen:** Steuert Eile vs. Ausdauer.

**Wofür:** Labyrinthe (Strafe), Balancieren (Bonus).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'strafe'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art.<br>**Nutzen:** Einordnung.<br>**Wofür:** strafe (Zeitstrafe) oder bonus (Überlebensbonus). |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `value` | Kommazahl | `-0.01` |  | Basis | **Was:** Wert je Schritt (negativ = Zeitstrafe, positiv = Überlebensbonus).<br>**Nutzen:** Zeitstrafe → schnelle Lösungen; Bonus → möglichst lange 'überleben'.<br>**Wofür:** -0.01 bis 0.01. |

#### `reward:action_penalty` (Aliase: `energie`) — Stufe Basis

**Was:** Strafe für Aktionsgröße/-kosten.

**Nutzen:** Energiesparendes, sanftes Verhalten.

**Wofür:** Robotik, Steuerung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'strafe'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art.<br>**Nutzen:** Einordnung.<br>**Wofür:** strafe. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `norm` | Text | `'l2'` | Auswahl: l1 \| l2 | Basis | **Was:** Maß der Aktionsgröße (kontinuierlich).<br>**Nutzen:** Bestraft Kraft-/Energieaufwand.<br>**Wofür:** Robotik: l2. |
| `costs` | Liste[Kommazahl] | `[]` |  | Basis | **Was:** Kosten je diskreter Aktion (leer = keine).<br>**Nutzen:** Manche Aktionen sind teurer (z. B. 'springen').<br>**Wofür:** Spiele. |

#### `reward:action_change` (Aliase: `smoothness`) — Stufe Fortgeschritten

**Was:** Strafe für Aktionswechsel (Ruckeln).

**Nutzen:** Glattere Steuerung.

**Wofür:** Roboter, Fahrzeuge.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'extrinsisch'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art der Belohnung (Einordnung + getrennte Protokollierung).<br>**Nutzen:** Übersicht, welcher Anteil wofür sorgt.<br>**Wofür:** Analyse von Reward-Hacking. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |

#### `reward:info` (Aliase: `info_value`) — Stufe Basis

**Was:** Beliebige Umgebungsgröße (info-dict) als Belohnung.

**Nutzen:** Maximale Freiheit ohne eigenen Code.

**Wofür:** Abstand, Winkel, Geschwindigkeit ...

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'extrinsisch'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art der Belohnung (Einordnung + getrennte Protokollierung).<br>**Nutzen:** Übersicht, welcher Anteil wofür sorgt.<br>**Wofür:** Analyse von Reward-Hacking. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `key` | Text | `'distance'` |  | Basis | **Was:** Schlüssel im info-dict der Umgebung (z. B. distance, angle, upright).<br>**Nutzen:** Beliebige von der Umgebung gemeldete Größe wird zur Belohnung.<br>**Wofür:** Je nach Umgebung. |
| `transform` | Text | `'identity'` | Auswahl: identity \| neg \| abs \| neg_abs \| square \| neg_square | Basis | **Was:** Umformung des Werts.<br>**Nutzen:** Z. B. -\|Winkel\| belohnt aufrechte Haltung.<br>**Wofür:** Je nach Ziel. |

#### `reward:potential` (Aliase: `shaping`) — Stufe Basis

**Was:** Potentialbasierte Formung γΦ(s')−Φ(s).

**Nutzen:** Hilft beim Lernen, ohne das optimale Verhalten zu verfälschen.

**Wofür:** Spärliche Belohnungen (Labyrinthe, Robotik).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'formend'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art.<br>**Nutzen:** Einordnung.<br>**Wofür:** formend. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `key` | Text | `'distance'` |  | Basis | **Was:** Schlüssel im info-dict der Umgebung (z. B. distance, angle, upright).<br>**Nutzen:** Beliebige von der Umgebung gemeldete Größe wird zur Belohnung.<br>**Wofür:** Je nach Umgebung. |
| `transform` | Text | `'neg'` | Auswahl: identity \| neg \| abs \| neg_abs \| square \| neg_square | Basis | **Was:** Potential Φ = transform(info[key]); Standard: -Abstand (näher = besser).<br>**Nutzen:** Richtung der Formung.<br>**Wofür:** neg. |

#### `reward:event` (Aliase: `ereignis`) — Stufe Basis

**Was:** Bonus/Strafe bei einem Ereignis.

**Nutzen:** Belohnt Meilensteine, bestraft Kollisionen.

**Wofür:** Ziel erreicht, Wand berührt.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'bonus'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art.<br>**Nutzen:** Einordnung.<br>**Wofür:** bonus oder strafe. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `key` | Text | `'reached_goal'` |  | Basis | **Was:** Ereignis-Schlüssel im info-dict (z. B. reached_goal, hit_wall, failed).<br>**Nutzen:** Belohnt/bestraft das Eintreten eines Ereignisses.<br>**Wofür:** Je nach Umgebung. |
| `value` | Kommazahl | `1.0` |  | Basis | **Was:** Belohnung beim Ereignis (negativ = Strafe).<br>**Nutzen:** Stärke.<br>**Wofür:** ±0.1 bis ±10. |

#### `reward:terminal` — Stufe Basis

**Was:** Wert am Episodenende.

**Nutzen:** Unterscheidet Scheitern von Zeitablauf.

**Wofür:** Absturzstrafe.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `1.0` |  | Basis | **Was:** Gewicht des Terms in der Gesamtbelohnung.<br>**Nutzen:** Balanciert konkurrierende Ziele.<br>**Wofür:** Hauptziel 1; Formung/Strafen meist 0.001–0.1.<br>**Tipp:** Negative Gewichte kehren den Term um. |
| `kind` | Text | `'strafe'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art.<br>**Nutzen:** Einordnung.<br>**Wofür:** strafe (Absturz) oder bonus. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `on` | Text | `'terminated'` | Auswahl: terminated \| truncated \| any | Basis | **Was:** Wann der Term greift.<br>**Nutzen:** Unterscheidet Scheitern von Zeitablauf.<br>**Wofür:** terminated. |
| `value` | Kommazahl | `-1.0` |  | Basis | **Was:** Wert am Episodenende.<br>**Nutzen:** Z. B. Strafe fürs Umfallen.<br>**Wofür:** -1 bis -10. |

#### `reward:rnd` (Aliase: `curiosity`, `neugier`) — Stufe Fortgeschritten

**Was:** Neugier per Random Network Distillation.

**Nutzen:** Intrinsische Erkundung bei spärlichen Belohnungen.

**Wofür:** Schwierige Erkundungsprobleme (Montezuma-artig).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `0.1` |  | Basis | **Was:** Gewicht der Neugier.<br>**Nutzen:** Stärke der Erkundung.<br>**Wofür:** 0.01–0.5. |
| `kind` | Text | `'intrinsisch'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art.<br>**Nutzen:** Einordnung.<br>**Wofür:** intrinsisch. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `hidden_dim` | Ganzzahl | `64` | ≥ 1 | Fortgeschritten | **Was:** Breite der RND-Netze.<br>**Nutzen:** Kapazität.<br>**Wofür:** 64–256. |
| `feature_dim` | Ganzzahl | `32` | ≥ 1 | Fortgeschritten | **Was:** Ausgabegröße der RND-Netze.<br>**Nutzen:** Merkmalsraum.<br>**Wofür:** 32–128. |
| `lr` | Kommazahl | `0.001` | ≥ 0.0 | Fortgeschritten | **Was:** Lernrate des Prädiktors.<br>**Nutzen:** Wie schnell Neues 'bekannt' wird.<br>**Wofür:** 1e-4–1e-3. |
| `normalize` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Neugier durch laufende Standardabweichung teilen.<br>**Nutzen:** Stabile Skala.<br>**Wofür:** an. |

#### `reward:count` (Aliase: `novelty`, `neuheit`) — Stufe Fortgeschritten

**Was:** Zählbasierter Neuheitsbonus.

**Nutzen:** Einfache, wirksame Erkundung in kleinen Zustandsräumen.

**Wofür:** Gitterwelten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Kommazahl | `0.1` |  | Basis | **Was:** Gewicht des Neuheitsbonus.<br>**Nutzen:** Stärke der Erkundung.<br>**Wofür:** 0.01–1. |
| `kind` | Text | `'intrinsisch'` | Auswahl: extrinsisch \| intrinsisch \| formend \| strafe \| bonus | Basis | **Was:** Art.<br>**Nutzen:** Einordnung.<br>**Wofür:** intrinsisch. |
| `label` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Anzeigename im Protokoll (None = Komponentenname).<br>**Nutzen:** Übersicht.<br>**Wofür:** Optional. |
| `clip_min` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Untergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `clip_max` | Kommazahl (oder None) | `None` |  | Fortgeschritten | **Was:** Obergrenze je Schritt.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** Optional. |
| `start_step` | Ganzzahl | `0` | ≥ 0 | Fortgeschritten | **Was:** Erst ab diesem Umgebungsschritt aktiv.<br>**Nutzen:** Belohnungs-Curriculum.<br>**Wofür:** Späte Zusatzziele. |
| `end_step` | Ganzzahl (oder None) | `None` | ≥ 0 | Fortgeschritten | **Was:** Nur bis zu diesem Schritt aktiv.<br>**Nutzen:** Hilfsbelohnungen später abschalten (Shaping-Gerüst).<br>**Wofür:** Optional. |
| `anneal` | Text | `'constant'` | Auswahl: constant \| linear_decay \| linear_increase | Fortgeschritten | **Was:** Gewichtsverlauf über die Zeit.<br>**Nutzen:** Z. B. Neugier anfangs stark, später schwach.<br>**Wofür:** intrinsisch: linear_decay. |
| `anneal_steps` | Ganzzahl | `10000` | ≥ 1 | Fortgeschritten | **Was:** Dauer des Gewichtsverlaufs (Schritte).<br>**Nutzen:** Tempo.<br>**Wofür:** Trainingslänge. |
| `resolution` | Kommazahl | `0.1` | ≥ 1e-06 | Fortgeschritten | **Was:** Rasterweite zum Zählen von Zuständen.<br>**Nutzen:** Gröber = mehr Zustände gelten als gleich.<br>**Wofür:** Je nach Skala. |


### Registry `rl_algorithm`

<a id="registry-rl-algorithm"></a>

**RL-Algorithmen** – Lernverfahren für Agenten in Umgebungen. Bestimmen Stabilität, Dateneffizienz und geeignete Aktionsräume.

Fabrik-Signatur: `factory(args) -> RLAlgorithm mit run(trainer, rl_cfg)`

#### `rl_algorithm:ppo` — Stufe Basis

**Was:** Proximal Policy Optimization.

**Nutzen:** Robust, vielseitig, Standard für viele RL-Aufgaben.

**Wofür:** Diskrete & kontinuierliche Steuerung, Robotik, Spiele, RLHF.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_steps` | Ganzzahl | `128` | ≥ 1 | Basis | **Was:** Schritte pro Umgebung je Rollout.<br>**Nutzen:** Länger = bessere Schätzungen, seltenere Updates.<br>**Wofür:** PPO: 128–2048; A2C: 5. |
| `value_coef` | Kommazahl | `0.5` | ≥ 0.0 | Basis | **Was:** Gewicht des Value-Verlusts.<br>**Nutzen:** Balanciert Kritiker- gegen Policy-Lernen.<br>**Wofür:** 0.5. |
| `entropy_coef` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Entropie-Bonus.<br>**Nutzen:** Fördert Erkundung, verhindert vorzeitig deterministische Policies.<br>**Wofür:** 0–0.05. |
| `gae_lambda` | Kommazahl | `0.95` | Bereich 0.0 … 1.0 | Basis | **Was:** λ der Generalized Advantage Estimation.<br>**Nutzen:** Bias-Varianz-Abwägung (1 = Monte-Carlo, 0 = TD).<br>**Wofür:** 0.9–0.99. |
| `normalize_advantages` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Vorteile pro Batch standardisieren.<br>**Nutzen:** Stabilere Gradienten.<br>**Wofür:** PPO: an. |
| `n_epochs` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Durchläufe über jeden Rollout.<br>**Nutzen:** Mehr = dateneffizienter, aber Gefahr zu großer Policy-Sprünge.<br>**Wofür:** 3–10. |
| `minibatch_size` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Minibatch-Größe.<br>**Nutzen:** Gradienten-Rauschen vs. Geschwindigkeit.<br>**Wofür:** 64–512. |
| `clip_range` | Kommazahl | `0.2` | ≥ 0.0 | Basis | **Was:** PPO-Clipping ε des Wahrscheinlichkeitsverhältnisses.<br>**Nutzen:** Begrenzt Policy-Änderungen pro Update – Kern von PPO.<br>**Wofür:** 0.1–0.3. |
| `clip_value` | Kommazahl (oder None) | `None` | ≥ 0.0 | Fortgeschritten | **Was:** Clipping der Value-Updates (None = aus).<br>**Nutzen:** Stabilisiert den Kritiker.<br>**Wofür:** 0.2 oder None. |
| `target_kl` | Kommazahl (oder None) | `None` | ≥ 0.0 | Fortgeschritten | **Was:** Abbruch der Epochen, wenn die KL-Divergenz diesen Wert überschreitet.<br>**Nutzen:** Schutz vor Policy-Kollaps.<br>**Wofür:** 0.01–0.05. |

#### `rl_algorithm:a2c` — Stufe Basis

**Was:** Advantage Actor-Critic.

**Nutzen:** Einfach und schnell, weniger stabil als PPO.

**Wofür:** Schnelle Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_steps` | Ganzzahl | `5` | ≥ 1 | Basis | **Was:** Schritte pro Umgebung je Update.<br>**Nutzen:** Kurze Rollouts, häufige Updates.<br>**Wofür:** 5–20. |
| `value_coef` | Kommazahl | `0.5` | ≥ 0.0 | Basis | **Was:** Gewicht des Value-Verlusts.<br>**Nutzen:** Balanciert Kritiker- gegen Policy-Lernen.<br>**Wofür:** 0.5. |
| `entropy_coef` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Entropie-Bonus.<br>**Nutzen:** Fördert Erkundung, verhindert vorzeitig deterministische Policies.<br>**Wofür:** 0–0.05. |
| `gae_lambda` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Basis | **Was:** λ der GAE.<br>**Nutzen:** 1 = n-Schritt-Ertrag.<br>**Wofür:** 1.0. |
| `normalize_advantages` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Vorteile standardisieren.<br>**Nutzen:** Optional.<br>**Wofür:** aus. |

#### `rl_algorithm:reinforce` — Stufe Basis

**Was:** REINFORCE (Monte-Carlo-Policy-Gradient).

**Nutzen:** Der einfachste Policy-Gradient – lehrreich, aber verrauscht.

**Wofür:** Lehre, Bandits, kurze Episoden.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_steps` | Ganzzahl | `200` | ≥ 1 | Basis | **Was:** Schritte pro Umgebung je Update.<br>**Nutzen:** Lange Rollouts (Monte-Carlo).<br>**Wofür:** ≥ Episodenlänge. |
| `value_coef` | Kommazahl | `0.5` | ≥ 0.0 | Basis | **Was:** Gewicht des Value-Verlusts.<br>**Nutzen:** Balanciert Kritiker- gegen Policy-Lernen.<br>**Wofür:** 0.5. |
| `entropy_coef` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Entropie-Bonus.<br>**Nutzen:** Erkundung.<br>**Wofür:** 0–0.01. |
| `gae_lambda` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Experte | **Was:** λ (REINFORCE: 1 = reine Monte-Carlo-Erträge).<br>**Nutzen:** MC.<br>**Wofür:** 1.0. |
| `normalize_advantages` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Vorteile pro Batch standardisieren.<br>**Nutzen:** Stabilere Gradienten.<br>**Wofür:** PPO: an. |
| `baseline` | Text | `'value'` | Auswahl: none \| mean \| value | Basis | **Was:** Baseline zur Varianzreduktion.<br>**Nutzen:** Ohne Baseline ist REINFORCE sehr verrauscht.<br>**Wofür:** value. |

#### `rl_algorithm:dqn` — Stufe Basis

**Was:** Deep Q-Network (+ Double, n-Schritt, Priorisierung).

**Nutzen:** Sehr dateneffizient durch Replay.

**Wofür:** Diskrete Aktionen (Spiele, Gitterwelten).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `buffer_size` | Ganzzahl | `50000` | ≥ 1 | Basis | **Was:** Kapazität des Replay-Speichers.<br>**Nutzen:** Mehr = vielfältigere, ältere Erfahrungen.<br>**Wofür:** 10⁴–10⁶. |
| `learning_starts` | Ganzzahl | `1000` | ≥ 0 | Basis | **Was:** Zufällige Schritte bevor gelernt wird.<br>**Nutzen:** Füllt den Speicher mit vielfältigen Daten.<br>**Wofür:** 1000–50000. |
| `batch_size` | Ganzzahl | `64` | ≥ 1 | Basis | **Was:** Batchgröße der Updates.<br>**Nutzen:** Gradientenqualität.<br>**Wofür:** 32–256. |
| `train_freq` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Update alle n Umgebungsschritte (je Umgebung).<br>**Nutzen:** Daten/Update-Verhältnis.<br>**Wofür:** 1–4. |
| `gradient_steps` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Updates je Trainingszeitpunkt.<br>**Nutzen:** Mehr = dateneffizienter.<br>**Wofür:** 1. |
| `target_update_interval` | Ganzzahl | `500` | ≥ 1 | Basis | **Was:** Ziel-Netz-Update alle n Gradientenschritte.<br>**Nutzen:** Stabilisiert die Bootstrapping-Ziele.<br>**Wofür:** 500–10000. |
| `tau` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Weiches Update θ' ← τθ + (1-τ)θ' (1 = hartes Kopieren).<br>**Nutzen:** Glattere Ziele.<br>**Wofür:** 1.0 oder 0.005 (jeden Schritt). |
| `double` | Ja/Nein | `True` |  | Basis | **Was:** Double-DQN: Aktion mit Online-Netz wählen, mit Ziel-Netz bewerten.<br>**Nutzen:** Reduziert Überschätzung von Q-Werten.<br>**Wofür:** Fast immer an. |
| `n_step` | Ganzzahl | `1` | ≥ 1 | Fortgeschritten | **Was:** n-Schritt-Erträge.<br>**Nutzen:** Schnellere Belohnungsausbreitung, etwas mehr Varianz.<br>**Wofür:** 1–5 (Rainbow: 3). |
| `epsilon_start` | Kommazahl | `1.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Start-ε (Zufallsaktionen).<br>**Nutzen:** Anfangs viel erkunden.<br>**Wofür:** 1.0. |
| `epsilon_end` | Kommazahl | `0.05` | Bereich 0.0 … 1.0 | Basis | **Was:** End-ε.<br>**Nutzen:** Rest-Erkundung.<br>**Wofür:** 0.01–0.05. |
| `epsilon_fraction` | Kommazahl | `0.3` | Bereich 0.0 … 1.0 | Basis | **Was:** Anteil des Trainings, über den ε linear sinkt.<br>**Nutzen:** Dauer der Erkundungsphase.<br>**Wofür:** 0.1–0.5. |
| `loss` | Text | `'huber'` | Auswahl: huber \| mse | Basis | **Was:** TD-Verlust.<br>**Nutzen:** Huber ist robust gegen große TD-Fehler.<br>**Wofür:** huber. |
| `prioritized` | Ja/Nein | `False` |  | Fortgeschritten | **Was:** Priorisiertes Replay (große TD-Fehler öfter ziehen).<br>**Nutzen:** Lernt schneller aus überraschenden Erfahrungen.<br>**Wofür:** Rainbow-DQN. |
| `alpha` | Kommazahl | `0.6` | Bereich 0.0 … 1.0 | Experte | **Was:** Stärke der Priorisierung.<br>**Nutzen:** 0 = gleichverteilt.<br>**Wofür:** 0.6. |
| `beta_start` | Kommazahl | `0.4` | Bereich 0.0 … 1.0 | Experte | **Was:** Start der Importance-Sampling-Korrektur (steigt auf 1).<br>**Nutzen:** Korrigiert den Priorisierungs-Bias.<br>**Wofür:** 0.4. |


### Registry `module_op`

<a id="registry-module-op"></a>

**Modul-Operationen** – Eingriffe in ganze Schichten/Module (ersetzen, umbauen, einfrieren, Datenfluss verändern …). Jede Schicht jeder Art einzeln anpassen – auch nach dem Bau, ohne Code.

Fabrik-Signatur: `factory(args) -> callable(model, name, module, ctx)`

#### `module_op:freeze` — Stufe Basis

**Was:** Alle Parameter des Moduls einfrieren.

**Nutzen:** Schützt Wissen, spart Gradienten.

**Wofür:** Transfer-Learning.

#### `module_op:unfreeze` — Stufe Basis

**Was:** Alle Parameter des Moduls auftauen.

**Nutzen:** Hebt freeze auf.

**Wofür:** 'Alles einfrieren außer …'.

#### `module_op:replace_activation` — Stufe Basis

**Was:** Aktivierungsmodul ersetzen.

**Nutzen:** Andere Nichtlinearität in einzelnen Schichten.

**Wofür:** Ablationen, Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `activation` | Komponente aus Registry 'activation' | `gelu` | Auswahl: relu \| leaky_relu \| prelu \| rrelu \| elu \| celu \| selu \| gelu \| quick_gelu \| silu \| swish \| mish \| tanh \| sigmoid \| … | Basis | **Was:** Neue Aktivierung.<br>**Nutzen:** Austausch der Nichtlinearität einzelner Schichten.<br>**Wofür:** silu, gelu, relu … |
| `activation.approximate` | Text | `'none'` | Auswahl: none \| tanh | Fortgeschritten | **Was:** Berechnungsart der GELU.<br>**Nutzen:** Die tanh-Näherung ist auf mancher Hardware schneller, numerisch minimal anders.<br>**Wofür:** Kompatibilität mit vortrainierten Gewichten (GPT-2 nutzt tanh). |

#### `module_op:replace_norm` — Stufe Basis

**Was:** Normalisierungsmodul ersetzen.

**Nutzen:** Andere Norm in einzelnen Schichten.

**Wofür:** LayerNorm → RMSNorm.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `norm` | Komponente aus Registry 'normalization' | `rmsnorm` | Auswahl: layernorm \| rmsnorm \| batchnorm \| groupnorm \| instancenorm \| scalenorm \| l2norm \| dyt \| identity | Basis | **Was:** Neue Normalisierung.<br>**Nutzen:** Austausch einzelner Normschichten.<br>**Wofür:** rmsnorm, layernorm … |
| `norm.eps` | Kommazahl | `1e-06` | ≥ 0.0 | Experte | **Was:** Kleine Zahl gegen Division durch 0.<br>**Nutzen:** Zu klein → numerische Probleme bei niedriger Präzision; zu groß → verfälschte Normalisierung.<br>**Wofür:** fp16/bf16-Training (dann eher 1e-5 bis 1e-6 statt 1e-8). |
| `norm.learnable` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Lernbare Skala pro Feature.<br>**Nutzen:** Wie γ bei LayerNorm.<br>**Wofür:** Standard: an. |
| `norm.scale_offset` | Ja/Nein | `False` |  | Experte | **Was:** Skala als (1 + w) parametrisieren, w startet bei 0.<br>**Nutzen:** Weight-Decay zieht die Skala Richtung 1 statt Richtung 0.<br>**Wofür:** Gemma-Stil. |
| `norm.compute_in_fp32` | Ja/Nein | `True` |  | Experte | **Was:** Normalisierung intern in 32-Bit rechnen.<br>**Nutzen:** Verhindert Genauigkeitsverlust bei fp16/bf16-Training.<br>**Wofür:** Mixed-Precision-Training. |
| `num_features` | Ganzzahl (oder None) | `None` | ≥ 1 | Experte | **Was:** Merkmalsbreite (None = vom alten Modul übernehmen).<br>**Nutzen:** Nur nötig, wenn nicht erkennbar.<br>**Wofür:** None. |

#### `module_op:identity` (Aliase: `entfernen`) — Stufe Basis

**Was:** Modul durch Identität ersetzen (entfernen).

**Nutzen:** Ablationen: Was passiert ohne diese Schicht?

**Wofür:** Form muss erhalten bleiben (z. B. Dropout, Norm, Aktivierung, ganzer Residual-Block).

#### `module_op:set_attr` — Stufe Basis

**Was:** Beliebiges Attribut setzen.

**Nutzen:** Feinste Eingriffe in gebaute Module.

**Wofür:** Dropout.p, eps, scale …

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `attribute` | Text | `'p'` |  | Basis | **Was:** Attributname (auch Punkt-Pfad, z. B. 'attn.scale').<br>**Nutzen:** Verändert Eigenschaften gebauter Module, die im Forward gelesen werden.<br>**Wofür:** Dropout 'p', Norm 'eps', Attention 'scale', LeakyReLU 'negative_slope'. |
| `value` | Ganzzahl oder Kommazahl oder Ja/Nein oder Text (oder None) | `0.0` |  | Basis | **Was:** Neuer Wert.<br>**Nutzen:** Wird unverändert gesetzt.<br>**Wofür:** Je nach Attribut. |
| `must_exist` | Ja/Nein | `True` |  | Experte | **Was:** Fehler, wenn das Attribut nicht existiert.<br>**Nutzen:** Schützt vor Tippfehlern.<br>**Wofür:** an. |

#### `module_op:dtype` — Stufe Fortgeschritten

**Was:** Datentyp des Moduls ändern.

**Nutzen:** Gemischte Präzision pro Schicht.

**Wofür:** Normen/Router in fp32.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `dtype` | Text | `'float32'` | Auswahl: float32 \| bfloat16 \| float16 \| float64 | Basis | **Was:** Datentyp der Parameter dieses Moduls.<br>**Nutzen:** Empfindliche Teile (Normen, Router) in fp32 halten, Rest in bf16.<br>**Wofür:** float32 für Normen. |

#### `module_op:quantize` — Stufe Fortgeschritten

**Was:** Lineare Schichten in diesem Modul quantisieren.

**Nutzen:** Gezielte Kompression einzelner Teile.

**Wofür:** Große FFNs.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `format` | Text | `'int8'` | Auswahl: int8 \| int4 \| nf4 | Basis | **Was:** Quantisierungsformat (eigene Implementierung).<br>**Nutzen:** Speicher dieses Moduls verkleinern.<br>**Wofür:** int8. |
| `group_size` | Ganzzahl (oder None) | `128` | ≥ 8 | Basis | **Was:** Gruppengröße (None = pro Kanal).<br>**Nutzen:** Genauigkeit vs. Skalen-Speicher.<br>**Wofür:** 128. |

#### `module_op:lora` — Stufe Fortgeschritten

**Was:** LoRA-Adapter für die linearen Schichten dieses Moduls.

**Nutzen:** Adapter nur dort, wo gewünscht.

**Wofür:** Feintuning ausgewählter Schichten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `rank` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Rang.<br>**Nutzen:** Adapter-Kapazität.<br>**Wofür:** 8–32. |
| `alpha` | Kommazahl | `16.0` | ≥ 0.0 | Basis | **Was:** Skalierung α.<br>**Nutzen:** Faktor α/r.<br>**Wofür:** 2·rank. |
| `dropout` | Kommazahl | `0.0` | Bereich 0.0 … 1.0 | Basis | **Was:** Dropout.<br>**Nutzen:** Regularisierung.<br>**Wofür:** 0–0.1. |
| `use_dora` | Ja/Nein | `False` |  | Experimentell | **Was:** DoRA.<br>**Nutzen:** Betrag + Richtung.<br>**Wofür:** Experimente. |
| `freeze_module` | Ja/Nein | `True` |  | Basis | **Was:** Basisgewichte dieses Moduls einfrieren.<br>**Nutzen:** Nur die Adapter lernen.<br>**Wofür:** an. |

#### `module_op:spectral_norm` — Stufe Fortgeschritten

**Was:** Spektralnormierung des Gewichts.

**Nutzen:** Begrenzt die Lipschitz-Konstante → stabile GANs/Diskriminatoren.

**Wofür:** GAN-Diskriminatoren.

#### `module_op:weight_norm` — Stufe Fortgeschritten

**Was:** Gewichtsnormierung (Richtung + Betrag).

**Nutzen:** Schnellere, stabilere Optimierung.

**Wofür:** Audio-/RL-Netze.

#### `module_op:checkpoint` — Stufe Fortgeschritten

**Was:** Gradient-Checkpointing für dieses Modul.

**Nutzen:** Spart Aktivierungsspeicher genau hier.

**Wofür:** Speicherintensive Teile.

#### `module_op:init` — Stufe Basis

**Was:** Alle Parameter des Moduls neu initialisieren.

**Nutzen:** Gezielte Startwerte.

**Wofür:** Neuer Kopf, Re-Init.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `weight` | Komponente aus Registry 'initializer' | `xavier_uniform` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Basis | **Was:** Initialisierer für Gewichte (≥ 2 Dimensionen).<br>**Nutzen:** Neue Startwerte.<br>**Wofür:** xavier_uniform, orthogonal … |
| `weight.gain` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Skalierungsfaktor der Streuung.<br>**Nutzen:** Kompensiert die Varianzänderung durch die Aktivierung.<br>**Wofür:** Wird ignoriert, wenn 'nonlinearity' gesetzt ist (dann automatisch berechnet). |
| `weight.nonlinearity` | Text (oder None) | `None` | Auswahl: linear \| conv1d \| conv2d \| conv3d \| sigmoid \| tanh \| relu \| leaky_relu \| selu | Fortgeschritten | **Was:** Aktivierung, für die der Gain automatisch berechnet wird.<br>**Nutzen:** z. B. relu → √2, tanh → 5/3.<br>**Wofür:** Wenn die nachfolgende Aktivierung bekannt ist. |
| `bias` | Komponente aus Registry 'initializer' (oder None) | `zeros` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Basis | **Was:** Initialisierer für Biases/1D-Parameter (None = unverändert).<br>**Nutzen:** Startwerte.<br>**Wofür:** zeros. |

#### `module_op:reset` — Stufe Basis

**Was:** PyTorch-Standard-Initialisierung (reset_parameters).

**Nutzen:** Zurück auf Anfang.

**Wofür:** Re-Init einzelner Schichten.

#### `module_op:copy_from` (Aliase: `uebernehmen`) — Stufe Basis

**Was:** Gewichte eines anderen Moduls gleicher Struktur übernehmen.

**Nutzen:** Übernahme ganzer Schichten (alle Tensoren auf einmal).

**Wofür:** Schichten klonen, Block-Wiederholung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `source` | Text | `''` |  | Basis | **Was:** Name des Quellmoduls mit gleicher Struktur (z. B. 'blocks.0').<br>**Nutzen:** Übernimmt alle Gewichte und Puffer eines anderen Moduls.<br>**Wofür:** Schichten klonen, Warm-Start. |
| `mode` | Text | `'replace'` | Auswahl: replace \| interpolate | Basis | **Was:** Art der Übernahme.<br>**Nutzen:** Vollständig oder anteilig.<br>**Wofür:** replace. |
| `alpha` | Kommazahl | `0.5` | Bereich 0.0 … 1.0 | Basis | **Was:** Anteil der Quelle (interpolate).<br>**Nutzen:** Mischung.<br>**Wofür:** 0.5. |

#### `module_op:scale_output` — Stufe Basis

**Was:** Ausgabe mit Faktor multiplizieren (Hook).

**Nutzen:** Beitrag einzelner Module dämpfen/verstärken.

**Wofür:** Ablationen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `factor` | Kommazahl | `1.0` |  | Basis | **Was:** Faktor auf die Ausgabe.<br>**Nutzen:** Dämpft/verstärkt den Beitrag des Moduls.<br>**Wofür:** 0.5. |

#### `module_op:output_noise` — Stufe Fortgeschritten

**Was:** Rauschen auf die Ausgabe (Hook).

**Nutzen:** Robustheit, Regularisierung.

**Wofür:** Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `std` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Standardabweichung.<br>**Nutzen:** Stärke des Rauschens.<br>**Wofür:** 0.01–0.1. |
| `only_training` | Ja/Nein | `True` |  | Basis | **Was:** Nur im Training.<br>**Nutzen:** Inferenz bleibt deterministisch.<br>**Wofür:** an. |

#### `module_op:clamp_output` — Stufe Fortgeschritten

**Was:** Ausgabe begrenzen (Hook).

**Nutzen:** Gegen Aktivierungsausreißer.

**Wofür:** Stabilisierung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `min` | Kommazahl (oder None) | `None` |  | Basis | **Was:** Untergrenze.<br>**Nutzen:** Begrenzt Aktivierungen.<br>**Wofür:** -10. |
| `max` | Kommazahl (oder None) | `None` |  | Basis | **Was:** Obergrenze.<br>**Nutzen:** Begrenzt Ausreißer.<br>**Wofür:** 10. |

#### `module_op:stop_gradient` — Stufe Fortgeschritten

**Was:** Gradient an der Ausgabe abschneiden (Hook).

**Nutzen:** Alles davor lernt nicht über diesen Pfad.

**Wofür:** Experimente.

#### `module_op:record` — Stufe Basis

**Was:** Statistik der Ausgabe aufzeichnen (Mittelwert, Std, Max) → ModuleSurgeon.activation_report().

**Nutzen:** Analyse: Wo explodieren oder verschwinden Aktivierungen?

**Wofür:** Debugging, Interpretierbarkeit.


### Registry `tensor_op`

<a id="registry-tensor-op"></a>

**Tensor-Operationen** – Eingriffe in einzelne Gewichtstensoren (oder Ausschnitte). Feinste Kontrolle: Einfrieren, Initialisieren, Beschneiden, Beschränken, eigene Lernraten ...

Fabrik-Signatur: `factory(args) -> TensorOp`

#### `tensor_op:freeze` — Stufe Basis

**Was:** Einfrieren: nicht mehr trainieren (mit index: nur der Ausschnitt).

**Nutzen:** Schützt vortrainiertes Wissen, spart Rechenzeit und Speicher.

**Wofür:** Transfer-Learning, Adapter-Training.

#### `tensor_op:unfreeze` — Stufe Basis

**Was:** Auftauen: wieder trainierbar machen.

**Nutzen:** Hebt ein früheres freeze auf (Regeln wirken in Reihenfolge).

**Wofür:** 'Alles einfrieren außer ...'.

#### `tensor_op:init` — Stufe Basis

**Was:** Neu initialisieren mit beliebigem Initialisierer.

**Nutzen:** Gezielte Startwerte einzelner Tensoren.

**Wofür:** Zero-Init von Ausgangsschichten, Re-Init von Köpfen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `initializer` | Komponente aus Registry 'initializer' | `normal` | Auswahl: keep \| zeros \| ones \| constant \| normal \| uniform \| trunc_normal \| xavier_uniform \| xavier_normal \| kaiming_uniform \| kaiming_normal \| … | Basis | **Was:** Initialisierer für diesen Tensor(-Ausschnitt).<br>**Nutzen:** Gezielte Startwerte.<br>**Wofür:** z. B. zeros für Zero-Init. |
| `initializer.mean` | Kommazahl | `0.0` |  | Basis | **Was:** Mittelwert der Normalverteilung.<br>**Nutzen:** Verschiebt alle Startwerte.<br>**Wofür:** Fast immer 0. |
| `initializer.std` | Kommazahl | `0.02` | ≥ 0.0 | Basis | **Was:** Standardabweichung.<br>**Nutzen:** Bestimmt die Startgröße der Gewichte.<br>**Wofür:** Transformer: 0.02 (GPT/BERT-Konvention). |

#### `tensor_op:scale` — Stufe Basis

**Was:** Werte mit Faktor multiplizieren.

**Nutzen:** Abschwächen/Verstärken einzelner Verbindungen.

**Wofür:** Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `factor` | Kommazahl | `1.0` |  | Basis | **Was:** Faktor.<br>**Nutzen:** Skaliert Werte bzw. Lernrate/Gradienten.<br>**Wofür:** 0.1–10. |

#### `tensor_op:add` — Stufe Basis

**Was:** Konstante addieren.

**Nutzen:** Verschieben von Werten (z. B. Gate-Bias).

**Wofür:** Forget-Gate-Bias +1.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `value` | Kommazahl | `0.0` |  | Basis | **Was:** Wert.<br>**Nutzen:** Wird gesetzt bzw. addiert.<br>**Wofür:** Beliebig. |

#### `tensor_op:set` — Stufe Basis

**Was:** Auf festen Wert setzen.

**Nutzen:** Exakte Kontrolle.

**Wofür:** Ablationen (z. B. einen Kopf auf 0).

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `value` | Kommazahl | `0.0` |  | Basis | **Was:** Wert.<br>**Nutzen:** Wird gesetzt bzw. addiert.<br>**Wofür:** Beliebig. |

#### `tensor_op:noise` — Stufe Basis

**Was:** Rauschen hinzufügen.

**Nutzen:** Symmetriebruch, Robustheitstests, 'Shrink & Perturb'.

**Wofür:** Warm-Starts.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `std` | Kommazahl | `0.01` | ≥ 0.0 | Basis | **Was:** Rauschstärke.<br>**Nutzen:** Störung der Werte.<br>**Wofür:** 1e-3–1e-1. |
| `distribution` | Text | `'normal'` | Auswahl: normal \| uniform | Basis | **Was:** Verteilung.<br>**Nutzen:** Form des Rauschens.<br>**Wofür:** normal. |
| `relative` | Ja/Nein | `False` |  | Basis | **Was:** Relativ zur Streuung des Tensors.<br>**Nutzen:** Skaleninvariante Störung.<br>**Wofür:** Robustheitstests. |

#### `tensor_op:clamp` — Stufe Basis

**Was:** Werte einmalig begrenzen.

**Nutzen:** Entfernt Ausreißer.

**Wofür:** Reparatur.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `min` | Kommazahl (oder None) | `None` |  | Basis | **Was:** Untergrenze.<br>**Nutzen:** Begrenzt Werte nach unten.<br>**Wofür:** Beliebig. |
| `max` | Kommazahl (oder None) | `None` |  | Basis | **Was:** Obergrenze.<br>**Nutzen:** Begrenzt Werte nach oben.<br>**Wofür:** Beliebig. |

#### `tensor_op:prune` — Stufe Basis

**Was:** Pruning: Gewichte entfernen (Maske bleibt im Training erhalten).

**Nutzen:** Kleinere, spärliche Modelle.

**Wofür:** Kompression, Lottery-Ticket-Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `amount` | Kommazahl | `0.5` | Bereich 0.0 … 1.0 | Basis | **Was:** Anteil zu entfernender Gewichte.<br>**Nutzen:** Mehr = kleineres/schnelleres, aber schwächeres Modell.<br>**Wofür:** 0.3–0.9. |
| `method` | Text | `'magnitude'` | Auswahl: magnitude \| random | Basis | **Was:** Auswahlkriterium.<br>**Nutzen:** Magnitude ist fast immer besser.<br>**Wofür:** magnitude. |
| `structure` | Text | `'none'` | Auswahl: none \| rows \| cols | Fortgeschritten | **Was:** Struktur des Prunings.<br>**Nutzen:** Strukturiertes Pruning beschleunigt echte Hardware.<br>**Wofür:** Kompression. |

#### `tensor_op:low_rank` — Stufe Fortgeschritten

**Was:** Auf niedrigen Rang projizieren (SVD).

**Nutzen:** Begrenzt die Komplexität einer Abbildung.

**Wofür:** Kompressionsstudien.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `rank` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Ziel-Rang (SVD-Abschneiden).<br>**Nutzen:** Entfernt 'feine' Richtungen, glättet die Abbildung.<br>**Wofür:** Kompressionsstudien. |

#### `tensor_op:orthogonalize` — Stufe Fortgeschritten

**Was:** Einmalig orthogonalisieren.

**Nutzen:** Normerhaltende Abbildung.

**Wofür:** RNNs, tiefe Netze.

#### `tensor_op:load` — Stufe Basis

**Was:** Werte aus Datei laden.

**Nutzen:** Externe/vortrainierte Gewichte für einzelne Tensoren.

**Wofür:** Transfer, Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `path` | Text | `'gewichte.pt'` |  | Basis | **Was:** Datei mit Werten (.pt: Tensor oder dict; .npy: NumPy-Array).<br>**Nutzen:** Übernimmt externe Gewichte.<br>**Wofür:** Transfer. |
| `key` | Text (oder None) | `None` |  | Basis | **Was:** Schlüssel im dict (.pt).<br>**Nutzen:** Welcher Tensor.<br>**Wofür:** Bei dict-Dateien. |

#### `tensor_op:custom` — Stufe Experte

**Was:** Eigene Funktion anwenden.

**Nutzen:** Unbegrenzte Freiheit.

**Wofür:** Forschung.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `function` | Komponente aus Registry 'tensor_function' | `identity` | Auswahl: identity \| abs \| sign \| normalize_rows \| flip \| shuffle | Basis | **Was:** Eigene Funktion aus der Registry 'tensor_function'.<br>**Nutzen:** Beliebige Manipulation.<br>**Wofür:** Eigene Registrierung. |

#### `tensor_op:copy` (Aliase: `kopieren`, `uebernehmen`) — Stufe Basis

**Was:** Werte aus einem anderen Parameter übernehmen (kopieren, addieren, mitteln, interpolieren).

**Nutzen:** Übernahme von Gewichten zwischen Schichten – ganz oder ausschnittsweise.

**Wofür:** Gleiche Startgewichte für mehrere Schichten, Wissenstransfer zwischen Köpfen.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `source` | Text | `''` |  | Basis | **Was:** Name des Quell-Parameters (z. B. 'blocks.0.mixer.attn.q_proj.weight').<br>**Nutzen:** Werte eines anderen Tensors übernehmen (statt nur Einstellungen).<br>**Wofür:** Schichten gleich initialisieren, Gewichte von Schicht 0 in Schicht 5 übernehmen. |
| `source_index` | Text (oder None) | `None` |  | Fortgeschritten | **Was:** Ausschnitt der Quelle (Python-Syntax wie 'index').<br>**Nutzen:** Nur Teile übernehmen (z. B. einen Kopf).<br>**Wofür:** '0:8, :'. |
| `mode` | Text | `'replace'` | Auswahl: replace \| add \| mean \| interpolate | Basis | **Was:** Wie Quelle und Ziel kombiniert werden.<br>**Nutzen:** Vom Kopieren bis zum sanften Angleichen.<br>**Wofür:** replace. |
| `alpha` | Kommazahl | `0.5` | Bereich 0.0 … 1.0 | Fortgeschritten | **Was:** Anteil der Quelle bei mode = interpolate.<br>**Nutzen:** Mischungsverhältnis.<br>**Wofür:** 0.5. |

#### `tensor_op:tie` — Stufe Fortgeschritten

**Was:** Gewichte mit anderem Parameter teilen (gleiches Tensor-Objekt).

**Nutzen:** Weight Tying.

**Wofür:** Kopplung von Schichten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `source` | Text | `''` |  | Basis | **Was:** Exakter Name des Quellparameters, dessen Gewichte geteilt werden.<br>**Nutzen:** Weight Tying: beide Stellen nutzen denselben Tensor (weniger Parameter, Regularisierung).<br>**Wofür:** Encoder/Decoder-Kopplung, Siamesische Teile. |

#### `tensor_op:lr_scale` — Stufe Basis

**Was:** Eigene Lernrate (Faktor auf die Basis-Lernrate).

**Nutzen:** Langsameres/schnelleres Lernen einzelner Teile.

**Wofür:** Kleine LR für Embeddings, große für neue Köpfe.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `factor` | Kommazahl | `1.0` |  | Basis | **Was:** Faktor.<br>**Nutzen:** Skaliert Werte bzw. Lernrate/Gradienten.<br>**Wofür:** 0.1–10. |

#### `tensor_op:weight_decay` — Stufe Basis

**Was:** Eigener Weight-Decay.

**Nutzen:** Individuelle Regularisierung.

**Wofür:** Feintuning.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `value` | Kommazahl | `0.0` | ≥ 0.0 | Basis | **Was:** Weight-Decay nur für diese Parameter.<br>**Nutzen:** Individuelle Regularisierung.<br>**Wofür:** 0 (aus) bis 0.1. |

#### `tensor_op:grad_scale` — Stufe Basis

**Was:** Gradienten skalieren.

**Nutzen:** Wie lr_scale, aber unabhängig vom Optimierer-Zustand.

**Wofür:** Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `factor` | Kommazahl | `1.0` |  | Basis | **Was:** Faktor.<br>**Nutzen:** Skaliert Werte bzw. Lernrate/Gradienten.<br>**Wofür:** 0.1–10. |

#### `tensor_op:grad_clip` — Stufe Basis

**Was:** Gradientennorm dieses Tensors begrenzen.

**Nutzen:** Lokale Stabilität.

**Wofür:** Instabile Einzelschichten.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `max_norm` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Maximale Norm des Gradienten dieses Tensors.<br>**Nutzen:** Lokale Stabilisierung einzelner Schichten.<br>**Wofür:** 0.1–10. |

#### `tensor_op:grad_noise` — Stufe Fortgeschritten

**Was:** Rauschen auf Gradienten.

**Nutzen:** Regularisierung/Erkundung.

**Wofür:** Experimente.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `std` | Kommazahl | `0.001` | ≥ 0.0 | Basis | **Was:** Rauschstärke auf den Gradienten.<br>**Nutzen:** Erkundung, Regularisierung.<br>**Wofür:** 1e-4–1e-2. |

#### `tensor_op:grad_mask` — Stufe Fortgeschritten

**Was:** Gradienten nullen (Werte können sich durch Weight-Decay/Momentum noch ändern; freeze hält sie fest).

**Nutzen:** Weiches Einfrieren.

**Wofür:** Experimente.

#### `tensor_op:constraint` — Stufe Basis

**Was:** Beschränkung nach jedem Schritt erzwingen.

**Nutzen:** Stabilität, physikalische/strukturelle Randbedingungen.

**Wofür:** Max-Norm-Regularisierung, nicht-negative Gewichte, Orthogonalität.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `kind` | Text | `'max_norm'` | Auswahl: max_norm \| unit_norm \| non_negative \| clip \| orthogonal | Basis | **Was:** Art der Beschränkung.<br>**Nutzen:** Wird nach jedem Optimierungsschritt erzwungen.<br>**Wofür:** Stabilität, Interpretierbarkeit, physikalische Randbedingungen. |
| `value` | Kommazahl | `1.0` | ≥ 0.0 | Basis | **Was:** Grenzwert.<br>**Nutzen:** Stärke der Beschränkung.<br>**Wofür:** 1–4. |
| `every_n_steps` | Ganzzahl | `1` | ≥ 1 | Fortgeschritten | **Was:** Nur alle n Schritte erzwingen.<br>**Nutzen:** Spart Zeit (orthogonal ist teuer).<br>**Wofür:** 1; orthogonal: 10. |


### Registry `tensor_function`

<a id="registry-tensor-function"></a>

**Tensor-Funktionen (Chirurgie)** – Eigene Funktionen f(tensor) → tensor für die Operation 'custom'. Beliebige, selbst definierte Gewichtsmanipulationen.

Fabrik-Signatur: `factory(args) -> callable(tensor) -> tensor`

#### `tensor_function:identity` — Stufe Basis

**Was:** Unverändert.

**Nutzen:** Platzhalter.

**Wofür:** Test.

#### `tensor_function:abs` — Stufe Basis

**Was:** Beträge.

**Nutzen:** Nur positive Gewichte.

**Wofür:** Experimente.

#### `tensor_function:sign` — Stufe Basis

**Was:** Vorzeichen (-1/0/1).

**Nutzen:** Binäre Gewichte.

**Wofür:** Binarisierungs-Experimente.

#### `tensor_function:normalize_rows` — Stufe Basis

**Was:** Zeilen auf Länge 1.

**Nutzen:** Normierte Gewichtsvektoren.

**Wofür:** Kosinus-Klassifikatoren.

#### `tensor_function:flip` — Stufe Basis

**Was:** Reihenfolge umkehren (Achse 0).

**Nutzen:** Permutation.

**Wofür:** Ablationen.

#### `tensor_function:shuffle` — Stufe Basis

**Was:** Werte zufällig permutieren (Verteilung bleibt, Struktur geht verloren).

**Nutzen:** Kontrollexperiment.

**Wofür:** Ablationen.


### Registry `dataset`

<a id="registry-dataset"></a>

**Datensätze** – Datenquellen: synthetische Beispielaufgaben und Lader für eigene Dateien. Sofort lauffähige Experimente und einfacher Weg zu eigenen Daten.

Fabrik-Signatur: `factory(args) -> DatasetBundle(train, val, info)`

#### `dataset:multimodal` — Stufe Basis

**Was:** Bild + Metadaten, Label hängt von beiden ab.

**Nutzen:** Testet echte Fusion zweier Modalitäten.

**Wofür:** Fusion-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `size` | Ganzzahl | `16` | ≥ 8 | Basis | **Was:** Bildgröße (quadratisch).<br>**Nutzen:** Auflösung.<br>**Wofür:** 16–32. |
| `channels` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Kanäle.<br>**Nutzen:** 1 = grau, 3 = farbig.<br>**Wofür:** 1 oder 3. |
| `noise` | Kommazahl | `0.1` | ≥ 0.0 | Basis | **Was:** Pixelrauschen.<br>**Nutzen:** Schwierigkeit.<br>**Wofür:** 0.05–0.3. |
| `meta_dim` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Länge des Zusatzvektors (Metadaten).<br>**Nutzen:** Zweite Modalität.<br>**Wofür:** 2–16. |

#### `dataset:classification` (Aliase: `klassifikation`) — Stufe Basis

**Was:** Synthetische 2D-Klassifikation (Spiralen, Monde, XOR …).

**Nutzen:** Sichtbar nichtlineare Entscheidungsgrenzen.

**Wofür:** MLP-Tests, Tabellen-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `kind` | Text | `'spirals'` | Auswahl: blobs \| xor \| spirals \| moons \| rings | Basis | **Was:** Form der Klassen.<br>**Nutzen:** Steuert die Schwierigkeit (blobs leicht, spirals schwer).<br>**Wofür:** spirals für Nichtlinearität. |
| `n_classes` | Ganzzahl | `3` | ≥ 2 | Basis | **Was:** Anzahl Klassen.<br>**Nutzen:** Ausgabegröße.<br>**Wofür:** 2–10. |
| `n_features` | Ganzzahl | `2` | ≥ 2 | Basis | **Was:** Merkmale (2 = sichtbar; weitere sind Rauschdimensionen).<br>**Nutzen:** Zusätzliche irrelevante Merkmale testen Robustheit.<br>**Wofür:** 2–50. |
| `noise` | Kommazahl | `0.05` | ≥ 0.0 | Basis | **Was:** Rauschen.<br>**Nutzen:** Überlappung der Klassen (ab ~0.1 überlappen die Spiralarme im Zentrum stark).<br>**Wofür:** 0.02–0.2. |

#### `dataset:regression` — Stufe Basis

**Was:** Synthetische Regression (sin, Polynom, Friedman).

**Nutzen:** Kontinuierliche Zielwerte.

**Wofür:** Regressions-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `function` | Text | `'sin'` | Auswahl: linear \| sin \| poly \| friedman | Basis | **Was:** Zielfunktion.<br>**Nutzen:** Schwierigkeit der Regression.<br>**Wofür:** sin oder friedman. |
| `n_features` | Ganzzahl | `5` | ≥ 1 | Basis | **Was:** Merkmale.<br>**Nutzen:** Eingabebreite.<br>**Wofür:** 5–20. |
| `noise` | Kommazahl | `0.05` | ≥ 0.0 | Basis | **Was:** Rauschen auf dem Ziel.<br>**Nutzen:** Messfehler.<br>**Wofür:** 0–0.2. |

#### `dataset:sequence_classification` (Aliase: `sequenz`) — Stufe Basis

**Was:** Sequenzklassifikation (Parität, Enthaltensein, Mehrheit).

**Nutzen:** Testet Gedächtnis von RNNs/Transformern.

**Wofür:** Sequenzmodelle.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `task` | Text | `'parity'` | Auswahl: parity \| contains \| majority | Basis | **Was:** Sequenzklassifikationsaufgabe.<br>**Nutzen:** Testet Langzeitgedächtnis / globale Aggregation.<br>**Wofür:** parity ist schwer. |
| `seq_len` | Ganzzahl | `16` | ≥ 2 | Basis | **Was:** Folgenlänge.<br>**Nutzen:** Länger = schwieriger.<br>**Wofür:** 8–64. |
| `vocab_size` | Ganzzahl | `8` | ≥ 2 | Basis | **Was:** Vokabular.<br>**Nutzen:** Anzahl Symbole.<br>**Wofür:** 2–20. |

#### `dataset:text` (Aliase: `char_lm`, `zeichen`) — Stufe Basis

**Was:** Zeichenbasierter Text für Sprachmodelle (eigener Text oder Beispieltext).

**Nutzen:** Sofort lauffähiges Mini-LLM.

**Wofür:** causal_lm-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `text` | Text (oder None) | `None` |  | Basis | **Was:** Eigener Text (None = eingebauter Beispieltext).<br>**Nutzen:** Trainingsmaterial des Sprachmodells.<br>**Wofür:** Beliebiger Text. |
| `path` | Text (oder None) | `None` |  | Basis | **Was:** Textdatei (UTF-8) statt 'text'.<br>**Nutzen:** Eigene Korpora.<br>**Wofür:** z. B. buch.txt. |
| `seq_len` | Ganzzahl | `64` | ≥ 2 | Basis | **Was:** Länge der Trainingsausschnitte.<br>**Nutzen:** Kontextlänge beim Training.<br>**Wofür:** 32–512. |
| `stride` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Schrittweite der Ausschnitte.<br>**Nutzen:** Kleiner = mehr (überlappende) Beispiele.<br>**Wofür:** seq_len/4. |
| `repeat` | Ganzzahl | `4` | ≥ 1 | Fortgeschritten | **Was:** Text so oft wiederholen (für sehr kurze Texte).<br>**Nutzen:** Mehr Trainingsbeispiele.<br>**Wofür:** 1–20. |

#### `dataset:seq2seq_toy` (Aliase: `seq2seq`) — Stufe Basis

**Was:** Sequenz-Transformationen (umkehren, sortieren …).

**Nutzen:** Encoder-Decoder-Tests.

**Wofür:** seq2seq-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `task` | Text | `'reverse'` | Auswahl: reverse \| copy \| sort \| shift | Basis | **Was:** Sequenz-zu-Sequenz-Aufgabe.<br>**Nutzen:** Testet Encoder-Decoder-Fähigkeiten.<br>**Wofür:** reverse. |
| `seq_len` | Ganzzahl | `8` | ≥ 2 | Basis | **Was:** Folgenlänge.<br>**Nutzen:** Schwierigkeit.<br>**Wofür:** 6–20. |
| `vocab_size` | Ganzzahl | `12` | ≥ 4 | Basis | **Was:** Vokabular (0 = Start-Token, 1.. = Symbole).<br>**Nutzen:** Anzahl Symbole.<br>**Wofür:** 10–50. |

#### `dataset:time_series` (Aliase: `zeitreihe`) — Stufe Basis

**Was:** Sinus-Mischungen für Prognosen.

**Nutzen:** Fenster → Zukunft.

**Wofür:** Zeitreihen-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `window` | Ganzzahl | `32` | ≥ 2 | Basis | **Was:** Eingabefenster (vergangene Schritte).<br>**Nutzen:** Wie weit zurückgeschaut wird.<br>**Wofür:** 16–128. |
| `horizon` | Ganzzahl | `4` | ≥ 1 | Basis | **Was:** Vorhersagehorizont (zukünftige Schritte).<br>**Nutzen:** Prognosereichweite.<br>**Wofür:** 1–24. |
| `n_series` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Variablen je Zeitschritt.<br>**Nutzen:** Multivariate Reihen.<br>**Wofür:** 1–8. |
| `noise` | Kommazahl | `0.05` | ≥ 0.0 | Basis | **Was:** Rauschen.<br>**Nutzen:** Schwierigkeit.<br>**Wofür:** 0–0.2. |
| `output` | Text | `'vector'` | Auswahl: vector \| sequence | Fortgeschritten | **Was:** Form des Ziels.<br>**Nutzen:** Passend zum Modell.<br>**Wofür:** vector. |

#### `dataset:shapes` (Aliase: `bilder`, `images`) — Stufe Basis

**Was:** Synthetische Bilder: Quadrat, Kreis, Kreuz.

**Nutzen:** Bildklassifikation ohne Download.

**Wofür:** CNN/ViT-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `size` | Ganzzahl | `16` | ≥ 8 | Basis | **Was:** Bildgröße (quadratisch).<br>**Nutzen:** Auflösung.<br>**Wofür:** 16–32. |
| `channels` | Ganzzahl | `1` | ≥ 1 | Basis | **Was:** Kanäle.<br>**Nutzen:** 1 = grau, 3 = farbig.<br>**Wofür:** 1 oder 3. |
| `noise` | Kommazahl | `0.1` | ≥ 0.0 | Basis | **Was:** Pixelrauschen.<br>**Nutzen:** Schwierigkeit.<br>**Wofür:** 0.05–0.3. |

#### `dataset:graphs` (Aliase: `graphen`) — Stufe Basis

**Was:** Kleine Graphen: Ring vs. Baum.

**Nutzen:** Graph-Klassifikation.

**Wofür:** GNN-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `400` | ≥ 2 | Basis | **Was:** Anzahl Graphen.<br>**Nutzen:** Datenmenge.<br>**Wofür:** 200–2000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `min_nodes` | Ganzzahl | `6` | ≥ 3 | Basis | **Was:** Minimale Knotenzahl.<br>**Nutzen:** Graphgröße.<br>**Wofür:** 5–10. |
| `max_nodes` | Ganzzahl | `14` | ≥ 3 | Basis | **Was:** Maximale Knotenzahl.<br>**Nutzen:** Graphgröße.<br>**Wofür:** 10–30. |
| `feature_dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Merkmale je Knoten.<br>**Nutzen:** Eingabebreite.<br>**Wofür:** 4–16. |

#### `dataset:distribution_2d` (Aliase: `verteilung`) — Stufe Basis

**Was:** 2D-Verteilungen für generative Modelle.

**Nutzen:** Sichtbarer Generierungserfolg.

**Wofür:** GAN/Diffusion-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `kind` | Text | `'moons'` | Auswahl: moons \| gaussians \| ring \| spiral | Basis | **Was:** 2D-Zielverteilung für generative Modelle.<br>**Nutzen:** Sichtbarer Generierungserfolg.<br>**Wofür:** moons, gaussians. |
| `noise` | Kommazahl | `0.05` | ≥ 0.0 | Basis | **Was:** Rauschen.<br>**Nutzen:** Breite der Verteilung.<br>**Wofür:** 0.02–0.1. |

#### `dataset:preferences` (Aliase: `praeferenzen`) — Stufe Basis

**Was:** Präferenzpaare (bevorzugt/abgelehnt) aus verstecktem Urteil.

**Nutzen:** Reward-Modell-Training.

**Wofür:** Präferenz-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `dim` | Ganzzahl | `8` | ≥ 1 | Basis | **Was:** Merkmale je Option.<br>**Nutzen:** Eingabebreite des Reward-Modells.<br>**Wofür:** 4–32. |
| `noise` | Kommazahl | `0.1` | ≥ 0.0 | Basis | **Was:** Rauschen in den Präferenzen.<br>**Nutzen:** Unzuverlässige Bewerter.<br>**Wofür:** 0–0.5. |

#### `dataset:anomaly` (Aliase: `anomalien`) — Stufe Basis

**Was:** Normaldaten zum Training + Testmenge mit Anomalien.

**Nutzen:** Anomalieerkennung per Autoencoder.

**Wofür:** AE-Presets.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `n_samples` | Ganzzahl | `1000` | ≥ 2 | Basis | **Was:** Anzahl Beispiele.<br>**Nutzen:** Mehr Daten = bessere Generalisierung.<br>**Wofür:** 500–10000. |
| `seed` | Ganzzahl | `0` |  | Fortgeschritten | **Was:** Seed der Datenerzeugung.<br>**Nutzen:** Reproduzierbare Daten.<br>**Wofür:** 0. |
| `n_features` | Ganzzahl | `16` | ≥ 2 | Basis | **Was:** Merkmale.<br>**Nutzen:** Eingabebreite.<br>**Wofür:** 8–64. |
| `anomaly_fraction` | Kommazahl | `0.05` | Bereich 0.0 … 0.5 | Basis | **Was:** Anteil Anomalien in der Testmenge.<br>**Nutzen:** Seltenheit der Ausreißer.<br>**Wofür:** 0.01–0.1. |

#### `dataset:csv` — Stufe Basis

**Was:** Eigene CSV-Datei (numerische Spalten).

**Nutzen:** Eigene Tabellendaten ohne Code.

**Wofür:** Echte Projekte.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `path` | Text | `'daten.csv'` |  | Basis | **Was:** CSV-Datei mit Kopfzeile.<br>**Nutzen:** Eigene Tabellendaten.<br>**Wofür:** Nur numerische Spalten. |
| `target` | Text (oder None) | `None` |  | Basis | **Was:** Zielspalte (None = letzte Spalte).<br>**Nutzen:** Was vorhergesagt werden soll.<br>**Wofür:** Spaltenname. |
| `task` | Text | `'auto'` | Auswahl: auto \| classification \| regression | Basis | **Was:** Aufgabentyp.<br>**Nutzen:** Datentyp des Ziels.<br>**Wofür:** auto. |
| `delimiter` | Text | `','` |  | Fortgeschritten | **Was:** Trennzeichen.<br>**Nutzen:** CSV-Dialekt.<br>**Wofür:** ',' oder ';'. |
| `standardize` | Ja/Nein | `True` |  | Fortgeschritten | **Was:** Merkmale standardisieren (Mittel 0, Std 1).<br>**Nutzen:** Wichtig für neuronale Netze.<br>**Wofür:** an. |

#### `dataset:tensor_file` (Aliase: `pt`) — Stufe Basis

**Was:** Eigene torch.save-Datei mit Tensoren.

**Nutzen:** Beliebige vorbereitete Daten.

**Wofür:** Echte Projekte.

| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |
|---|---|---|---|---|---|
| `path` | Text | `'daten.pt'` |  | Basis | **Was:** torch.save-Datei mit dict von Tensoren (z. B. inputs/targets).<br>**Nutzen:** Beliebige vorbereitete Daten.<br>**Wofür:** Eigene Daten. |
| `val_path` | Text (oder None) | `None` |  | Basis | **Was:** Optionale Validierungsdatei.<br>**Nutzen:** Eigene Validierung.<br>**Wofür:** Optional. |

#### `dataset:none` (Aliase: `keine`) — Stufe Basis

**Was:** Keine Daten (z. B. Reinforcement Learning).

**Nutzen:** RL erzeugt Daten selbst.

**Wofür:** RL-Presets.


### Registry `preset`

<a id="registry-preset"></a>

**Vorlagen (Presets)** – Vollständig vorkonfigurierte KI-Projekte je Anwendungsgebiet. Sinnvolle Standardwerte als Startpunkt – alles bleibt änderbar.

Fabrik-Signatur: `factory(args) -> dict (ProjectConfig)`

#### `preset:sprachmodell_mini` — Stufe Basis

**Was:** Kleines GPT-artiges Sprachmodell (RoPE, SwiGLU, RMSNorm, Pre-Norm) auf Zeichenebene.

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'text'.

#### `preset:sprachmodell_moe` — Stufe Basis

**Was:** Sprachmodell mit Mixture-of-Experts in jeder zweiten Schicht.

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'text'.

#### `preset:sprachmodell_mamba` — Stufe Basis

**Was:** Mamba-artiges State-Space-Sprachmodell (ohne Attention).

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'ssm', Daten 'text'.

#### `preset:sprachmodell_hybrid` — Stufe Basis

**Was:** Hybrid aus SSM- und Attention-Schichten (Jamba-Prinzip).

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'hybrid', Daten 'text'.

#### `preset:sprachmodell_speichersparend` — Stufe Basis

**Was:** Größeres Sprachmodell mit allen Speicher-Weichen: GQA, Gradient-Checkpointing, bf16-Autocast, Akkumulation.

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'text'.

#### `preset:sprachmodell_qlora` — Stufe Basis

**Was:** Feintuning wie QLoRA: Basis in nf4 (4 Bit) eingefroren, nur LoRA-Adapter + Normen lernen, 8-Bit-AdamW.

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'text'.

#### `preset:sprachmodell_ram_streaming` — Stufe Basis

**Was:** Großes Modell auf kleiner GPU trainieren: Blöcke samt Optimierer-Zuständen im RAM, blockweise gestreamt.

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'text'.

#### `preset:text_klassifikation` — Stufe Basis

**Was:** Encoder-Transformer (BERT-artig) mit CLS-Token zur Sequenzklassifikation.

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'sequence_classification'.

#### `preset:bert_vortraining` — Stufe Basis

**Was:** Maskiertes Sprachmodell-Vortraining (BERT-Stil).

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'text'.

#### `preset:uebersetzung_seq2seq` — Stufe Basis

**Was:** Encoder-Decoder-Transformer für Sequenz-zu-Sequenz (hier: Folgen umkehren).

**Nutzen:** Anwendungsgebiet: Sprachmodelle, Textverständnis, Übersetzung.

**Wofür:** Modell 'transformer', Daten 'seq2seq_toy'.

#### `preset:bildklassifikation_cnn` — Stufe Basis

**Was:** ResNet-artiges CNN mit Label Smoothing und Mixup.

**Nutzen:** Anwendungsgebiet: Bildklassifikation, Bildgenerierung.

**Wofür:** Modell 'cnn', Daten 'shapes'.

#### `preset:bildklassifikation_vit` — Stufe Basis

**Was:** Vision Transformer (Patches, CLS-Token, Stochastic Depth).

**Nutzen:** Anwendungsgebiet: Bildklassifikation, Bildgenerierung.

**Wofür:** Modell 'transformer', Daten 'shapes'.

#### `preset:bildklassifikation_mobil` — Stufe Basis

**Was:** Effizientes MobileNet/EfficientNet-artiges CNN (Inverted Residuals + SE).

**Nutzen:** Anwendungsgebiet: Bildklassifikation, Bildgenerierung.

**Wofür:** Modell 'cnn', Daten 'shapes'.

#### `preset:vae_bilder` — Stufe Basis

**Was:** Faltungs-VAE für Bilder (β-VAE mit KL-Annealing).

**Nutzen:** Anwendungsgebiet: Daten erzeugen: GAN, Diffusion, VAE.

**Wofür:** Modell 'autoencoder', Daten 'shapes'.

#### `preset:zeitreihen_lstm` — Stufe Basis

**Was:** LSTM-Prognose mehrerer Zukunftsschritte (Huber-Verlust).

**Nutzen:** Anwendungsgebiet: Prognosen, Sensordaten, Signale.

**Wofür:** Modell 'rnn', Daten 'time_series'.

#### `preset:zeitreihen_transformer` — Stufe Basis

**Was:** Transformer-Encoder für kontinuierliche Zeitreihen.

**Nutzen:** Anwendungsgebiet: Prognosen, Sensordaten, Signale.

**Wofür:** Modell 'transformer', Daten 'time_series'.

#### `preset:sequenz_gedaechtnis_rnn` — Stufe Basis

**Was:** Gemischter RNN-Stapel (LSTM → GRU → minGRU) auf einer Gedächtnisaufgabe.

**Nutzen:** Anwendungsgebiet: Prognosen, Sensordaten, Signale.

**Wofür:** Modell 'rnn', Daten 'sequence_classification'.

#### `preset:tabellen_klassifikation` — Stufe Basis

**Was:** Tiefes Residual-MLP mit LayerNorm, GELU und Dropout.

**Nutzen:** Anwendungsgebiet: Tabellendaten: Klassifikation und Regression.

**Wofür:** Modell 'mlp', Daten 'classification'.

#### `preset:tabellen_regression` — Stufe Basis

**Was:** MLP-Regression (Friedman-Benchmark) mit robustem Huber-Verlust.

**Nutzen:** Anwendungsgebiet: Tabellendaten: Klassifikation und Regression.

**Wofür:** Modell 'mlp', Daten 'regression'.

#### `preset:gan_2d` — Stufe Basis

**Was:** GAN auf einer 2D-Verteilung (Hinge-Verlust, Spectral-Norm, TTUR).

**Nutzen:** Anwendungsgebiet: Daten erzeugen: GAN, Diffusion, VAE.

**Wofür:** Modell 'gan', Daten 'distribution_2d'.

#### `preset:diffusion_2d` — Stufe Basis

**Was:** Diffusionsmodell auf 2D-Gauß-Ring (v-Vorhersage, Min-SNR, EMA, DDIM).

**Nutzen:** Anwendungsgebiet: Daten erzeugen: GAN, Diffusion, VAE.

**Wofür:** Modell 'diffusion', Daten 'distribution_2d'.

#### `preset:graph_klassifikation` — Stufe Basis

**Was:** Graph Isomorphism Network (GIN) zur Unterscheidung von Ringen und Bäumen.

**Nutzen:** Anwendungsgebiet: Graphen: Moleküle, Netzwerke.

**Wofür:** Modell 'gnn', Daten 'graphs'.

#### `preset:anomalie_autoencoder` — Stufe Basis

**Was:** Denoising-Autoencoder: hoher Rekonstruktionsfehler = Anomalie.

**Nutzen:** Anwendungsgebiet: Anomalie- und Ausreißererkennung.

**Wofür:** Modell 'autoencoder', Daten 'anomaly'.

#### `preset:reward_modell` — Stufe Basis

**Was:** Reward-Modell aus Präferenzpaaren (Bradley-Terry) – Baustein für RLHF.

**Nutzen:** Anwendungsgebiet: Lernen aus Präferenzen (Reward-Modelle, RLHF-Bausteine).

**Wofür:** Modell 'mlp', Daten 'preferences'.

#### `preset:rl_cartpole_ppo` — Stufe Basis

**Was:** PPO balanciert einen Stab (CartPole).

**Nutzen:** Anwendungsgebiet: Reinforcement Learning mit frei kombinierbaren Belohnungen.

**Wofür:** Modell 'actor_critic', Daten 'none'.

#### `preset:rl_cartpole_dqn` — Stufe Basis

**Was:** Double-DQN mit n-Schritt-Erträgen und priorisiertem Replay (CartPole).

**Nutzen:** Anwendungsgebiet: Reinforcement Learning mit frei kombinierbaren Belohnungen.

**Wofür:** Modell 'q_network', Daten 'none'.

#### `preset:rl_gitterwelt_belohnungen` — Stufe Basis

**Was:** Gitterwelt mit kombinierter Belohnung: Ziel + Zeitstrafe + Potential-Formung + Wandstrafe + Neuheitsbonus.

**Nutzen:** Anwendungsgebiet: Reinforcement Learning mit frei kombinierbaren Belohnungen.

**Wofür:** Modell 'actor_critic', Daten 'none'.

#### `preset:rl_pendel_kontinuierlich` — Stufe Basis

**Was:** PPO mit Gauß-Policy für kontinuierliche Steuerung (Pendel) + Energiestrafe.

**Nutzen:** Anwendungsgebiet: Reinforcement Learning mit frei kombinierbaren Belohnungen.

**Wofür:** Modell 'actor_critic', Daten 'none'.

#### `preset:ensemble_tabellen` — Stufe Basis

**Was:** Ensemble dreier unterschiedlicher MLPs (Wahrscheinlichkeits-Mittel).

**Nutzen:** Anwendungsgebiet: Kombinationen mehrerer Modelle (Ensemble, Fusion, Mixture).

**Wofür:** Modell 'ensemble', Daten 'classification'.

#### `preset:multimodal_fusion` — Stufe Basis

**Was:** Fusion von Bild (CNN) und Metadaten (MLP) per Attention.

**Nutzen:** Anwendungsgebiet: Kombinationen mehrerer Modelle (Ensemble, Fusion, Mixture).

**Wofür:** Modell 'fusion', Daten 'multimodal'.

#### `preset:mixture_of_models` — Stufe Basis

**Was:** Router wählt pro Beispiel zwischen verschiedenen Experten-Modellen.

**Nutzen:** Anwendungsgebiet: Kombinationen mehrerer Modelle (Ensemble, Fusion, Mixture).

**Wofür:** Modell 'mixture_of_models', Daten 'regression'.

#### `preset:freies_netz` — Stufe Basis

**Was:** Frei definierter Netzgraph mit Verzweigung, Gating und Residual.

**Nutzen:** Anwendungsgebiet: Experimentelle und exotische Konfigurationen.

**Wofür:** Modell 'graph', Daten 'classification'.

#### `preset:kontrastiv_vortraining` — Stufe Basis

**Was:** Selbstüberwachtes kontrastives Lernen (SimCLR-artig) ohne Labels.

**Nutzen:** Anwendungsgebiet: Experimentelle und exotische Konfigurationen.

**Wofür:** Modell 'mlp', Daten 'classification'.

#### `preset:forschung_experimentell` — Stufe Basis

**Was:** Exotischer Transformer: Sigmoid-Attention, QK-Norm, Output-Gate, DyT statt Norm, Sandwich, Muon + WSD.

**Nutzen:** Anwendungsgebiet: Experimentelle und exotische Konfigurationen.

**Wofür:** Modell 'transformer', Daten 'text'.

