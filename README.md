# KI-Baukasten

Ein Python-Baukasten (PyTorch), mit dem sich **KIs und ganze KI-Systeme bis ins kleinste Detail** zusammenstellen,
kombinieren, trainieren und speichern lassen – mit einer **Erklärung zu jeder einzelnen Einstellung**.

- **982 einzeln dokumentierte Einstellungen** – jede mit Tooltip: *Was ist das? Warum/Nutzen? Wofür wichtig? Tipp*
- **333 austauschbare Komponenten** in 24 Registries (Aktivierungen, Normen, Attention-Kernel, Mixer, FFN/MoE/KAN,
  Modelle, Optimierer, Zeitpläne, Verluste, Belohnungen, Tensor- und Modul-Operationen …) – alle per Name überall einsetzbar
- **18 Modelle**, beliebig kombinierbar und verschachtelbar (Ensemble aus Pipelines aus Transformern …)
- **12 Trainings-Paradigmen** inkl. Reinforcement Learning mit frei kombinierbaren Belohnungen
- **34 Vorlagen** für 11 Anwendungsgebiete – sofort lauffähig ohne Downloads
- **Jede Schicht darf anders sein** (schichtweise Overrides mit **Übernahme** von anderen Schichten), **jedes gebaute
  Modul** (Modul-Chirurgie) und **jeder Tensor-Ausschnitt** (Tensor-Chirurgie) ist einzeln steuerbar – jeweils mit
  Geltungsbereich **für alle / nur bestimmte / nur genau einen / alle außer**
- **Schnelles Mamba**: fusionierter Scan mit eigenem Backward – Training ~12× schneller, ~23× weniger Aktivierungsspeicher
- **KV-Cache & Zustands-Caches** für die Generierung (alle kausalen Mixer; float16/bfloat16/int8, Fenster + Attention-Sinks,
  MLA-lite)
- **Speicher-Weichen**: Präzision, Gradient-Checkpointing, Aktivierungs-Offload, **trainierbares Auslagern von Blöcken
  in den RAM**, Layer-Swapping (CPU/Festplatte), **eigene int8/int4/nf4-Quantisierung** (ohne torch.ao), **LoRA/DoRA/
  QLoRA**, **8-Bit-AdamW**, Speicherschätzung
- **Ratgeber & Aufbau**: Optimierungs-/Speichervorschläge zum direkten Übernehmen, komplette Aufbau-Darstellung
  (Text, Markdown, Mermaid, JSON), Export (torch.export, TorchScript, ONNX)
- **Drei Oberflächen**: Python-API, Kommandozeile (deutsch), grafischer Editor mit Maus-Tooltips, Rückgängig/Wiederholen

Es gibt keine „verbotenen“ Einstellungen: auch experimentelle, exotische oder scheinbar unnötige Optionen sind
änderbar. Erfahrungsstufen (*Basis → Fortgeschritten → Experte → Experimentell*) dienen nur der Orientierung.

---

## Inhalt

1. [Installation](#installation)
2. [Schnellstart](#schnellstart) – Python, Kommandozeile, GUI
3. [Grundkonzepte](#grundkonzepte)
4. [Was alles drin ist](#was-alles-drin-ist)
5. [Vorlagen nach Anwendungsgebiet](#vorlagen-nach-anwendungsgebiet)
6. [Speicher & große Modelle](#speicher--große-modelle) – Streaming, Quantisierung, LoRA, 8-Bit-Optimierer
7. [Mamba, Generierung & KV-Cache](#mamba-generierung--kv-cache)
8. [Aufbau, Ratgeber & Export](#aufbau-ratgeber--export)
9. [Reinforcement Learning & Belohnungen](#reinforcement-learning--belohnungen)
10. [Eigene Komponenten](#eigene-komponenten)
11. [Projektstruktur](#projektstruktur)
12. [Tests](#tests)
13. [Ehrliche Grenzen](#ehrliche-grenzen)

Ausführliche Erklärung des inneren Aufbaus und aller Eingriffspunkte: [`docs/ARCHITEKTUR.md`](docs/ARCHITEKTUR.md).

---

## Installation

```bash
pip install torch            # PyTorch ≥ 2.3 (CPU oder GPU)
pip install -e .             # dieses Paket (optional: .[yaml,rl,dev])
```

Optional: `pyyaml` (Konfiguration als YAML), `gymnasium` (Standard-RL-Umgebungen), `pytest`.
Die grafische Oberfläche nutzt Tkinter (in Python enthalten; unter Debian/Ubuntu ggf. `sudo apt install python3-tk`).

---

## Schnellstart

### Python

```python
from ki_baukasten import Project

p = Project.from_preset("sprachmodell_mini")        # Vorlage: kleines GPT-artiges Sprachmodell
p.set("model.n_layers", 6)                          # jede Einstellung per Pfad …
p.set("model.block.mixer.num_heads", 8)             # … auch tief verschachtelt
p.set("model.layer_overrides", {"-1": {"ffn": "moe"}})   # nur die letzte Schicht als Mixture-of-Experts

p.set_all("dropout", 0.1)                           # eine Einstellung überall setzen (Platzhalter: "model.**.dropout")

print(p.explain("model.block.norm_position"))       # Tooltip + Typ + Bereich + alle Optionen
print(p.search("speicher sparen"))                  # Volltextsuche über Pfade, Tooltips und Komponenten
print(p.memory_report())                            # Speicherschätzung mit Empfehlungen
for tipp in p.advise():                             # Ratgeber: Speicher, Tempo, Stabilität, Struktur
    print(tipp)
p.apply_advice(category="speicher")                 # Vorschläge übernehmen … und bei Bedarf p.undo()
print(p.architecture("text"))                       # komplette Auflistung: Module, Formen, Schichten, Pfade

p.train()                                           # bauen (Größen passen sich den Daten an) + trainieren
print(p.generate("Der Baukasten", max_new_tokens=80))
p.save("ausgaben/mein_llm")                         # projekt.json + modell.pt + verlauf.json
```

### Kommandozeile

```bash
python -m ki_baukasten presets                              # Vorlagen nach Anwendungsgebiet
python -m ki_baukasten modelle                              # alle Modelle
python -m ki_baukasten komponenten                          # alle Registries
python -m ki_baukasten komponenten reward                   # z. B. alle Belohnungsterme
python -m ki_baukasten zeige sprachmodell_mini --stufe experte
python -m ki_baukasten erklaere model.block.mixer.num_kv_heads --quelle sprachmodell_mini
python -m ki_baukasten erklaere optimizer:lion               # Komponente erklären
python -m ki_baukasten vorlage bildklassifikation_cnn -o meine_ki.json --nur-geaendert
python -m ki_baukasten trainieren meine_ki.json --setze training.epochs=20 --speicher sparsam
python -m ki_baukasten speicher sprachmodell_speichersparend --batch 64
python -m ki_baukasten tensoren meine_ki.json --filter stages.1  # Parameternamen für Tensor-Regeln
python -m ki_baukasten aufbau sprachmodell_hybrid --format markdown -o aufbau.md   # Aufbau inkl. Mermaid-Diagramm
python -m ki_baukasten optimieren sprachmodell_mini --anwenden adamw8bit,gqa      # Ratgeber + übernehmen
python -m ki_baukasten suche "kv cache" --quelle sprachmodell_mini               # Einstellungen finden
python -m ki_baukasten vergleiche sprachmodell_mini sprachmodell_mamba           # Unterschiede
python -m ki_baukasten zeige sprachmodell_mini --setze "dropout=0.2"             # Platzhalter: alle 'dropout'
python -m ki_baukasten exportieren ausgaben/sprachmodell_mini --format export    # torch.export (.pt2)
python -m ki_baukasten generieren ausgaben/sprachmodell_mini --prompt "Die " --laenge 120
python -m ki_baukasten doku -o docs/PARAMETER_REFERENZ.md  # komplette Referenz aus den Tooltips
python -m ki_baukasten gui sprachmodell_mini
```

### Grafische Oberfläche

`python -m ki_baukasten gui [vorlage|datei]` öffnet einen Editor mit

- **Baum** aller Einstellungen (Modell, Daten, Training, Speicher, Tensor-Regeln),
- **Formular** des gewählten Knotens – **Maus über einen Namen zeigt den Tooltip**, `?` zeigt alle Details und Optionen,
- **Stufenfilter** (Basis … Experimentell), **Suche** über alle Einstellungen,
- Komponentenwechsel per Auswahlliste, Listen (Verluste, Belohnungen, Stufen, Regeln) hinzufügen/entfernen,
- **Prüfen / Bauen / Speicher / Trainieren** – Training läuft im Hintergrund mit Live-Protokoll,
- **Aufbau** (komplette Auflistung), **Optimieren** (Ratgeber mit Übernahme-Dialog), **↶ Rückgängig / ↷ Wiederholen**.

---

## Grundkonzepte

### 1. Jede Einstellung ist ein erklärter `Param`

```python
class AttentionConfig(BaseConfig):
    num_heads = Param(8, min=1, tooltip=T(
        "Anzahl paralleler Attention-Köpfe.",                       # Was?
        "Jeder Kopf kann andere Beziehungen lernen …",               # Warum / Nutzen
        "Alle Attention-Modelle.",                                   # Wofür wichtig
        "Faustregel: head_dim zwischen 32 und 128."))                # Tipp
```

Werte werden **beim Setzen geprüft** (Typ, Bereich, Auswahl); Tippfehler werden erkannt
(`num_haeds` → *„Meinten Sie 'num_heads'?“*); Abhängigkeiten zwischen Werten prüft `validate()`.

### 2. Pfade

Alles ist per Punkt-Pfad erreichbar – auch Listen, Komponenten-Argumente und dicts:

| Pfad | Bedeutung |
|---|---|
| `model.d_model` | Parameter des Modells |
| `model.block.mixer.num_heads` | Argument der gewählten Komponente (Kurzform für `….args.num_heads`) |
| `model.block.mixer.name` | Komponente wechseln (kompatible Argumente bleiben) |
| `model.stages.2.channels` | Element einer Liste |
| `training.paradigm.rewards.1.weight` | Gewicht des zweiten Belohnungsterms |
| `model.layer_overrides.3.mixer.window_size` | schichtweise Abweichung |

### 3. Registries & Komponenten

Austauschbare Teile stehen in **Registries** und werden in Konfigurationen nur **per Name** referenziert –
so bleibt jede KI als JSON/YAML speicherbar:

```json
"optimizer": {"name": "adamw", "args": {"lr": 0.0003, "weight_decay": 0.1}}
```

### 4. Schichtweise Abweichungen (Overrides)

```python
"layer_overrides": {
    "*":       {"mixer": "ssm"},                    # alle Schichten Mamba-artig …
    "3::4":    {"mixer": "attention"},              # … jede vierte Attention (Hybrid)
    "1::2":    {"ffn": {"name": "moe"}},            # MoE in jeder zweiten Schicht
    "-1":      {"mixer.num_heads": 16},             # letzte Schicht mehr Köpfe
}
```

Selektoren: `5`, `-1`, `2-5` (inklusive), `2:6`, `::2`, `*`/`alle`, `gerade`, `ungerade`, `erste`, `letzte`, `mitte`,
kombinierbar mit Kommas. Spätere Regeln überschreiben frühere. Gibt es für MLP, CNN (pro Block), RNN, Transformer
(Decoder und Encoder getrennt) und GNN.

**Übernahme** – eine Schicht übernimmt die fertige Konfiguration einer anderen, ganz oder teilweise:

```python
"layer_overrides": {
    "0":  {"mixer.num_heads": 16, "ffn": "moe"},
    "-1": {"@von": 0},                                   # alles von Schicht 0
    "5":  {"@von": 0, "@nur": ["mixer.num_heads"]},      # nur diese eine Einstellung
    "6":  {"@von": 0, "@ausser": ["ffn"], "dropout": 0.2},  # alles außer ffn, danach eigene Änderung
    "7":  {"@von": "basis"},                             # zurück auf den Basis-Block
}
```

Ketten sind erlaubt, Kreise werden erkannt (`@from`/`@only`/`@except` funktionieren ebenso).

### 5. Tensor-Chirurgie

Regeln wählen Parameter per Regex, Modultyp, Art und Dimension – optional nur einen **Ausschnitt** des Tensors –
und wenden eine Operation an:

```python
p.set("tensor_rules", [
    {"target": "embed", "op": "freeze"},
    {"target": r"blocks\.0\..*q_proj\.weight", "index": "0:8, :", "op": "freeze"},   # nur 8 Zeilen einfrieren
    {"module_type": "Linear", "param_kind": "weight", "op": {"name": "prune", "args": {"amount": 0.5}}},
    {"target": "lm_head", "op": {"name": "lr_scale", "args": {"factor": 0.1}}},
    {"target": "o_proj", "op": {"name": "constraint", "args": {"kind": "max_norm", "value": 2.0}}},
])
```

Operationen (22): `freeze`, `unfreeze`, `init`, `scale`, `add`, `set`, `noise`, `clamp`, `prune` (Maske bleibt im
Training), `low_rank`, `orthogonalize`, `load` (Werte aus Datei), `copy` (Werte eines anderen Tensors übernehmen –
ersetzen, addieren, mitteln, interpolieren, auch ausschnittsweise), `custom`, `tie` (Gewichte teilen), `lr_scale`,
`weight_decay`, `grad_scale`, `grad_clip`, `grad_noise`, `grad_mask`, `constraint` (max_norm, unit_norm,
non_negative, clip, orthogonal – nach jedem Schritt erzwungen).

**Geltungsbereich** jeder Regel (auch bei Modul-Regeln):

| Einstellung | Wirkung |
|---|---|
| `layers` | nur innerhalb dieser Blöcke (`"0"`, `"-2:"`, `"::2"`, `"gerade"` …) |
| `match` | `search` (Regex irgendwo), `full` (ganzer Name), `exact` (wörtlich – genau dieser eine) |
| `scope` + `pick` | `alle` · `nur_auswahl` (Positionen `pick` der Trefferliste) · `nur_einer` (genau ein Treffer, sonst Fehler) · `alle_ausser` |

```python
{"target": "o_proj.weight", "scope": "nur_einer", "pick": "-1", "op": {"name": "set", "value": 0.0}}
{"target": "k_proj.weight", "scope": "alle_ausser", "pick": "0", "op": {"name": "scale", "factor": 0.5}}
{"target": "v_proj.weight", "layers": "3", "index": "0:4, :",
 "op": {"name": "copy", "source": "blocks.1.mixer.attn.v_proj.weight", "source_index": "4:8, :"}}
```

### 5b. Modul-Chirurgie – jede Schicht jeder Art

`module_rules` greifen nach dem Bau in **ganze Module** ein (Auswahl per Name, `module_type`, `layers`, `scope`):

```python
p.set("module_rules", [
    {"module_type": "Dropout", "op": {"name": "set_attr", "attribute": "p", "value": 0.0}},     # alle Dropouts aus
    {"target": "blocks.3", "match": "exact", "op": {"name": "copy_from", "source": "blocks.0"}},  # Schicht übernehmen
    {"target": "ffn", "layers": "-1", "op": {"name": "lora", "rank": 16}},                      # LoRA nur hier
    {"module_type": "RMSNorm", "scope": "nur_einer", "pick": "0",
     "op": {"name": "replace_norm", "norm": "layernorm"}},                                       # genau eine Norm tauschen
    {"target": "q_proj", "layers": "0", "op": {"name": "quantize", "format": "int8"}},
    {"target": "mixer", "layers": "::2", "op": {"name": "scale_output", "factor": 0.5}},        # Ausgabe-Hook
    {"target": r"blocks\.\d+$", "match": "full", "op": "record"},                                # Aktivierungsstatistik
])
```

Operationen (20): `freeze`, `unfreeze`, `replace_activation`, `replace_norm`, `identity` (entfernen), `set_attr`,
`dtype` (Präzision pro Schicht), `quantize`, `lora`, `spectral_norm`, `weight_norm`, `checkpoint`, `init`, `reset`,
`copy_from`, `scale_output`, `output_noise`, `clamp_output`, `stop_gradient`, `record`.

### 6. Handling: Platzhalter, Suche, Rückgängig, Vergleich

```python
p.find("dropout")                          # alle Pfade namens dropout ("model.**.num_heads", "model.block.*.dropout" …)
p.set_all("dropout", 0.1, scope="alle_ausser", pick="-1")   # für alle / bestimmte / einen – alles oder nichts
p.set("model.**.num_heads", 8)             # Platzhalter direkt in set
p.search("kv cache")                       # Volltext über Pfade, Tooltips und Komponenten
p.undo(); p.redo()                         # Verlauf (set, set_all, update, Profile, Ratgeber)
p.diff("sprachmodell_mamba")               # Unterschiede zu Vorlage/Datei/Projekt
```

### 7. Projekt = eine Datei

Eine komplette KI (Modell, Daten, Training, Speicher, Tensor-Regeln) ist **eine** `ProjectConfig`.
`save(..., only_changed=True)` schreibt nur Abweichungen vom Standard – kompakt und gut lesbar
(siehe [`examples/konfiguration_beispiel.json`](examples/konfiguration_beispiel.json)).
Mit `auto_fit_dims` (Standard: an) passen sich Ein-/Ausgabegrößen, Vokabular, Beobachtungs- und Aktionsräume
automatisch an Daten bzw. RL-Umgebung an – jede Anpassung wird gemeldet.

---

## Was alles drin ist

### Modelle (`model`)

| Name | Beschreibung |
|---|---|
| `mlp` | MLP: Breite je Schicht, Reihenfolge der Operationen, Residual/Projektion, Weight-/Spectral-Norm |
| `cnn` | 1D/2D/3D; Stufen mit Blocktypen plain, residual, bottleneck, inverted_residual (MobileNet), convnext, depthwise_separable; SE, Stochastic Depth, Klassifikations- oder dichter Kopf |
| `rnn` | LSTM/GRU/RNN/minGRU – **jede Schicht eigene Zellart/Breite/Richtung**, Pooling inkl. Attention |
| `transformer` | decoder_only / encoder_only / encoder_decoder; Eingaben Tokens, kontinuierlich oder Bild-Patches (ViT); Generierung mit top-k/top-p/min-p/Wiederholungsstrafe |
| `ssm` | Mamba-artiges selektives State-Space-Modell |
| `hybrid` | SSM- und Attention-Schichten gemischt (Jamba-Prinzip) |
| `autoencoder` | AE, VAE (β, Free Bits), Denoising, Sparse (L1/KL), VQ-VAE; MLP- oder Faltungs-Rückgrat |
| `gan` | Generator/Diskriminator (MLP oder **beliebiges Modell**), bedingt per Klassen |
| `diffusion` | DDPM/DDIM, Fahrpläne linear/cosine/quadratic/sigmoid, ε/x₀/v-Vorhersage, Classifier-Free Guidance |
| `gnn` | GCN, GraphSAGE, GIN, GAT; Knoten- oder Graph-Ebene, Jumping Knowledge, DropEdge |
| `actor_critic` | Policy + Value (diskret/kontinuierlich, Gauß-Policy, Squashing, PPO-Initialisierung) |
| `q_network` | Q-Netz mit Dueling und Noisy Nets |
| `graph` | **Freies Netz**: beliebiger Graph aus 28 Schichtarten, Verzweigungen, Merges, geteilte Gewichte, Wiederholungen, mehrere Ein-/Ausgänge |
| `ensemble` | Mittel, gewichtet (lernbar), Median, Max, Votum, Softmax-Mittel, Stacking; Uneinigkeit als Unsicherheit |
| `pipeline` | Modelle hintereinander mit automatischen Adaptern |
| `fusion` | Multimodal: mehrere Zweige mit eigenen Eingaben, Fusion concat/sum/gated/attention, Modality-Dropout |
| `mixture_of_models` | Router über ganze Modelle (top-k, Lastverteilung) |
| `multitask` | gemeinsamer Rumpf, mehrere Köpfe |

### Transformer-Bausteine

- **Mixer** (`sequence_mixer`, pro Schicht wählbar): attention, ssm (schneller Scan, optional mamba-ssm-Kernel),
  rnn (GRU/LSTM/RNN/**minGRU** mit parallelem Scan), conv, fourier (FNet), none – alle kausalen mit Generierungs-Cache
- **Attention**: Köpfe, GQA/MQA (`num_kv_heads`), Kopfdimension, Kernel `softmax` (SDPA/Flash oder manuell),
  `sigmoid`, `relu`, `cosine`, `linear`; Temperatur, Logit-Softcap, Sliding Window, QK-Norm, Head-Dropout,
  Output-Gate, Biases, Skalierung, **MLA-lite** (`kv_compression_dim`: komprimierter KV-Latent)
- **Positionen**: none (NoPE), sinusoidal, learned, rope (θ, partiell, linear/NTK/**YaRN**-Skalierung), alibi,
  relative_bias (T5), combined
- **FFN**: mlp, glu (SwiGLU/GeGLU/…), moe (Experten = beliebiges FFN, shared experts, Kapazität, Lastverteilungs- und
  z-Loss, Router-Rauschen), **kan** (Kolmogorov-Arnold, FastKAN), none
- **Block**: Norm (9 Arten inkl. RMSNorm, DyT), Position pre/post/sandwich/none, parallele Zweige, LayerScale,
  Stochastic Depth, Residual-/Skip-Skalierung

### Weitere Komponenten

- **Aktivierungen (33)**: relu … gelu, silu, swish (lernbar), mish, snake, sine (SIREN), squared_relu, star_relu, softcap, lernbares Polynom …
- **Initialisierer (20)**: xavier, kaiming, lecun, orthogonal, trunc_normal, small_init, scaled_normal (GPT-2), siren, dirac, sparse …
- **Optimierer (13)**: sgd, adam, adamw, nadam, radam, adamax, adagrad, adadelta, rmsprop, adafactor, lion, muon, **adamw8bit**
- **Zeitpläne (12)**: constant, linear, warmup_cosine, cosine_restarts, polynomial, exponential, step, inverse_sqrt, one_cycle, wsd, cyclic, plateau
- **Verluste (17)**, frei gewichtet kombinierbar: cross_entropy, bce, mse, l1, huber, smooth_l1, log_cosh, quantile, gaussian_nll, poisson_nll, focal, dice, kl_div, cosine, info_nce, hinge, nll
- **Paradigmen (12)**: supervised, causal_lm, masked_lm, seq2seq, autoencoder, contrastive, distillation, gan (5 Verlustarten, GP, R1, TTUR), diffusion (Min-SNR), preference (Reward-Modell), dpo, reinforcement
- **Training**: Gradientenakkumulation, Clipping (Norm/Wert/AGC), EMA, L1/L2/Orthogonalität, Mixup, Eingaberauschen, Gradientenrauschen, Flooding, SAM, Curriculum, schichtweise Lernraten, Weight-Decay-Ausnahmen, NaN-Behandlung, getrennte Optimierer je Gruppe
- **Callbacks (7)**: early_stopping, checkpoint, csv_logger, jsonl_logger, gradient_monitor, time_limit, gradual_unfreeze
- **Datensätze (15)**: synthetische Aufgaben für jede Modellart + `csv`, `text` (eigene Textdatei), `tensor_file`

Die vollständige Liste mit allen Argumenten und Tooltips steht in
[`docs/PARAMETER_REFERENZ.md`](docs/PARAMETER_REFERENZ.md) (automatisch aus dem Code erzeugt).

---

## Vorlagen nach Anwendungsgebiet

| Gebiet | Vorlagen |
|---|---|
| Sprache | `sprachmodell_mini`, `sprachmodell_moe`, `sprachmodell_mamba`, `sprachmodell_hybrid`, `sprachmodell_speichersparend`, `sprachmodell_qlora`, `sprachmodell_ram_streaming`, `text_klassifikation`, `bert_vortraining`, `uebersetzung_seq2seq` |
| Bild | `bildklassifikation_cnn`, `bildklassifikation_vit`, `bildklassifikation_mobil` |
| Zeitreihen | `zeitreihen_lstm`, `zeitreihen_transformer`, `sequenz_gedaechtnis_rnn` |
| Tabellen | `tabellen_klassifikation`, `tabellen_regression` |
| Generativ | `gan_2d`, `diffusion_2d`, `vae_bilder` |
| Graph | `graph_klassifikation` |
| Anomalie | `anomalie_autoencoder` |
| Präferenz | `reward_modell` |
| RL | `rl_cartpole_ppo`, `rl_cartpole_dqn`, `rl_gitterwelt_belohnungen`, `rl_pendel_kontinuierlich` |
| Kombination | `ensemble_tabellen`, `multimodal_fusion`, `mixture_of_models` |
| Forschung | `freies_netz`, `kontrastiv_vortraining`, `forschung_experimentell` |

Alle laufen ohne Downloads auf eingebauten synthetischen Daten. Beispiele mit Standardeinstellungen (CPU):
Bild-CNN 100 %, Text-Klassifikation 100 %, Graph-Klassifikation 100 %, Multimodal-Fusion 99 %, Seq2Seq-Umkehrung 100 %,
Anomalie-AUROC 1.0, PPO/DQN lösen CartPole (200/200).

---

## Speicher & große Modelle

```python
p.set("memory.mixed_precision", "bf16")          # Rechnen in bfloat16
p.set("memory.gradient_checkpointing", True)     # Aktivierungen neu berechnen statt speichern
p.set("memory.checkpoint_layers", "::2")         # … nur jeden zweiten Block
p.set("memory.activation_offload", True)         # Aktivierungen im CPU-RAM
p.set("memory.param_dtype", "bfloat16")          # reines bf16-Training
p.set("memory.quantization", "fake_quant")       # Quantization-Aware Training (4–8 Bit)
p.set("training.gradient_accumulation_steps", 8) # große effektive Batches
p.apply_memory_profile("extrem_sparsam")         # oder alles auf einmal: sparsam / extrem_sparsam / schnell
print(p.memory_report())                         # Gewichte, Gradienten, Optimierer, gemessene Aktivierungen + Empfehlungen
```

### Modellblöcke trainierbar in den RAM auslagern (`offload.mode = "stream"`)

```python
p.set("memory.offload.mode", "stream")   # Gewichte, Gradienten UND Optimierer-Zustände der Blöcke liegen im CPU-RAM
p.set("memory.offload.layers", "0:12")   # optional nur ein Teil der Blöcke
p.set("memory.offload.prefetch", True)   # (experimentell, CUDA) nächsten Block auf eigenem Stream vorladen
```

Pro Block liegt nur während seiner Berechnung eine Kopie auf dem Gerät. Im Backward wird der Block mit identischem
Zufalls- und autocast-Zustand neu gerechnet (wie Checkpointing); die Gradienten wandern in den RAM, der
Optimierer-Schritt dieser Gewichte läuft auf der CPU (ZeRO-Offload-Prinzip). Ergebnis: **bitgenau dasselbe Training**,
auch mit Dropout, BatchNorm, bf16 und Checkpointing. Für reine **Inferenz** gibt es weiterhin `offload.mode = "cpu" | "disk"`
(Layer-Swapping inkl. Memory-Map von der Festplatte, `keep_resident`).

### Quantisierung ohne torch.ao, LoRA/QLoRA, 8-Bit-Optimierer

```python
p.set("memory.quantization", "nf4")      # eigene Gewichts-Quantisierung: int8 | int4 | nf4 (alle Geräte)
p.set("memory.quant_group_size", 64)     # Skala pro 64 Gewichte (None = pro Kanal)
p.set("memory.quant_skip", "lm_head")    # Ausnahmen (Regex)
p.set("memory.lora.enabled", True)       # LoRA-Adapter (rank, alpha, targets, layers, DoRA, PiSSA-Init, train_also)
p.set("training.optimizer", "adamw8bit") # Adam-Zustände in 8 Bit (~¼ Optimierer-Speicher)
```

| Format | Speicher/Gewicht | Einsatz |
|---|---|---|
| `int8` | 1 Byte | Inferenz, eingefrorene Basis |
| `int4` | ½ Byte | aggressive Kompression (kleine Gruppen) |
| `nf4` | ½ Byte | QLoRA – beste 4-Bit-Qualität für normalverteilte Gewichte |
| `torchao_int8/int4` | – | optimierte Kernel über das Paket `torchao` (optional) |
| `fake_quant` | – | Quantization-Aware Training (voll trainierbar) |
| `dynamic_int8` | – | **veraltet** (torch.ao), nur noch aus Kompatibilitätsgründen |

Gespeicherte quantisierte/LoRA-Projekte werden beim Laden automatisch in dieselbe Struktur gebracht;
`merge_lora` bzw. der Export rechnen Adapter in die Gewichte ein.

Weitere Schalter: `device`, `compile`, `channels_last`, `matmul_precision`, `cpu_threads`, `max_memory_fraction`,
`empty_cache_every_n_steps`. `memory_report()` berücksichtigt Streaming, 8-Bit-Optimierer und schlägt passende
Einstellungen vor.

---

## Mamba, Generierung & KV-Cache

**Mamba-Training**: Der Zustands-Scan `h_t = exp(Δ·A)·h_{t-1} + Δ·B·x_t` läuft als eigene Autograd-Funktion
(`components/scan.py`): Forward und Backward ohne Aufzeichnung jedes Zeitschritts, im Backward werden die großen
`[B, T, E, N]`-Tensoren neu berechnet (Prinzip des offiziellen CUDA-Kernels, hier in reinem PyTorch).

| `model.block.mixer.scan_mode` | Algorithmus |
|---|---|
| `auto` | CPU: fusionierte In-place-Schleife (O(T)); GPU: chunked |
| `chunked` / `parallel` | Hillis-Steele-Scan (log₂T Runden) in Blöcken bzw. über die ganze Folge |
| `sequential` | Referenz |
| `mamba_ssm` | offizieller Kernel, falls `mamba-ssm` installiert (sonst automatisch `auto`) |

`scan_backward = fused | scan | autograd` wählt den Speicher/Zeit-Kompromiss. Gemessen (CPU, Batch 32, T 128,
4 Blöcke): Forward+Backward 16,2 s → 1,3 s, gespeicherte Aktivierungen 5,2 GiB → 0,23 GiB.

**Generierung mit Cache** (`model.generate(..., use_cache=True)` ist Standard): Prompt einmal verarbeiten, danach pro
Token nur ein Schritt. Attention speichert Keys/Values, SSM Faltungsfenster + Zustand, RNN den versteckten Zustand,
Faltung die letzten Eingaben – Ergebnis identisch zur Neuberechnung (getestet für alle Mixer und Positionskodierungen).

```python
ids = model.generate(prompt, max_new_tokens=200, cache={"dtype": "int8"})          # ¼ Cache-Speicher
ids = model.generate(prompt, cache={"max_length": 1024, "sink_tokens": 4})         # Fenster + Attention-Sinks
cache = model.new_cache(); model(prompt, cache=cache); model(next_token, cache=cache)  # eigene Schleifen
```

`kv_compression_dim` (MLA-lite) und `num_kv_heads` (GQA) verkleinern den Cache schon beim Modellbau.

---

## Aufbau, Ratgeber & Export

```python
print(p.architecture("text", max_depth=3))        # alle Module mit Pfad, Typ, Parametern, Ausgabeform, Markierungen
p.architecture("markdown", path="aufbau.md")       # inkl. Mermaid-Datenflussdiagramm und Tensor-Tabelle
data = json.loads(p.architecture("json"))          # maschinenlesbar zum Integrieren
```

Die **Schichttabelle** zeigt für jeden Block Mixer/FFN/Köpfe, Parameter, **die Abweichungen von der Basis** und den
Pfad zum Anpassen (`model.layer_overrides.<i>.…` bzw. Modul-Regel). Markierungen: eingefroren, LoRA, quantisiert,
Checkpointing, RAM-Streaming, Hooks, Cache.

**Ratgeber** (`p.advise()`, CLI `optimieren`, GUI „Optimieren“): analysiert Konfiguration, Modell und
Speicherschätzung – z. B. 8-Bit-AdamW, Checkpointing, bf16, Akkumulation, Streaming, QLoRA, GQA, Weight Tying,
Kopfgrößen, KV-Cache in int8, Sliding Window, schneller SSM-Scan, Clipping, Lernrate, Warmup, torch.compile, TF32.
Jeder Vorschlag hat Begründung, geschätzten Effekt und direkt anwendbare Änderungen (`apply_advice`, rückgängig mit `undo`).

**Export**: `p.export("modell", "export")` (torch.export `.pt2`), `"torchscript"`, `"onnx"` (mit Paket `onnx`),
`"state_dict"` – LoRA wird dabei eingerechnet.

---

## Reinforcement Learning & Belohnungen

Die Gesamtbelohnung ist eine **gewichtete Summe frei kombinierbarer Terme** – jeder mit Art
(extrinsisch / intrinsisch / formend / strafe / bonus), Gewicht, Begrenzung, Aktivitätsfenster und Gewichtsverlauf:

```python
"rewards": [
    {"name": "env"},                                                    # Aufgabe der Umgebung
    {"name": "constant",  "args": {"value": -0.01}},                    # Zeitstrafe
    {"name": "potential", "args": {"key": "distance", "weight": 0.2}},  # Formung γΦ(s')−Φ(s)
    {"name": "event",     "args": {"key": "hit_wall", "value": -0.05}}, # Ereignis-Strafe
    {"name": "count",     "args": {"weight": 0.05, "anneal": "linear_decay"}},  # Neuheitsbonus, klingt ab
    {"name": "rnd",       "args": {"weight": 0.1}},                     # Neugier (Random Network Distillation)
]
```

Terme: `env`, `constant`, `action_penalty`, `action_change`, `info` (beliebige Umgebungsgröße), `potential`,
`event`, `terminal`, `rnd`, `count` – plus eigene. Belohnungs-/Beobachtungsnormalisierung und Clipping sind einstellbar.
Das Protokoll zeigt den **Ertrag je Term** und deckt so „Reward Hacking“ auf.
Algorithmen: PPO, A2C, REINFORCE (Baselines), DQN (Double, n-Schritt, priorisiertes Replay, weiche/harte Ziel-Updates).
Umgebungen: gridworld, cartpole, pendulum, bandit, gymnasium (alle Gymnasium-Umgebungen).

---

## Eigene Komponenten

Einmal registrieren – danach überall per Name nutzbar (Python, JSON, CLI, GUI, mit Tooltips):

```python
from torch import nn
from ki_baukasten import BaseConfig, Param, T, registry

class WelleConfig(BaseConfig):
    frequenz = Param(2.0, min=0.0, tooltip=T("Frequenz.", "Steuert die Oszillation.", "Periodische Daten."))

class Welle(nn.Module):
    def __init__(self, f): super().__init__(); self.f = f
    def forward(self, x): return x + (self.f * x).sin() / self.f

registry("activation").register("welle", lambda a: Welle(a.frequenz), config=WelleConfig,
                                tooltip=T("Eigene Wellen-Aktivierung.", "…", "…"))

p.set("model.layer.activation", {"name": "welle", "args": {"frequenz": 3.0}})
```

Gleiches gilt für Modelle, Schichten, Mixer, Verluste, Optimierer, Zeitpläne, Paradigmen, Callbacks,
Umgebungen, Belohnungen, Tensor-Operationen, Datensätze und Vorlagen. Ohne Schema geht es auch:
`register(..., free_args=True)` reicht beliebige Argumente durch. Für Ad-hoc-Experimente hat jedes Modell
außerdem ein freies `extras`-dict.

---

## Projektstruktur

```
ki_baukasten/
├── core/          Param, BaseConfig, Tooltip/Level, Registry, ComponentSpec, Selektoren, Fehler
├── components/    activations, normalization, initializers, positional, attention, feedforward, mixers, layers,
│                  scan (schnelle lineare Rekurrenzen), cache (KV-/Zustands-Cache)
├── models/        base, mlp, cnn, rnn, transformer (+ssm/hybrid), autoencoder, generative (gan/diffusion),
│                  gnn, rl_nets, graph (freies Netz), composite (ensemble/pipeline/fusion/mixture/multitask)
├── training/      config, trainer, paradigms, optimizers, schedulers, losses, callbacks, data_utils
│   └── rl/        environments, rewards, algorithms (ppo/a2c/reinforce/dqn)
├── memory/        config, manager, streaming (RAM-Training), offload (Swapping), quant (int8/int4/nf4),
│                  lora (LoRA/DoRA/QLoRA), estimator
├── tensor/        surgery (Tensor-Chirurgie), modules (Modul-Chirurgie), selection (Geltungsbereiche)
├── analysis/      architecture (Aufbau-Darstellung), advisor (Ratgeber), export
├── data/          datasets (synthetisch + CSV/Text/Tensor-Dateien)
├── presets/       Vorlagen je Anwendungsgebiet, Speicherprofile
├── ui/            cli, gui (Tkinter), docs (Referenz-Export)
└── project.py     Project-Fassade
examples/          14 ausführlich kommentierte Beispiele + Beispiel-Konfigurationsdatei
tests/             ~365 Tests (pytest)
docs/              ARCHITEKTUR.md (innerer Aufbau, Eingriffspunkte) + automatisch erzeugte Parameter-Referenz
```

| Beispiel | Inhalt |
|---|---|
| `01_schnellstart.py` … `09_generativ.py` | Grundlagen, eigene Architekturen, freies Netz, Kombinationen, Tensor-Chirurgie, Speicher, RL, eigene Komponenten, generative Modelle |
| `10_mamba_schnell_und_cache.py` | schneller SSM-Scan (Zeit/Speicher gemessen), Generierung mit Cache, Cache-Speicher |
| `11_qlora_feintuning.py` | nf4 + LoRA + 8-Bit-AdamW, Speichern/Laden, Export |
| `12_ram_streaming.py` | trainierbares Auslagern in den RAM – bitgenau wie ohne |
| `13_regeln_geltungsbereich.py` | Übernahme zwischen Schichten, Modul-/Tensor-Regeln mit Geltungsbereich, Platzhalter, Rückgängig |
| `14_ratgeber_aufbau_export.py` | Ratgeber übernehmen, Unterschiede, Aufbau (Text/Markdown/JSON), Export |

Konvention: **Code-Bezeichner englisch** (wie in PyTorch und Fachliteratur), **alle Erklärungen,
Kommentare, Tooltips und Ausgaben deutsch**.

---

## Tests

```bash
pip install pytest
pytest            # ~365 Tests, ca. 30–60 s auf CPU
```

Geprüft werden u. a.: jede Komponente jeder Registry, jedes Modell vorwärts/rückwärts, Kausalität aller Mixer und
Positionskodierungen, SDPA = manuelle Attention, jedes Paradigma, jeder Optimierer und Zeitplan, alle Speicheroptionen
(Checkpointing gradientengleich, Offloading bitgenau), Tensor-Chirurgie (Teil-Einfrieren hält exakt, Pruning-Maske bleibt),
alle Vorlagen (Bauen + Konfigurations-Rundlauf), CLI und Doku-Export – und dass PPO CartPole löst. Neu:
Scan-Gradienten gegen Referenz (inkl. gradcheck), Cache = Neuberechnung für alle Mixer/Positionen, RAM-Streaming
gradienten- und trainingsgleich, Quantisierungsgenauigkeit, LoRA-Start/-Merge, QLoRA speichern/laden, 8-Bit-AdamW-
Konvergenz, Übernahme/Geltungsbereiche/Modul-Regeln, Ratgeber/Aufbau/Export/Rückgängig.

---

## Ehrliche Grenzen

- Der **SSM-Scan** ist reines PyTorch: auf der GPU schnell, aber ohne Fusion in einen einzigen Kernel wie beim
  offiziellen `mamba-ssm` (der mit `scan_mode = "mamba_ssm"` genutzt wird, wenn installiert).
- **RAM-Streaming** überträgt die Block-Gewichte zweimal pro Schritt und rechnet jeden Block neu (typisch 1.3–3×
  langsamer); der CUDA-Prefetch ist experimentell und in dieser Testumgebung (nur CPU) nicht geprüft.
- **Eigene Quantisierung** entpackt die Gewichte pro Schicht vor der Matrixmultiplikation (spart Speicher, keine
  int8-Rechenkernel); für maximale Geschwindigkeit `torchao_int8/int4`. `dynamic_int8` (torch.ao) ist veraltet.
- **MLA-lite** ist vereinfacht (RoPE auf rekonstruierte Keys, keine entkoppelten RoPE-Köpfe wie bei DeepSeek).
- Keine verteilte Mehr-GPU-Ausführung (DDP/FSDP) – eine natürliche Erweiterung.
- Die Vorlagen sind auf kleine synthetische Daten abgestimmt; für echte Projekte `data` auf `csv`, `text` oder
  `tensor_file` umstellen (Größen passen sich automatisch an) und Umfang/Lernrate anpassen.
