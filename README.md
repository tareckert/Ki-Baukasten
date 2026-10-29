# KI-Baukasten

Ein Python-Baukasten (PyTorch), mit dem sich **KIs und ganze KI-Systeme bis ins kleinste Detail** zusammenstellen,
kombinieren, trainieren und speichern lassen – mit einer **Erklärung zu jeder einzelnen Einstellung**.

- **899 einzeln dokumentierte Einstellungen** – jede mit Tooltip: *Was ist das? Warum/Nutzen? Wofür wichtig? Tipp*
- **310 austauschbare Komponenten** in 23 Registries (Aktivierungen, Normen, Attention-Kernel, Mixer, FFN/MoE,
  Modelle, Optimierer, Zeitpläne, Verluste, Belohnungen, Tensor-Operationen …) – alle per Name überall einsetzbar
- **18 Modelle**, beliebig kombinierbar und verschachtelbar (Ensemble aus Pipelines aus Transformern …)
- **12 Trainings-Paradigmen** inkl. Reinforcement Learning mit frei kombinierbaren Belohnungen
- **32 Vorlagen** für 11 Anwendungsgebiete – sofort lauffähig ohne Downloads
- **Jede Schicht darf anders sein** (schichtweise Overrides) und **jeder Tensor-Ausschnitt** ist einzeln steuerbar
  (Tensor-Chirurgie)
- **Speicher-Weichen**: Präzision, Gradient-Checkpointing, Aktivierungs-Offload, Layer-Swapping (CPU/Festplatte),
  Quantisierung, Speicherschätzung mit Empfehlungen
- **Drei Oberflächen**: Python-API, Kommandozeile (deutsch), grafischer Editor mit Maus-Tooltips

Es gibt keine „verbotenen“ Einstellungen: auch experimentelle, exotische oder scheinbar unnötige Optionen sind
änderbar. Erfahrungsstufen (*Basis → Fortgeschritten → Experte → Experimentell*) dienen nur der Orientierung.

---

## Inhalt

1. [Installation](#installation)
2. [Schnellstart](#schnellstart) – Python, Kommandozeile, GUI
3. [Grundkonzepte](#grundkonzepte)
4. [Was alles drin ist](#was-alles-drin-ist)
5. [Vorlagen nach Anwendungsgebiet](#vorlagen-nach-anwendungsgebiet)
6. [Speicher & große Modelle](#speicher--große-modelle)
7. [Reinforcement Learning & Belohnungen](#reinforcement-learning--belohnungen)
8. [Eigene Komponenten](#eigene-komponenten)
9. [Projektstruktur](#projektstruktur)
10. [Tests](#tests)
11. [Ehrliche Grenzen](#ehrliche-grenzen)

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

print(p.explain("model.block.norm_position"))       # Tooltip + Typ + Bereich + alle Optionen
print(p.memory_report())                            # Speicherschätzung mit Empfehlungen

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
- **Prüfen / Bauen / Speicher / Trainieren** – Training läuft im Hintergrund mit Live-Protokoll.

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

Operationen (21): `freeze`, `unfreeze`, `init`, `scale`, `add`, `set`, `noise`, `clamp`, `prune` (Maske bleibt im
Training), `low_rank`, `orthogonalize`, `load` (Werte aus Datei), `custom`, `tie` (Gewichte teilen), `lr_scale`,
`weight_decay`, `grad_scale`, `grad_clip`, `grad_noise`, `grad_mask`, `constraint` (max_norm, unit_norm,
non_negative, clip, orthogonal – nach jedem Schritt erzwungen).

### 6. Projekt = eine Datei

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

- **Mixer** (`sequence_mixer`, pro Schicht wählbar): attention, ssm, rnn, conv, fourier (FNet), none
- **Attention**: Köpfe, GQA/MQA (`num_kv_heads`), Kopfdimension, Kernel `softmax` (SDPA/Flash oder manuell),
  `sigmoid`, `relu`, `cosine`, `linear`; Temperatur, Logit-Softcap, Sliding Window, QK-Norm, Head-Dropout,
  Output-Gate, Biases, Skalierung
- **Positionen**: none (NoPE), sinusoidal, learned, rope (θ, partiell, linear/NTK-Skalierung), alibi, relative_bias (T5),
  combined
- **FFN**: mlp, glu (SwiGLU/GeGLU/…), moe (Experten = beliebiges FFN, shared experts, Kapazität, Lastverteilungs- und
  z-Loss, Router-Rauschen), none
- **Block**: Norm (9 Arten inkl. RMSNorm, DyT), Position pre/post/sandwich/none, parallele Zweige, LayerScale,
  Stochastic Depth, Residual-/Skip-Skalierung

### Weitere Komponenten

- **Aktivierungen (33)**: relu … gelu, silu, swish (lernbar), mish, snake, sine (SIREN), squared_relu, star_relu, softcap, lernbares Polynom …
- **Initialisierer (20)**: xavier, kaiming, lecun, orthogonal, trunc_normal, small_init, scaled_normal (GPT-2), siren, dirac, sparse …
- **Optimierer (12)**: sgd, adam, adamw, nadam, radam, adamax, adagrad, adadelta, rmsprop, adafactor, lion, muon
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
| Sprache | `sprachmodell_mini`, `sprachmodell_moe`, `sprachmodell_mamba`, `sprachmodell_hybrid`, `sprachmodell_speichersparend`, `text_klassifikation`, `bert_vortraining`, `uebersetzung_seq2seq` |
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

**Layer-Swapping** für Inferenz von Modellen, die größer als der Gerätespeicher sind:
`memory.offload.mode = "cpu" | "disk"` – Blöcke liegen im RAM bzw. auf der Festplatte und werden nur für ihre eigene
Berechnung geladen (`keep_resident` steuert, wie viele bleiben dürfen). Weitere Schalter: `device`, `compile`,
`channels_last`, `matmul_precision`, `cpu_threads`, `max_memory_fraction`, `empty_cache_every_n_steps`.

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
├── components/    activations, normalization, initializers, positional, attention, feedforward, mixers, layers
├── models/        base, mlp, cnn, rnn, transformer (+ssm/hybrid), autoencoder, generative (gan/diffusion),
│                  gnn, rl_nets, graph (freies Netz), composite (ensemble/pipeline/fusion/mixture/multitask)
├── training/      config, trainer, paradigms, optimizers, schedulers, losses, callbacks, data_utils
│   └── rl/        environments, rewards, algorithms (ppo/a2c/reinforce/dqn)
├── memory/        config, manager (Präzision, Checkpointing, Quantisierung), offload (Swapping), estimator
├── tensor/        surgery (Tensor-Chirurgie)
├── data/          datasets (synthetisch + CSV/Text/Tensor-Dateien)
├── presets/       Vorlagen je Anwendungsgebiet, Speicherprofile
├── ui/            cli, gui (Tkinter), docs (Referenz-Export)
└── project.py     Project-Fassade
examples/          9 ausführlich kommentierte Beispiele + Beispiel-Konfigurationsdatei
tests/             257 Tests (pytest)
docs/              automatisch erzeugte Parameter-Referenz
```

Konvention: **Code-Bezeichner englisch** (wie in PyTorch und Fachliteratur), **alle Erklärungen,
Kommentare, Tooltips und Ausgaben deutsch**.

---

## Tests

```bash
pip install pytest
pytest            # ~260 Tests, ca. 40 s auf CPU
```

Geprüft werden u. a.: jede Komponente jeder Registry, jedes Modell vorwärts/rückwärts, Kausalität aller Mixer und
Positionskodierungen, SDPA = manuelle Attention, jedes Paradigma, jeder Optimierer und Zeitplan, alle Speicheroptionen
(Checkpointing gradientengleich, Offloading bitgenau), Tensor-Chirurgie (Teil-Einfrieren hält exakt, Pruning-Maske bleibt),
alle Vorlagen (Bauen + Konfigurations-Rundlauf), CLI und Doku-Export – und dass PPO CartPole löst.

---

## Ehrliche Grenzen

- **SSM/minGRU** rechnen den Scan als Python-Schleife (korrekt und lesbar, aber deutlich langsamer als die
  CUDA-Kernel des Mamba-Originals).
- **Layer-Swapping** (`offload`) ist für Inferenz bzw. eingefrorene Blöcke gedacht; fürs Training großer Modelle sind
  Gradient-Checkpointing, Aktivierungs-Offload, bf16 und Akkumulation die richtigen Werkzeuge.
- **`dynamic_int8`** nutzt `torch.ao.quantization`, das PyTorch zugunsten von `torchao` ausmustern will.
- Keine verteilte Mehr-GPU-Ausführung (DDP/FSDP) und kein KV-Cache bei der Generierung (jeder Schritt rechnet den
  Kontext neu) – beides wäre eine natürliche Erweiterung.
- Die Vorlagen sind auf kleine synthetische Daten abgestimmt; für echte Projekte `data` auf `csv`, `text` oder
  `tensor_file` umstellen (Größen passen sich automatisch an) und Umfang/Lernrate anpassen.
