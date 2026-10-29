# KI-Baukasten – alle Möglichkeiten

Automatisch erzeugt (`python -m ki_baukasten moeglichkeiten -o docs/MOEGLICHKEITEN.md`). 352 Komponenten in 25 Registries – jede überall per Name einsetzbar. Einzelheiten zu jeder Einstellung: [PARAMETER_REFERENZ.md](PARAMETER_REFERENZ.md), innerer Aufbau: [ARCHITEKTUR.md](ARCHITEKTUR.md).

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
- **Modelle** (`model`, 19): `autoencoder`, `cnn`, `ensemble`, `pipeline`, `fusion`, `mixture_of_models`, `multitask`, `extern`, `mlp`, `gan`, `diffusion`, `gnn`, `graph`, `actor_critic`, `q_network`, `rnn`, `transformer`, `ssm`, `hybrid`
- **Callbacks** (`callback`, 7): `early_stopping`, `checkpoint`, `csv_logger`, `jsonl_logger`, `gradient_monitor`, `time_limit`, `gradual_unfreeze`
- **Verlustfunktionen** (`loss`, 17): `cross_entropy`, `bce`, `mse`, `l1`, `huber`, `smooth_l1`, `log_cosh`, `quantile`, `gaussian_nll`, `poisson_nll`, `focal`, `dice`, `kl_div`, `cosine`, `info_nce`, `hinge`, `nll`
- **Optimierer** (`optimizer`, 13): `sgd`, `adam`, `adamw`, `nadam`, `radam`, `adamax`, `adagrad`, `adadelta`, `rmsprop`, `adafactor`, `lion`, `adamw8bit`, `muon`
- **Trainings-Paradigmen** (`paradigm`, 12): `supervised`, `causal_lm`, `masked_lm`, `seq2seq`, `autoencoder`, `contrastive`, `distillation`, `gan`, `diffusion`, `preference`, `dpo`, `reinforcement`
- **Lernraten-Zeitpläne** (`scheduler`, 12): `constant`, `linear`, `warmup_cosine`, `cosine_restarts`, `polynomial`, `exponential`, `step`, `inverse_sqrt`, `one_cycle`, `wsd`, `cyclic`, `plateau`
- **RL-Umgebungen** (`environment`, 5): `gridworld`, `cartpole`, `pendulum`, `bandit`, `gymnasium`
- **Belohnungsterme** (`reward`, 10): `env`, `constant`, `action_penalty`, `action_change`, `info`, `potential`, `event`, `terminal`, `rnd`, `count`
- **RL-Algorithmen** (`rl_algorithm`, 4): `ppo`, `a2c`, `reinforce`, `dqn`
- **Modul-Operationen** (`module_op`, 25): `freeze`, `unfreeze`, `replace_activation`, `replace_norm`, `identity`, `set_attr`, `dtype`, `quantize`, `lora`, `spectral_norm`, `weight_norm`, `checkpoint`, `init`, `reset`, `copy_from`, `scale_output`, `output_noise`, `clamp_output`, `stop_gradient`, `record`, `resize_linear`, `resize_norm`, `resize_attention`, `factorize`, `drop_items`
- **Tensor-Operationen** (`tensor_op`, 22): `freeze`, `unfreeze`, `init`, `scale`, `add`, `set`, `noise`, `clamp`, `prune`, `low_rank`, `orthogonalize`, `load`, `custom`, `copy`, `tie`, `lr_scale`, `weight_decay`, `grad_scale`, `grad_clip`, `grad_noise`, `grad_mask`, `constraint`
- **Tensor-Funktionen (Chirurgie)** (`tensor_function`, 6): `identity`, `abs`, `sign`, `normalize_rows`, `flip`, `shuffle`
- **Kompressions-Schritte** (`compress_step`, 11): `prune_ffn`, `prune_heads`, `drop_layers`, `prune_mlp`, `low_rank`, `sparsify`, `heal`, `finetune`, `quantize`, `dtype`, `merge_lora`
- **Datensätze** (`dataset`, 15): `multimodal`, `classification`, `regression`, `sequence_classification`, `text`, `seq2seq_toy`, `time_series`, `shapes`, `graphs`, `distribution_2d`, `preferences`, `anomaly`, `csv`, `tensor_file`, `none`
- **Vorlagen (Presets)** (`preset`, 34): `sprachmodell_mini`, `sprachmodell_moe`, `sprachmodell_mamba`, `sprachmodell_hybrid`, `sprachmodell_speichersparend`, `sprachmodell_qlora`, `sprachmodell_ram_streaming`, `text_klassifikation`, `bert_vortraining`, `uebersetzung_seq2seq`, `bildklassifikation_cnn`, `bildklassifikation_vit`, `bildklassifikation_mobil`, `vae_bilder`, `zeitreihen_lstm`, `zeitreihen_transformer`, `sequenz_gedaechtnis_rnn`, `tabellen_klassifikation`, `tabellen_regression`, `gan_2d`, `diffusion_2d`, `graph_klassifikation`, `anomalie_autoencoder`, `reward_modell`, `rl_cartpole_ppo`, `rl_cartpole_dqn`, `rl_gitterwelt_belohnungen`, `rl_pendel_kontinuierlich`, `ensemble_tabellen`, `multimodal_fusion`, `mixture_of_models`, `freies_netz`, `kontrastiv_vortraining`, `forschung_experimentell`

## Kommandozeile

| Befehl | Zweck |
|---|---|
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
| `vergleiche` | zwei Vorlagen/Dateien vergleichen |
| `exportieren` | Modell exportieren |
| `importieren` | fremdes Modell importieren |
| `komprimieren` | Modell eindampfen |
| `zielsysteme` | Geräteprofile bzw. Übersicht für ein Modell |
| `reduzieren` | Empfehlung für ein Zielsystem (und anwenden) |
| `schueler` | kleineres Schüler-Modell per Distillation |
| `probelauf` | alle Wege durchs Netz mit Statistiken |
| `moeglichkeiten` | Katalog aller Möglichkeiten |
| `trainieren` | trainieren |
| `generieren` | aus gespeichertem Projekt generieren |
| `doku` | Parameter-Referenz exportieren |
| `gui` | grafische Oberfläche |

Grafische Oberfläche: `python -m ki_baukasten gui` – Einstellungsbaum mit Tooltips, Prüfen, Bauen, Speicher, Trainieren, Aufbau, Optimieren, Probelauf & Gewichte, Reduzieren/Export, Modell importieren, Rückgängig/Wiederholen.
