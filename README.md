# KI-Baukasten

Ein Python-Baukasten (PyTorch), mit dem sich **KIs und ganze KI-Systeme bis ins kleinste Detail** zusammenstellen,
kombinieren, trainieren und speichern lassen – mit einer **Erklärung zu jeder einzelnen Einstellung**.

- **1.169 einzeln dokumentierte Einstellungen** – jede mit Tooltip: *Was ist das? Warum/Nutzen? Wofür wichtig? Tipp* –
  und jede mit Schalter **An · Aus · Eigen**
- **Übersichtlich in drei Ebenen**: **Ziel-Assistent** (Zweck und Prioritäten angeben → passende Modelle werden
  empfohlen, unpassende abgeraten), **Techniken-Katalog** (92 Verfahren aus Community, Forschung und Industrie in
  13 Kategorien, je mit einem Häkchen an/aus) und **Einstellungen** (jede einzeln, mit Ansichtsfilter)
- **363 austauschbare Komponenten** in 26 Registries (Aktivierungen, Normen, Attention-Kernel, Mixer, FFN/MoE/KAN,
  Modelle, Optimierer, Zeitpläne, Verluste, Belohnungen, Tensor- und Modul-Operationen, Kompressions-Schritte …) – alle
  per Name überall einsetzbar; Gesamtübersicht in [`docs/MOEGLICHKEITEN.md`](docs/MOEGLICHKEITEN.md)
- **19 Modelle** (inkl. **Import fremder Modelle**), beliebig kombinierbar und verschachtelbar
- **13 Trainings-Paradigmen** inkl. Reinforcement Learning mit frei kombinierbaren Belohnungen und gradientenfreien
  **Evolution Strategies**
- **34 Vorlagen** für 11 Anwendungsgebiete – sofort lauffähig ohne Downloads; **23 kommentierte Beispiele**
- **Jede Schicht darf anders sein** (schichtweise Overrides mit **Übernahme** von anderen Schichten), **jedes gebaute
  Modul** (Modul-Chirurgie) und **jeder Tensor-Ausschnitt** (Tensor-Chirurgie) ist einzeln steuerbar – jeweils mit
  Geltungsbereich **für alle / nur bestimmte / nur genau einen / alle außer**
- **Schnelles Mamba**: fusionierter Scan mit eigenem Backward – Training ~12× schneller, ~23× weniger Aktivierungsspeicher
- **KV-Cache & Zustands-Caches** für die Generierung (alle kausalen Mixer; float16/bfloat16/int8, Fenster + Attention-Sinks,
  MLA-lite)
- **Speicher-Weichen**: Präzision, Gradient-Checkpointing, Aktivierungs-Offload, **trainierbares Auslagern von Blöcken
  in den RAM**, Layer-Swapping (CPU/Festplatte), **eigene int8/int4/nf4-Quantisierung** (ohne torch.ao), **LoRA/DoRA/
  QLoRA**, **8-Bit-AdamW**, Speicherschätzung
- **Modelle eindampfen**: Attention-Köpfe, FFN-Neuronen und ganze Blöcke entfernen, Low-Rank, Sparsity, int8/int4/nf4/
  4-Bit-Palette, 16-Bit – mit **Heilen per Distillation** und Messung (Größe, MACs, Latenz, Qualität) nach jedem Schritt
- **Lehrer-Schüler & Weiterlernen**: automatisch verkleinerte Schüler-Modelle, Feature-Distillation, Weiterlernen ohne
  Vergessen (LwF, EWC, Replay)
- **Zielsysteme**: nach dem Training zeigt der Baukasten, ob/wie das Modell auf Rechenzentrums-GPU, Gaming-GPU, Apple,
  CPU, Smartphone, Raspberry Pi, Mikrocontroller oder Browser passt – inkl. Rezept, `reduce_for(ziel)` und Export
- **Fremde Modelle**: Import (torch.export, TorchScript, state_dict mit Schlüssel-Abbildung, safetensors, Hugging Face),
  ändern, nachtrainieren, komprimieren und neu exportieren
- **Probelauf & Einblick**: alle Wege durchs Netz mit Aktivierungen, Gradienten, Zeiten und Warnungen; Gewichts-
  Histogramme/-Heatmaps, Attention-Karten, Testlauf mit Konfusionsmatrix – auch in der GUI
- **Ratgeber & Aufbau**: Optimierungs-/Speichervorschläge zum direkten Übernehmen, komplette Aufbau-Darstellung
  (Text, Markdown, Mermaid, JSON), Export (torch.export mit dynamischen Formen, TorchScript, ONNX)
- **Notfall-Ausführung**: passt ein Modell trotz Verkleinern nicht, wird stückweise gerechnet – Schicht für Schicht
  von der SSD, eingefrorene Basis auf der SSD mit Adaptern im RAM, Teile auf der CPU, LISA, Batches automatisch teilen,
  lange Prompts in Stücken – jeweils mit Schätzung von Speicher und Tempo
- **Cluster**: verteiltes Training mit DDP, ZeRO-1 und FSDP über `torchrun` (`python -m ki_baukasten verteilt …`)
- **Schwarm & Evolution**: Hyperparameter-Suche per Zufall, Gitter, genetischem Algorithmus, Partikelschwarm und
  Population Based Training; föderiertes Lernen (FedAvg/FedProx), Model Soups, Kreuzvalidierung, LR-Finder – als
  schaltbarer Ablauf (`Project.run()`)
- **Logger**: Umfang von „still“ bis „debug“, Ziele Konsole/Datei (Rotation)/JSONL/CSV/Python-`logging`/TensorBoard/
  W&B/MLflow – je Ziel eigene Stufe – und eigene Logger
- **Drei Oberflächen**: Python-API, Kommandozeile (deutsch), grafischer Editor mit Menüs, Assistent, Techniken-Fenster,
  Maus-Tooltips und Rückgängig/Wiederholen

Es gibt keine „verbotenen“ Einstellungen: auch experimentelle, exotische oder scheinbar unnötige Optionen sind
änderbar. Erfahrungsstufen (*Basis → Fortgeschritten → Experte → Experimentell*) dienen nur der Orientierung.

---

## Inhalt

1. [Installation](#installation)
2. [Schnellstart](#schnellstart) – Python, Kommandozeile, GUI
3. [Ziele, Techniken, Schalter – die drei Ebenen](#ziele-techniken-schalter--die-drei-ebenen)
4. [Grundkonzepte](#grundkonzepte)
5. [Was alles drin ist](#was-alles-drin-ist)
6. [Vorlagen nach Anwendungsgebiet](#vorlagen-nach-anwendungsgebiet)
7. [Speicher & große Modelle](#speicher--große-modelle) – Streaming, Quantisierung, LoRA, 8-Bit-Optimierer
8. [Notfall-Ausführung](#notfall-ausführung-wenn-es-trotz-verkleinern-nicht-passt)
9. [Mamba, Generierung & KV-Cache](#mamba-generierung--kv-cache)
10. [Modelle eindampfen, Lehrer-Schüler & Zielsysteme](#modelle-eindampfen-lehrer-schüler--zielsysteme)
11. [Ablauf: Suche, Schwarm, Evolution, föderiert](#ablauf-suche-schwarm-evolution-föderiert)
12. [Logger & Cluster](#logger--cluster)
13. [Fremde Modelle importieren & Probelauf](#fremde-modelle-importieren--probelauf)
14. [Aufbau, Ratgeber & Export](#aufbau-ratgeber--export)
15. [Reinforcement Learning & Belohnungen](#reinforcement-learning--belohnungen)
16. [Eigene Komponenten](#eigene-komponenten)
17. [Projektstruktur](#projektstruktur)
18. [Tests](#tests)
19. [Ehrliche Grenzen](#ehrliche-grenzen)

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

# Variante A: nach Zielen – der Assistent wählt die Vorlage und schaltet passende Techniken ein
p = Project.from_goals(description="Spam-Filter direkt auf dem Handy", task="text_klassifizieren",
                       priorities=["geschwindigkeit", "datenschutz"], data_size="klein")
print(p.goal_report)                                # empfohlen / abgeraten (mit Begründung), Techniken, Hinweise

# Variante B: direkt eine Vorlage
p = Project.from_preset("sprachmodell_mini")        # Vorlage: kleines GPT-artiges Sprachmodell
p.set("model.n_layers", 6)                          # jede Einstellung per Pfad …
p.set("model.block.mixer.num_heads", 8)             # … auch tief verschachtelt
p.set("model.layer_overrides", {"-1": {"ffn": "moe"}})   # nur die letzte Schicht als Mixture-of-Experts

p.set_all("dropout", 0.1)                           # eine Einstellung überall setzen (Platzhalter: "model.**.dropout")
p.set("**.dropout", "aus")                          # … oder per Schalter: an · aus · eigen · standard
p.enable("qlora", "spike_schutz", "ema")            # Techniken aus dem Katalog an- und abwählen

print(p.explain("model.block.norm_position"))       # Tooltip + Typ + Bereich + alle Optionen
print(p.search("speicher sparen"))                  # Volltextsuche über Pfade, Tooltips und Komponenten
print(p.memory_report())                            # Speicherschätzung mit Empfehlungen
for tipp in p.advise():                             # Ratgeber: Speicher, Tempo, Stabilität, Struktur
    print(tipp)
p.apply_advice(category="speicher")                 # Vorschläge übernehmen … und bei Bedarf p.undo()
print(p.architecture("text"))                       # komplette Auflistung: Module, Formen, Schichten, Pfade

p.train()                                           # bauen + trainieren; danach Übersicht für Zielsysteme
print(p.generate("Der Baukasten", max_new_tokens=80))
p.save("ausgaben/mein_llm")                         # projekt.json + modell.pt + verlauf.json

klein = p.reduce_for("smartphone", export="ausgaben/handy/modell")   # Empfehlung anwenden + exportieren
schueler = p.distill(width=0.5, depth=0.5)          # Lehrer-Schüler: kleineres Modell lernt vom großen
```

### Kommandozeile

```bash
python -m ki_baukasten ziele --text "Bilder auf dem Handy in Echtzeit erkennen" --erstellen neu.json  # Assistent
python -m ki_baukasten techniken sprachmodell_mini --an qlora --an spike_schutz --speichern llm.json  # Katalog
python -m ki_baukasten schalter sprachmodell_mini --aus dropout --an training.ema       # An/Aus/Eigen je Einstellung
python -m ki_baukasten notfall sprachmodell_mini --budget-mb 8 --training --anwenden auto  # wenn es nicht passt
python -m ki_baukasten automl tabellen_klassifikation --methode pso --versuche 12       # Schwarm-/Evolutions-Suche
python -m ki_baukasten ablauf meine_ki.json --setze workflow.cross_validation=5         # ganzer Ablauf
python -m ki_baukasten verteilt meine_ki.json --prozesse 4 --strategie fsdp            # Cluster (torchrun)
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
python -m ki_baukasten zielsysteme ausgaben/sprachmodell_mini                    # passt es auf welche Geräte?
python -m ki_baukasten reduzieren ausgaben/sprachmodell_mini --ziel raspberry_pi --anwenden --export pi/modell
python -m ki_baukasten komprimieren ausgaben/sprachmodell_mini --schritte '["prune_ffn","heal","quantize"]'
python -m ki_baukasten schueler ausgaben/sprachmodell_mini --breite 0.5 --tiefe 0.5 --speichern ausgaben/klein
python -m ki_baukasten importieren fremd.pt2 --form 1,32 --speichern ausgaben/fremd   # fremdes Modell → Projekt
python -m ki_baukasten probelauf sprachmodell_mini --test                         # alle Wege + Testlauf
python -m ki_baukasten moeglichkeiten -o docs/MOEGLICHKEITEN.md                   # Katalog aller Möglichkeiten
python -m ki_baukasten generieren ausgaben/sprachmodell_mini --prompt "Die " --laenge 120
python -m ki_baukasten doku -o docs/PARAMETER_REFERENZ.md  # komplette Referenz aus den Tooltips
python -m ki_baukasten gui sprachmodell_mini
```

### Grafische Oberfläche

`python -m ki_baukasten gui [vorlage|datei]` öffnet einen Editor mit

- **Menü** Datei · Assistent · Einstellungen · Techniken · Ausführen · Analyse · Einsatz · Hilfe und einer schlanken
  Werkzeugleiste für das Häufigste,
- **✚ Neue KI nach Zielen…**: Aufgabe, Beschreibung, Prioritäten (Häkchen), Datenmenge, Zielgerät → Empfehlungen und
  Abratungen mit Begründung → neues Projekt; später „Ziele prüfen / ändern“,
- **Techniken**: Fenster mit Reitern je Kategorie, Häkchen zum An-/Abwählen, Status, Suche, Tooltips (wer nutzt es),
- **An · Aus · Eigen** neben jeder Einstellung und **Ansichtsfilter** (alle, nur eingeschaltete, nur ausgeschaltete,
  nur eigene Werte) – so bleibt es bei über tausend Einstellungen übersichtlich,
- **Baum** aller Einstellungen (Modell, Daten, Training, Speicher, Tensor-Regeln, Ziele, Ablauf),
- **Formular** des gewählten Knotens – **Maus über einen Namen zeigt den Tooltip**, `?` zeigt alle Details und Optionen,
- **Stufenfilter** (Basis … Experimentell), **Suche** über alle Einstellungen,
- Komponentenwechsel per Auswahlliste, Listen (Verluste, Belohnungen, Stufen, Regeln) hinzufügen/entfernen,
- **Prüfen / Bauen / Speicher / Trainieren** – Training läuft im Hintergrund mit Live-Protokoll,
- **Aufbau** (komplette Auflistung), **Optimieren** (Ratgeber mit Übernahme-Dialog), **↶ Rückgängig / ↷ Wiederholen**,
- **Probelauf & Gewichte**: Fenster mit allen Wegen durchs Netz (Ausgabeform, Statistik, Gradient, Zeit, ⚠-Warnungen),
  Histogramm + Heatmap jeder Modulausgabe und jedes Gewichts, Attention-Karten je Kopf, Testlauf mit Konfusionsmatrix,
- **Reduzieren / Export**: Zielsystem wählen → Empfehlung lesen → anwenden → exportieren; **Notfall-Ausführung…**
  (Optionen mit Schätzung, eine wählen und einschalten); **Modell importieren…**,
- **Ausführen**: Ablauf, automatische Suche, föderiert, Model Soup, Kreuzvalidierung, LR-Finder – im Hintergrund.

---

## Ziele, Techniken, Schalter – die drei Ebenen

Trotz über tausend Einstellungen bleibt es übersichtlich, weil man auf drei Ebenen arbeiten kann:

| Ebene | Wofür | Python | CLI | GUI |
|---|---|---|---|---|
| **1. Ziele** | Zweck, wichtige Punkte, Zielgerät → Modell wird empfohlen, Unpassendes abgeraten | `Project.from_goals`, `set_goals`, `goal_check` | `ziele` | ✚ Neue KI nach Zielen |
| **2. Techniken** | 92 bewährte Verfahren in 13 Kategorien mit einem Häkchen | `enable`, `disable`, `techniques` | `techniken` | Techniken |
| **3. Einstellungen** | jede einzelne: An (empfohlen) · Aus · Eigen | `switch`, `set(…, "aus")`, `switches` | `schalter` | Formular + Ansicht |

**Ziel-Assistent** – Beispiel `evaluate_goals({"description": "Pflanzenkrankheiten per Handykamera in Echtzeit",
"data_size": "klein"})`: erkannt werden Aufgabe *Bilder erkennen*, Zielgerät *Smartphone* und die Prioritäten *wenig
Speicher, Energie, Echtzeit*; empfohlen wird das mobile CNN (✓ passt auf das Smartphone, ✓ Echtzeit …), abgeraten vom
Vision Transformer (✗ braucht mehr Daten, ✗ zu groß fürs Smartphone). Vorgeschlagene Techniken (Checkpointing,
Augmentierung, Early Stopping, Kompression nach dem Training …) werden eingeschaltet; aufwendige wie DP-SGD oder
föderiertes Lernen nur vorgeschlagen. Die Ziele bleiben im Projekt (`goals`) und lassen sich jederzeit ändern.

**Techniken-Katalog** – Kategorien: Stabilität (Gradient-Clipping, Warmup, Verlust-Spitzen-Schutz mit Rücksprung, EMA,
QK-Norm, z-Loss …), gegen Auswendiglernen (Dropout, DropPath, Weight-Decay, Label Smoothing, Mixup, CutMix, SAM, SWA,
Early Stopping, Kreuzvalidierung …), Performance (bf16, torch.compile, TF32, GQA …), große Modelle auf kleiner Hardware
(Checkpointing, 8-Bit-AdamW, Adafactor, LoRA/QLoRA, Streaming, SSD-Streaming, device_map, LISA …), Cluster (DDP, ZeRO-1,
FSDP), automatische Suche, Schwarmintelligenz (Partikelschwarm, föderiert, Model Soup), Evolution (genetisch, PBT,
Evolution Strategies), Daten, Datenschutz (DP-SGD), Überwachung, Kompression, Experimentelles (Lookahead, Gradient
Centralization, Grokfast, Lion, Muon …). Jede Technik zeigt, wer sie nutzt, und `disable` stellt den vorherigen
Zustand exakt wieder her. Vollständige Liste: [`docs/MOEGLICHKEITEN.md`](docs/MOEGLICHKEITEN.md#techniken-katalog).

**Schalter je Einstellung** – `an` = empfohlener Wert, `aus` = wirkungslos (Dropout 0, Mixup 0, Komponente `none`, Liste
leer …), `eigen` = eigener Wert (wird beim Ausschalten gemerkt). Gruppen lassen sich als Ganzes schalten
(`p.switch("training.regularization", "aus")`), `p.switch_overview()` zeigt kompakt, was an ist.

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

## Notfall-Ausführung: wenn es trotz Verkleinern nicht passt

```python
print(p.recommend_for("mikrocontroller"))           # enthält bei Bedarf "Passt trotz Verkleinern nicht? …"
for o in p.fallback_options(budget_mb=500, training=True):
    print(o)                                        # ✓/✗, Gerätespeicher, RAM/SSD, Verlangsamung, Einstellungen
p.apply_fallback("auto", budget_mb=500, training=True)   # schnellste passende Option ohne Qualitätsverlust
klein = p.reduce_for("raspberry_pi", fallback="auto")    # verkleinern und – falls nötig – Notfall einschalten
```

| Inferenz | Training |
|---|---|
| Eingaben stückweise (`predict(chunk_size=…)`) | Checkpointing + Mikro-Batches + Akkumulation + bf16 |
| hintere Blöcke auf der CPU (`offload.mode = "split"`) | 8-Bit-AdamW, LISA (nur wenige Blöcke gleichzeitig) |
| Blöcke im RAM bzw. Schicht für Schicht von der SSD | LoRA / QLoRA |
| große Stapel: jede Schicht nur einmal laden (AirLLM-Prinzip) | Blöcke im RAM streamen (`offload.mode = "stream"`) |
| 4-Bit-Gewichte (nf4) | eingefrorene Basis auf der SSD + Adapter im RAM (`offload.storage = "disk"`) |
| auf der CPU rechnen | CPU, oder FSDP über mehrere Geräte |

Dazu: **Speichermangel = kein Abbruch** (`memory.oom_strategy = "batch_teilen"` teilt den Batch und akkumuliert –
gleiches Ergebnis), **Chunked Prefill** für lange Prompts (`generate(..., prefill_chunk=512)`, identisches Ergebnis)
und **Laden per Memory-Map** (`Project.load(ordner, mmap=True)`). Mit `memory.memory_budget_gb` warnt `train()` vorab und
nennt passende Optionen.

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

## Modelle eindampfen, Lehrer-Schüler & Zielsysteme

**Kompression** (`p.compress([...])`, Registry `compress_step`): jeder Schritt wird gemessen.

```python
p.compress([
    {"name": "prune_heads", "args": {"amount": 0.5, "importance": "activation"}},   # Köpfe (bei GQA gruppenweise)
    {"name": "prune_ffn",   "args": {"amount": 0.5}},                               # FFN-Neuronen (MLP/GLU/MoE)
    {"name": "drop_layers", "args": {"count": 1}},                                  # Block mit geringstem Einfluss
    {"name": "heal",        "args": {"steps": 300, "feature_weight": 0.5}},         # Distillation vom Original
    {"name": "quantize",    "args": {"format": "int8"}},                            # int8 | int4 | nf4 | pal4
])
```

Beispiel (Sprachmodell aus der Vorlage, CPU): **Größe ÷ 9,4**, Rechenaufwand ÷ 2,7, Verlust 2,07 → 2,20 (dank Heilen).
Weitere Schritte: `prune_mlp`, `low_rank` (SVD), `sparsify`, `dtype` (16 Bit), `finetune`, `merge_lora`.
Wichtigkeit wahlweise `magnitude`, `activation` (Wanda-Prinzip), `taylor` oder `random`. Alle Formänderungen
werden als `module_rules` protokolliert – `save()`/`load()` stellen das kleinere Modell exakt wieder her.

**Lehrer-Schüler**: `p.distill(width=0.5, depth=0.5, feature_weight=0.5)` baut ein verkleinertes Schüler-Projekt und
trainiert es mit dem Paradigma `distillation` (KL, reverse KL, JS, MSE, Kosinus; Feature-Distillation auch bei
unterschiedlicher Breite; Lehrer auch aus gespeichertem Projekt oder `.pt2`). Beispiel: 13 % der Parameter, 91 % statt
97 % Genauigkeit.

**Weiterlernen ohne Vergessen**: `p.continue_training(neue_daten, method="lwf" | "ewc" | "replay")`.

**Zielsysteme** (`analysis/targets.py`): Nach `train()` erscheint automatisch eine Übersicht (`deployment.targets`):

```
Zielsystem                              jetzt    Latenz  passt  Empfehlung                                      danach
Gaming-/Workstation-GPU (8–24 GB)     3,0 MB    0.0 ms  ja     dtype(float16)                                  1,5 MB ✓
Smartphone (ARM, 6–12 GB RAM, Akku)   3,0 MB    1.1 ms  ja     quantize(int8)                                  0,8 MB ✓
Mikrocontroller (Cortex-M, ≤ 1 MB)    3,0 MB 1115.7 ms  nein   prune_heads → prune_ffn → heal → quantize(int8) 0,6 MB ✗
```

`p.recommend_for("smartphone")` erklärt das Rezept (inkl. Schüler-Modell, falls nötig, und Exportformat),
`p.reduce_for("smartphone", export=...)` führt es aus. Profile: `rechenzentrum_gpu`, `gaming_gpu`, `laptop_gpu`,
`apple_silicon`, `server_cpu`, `laptop_cpu`, `smartphone`, `raspberry_pi`, `mikrocontroller`, `browser`.

---

## Ablauf: Suche, Schwarm, Evolution, föderiert

```python
p.tune("pso", space={"training.optimizer.lr": {"min": 1e-4, "max": 1e-2, "log": True},
                     "model.layer.dropout": [0.0, 0.1, 0.2]}, trials=20)   # random | grid | evolution | pso | pbt
p.federated_train(clients=10, rounds=20, split="nicht_iid", algorithm="fedprox")   # Daten bleiben verteilt
p.model_soup(4)                                     # 4 Läufe vom selben Start, Gewichte gemittelt (greedy)
p.cross_validate(k=5)                               # Mittel ± Streuung
p.find_lr(apply=True)                               # LR-Range-Test, Empfehlung übernehmen
p.set("training.paradigm", "evolution_strategies")  # gradientenfrei (OpenAI-ES)

p.update({"workflow.search.method": "evolution", "workflow.soup_runs": 3, "workflow.compress_after_training": True,
          "workflow.export_format": "export"})
p.run()                                             # Suche → Training (Soup) → Kompression → Export
```

PBT verändert Trainings-Einstellungen *während* des Trainings (schlechte Mitglieder übernehmen Gewichte und
Einstellungen der guten und variieren sie). Neue Trainings-Techniken: Verlust-Spitzen-Schutz mit Rücksprung, SWA,
Lookahead, Gradient Centralization, Grokfast, Bild-Augmentierung inkl. CutMix, Token-Rauschen, Klassen-Ausgleich,
Label Smoothing für jede Klassifikation, DP-SGD (Gradient je Beispiel begrenzen + Rauschen).

## Logger & Cluster

```python
p.set("training.logging", {"level": "minimal",                        # still | minimal | normal | ausfuehrlich | debug
                           "backends": ["console", {"name": "jsonl", "args": {"level": "debug"}}, "csv", "datei",
                                        "tensorboard"],
                           "weights_every_n_steps": 200, "gradients_every_n_steps": 100, "max_file_mb": 50})
p.add_logger(logging.getLogger("firma.ki"))         # eigener Logger (oder Objekt mit message/metrics/…)
```

Verteilt: `training.distributed.strategy = ddp | zero1 | fsdp`, Start mit
`python -m ki_baukasten verteilt projekt.json --prozesse 4` (mehrere Rechner: `--knoten 2 --knoten-rang 0 --master
host:29500`). Jeder Prozess sieht seinen Teil der Daten; Ausgaben und Speichern übernimmt Prozess 0. Getestet ist,
dass 2 Prozesse exakt dasselbe Ergebnis liefern wie ein Prozess mit doppeltem Batch – für DDP, ZeRO-1 und FSDP.

## Fremde Modelle importieren & Probelauf

```python
p = Project.import_model("modell.pt2", input_shape=[1, 32])              # torch.export / TorchScript / …
p = Project.import_model("gewichte.pt", format="state_dict",             # Gewichte in eine Baukasten-Architektur
                         architecture={"name": "transformer", "args": {...}}, key_map={"^model\\.": ""})
p.set("tensor_rules", [{"target": "embed", "op": "freeze"}]); p.train()  # ändern, nachtrainieren …
p.compress([...]); p.export("modell_neu", "export")                       # … eindampfen, neu exportieren
```

Hugging Face (`format="huggingface"`) und safetensors werden genutzt, wenn die Pakete `transformers` bzw.
`safetensors` installiert sind. Gepickelte Komplettmodelle nur mit `allow_pickle=True` (Pickle führt Code aus).

**Probelauf** (`analysis/probe.py`, GUI „Probelauf & Gewichte“, CLI `probelauf`): ein Forward + Backward, bei dem jedes
Modul in Ausführungsreihenfolge aufgezeichnet wird – Ein-/Ausgabeform, Mittel/Std/Min/Max, Anteil Nullen, NaN/Inf,
Gradient, Zeit – mit Warnungen bei toten Neuronen, explodierenden oder verschwindenden Werten. Dazu
`weight_stats`/`histogram`/`heatmap`, `attention_maps` und `evaluate_model` (Metriken, Konfusionsmatrix, Beispiele).

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
├── core/          Param, BaseConfig, Tooltip/Level, Registry, ComponentSpec, Selektoren, Schalter, Fehler
├── components/    activations, normalization, initializers, positional, attention, feedforward, mixers, layers,
│                  scan (schnelle lineare Rekurrenzen), cache (KV-/Zustands-Cache)
├── models/        base, mlp, cnn, rnn, transformer (+ssm/hybrid), autoencoder, generative (gan/diffusion),
│                  gnn, rl_nets, graph (freies Netz), composite (ensemble/pipeline/fusion/mixture/multitask),
│                  external (Import fremder Modelle)
├── training/      config, trainer, paradigms (+ Evolution Strategies), optimizers, schedulers, losses, callbacks
│                  (+ LISA, SWA), data_utils, continual, distributed (DDP/ZeRO-1/FSDP), logger, lr_finder
│   └── rl/        environments, rewards, algorithms (ppo/a2c/reinforce/dqn)
├── memory/        config, manager, streaming (RAM-/SSD-Training), offload (Swapping, device_map), quant
│                  (int8/int4/nf4), lora (LoRA/DoRA/QLoRA), estimator, execution (stückweise, OOM-Teilung)
├── tensor/        surgery (Tensor-Chirurgie), modules (Modul-Chirurgie), selection (Geltungsbereiche)
├── compress/      structure, methods (Pruning, Low-Rank, Sparsity), pipeline (Rezepte + Messung)
├── analysis/      architecture, advisor, export, benchmark (Größe/MACs/Latenz), targets (Zielsysteme), probe,
│                  fallback (Notfall-Ausführung), goals (Ziel-Assistent)
├── workflow/      config (Ablauf), search (Zufall/Gitter/Evolution/PSO/PBT), federated (FedAvg/FedProx, Model Soup)
├── techniques.py  Techniken-Katalog (an/aus, Status)
├── data/          datasets (synthetisch + CSV/Text/Tensor-Dateien)
├── presets/       Vorlagen je Anwendungsgebiet, Speicherprofile
├── ui/            cli, gui (Tkinter), assistant (Ziel-Assistent, Techniken-Fenster), inspector (Probelauf-Fenster),
│                  docs (Referenz + Möglichkeiten-Katalog)
└── project.py     Project-Fassade
examples/          23 ausführlich kommentierte Beispiele + Beispiel-Konfigurationsdatei
tests/             ~460 Tests (pytest), CI per GitHub Actions (Python 3.10 und 3.12)
docs/              ARCHITEKTUR.md, MOEGLICHKEITEN.md (Katalog), PARAMETER_REFERENZ.md (beide automatisch erzeugt)
```

| Beispiel | Inhalt |
|---|---|
| `01_schnellstart.py` … `09_generativ.py` | Grundlagen, eigene Architekturen, freies Netz, Kombinationen, Tensor-Chirurgie, Speicher, RL, eigene Komponenten, generative Modelle |
| `10_mamba_schnell_und_cache.py` | schneller SSM-Scan (Zeit/Speicher gemessen), Generierung mit Cache, Cache-Speicher |
| `11_qlora_feintuning.py` | nf4 + LoRA + 8-Bit-AdamW, Speichern/Laden, Export |
| `12_ram_streaming.py` | trainierbares Auslagern in den RAM – bitgenau wie ohne |
| `13_regeln_geltungsbereich.py` | Übernahme zwischen Schichten, Modul-/Tensor-Regeln mit Geltungsbereich, Platzhalter, Rückgängig |
| `14_ratgeber_aufbau_export.py` | Ratgeber übernehmen, Unterschiede, Aufbau (Text/Markdown/JSON), Export |
| `15_modell_eindampfen.py` | Kompressions-Rezept mit Messung, Speichern/Laden, Zielsystem-Empfehlung, `reduce_for` |
| `16_lehrer_schueler_weiterlernen.py` | Schüler per Distillation, Weiterlernen mit LwF/EWC/Replay im Vergleich |
| `17_import_aendern_export.py` | `.pt2` importieren und nachtrainieren, Gewichte mit `key_map` laden, komprimieren, neu exportieren |
| `18_probelauf_gewichte.py` | Probelauf aller Wege, Gewichte, Histogramm/Heatmap, Attention-Karten, Testlauf |
| `19_ziele_assistent.py` | Zweck & Prioritäten → Empfehlen/Abraten, Projekt nach Zielen, Ziele später ändern |
| `20_techniken_und_schalter.py` | Techniken an/ab, An/Aus/Eigen je Einstellung, Gruppen, Übersicht |
| `21_notfall_ausfuehrung.py` | Notfall-Optionen mit Schätzung, Training mit Basis auf der SSD, stückweise Inferenz |
| `22_ablauf_suche_schwarm_evolution.py` | PSO, genetisch, PBT, föderiert, Model Soup, Evolution Strategies, `run()` |
| `23_logger_und_cluster.py` | Logger-Ziele und -Stufen, eigener Logger, verteiltes Training mit 2 Prozessen |

Konvention: **Code-Bezeichner englisch** (wie in PyTorch und Fachliteratur), **alle Erklärungen,
Kommentare, Tooltips und Ausgaben deutsch**.

---

## Tests

```bash
pip install pytest
pytest            # ~460 Tests, ca. 60–90 s auf CPU
```

Geprüft werden u. a.: jede Komponente jeder Registry, jedes Modell vorwärts/rückwärts, Kausalität aller Mixer und
Positionskodierungen, SDPA = manuelle Attention, jedes Paradigma, jeder Optimierer und Zeitplan, alle Speicheroptionen
(Checkpointing gradientengleich, Offloading bitgenau), Tensor-Chirurgie (Teil-Einfrieren hält exakt, Pruning-Maske bleibt),
alle Vorlagen (Bauen + Konfigurations-Rundlauf), CLI und Doku-Export – und dass PPO CartPole löst. Neu:
Scan-Gradienten gegen Referenz (inkl. gradcheck), Cache = Neuberechnung für alle Mixer/Positionen, RAM-Streaming
gradienten- und trainingsgleich, Quantisierungsgenauigkeit, LoRA-Start/-Merge, QLoRA speichern/laden, 8-Bit-AdamW-
Konvergenz, Übernahme/Geltungsbereiche/Modul-Regeln, Ratgeber/Aufbau/Export/Rückgängig. Runde 3: Schalter
(Aus/An/Eigen, Gruppen, Rückgängig), jede Technik an → aus stellt den Zustand exakt wieder her, Ziel-Assistent
(Empfehlungen je Szenario), SSD-Streaming trainingsgleich zu RAM-Streaming, device_map exakt, OOM-Teilung, LISA,
Chunked Prefill, verteiltes Training (2 gloo-Prozesse = 1 Prozess mit doppeltem Batch für DDP/ZeRO-1/FSDP, GAN per
All-Reduce), Logger-Stufen/-Ziele/Rotation, alle Suchverfahren, föderiert, Model Soup, Evolution Strategies,
Spike-Schutz mit Rücksprung, Lookahead, DP-SGD.

---

## Ehrliche Grenzen

- Der **SSM-Scan** ist reines PyTorch: auf der GPU schnell, aber ohne Fusion in einen einzigen Kernel wie beim
  offiziellen `mamba-ssm` (der mit `scan_mode = "mamba_ssm"` genutzt wird, wenn installiert).
- **RAM-Streaming** überträgt die Block-Gewichte zweimal pro Schritt und rechnet jeden Block neu (typisch 1.3–3×
  langsamer); der CUDA-Prefetch ist experimentell und in dieser Testumgebung (nur CPU) nicht geprüft.
- **Eigene Quantisierung** entpackt die Gewichte pro Schicht vor der Matrixmultiplikation (spart Speicher, keine
  int8-Rechenkernel); für maximale Geschwindigkeit `torchao_int8/int4`. `dynamic_int8` (torch.ao) ist veraltet.
- **MLA-lite** ist vereinfacht (RoPE auf rekonstruierte Keys, keine entkoppelten RoPE-Köpfe wie bei DeepSeek).
- **Gewichts-Quantisierung spart Speicher, nicht CPU-Zeit**: die Gewichte werden pro Schicht entpackt – auf der CPU
  ist das oft sogar langsamer als float32 (die Kompressions-Messung zeigt das ehrlich an).
- **Strukturiertes Pruning** gibt es für Transformer/SSM/Hybrid (Köpfe, FFN, Blöcke) und das MLP-Modell; für CNNs
  (Residual-Kanäle) bisher nur Low-Rank, Sparsity und Quantisierung.
- **Zielsystem-Profile** sind grobe Richtwerte (effektive Leistung bei Batch 1) – sie ersetzen kein Benchmarking auf
  dem echten Gerät. ExecuTorch/CoreML/TensorRT selbst sind nicht eingebaut; der Baukasten liefert die `.pt2`/ONNX-Datei.
- **Import** über `safetensors`/Hugging Face braucht die jeweiligen Pakete und ist in dieser Testumgebung nicht
  geprüft; aus `torch.export` importierte Graphen behalten Namen, aber keine `nn.Linear`-Objekte (LoRA/Quantisierung
  dort erst nach Import der Gewichte in eine Baukasten-Architektur). Exportierte SSM-Modelle sind reine Inferenz.
- **Verteiltes Training** ist mit 2 CPU-Prozessen (gloo) geprüft; NCCL/Mehr-GPU und mehrere Rechner sind in dieser
  Umgebung nicht testbar. FSDP setzt voraus, dass das Paradigma das Modell nur über `model(...)` aufruft.
- **DP-SGD** rechnet jedes Beispiel einzeln (exakte Begrenzung, aber langsam) und berechnet kein Privacy-Budget (ε);
  für formale Garantien Opacus nutzen. Nicht mit BatchNorm kombinierbar.
- **Föderiertes Lernen** wird simuliert (Teilnehmer = Teile des Datensatzes); echte Geräte-Kommunikation ist nicht
  eingebaut. **Model Soups** sind für Läufe vom selben Startpunkt gedacht.
- Der **Ziel-Assistent** ist ein nachvollziehbares Regelwerk (jede Begründung wird angezeigt), kein Orakel; die
  Profile beziehen sich auf die mitgelieferten Vorlagen.
- **Notfall-Schätzungen** (Speicher, Verlangsamung) sind Größenordnungen; die Schicht-für-Schicht-Ausführung von der
  SSD ist für Inferenz gedacht (Training mit Basis auf der SSD: `stream` + `storage = "disk"`).
- **TensorBoard, W&B, MLflow** sind als Logger-Ziele eingebunden, in dieser Umgebung aber nicht installiert und daher
  nur auf „fehlt → Warnung, Training läuft weiter“ geprüft.
- Die Vorlagen sind auf kleine synthetische Daten abgestimmt; für echte Projekte `data` auf `csv`, `text` oder
  `tensor_file` umstellen (Größen passen sich automatisch an) und Umfang/Lernrate anpassen.
