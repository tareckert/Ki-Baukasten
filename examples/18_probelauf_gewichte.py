"""
Beispiel 18 – Probelauf: alle Wege, Aktivierungen, Gradienten, Gewichte, Attention, Testlauf
==========================================================================================

Dasselbe gibt es interaktiv in der GUI (Knopf "Probelauf & Gewichte") – mit Tabellen,
Histogrammen und Heatmaps.

Ausführen:  python examples/18_probelauf_gewichte.py
"""

import warnings

from ki_baukasten import Project
from ki_baukasten.analysis.probe import (attention_maps, evaluate_model, evaluation_text, get_weight, heatmap,
                                         histogram, probe, weight_stats)

warnings.simplefilter("ignore")

p = Project.from_preset("sprachmodell_mini")
p.set("training.max_steps", 30)
p.set("training.verbose", False)
p.set("deployment.show_after_training", False)
p.train()
bundle = p.data()
daten = bundle.val if bundle.val is not None else bundle.train

ergebnis = probe(p.model, daten["inputs"][:2])           # Forward + Backward mit Aufzeichnung
print(ergebnis.to_text(max_depth=2))

print("\nGewichte mit der größten Streuung:")
for s in sorted(weight_stats(p.model), key=lambda r: -r["std"])[:5]:
    print(f"  {s['name']:<40} {str(s['shape']):<14} std={s['std']:.4f} |max|={s['absmax']:.3f}")
w = get_weight(p.model, "blocks.0.mixer.attn.q_proj.weight")
edges, counts = histogram(w, 12)
print("\nHistogramm q_proj (Block 0):", counts)
print("Heatmap-Größe:", len(heatmap(w, 16, 16)), "×", len(heatmap(w, 16, 16)[0]))

karten = attention_maps(p.model, daten["inputs"][:1, :12])
erste = next(iter(karten))
print(f"\nAttention {erste}: Kopf 0, Zeile 11 →", [round(v, 2) for v in karten[erste][0, 11].tolist()])

print()
print(evaluation_text(evaluate_model(p.model, daten, max_batches=4)))
