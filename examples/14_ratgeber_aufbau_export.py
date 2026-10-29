"""
Beispiel 14 – Ratgeber, komplette Aufbau-Darstellung und Export
==============================================================

* ``advise()`` – Vorschläge für Speicher, Tempo, Stabilität und Struktur, direkt übernehmbar
* ``architecture()`` – Text, Markdown (mit Mermaid-Diagramm) oder JSON
* ``diff()`` – was hat der Ratgeber geändert?
* ``export()`` – torch.export / TorchScript / state_dict

Ausführen:  python examples/14_ratgeber_aufbau_export.py
"""

import json
import warnings

from ki_baukasten import Project

warnings.simplefilter("ignore")

p = Project.from_preset("sprachmodell_mini")
p.set("training.verbose", False)
p.set("model.block.mixer.num_heads", 8)
p.set("training.gradient_clipping.mode", "none")
p.set("memory.memory_budget_gb", 0.05)                 # knappes Budget, damit Speichervorschläge erscheinen

vorher = Project(p.config.copy())
for tipp in p.advise():
    print(tipp, end="\n\n")

angewandt = p.apply_advice(category="speicher") + p.apply_advice(["gqa", "clipping"])
print("Übernommen:", [a.key for a in angewandt])
for pfad, neu, alt in p.diff(vorher):
    print(f"  {pfad:<45} {alt!r} → {neu!r}")

print()
print(p.architecture("text", max_depth=2))
p.architecture("markdown", path="ausgaben/aufbau_sprachmodell.md")
baum = json.loads(p.architecture("json"))
print("\nJSON-Schlüssel:", list(baum), "– Schichten:", len(baum["layers"]))

p.set("training.max_steps", 10)
p.train()
for fmt in ("export", "torchscript", "state_dict"):
    print("Export:", p.export(f"ausgaben/sprachmodell_export/modell_{fmt}", fmt))
