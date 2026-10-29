"""
Beispiel 17 – Fremde Modelle importieren, verändern und neu exportieren
======================================================================

1. Ein Modell als torch.export-Datei (.pt2) und als reine Gewichte erzeugen (stellvertretend
   für ein "fremdes" Modell)
2. .pt2 importieren → Aufbau ansehen, Tensor-Regel anwenden, nachtrainieren
3. Gewichte mit anderen Schlüsselnamen in eine Baukasten-Architektur laden (key_map)
   → komprimieren → speichern → erneut exportieren

Ausführen:  python examples/17_import_aendern_export.py
"""

import warnings

import torch

from ki_baukasten import Project

warnings.simplefilter("ignore")

quelle = Project.from_preset("sprachmodell_mini")
quelle.set("training.max_steps", 40)
quelle.set("training.verbose", False)
quelle.set("deployment.show_after_training", False)
quelle.train()
pt2 = quelle.export("ausgaben/fremd/modell", "export")
torch.save({"netz." + k: v for k, v in quelle.model.state_dict().items()}, "ausgaben/fremd/gewichte.pt")

# --- 2. .pt2 importieren und ändern ---------------------------------------------------------------
imp = Project.import_model(pt2, input_shape=[1, 32], input_high=40, data="text")
print(imp.architecture("text", max_depth=2))
imp.set("training.paradigm", "causal_lm")
imp.set("training.max_steps", 20)
imp.set("training.verbose", False)
imp.set("tensor_rules", [{"target": "embed", "op": "freeze"}])
imp.train()
print("Nachtraining des importierten Modells:", imp.history[-1])

# --- 3. Gewichte mit fremden Namen in eine Baukasten-Architektur --------------------------------------
gw = Project.import_model("ausgaben/fremd/gewichte.pt", format="state_dict",
                          architecture=quelle.config.model.to_dict(), key_map={r"^netz\.": ""})
print("Import-Bericht:", gw.model.import_report)
gw.config.training.verbose = True
gw.compress([{"name": "prune_ffn", "args": {"amount": 0.4, "importance": "magnitude"}},
             {"name": "quantize", "args": {"format": "int8"}}])
gw.save("ausgaben/fremd/projekt")
print("Neu exportiert:", gw.export("ausgaben/fremd/modell_klein", "export"))
