"""
Beispiel 15 – Modelle eindampfen und für Zielsysteme reduzieren
===============================================================

1. Trainieren – danach erscheint automatisch die Reduktions-Übersicht für Zielsysteme
2. Eigenes Kompressions-Rezept: Köpfe + FFN-Neuronen + Blöcke entfernen, heilen, quantisieren
   (mit Messung nach jedem Schritt)
3. Speichern/Laden des kleineren Modells (die Formänderungen stehen in module_rules)
4. Empfehlung für ein Zielsystem anzeigen und automatisch anwenden + Export

Ausführen:  python examples/15_modell_eindampfen.py
"""

import warnings

import torch

from ki_baukasten import Project

warnings.simplefilter("ignore")

p = Project.from_preset("sprachmodell_mini")
p.set("training.max_steps", 120)
p.set("deployment.targets", ["gaming_gpu", "laptop_cpu", "smartphone", "raspberry_pi", "mikrocontroller"])
p.train()                                         # → am Ende: Übersicht "Reduktion für Zielsysteme"

# --- eigenes Rezept ------------------------------------------------------------------------------
klein = Project(p.config.copy())
klein._bundle, klein.model = p._bundle, __import__("copy").deepcopy(p.model)
klein.config.training.verbose = True
klein.compress([
    {"name": "prune_heads", "args": {"amount": 0.5, "importance": "activation"}},
    {"name": "prune_ffn", "args": {"amount": 0.5, "importance": "activation"}},
    {"name": "drop_layers", "args": {"count": 1}},            # Block mit dem geringsten Einfluss
    {"name": "heal", "args": {"steps": 80, "feature_weight": 0.5}},
    {"name": "quantize", "args": {"format": "int8"}},
])
ordner = klein.save("ausgaben/eingedampft")
wieder = Project.load(ordner)
x = torch.randint(0, 40, (1, 24))
klein.model.cpu().eval()
with torch.no_grad():
    print("Nach dem Laden identisch:", torch.allclose(klein.model(x), wieder.model.eval()(x)))

# --- Zielsystem-Empfehlung --------------------------------------------------------------------
print()
print(p.recommend_for("raspberry_pi"))
p.config.training.verbose = False
pi = p.reduce_for("raspberry_pi", export="ausgaben/raspberry_pi/modell")
print("\nAngewandt:", pi.last_compression.after if pi.last_compression else "nichts nötig")
