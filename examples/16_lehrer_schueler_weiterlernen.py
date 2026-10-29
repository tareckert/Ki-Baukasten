"""
Beispiel 16 – Lehrer-Schüler (Distillation) und Weiterlernen ohne Vergessen
==========================================================================

* ``distill``: ein automatisch verkleinertes Schüler-Modell lernt vom trainierten Lehrer
  (weiche Ziele + optional innere Zustände = Feature-Distillation)
* ``continue_training``: neue Daten lernen – mit LwF (alte Version als Lehrer), EWC
  (wichtige Gewichte festhalten) oder Replay (alte Beispiele beimischen) im Vergleich

Ausführen:  python examples/16_lehrer_schueler_weiterlernen.py
"""

import copy
import warnings

import torch

from ki_baukasten import Project
from ki_baukasten.core import ComponentSpec

warnings.simplefilter("ignore")

# --- Lehrer → Schüler ----------------------------------------------------------------------------
lehrer = Project.from_preset("sprachmodell_mini")
lehrer.set("training.max_steps", 150)
lehrer.set("training.verbose", False)
lehrer.set("deployment.show_after_training", False)
lehrer.train()
schueler = lehrer.distill(width=0.5, depth=0.5, steps=150, feature_weight=0.5, kd_loss="kl")
n_l = sum(t.numel() for t in lehrer.model.parameters())
n_s = sum(t.numel() for t in schueler.model.parameters())
print(f"Lehrer {n_l:,} Parameter, Schüler {n_s:,} ({100 * n_s / n_l:.0f} %)")
print("Lehrer :", {k: round(v, 3) for k, v in lehrer.history[-1].items() if k.startswith("val_")})
print("Schüler:", {k: round(v, 3) for k, v in schueler.history[-1].items() if k.startswith("val_")})
print(lehrer.generate("Der ", max_new_tokens=40, temperature=0.7))
print(schueler.generate("Der ", max_new_tokens=40, temperature=0.7))

# --- Weiterlernen ---------------------------------------------------------------------------------
basis = Project.from_preset("tabellen_klassifikation")
basis.set("training.verbose", False)
basis.set("deployment.show_after_training", False)
basis.train()
alt = basis.data().train
a = basis.config.model.args
neu = ComponentSpec.from_value("dataset", {"name": "classification", "args": {
    "n_features": a.input_dim, "n_classes": a.output_dim, "seed": 123}}).build().train


def genauigkeit(model, daten):
    model.eval()
    with torch.no_grad():
        return float((model(daten["inputs"]).argmax(-1) == daten["targets"]).float().mean())


print(f"\nVorher: alt {genauigkeit(basis.model, alt):.3f}")
for methode in ["none", "lwf", "ewc", "replay"]:
    kopie = Project(basis.config.copy())
    kopie._bundle, kopie.model = basis._bundle, copy.deepcopy(basis.model)
    kopie.continue_training(neu, method=methode, strength=0.8 if methode == "lwf" else 1.0, steps=150)
    print(f"{methode:<7} alt {genauigkeit(kopie.model, alt):.3f}   neu {genauigkeit(kopie.model, neu):.3f}")
