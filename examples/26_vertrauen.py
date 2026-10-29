"""
Beispiel 26 – Vertrauen: Kalibrierung, Unsicherheit, Erklärungen, Fairness, Robustheit, wenige Labels
=====================================================================================================

* ``calibrate``: Temperature Scaling (+ Conformal-Grenze) auf der Validierung
* ``predict_with_uncertainty``: Wahrscheinlichkeit, Entropie, MC-Dropout-Streuung, Conformal-Menge, Enthaltung
* ``explain_prediction`` / ``feature_importance``: warum diese Vorhersage, welche Merkmale zählen?
* ``fairness_report``: werden Gruppen gleich behandelt?
* ``training.adversarial`` + ``robustness_report``: robust gegen Störungen
* ``select_for_labeling`` / ``pseudo_label``: mit wenigen Labels weit kommen

Ausführen:  python examples/26_vertrauen.py
"""

import warnings

import torch

from ki_baukasten import Project

warnings.simplefilter("ignore")

p = Project.from_preset("tabellen_klassifikation", **{"training.max_steps": 300, "training.verbose": False,
                                                      "deployment.show_after_training": False})
p.update({"inference.conformal_alpha": 0.1, "inference.uncertainty": "mc_dropout", "inference.samples": 8})
p.train()
print("Kalibrierung:", p.calibrate())

x = p.data().train["inputs"][:4]
u = p.predict_with_uncertainty(x)
for i in range(4):
    print(f"  Beispiel {i}: {u['vorhersage'][i]}  p={float(u['wahrscheinlichkeit'][i]):.2f}  "
          f"Streuung={float(u['streuung'][i]):.3f}  Conformal-Menge={u['menge'][i]}")
p.set("inference.abstain_below", 0.8)                 # unsichere Fälle an Menschen weitergeben
print("Mit Enthaltung:", p.predict_labels(x))
p.set("inference.abstain_below", None)

print(p.explain_prediction(x, index=0))
print(p.feature_importance(repeats=2))

# Fairness: Gruppe = Vorzeichen des ersten Merkmals (bei echten Daten z. B. 'geschlecht' als Spaltenname)
val = p.trainer.val_data
gruppen = [int(val[i]["inputs"][0] > 0) for i in range(len(val))]
print(p.fairness_report(gruppen))

# Robustheit vor und nach adversarialem Training (PGD)
print(p.robustness_report(epsilons=[0.05, 0.1], methods=("rauschen", "pgd")))
r = Project.from_preset("tabellen_klassifikation", **{"training.max_steps": 300, "training.verbose": False,
                                                      "deployment.show_after_training": False})
r.enable("adversarial_training")
r.set("training.adversarial.epsilon", 0.1)
r.train()
print(r.robustness_report(epsilons=[0.05, 0.1], methods=("rauschen", "pgd")))

# Wenige Labels: welche Beispiele sollen Menschen beschriften? Danach sichere Vorhersagen mittrainieren
pool = torch.randn(300, 2)
print(p.select_for_labeling(pool, n=5, strategy="marge"))
print("Pseudo-Labeling:", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r_.items()}
                           for r_ in p.pseudo_label(pool, threshold=0.9)["runden"]])
