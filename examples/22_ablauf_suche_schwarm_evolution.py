"""
Beispiel 22 – Ablauf: automatische Suche, Schwarm, Evolution, föderiert, Model Soup
==================================================================================

* ``tune``: Zufall, Gitter, genetischer Algorithmus, Partikelschwarm (PSO), Population Based Training
* ``federated_train``: föderiertes Lernen (FedAvg/FedProx), Daten bleiben bei den Teilnehmern
* ``model_soup``: mehrere Läufe vom selben Start mitteln
* Paradigma ``evolution_strategies``: Training ganz ohne Backward
* ``run()``: der ganze Ablauf (workflow) in einem Befehl – jede Stufe schaltbar

Ausführen:  python examples/22_ablauf_suche_schwarm_evolution.py
"""

import tempfile
import warnings

from ki_baukasten import Project

warnings.simplefilter("ignore")


def neu(**over):
    base = {"training.callbacks": [], "training.verbose": False, "deployment.show_after_training": False}
    return Project.from_preset("tabellen_klassifikation", **{**base, **over})


# Schwarmintelligenz: Partikelschwarm sucht Lernrate und Weight-Decay
p = neu()
ergebnis = p.tune("pso", space={"training.optimizer.lr": {"min": 3e-4, "max": 3e-2, "log": True},
                                "training.optimizer.weight_decay": [0.0, 0.01, 0.1]},
                  steps_per_trial=40, population=4, generations=3)
print(ergebnis)

# Evolution: genetischer Algorithmus und PBT (Einstellungen ändern sich während des Trainings)
print(neu().tune("evolution", trials=4, steps_per_trial=30, population=4, generations=2).best_params)
print(neu().tune("pbt", space={"training.optimizer.lr": {"min": 1e-3, "max": 3e-2, "log": True}},
                 steps_per_trial=30, population=4, generations=3).best_params)

# Föderiert: 5 Teilnehmer mit ungleich verteilten Daten
f = neu()
verlauf = f.federated_train(clients=5, rounds=4, local_steps=20, split="nicht_iid", algorithm="fedprox")
print("Föderiert, Genauigkeit je Runde:", [round(r["val_accuracy"], 3) for r in verlauf])

# Model Soup
print("Model Soup:", neu(**{"training.max_steps": 60}).model_soup(3))

# Evolution Strategies: gradientenfrei
es = neu(**{"training.paradigm": "evolution_strategies", "training.max_steps": 60, "training.optimizer.lr": 0.01})
es.train()
print("Evolution Strategies:", {k: round(v, 3) for k, v in es.evaluate().items()})

# Der ganze Ablauf: Training → 3-fache Kreuzvalidierung → Kompression → Export
with tempfile.TemporaryDirectory() as ordner:
    w = neu(**{"training.max_steps": 60, "workflow.cross_validation": 3, "workflow.compress_after_training": True,
               "workflow.export_format": "state_dict", "workflow.export_path": f"{ordner}/modell",
               "compression.steps": [{"name": "quantize", "args": {"format": "int8"}}]})
    bericht = w.run()
    print("Ablauf:", list(bericht), "| Kreuzvalidierung:",
          {k: round(v, 3) for k, v in bericht["kreuzvalidierung"]["mittel"].items()})
