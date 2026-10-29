"""
Beispiel 03 – Ein frei definiertes Netz (Graph aus Schichten)
=============================================================

Wenn keine vorgefertigte Architektur passt: Knoten für Knoten selbst bauen.
Eingangsgrößen werden automatisch ermittelt – man gibt nur Ausgangsgrößen an.

Netz: zwei Zweige (GELU- und Sinus-Aktivierung) → gelerntes Gate → MLP → Residual → Ausgabe

Ausführen:  python examples/03_freies_netz.py
"""

from ki_baukasten import Project

projekt = Project({
    "name": "freies_netz_beispiel",
    "data": {"name": "classification", "args": {"kind": "spirals", "n_classes": 3}},
    "model": {"name": "graph", "args": {
        "inputs": [{"name": "x", "shape": [2]}],
        "nodes": [
            {"name": "zweig_a", "layer": {"name": "linear", "args": {"out_features": 64}}, "inputs": ["x"]},
            {"name": "a_akt", "layer": {"name": "activation", "args": {"activation": "gelu"}}},
            {"name": "zweig_b", "layer": {"name": "linear", "args": {"out_features": 64}}, "inputs": ["x"]},
            {"name": "b_akt", "layer": {"name": "activation", "args": {"activation": {"name": "sine", "args": {"w0": 3.0}}}}},
            {"name": "gate", "layer": "identity", "inputs": ["a_akt", "b_akt"], "merge": "gated"},
            {"name": "tief", "layer": {"name": "mlp", "args": {"hidden_dims": [64], "output_dim": 64}}},
            {"name": "residual", "layer": {"name": "normalization", "args": {"norm": "layernorm"}},
             "inputs": ["gate", "tief"], "merge": "add"},
            {"name": "ausgabe", "layer": {"name": "linear", "args": {"out_features": 3}}},
        ],
    }},
    "training": {"epochs": 60, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}, "verbose": False},
})

modell = projekt.build()
print("Formen je Knoten:", modell.shapes())
projekt.train()
print("Ergebnis:", projekt.evaluate())
