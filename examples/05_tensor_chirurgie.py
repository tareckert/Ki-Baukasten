"""
Beispiel 05 – Tensor-Chirurgie: Eingriffe bis auf einzelne Gewichte
===================================================================

Regeln (Auswahl per Regex/Modultyp + optionaler Ausschnitt + Operation):

* Embeddings einfrieren
* in Schicht 0 nur die ersten 8 Zeilen der Query-Projektion einfrieren (Ausschnitt!)
* alle linearen Gewichte in Schicht 1 zu 50 % prunen (Maske bleibt im Training)
* Ausgangsprojektionen per Max-Norm beschränken (nach jedem Schritt)
* Kopf mit 10-fach kleinerer Lernrate trainieren
* einen Tensor auf Rang 4 projizieren, einen anderen mit Rauschen versehen

Ausführen:  python examples/05_tensor_chirurgie.py
"""

from ki_baukasten import Project

projekt = Project.from_preset("sprachmodell_mini")
projekt.update({"model.n_layers": 2, "training.max_steps": 40})
projekt.set("tensor_rules", [
    {"label": "Embeddings einfrieren", "target": "embed", "op": "freeze"},
    {"label": "Teil-Einfrieren", "target": r"blocks\.0\..*q_proj\.weight", "index": "0:8, :", "op": "freeze"},
    {"label": "Pruning", "target": r"blocks\.1\.", "module_type": "Linear", "param_kind": "weight",
     "op": {"name": "prune", "args": {"amount": 0.5}}},
    {"label": "Max-Norm", "target": "o_proj", "op": {"name": "constraint", "args": {"kind": "max_norm", "value": 1.0}}},
    {"label": "Kopf langsamer", "target": "lm_head", "op": {"name": "lr_scale", "args": {"factor": 0.1}}},
    {"label": "Rang 4", "target": r"blocks\.1\..*k_proj\.weight", "op": {"name": "low_rank", "args": {"rank": 4}}},
    {"label": "Rauschen", "target": r"blocks\.0\.ffn\.up", "op": {"name": "noise", "args": {"std": 0.01}}},
])

modell = projekt.build()
print(projekt.surgeon.preview())
print(projekt.surgeon.report())
projekt.train()
w = modell.blocks[1].mixer.attn.q_proj.weight
print(f"Anteil Nullen nach dem Training (Pruning gehalten): {(w == 0).float().mean():.2f}")
