"""
Beispiel 13 – Jede Schicht und jeder Tensor: für alle, nur bestimmte oder genau einen
===================================================================================

1. ``layer_overrides`` mit Übernahme (@von / @nur / @ausser)
2. Modul-Chirurgie (``module_rules``) mit Geltungsbereich
3. Tensor-Chirurgie (``tensor_rules``) mit layers / scope / pick / copy
4. Platzhalter-Pfade, Suche, Rückgängig

Ausführen:  python examples/13_regeln_geltungsbereich.py
"""

from ki_baukasten import Project

p = Project.from_preset("sprachmodell_mini")
p.set("training.verbose", False)
p.set("model.n_layers", 6)

# 1) Schicht 0 bekommt eine Sonderrolle; andere Schichten übernehmen sie ganz oder teilweise.
p.set("model.layer_overrides", {
    "0": {"mixer.num_heads": 8, "ffn": "mlp", "dropout": 0.0},
    "-1": {"@von": 0},                                  # letzte Schicht = Kopie der Konfiguration von Schicht 0
    "3": {"@von": 0, "@nur": ["mixer.num_heads"]},      # nur die Kopfzahl übernehmen
    "4": {"@von": 0, "@ausser": ["ffn"], "dropout": 0.2},
})

# 2) Eingriffe in gebaute Module
p.set("module_rules", [
    {"label": "Schicht 5 startet mit den Gewichten von Schicht 0", "target": "blocks.5", "match": "exact",
     "op": {"name": "copy_from", "source": "blocks.0"}},
    {"label": "nur die erste Norm wird LayerNorm", "module_type": "RMSNorm", "scope": "nur_einer", "pick": "0",
     "op": {"name": "replace_norm", "norm": "layernorm"}},
    {"label": "Normen außer der ersten in fp32 halten", "module_type": "RMSNorm", "scope": "alle_ausser", "pick": "0",
     "op": {"name": "dtype", "dtype": "float32"}},
    {"label": "Aktivierungen jedes Blocks aufzeichnen", "target": r"blocks\.\d+$", "match": "full", "op": "record"},
])

# 3) Eingriffe in einzelne Tensoren
p.set("tensor_rules", [
    {"label": "Q-Projektion in erster und letzter Schicht einfrieren", "target": "q_proj.weight", "layers": "0,-1",
     "op": "freeze"},
    {"label": "genau eine Ausgabeprojektion nullen", "target": "o_proj.weight", "scope": "nur_einer", "pick": "2",
     "op": {"name": "set", "value": 0.0}},
    {"label": "Kopf 0 von Schicht 1 nach Kopf 1 von Schicht 2 kopieren", "target": "v_proj.weight", "layers": "2",
     "index": "16:32, :", "op": {"name": "copy", "source": "blocks.1.mixer.attn.v_proj.weight",
                                 "source_index": "0:16, :"}},
])

# 4) Platzhalter, Suche, Rückgängig
print("Alle dropout-Pfade:", p.find("dropout"))
p.set_all("dropout", 0.05, scope="alle_ausser", pick="-1")      # alle außer dem letzten Treffer
print("Suche 'sliding':", p.search("sliding")[:3])
p.set("model.d_model", 192)
p.undo()                                                         # d_model wieder zurück

p.set("training.max_steps", 20)
p.train()
print()
print(p.module_surgeon.report())
print()
print(p.surgeon.report())
print()
print(p.module_surgeon.activation_report())
print()
print(p.architecture("text", max_depth=1))
