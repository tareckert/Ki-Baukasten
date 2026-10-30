"""
Beispiel 20 – Techniken-Katalog und Schalter An / Aus / Eigen
============================================================

Ebene 2 (Techniken): bewährte Verfahren von Communitys, Forschung und Firmen mit einem Befehl
an- und abwählen – Status ● an · ○ aus · ◐ teilweise · – passt nicht.
Ebene 3 (Einstellungen): jede einzelne Einstellung An (empfohlen), Aus oder Eigen.

GUI: Knopf "Techniken" (Reiter je Kategorie, Häkchen) und im Formular die Schalter An/Aus/Eigen je
Einstellung, dazu der Ansichtsfilter ("nur eingeschaltete" …).
Kommandozeile:  python -m ki_baukasten techniken sprachmodell_mini --an qlora --aus dropout
                python -m ki_baukasten schalter sprachmodell_mini --aus dropout --an training.ema

Ausführen:  python examples/20_techniken_und_schalter.py
"""

import warnings

from ki_baukasten import Project

warnings.simplefilter("ignore")

p = Project.from_preset("sprachmodell_mini", **{"training.max_steps": 20, "training.verbose": False,
                                                 "deployment.show_after_training": False})

# --- Ebene 2: Techniken ---------------------------------------------------------------------
print(p.technique_overview(only_active=True))
for r in p.techniques("schwarm") + p.techniques("evolution"):
    print(" ", r)
p.enable("qlora", "spike_schutz", "z_loss", "swa", "lookahead")    # mehrere auf einmal
print("\nQLoRA an? ", next(r.status for r in p.techniques() if r.technique.key == "qlora"))
p.disable("qlora")                                                 # vorheriger Zustand kommt exakt zurück
print("Quantisierung nach dem Ausschalten:", p.get("memory.quantization"))

# --- Ebene 3: Schalter je Einstellung ---------------------------------------------------------
p.set("**.dropout", "aus")                                         # alle Dropouts aus
p.switch("training.regularization.mixup_alpha", "an")              # empfohlener An-Wert (0.2)
p.set("training.regularization.sam_rho", 0.08)                     # eigener Wert …
p.switch("training.regularization.sam_rho", "aus")                 # … aus (gemerkt) …
p.switch("training.regularization.sam_rho", "eigen")               # … und wieder da
print("\nsam_rho:", p.get("training.regularization.sam_rho"))
try:
    p.set("model.d_model", "aus")
except Exception as exc:                                           # Pflicht-Einstellungen kennen kein Aus
    print("Pflicht:", exc)
print(p.switch_overview().splitlines()[0])
for info in p.switches("training.regularization", state="an"):
    print(" ", info)

hist = p.train()
print("\nTraining mit allen Techniken ok, letzter Verlust:", round(next(h["loss"] for h in reversed(hist) if "loss" in h), 3))
p.undo()                                                           # jede Änderung ist rückgängig machbar
