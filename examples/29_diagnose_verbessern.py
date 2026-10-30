"""
Beispiel 29 – Diagnose nach dem Training, automatisch verbessern, nichts geht stillschweigend verloren
=====================================================================================================

* ``diagnose``: was bremst das Modell? (instabil, lernt kaum, zu kurz, Überanpassung, Basis …) – mit Maßnahmen
* ``improve``: Diagnose → Maßnahmen → neu trainieren; nur, was die Validierung verbessert, bleibt
* ``inactive_settings``: geänderte Einstellungen, die gerade nichts bewirken – mit Grund und Abhilfe
* Technik-Konflikte werden gemeldet (z. B. ein Optimierer ersetzt den anderen)

Ausführen:  python examples/29_diagnose_verbessern.py
"""

import warnings

from ki_baukasten import Project

warnings.simplefilter("ignore")
LEISE = {"training.verbose": False, "deployment.show_after_training": False}

# --- 1. Ein absichtlich schlecht eingestelltes Training ------------------------------------------
p = Project.from_preset("tabellen_klassifikation", **LEISE, **{"training.max_steps": 200,
                                                                 "training.optimizer.lr": 1e-6})
p.train()
print(p.diagnose())          # ✗ Modell lernt kaum … → LR-Finder, länger trainieren

# --- 2. Automatisch verbessern -------------------------------------------------------------------
ergebnis = p.improve(rounds=2)
for runde in ergebnis["runden"]:
    print(f"Runde {runde['runde']}: {ergebnis['metrik']} {runde['vorher']:.3f} → {runde['nachher']:.3f} "
          f"({'behalten' if runde['behalten'] else 'verworfen'}) – " + "; ".join(runde["massnahmen"]))
print(f"Insgesamt: {ergebnis['metrik']} {ergebnis['start']:.3f} → {ergebnis['ende']:.3f}")
print(f"Lernrate jetzt: {p.config.training.optimizer.args.lr:.3g} (vorher 1e-06)\n")

# --- 3. Wirkungslose Einstellungen und Technik-Konflikte -----------------------------------------
q = Project.from_preset("sprachmodell_mini", **LEISE)
q.set("memory.lora.rank", 16)            # LoRA ist aus → wirkt nicht
q.set("training.ema.decay", 0.99)        # EMA ist aus → wirkt nicht
for eintrag in q.inactive_settings():
    print("⚠", eintrag)
q.enable("lion")
q.enable("muon")                          # ersetzt 'lion' – wird gemeldet statt stillschweigend
print(q.notes[-1])
