"""
Beispiel 24 – Perfektions-Check: was braucht meine KI, und ist es eingeschaltet?
================================================================================

Der Anforderungs-Katalog beschreibt, was eine KI je nach Anwendung (Tabellen, Bilder, Segmentierung,
Sprache, Zeitreihen, Audio, RL, Entscheidungen über Menschen …) und Phase (Daten → Modell → Training →
Bewertung → Vertrauen → Einsatz) braucht. Der Perfektions-Check wendet ihn auf ein Projekt an:

* ✓ aktiv/eingebaut · ○ aus (mit Technik zum Einschalten) · ◇ offen (Schritt ausführen) · ✗ fehlt
* ``apply_perfection()`` schaltet alles Empfohlene ein – alles bleibt einzeln abwählbar (``disable``/``undo``)

Ausführen:  python examples/24_perfektions_check.py
"""

import warnings

from ki_baukasten import Project
from ki_baukasten.requirements import coverage

warnings.simplefilter("ignore")

# Abdeckung des Baukastens insgesamt (ehrlich: zwei Anforderungen fehlen)
cov = coverage()
print(f"Baukasten: {cov['vorhanden']} von {cov['gesamt']} Anforderungen umsetzbar")
print("Lücken:", [r.title for r, ok in cov["zeilen"] if not ok])

# Je Anwendung gelten andere Anforderungen – z. B. Zeitreihen: zeitliche Aufteilung ist Pflicht
p = Project.from_preset("zeitreihen_lstm", **{"training.max_steps": 80, "training.verbose": False,
                                              "deployment.show_after_training": False})
vorher = p.perfection_check()
print(vorher)

# Alles Empfohlene einschalten (nur Techniken aus dem Katalog – abwählbar)
an = p.apply_perfection()
print("\nEingeschaltet:", an)
for key in ("checkpoints", "log_datei"):          # für dieses Beispiel keine Dateien schreiben – einzeln abwählbar
    if key in an:
        p.disable(key)

# Offene Schritte ausführen: trainieren, testen, mit einer Basis vergleichen
p.train()
print("Test:", {k: round(v, 3) for k, v in p.test().items()})
print("Basis vs. Modell:", {k: round(v, 3) for k, v in p.baseline().items()})
nachher = p.perfection_check()
print(f"\nPunktzahl: {vorher.score:.0f} → {nachher.score:.0f} / 100")

# Ziele verändern die Anforderungen: Entscheidungen über Menschen → Fairness wird Pflicht
k = Project.from_preset("tabellen_klassifikation")
k.set("goals.description", "Kreditentscheidungen für Kunden")
fair = [c for c in k.perfection_check().checks if c.requirement.key == "fairness"][0]
print(f"\nFairness bei Kreditentscheidungen: {fair.requirement.importance} – {fair.status}: {fair.hint}")
