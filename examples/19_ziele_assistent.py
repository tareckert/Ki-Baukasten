"""
Beispiel 19 – Ziel-Assistent: Wofür soll die KI genutzt werden?
==============================================================

Aufgabe, wichtige Punkte und Zielgerät angeben → der Assistent empfiehlt passende Modelle
(mit Begründung), rät von unpassenden ab, schaltet passende Techniken ein und legt das Projekt
an. Die Ziele bleiben im Projekt gespeichert und können später geändert werden.

Dasselbe in der GUI: "✚ Neue KI nach Zielen…" bzw. Menü Assistent → "Ziele prüfen / ändern…".
Kommandozeile:  python -m ki_baukasten ziele --text "…" --erstellen projekt.json

Ausführen:  python examples/19_ziele_assistent.py
"""

import warnings

from ki_baukasten import Project
from ki_baukasten.analysis.goals import evaluate_goals

warnings.simplefilter("ignore")

# 1) Nur bewerten (ohne Projekt): Freitext wird ausgewertet (Handy → Zielgerät + wenig Speicher/Energie …)
bericht = evaluate_goals({"description": "Ich möchte Pflanzenkrankheiten per Handykamera in Echtzeit erkennen.",
                          "data_size": "klein", "priorities": ["robustheit"]})
print(bericht)

# 2) Projekt direkt aus den Zielen anlegen – beste Empfehlung, passende Techniken eingeschaltet
p = Project.from_goals(task="text_klassifizieren", description="Spam-Filter direkt auf dem Smartphone",
                       priorities=["geschwindigkeit", "datenschutz"], data_size="klein")
print("\nGewählte Vorlage:", p.config.name, "| Zielgerät:", p.config.deployment.main_target)
print("Eingeschaltete Techniken:", [r.technique.key for r in p.techniques(status="an")])
print("Empfohlen, aber aufwendig (◇, nicht automatisch an):",
      [k for k, _ in p.goal_report.techniques if k in ("dp_sgd", "foederiert")])

# 3) Später: Ziele ändern und das aktuelle Modell neu bewerten lassen
neu = p.set_goals(task="text_erzeugen", sequence_length="sehr_lang", priorities=["lange_kontexte"])
print("\nNach der Änderung:", neu.current_verdict)
print("Jetzt empfohlen:", [s.preset for s in neu.recommended])
print("Abgeraten:", [(s.preset, [r for r in s.reasons if r.startswith("✗")][:1]) for s in neu.discouraged][:3])

# 4) Neues Projekt mit den geänderten Zielen (z. B. Mamba für sehr lange Texte)
q = Project.from_goals(p.config.goals)
print("\nNeues Projekt nach geänderten Zielen:", q.config.name, "→ Modell", q.config.model.name)
