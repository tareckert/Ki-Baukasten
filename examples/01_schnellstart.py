"""
Beispiel 01 – Schnellstart
==========================

Zeigt den typischen Ablauf in fünf Schritten:

1. Vorlage (Preset) für ein Anwendungsgebiet laden
2. Einstellungen per Pfad ändern
3. Erklärungen (Tooltips) abrufen
4. trainieren
5. speichern, laden, benutzen

Ausführen:  python examples/01_schnellstart.py
"""

from ki_baukasten import Project

# 1) Vorlage laden: ein tiefes Residual-MLP für Tabellendaten (synthetische Spiralen).
projekt = Project.from_preset("tabellen_klassifikation")

# 2) Jede Einstellung ist über ihren Pfad erreichbar – auch tief verschachtelte.
projekt.set("model.hidden_dims", [128, 128, 128])          # Breite/Tiefe
projekt.set("model.layer.activation", "silu")               # Aktivierung aller Schichten …
projekt.set("model.layer_overrides", {"-1": {"dropout": 0.0}})  # … außer: letzte Schicht ohne Dropout
projekt.set("training.epochs", 40)
projekt.set("training.optimizer.lr", 5e-3)

# 3) Erklärungen: Was ist das? Warum/Nutzen? Wofür wichtig? Tipp?
print(projekt.explain("model.layer.residual"))
print()
print(projekt.describe(max_level="basic", only_changed=True))  # nur Abweichungen vom Standard

# 4) Trainieren (Modell wird automatisch gebaut; Ein-/Ausgabegrößen passen sich den Daten an).
projekt.train()
print("Ergebnis:", projekt.evaluate())

# 5) Speichern (Konfiguration als JSON + Gewichte) und wieder laden.
ordner = projekt.save("ausgaben/beispiel_01")
geladen = Project.load(ordner)
import torch  # noqa: E402  (nur für das Beispiel-Tensor unten)

vorhersage = geladen.predict(torch.tensor([[0.2, -0.4]]))
print("Vorhersage (Logits):", vorhersage)
