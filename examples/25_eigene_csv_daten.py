"""
Beispiel 25 – Eigene Tabellendaten: von der CSV zur ehrlich bewerteten KI
==========================================================================

* CSV mit Zahlen- **und** Kategorie-Spalten und fehlenden Werten – Vorverarbeitung nur auf Trainingszeilen gelernt
* Datenprüfung mit Abhilfen, stratifizierte Aufteilung, zurückgehaltene Testmenge, Klassengewichte
* Metriken über alle Beispiele (F1, AUROC, Kalibrierungsfehler …), Early Stopping auf val_f1 (Richtung automatisch)
* Basis-Vergleich, Fehleranalyse, Drift-Bericht für neue Daten

Ausführen:  python examples/25_eigene_csv_daten.py
"""

import csv
import random
import tempfile
import warnings
from pathlib import Path

import torch

from ki_baukasten import Project

warnings.simplefilter("ignore")
ordner = Path(tempfile.mkdtemp())
datei = ordner / "kunden.csv"

# Beispieldaten: wer kauft? (unausgewogen: ~20 % "ja", einige Lücken)
rng = random.Random(0)
with open(datei, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kunden_id", "alter", "stadt", "einkommen", "besuche", "kauft"])
    for i in range(1500):
        alter, stadt = rng.randint(18, 75), rng.choice(["Berlin", "Hamburg", "München", "Köln"])
        einkommen, besuche = rng.randint(1500, 7000), rng.randint(0, 12)
        punkte = (einkommen > 5000) + (besuche > 8) + (stadt == "München") + rng.random() * 0.8
        w.writerow([i, alter, stadt, "" if rng.random() < 0.05 else einkommen, besuche,
                    "ja" if punkte > 1.6 else "nein"])

p = Project.from_preset("tabellen_klassifikation", **{"training.verbose": False, "training.max_steps": 400,
                                                      "deployment.show_after_training": False})
p.set("data", {"name": "csv", "path": str(datei), "target": "kauft", "drop_columns": ["kunden_id"],
               "missing": "median", "scaling": "robust", "split": "stratifiziert"})
print("Merkmale:", p.data().info["features"])

bericht = p.data_check()
print(bericht)
print("Übernommen:", p.apply_data_fixes())          # z. B. Klassengewichte, Testmenge

p.set("training.callbacks", [{"name": "early_stopping", "monitor": "val_f1", "patience": 6}])
p.train()
print("Validierung:", {k: round(v, 3) for k, v in p.evaluate().items()})
print("Test:       ", {k: round(v, 3) for k, v in p.test().items()})
print("Basis:      ", {k: round(v, 3) for k, v in p.baseline().items() if k.startswith("basis_")})
fehler = p.error_analysis(top=3)
print("Konfusionsmatrix:", fehler["konfusionsmatrix"], "– sicherste Fehler:", fehler["sicherste_fehler"])

# Im Einsatz: neue Kunden sind deutlich reicher → Drift
neu = p.data().train["inputs"].clone()
spalte = p.data().info["features"].index("einkommen")
neu[:, spalte] += 1.5
print(p.drift_report({"inputs": neu}))
print("Vorhersage aus Rohdaten:", p.predict_raw([{"alter": 40, "stadt": "München", "einkommen": 6500,
                                                  "besuche": 10}])["vorhersage"])
