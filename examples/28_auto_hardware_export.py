"""
Beispiel 28 – Von den Daten zur fertigen KI, Hardware-Automatik und schlanker Export
====================================================================================

* ``Project.auto``: Datenart erkennen → Vorlage → Empfehlungen → Training → Test → Basis-Vergleich
* ``hardware_report``: welches Gerät erkannt wurde und was automatisch gewählt ist
* Tempo: ein Trainingsschritt des Baukastens gegen eine handgeschriebene PyTorch-Schleife
* ``export``: ohne Trainings-Ballast, geprüft gegen das Original

Ausführen:  python examples/28_auto_hardware_export.py
"""

import csv
import random
import tempfile
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from ki_baukasten import Project, Trainer

warnings.simplefilter("ignore")
ordner = Path(tempfile.mkdtemp())

# --- 1. Ein Aufruf: CSV → fertige KI --------------------------------------------------------------
rng = random.Random(0)
with open(ordner / "preise.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["flaeche", "zimmer", "lage", "preis"])
    for _ in range(800):
        fl, lage = rng.uniform(20, 200), rng.choice(["zentral", "rand", "land"])
        faktor = {"zentral": 4000, "rand": 3000, "land": 2000}[lage]
        w.writerow([round(fl, 1), rng.randint(1, 6), lage, round(fl * faktor + rng.gauss(0, 10000))])

p = Project.auto(ordner / "preise.csv", steps=400, **{"training.verbose": False,
                                                         "deployment.show_after_training": False})
print("Vorlage:", p.config.name, "· Test:", {k: round(v, 3) for k, v in p.last_test.items()})
print("Basis (Mittelwert) vs. Modell, MAE:", round(p.last_baseline["basis_mae"]), "→",
      round(p.last_baseline["modell_mae"]))
print("Vorhersage 100 m², 3 Zimmer, zentral:",
      p.predict_raw([{"flaeche": 100, "zimmer": 3, "lage": "zentral"}])["vorhersage"][0], "(≈ 400.000)")

# --- 2. Hardware ------------------------------------------------------------------------------------
print()
print(p.hardware_report())

# --- 3. Tempo: Baukasten-Schritt vs. reines PyTorch (gleiches Modell, gleicher Batch) -------------
lm = Project.from_preset("sprachmodell_mini")
lm.set("model.n_layers", 2)
lm.config.training.verbose = False
lm.build()
tr = Trainer(lm.model, lm.config.training, lm.config.memory)
tr._build_schedulers(1000)
x = lm.data().train["inputs"][:32]
opt = torch.optim.AdamW(lm.model.parameters(), lr=1e-3, fused=True)


def von_hand():
    out = lm.model(x[:, :-1])
    loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lm.model.parameters(), 1.0)
    opt.step()


zeiten: dict[str, list[float]] = {"PyTorch von Hand": [], "Baukasten": []}
for _ in range(5):                       # abwechselnd messen, Median – einzelne Messungen schwanken stark
    for name, fn in (("PyTorch von Hand", von_hand), ("Baukasten", lambda: tr._train_step({"inputs": x}))):
        for _ in range(3):
            fn()
        t = time.perf_counter()
        for _ in range(20):
            fn()
        zeiten[name].append((time.perf_counter() - t) / 20 * 1000)
for name, werte in zeiten.items():
    print(f"{name:<18} {sorted(werte)[len(werte) // 2]:6.1f} ms/Schritt (Median)")

# --- 4. Schlanker Export -------------------------------------------------------------------------
gan = Project.from_preset("gan_2d", **{"training.max_steps": 50, "training.verbose": False,
                                       "deployment.show_after_training": False})
gan.train()
gan.export(ordner / "generator", "export")
print()
print(gan.last_export)                 # Diskriminator wird für den Einsatz nicht gebraucht → entfernt
