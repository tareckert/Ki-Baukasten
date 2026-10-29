"""
Beispiel 09 – Generative Modelle: GAN und Diffusion
===================================================

Beide lernen eine 2D-Verteilung und erzeugen danach neue Punkte.
Vergleich über Mittelwert und Streuung echter vs. erzeugter Daten.

Ausführen:  python examples/09_generativ.py
"""

from ki_baukasten import Project

for name in ["gan_2d", "diffusion_2d"]:
    projekt = Project.from_preset(name)
    projekt.set("training.verbose", False)
    projekt.train()
    echt = projekt.data().train["inputs"]
    neu = projekt.sample(1000).cpu()
    print(f"{name}: echt μ={echt.mean(0).tolist()} σ={echt.std(0).tolist()}")
    print(f"{' ' * len(name)}  neu  μ={neu.mean(0).tolist()} σ={neu.std(0).tolist()}")
