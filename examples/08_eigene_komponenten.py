"""
Beispiel 08 – Eigene Komponenten registrieren
=============================================

Alles im Baukasten ist erweiterbar. Eine eigene Komponente wird einmal
registriert und ist danach überall per Name nutzbar – in JSON-Dateien, in der
GUI, in der CLI, mit Tooltips.

Hier: eine eigene Aktivierungsfunktion und eine eigene Verlustfunktion.

Ausführen:  python examples/08_eigene_komponenten.py
"""

import torch
from torch import nn

from ki_baukasten import BaseConfig, Level, Param, Project, T, registry


class WelleConfig(BaseConfig):
    frequenz = Param(2.0, min=0.0, tooltip=T(
        "Frequenz der Welle.", "Steuert, wie oft die Funktion oszilliert.", "Experimente mit periodischen Daten."))


class Welle(nn.Module):
    """x + sin(f·x)/f – eine eigene, glatte Aktivierung."""

    def __init__(self, f: float):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + torch.sin(self.f * x) / self.f


registry("activation").register("welle", lambda a: Welle(a.frequenz), config=WelleConfig, level=Level.EXPERIMENTAL,
                                override=True, tooltip=T("Eigene Wellen-Aktivierung.", "Demonstriert Erweiterbarkeit.",
                                                         "Beispiel 08."))


class AsymmetrischConfig(BaseConfig):
    weight = Param(1.0, tooltip=T("Gewicht.", "Im Gesamtverlust.", "1."))
    output_key = Param(None, type=str, tooltip=T("Ausgabeschlüssel.", "Multi-Task.", "None."))
    target_key = Param(None, type=str, tooltip=T("Zielschlüssel.", "Multi-Task.", "None."))
    unterschaetzung = Param(3.0, min=0.0, tooltip=T(
        "Strafe für Unterschätzung relativ zu Überschätzung.", "Asymmetrische Kosten (z. B. Lagerbestand).", "2–5."))


class Asymmetrisch(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, pred, target):
        diff = target.float().reshape(pred.shape) - pred.float()
        return torch.where(diff > 0, self.a.unterschaetzung * diff, -diff).mean()


registry("loss").register("asymmetrisch", lambda a: Asymmetrisch(a), config=AsymmetrischConfig, override=True,
                          tooltip=T("Asymmetrischer absoluter Fehler.", "Unterschätzung teurer als Überschätzung.",
                                    "Nachfrageprognosen."))

projekt = Project.from_preset("tabellen_regression")
projekt.update({"model.layer.activation": {"name": "welle", "args": {"frequenz": 3.0}},
                "training.losses": [{"name": "asymmetrisch", "args": {"unterschaetzung": 4.0}}],
                "training.epochs": 10})
print(projekt.explain("model.layer.activation"))
projekt.train()
