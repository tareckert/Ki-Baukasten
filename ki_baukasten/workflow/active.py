"""
ki_baukasten.workflow.active
============================

**Mit wenigen Labels weit kommen** – Active Learning und semi-überwachtes Lernen.

=================  ===============================================================================
Verfahren          Idee
=================  ===============================================================================
Active Learning    Die KI schlägt vor, *welche* Beispiele Menschen als Nächstes beschriften sollen –
                   die informativsten statt zufälliger (spart oft 50–90 % Beschriftungsaufwand).
Pseudo-Labeling    Sichere Vorhersagen auf unbeschrifteten Daten werden als Labels übernommen und
                   mittrainiert (Self-Training, Lee 2013; Basis von FixMatch/Noisy Student).
=================  ===============================================================================

Auswahl-Strategien (``select``):

=============  ===================================================================================
unsicherheit   geringste Konfidenz der besten Klasse
marge          kleinster Abstand zwischen den zwei besten Klassen
entropie       höchste Entropie der Wahrscheinlichkeiten
vielfalt       k-Center: möglichst unterschiedliche Beispiele (deckt den Datenraum ab)
zufall         Vergleichsbasis
=============  ===================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

STRATEGIES = {"unsicherheit": "geringste Konfidenz", "marge": "kleinster Abstand der zwei besten Klassen",
              "entropie": "höchste Entropie", "vielfalt": "k-Center (möglichst unterschiedliche Beispiele)",
              "zufall": "zufällig (Vergleich)"}


def informativeness(probs: torch.Tensor, strategy: str) -> torch.Tensor:
    """Je größer, desto lohnender ist es, das Beispiel zu beschriften."""
    if strategy == "unsicherheit":
        return 1 - probs.max(-1).values
    if strategy == "marge":
        top = probs.topk(min(2, probs.shape[-1]), -1).values
        return -(top[:, 0] - top[:, -1])
    if strategy == "entropie":
        return -(probs.clamp_min(1e-12).log() * probs).sum(-1)
    raise ValueError(f"Unbekannte Strategie '{strategy}'. Erlaubt: {', '.join(STRATEGIES)}.")


def k_center(features: torch.Tensor, n: int, seed: int = 0) -> list[int]:
    """Greedy k-Center: wählt nacheinander das Beispiel, das am weitesten von allen gewählten entfernt ist."""
    f = features.reshape(features.shape[0], -1).float()
    g = torch.Generator().manual_seed(seed)
    chosen = [int(torch.randint(0, f.shape[0], (1,), generator=g))]
    dist = (f - f[chosen[0]]).norm(dim=1)
    while len(chosen) < min(n, f.shape[0]):
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = torch.minimum(dist, (f - f[nxt]).norm(dim=1))
    return chosen


@dataclass
class Selection:
    strategy: str
    indices: list[int]
    scores: list[float]
    predictions: list[Any]

    def __str__(self) -> str:
        lines = [f"Zum Beschriften vorgeschlagen ({self.strategy}, {len(self.indices)} Beispiele):",
                 f"  {'Index':>6} {'Wert':>8}  aktuelle Vorhersage"]
        for i, s, p in list(zip(self.indices, self.scores, self.predictions))[:25]:
            lines.append(f"  {i:>6} {s:>8.3f}  {p}")
        if len(self.indices) > 25:
            lines.append(f"  … und {len(self.indices) - 25} weitere")
        return "\n".join(lines)


def select(probs: torch.Tensor, n: int, strategy: str = "unsicherheit", features: torch.Tensor | None = None,
           seed: int = 0) -> tuple[list[int], list[float]]:
    n = min(n, probs.shape[0])
    if strategy == "zufall":
        idx = torch.randperm(probs.shape[0], generator=torch.Generator().manual_seed(seed))[:n].tolist()
        return idx, [0.0] * n
    if strategy == "vielfalt":
        idx = k_center(features if features is not None else probs, n, seed)
        return idx, informativeness(probs, "entropie")[idx].tolist()
    scores = informativeness(probs, strategy)
    order = scores.argsort(descending=True)[:n]
    return order.tolist(), scores[order].tolist()


def pseudo_labels(probs: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(Maske der sicheren Beispiele, Labels) – nur Vorhersagen mit Konfidenz ≥ ``threshold``."""
    conf, labels = probs.max(-1)
    return conf >= threshold, labels
