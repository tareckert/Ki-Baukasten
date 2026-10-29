"""
ki_baukasten.analysis.fairness
==============================

**Fairness-Bericht** – behandelt die KI Gruppen (z. B. Geschlecht, Alter, Region) gleich?

=============================  =====================================================================
Kennzahl                       Bedeutung
=============================  =====================================================================
Genauigkeit je Gruppe          Ist das Modell für manche Gruppen schlechter?
Positiv-Rate je Gruppe         Wie oft bekommt eine Gruppe die positive Entscheidung?
Demographic-Parity-Differenz   größter Unterschied der Positiv-Raten (0 = gleich)
Disparate-Impact-Verhältnis    kleinste / größte Positiv-Rate (< 0.8 verletzt die "4/5-Regel")
Equalized-Odds-Differenz       größter Unterschied der Trefferquote (TPR) bzw. Fehlalarmquote (FPR)
=============================  =====================================================================

Abhilfen: ausgewogenere Daten je Gruppe, Klassengewichte, Schwellen je Gruppe prüfen, sensible
Merkmale (und ihre Stellvertreter) entfernen – und immer fachlich bewerten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FairnessReport:
    groups: dict[str, dict[str, float]]
    parity_difference: float
    disparate_impact: float
    equalized_odds_difference: float
    positive_class: int

    @property
    def fair(self) -> bool:
        return self.disparate_impact >= 0.8 and self.equalized_odds_difference <= 0.1

    def __str__(self) -> str:
        lines = [f"Fairness-Bericht (positive Klasse {self.positive_class}):",
                 f"{'Gruppe':<20} {'n':>6} {'Genauigk.':>10} {'Positiv-Rate':>13} {'TPR':>7} {'FPR':>7}"]
        for name, g in self.groups.items():
            lines.append(f"{name[:20]:<20} {int(g['n']):>6} {g['accuracy']:>10.3f} {g['positive_rate']:>13.3f} "
                         f"{g['tpr']:>7.3f} {g['fpr']:>7.3f}")
        lines += [f"Demographic-Parity-Differenz: {self.parity_difference:.3f}",
                  f"Disparate-Impact-Verhältnis:  {self.disparate_impact:.3f}"
                  + ("  ⚠ unter 0.8 (4/5-Regel)" if self.disparate_impact < 0.8 else ""),
                  f"Equalized-Odds-Differenz:     {self.equalized_odds_difference:.3f}"
                  + ("  ⚠ über 0.1" if self.equalized_odds_difference > 0.1 else ""),
                  "→ " + ("keine auffälligen Unterschiede." if self.fair else
                          "Unterschiede prüfen: Daten je Gruppe ausgleichen, Klassengewichte, sensible Merkmale und "
                          "Stellvertreter entfernen, Schwellen bewerten.")]
        return "\n".join(lines)


def fairness_report(pred: torch.Tensor, target: torch.Tensor, groups: Any, positive_class: int = 1,
                    names: dict | None = None) -> FairnessReport:
    """``pred``/``target``: Klassen ``[N]``; ``groups``: Gruppen-Kennung je Beispiel (Zahlen oder Texte)."""
    pred, target = pred.reshape(-1).long(), target.reshape(-1).long()
    keys = groups.tolist() if isinstance(groups, torch.Tensor) else list(groups)
    out: dict[str, dict[str, float]] = {}
    for key in sorted(set(keys), key=str):
        m = torch.tensor([k == key for k in keys])
        p, t = pred[m], target[m]
        pos, neg = t == positive_class, t != positive_class
        hit = p == positive_class
        out[str(names.get(key, key) if names else key)] = {
            "n": float(m.sum()), "accuracy": float((p == t).float().mean()),
            "positive_rate": float(hit.float().mean()),
            "tpr": float(hit[pos].float().mean()) if pos.any() else float("nan"),
            "fpr": float(hit[neg].float().mean()) if neg.any() else float("nan")}
    rates = [g["positive_rate"] for g in out.values()]
    tprs = [g["tpr"] for g in out.values() if g["tpr"] == g["tpr"]]
    fprs = [g["fpr"] for g in out.values() if g["fpr"] == g["fpr"]]
    spread = (lambda v: max(v) - min(v) if v else 0.0)
    return FairnessReport(out, spread(rates), (min(rates) / max(rates)) if rates and max(rates) > 0 else 1.0,
                          max(spread(tprs), spread(fprs)), positive_class)


def groups_from_features(x: torch.Tensor, feature_names: list[str], column: str) -> tuple[list[Any], dict]:
    """Gruppen aus einer (CSV-)Spalte: One-Hot-Kategorien ``spalte=wert`` oder Zahl (≤/> Median)."""
    onehot = [i for i, n in enumerate(feature_names) if n.startswith(f"{column}=")]
    if onehot:
        idx = x[:, onehot].argmax(1).tolist()
        labels = {i: feature_names[onehot[i]].split("=", 1)[1] for i in range(len(onehot))}
        return idx, labels
    if column in feature_names:
        col = x[:, feature_names.index(column)]
        med = float(col.median())
        return (col > med).long().tolist(), {0: f"{column} ≤ Median", 1: f"{column} > Median"}
    raise KeyError(f"Spalte '{column}' nicht unter den Merkmalen ({', '.join(feature_names[:10])} …).")
