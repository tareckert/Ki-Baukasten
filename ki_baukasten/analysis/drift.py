"""
ki_baukasten.analysis.drift
===========================

**Drift-Erkennung** – ändern sich die Daten im Einsatz, wird die KI unbemerkt schlechter.
``drift_report`` vergleicht neue Daten mit den Trainingsdaten (Referenz):

==========================  ==========================================================================
Maß                         Bedeutung
==========================  ==========================================================================
PSI (Population Stability   Verschiebung der Verteilung je Merkmal: < 0.1 stabil, 0.1–0.25 mäßig,
Index)                      > 0.25 deutlich (Branchenstandard bei Banken/Versicherungen)
KS (Kolmogorow-Smirnow)     größter Abstand der Verteilungsfunktionen (0 … 1)
Mittelwert-Verschiebung     Abstand der Mittelwerte in Standardabweichungen der Referenz
Token-Verteilung            PSI über die Häufigkeit der Tokens (Texte)
Kanal-Statistik             Mittel/Std je Farbkanal (Bilder)
Vorhersage-Drift            Verteilung der vorhergesagten Klassen bzw. Werte (mit Modell)
==========================  ==========================================================================

Typische Abhilfe bei Drift: neu trainieren (``Project.continue_training`` mit den neuen Daten).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

STABLE, MODERATE = 0.1, 0.25


def _psi(ref: torch.Tensor, cur: torch.Tensor, bins: int = 10) -> float:
    """PSI mit Quantil-Grenzen der Referenz."""
    ref, cur = ref.float(), cur.float()
    qs = torch.quantile(ref, torch.linspace(0, 1, bins + 1)[1:-1]) if ref.numel() > 1 else ref[:0]
    edges = torch.unique(qs)
    a = torch.bincount(torch.bucketize(ref, edges), minlength=len(edges) + 1).float()
    b = torch.bincount(torch.bucketize(cur, edges), minlength=len(edges) + 1).float()
    return _psi_counts(a, b)


def _psi_counts(a: torch.Tensor, b: torch.Tensor) -> float:
    eps = 1e-4
    pa = (a / a.sum().clamp_min(1)).clamp_min(eps)
    pb = (b / b.sum().clamp_min(1)).clamp_min(eps)
    return float(((pb - pa) * (pb / pa).log()).sum())


def _ks(ref: torch.Tensor, cur: torch.Tensor) -> float:
    ref, cur = ref.float().sort().values, cur.float().sort().values
    grid = torch.cat([ref, cur]).unique()
    cdf_a = torch.searchsorted(ref, grid, right=True).float() / max(1, len(ref))
    cdf_b = torch.searchsorted(cur, grid, right=True).float() / max(1, len(cur))
    return float((cdf_a - cdf_b).abs().max()) if len(grid) else 0.0


def _verdict(psi: float) -> str:
    return "stabil" if psi < STABLE else ("mäßig" if psi < MODERATE else "deutlich")


def feature_drift(ref: torch.Tensor, cur: torch.Tensor, names: list[str] | None = None,
                  max_features: int = 500) -> list[dict[str, Any]]:
    """Drift je Merkmal für Tabellen ``[N, D]``."""
    ref, cur = ref.reshape(ref.shape[0], -1).float(), cur.reshape(cur.shape[0], -1).float()
    out = []
    for j in range(min(ref.shape[1], max_features)):
        r, c = ref[:, j], cur[:, j]
        r, c = r[torch.isfinite(r)], c[torch.isfinite(c)]
        if not len(r) or not len(c):
            continue
        psi = _psi(r, c)
        sd = float(r.std()) if len(r) > 1 else 0.0
        out.append({"merkmal": names[j] if names and j < len(names) else f"Merkmal {j}", "psi": psi, "ks": _ks(r, c),
                    "verschiebung_std": (float(c.mean()) - float(r.mean())) / sd if sd > 0 else 0.0,
                    "bewertung": _verdict(psi)})
    return out


@dataclass
class DriftReport:
    """Ergebnis von ``drift_report``."""

    kind: str                                   # tabelle | tokens | bilder | allgemein
    features: list[dict[str, Any]] = field(default_factory=list)
    overall_psi: float = 0.0
    prediction: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        worst = max([self.overall_psi] + [d["psi"] for d in self.features] +
                    ([self.prediction["psi"]] if self.prediction else []))
        return _verdict(worst)

    @property
    def drifted(self) -> list[dict[str, Any]]:
        return [d for d in self.features if d["psi"] >= MODERATE]

    def __str__(self) -> str:
        lines = [f"Drift-Bericht ({self.kind}): Gesamturteil {self.verdict.upper()}  (PSI gesamt {self.overall_psi:.3f})"]
        if self.features:
            lines.append(f"{'Merkmal':<28} {'PSI':>7} {'KS':>6} {'Δ Mittel (σ)':>13}  Bewertung")
            for d in sorted(self.features, key=lambda d: -d["psi"])[:15]:
                lines.append(f"{d['merkmal'][:28]:<28} {d['psi']:>7.3f} {d['ks']:>6.3f} {d['verschiebung_std']:>13.2f}  "
                             f"{d['bewertung']}")
            if len(self.features) > 15:
                lines.append(f"… und {len(self.features) - 15} weitere Merkmale")
        if self.prediction:
            p = self.prediction
            lines.append(f"Vorhersage-Drift: PSI {p['psi']:.3f} ({_verdict(p['psi'])}) – {p['beschreibung']}")
        lines += [f"ℹ {n}" for n in self.notes]
        if self.verdict != "stabil":
            lines.append("→ Neu trainieren oder weiterlernen: project.continue_training(neue_daten, method='lwf'); "
                         "vorher Ursache prüfen (Messfehler? neue Kundengruppe? Saison?).")
        return "\n".join(lines)


def _inputs(data: Any) -> Any:
    from .datacheck import _tensors

    x, _, _ = _tensors(data)
    return x


def drift_report(reference: Any, current: Any, names: list[str] | None = None, model: Any = None,
                 predict: Any = None) -> DriftReport:
    """Vergleicht ``current`` (neue Daten) mit ``reference`` (Trainingsdaten).

    ``predict``: optional ``f(inputs) -> Ausgabe`` für die Vorhersage-Drift.
    """
    ref, cur = _inputs(reference), _inputs(current)
    if not isinstance(ref, torch.Tensor) or not isinstance(cur, torch.Tensor):
        return DriftReport("allgemein", notes=["Daten liegen nicht als Tensoren vor – kein Vergleich möglich."])
    if not ref.is_floating_point() and ref.dim() == 2:          # Tokens
        vocab = int(max(ref.max(), cur.max())) + 1
        a = torch.bincount(ref.reshape(-1).clamp_min(0), minlength=vocab).float()
        b = torch.bincount(cur.reshape(-1).clamp_min(0), minlength=vocab).float()
        a[0] = b[0] = 0                                          # Padding ignorieren
        rep = DriftReport("tokens", overall_psi=_psi_counts(a, b))
        unk_a, unk_b = float(a[1] / a.sum().clamp_min(1)), float(b[1] / b.sum().clamp_min(1))
        if unk_b > unk_a + 0.02:
            rep.notes.append(f"Anteil unbekannter Tokens gestiegen: {unk_a:.1%} → {unk_b:.1%} (neues Vokabular?).")
        len_a, len_b = float((ref != 0).sum(1).float().mean()), float((cur != 0).sum(1).float().mean())
        rep.notes.append(f"Mittlere Länge: {len_a:.1f} → {len_b:.1f} Tokens.")
    elif ref.is_floating_point() and ref.dim() == 4:            # Bilder: je Kanal
        feats = []
        for ch in range(ref.shape[1]):
            r, c = ref[:, ch].reshape(-1), cur[:, ch].reshape(-1)
            r = r[torch.randperm(len(r), generator=torch.Generator().manual_seed(0))[:200000]]
            c = c[torch.randperm(len(c), generator=torch.Generator().manual_seed(1))[:200000]]
            psi = _psi(r, c)
            feats.append({"merkmal": f"Kanal {ch}", "psi": psi, "ks": _ks(r, c),
                          "verschiebung_std": (float(c.mean()) - float(r.mean())) / max(1e-12, float(r.std())),
                          "bewertung": _verdict(psi)})
        rep = DriftReport("bilder", feats, overall_psi=max(d["psi"] for d in feats))
    else:
        feats = feature_drift(ref, cur, names)
        overall = sum(d["psi"] for d in feats) / max(1, len(feats))
        rep = DriftReport("tabelle", feats, overall_psi=overall)
    if predict is not None:
        rep.prediction = prediction_drift(predict, ref, cur)
    return rep


@torch.no_grad()
def prediction_drift(predict: Any, ref: torch.Tensor, cur: torch.Tensor, limit: int = 2000) -> dict[str, Any]:
    from ..training.paradigms import main_output

    a = main_output(predict(ref[:limit])).float().cpu()
    b = main_output(predict(cur[:limit])).float().cpu()
    if a.dim() == 2 and a.shape[1] > 1:
        k = a.shape[1]
        ca = torch.bincount(a.argmax(1), minlength=k).float()
        cb = torch.bincount(b.argmax(1), minlength=k).float()
        share_a, share_b = ca / ca.sum(), cb / cb.sum()
        top = int((share_b - share_a).abs().argmax())
        return {"psi": _psi_counts(ca, cb), "anteile_referenz": share_a.tolist(), "anteile_neu": share_b.tolist(),
                "beschreibung": f"stärkste Änderung bei Klasse {top}: {float(share_a[top]):.0%} → "
                                f"{float(share_b[top]):.0%}"}
    a, b = a.reshape(-1), b.reshape(-1)
    return {"psi": _psi(a, b), "beschreibung": f"Mittel der Vorhersagen {float(a.mean()):.4g} → {float(b.mean()):.4g}"}
