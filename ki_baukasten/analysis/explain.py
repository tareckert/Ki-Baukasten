"""
ki_baukasten.analysis.explain
=============================

**Erklärbarkeit** – *warum* hat die KI so entschieden? Wichtig für Vertrauen, Fehlersuche,
Datenlecks und Vorgaben (z. B. EU AI Act: Nachvollziehbarkeit bei Hochrisiko-Systemen).

========================  ==========================================================================
Verfahren                 Idee
========================  ==========================================================================
gradient (Saliency)       Gradient × Eingabe: wie stark ändert eine kleine Änderung die Ausgabe?
integrated_gradients      Gradienten entlang des Wegs von einer Basis (Null) zur Eingabe gemittelt –
                          erfüllt "Vollständigkeit" (Summe = Ausgabe − Basis-Ausgabe)
occlusion                 Teile (Bildfelder, Tokens, Merkmale) abdecken und den Effekt messen –
                          modellunabhängig, auch für Token-Eingaben
permutation               Merkmal über alle Beispiele mischen → wie viel schlechter wird die Metrik?
                          (globale Merkmals-Wichtigkeit)
========================  ==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

METHODS = {"auto": "gradient für Zahlen/Bilder, occlusion für Tokens", "gradient": "Gradient × Eingabe",
           "integrated_gradients": "Integrated Gradients", "occlusion": "Abdecken/Weglassen"}


def _score(out: torch.Tensor, target: Any) -> torch.Tensor:
    """Skalarer Wert je Beispiel: Logit der Zielklasse (bzw. Ausgabe bei Regression)."""
    if out.dim() == 1:
        return out
    if out.shape[-1] == 1:
        return out.reshape(out.shape[0], -1).sum(-1)
    if out.dim() == 3:                                   # Sprachmodell: letzte Position
        out = out[:, -1]
    if target is None:
        idx = out.argmax(-1)
    elif isinstance(target, torch.Tensor):
        idx = target.to(out.device).reshape(-1)
    else:
        idx = torch.full((out.shape[0],), int(target), device=out.device)
    return out.gather(-1, idx.reshape(-1, 1)).squeeze(-1)


@torch.no_grad()
def _fixed_target(forward: Callable, x: torch.Tensor, target: Any) -> Any:
    """Erklärte Klasse einmal festhalten (sonst wechselt sie beim Abdecken bzw. entlang des Pfads)."""
    if target is not None:
        return target
    out = forward(x)
    if out.dim() == 1 or out.shape[-1] == 1:
        return None
    return (out[:, -1] if out.dim() == 3 else out).argmax(-1)


def _need_float(x: torch.Tensor, method: str) -> None:
    if not x.is_floating_point():
        raise ValueError(f"'{method}' braucht Zahlen-Eingaben – für Tokens method='occlusion' verwenden.")


def gradient_x_input(forward: Callable, x: torch.Tensor, target: Any = None) -> torch.Tensor:
    _need_float(x, "gradient")
    x = x.detach().clone().float().requires_grad_(True)
    _score(forward(x), target).sum().backward()
    return (x.grad * x).detach()


def integrated_gradients(forward: Callable, x: torch.Tensor, target: Any = None, steps: int = 32,
                         baseline: torch.Tensor | None = None) -> torch.Tensor:
    _need_float(x, "integrated_gradients")
    target = _fixed_target(forward, x.float(), target)
    x = x.detach().float()
    base = torch.zeros_like(x) if baseline is None else baseline.to(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps).tolist():
        point = (base + a * (x - base)).requires_grad_(True)
        _score(forward(point), target).sum().backward()
        total += point.grad
    return ((x - base) * total / steps).detach()


@torch.no_grad()
def occlusion(forward: Callable, x: torch.Tensor, target: Any = None, patch: int = 4,
              fill: float | int = 0) -> torch.Tensor:
    """Wichtigkeit = Abfall des Zielwerts, wenn ein Teil abgedeckt wird (Tokens: durch Padding ersetzt)."""
    target_idx = _fixed_target(forward, x, target)
    ref = _score(forward(x), target_idx)
    attr = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    if x.dim() == 4:                                      # Bilder: Felder
        _, _, h, w = x.shape
        for i in range(0, h, patch):
            for j in range(0, w, patch):
                xo = x.clone()
                xo[:, :, i:i + patch, j:j + patch] = fill
                drop = ref - _score(forward(xo), target_idx)
                attr[:, :, i:i + patch, j:j + patch] = drop.reshape(-1, 1, 1, 1)
        return attr
    flat = x.reshape(x.shape[0], -1)
    for j in range(flat.shape[1]):                        # Tokens/Merkmale einzeln
        xo = flat.clone()
        xo[:, j] = fill
        attr.reshape(x.shape[0], -1)[:, j] = ref - _score(forward(xo.reshape(x.shape)), target_idx)
    return attr


@dataclass
class Explanation:
    """Erklärung einer Vorhersage."""

    method: str
    attributions: torch.Tensor
    names: list[str] = field(default_factory=list)
    prediction: Any = None
    target: Any = None

    def top(self, k: int = 10) -> list[tuple[str, float]]:
        a = self.attributions[0]
        if a.dim() == 3:                                  # Bild: je Pixel über Kanäle summiert → kein Name
            return []
        flat = a.reshape(-1)
        order = flat.abs().argsort(descending=True)[:k].tolist()
        names = self.names or [f"Merkmal {i}" for i in range(flat.numel())]
        return [(names[i] if i < len(names) else str(i), float(flat[i])) for i in order]

    def heatmap(self) -> torch.Tensor:
        """Bilder: Wichtigkeit je Pixel ``[H, W]`` (Betrag über Kanäle summiert, auf 0…1 normiert)."""
        a = self.attributions[0]
        if a.dim() != 3:
            raise ValueError("heatmap() nur für Bild-Eingaben.")
        m = a.abs().sum(0)
        return m / m.max().clamp_min(1e-12)

    def __str__(self) -> str:
        head = f"Erklärung ({self.method}) – Vorhersage: {self.prediction}"
        if self.target is not None:
            head += f", erklärt für: {self.target}"
        lines = [head]
        top = self.top()
        if not top:
            h = self.heatmap()
            ys, xs = divmod(int(h.reshape(-1).argmax()), h.shape[1])
            lines.append(f"  wichtigster Bildbereich um Zeile {ys}, Spalte {xs} (heatmap() für die ganze Karte)")
        width = max(len(n) for n, _ in top) if top else 0
        scale = max(abs(v) for _, v in top) if top else 1.0
        for name, v in top:
            bar = "█" * max(1, int(20 * abs(v) / (scale or 1.0)))
            lines.append(f"  {name:<{width}}  {'+' if v >= 0 else '−'}{abs(v):.4g}  {bar}")
        lines.append("  (+ spricht für die Vorhersage, − dagegen)")
        return "\n".join(lines)


@dataclass
class Importance:
    """Globale Merkmals-Wichtigkeit (Permutation)."""

    metric: str
    baseline: float
    rows: list[tuple[str, float, float]]          # (Name, Verschlechterung, Streuung)

    def __str__(self) -> str:
        lines = [f"Merkmals-Wichtigkeit (Permutation, Metrik {self.metric}, Ausgangswert {self.baseline:.4g}):"]
        width = max((len(r[0]) for r in self.rows), default=0)
        scale = max((abs(r[1]) for r in self.rows), default=1.0) or 1.0
        for name, drop, sd in self.rows:
            bar = "█" * max(0, int(25 * max(0.0, drop) / scale))
            lines.append(f"  {name:<{width}}  {drop:+.4g} ± {sd:.2g}  {bar}")
        lines.append("  (Verschlechterung, wenn das Merkmal zufällig gemischt wird – größer = wichtiger)")
        return "\n".join(lines)


@torch.no_grad()
def permutation_importance(score_fn: Callable[[torch.Tensor], float], x: torch.Tensor, higher_is_better: bool,
                           names: list[str] | None = None, repeats: int = 3, seed: int = 0,
                           metric: str = "") -> Importance:
    """``score_fn(x) -> Metrik`` auf allen Beispielen; Merkmale (Spalten von ``x``) einzeln mischen."""
    g = torch.Generator().manual_seed(seed)
    base = score_fn(x)
    rows = []
    flat = x.reshape(x.shape[0], -1)
    for j in range(flat.shape[1]):
        drops = []
        for _ in range(repeats):
            xp = flat.clone()
            xp[:, j] = flat[torch.randperm(flat.shape[0], generator=g), j]
            s = score_fn(xp.reshape(x.shape))
            drops.append(base - s if higher_is_better else s - base)
        d = torch.tensor(drops)
        rows.append((names[j] if names and j < len(names) else f"Merkmal {j}", float(d.mean()),
                     float(d.std()) if len(drops) > 1 else 0.0))
    rows.sort(key=lambda r: -r[1])
    return Importance(metric, base, rows)
