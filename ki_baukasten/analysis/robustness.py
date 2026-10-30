"""
ki_baukasten.analysis.robustness
================================

**Robustheit** – hält die KI kleinen Störungen stand?

=====================  =============================================================================
Prüfung                Bedeutung
=====================  =============================================================================
fgsm                   Angriff in Richtung des Gradienten (ein Schritt) – zeigt Anfälligkeit sofort
pgd                    mehrstufiger Angriff innerhalb einer ε-Kugel – Maßstab für echte Robustheit
rauschen               Gaußsches Rauschen – Robustheit gegen Messfehler/Sensorrauschen
=====================  =============================================================================

Dieselben Angriffe erzeugen im Training die Beispiele für ``training.adversarial`` (FGSM/PGD-Training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


def attack(forward: Callable[[torch.Tensor], torch.Tensor], loss_fn: Callable[[torch.Tensor, Any], torch.Tensor],
           x: torch.Tensor, y: Any, method: str = "fgsm", epsilon: float = 0.03, steps: int = 5,
           step_size: float | None = None, seed: int = 0) -> torch.Tensor:
    """Erzeugt gestörte Eingaben (L∞, Abstand höchstens ``epsilon``), die den Verlust maximieren."""
    if not x.is_floating_point():
        raise ValueError("Angriffe brauchen Zahlen-Eingaben (keine Tokens).")
    x0 = x.detach()
    if method == "fgsm":
        xa = x0.clone().requires_grad_(True)
        grad, = torch.autograd.grad(loss_fn(forward(xa), y), xa)
        return (x0 + epsilon * grad.sign()).detach()
    if method != "pgd":
        raise ValueError(f"Unbekannter Angriff '{method}' (fgsm, pgd).")
    alpha = step_size or 2.5 * epsilon / steps
    g = torch.Generator(device="cpu").manual_seed(seed)
    delta = (torch.rand(x0.shape, generator=g) * 2 - 1).to(x0) * epsilon       # zufälliger Start
    for _ in range(steps):
        xa = (x0 + delta).requires_grad_(True)
        grad, = torch.autograd.grad(loss_fn(forward(xa), y), xa)
        delta = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon).detach()
    return (x0 + delta).detach()


@dataclass
class RobustnessReport:
    metric: str
    clean: float
    rows: list[tuple[str, float, float]]           # (Angriff, Stärke, Wert)

    def __str__(self) -> str:
        lines = [f"Robustheit ({self.metric}, sauber: {self.clean:.4g}):",
                 f"  {'Störung':<10} {'Stärke':>8} {self.metric:>12} {'Abfall':>8}"]
        for kind, eps, value in self.rows:
            lines.append(f"  {kind:<10} {eps:>8.3g} {value:>12.4g} {self.clean - value:>+8.3g}")
        worst = min((v for _, _, v in self.rows), default=self.clean)
        if self.clean - worst > 0.2 * abs(self.clean):
            lines.append("→ Anfällig: training.adversarial (fgsm/pgd) oder Augmentierung/Rauschen einschalten "
                         "(project.enable('adversarial_training')).")
        return "\n".join(lines)


def robustness_report(forward: Callable, loss_fn: Callable, metric_fn: Callable, x: torch.Tensor, y: Any,
                      epsilons: list[float], metric: str, methods: tuple[str, ...] = ("rauschen", "fgsm", "pgd"),
                      steps: int = 5) -> RobustnessReport:
    with torch.no_grad():
        clean = metric_fn(forward(x), y)
    rows = []
    std = float(x.float().std()) or 1.0
    g = torch.Generator(device="cpu").manual_seed(0)
    for eps in epsilons:
        for m in methods:
            if m == "rauschen":
                xa = x + (torch.randn(x.shape, generator=g) * eps * std).to(x)
            else:
                xa = attack(forward, loss_fn, x, y, m, eps * std, steps)
            with torch.no_grad():
                rows.append((m, eps, metric_fn(forward(xa), y)))
    return RobustnessReport(metric, clean, rows)
