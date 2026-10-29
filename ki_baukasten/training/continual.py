"""
ki_baukasten.training.continual
===============================

**Weiterlernen ohne Vergessen** und **Schüler-Modelle**.

Wenn ein trainiertes Modell neue Daten lernt, überschreibt es leicht altes Wissen
("katastrophales Vergessen"). Gegenmittel:

==========  ===========================================================================
Methode     Prinzip
==========  ===========================================================================
``lwf``     Learning without Forgetting: die *alte* Modellversion ist Lehrer – das neue
            Modell lernt die neuen Daten und bleibt gleichzeitig nah an den alten Ausgaben
``ewc``     Elastic Weight Consolidation: Gewichte, die für alte Daten wichtig waren
            (Fisher-Information), werden per Strafterm festgehalten
``replay``  ein Anteil alter Beispiele wird den neuen Daten beigemischt
``none``    normales Weitertrainieren (Vergleich)
==========  ===========================================================================

``shrink_config`` erzeugt aus einer Modell-Konfiguration eine kleinere Schüler-
Konfiguration (Breite/Tiefe skaliert) für ``Project.distill``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import torch
from torch import nn

from ..core import ConfigError

METHODS = {
    "lwf": "alte Modellversion als Lehrer (Distillation auf neuen Daten)",
    "ewc": "wichtige Gewichte per Fisher-Strafterm festhalten",
    "replay": "alte Beispiele beimischen",
    "none": "normal weitertrainieren",
}


def fisher_information(model: nn.Module, batches: list[Any], loss_fn: Callable[[nn.Module, Any], torch.Tensor]
                       ) -> dict[str, torch.Tensor]:
    """Diagonale (empirische) Fisher-Information: mittleres Quadrat der Gradienten pro Gewicht."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    was = model.training
    model.eval()
    try:
        for b in batches:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss_fn(model, b).backward()
            for n, p in model.named_parameters():
                if n in fisher and p.grad is not None:
                    fisher[n] += p.grad.detach().square() / len(batches)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was)
    return fisher


def ewc_penalty(model: nn.Module, fisher: dict[str, torch.Tensor], strength: float) -> Callable[[], torch.Tensor]:
    """Strafterm λ/2 · Σ F·(θ − θ_alt)² als Funktion (für ``trainer.penalties``)."""
    anchors = {n: p.detach().clone() for n, p in model.named_parameters() if n in fisher}
    # Normieren, damit 'strength' unabhängig von der Modellgröße ein ähnliches Gewicht hat.
    scale = 1.0 / max(1e-12, float(sum(f.sum() for f in fisher.values()) / max(1, sum(f.numel() for f in fisher.values()))))
    params = dict(model.named_parameters())

    def penalty() -> torch.Tensor:
        total = None
        for n, f in fisher.items():
            p = params[n]
            term = (f.to(p.device) * (p - anchors[n].to(p.device)).square()).sum()
            total = term if total is None else total + term
        return 0.5 * strength * scale * total / max(1, len(fisher)) if total is not None else torch.zeros(())
    return penalty


def mix_replay(new: dict[str, torch.Tensor], old: dict[str, torch.Tensor], fraction: float,
               seed: int = 0) -> dict[str, torch.Tensor]:
    """Mischt ``fraction`` · len(new) zufällige alte Beispiele unter die neuen (gleiche Schlüssel)."""
    if not isinstance(new, dict) or not isinstance(old, dict):
        raise ConfigError("replay braucht dict-Datensätze mit gleichen Schlüsseln (z. B. inputs/targets).")
    keys = [k for k in new if k in old and isinstance(new[k], torch.Tensor)]
    n_new = next(iter(new.values())).shape[0]
    n_old = next(iter(old.values())).shape[0]
    k = min(n_old, max(1, int(n_new * fraction)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_old, generator=g)[:k]
    out = {}
    for key in keys:
        a, b = new[key], old[key][idx]
        if a.shape[1:] != b.shape[1:]:
            width = max(a.shape[-1], b.shape[-1]) if a.dim() > 1 else None
            if width is None:
                raise ConfigError(f"replay: Formen von '{key}' passen nicht zusammen.")
            a = torch.nn.functional.pad(a, (0, width - a.shape[-1]))
            b = torch.nn.functional.pad(b, (0, width - b.shape[-1]))
        out[key] = torch.cat([a, b])
    return out


def shrink_config(model_spec: Any, width: float = 0.5, depth: float = 0.5) -> dict:
    """Kleinere Schüler-Konfiguration: Breite (d_model / versteckte Größen) und Tiefe skalieren.

    Unterstützt Transformer/SSM/Hybrid (d_model, n_layers, Köpfe), MLP (hidden_dims), RNN (Schichtgrößen)
    und CNN (Kanäle je Stufe). Ein- und Ausgabegrößen bleiben unverändert.
    """
    spec = model_spec.to_dict() if hasattr(model_spec, "to_dict") else copy.deepcopy(model_spec)
    name, args = spec["name"], copy.deepcopy(spec.get("args", {}))

    def scale(v: int, f: float, multiple: int = 1, minimum: int = 1) -> int:
        out = max(minimum, int(round(v * f)))
        return max(multiple, int(round(out / multiple)) * multiple) if multiple > 1 else out

    if "d_model" in args:
        heads = args.get("block", {}).get("mixer", {}).get("args", {}).get("num_heads")
        new_d = scale(args["d_model"], width, multiple=max(8, heads or 1) if heads else 8)
        if heads:
            hd = max(8, args["d_model"] // heads)
            new_heads = max(1, new_d // hd)
            kv = args["block"]["mixer"]["args"].get("num_kv_heads")
            while new_d % new_heads:
                new_heads -= 1
            args["block"]["mixer"]["args"]["num_heads"] = new_heads
            if kv:
                kv_new = max(1, min(kv, new_heads))
                while new_heads % kv_new:
                    kv_new -= 1
                args["block"]["mixer"]["args"]["num_kv_heads"] = kv_new
        args["d_model"] = new_d
        args["n_layers"] = scale(args.get("n_layers", 4), depth)
        args["layer_overrides"] = {}  # Overrides beziehen sich auf die alte Tiefe
    elif "hidden_dims" in args:
        dims = args["hidden_dims"]
        keep = max(1, int(round(len(dims) * depth))) if dims else 0
        args["hidden_dims"] = [scale(d, width) for d in dims[:keep]]
        args["layer_overrides"] = {}
    elif "hidden_sizes" in args:
        sizes = args["hidden_sizes"]
        keep = max(1, int(round(len(sizes) * depth)))
        args["hidden_sizes"] = [scale(h, width) for h in sizes[:keep]]
        args["layer_overrides"] = {}
    elif "stages" in args and isinstance(args["stages"], list):
        args["stages"] = [dict(st, channels=scale(st.get("channels", 32), width, multiple=8),
                               num_blocks=max(1, int(round(st.get("num_blocks", 1) * depth))))
                          for st in args["stages"]]
        if isinstance(args.get("stem"), dict) and "channels" in args["stem"]:
            args["stem"]["channels"] = scale(args["stem"]["channels"], width, multiple=8)
        args["block_overrides"] = {}
    else:
        raise ConfigError(f"shrink_config kennt keine Größenparameter für Modell '{name}' – Schüler bitte explizit "
                          "angeben (student=...).")
    return {"name": name, "args": args}
