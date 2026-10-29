"""
ki_baukasten.analysis.probe
===========================

**Probelauf & Einblick**: Was passiert *in* der KI – Schritt für Schritt?

* ``probe`` – ein Forward (+ optional Backward) mit Aufzeichnung jedes Moduls in
  Ausführungsreihenfolge ("alle Wege"): Ein-/Ausgabeformen, Statistik der Ausgaben
  (Mittel, Streuung, Min/Max, Anteil Nullen, NaN/Inf), Gradienten, Rechenzeit,
  Speicher der Ausgabe – plus automatische **Warnungen** (tote Neuronen, explodierende
  oder verschwindende Werte, NaN).
* ``weight_stats`` / ``histogram`` / ``heatmap`` – Gewichte ansehen.
* ``attention_maps`` – Attention-Gewichte jeder Schicht (welches Token schaut wohin?).
* ``evaluate_model`` – Testlauf auf Validierungsdaten mit Metriken, Konfusionsmatrix und
  Beispielvorhersagen.

Alle Funktionen liefern reine Python-Daten – die GUI (``ui.gui``) zeichnet sie,
die CLI (``probelauf``) gibt sie als Text aus.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


def tensor_stats(t: torch.Tensor) -> dict[str, float]:
    """Kennzahlen eines Tensors (auf der CPU, in float32)."""
    x = t.detach().float()
    finite = torch.isfinite(x)
    n = x.numel()
    if n == 0:
        return {"numel": 0}
    xf = x[finite] if not bool(finite.all()) else x
    out = {
        "numel": n,
        "mean": float(xf.mean()) if xf.numel() else float("nan"),
        "std": float(xf.std()) if xf.numel() > 1 else 0.0,
        "min": float(xf.min()) if xf.numel() else float("nan"),
        "max": float(xf.max()) if xf.numel() else float("nan"),
        "absmax": float(xf.abs().max()) if xf.numel() else float("nan"),
        "zeros": float((x == 0).float().mean()),
        "nan": int(torch.isnan(x).sum()),
        "inf": int(torch.isinf(x).sum()),
        "norm": float(xf.norm()) if xf.numel() else float("nan"),
    }
    return out


def histogram(t: torch.Tensor, bins: int = 40) -> tuple[list[float], list[int]]:
    """``(Kanten, Häufigkeiten)`` der Werte eines Tensors."""
    x = t.detach().float().flatten()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return [0.0, 1.0], [0]
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    counts = torch.histc(x, bins=bins, min=lo, max=hi)
    edges = torch.linspace(lo, hi, bins + 1)
    return edges.tolist(), [int(c) for c in counts.tolist()]


def heatmap(t: torch.Tensor, max_rows: int = 64, max_cols: int = 64) -> list[list[float]]:
    """Verkleinerte 2D-Ansicht eines Tensors (Mittelwert über Blöcke) für die Darstellung als Bild."""
    x = t.detach().float()
    if x.dim() == 0:
        x = x.view(1, 1)
    elif x.dim() == 1:
        x = x.view(1, -1)
    elif x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    rows, cols = min(max_rows, x.shape[0]), min(max_cols, x.shape[1])
    pooled = torch.nn.functional.adaptive_avg_pool2d(x.unsqueeze(0).unsqueeze(0), (rows, cols))[0, 0]
    return pooled.tolist()


@dataclass
class ProbeRecord:
    order: int
    name: str
    type: str
    depth: int
    in_shapes: list[list[int]]
    out_shapes: list[list[int]]
    out_stats: dict[str, float]
    grad_stats: dict[str, float] | None = None
    param_grad_norm: float | None = None
    time_ms: float = 0.0
    out_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ProbeResult:
    records: list[ProbeRecord]
    output_shapes: list[list[int]]
    output_stats: dict[str, float]
    loss: float | None
    total_ms: float
    outputs: Any = None              # Hauptausgabe (CPU-Tensor) für Detailansichten
    activations: dict[str, torch.Tensor] = field(default_factory=dict)  # erste Ausgabe je Modul (begrenzt)

    def warnings(self) -> list[tuple[str, str]]:
        return [(r.name, w) for r in self.records for w in r.warnings]

    def to_text(self, max_depth: int = 3) -> str:
        lines = [f"Probelauf: {len(self.records)} Modulaufrufe, {self.total_ms:.1f} ms"
                 + (f", Verlust {self.loss:.4f}" if self.loss is not None else ""),
                 f"  {'#':>3} {'Modul':<44} {'Typ':<20} {'Ausgabe':<16} {'Mittel':>9} {'Std':>9} {'|max|':>9} "
                 f"{'Null%':>6} {'Grad':>9} {'ms':>7}"]
        for r in self.records:
            if r.depth > max_depth:
                continue
            s = r.out_stats
            shape = "×".join(map(str, r.out_shapes[0])) if r.out_shapes else ""
            grad = f"{r.grad_stats['norm']:.3g}" if r.grad_stats else "–"
            mark = " ⚠" if r.warnings else ""
            lines.append(f"  {r.order:>3} {('  ' * (r.depth - 1) + r.name.split('.')[-1])[:44]:<44} {r.type[:20]:<20} "
                         f"{shape[:16]:<16} {s.get('mean', 0):>9.3g} {s.get('std', 0):>9.3g} {s.get('absmax', 0):>9.3g} "
                         f"{100 * s.get('zeros', 0):>5.1f} {grad:>9} {r.time_ms:>7.2f}{mark}")
        warns = self.warnings()
        if warns:
            lines.append("  Warnungen:")
            lines += [f"    ⚠ {n}: {w}" for n, w in warns[:30]]
        return "\n".join(lines)


def _warn(stats: dict[str, float], kind: str) -> list[str]:
    out = []
    if stats.get("nan", 0) or stats.get("inf", 0):
        out.append(f"{kind} enthält NaN/Inf ({stats.get('nan', 0)}/{stats.get('inf', 0)})")
    if stats.get("zeros", 0) > 0.9 and stats.get("numel", 0) > 16:
        out.append(f"{kind} zu {100 * stats['zeros']:.0f} % null (tote Neuronen?)")
    absmax = stats.get("absmax", 0)
    if isinstance(absmax, float) and math.isfinite(absmax) and absmax > 1e4:
        out.append(f"{kind} sehr groß (|max| = {absmax:.3g}) – explodierende Werte?")
    if 0 < stats.get("std", 1) < 1e-6 and stats.get("numel", 0) > 16:
        out.append(f"{kind} nahezu konstant (Std {stats['std']:.2g}) – verschwindendes Signal?")
    return out


def probe(model: nn.Module, inputs: Any, targets: Any = None, backward: bool = True, max_depth: int = 6,
          keep_activations: int = 64) -> ProbeResult:
    """Forward (+ Backward) mit Aufzeichnung aller Module in Ausführungsreihenfolge."""
    from ..compress.methods import loss_from_output
    from ..training.data_utils import to_device
    from ..training.paradigms import call_model, main_output

    dev = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    names = {id(m): n for n, m in model.named_modules()}
    by_name: dict[str, ProbeRecord] = {}
    starts: dict[int, float] = {}
    acts: dict[str, torch.Tensor] = {}
    counter = [0]
    handles = []

    def pre(module, args):
        starts[id(module)] = time.perf_counter()

    def post(module, args, output):
        name = names.get(id(module), "?")
        depth = name.count(".") + 1 if name else 0
        if not name or depth > max_depth or name in by_name:
            return
        out = output[0] if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor) \
            else output
        if isinstance(out, dict):
            out = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
        if not isinstance(out, torch.Tensor):
            return
        from .architecture import _shapes
        stats = tensor_stats(out)
        rec = ProbeRecord(counter[0], name, type(module).__name__, depth, _shapes(args), _shapes(output), stats,
                          time_ms=(time.perf_counter() - starts.get(id(module), time.perf_counter())) * 1000,
                          out_bytes=out.numel() * out.element_size(), warnings=_warn(stats, "Ausgabe"))
        counter[0] += 1
        by_name[name] = rec
        if len(acts) < keep_activations:
            acts[name] = out.detach().cpu()
        if backward and out.requires_grad:
            out.register_hook(lambda g, rec=rec: setattr(rec, "grad_stats", tensor_stats(g)))

    for m in model.modules():
        if m is model:
            continue
        try:
            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(post))
        except RuntimeError:  # TorchScript
            pass
    was = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    loss = None
    t0 = time.perf_counter()
    try:
        batch = {"inputs": inputs, **({"targets": targets} if targets is not None else {})}
        batch = to_device(batch, dev)
        with torch.set_grad_enabled(backward):
            out = main_output(call_model(model, batch["inputs"]))
            if backward:
                loss_t = loss_from_output(out, batch)
                loss = float(loss_t.detach())
                if loss_t.requires_grad:
                    loss_t.backward()
    finally:
        for h in handles:
            h.remove()
        model.train(was)
    total = (time.perf_counter() - t0) * 1000
    # Gradientennorm der Parameter je Modul (nur direkte Parameter)
    for name, rec in by_name.items():
        try:
            mod = model.get_submodule(name)
        except AttributeError:
            continue
        grads = [p.grad for p in mod.parameters(recurse=False) if p.grad is not None]
        if grads:
            rec.param_grad_norm = float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
        if rec.grad_stats:
            rec.warnings += [w for w in _warn(rec.grad_stats, "Gradient") if "null" not in w]
    from .architecture import _shapes
    result = ProbeResult(sorted(by_name.values(), key=lambda r: r.order), _shapes(out), tensor_stats(out), loss, total,
                         out.detach().cpu(), acts)
    model.zero_grad(set_to_none=True)
    return result


def weight_stats(model: nn.Module) -> list[dict[str, Any]]:
    """Kennzahlen aller Parameter (und quantisierten Gewichte) – Basis der Gewichtsansicht."""
    rows = []
    for name, p in model.named_parameters():
        s = tensor_stats(p)
        s.update(name=name, shape=list(p.shape), trainable=p.requires_grad,
                 grad_norm=float(p.grad.norm()) if p.grad is not None else None)
        rows.append(s)
    for mname, m in model.named_modules():
        if type(m).__name__ == "QuantLinear":
            s = tensor_stats(m.dequantize())
            s.update(name=f"{mname}.weight (quantisiert {m.fmt})", shape=[m.out_features, m.in_features],
                     trainable=False, grad_norm=None)
            rows.append(s)
    return rows


def get_weight(model: nn.Module, name: str) -> torch.Tensor:
    """Parameter oder (entpacktes) quantisiertes Gewicht nach Namen aus ``weight_stats``."""
    params = dict(model.named_parameters())
    if name in params:
        return params[name].detach()
    if " (quantisiert" in name:
        mod = model.get_submodule(name.split(".weight (")[0])
        return mod.dequantize()
    raise KeyError(name)


@torch.no_grad()
def attention_maps(model: nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
    """Attention-Gewichte jeder Softmax-Attention ``{modul: [H, T, S]}`` (erstes Beispiel des Batches)."""
    from ..components.attention import MultiHeadAttention, SoftmaxKernel
    from ..training.paradigms import call_model

    kernels = [(n, m.kernel) for n, m in model.named_modules()
               if isinstance(m, MultiHeadAttention) and isinstance(m.kernel, SoftmaxKernel)]
    old = [(k, k.cfg.store_weights) for _, k in kernels]
    for _, k in kernels:
        k.cfg.store_weights = True
    was = model.training
    model.eval()
    try:
        call_model(model, inputs)
        return {n: k.last_weights[0].float().cpu() for n, k in kernels if k.last_weights is not None}
    finally:
        for k, flag in old:
            k.cfg.store_weights = flag
            k.last_weights = None
        model.train(was)


@torch.no_grad()
def evaluate_model(model: nn.Module, data: dict[str, torch.Tensor], max_batches: int = 20, batch_size: int = 64,
             samples: int = 8) -> dict[str, Any]:
    """Testlauf: Verlust, Genauigkeit, Konfusionsmatrix (Klassifikation) und Beispielvorhersagen."""
    from ..compress.methods import calibration_loss
    from ..training.data_utils import to_device
    from ..training.paradigms import call_model, main_output

    dev = next(model.parameters()).device
    was = model.training
    model.eval()
    n = next(iter(data.values())).shape[0]
    losses, correct, total = [], 0, 0
    confusion: torch.Tensor | None = None
    examples: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        for b, start in enumerate(range(0, n, batch_size)):
            if b >= max_batches:
                break
            batch = to_device({k: v[start:start + batch_size] for k, v in data.items()
                               if isinstance(v, torch.Tensor)}, dev)
            out = main_output(call_model(model, batch["inputs"]))
            losses.append(float(calibration_loss(model, batch)))
            y = batch.get("targets")
            if y is not None and not y.is_floating_point() and out.dim() == y.dim() + 1:
                pred = out.argmax(-1)
                correct += int((pred == y).sum())
                total += y.numel()
                k = out.shape[-1]
                if k <= 50 and y.dim() == 1:
                    if confusion is None:
                        confusion = torch.zeros(k, k, dtype=torch.long)
                    confusion.index_put_((y.cpu().long(), pred.cpu().long()), torch.ones_like(y.cpu().long()),
                                         accumulate=True)
                for i in range(min(samples - len(examples), y.shape[0])):
                    probs = out[i].float().softmax(-1)
                    examples.append({"ziel": int(y[i]) if y.dim() == 1 else y[i].tolist(),
                                     "vorhersage": int(pred[i]) if pred.dim() == 1 else pred[i].tolist(),
                                     "sicherheit": float(probs.max(-1).values.mean())})
            elif y is not None:
                for i in range(min(samples - len(examples), y.shape[0])):
                    examples.append({"ziel": y[i].flatten()[:6].tolist(), "vorhersage": out[i].flatten()[:6].tolist()})
            else:
                x = batch["inputs"]
                if isinstance(x, torch.Tensor) and not x.is_floating_point() and out.dim() == 3:
                    pred = out[:, :-1].argmax(-1)
                    correct += int((pred == x[:, 1:]).sum())
                    total += pred.numel()
    finally:
        model.train(was)
    res: dict[str, Any] = {"loss": sum(losses) / max(1, len(losses)), "batches": len(losses),
                           "zeit_ms": (time.perf_counter() - t0) * 1000, "beispiele": examples}
    if total:
        res["accuracy"] = correct / total
    if confusion is not None:
        res["confusion"] = confusion.tolist()
    return res


def evaluation_text(res: dict[str, Any]) -> str:
    lines = [f"Testlauf: {res['batches']} Batches, {res['zeit_ms']:.0f} ms, Verlust {res['loss']:.4f}"
             + (f", Genauigkeit {100 * res['accuracy']:.1f} %" if "accuracy" in res else "")]
    if "confusion" in res:
        cm = res["confusion"]
        lines.append("  Konfusionsmatrix (Zeile = Ziel, Spalte = Vorhersage):")
        lines.append("       " + " ".join(f"{j:>5}" for j in range(len(cm))))
        lines += [f"  {i:>4} " + " ".join(f"{v:>5}" for v in row) for i, row in enumerate(cm)]
    if res["beispiele"]:
        lines.append("  Beispiele:")
        lines += [f"    {e}" for e in res["beispiele"]]
    return "\n".join(lines)
