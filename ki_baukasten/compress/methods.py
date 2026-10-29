"""
ki_baukasten.compress.methods
=============================

Die eigentlichen **Kompressionsverfahren** ("Modell eindampfen").

==================  ======================================================================
Verfahren           Was passiert
==================  ======================================================================
``prune_ffn``       unwichtige Neuronen der Feed-Forward-Schichten entfernen (MLP, GLU, MoE-Experten)
``prune_heads``     unwichtige Attention-Köpfe entfernen (bei GQA gruppenweise)
``drop_layers``     ganze Blöcke entfernen (Block-Einfluss: Blöcke, die ihre Eingabe kaum ändern)
``prune_mlp``       Neuronen der versteckten Schichten eines MLP-Modells entfernen
``low_rank``        lineare Schichten per SVD zerlegen (nur wo es Parameter spart)
``sparsify``        unstrukturiertes Magnitude-Pruning (Gewichte auf 0, global oder je Schicht)
==================  ======================================================================

Wichtigkeit (``importance``)
----------------------------
* ``magnitude`` – Gewichtsnormen (ohne Daten, schnell)
* ``activation`` – mittlere Aktivierung × Gewichtsnorm auf Kalibrierungsdaten (Wanda-Prinzip)
* ``taylor`` – |Gewicht × Gradient| (Taylor-Näherung des Verlustanstiegs beim Entfernen)
* ``random`` – Zufall (Kontrollexperiment)

Alle Verfahren liefern eine Liste von **Struktur-Regeln** (siehe ``compress.structure``),
die das Projekt in ``module_rules`` übernimmt – so lässt sich das kleinere Modell
speichern und exakt wieder laden.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from ..components.attention import MultiHeadAttention
from ..components.feedforward import GLUFeedForward, MLPFeedForward
from ..core import ConfigError, select_indices
from .structure import (_check_plain_linear, drop_list_items, factorize_linear, keep_cols, keep_heads, keep_norm,
                        keep_rows)

Rule = dict[str, Any]


# ===========================================================================
# Kalibrierung
# ===========================================================================

def _device(model: nn.Module) -> torch.device:
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _call(model: nn.Module, batch: Any) -> torch.Tensor:
    from ..training.data_utils import to_device
    from ..training.paradigms import call_model, main_output

    batch = to_device(batch, _device(model))
    if isinstance(batch, dict) and "inputs" in batch:
        extra = {"attention_mask": batch.get("attention_mask")}
        return main_output(call_model(model, batch["inputs"], **extra))
    return main_output(call_model(model, batch))


def calibration_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    """Sinnvoller Verlust ohne Paradigma: Labels falls vorhanden, sonst nächstes-Token-Vorhersage bzw. Ausgabenorm."""
    from ..training.paradigms import auto_loss

    out = _call(model, batch)
    if isinstance(batch, dict) and batch.get("targets") is not None:
        return auto_loss(out, batch["targets"].to(out.device))
    inputs = batch["inputs"] if isinstance(batch, dict) else batch
    if isinstance(inputs, torch.Tensor) and not inputs.is_floating_point() and out.dim() == 3 \
            and out.shape[1] == inputs.shape[1]:
        return F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]).float(), inputs[:, 1:].reshape(-1).to(out.device))
    return out.float().square().mean()


class _Recorder:
    """Sammelt Eingänge bestimmter Module (Summe |x| pro Kanal) über Kalibrierungsbatches."""

    def __init__(self, modules: dict[str, nn.Module], per_head: dict[str, int] | None = None):
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.handles = []
        per_head = per_head or {}
        for name, m in modules.items():
            self.handles.append(m.register_forward_pre_hook(self._hook(name, per_head.get(name))))

    def _hook(self, name: str, heads: int | None):
        def hook(module, args):
            x = args[0].detach().float()
            x = x.reshape(-1, x.shape[-1])
            if heads:  # Norm je Kopf: [N, H*hd] → [N, H]
                v = x.view(x.shape[0], heads, -1).norm(dim=-1).sum(0)
            else:
                v = x.abs().sum(0)
            self.sums[name] = self.sums.get(name, 0) + v
            self.counts[name] = self.counts.get(name, 0) + x.shape[0]
        return hook

    def mean(self, name: str) -> torch.Tensor:
        return self.sums[name] / max(1, self.counts[name])

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def _run_calibration(model: nn.Module, batches: list[Any], backward: bool) -> None:
    was = model.training
    model.eval()
    try:
        for b in batches:
            if backward:
                with torch.enable_grad():
                    calibration_loss(model, b).backward()
            else:
                with torch.no_grad():
                    _call(model, b)
    finally:
        model.train(was)


def _need_data(importance: str, batches: list[Any] | None, what: str) -> None:
    if importance in ("activation", "taylor") and not batches:
        raise ConfigError(f"{what}: importance='{importance}' braucht Kalibrierungsdaten (calibration).")


def _taylor(*params: torch.Tensor) -> list[torch.Tensor]:
    return [(p.detach() * p.grad).abs() if p.grad is not None else torch.zeros_like(p) for p in params]


# ===========================================================================
# Hilfen
# ===========================================================================

def _keep_count(n: int, amount: float, multiple_of: int = 1, minimum: int = 1) -> int:
    keep = max(minimum, int(round(n * (1 - amount))))
    if multiple_of > 1 and keep > multiple_of:
        keep = max(multiple_of, int(round(keep / multiple_of)) * multiple_of)
    return min(n, keep)


def _top(scores: torch.Tensor, k: int) -> torch.Tensor:
    return scores.topk(k).indices.sort().values.cpu()


def _blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    blocks = model.checkpointable_blocks() if hasattr(model, "checkpointable_blocks") else []
    names = {id(m): n for n, m in model.named_modules()}
    return [(names[id(b)], b) for b in blocks if id(b) in names]


def _in_layers(model: nn.Module, layers: str | None) -> Callable[[str], bool]:
    if layers is None or layers == "*":
        return lambda name: True
    blocks = [n for n, _ in _blocks(model)]
    chosen = [blocks[i] for i in select_indices(layers, len(blocks))]
    return lambda name: any(name == p or name.startswith(p + ".") for p in chosen)


def _set(model: nn.Module, name: str, module: nn.Module) -> None:
    from ..tensor.modules import _set_module
    _set_module(model, name, module)


def _lin_rule(name: str, in_f: int | None = None, out_f: int | None = None) -> Rule:
    op: dict[str, Any] = {"name": "resize_linear"}
    if in_f is not None:
        op["in_features"] = in_f
    if out_f is not None:
        op["out_features"] = out_f
    return {"label": "Kompression: Form", "target": name, "match": "exact", "op": op}


# ===========================================================================
# FFN-Neuronen
# ===========================================================================

def prune_ffn(model: nn.Module, amount: float = 0.3, importance: str = "activation", calibration: list | None = None,
              layers: str | None = None, multiple_of: int = 8) -> list[Rule]:
    """Entfernt den Anteil ``amount`` der versteckten FFN-Neuronen (pro FFN gleich viele)."""
    if not 0 <= amount < 1:
        raise ConfigError("prune_ffn: amount muss in [0, 1) liegen.")
    use = _in_layers(model, layers)
    ffns = {n: m for n, m in model.named_modules() if isinstance(m, (MLPFeedForward, GLUFeedForward)) and use(n)}
    if not ffns:
        raise ConfigError("prune_ffn: keine MLP-/GLU-Feed-Forward-Schicht gefunden.")
    _need_data(importance, calibration, "prune_ffn")
    rec = None
    if importance == "activation":
        rec = _Recorder({n: m.down for n, m in ffns.items()})
        _run_calibration(model, calibration, backward=False)
        rec.close()
    elif importance == "taylor":
        model.zero_grad(set_to_none=True)
        _run_calibration(model, calibration, backward=True)
    rules: list[Rule] = []
    for name, ffn in ffns.items():
        for lin in ([ffn.up, ffn.down] + ([ffn.gate] if isinstance(ffn, GLUFeedForward) else [])):
            _check_plain_linear(lin, name)
        h = ffn.up.out_features
        down_norm = ffn.down.weight.detach().float().norm(dim=0)
        if importance == "magnitude":
            score = ffn.up.weight.detach().float().norm(dim=1) * down_norm
            if isinstance(ffn, GLUFeedForward):
                score = score * ffn.gate.weight.detach().float().norm(dim=1)
        elif importance == "activation":
            score = rec.mean(name).to(down_norm.device) * down_norm
        elif importance == "taylor":
            parts = _taylor(ffn.up.weight, ffn.down.weight) + (
                _taylor(ffn.gate.weight) if isinstance(ffn, GLUFeedForward) else [])
            score = parts[0].sum(1) + parts[1].sum(0) + (parts[2].sum(1) if len(parts) > 2 else 0)
        else:
            score = torch.rand(h)
        keep = _keep_count(h, amount, multiple_of)
        idx = _top(score.float(), keep)
        ffn.up, ffn.down = keep_rows(ffn.up, idx), keep_cols(ffn.down, idx)
        rules += [_lin_rule(f"{name}.up", out_f=keep), _lin_rule(f"{name}.down", in_f=keep)]
        if isinstance(ffn, GLUFeedForward):
            ffn.gate = keep_rows(ffn.gate, idx)
            rules.append(_lin_rule(f"{name}.gate", out_f=keep))
    model.zero_grad(set_to_none=True)
    return rules


# ===========================================================================
# Attention-Köpfe
# ===========================================================================

def prune_heads(model: nn.Module, amount: float = 0.25, importance: str = "activation", calibration: list | None = None,
                layers: str | None = None, granularity: str = "auto") -> list[Rule]:
    """Entfernt den Anteil ``amount`` der Attention-Köpfe.

    ``granularity``: ``query`` (Query-Köpfe gleichmäßig in jeder KV-Gruppe), ``group`` (ganze KV-Gruppen
    samt ihrer Query-Köpfe – verkleinert auch den KV-Cache) oder ``auto`` (group bei GQA mit ≥ 2 Gruppen).
    """
    use = _in_layers(model, layers)
    attns = {n: m for n, m in model.named_modules() if isinstance(m, MultiHeadAttention) and use(n)
             and type(m.kernel).__name__ != "LinearKernel"}
    if not attns:
        raise ConfigError("prune_heads: keine Attention gefunden.")
    _need_data(importance, calibration, "prune_heads")
    rec = None
    if importance == "activation":
        rec = _Recorder({n: m.o_proj for n, m in attns.items()}, {n: m.num_heads for n, m in attns.items()})
        _run_calibration(model, calibration, backward=False)
        rec.close()
    elif importance == "taylor":
        model.zero_grad(set_to_none=True)
        _run_calibration(model, calibration, backward=True)
    rules: list[Rule] = []
    for name, a in attns.items():
        _check_plain_linear(a.q_proj, name)
        _check_plain_linear(a.o_proj, name)
        H, Hkv, hd = a.num_heads, a.num_kv_heads, a.head_dim
        G = H // Hkv
        o_norm = a.o_proj.weight.detach().float().view(a.o_proj.out_features, H, hd).norm(dim=(0, 2))
        if importance == "magnitude":
            score = a.q_proj.weight.detach().float().view(H, hd, -1).norm(dim=(1, 2)) * o_norm
        elif importance == "activation":
            score = rec.mean(name).to(o_norm.device) * o_norm
        elif importance == "taylor":
            tq, to = _taylor(a.q_proj.weight, a.o_proj.weight)
            score = tq.view(H, -1).sum(1) + to.view(to.shape[0], H, hd).sum(dim=(0, 2))
        else:
            score = torch.rand(H)
        mode = granularity if granularity != "auto" else ("group" if Hkv >= 2 and G >= 1 else "query")
        if mode == "group" and Hkv >= 2:
            group_score = score.view(Hkv, G).sum(1)
            kv_keep = _top(group_score, _keep_count(Hkv, amount))
            q_keep = torch.cat([torch.arange(G) + k * G for k in kv_keep.tolist()])
        else:
            g_keep = _keep_count(G, amount)
            kv_keep = torch.arange(Hkv)
            q_keep = torch.cat([_top(score.view(Hkv, G)[k], g_keep) + k * G for k in range(Hkv)])
        if len(q_keep) == H:
            continue
        keep_heads(a, q_keep, kv_keep)
        rules.append({"label": "Kompression: Form", "target": name, "match": "exact",
                      "op": {"name": "resize_attention", "num_heads": a.num_heads, "num_kv_heads": a.num_kv_heads}})
    model.zero_grad(set_to_none=True)
    return rules


# ===========================================================================
# Ganze Blöcke
# ===========================================================================

def block_influence(model: nn.Module, calibration: list) -> list[float]:
    """Block-Einfluss (ShortGPT): 1 − mittlere Kosinus-Ähnlichkeit von Block-Ein- und -Ausgabe."""
    blocks = _blocks(model)
    sims: dict[int, list[float]] = {i: [] for i in range(len(blocks))}
    handles = []
    for i, (_, b) in enumerate(blocks):
        def hook(module, args, output, i=i):
            x = args[0] if args else None
            y = output[0] if isinstance(output, tuple) else output
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape:
                cos = F.cosine_similarity(x.float().reshape(-1, x.shape[-1]), y.float().reshape(-1, y.shape[-1]), dim=-1)
                sims[i].append(float(cos.mean()))
        handles.append(b.register_forward_hook(hook))
    try:
        _run_calibration(model, calibration, backward=False)
    finally:
        for h in handles:
            h.remove()
    return [1.0 - (sum(v) / len(v)) if v else 1.0 for v in sims.values()]


def drop_layers(model: nn.Module, count: int | None = None, amount: float | None = None,
                importance: str = "block_influence", calibration: list | None = None,
                keep_first_last: bool = True) -> list[Rule]:
    """Entfernt ``count`` Blöcke (oder den Anteil ``amount``) mit dem geringsten Einfluss."""
    blocks = _blocks(model)
    n = len(blocks)
    if n < 2:
        raise ConfigError("drop_layers: Modell meldet weniger als 2 Blöcke.")
    count = count if count is not None else int(round(n * (amount or 0.0)))
    protected = {0, n - 1} if keep_first_last and n > 2 else set()
    max_drop = n - len(protected) if protected else n - 1  # mindestens ein Block bleibt immer
    count = max(0, min(count, max_drop))
    if count == 0:
        return []
    if importance == "block_influence":
        if not calibration:
            raise ConfigError("drop_layers: block_influence braucht Kalibrierungsdaten.")
        score = block_influence(model, calibration)
    elif importance == "last":
        score = [-float(i) for i in range(n)]  # hintere Blöcke zuerst entfernen
    elif importance == "magnitude":
        score = [float(sum(p.detach().float().norm() for p in b.parameters())) for _, b in blocks]
    else:
        score = torch.rand(n).tolist()
    order = sorted((i for i in range(n) if i not in protected), key=lambda i: score[i])
    drop = sorted(order[:count])
    # Blöcke liegen in einer (oder mehreren) ModuleLists – je Liste die Original-Indizes entfernen.
    by_list: dict[str, list[int]] = {}
    for i in drop:
        name = blocks[i][0]
        parent, _, idx = name.rpartition(".")
        if not idx.isdigit():
            raise ConfigError(f"drop_layers: Block '{name}' liegt nicht in einer ModuleList.")
        by_list.setdefault(parent, []).append(int(idx))
    rules = []
    for parent, idxs in by_list.items():
        owner = model.get_submodule(parent)
        _set(model, parent, drop_list_items(owner, idxs))
        rules.append({"label": "Kompression: Blöcke entfernt", "target": parent, "match": "exact",
                      "op": {"name": "drop_items", "indices": sorted(idxs)}})
    return rules


# ===========================================================================
# MLP-Modell (versteckte Schichten)
# ===========================================================================

def prune_mlp(model: nn.Module, amount: float = 0.3, importance: str = "magnitude", calibration: list | None = None,
              multiple_of: int = 1) -> list[Rule]:
    """Neuronen-Pruning für das Modell ``mlp`` (Schichten ohne Identitäts-Residual)."""
    from ..models.mlp import MLP

    if not isinstance(model, MLP):
        raise ConfigError("prune_mlp ist nur für das Modell 'mlp' gedacht (Transformer: prune_ffn).")
    _need_data(importance, calibration, "prune_mlp")
    layers = list(model.layers)
    rec = None
    if importance == "activation":
        targets = {f"layers.{i + 1}.ops.linear" if i + 1 < len(layers) else "out": None for i in range(len(layers))}
        mods = {k: model.get_submodule(k) for k in targets}
        rec = _Recorder(mods)
        _run_calibration(model, calibration, backward=False)
        rec.close()
    elif importance == "taylor":
        model.zero_grad(set_to_none=True)
        _run_calibration(model, calibration, backward=True)
    rules: list[Rule] = []
    for i, layer in enumerate(layers):
        lin = layer.ops["linear"]
        if type(lin) is not nn.Linear or (layer.skip is not None and isinstance(layer.skip, nn.Identity)):
            continue  # parametrisierte Schicht oder Identitäts-Residual → Breite muss bleiben
        nxt_name = f"layers.{i + 1}" if i + 1 < len(layers) else None
        nxt = layers[i + 1] if nxt_name else None
        if nxt is not None and (type(nxt.ops["linear"]) is not nn.Linear
                                or (nxt.skip is not None and isinstance(nxt.skip, nn.Identity))):
            continue
        consumer = nxt.ops["linear"] if nxt is not None else model.out
        h = lin.out_features
        if importance == "magnitude":
            score = lin.weight.detach().float().norm(dim=1) * consumer.weight.detach().float().norm(dim=0)
        elif importance == "activation":
            key = f"layers.{i + 1}.ops.linear" if nxt is not None else "out"
            score = rec.mean(key) * consumer.weight.detach().float().norm(dim=0)
        elif importance == "taylor":
            a, b = _taylor(lin.weight, consumer.weight)
            score = a.sum(1) + b.sum(0)
        else:
            score = torch.rand(h)
        keep = _keep_count(h, amount, multiple_of)
        idx = _top(score, keep)
        p = f"layers.{i}"
        layer.ops["linear"] = keep_rows(lin, idx)
        rules.append(_lin_rule(f"{p}.ops.linear", out_f=keep))
        norm_after = "norm" in layer.order and layer.order.index("linear") < layer.order.index("norm")
        if norm_after and any(t.dim() == 1 and t.shape[0] == h for t in layer.ops["norm"].parameters()):
            keep_norm(layer.ops["norm"], idx)
            rules.append({"label": "Kompression: Form", "target": f"{p}.ops.norm", "match": "exact",
                          "op": {"name": "resize_norm", "num_features": keep}})
        if isinstance(layer.skip, nn.Linear):
            layer.skip = keep_rows(layer.skip, idx)
            rules.append(_lin_rule(f"{p}.skip", out_f=keep))
        if nxt is not None:
            nxt.ops["linear"] = keep_cols(nxt.ops["linear"], idx)
            rules.append(_lin_rule(f"{nxt_name}.ops.linear", in_f=keep))
            if "norm" in nxt.order and nxt.order.index("norm") < nxt.order.index("linear"):
                keep_norm(nxt.ops["norm"], idx)
                rules.append({"label": "Kompression: Form", "target": f"{nxt_name}.ops.norm", "match": "exact",
                              "op": {"name": "resize_norm", "num_features": keep}})
            if isinstance(nxt.skip, nn.Linear):
                nxt.skip = keep_cols(nxt.skip, idx)
                rules.append(_lin_rule(f"{nxt_name}.skip", in_f=keep))
        else:
            model.out = keep_cols(model.out, idx)
            rules.append(_lin_rule("out", in_f=keep))
    model.zero_grad(set_to_none=True)
    return rules


# ===========================================================================
# Low-Rank
# ===========================================================================

def low_rank(model: nn.Module, energy: float = 0.9, rank_ratio: float | None = None, targets: str | None = None,
             exclude: str | None = "lm_head|embed|router", min_saving: float = 0.1) -> list[Rule]:
    """Zerlegt lineare Schichten per SVD, wenn das mindestens ``min_saving`` der Parameter spart.

    Rang: kleinster Rang, der den Anteil ``energy`` der quadrierten Singulärwerte erhält
    (oder fest ``rank_ratio`` · min(in, out)).
    """
    import re

    rx_t = re.compile(targets) if targets else None
    rx_e = re.compile(exclude) if exclude else None
    rules: list[Rule] = []
    for name, m in list(model.named_modules()):
        if type(m) is not nn.Linear or (rx_t and not rx_t.search(name)) or (rx_e and rx_e.search(name)):
            continue
        full_rank = min(m.in_features, m.out_features)
        if rank_ratio is not None:
            r = max(1, int(round(full_rank * rank_ratio)))
        else:
            s = torch.linalg.svdvals(m.weight.detach().float())
            cum = (s.square().cumsum(0) / s.square().sum().clamp_min(1e-12))
            r = int((cum < energy).sum()) + 1
        if r * (m.in_features + m.out_features) > (1 - min_saving) * m.in_features * m.out_features:
            continue
        _set(model, name, factorize_linear(m, r))
        rules.append({"label": "Kompression: Low-Rank", "target": name, "match": "exact",
                      "op": {"name": "factorize", "rank": r}})
    return rules


# ===========================================================================
# Unstrukturiert
# ===========================================================================

def sparsify(model: nn.Module, sparsity: float = 0.5, scope: str = "global", targets: str | None = None,
             exclude: str | None = "embed|norm|lm_head") -> dict[str, torch.Tensor]:
    """Setzt den Anteil ``sparsity`` der betragskleinsten Gewichte auf 0. Liefert die Masken (1 = behalten).

    Speicher sinkt erst mit spärlichen Formaten/Kompression (Zip, Sparse-Kernel); dafür bleibt die Struktur gleich.
    """
    import re

    rx_t = re.compile(targets) if targets else None
    rx_e = re.compile(exclude) if exclude else None
    params = [(n, p) for n, p in model.named_parameters() if p.dim() >= 2 and (not rx_t or rx_t.search(n))
              and not (rx_e and rx_e.search(n))]
    if not params:
        raise ConfigError("sparsify: keine passenden Gewichte gefunden.")
    masks: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        if scope == "global":
            flat = torch.cat([p.detach().abs().flatten().float() for _, p in params])
            k = int(flat.numel() * sparsity)
            thresh = flat.kthvalue(max(1, k)).values if k > 0 else flat.new_tensor(-1.0)
            for n, p in params:
                masks[n] = (p.detach().abs().float() > thresh).to(p.dtype)
        else:
            for n, p in params:
                flat = p.detach().abs().flatten().float()
                k = int(flat.numel() * sparsity)
                thresh = flat.kthvalue(max(1, k)).values if k > 0 else flat.new_tensor(-1.0)
                masks[n] = (p.detach().abs().float() > thresh).to(p.dtype)
        for n, p in params:
            p.mul_(masks[n])
    return masks


def sparsity_of(model: nn.Module) -> float:
    """Anteil exakter Nullen in allen Gewichtsmatrizen."""
    total = zeros = 0
    for p in model.parameters():
        if p.dim() >= 2:
            total += p.numel()
            zeros += int((p == 0).sum())
    return zeros / max(1, total)


def apply_rules(model: nn.Module, rules: Iterable[Rule]) -> None:
    """Struktur-Regeln auf ein (neu gebautes) Modell anwenden – wie beim Laden eines Projekts."""
    from ..tensor import ModuleRuleConfig, ModuleSurgeon

    ModuleSurgeon(model, [ModuleRuleConfig.from_dict(r) for r in rules]).apply()


__all__ = ["prune_ffn", "prune_heads", "drop_layers", "block_influence", "prune_mlp", "low_rank", "sparsify",
           "sparsity_of", "calibration_loss", "apply_rules"]
