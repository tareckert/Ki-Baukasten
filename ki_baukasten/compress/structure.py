"""
ki_baukasten.compress.structure
===============================

**Struktur-Bausteine** der Kompression: Module verkleinern, zerlegen oder entfernen.

Warum ein eigenes Modul?
------------------------
Strukturierte Kompression (Neuronen, Köpfe, Schichten entfernen; Low-Rank-
Zerlegung) ändert die **Form** des Netzes. Damit ein komprimiertes Projekt nach
dem Speichern wieder exakt geladen werden kann, wird jede Formänderung als
**Modul-Regel** protokolliert (``module_rules`` im Projekt). Beim Laden baut der
Baukasten das Originalmodell, wendet die Regeln an (gleiche Formen, zufällige
Werte) und lädt danach die gespeicherten Gewichte hinein.

Registrierte Modul-Operationen (Registry ``module_op``)
-------------------------------------------------------
==================  ==============================================================
``resize_linear``   ``nn.Linear`` mit neuer Ein-/Ausgabebreite
``resize_norm``     Normalisierung mit neuer Merkmalsbreite
``resize_attention`` Attention mit weniger Köpfen (``num_heads``/``num_kv_heads``)
``factorize``       ``nn.Linear`` → zwei schmale Schichten (Rang r): W ≈ B·A
``drop_items``      Einträge einer ``ModuleList`` entfernen (z. B. ganze Blöcke)
==================  ==============================================================

Die ``keep_*``-Funktionen erzeugen dieselben Formen **mit ausgewählten Werten**
(für die eigentliche Kompression).
"""

from __future__ import annotations

import torch
from torch import nn

from ..components.attention import MultiHeadAttention
from ..core import BaseConfig, ConfigError, Level, Param, T
from ..tensor.modules import MODULE_OPS, _set_module


# ===========================================================================
# Low-Rank-Schicht
# ===========================================================================

class LowRankLinear(nn.Module):
    """``y = second(first(x))`` mit ``first: in → r`` (ohne Bias) und ``second: r → out``.

    Spart Parameter, wenn ``r·(in + out) < in·out``. Verhält sich nach außen wie
    ``nn.Linear`` (Attribute ``in_features``, ``out_features``, ``rank``).
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.first = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.second = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:  # rekonstruierte Matrix (Analyse, Distillation)
        return self.second.weight @ self.first.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.second.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


# ===========================================================================
# Formen mit ausgewählten Werten (Kompression)
# ===========================================================================

def _like_linear(old: nn.Module, in_f: int, out_f: int) -> nn.Linear:
    w = old.weight
    return nn.Linear(in_f, out_f, bias=old.bias is not None, device=w.device, dtype=w.dtype)


def _check_plain_linear(module: nn.Module, name: str) -> None:
    if type(module) is not nn.Linear:
        raise ConfigError(f"'{name}' ist {type(module).__name__}, erwartet nn.Linear. Reihenfolge beachten: erst "
                          "strukturiert komprimieren, danach quantisieren / LoRA / Low-Rank.")


@torch.no_grad()
def keep_rows(linear: nn.Linear, idx: torch.Tensor) -> nn.Linear:
    """Nur die Ausgänge ``idx`` behalten (Zeilen von W, Einträge des Bias)."""
    new = _like_linear(linear, linear.in_features, len(idx))
    new.weight.copy_(linear.weight[idx])
    if linear.bias is not None:
        new.bias.copy_(linear.bias[idx])
    new.weight.requires_grad_(linear.weight.requires_grad)
    return new


@torch.no_grad()
def keep_cols(linear: nn.Linear, idx: torch.Tensor) -> nn.Linear:
    """Nur die Eingänge ``idx`` behalten (Spalten von W)."""
    new = _like_linear(linear, len(idx), linear.out_features)
    new.weight.copy_(linear.weight[:, idx])
    if linear.bias is not None:
        new.bias.copy_(linear.bias)
    new.weight.requires_grad_(linear.weight.requires_grad)
    return new


def _norm_tensors(norm: nn.Module) -> list[tuple[str, torch.Tensor, bool]]:
    out = [(n, p, True) for n, p in norm.named_parameters(recurse=False) if p is not None and p.dim() == 1]
    out += [(n, b, False) for n, b in norm.named_buffers(recurse=False) if b is not None and b.dim() == 1]
    return out


@torch.no_grad()
def keep_norm(norm: nn.Module, idx: torch.Tensor) -> None:
    """Normalisierung in-place auf die Merkmale ``idx`` verkleinern (LayerNorm, RMSNorm, BatchNorm …)."""
    n = len(idx)
    for name, t, is_param in _norm_tensors(norm):
        value = t.detach()[idx].clone()
        if is_param:
            setattr(norm, name, nn.Parameter(value, requires_grad=t.requires_grad))
        else:
            norm._buffers[name] = value
    _set_norm_size(norm, n)


def _set_norm_size(norm: nn.Module, n: int) -> None:
    if hasattr(norm, "normalized_shape"):
        norm.normalized_shape = (n,)
    for attr in ("num_features", "dim"):
        if isinstance(getattr(norm, attr, None), int):
            setattr(norm, attr, n)


@torch.no_grad()
def keep_heads(attn: MultiHeadAttention, q_heads: torch.Tensor, kv_heads: torch.Tensor | None = None) -> None:
    """Attention in-place auf die Query-Köpfe ``q_heads`` (und KV-Köpfe ``kv_heads``) verkleinern.

    Bei Grouped-Query-Attention muss jede verbleibende KV-Gruppe gleich viele
    Query-Köpfe behalten (``num_heads`` durch ``num_kv_heads`` teilbar).
    """
    hd = attn.head_dim
    kv_heads = torch.arange(attn.num_kv_heads) if kv_heads is None else kv_heads

    def rows(heads: torch.Tensor) -> torch.Tensor:
        return (heads.view(-1, 1) * hd + torch.arange(hd).view(1, -1)).reshape(-1)

    q_rows, kv_rows = rows(q_heads), rows(kv_heads)
    attn.q_proj = keep_rows(attn.q_proj, q_rows)
    attn.o_proj = keep_cols(attn.o_proj, q_rows)
    if attn.gate_proj is not None:
        attn.gate_proj = keep_rows(attn.gate_proj, q_rows)
    if len(kv_heads) != attn.num_kv_heads:
        if attn.cfg.kv_compression_dim is None:
            attn.k_proj, attn.v_proj = keep_rows(attn.k_proj, kv_rows), keep_rows(attn.v_proj, kv_rows)
        else:
            attn.k_up, attn.v_up = keep_rows(attn.k_up, kv_rows), keep_rows(attn.v_up, kv_rows)
    attn.num_heads, attn.num_kv_heads = len(q_heads), len(kv_heads)
    attn.cfg.num_heads, attn.cfg.num_kv_heads, attn.cfg.head_dim = attn.num_heads, attn.num_kv_heads, hd


@torch.no_grad()
def factorize_linear(linear: nn.Linear, rank: int) -> LowRankLinear:
    """SVD-Zerlegung W ≈ (U·√S)·(√S·Vᵀ) mit Rang ``rank`` – beste Näherung im Sinne der Frobenius-Norm."""
    w = linear.weight.detach().float()
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    sq = S[:rank].sqrt()
    new = LowRankLinear(linear.in_features, linear.out_features, rank, linear.bias is not None,
                        linear.weight.device, linear.weight.dtype)
    new.first.weight.copy_((sq.unsqueeze(1) * Vh[:rank]).to(new.first.weight.dtype))
    new.second.weight.copy_((U[:, :rank] * sq).to(new.second.weight.dtype))
    if linear.bias is not None:
        new.second.bias.copy_(linear.bias)
    return new


def drop_list_items(owner: nn.ModuleList, indices: list[int]) -> nn.ModuleList:
    keep = [m for i, m in enumerate(owner) if i not in set(indices)]
    if not keep:
        raise ConfigError("drop_items würde alle Einträge entfernen.")
    return nn.ModuleList(keep)


# ===========================================================================
# Modul-Operationen (gleiche Formen beim erneuten Laden)
# ===========================================================================

class ResizeLinearConfig(BaseConfig):
    in_features = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Neue Eingabebreite (None = unverändert).", "Formänderung nach Pruning.", "Wird von der Kompression gesetzt."))
    out_features = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Neue Ausgabebreite (None = unverändert).", "Formänderung nach Pruning.", "Wird von der Kompression gesetzt."))


class ResizeNormConfig(BaseConfig):
    num_features = Param(1, min=1, tooltip=T(
        "Neue Merkmalsbreite.", "Formänderung nach Neuronen-Pruning.", "Wird von der Kompression gesetzt."))


class ResizeAttentionConfig(BaseConfig):
    num_heads = Param(1, min=1, tooltip=T("Verbleibende Query-Köpfe.", "Nach Kopf-Pruning.", "Automatisch gesetzt."))
    num_kv_heads = Param(1, min=1, tooltip=T("Verbleibende KV-Köpfe.", "Nach Kopf-Pruning.", "Automatisch gesetzt."))


class FactorizeConfig(BaseConfig):
    rank = Param(8, min=1, tooltip=T(
        "Rang r der Zerlegung W ≈ B·A.", "Kleiner = weniger Parameter, größerer Fehler.", "Automatisch gesetzt."))


class DropItemsConfig(BaseConfig):
    indices = Param([], item_type=int, tooltip=T(
        "Zu entfernende Positionen der ModuleList (Original-Nummerierung).",
        "Entfernt z. B. ganze Transformer-Blöcke.", "Automatisch gesetzt."))


def _resize_linear(a: ResizeLinearConfig):
    def fn(model, name, module, ctx):
        _check_plain_linear(module, name)
        new = _like_linear(module, a.in_features or module.in_features, a.out_features or module.out_features)
        _set_module(model, name, new)
    return fn


def _resize_norm(a: ResizeNormConfig):
    def fn(model, name, module, ctx):
        keep_norm(module, torch.arange(a.num_features))
    return fn


def _resize_attention(a: ResizeAttentionConfig):
    def fn(model, name, module, ctx):
        if not isinstance(module, MultiHeadAttention):
            raise ConfigError(f"resize_attention: '{name}' ist keine MultiHeadAttention.")
        if a.num_heads % a.num_kv_heads:
            raise ConfigError("resize_attention: num_heads muss durch num_kv_heads teilbar sein.")
        g_old = module.num_heads // module.num_kv_heads
        g_new = a.num_heads // a.num_kv_heads
        q = torch.cat([torch.arange(g_new) + k * g_old for k in range(a.num_kv_heads)])
        keep_heads(module, q, torch.arange(a.num_kv_heads))
    return fn


def _factorize(a: FactorizeConfig):
    def fn(model, name, module, ctx):
        _check_plain_linear(module, name)
        _set_module(model, name, factorize_linear(module, min(a.rank, module.in_features, module.out_features)))
    return fn


def _drop_items(a: DropItemsConfig):
    def fn(model, name, module, ctx):
        if not isinstance(module, nn.ModuleList):
            raise ConfigError(f"drop_items: '{name}' ist keine ModuleList.")
        _set_module(model, name, drop_list_items(module, list(a.indices)))
    return fn


R = MODULE_OPS.register
R("resize_linear", lambda a: _resize_linear(a), config=ResizeLinearConfig, level=Level.EXPERT, tooltip=T(
    "Lineare Schicht mit neuer Breite anlegen (Struktur nach Pruning).",
    "Hält die Form eines komprimierten Modells beim Speichern/Laden reproduzierbar.",
    "Wird automatisch von der Kompression eingetragen."))
R("resize_norm", lambda a: _resize_norm(a), config=ResizeNormConfig, level=Level.EXPERT, tooltip=T(
    "Normalisierung mit neuer Breite (Struktur nach Pruning).", "Wie resize_linear.", "Automatisch."))
R("resize_attention", lambda a: _resize_attention(a), config=ResizeAttentionConfig, level=Level.EXPERT, tooltip=T(
    "Attention mit weniger Köpfen (Struktur nach Kopf-Pruning).", "Wie resize_linear.", "Automatisch."))
R("factorize", lambda a: _factorize(a), config=FactorizeConfig, level=Level.ADVANCED, tooltip=T(
    "Lineare Schicht per SVD in zwei schmale Schichten zerlegen (Rang r).",
    "Spart Parameter und Rechenzeit, wenn die Matrix 'eigentlich' niedrigrangig ist.",
    "Kompression großer Projektionen; auch manuell als Modul-Regel nutzbar."))
R("drop_items", lambda a: _drop_items(a), config=DropItemsConfig, level=Level.ADVANCED, tooltip=T(
    "Einträge einer ModuleList entfernen (z. B. Transformer-Blöcke).",
    "Tiefe reduzieren: weniger Schichten = schneller und kleiner.",
    "Layer-Pruning (wird von der Kompression gesetzt, auch manuell nutzbar)."))
