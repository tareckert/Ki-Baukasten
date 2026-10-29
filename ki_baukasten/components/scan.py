"""
ki_baukasten.components.scan
============================

Schnelle **lineare Rekurrenzen** – das Herzstück von Mamba, minGRU & Co.

Problem
-------
Ein selektives SSM (Mamba) und viele moderne RNNs (minGRU, minLSTM, RG-LRU)
berechnen pro Zeitschritt::

    h_t = a_t ⊙ h_{t-1} + b_t          (elementweise, a_t und b_t hängen nur von der Eingabe ab)

Eine Python-Schleife über die Zeit ist korrekt, aber langsam: Für T Schritte
entstehen T kleine Rechenoperationen und T Knoten im Autograd-Graphen – genau
das machte das Mamba-Training im Baukasten bisher zäh.

Lösung: assoziativer Scan
-------------------------
Zwei Schritte lassen sich zu **einem** zusammenfassen::

    (a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂,  a₂·b₁ + b₂)

Weil ``∘`` assoziativ ist, braucht man nur ``log₂(T)`` Runden
(Hillis-Steele-Scan), in denen *alle* Positionen gleichzeitig verarbeitet
werden – ideal für GPUs und auch auf der CPU um ein Vielfaches schneller.

Modi (``mode``)
---------------
==============  =========================================================================
``auto``        CPU: ``sequential`` (fusioniert, in-place); GPU: ``chunked``
``parallel``    ein Hillis-Steele-Scan über die ganze Sequenz (log₂T Runden)
``chunked``     parallel innerhalb von Blöcken der Länge ``chunk_size``, Übertrag
                zwischen Blöcken sequenziell – wenige große Rechenschritte (GPU)
``sequential``  Schleife über die Zeit. Ohne Gradientenverfolgung (also innerhalb
                der eigenen Backward-Funktionen) als fusionierter In-place-Schritt
                pro Zeitpunkt – auf der CPU die schnellste Variante (O(T) Arbeit)
==============  =========================================================================

Warum war das alte Mamba-Training langsam? Nicht die Schleife an sich, sondern
dass **Autograd jeden Zeitschritt einzeln aufzeichnete** (T·3 Graph-Knoten,
T Zwischenkopien). Die eigenen Autograd-Funktionen unten rechnen Forward und
Backward ohne Aufzeichnung – das macht auch die Schleife schnell.

Speichersparender Backward
--------------------------
Der Gradient einer linearen Rekurrenz ist selbst wieder eine lineare
Rekurrenz – nur rückwärts in der Zeit::

    g_t = ∂L/∂h_t + a_{t+1} ⊙ g_{t+1}       ∂L/∂b_t = g_t       ∂L/∂a_t = g_t ⊙ h_{t-1}

* ``LinearScanFunction`` speichert deshalb nur ``a`` und ``h`` und rechnet den
  Backward mit demselben schnellen Scan.
* ``SelectiveScanFunction`` (Mamba) geht weiter: Sie speichert nur die kleinen
  Eingaben ``u, Δ, B, C`` und berechnet die großen ``[B, T, E, N]``-Tensoren im
  Backward neu – wie der offizielle CUDA-Kernel, nur in reinem PyTorch.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ..core.runtime import is_exporting

SCAN_MODES = {
    "auto": "CPU: fusionierte Schleife, GPU: chunked",
    "parallel": "Hillis-Steele über die ganze Sequenz (log₂T Runden)",
    "chunked": "parallel in Blöcken (chunk_size), Übertrag sequenziell – schnell und sparsam",
    "sequential": "Referenz-Schleife über die Zeit (langsam, zum Prüfen)",
}


# ===========================================================================
# Kern-Algorithmen (ohne Autograd-Besonderheiten)
# ===========================================================================

def _hillis_steele(a: torch.Tensor, b: torch.Tensor, dim: int, need_a: bool,
                   inplace: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Inklusiver Scan von ``(a, b)`` entlang ``dim`` in ⌈log₂T⌉ Runden.

    Nach Runde k enthält Position t die Zusammenfassung der letzten 2^k
    Schritte. ``need_a=False`` spart in der letzten Runde die (dann unnötige)
    Produktbildung von ``a``. ``inplace=True`` (nur ohne Gradientenverfolgung)
    überschreibt Kopien der Eingaben statt neue Tensoren zusammenzusetzen.

    Returns:
        ``(A, H)`` mit ``A_t = a_1·…·a_t`` (kumuliertes Produkt) und ``H_t`` = Zustand bei h_0 = 0.
    """
    T = a.shape[dim]
    if inplace:
        a, b = a.clone(), b.clone()
    offset = 1
    while offset < T:
        a_hi, a_lo = a.narrow(dim, offset, T - offset), a.narrow(dim, 0, T - offset)
        update_a = need_a or offset * 2 < T
        if inplace:
            b.narrow(dim, offset, T - offset).add_(a_hi * b.narrow(dim, 0, T - offset))
            if update_a:
                a_hi.copy_(a_hi * a_lo)  # rechte Seite wird vollständig berechnet, bevor überschrieben wird
        else:
            b = torch.cat([b.narrow(dim, 0, offset), a_hi * b.narrow(dim, 0, T - offset)
                           + b.narrow(dim, offset, T - offset)], dim)
            if update_a:
                a = torch.cat([a.narrow(dim, 0, offset), a_hi * a_lo], dim)
        offset *= 2
    return a, b


def _sequential(a: torch.Tensor, b: torch.Tensor, inplace: bool, reverse: bool = False) -> torch.Tensor:
    steps = range(a.shape[1] - 1, -1, -1) if reverse else range(a.shape[1])
    if inplace:
        # Ohne Autograd: ein fusionierter Schritt pro Zeitpunkt direkt in den Ausgabepuffer (O(T) Arbeit).
        out = torch.empty_like(b)
        h = torch.zeros_like(b[:, 0])
        for t in steps:
            h = torch.addcmul(b[:, t], a[:, t], h, out=out[:, t])
        return out
    h = torch.zeros_like(b[:, 0])
    outs = []
    for t in steps:
        h = a[:, t] * h + b[:, t]
        outs.append(h)
    if reverse:
        outs.reverse()
    return torch.stack(outs, 1)


def _chunked(a: torch.Tensor, b: torch.Tensor, chunk_size: int, inplace: bool) -> torch.Tensor:
    B, T = a.shape[:2]
    rest = a.shape[2:]
    L = max(1, min(chunk_size, T))
    n_chunks = -(-T // L)
    pad = n_chunks * L - T
    if pad:
        # Auffüllen mit dem neutralen Element (a = 1, b = 0) – ändert keine echten Positionen.
        a = torch.cat([a, a.new_ones((B, pad) + rest)], 1)
        b = torch.cat([b, b.new_zeros((B, pad) + rest)], 1)
    a = a.reshape((B, n_chunks, L) + rest)
    b = b.reshape((B, n_chunks, L) + rest)
    A, H = _hillis_steele(a, b, dim=2, need_a=n_chunks > 1, inplace=inplace)
    if n_chunks == 1:
        out = H
    elif inplace:
        for c in range(1, n_chunks):
            H[:, c].addcmul_(A[:, c], H[:, c - 1, -1].unsqueeze(1))
        out = H
    else:
        chunks = [H[:, 0]]
        carry = H[:, 0, -1]
        for c in range(1, n_chunks):
            hc = H[:, c] + A[:, c] * carry.unsqueeze(1)
            chunks.append(hc)
            carry = hc[:, -1]
        out = torch.stack(chunks, 1)
    out = out.reshape((B, n_chunks * L) + rest)
    return out[:, :T] if pad else out


def scan_raw(a: torch.Tensor, b: torch.Tensor, mode: str = "auto", chunk_size: int = 32,
             reverse: bool = False) -> torch.Tensor:
    """``h_t = a_t·h_{t-1} + b_t`` entlang Dimension 1 (``h_0 = 0``).

    ``reverse=True`` läuft rückwärts in der Zeit (``h_t = a_t·h_{t+1} + b_t``) –
    gebraucht für Backward-Rechnungen. Ohne Gradientenverfolgung wird
    automatisch speicher- und zeitsparend in-place gerechnet.
    """
    T = a.shape[1]
    if T == 0:
        return b
    inplace = not (torch.is_grad_enabled() and (a.requires_grad or b.requires_grad))
    if mode == "auto":
        # CPU: die fusionierte Schleife ist speicherbandbreiten-optimal (O(T) Arbeit);
        # GPU & Co.: wenige große Runden (chunked) statt T kleiner Kernel-Starts.
        mode = "sequential" if (a.device.type == "cpu" and inplace) or T <= 4 else "chunked"
    if mode == "sequential":
        return _sequential(a, b, inplace, reverse)
    if mode not in ("parallel", "chunked"):
        raise ValueError(f"Unbekannter Scan-Modus '{mode}'. Erlaubt: {', '.join(SCAN_MODES)}.")
    if reverse:
        a, b = a.flip(1), b.flip(1)
    out = _hillis_steele(a, b, dim=1, need_a=False, inplace=inplace)[1] if mode == "parallel" \
        else _chunked(a, b, chunk_size, inplace)
    return out.flip(1) if reverse else out


# ===========================================================================
# Speichersparender Backward
# ===========================================================================

class LinearScanFunction(torch.autograd.Function):
    """Autograd-Funktion: speichert nur ``a`` und ``h``; Backward = Rückwärts-Scan."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, mode: str, chunk_size: int) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():
            h = scan_raw(a, b, mode, chunk_size)
        ctx.save_for_backward(a, h)
        ctx.mode, ctx.chunk_size = mode, chunk_size
        return h

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_h: torch.Tensor):  # type: ignore[override]
        a, h = ctx.saved_tensors
        # g_t = grad_h_t + a_{t+1} · g_{t+1}  →  Rückwärts-Scan.
        a_next = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], 1)
        g = scan_raw(a_next, grad_h.contiguous(), ctx.mode, ctx.chunk_size, reverse=True)
        grad_a = None
        if ctx.needs_input_grad[0]:
            h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
            grad_a = g * h_prev
        grad_b = g if ctx.needs_input_grad[1] else None
        return grad_a, grad_b, None, None


def linear_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None, mode: str = "auto",
                chunk_size: int = 32, memory_efficient: bool = True) -> torch.Tensor:
    """Berechnet alle Zustände der Rekurrenz ``h_t = a_t ⊙ h_{t-1} + b_t``.

    Args:
        a, b: ``[B, T, ...]`` (Zeit = Dimension 1; werden bei Bedarf gebroadcastet).
        h0:   Anfangszustand ``[B, ...]`` (z. B. aus einem Zustands-Cache) oder ``None`` (= 0).
        mode: siehe ``SCAN_MODES``.
        chunk_size: Blocklänge für ``chunked``.
        memory_efficient: eigener Backward (spart Zwischenspeicher, siehe Modul-Doku).

    Returns:
        ``h`` ``[B, T, ...]``.
    """
    if a.shape != b.shape:
        a, b = torch.broadcast_tensors(a, b)
    if h0 is not None:
        # h0 in den ersten Schritt einrechnen: b_0 ← b_0 + a_0·h0 (Rest bleibt unverändert).
        b = torch.cat([b[:, :1] + a[:, :1] * h0.unsqueeze(1), b[:, 1:]], 1)
    if memory_efficient and torch.is_grad_enabled() and (a.requires_grad or b.requires_grad) and not is_exporting():
        return LinearScanFunction.apply(a, b, mode, chunk_size)
    return scan_raw(a, b, mode, chunk_size)


# ===========================================================================
# Fusionierter selektiver Scan (Mamba) mit Neuberechnung im Backward
# ===========================================================================

def _discretize(u, delta, A, B, h0):
    dA = torch.exp(delta.unsqueeze(-1) * A)                          # [B, T, E, N]  Zerfall pro Schritt
    dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)     # [B, T, E, N]  Eingang pro Schritt
    if h0 is not None:
        dBu[:, 0] += dA[:, 0] * h0
    return dA, dBu


class SelectiveScanFunction(torch.autograd.Function):
    """Mamba-Scan ``y = C·h + D·u`` als eine Autograd-Funktion.

    Speichert für den Backward **nur die Eingaben** (``u, Δ, B, C`` sind
    ``[B, T, E]`` bzw. ``[B, T, N]``) – die großen ``[B, T, E, N]``-Tensoren
    (Zerfall, Eingang, Zustände) werden im Backward neu berechnet. Das ist das
    Prinzip des offiziellen Mamba-Kernels, hier in reinem PyTorch: der
    Aktivierungsspeicher des Scans sinkt um etwa den Faktor ``d_state``.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, h0, mode, chunk_size):  # type: ignore[override]
        with torch.no_grad():
            dA, dBu = _discretize(u, delta, A, B, h0)
            h = scan_raw(dA, dBu, mode, chunk_size)
            y = torch.einsum("bten,btn->bte", h, C) + u * D
            last = h[:, -1].clone()
        ctx.save_for_backward(u, delta, A, B, C, D, h0 if h0 is not None else torch.empty(0))
        ctx.has_h0, ctx.mode, ctx.chunk_size = h0 is not None, mode, chunk_size
        return y, last

    @staticmethod
    @once_differentiable
    def backward(ctx, gy, glast):  # type: ignore[override]
        u, delta, A, B, C, D, h0 = ctx.saved_tensors
        h0 = h0 if ctx.has_h0 else None
        dA, dBu = _discretize(u, delta, A, B, h0)
        h = scan_raw(dA, dBu, ctx.mode, ctx.chunk_size)
        del dBu
        gy = gy.contiguous()
        gh = gy.unsqueeze(-1) * C.unsqueeze(2)                        # ∂L/∂h_t aus y_t = Σ_n C_tn·h_tn
        if glast is not None:
            gh[:, -1] += glast
        a_next = torch.cat([dA[:, 1:], torch.zeros_like(dA[:, :1])], 1)
        g = scan_raw(a_next, gh, ctx.mode, ctx.chunk_size, reverse=True)  # ∂L/∂(Eingang dBu)
        del a_next, gh
        grad_C = torch.einsum("bten,bte->btn", h, gy)
        h_prev = torch.cat([h0.unsqueeze(1) if h0 is not None else torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
        del h
        t1 = g * h_prev * dA                                          # ∂L/∂(Δ·A) über dA = exp(Δ·A)
        del h_prev
        gB = torch.einsum("bten,btn->bte", g, B)
        grad_delta = torch.einsum("bten,en->bte", t1, A) + gB * u
        grad_A = torch.einsum("bten,bte->en", t1, delta)
        del t1
        grad_B = torch.einsum("bten,bte->btn", g, delta * u)
        grad_u = gB * delta + gy * D
        grad_D = (gy * u).sum((0, 1))
        grad_h0 = g[:, 0] * dA[:, 0] if h0 is not None else None
        return grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D, grad_h0, None, None


def selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                   D: torch.Tensor, h0: torch.Tensor | None = None, mode: str = "auto", chunk_size: int = 32,
                   backward: str = "fused") -> tuple[torch.Tensor, torch.Tensor]:
    """Selektiver Scan nach Mamba.

    Args:
        u, delta: ``[B, T, E]``;  A: ``[E, N]`` (negativ);  B, C: ``[B, T, N]``;  D: ``[E]``.
        h0: Anfangszustand ``[B, E, N]`` (Cache) oder ``None``.
        backward: ``fused`` (Neuberechnung, minimaler Speicher), ``scan`` (speichert Zerfall + Zustände),
                  ``autograd`` (alles verfolgt – langsam, nur zum Vergleichen).

    Returns:
        ``(y [B, T, E], letzter Zustand [B, E, N])``.
    """
    tracking = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (u, delta, A, B, C, D, h0))
    if backward == "fused" and tracking and not is_exporting():
        return SelectiveScanFunction.apply(u, delta, A, B, C, D, h0, mode, chunk_size)
    dA = torch.exp(delta.unsqueeze(-1) * A)
    dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    h = linear_scan(dA, dBu, h0, mode=mode, chunk_size=chunk_size, memory_efficient=backward != "autograd")
    return torch.einsum("bten,btn->bte", h, C) + u * D, h[:, -1]
