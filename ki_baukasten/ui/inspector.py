"""
ki_baukasten.ui.inspector
=========================

**Probelauf- und Gewichtsfenster** der GUI: ein Blick in die laufende KI.

::

    ┌ Probelauf (Forward+Backward) | Testlauf | Attention | Gewichte neu laden | Tiefe ▾ | ☑ Backward ┐
    ├ Reiter: Wege · Gewichte · Attention · Test ──────────────────────────────────────────────────┤
    │ Tabelle (Module in Ausführungsreihenfolge     │ Histogramm der Werte                          │
    │ bzw. alle Gewichte) – Warnungen farbig         │ Heatmap (blau = negativ, rot = positiv)       │
    │                                                │ Kennzahlen                                    │
    └────────────────────────────────────────────────┴───────────────────────────────────────────────┘

* **Wege**: jedes Modul in der Reihenfolge, in der die Daten es durchlaufen – mit
  Ausgabeform, Statistik, Gradient, Zeit; ⚠ markiert tote/explodierende/NaN-Werte.
  Auswahl zeigt Histogramm und Heatmap der Ausgabe dieses Moduls.
* **Gewichte**: alle Parameter (auch quantisierte) mit Kennzahlen; Auswahl zeigt
  Verteilung und Gewichtsmatrix.
* **Attention**: Attention-Karte je Schicht und Kopf.
* **Test**: Testlauf auf Validierungsdaten (Metriken, Konfusionsmatrix, Beispiele).

Rechnungen laufen in einem Hintergrund-Thread; Tk wird nur im Hauptthread verändert.
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable

import tkinter as tk
from tkinter import messagebox, ttk


def _color(v: float, lo: float, hi: float) -> str:
    """Divergierende Farbskala: blau (negativ) – weiß (0) – rot (positiv)."""
    m = max(abs(lo), abs(hi), 1e-12)
    t = max(-1.0, min(1.0, v / m))
    if t >= 0:
        r, g, b = 255, int(255 * (1 - t)), int(255 * (1 - t))
    else:
        r, g, b = int(255 * (1 + t)), int(255 * (1 + t)), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_histogram(canvas: tk.Canvas, edges: list[float], counts: list[int], title: str = "") -> None:
    canvas.delete("all")
    w, h = int(canvas.winfo_width() or 400), int(canvas.winfo_height() or 160)
    pad = 24
    top = max(counts) if counts and max(counts) > 0 else 1
    bw = (w - 2 * pad) / max(1, len(counts))
    for i, c in enumerate(counts):
        x0 = pad + i * bw
        y0 = h - pad - (h - 2 * pad) * c / top
        mid = (edges[i] + edges[i + 1]) / 2
        canvas.create_rectangle(x0, y0, x0 + bw - 1, h - pad, fill=_color(mid, edges[0], edges[-1]), outline="#888")
    canvas.create_text(pad, h - 8, text=f"{edges[0]:.3g}", anchor="w", font=("TkDefaultFont", 8))
    canvas.create_text(w - pad, h - 8, text=f"{edges[-1]:.3g}", anchor="e", font=("TkDefaultFont", 8))
    canvas.create_text(w / 2, 10, text=title or "Verteilung der Werte", font=("TkDefaultFont", 9, "bold"))


def draw_heatmap(canvas: tk.Canvas, matrix: list[list[float]], title: str = "") -> None:
    canvas.delete("all")
    w, h = int(canvas.winfo_width() or 400), int(canvas.winfo_height() or 300)
    if not matrix or not matrix[0]:
        return
    rows, cols = len(matrix), len(matrix[0])
    flat = [v for row in matrix for v in row]
    lo, hi = min(flat), max(flat)
    top = 20
    cw, ch = (w - 10) / cols, (h - top - 20) / rows
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            x0, y0 = 5 + j * cw, top + i * ch
            canvas.create_rectangle(x0, y0, x0 + cw, y0 + ch, fill=_color(v, lo, hi), outline="")
    canvas.create_text(w / 2, 10, text=title or f"Heatmap {rows}×{cols}", font=("TkDefaultFont", 9, "bold"))
    canvas.create_text(5, h - 8, text=f"min {lo:.3g}", anchor="w", font=("TkDefaultFont", 8), fill="#2244cc")
    canvas.create_text(w - 5, h - 8, text=f"max {hi:.3g}", anchor="e", font=("TkDefaultFont", 8), fill="#cc2222")


class InspectorWindow(tk.Toplevel):
    """Probelauf, Gewichte, Attention und Testlauf eines Projekts."""

    def __init__(self, master: tk.Misc, project: Any):
        super().__init__(master)
        self.project = project
        self.title(f"Probelauf & Gewichte – {project.config.name}")
        self.geometry("1300x820")
        self.queue: "queue.Queue[tuple[Callable, Any]]" = queue.Queue()
        self.busy = False
        self.probe_result: Any = None
        self.weights: list[dict] = []
        self.attn: dict[str, Any] = {}
        self.depth = tk.IntVar(value=3)
        self.backward = tk.BooleanVar(value=True)
        self._build()
        self.after(100, self._poll)

    # ------------------------------------------------------------ Aufbau
    def _build(self) -> None:
        bar = ttk.Frame(self, padding=4)
        bar.pack(fill="x")
        for text, cmd in (("Probelauf (Forward + Backward)", self.run_probe), ("Testlauf (Validierung)", self.run_test),
                          ("Attention-Karten", self.run_attention), ("Gewichte neu laden", self.load_weights)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(bar, text="  Tiefe:").pack(side="left")
        ttk.Spinbox(bar, from_=1, to=8, textvariable=self.depth, width=3,
                    command=self._fill_paths).pack(side="left")
        ttk.Checkbutton(bar, text="Backward (Gradienten)", variable=self.backward).pack(side="left", padx=8)
        self.status = ttk.Label(bar, text="Bereit.", foreground="#555")
        self.status.pack(side="left", padx=12)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        # Wege
        f, self.path_tree, self.path_hist, self.path_map, self.path_info = self._split_tab(
            ("#", "Modul", "Typ", "Ausgabe", "Mittel", "Std", "|max|", "Null %", "Grad", "ms"))
        self.tabs.add(f, text="Wege (Probelauf)")
        self.path_tree.bind("<<TreeviewSelect>>", lambda e: self._show_activation())
        # Gewichte
        f, self.w_tree, self.w_hist, self.w_map, self.w_info = self._split_tab(
            ("Name", "Form", "Mittel", "Std", "|max|", "Null %", "Grad-Norm", "lernt"))
        self.tabs.add(f, text="Gewichte")
        self.w_tree.bind("<<TreeviewSelect>>", lambda e: self._show_weight())
        # Attention
        f = ttk.Frame(self.tabs)
        top = ttk.Frame(f, padding=4)
        top.pack(fill="x")
        ttk.Label(top, text="Schicht:").pack(side="left")
        self.attn_layer = ttk.Combobox(top, width=40, state="readonly")
        self.attn_layer.pack(side="left", padx=4)
        ttk.Label(top, text="Kopf:").pack(side="left")
        self.attn_head = tk.IntVar(value=0)
        ttk.Spinbox(top, from_=0, to=63, textvariable=self.attn_head, width=4, command=self._show_attention).pack(
            side="left")
        self.attn_layer.bind("<<ComboboxSelected>>", lambda e: self._show_attention())
        self.attn_canvas = tk.Canvas(f, background="white")
        self.attn_canvas.pack(fill="both", expand=True)
        self.tabs.add(f, text="Attention")
        # Test
        f = ttk.PanedWindow(self.tabs, orient="horizontal")
        self.test_output = tk.Text(f, wrap="none", font=("TkFixedFont", 9))
        self.test_canvas = tk.Canvas(f, background="white", width=420)
        f.add(self.test_output, weight=2)
        f.add(self.test_canvas, weight=1)
        self.tabs.add(f, text="Test")

    def _split_tab(self, columns: tuple[str, ...]):
        pane = ttk.PanedWindow(self.tabs, orient="horizontal")
        left = ttk.Frame(pane)
        tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, width=70 if c not in ("Modul", "Name") else 280, anchor="w" if c in ("Modul", "Name", "Typ")
                        else "e")
        tree.tag_configure("warn", background="#ffe2c0")
        tree.tag_configure("bad", background="#ffc0c0")
        ys = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ys.set)
        tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        right = ttk.Frame(pane)
        hist = tk.Canvas(right, height=170, background="white")
        hist.pack(fill="x", padx=4, pady=4)
        heat = tk.Canvas(right, height=330, background="white")
        heat.pack(fill="both", expand=True, padx=4, pady=4)
        info = tk.Text(right, height=8, wrap="word", font=("TkFixedFont", 9), background="#f6f8fa")
        info.pack(fill="x", padx=4, pady=4)
        pane.add(left, weight=3)
        pane.add(right, weight=2)
        return pane, tree, hist, heat, info

    # ------------------------------------------------------------ Hintergrund-Rechnung
    def _async(self, title: str, fn: Callable[[], Any], done: Callable[[Any], None]) -> None:
        if self.busy:
            return
        self.busy = True
        self.status.configure(text=f"{title} läuft …")

        def worker():
            try:
                self.queue.put((done, fn()))
            except Exception:  # noqa: BLE001 - Fehler im Fenster anzeigen
                self.queue.put((self._error, traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                cb, value = self.queue.get_nowait()
                self.busy = False
                self.status.configure(text="Fertig.")
                cb(value)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    def _error(self, text: str) -> None:
        self.status.configure(text="Fehler – Details im Dialog.")
        messagebox.showerror("Probelauf", text[-2000:], parent=self)

    def _model_and_batch(self, n: int = 2):
        model = self.project.model if self.project.model is not None else self.project.build()
        inputs = targets = None
        try:
            bundle = self.project.data()
            data = bundle.val if bundle.val is not None else bundle.train
            if isinstance(data, dict) and "inputs" in data:
                inputs = data["inputs"][:n]
                targets = data["targets"][:n] if "targets" in data else None
        except Exception:  # noqa: BLE001 - ohne Daten: Beispiel-Eingabe des Modells
            inputs = None
        if inputs is None:
            inputs = model.example_input(n)
        return model, inputs, targets

    # ------------------------------------------------------------ Aktionen
    def run_probe(self) -> None:
        from ..analysis.probe import probe

        backward = self.backward.get()  # Tk-Variablen nur im Hauptthread lesen

        def fn():
            model, x, y = self._model_and_batch()
            return probe(model, x, y, backward=backward, max_depth=8)

        self._async("Probelauf", fn, self._probe_done)

    def _probe_done(self, result: Any) -> None:
        self.probe_result = result
        self._fill_paths()
        self.tabs.select(0)
        n_warn = len(result.warnings())
        self.status.configure(text=f"Probelauf: {len(result.records)} Module, {result.total_ms:.0f} ms"
                                   + (f", Verlust {result.loss:.4f}" if result.loss is not None else "")
                                   + (f", ⚠ {n_warn} Warnungen" if n_warn else ""))
        self.load_weights()

    def _fill_paths(self) -> None:
        r = self.probe_result
        if r is None:
            return
        self.path_tree.delete(*self.path_tree.get_children())
        for rec in r.records:
            if rec.depth > self.depth.get():
                continue
            s = rec.out_stats
            tag = ("bad",) if any("NaN" in w for w in rec.warnings) else (("warn",) if rec.warnings else ())
            grad = f"{rec.grad_stats['norm']:.3g}" if rec.grad_stats else "–"
            shape = "×".join(map(str, rec.out_shapes[0])) if rec.out_shapes else ""
            self.path_tree.insert("", "end", iid=rec.name, tags=tag, values=(
                rec.order, "  " * (rec.depth - 1) + rec.name, rec.type, shape, f"{s.get('mean', 0):.3g}",
                f"{s.get('std', 0):.3g}", f"{s.get('absmax', 0):.3g}", f"{100 * s.get('zeros', 0):.1f}", grad,
                f"{rec.time_ms:.2f}"))

    def _show_activation(self) -> None:
        from ..analysis.probe import heatmap, histogram

        sel = self.path_tree.selection()
        if not sel or self.probe_result is None:
            return
        name = sel[0]
        rec = next((r for r in self.probe_result.records if r.name == name), None)
        act = self.probe_result.activations.get(name)
        self.path_info.delete("1.0", "end")
        if rec is not None:
            self.path_info.insert("1.0", f"{rec.name} ({rec.type})\nEingabe {rec.in_shapes} → Ausgabe {rec.out_shapes}\n"
                                         f"Ausgabe: {rec.out_stats}\nGradient: {rec.grad_stats}\n"
                                         f"Parameter-Gradientennorm: {rec.param_grad_norm}\n"
                                         + ("\n".join("⚠ " + w for w in rec.warnings)))
        if act is not None:
            draw_histogram(self.path_hist, *histogram(act), title=f"Ausgabe von {name}")
            mat = act[0] if act.dim() >= 3 else act
            draw_heatmap(self.path_map, heatmap(mat), title="Aktivierungen (1. Beispiel: Zeilen = Positionen)")

    def load_weights(self) -> None:
        from ..analysis.probe import weight_stats

        model = self.project.model if self.project.model is not None else self.project.build()
        self.weights = weight_stats(model)
        self.w_tree.delete(*self.w_tree.get_children())
        for i, s in enumerate(self.weights):
            tag = ("bad",) if s.get("nan") or s.get("inf") else (("warn",) if s.get("zeros", 0) > 0.5 else ())
            g = f"{s['grad_norm']:.3g}" if s.get("grad_norm") is not None else "–"
            self.w_tree.insert("", "end", iid=str(i), tags=tag, values=(
                s["name"], "×".join(map(str, s["shape"])), f"{s.get('mean', 0):.3g}", f"{s.get('std', 0):.3g}",
                f"{s.get('absmax', 0):.3g}", f"{100 * s.get('zeros', 0):.1f}", g, "ja" if s["trainable"] else "nein"))

    def _show_weight(self) -> None:
        from ..analysis.probe import get_weight, heatmap, histogram

        sel = self.w_tree.selection()
        if not sel:
            return
        s = self.weights[int(sel[0])]
        model = self.project.model
        w = get_weight(model, s["name"])
        draw_histogram(self.w_hist, *histogram(w), title=f"Verteilung {s['name']}")
        draw_heatmap(self.w_map, heatmap(w), title=f"Gewichtsmatrix {tuple(w.shape)} (verkleinert)")
        self.w_info.delete("1.0", "end")
        self.w_info.insert("1.0", "\n".join(f"{k}: {v}" for k, v in s.items()))

    def run_attention(self) -> None:
        from ..analysis.probe import attention_maps

        def fn():
            model, x, _ = self._model_and_batch(1)
            return attention_maps(model, x)

        self._async("Attention", fn, self._attention_done)

    def _attention_done(self, maps: dict[str, Any]) -> None:
        self.attn = maps
        self.attn_layer.configure(values=list(maps))
        if not maps:
            messagebox.showinfo("Attention", "Keine Softmax-Attention im Modell gefunden.", parent=self)
            return
        self.attn_layer.set(next(iter(maps)))
        self.tabs.select(2)
        self._show_attention()

    def _show_attention(self) -> None:
        from ..analysis.probe import heatmap

        name = self.attn_layer.get()
        if name not in self.attn:
            return
        a = self.attn[name]
        h = max(0, min(self.attn_head.get(), a.shape[0] - 1))
        draw_heatmap(self.attn_canvas, heatmap(a[h], 96, 96),
                     title=f"{name} – Kopf {h}: Zeile = Abfrage-Position, Spalte = beachtete Position")

    def run_test(self) -> None:
        from ..analysis.probe import evaluate_model, evaluation_text

        def fn():
            model, _, _ = self._model_and_batch()
            bundle = self.project.data()
            data = bundle.val if bundle.val is not None else bundle.train
            res = evaluate_model(model, data)
            return res, evaluation_text(res)

        self._async("Testlauf", fn, self._test_done)

    def _test_done(self, value: Any) -> None:
        res, text = value
        self.tabs.select(3)
        self.test_output.delete("1.0", "end")
        self.test_output.insert("1.0", text)
        if "confusion" in res:
            draw_heatmap(self.test_canvas, [[float(v) for v in row] for row in res["confusion"]],
                         title="Konfusionsmatrix (Zeile = Ziel)")
