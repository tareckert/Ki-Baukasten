"""
ki_baukasten.ui.gui
===================

Grafischer Konfigurations-Editor (Tkinter, Teil der Python-Standardbibliothek).

Aufbau des Fensters
-------------------
::

    ┌ Werkzeugleiste: Vorlage ▾ | Öffnen | Speichern | Stufe ▾ | Suchen | Prüfen | Bauen | Speicher | Trainieren ┐
    ├──────────────────────────────┬───────────────────────────────────────────────────────────────────────┤
    │ Baum der Konfiguration       │ Formular des gewählten Knotens                                        │
    │  ▸ model ⟨transformer⟩       │   Name          [ Wert       ]  ?    ← Maus über den Namen = Tooltip  │
    │    ▸ block [Transformer-…]   │   …                                                                    │
    │  ▸ training                  ├───────────────────────────────────────────────────────────────────────┤
    │  …                           │ Erklärung (Was / Warum / Wofür / Tipp, Typ, Bereich, Optionen)        │
    ├──────────────────────────────┴───────────────────────────────────────────────────────────────────────┤
    │ Protokoll (Bauen, Speicherschätzung, Trainingsfortschritt)                                           │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────┘

* **Tooltips**: Maus über einen Einstellungsnamen halten.
* **?**: ausführliche Erklärung inkl. aller Optionen im Erklärungsbereich.
* **Stufe**: blendet Einstellungen nach Erfahrungsstufe ein/aus (nichts wird gesperrt).
* **An / Aus / Eigen**: jede Einstellung hat drei Schalter – empfohlener Wert, abgeschaltet
  (z. B. Dropout 0) oder eigener Wert. Pflicht-Einstellungen kennen nur An/Eigen.
* **Ansicht**: nur eingeschaltete, nur ausgeschaltete oder nur eigene Werte zeigen – so bleibt
  auch bei über tausend Einstellungen alles übersichtlich.
* Komponenten (Aktivierung, Optimierer, Mixer, Belohnungen …) per Auswahlliste
  wechseln; kompatible Argumente bleiben erhalten.
* Training läuft in einem Hintergrund-Thread; die Oberfläche bleibt bedienbar.

Start: ``python -m ki_baukasten gui [preset|datei]``
"""

from __future__ import annotations

import io
import json
import queue
import sys
import threading
import traceback
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - abhängig von der Python-Installation
    raise ImportError(
        "Tkinter ist nicht installiert. Unter Debian/Ubuntu: 'sudo apt install python3-tk'. "
        "Alternativ die Kommandozeile nutzen: python -m ki_baukasten --help"
    ) from exc

from ..core import BaseConfig, ConfigError, Level
from ..core.config import _is_spec

#: Markierung in der Protokoll-Warteschlange: "Hintergrundaufgabe fertig" (Tk darf nur im Hauptthread angefasst werden).
_DONE = object()
_LEVELS = [("Basis", "basic"), ("Fortgeschritten", "advanced"), ("Experte", "expert"), ("Experimentell", "experimental")]
_NONE = "(keine)"
_VIEWS = {"Alle": None, "Nur eingeschaltete": ("an", "eigen"), "Nur ausgeschaltete": ("aus",),
          "Nur eigene Werte": ("eigen",), "Nur schaltbare": "schaltbar"}


class HoverTip:
    """Tooltip-Fenster, das beim Überfahren eines Widgets erscheint."""

    def __init__(self, widget: tk.Widget, text_fn, delay: int = 450):
        self.widget, self.text_fn, self.delay = widget, text_fn, delay
        self.win: tk.Toplevel | None = None
        self.job: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._cancel()
        self.job = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None

    def _show(self):
        text = self.text_fn()
        if not text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{x}+{y}")
        tk.Label(self.win, text=text, justify="left", background="#fffbe6", relief="solid", borderwidth=1,
                 wraplength=560, font=("TkDefaultFont", 9), padx=8, pady=6).pack()

    def _hide(self, _e=None):
        self._cancel()
        if self.win is not None:
            self.win.destroy()
            self.win = None


class _StreamToQueue(io.TextIOBase):
    def __init__(self, q: "queue.Queue[str]"):
        self.q = q

    def write(self, s: str) -> int:
        self.q.put(s)
        return len(s)


class BaukastenGUI(tk.Tk):
    """Hauptfenster (siehe Modul-Dokumentation)."""

    def __init__(self, source: str | None = None):
        super().__init__()
        from ..project import Project

        self.title("KI-Baukasten – Konfiguration")
        self.geometry("1400x900")
        from pathlib import Path

        if source and Path(source).exists():
            self.project = Project.from_file(source)
        else:
            self.project = Project.from_preset(source or "tabellen_klassifikation")
        self.level = tk.StringVar(value="Fortgeschritten")
        self.view = tk.StringVar(value="Alle")
        self.current_path = ""
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.busy = False
        self._row_widgets: dict[str, tk.Widget] = {}
        self._build_ui()
        self.refresh_tree()
        self.select_path("")
        self.after(100, self._poll_log)

    # ================================================================ Aufbau
    def _build_ui(self) -> None:
        from ..core import registry

        bar = ttk.Frame(self, padding=4)
        bar.pack(fill="x")
        ttk.Label(bar, text="Vorlage:").pack(side="left")
        self.preset_var = tk.StringVar(value=self.project.config.name)
        presets = registry("preset").names()
        cb = ttk.Combobox(bar, textvariable=self.preset_var, values=presets, width=28, state="readonly")
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.preset_var.get()))
        HoverTip(cb, lambda: registry("preset").get(self.preset_var.get()).tooltip.format()
                 if self.preset_var.get() in registry("preset") else "")
        for text, cmd in (("Öffnen…", self.open_file), ("Speichern…", self.save_file),
                          ("Modell importieren…", self.import_model)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(bar, text="Stufe:").pack(side="left")
        lv = ttk.Combobox(bar, textvariable=self.level, values=[l[0] for l in _LEVELS], width=15, state="readonly")
        lv.pack(side="left", padx=4)
        lv.bind("<<ComboboxSelected>>", lambda e: (self.refresh_tree(), self.select_path(self.current_path)))
        HoverTip(lv, lambda: "Erfahrungsstufe: blendet Einstellungen ein/aus.\nNichts ist gesperrt – "
                              "'Experimentell' zeigt wirklich alles.")
        ttk.Label(bar, text="Ansicht:").pack(side="left", padx=(8, 0))
        vw = ttk.Combobox(bar, textvariable=self.view, values=list(_VIEWS), width=18, state="readonly")
        vw.pack(side="left", padx=4)
        vw.bind("<<ComboboxSelected>>", lambda e: self.show_form(self.current_path))
        HoverTip(vw, lambda: "Filtert das Formular nach Schalterzustand:\n● an  ○ aus  ◆ eigener Wert.\n"
                              "Gruppen (öffnen →) bleiben immer sichtbar.")
        self.search_var = tk.StringVar()
        se = ttk.Entry(bar, textvariable=self.search_var, width=22)
        se.pack(side="left", padx=(12, 2))
        se.bind("<Return>", lambda e: self.search())
        ttk.Button(bar, text="Suchen", command=self.search).pack(side="left")
        bar = ttk.Frame(self, padding=(4, 0, 4, 4))  # zweite Zeile: Aktionen
        bar.pack(fill="x")
        self.action_buttons = []
        for text, cmd, tip in (
            ("Prüfen", self.validate, "Alle Einstellungen und Abhängigkeiten prüfen."),
            ("Bauen", self.build_model, "Modell bauen und Aufbau anzeigen."),
            ("Speicher", self.memory, "Speicherbedarf schätzen und Empfehlungen anzeigen."),
            ("Trainieren", self.train, "Training im Hintergrund starten (Fortschritt im Protokoll)."),
            ("Aufbau", self.architecture, "Komplette Auflistung des Modells: Module, Formen, Schichten, Anpassungspfade."),
            ("Optimieren", self.advise, "Ratgeber: Vorschläge für Speicher, Tempo und Stabilität (mit Übernahme)."),
            ("Probelauf & Gewichte", self.inspector,
             "Testlauf im Fenster: alle Wege durchs Netz, Aktivierungen, Gradienten, Gewichte, Attention, Metriken."),
            ("Reduzieren / Export", self.reduce_dialog,
             "Zielsystem wählen (Smartphone, Raspberry Pi, CPU, GPU …): Empfehlung, Verkleinern, Exportieren."),
            ("↶ Rückgängig", self.undo, "Letzte Änderung zurücknehmen."),
            ("↷ Wiederholen", self.redo, "Zurückgenommene Änderung wiederherstellen."),
        ):
            b = ttk.Button(bar, text=text, command=cmd)
            b.pack(side="left", padx=2)
            HoverTip(b, lambda t=tip: t)
            self.action_buttons.append(b)
        ttk.Label(bar, text="Tipp: Maus über einen Einstellungsnamen = Kurzerklärung · '?' = alle Details und Optionen",
                  foreground="#666").pack(side="left", padx=12)

        outer = ttk.PanedWindow(self, orient="vertical")
        outer.pack(fill="both", expand=True)
        main = ttk.PanedWindow(outer, orient="horizontal")
        outer.add(main, weight=4)

        tree_frame = ttk.Frame(main)
        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse")
        ys = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        main.add(tree_frame, weight=1)

        right = ttk.PanedWindow(main, orient="vertical")
        main.add(right, weight=3)
        form_outer = ttk.Frame(right)
        self.form_canvas = tk.Canvas(form_outer, highlightthickness=0)
        fs = ttk.Scrollbar(form_outer, orient="vertical", command=self.form_canvas.yview)
        self.form = ttk.Frame(self.form_canvas, padding=8)
        self.form.bind("<Configure>", lambda e: self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all")))
        self.form_canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.form_canvas.configure(yscrollcommand=fs.set)
        self.form_canvas.pack(side="left", fill="both", expand=True)
        fs.pack(side="right", fill="y")
        right.add(form_outer, weight=3)
        self.explain_text = tk.Text(right, height=12, wrap="word", font=("TkDefaultFont", 10), background="#f6f8fa")
        right.add(self.explain_text, weight=1)

        self.log = tk.Text(outer, height=10, wrap="none", font=("TkFixedFont", 9), background="#111", foreground="#ddd")
        outer.add(self.log, weight=1)

    # ================================================================ Hilfen
    def max_level(self) -> str:
        return dict(_LEVELS)[self.level.get()]

    def obj_at(self, path: str) -> Any:
        return self.project.config if not path else self.project.config.get(path)

    def log_write(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def explain(self, path: str) -> None:
        try:
            text = self.project.config.explain(path or None)
        except Exception as exc:  # noqa: BLE001 - reine Anzeige
            text = f"(keine Erklärung: {exc})"
        self.explain_text.delete("1.0", "end")
        self.explain_text.insert("1.0", text)

    def _poll_log(self) -> None:  # läuft im Hauptthread (Tk-after-Schleife)
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item is _DONE:
                    self._done()
                else:
                    self.log_write(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ================================================================ Baum
    def refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        cfg = self.project.config
        self.tree.insert("", "end", iid="@", text=f"{cfg.title()} – {cfg.name}", open=True)
        self._add_children("@", "", cfg, Level.BASIC)

    def _iid(self, path: str) -> str:
        return path or "@"

    def _add_children(self, parent_iid: str, prefix: str, obj: Any, level: Level) -> None:
        target = obj.args if _is_spec(obj) else obj
        if not isinstance(target, BaseConfig):
            return
        limit = Level.parse(self.max_level())
        for name, param in target._iter_params():
            eff = param.level if param.level.rank > level.rank else level
            if not eff.visible_at(limit):
                continue
            value = target._values[name]
            path = f"{prefix}{name}"
            if isinstance(value, BaseConfig):
                mark = ("● " if value._values.get("enabled") else "○ ") if "enabled" in type(value)._params else ""
                self.tree.insert(parent_iid, "end", iid=path, text=f"{mark}{name}  [{value.title()}]")
                self._add_children(path, path + ".", value, eff)
            elif _is_spec(value):
                comp_level = value.component.level
                lv = comp_level if comp_level.rank > eff.rank else eff
                self.tree.insert(parent_iid, "end", iid=path, text=f"{name}  ⟨{value.name}⟩")
                self._add_children(path, path + ".", value, lv)
            elif isinstance(value, list) and getattr(param, "kind", "") in ("config_list", "component_list"):
                self.tree.insert(parent_iid, "end", iid=path, text=f"{name}  [{len(value)} Elemente]")
                for i, item in enumerate(value):
                    ip = f"{path}.{i}"
                    label = f"{i}: ⟨{item.name}⟩" if _is_spec(item) else f"{i}: {getattr(item, 'name', item.title())}"
                    self.tree.insert(path, "end", iid=ip, text=label)
                    self._add_children(ip, ip + ".", item, eff)

    def _on_tree_select(self, _e=None) -> None:
        sel = self.tree.selection()
        if sel:
            iid = sel[0]
            self.show_form("" if iid == "@" else iid)

    def select_path(self, path: str) -> None:
        iid = self._iid(path)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)
        self.show_form(path)

    # ================================================================ Formular
    def show_form(self, path: str) -> None:
        self.current_path = path
        for w in self.form.winfo_children():
            w.destroy()
        self._row_widgets.clear()
        obj = self.obj_at(path)
        self.explain(path)
        title = ttk.Label(self.form, text=path or "Projekt", font=("TkDefaultFont", 12, "bold"))
        title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        row = 1
        prefix = f"{path}." if path else ""
        if isinstance(obj, list):
            self._list_form(path, obj, row)
            return
        if _is_spec(obj):
            row = self._component_chooser(path, obj, row)
            target = obj.args
        else:
            target = obj
        limit = Level.parse(self.max_level())
        view = _VIEWS.get(self.view.get())
        hidden = 0
        for name, param in target._iter_params():
            if not param.level.visible_at(limit):
                continue
            if view is not None and not self._view_match(param, target._values[name], view):
                hidden += 1
                continue
            self._param_row(f"{prefix}{name}", name, param, target._values[name], row)
            row += 1
        if hidden:
            ttk.Label(self.form, text=f"({hidden} Einstellungen durch die Ansicht '{self.view.get()}' ausgeblendet)",
                      foreground="#888").grid(row=row, column=0, columnspan=4, sticky="w", pady=(8, 0))

    @staticmethod
    def _view_match(param: Any, value: Any, view: Any) -> bool:
        from ..core import UNSET, off_value, state_of

        if getattr(param, "kind", "value") in ("nested", "config_list"):
            return True
        if view == "schaltbar":
            return off_value(param) is not UNSET
        return state_of(param, value) in view

    def _label(self, parent: tk.Widget, text: str, path: str, param: Any, row: int, off: bool = False) -> None:
        badge = "" if param.level is Level.BASIC else f"  ·{param.level.label}"
        lbl = ttk.Label(parent, text=text + badge, cursor="question_arrow", foreground="#999" if off else "")
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        HoverTip(lbl, lambda: f"{path}\n\n" + param.tooltip.format(width=80) +
                 (f"\n\nTyp: {param.type_label()}" + (f"   {param.constraint_label()}" if param.constraint_label() else "")))
        ttk.Button(parent, text="?", width=2, command=lambda: self.explain(path)).grid(row=row, column=2, padx=4)

    def _param_row(self, path: str, name: str, param: Any, value: Any, row: int) -> None:
        from ..core import state_of

        kind = getattr(param, "kind", "value")
        leaf = kind not in ("nested", "config_list")
        self._label(self.form, name, path, param, row, off=leaf and state_of(param, value) == "aus")
        if leaf:
            self._switch_buttons(path, param, value, row)
        if kind in ("nested", "config_list", "component_list"):
            text = f"{value.title()} öffnen →" if isinstance(value, BaseConfig) else \
                f"{len(value or [])} Elemente öffnen →" if isinstance(value, list) else "öffnen →"
            ttk.Button(self.form, text=text, command=lambda: self.select_path(path)).grid(row=row, column=1, sticky="w")
            return
        if kind == "component":
            names = ([_NONE] if param.nullable else []) + list(param.choices or [])
            var = tk.StringVar(value=value.name if value is not None else _NONE)
            cb = ttk.Combobox(self.form, textvariable=var, values=names, width=34, state="readonly")
            cb.grid(row=row, column=1, sticky="w")
            cb.bind("<<ComboboxSelected>>", lambda e: self._set_component(path, var.get()))
            if value is not None and type(value.args)._params:
                ttk.Button(self.form, text="Argumente →", command=lambda: self.select_path(path)).grid(row=row, column=4)
            self._row_widgets[path] = cb
            return
        is_bool = bool in param.types and not param.nullable
        if is_bool:
            var = tk.BooleanVar(value=bool(value))
            w = ttk.Checkbutton(self.form, variable=var, command=lambda: self._commit(path, var.get()))
        elif param.choices:
            vals = ([_NONE] if param.nullable else []) + [str(c) for c in param.choices]
            var = tk.StringVar(value=_NONE if value is None else str(value))
            w = ttk.Combobox(self.form, textvariable=var, values=vals, width=34)
            w.bind("<<ComboboxSelected>>", lambda e: self._commit_text(path, var.get(), param))
            w.bind("<Return>", lambda e: self._commit_text(path, var.get(), param))
        else:
            shown = "" if value is None else (json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                                              else str(value))
            var = tk.StringVar(value=shown)
            w = ttk.Entry(self.form, textvariable=var, width=36)
            w.bind("<Return>", lambda e: self._commit_text(path, var.get(), param))
            w.bind("<FocusOut>", lambda e: self._commit_text(path, var.get(), param, quiet=True))
        w.grid(row=row, column=1, sticky="w")
        w.bind("<FocusIn>", lambda e: self.explain(path), add="+")
        self._row_widgets[path] = w

    def _switch_buttons(self, path: str, param: Any, value: Any, row: int) -> None:
        """Drei Schalter je Einstellung: An (empfohlen) · Aus · Eigen (benutzerdefiniert)."""
        from ..core import UNSET, off_value, on_value, state_of
        from ..core.switches import _show, off_label

        off, on = off_value(param), on_value(param, self.project._switch_memory.get(path, UNSET))
        frame = ttk.Frame(self.form)
        frame.grid(row=row, column=3, sticky="w", padx=(6, 0))
        var = tk.StringVar(value=state_of(param, value))
        tips = {
            "an": f"An: empfohlener Wert {_show(on)}" if on is not UNSET else
                  "An: kein empfohlener Wert hinterlegt – bitte links einen eigenen Wert eingeben.",
            "aus": f"Aus: {off_label(param)}" if off is not UNSET else
                   "Pflicht-Einstellung: kann nicht ausgeschaltet werden (nur An oder Eigen).",
            "eigen": "Eigen: benutzerdefinierter Wert – im Feld links eingeben (der zuletzt eigene Wert wird "
                     "beim Aus-Schalten gemerkt und hier wiederhergestellt).",
        }
        for key, text in (("an", "An"), ("aus", "Aus"), ("eigen", "Eigen")):
            rb = ttk.Radiobutton(frame, text=text, value=key, variable=var, style="Toolbutton",
                                 command=lambda k=key: self._switch(path, k))
            if key == "aus" and off is UNSET:
                rb.state(["disabled"])
            rb.pack(side="left")
            HoverTip(rb, lambda k=key: tips[k])

    def _switch(self, path: str, state: str) -> None:
        try:
            changed = self.project.switch(path, state)
        except ConfigError as exc:
            if state == "an":
                messagebox.showinfo("Kein empfohlener Wert", f"{exc}\n\nBitte im Feld einen eigenen Wert eingeben.")
            else:
                messagebox.showerror("Schalter", str(exc))
            changed = []
        for c in changed:
            self.log_write(f"✓ {c} → {state}: {self.project.config.get(c)!r}\n")
        current = self.current_path
        self.refresh_tree()
        self.select_path(current if self._exists(current) else "")
        w = self._row_widgets.get(path)
        if state == "eigen" and w is not None:
            w.focus_set()

    def _component_chooser(self, path: str, spec: Any, row: int) -> int:
        from ..core import registry

        reg = registry(spec.registry)
        ttk.Label(self.form, text=f"Komponente ({reg.title})").grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=spec.name)
        cb = ttk.Combobox(self.form, textvariable=var, values=reg.names(), width=34, state="readonly")
        cb.grid(row=row, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self._set_component(path, var.get(), via_name=True))
        HoverTip(cb, lambda: reg.get(var.get()).tooltip.format(width=80))
        desc = ttk.Label(self.form, text=spec.tooltip.short(), foreground="#555", wraplength=520)
        desc.grid(row=row + 1, column=0, columnspan=4, sticky="w", pady=(2, 8))
        return row + 2

    def _list_form(self, path: str, items: list, row: int) -> None:
        ref = self.project.config.param_at(path)
        param = ref.param
        ttk.Label(self.form, text=param.tooltip.short(), wraplength=600).grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1
        lb = tk.Listbox(self.form, height=min(12, max(3, len(items))), width=60)
        for i, it in enumerate(items):
            lb.insert("end", f"{i}: ⟨{it.name}⟩ {it.args.to_dict(only_changed=True)}" if _is_spec(it) else
                      f"{i}: {it.to_dict(only_changed=True)}")
        lb.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
        lb.bind("<Double-Button-1>", lambda e: lb.curselection() and self.select_path(f"{path}.{lb.curselection()[0]}"))
        row += 1
        btns = ttk.Frame(self.form)
        btns.grid(row=row, column=0, columnspan=4, sticky="w")
        if getattr(param, "kind", "") == "component_list":
            from ..core import registry

            var = tk.StringVar(value=registry(param.registry_kind).names()[0])
            ttk.Combobox(btns, textvariable=var, values=registry(param.registry_kind).names(), width=24,
                         state="readonly").pack(side="left")
            ttk.Button(btns, text="Hinzufügen", command=lambda: self._list_add(path, var.get())).pack(side="left", padx=4)
        else:
            ttk.Button(btns, text="Element hinzufügen", command=lambda: self._list_add(path, None)).pack(side="left")
        ttk.Button(btns, text="Ausgewähltes entfernen",
                   command=lambda: lb.curselection() and self._list_remove(path, lb.curselection()[0])).pack(side="left",
                                                                                                          padx=4)

    # ================================================================ Änderungen
    def _commit_text(self, path: str, text: str, param: Any, quiet: bool = False) -> None:
        value: Any = text
        if text in ("", _NONE) and param.nullable:
            value = None
        elif (list in param.types or dict in param.types) and text.strip():
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                pass
        self._commit(path, value, quiet)

    def _commit(self, path: str, value: Any, quiet: bool = False) -> None:
        try:
            old = self.project.config.get(path)
            if old == value:
                return
            self.project.set(path, value)
            self.log_write(f"✓ {path} = {self.project.config.get(path)!r}\n")
        except ConfigError as exc:
            if not quiet:
                messagebox.showerror("Ungültiger Wert", str(exc))
            self.show_form(self.current_path)

    def _set_component(self, path: str, name: str, via_name: bool = False) -> None:
        try:
            if name == _NONE:
                self.project.set(path, None)
            elif via_name or self.project.config.get(path) is not None:
                self.project.set(f"{path}.name", name)  # kompatible Argumente bleiben erhalten
            else:
                self.project.set(path, name)
            self.log_write(f"✓ {path} → ⟨{name}⟩\n")
        except ConfigError as exc:
            messagebox.showerror("Komponente", str(exc))
        current = self.current_path
        self.refresh_tree()
        self.select_path(current if self._exists(current) else "")

    def _exists(self, path: str) -> bool:
        try:
            self.obj_at(path)
            return True
        except Exception:
            return False

    def _list_add(self, path: str, component: str | None) -> None:
        items = list(self.project.config.get(path))
        if component is not None:
            items.append(component)
        else:
            items.append(self.project.config.param_at(path).param.new_item())
        self.project.set(path, items)
        self.refresh_tree()
        self.select_path(path)

    def _list_remove(self, path: str, index: int) -> None:
        items = list(self.project.config.get(path))
        items.pop(index)
        try:
            self.project.set(path, items)
        except ConfigError as exc:
            messagebox.showerror("Liste", str(exc))
        self.refresh_tree()
        self.select_path(path)

    # ================================================================ Aktionen
    def load_preset(self, name: str) -> None:
        from ..project import Project

        self.project = Project.from_preset(name)
        self.refresh_tree()
        self.select_path("")
        self.log_write(f"Vorlage geladen: {name}\n")

    def open_file(self) -> None:
        from ..project import Project

        path = filedialog.askopenfilename(filetypes=[("Konfiguration", "*.json *.yaml *.yml"), ("Alle", "*")])
        if path:
            self.project = Project.from_file(path)
            self.refresh_tree()
            self.select_path("")
            self.log_write(f"Geöffnet: {path}\n")

    def save_file(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=f"{self.project.config.name}.json",
                                            filetypes=[("JSON", "*.json"), ("YAML", "*.yaml")])
        if path:
            self.project.config.save(path)
            self.log_write(f"Gespeichert: {path}\n")

    def search(self) -> None:
        text = self.search_var.get().strip().lower()
        if not text:
            return
        found = {path for path, _ in self.project.search(text, limit=500)}
        hits = [ref for ref in self.project.config.walk() if ref.path in found]
        if not hits:
            messagebox.showinfo("Suche", f"Nichts gefunden für '{text}'.")
            return
        win = tk.Toplevel(self)
        win.title(f"Suche: {text} ({len(hits)} Treffer)")
        lb = tk.Listbox(win, width=110, height=min(25, len(hits)))
        for ref in hits:
            lb.insert("end", f"{ref.path}   —   {ref.param.tooltip.short()}")
        lb.pack(fill="both", expand=True)

        def go(_e=None):
            if not lb.curselection():
                return
            ref = hits[lb.curselection()[0]]
            parent = ref.path.rsplit(".", 1)[0] if "." in ref.path else ""
            self.level.set("Experimentell")
            self.refresh_tree()
            self.select_path(parent if self._exists(parent) else "")
            self.explain(ref.path)

        lb.bind("<Double-Button-1>", go)
        lb.bind("<Return>", go)

    def validate(self) -> None:
        problems = self.project.config.problems()
        if problems:
            messagebox.showwarning("Prüfung", "Probleme:\n\n" + "\n".join(f"• {p}" for p in problems))
        else:
            messagebox.showinfo("Prüfung", "✓ Konfiguration ist gültig.")

    def _run_bg(self, title: str, fn) -> None:
        if self.busy:
            messagebox.showinfo("Bitte warten", "Es läuft bereits eine Aufgabe.")
            return
        self.busy = True
        for b in self.action_buttons:
            b.state(["disabled"])
        self.log_write(f"\n=== {title} ===\n")

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _StreamToQueue(self.log_queue)
            try:
                result = fn()
                if result:
                    print(result)
            except Exception:  # noqa: BLE001 - Fehler ins Protokoll
                print(traceback.format_exc())
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self.log_queue.put(_DONE)  # Hauptthread übernimmt (siehe _poll_log)

        threading.Thread(target=worker, daemon=True).start()

    def _done(self) -> None:
        self.busy = False
        for b in self.action_buttons:
            b.state(["!disabled"])
        self.refresh_tree()
        self.select_path(self.current_path if self._exists(self.current_path) else "")

    def build_model(self) -> None:
        self._run_bg("Bauen", lambda: (self.project.build(), self.project.summary())[1])

    def memory(self) -> None:
        self._run_bg("Speicherschätzung", lambda: str(self.project.memory_report()))

    def architecture(self) -> None:
        self._run_bg("Aufbau", lambda: self.project.architecture("text", max_depth=3))

    def advise(self) -> None:
        def job():
            tips = self.project.advise()
            self._last_tips = tips
            if not tips:
                return "Keine Vorschläge – die Konfiguration sieht gut aus."
            return "\n\n".join(str(t) for t in tips) + "\n\nÜbernehmen: Knopf 'Optimieren' erneut nach Auswahl " \
                "im Dialog (nach Abschluss)."

        self._run_bg("Ratgeber", job)
        self.after(300, self._offer_advice)

    def _offer_advice(self) -> None:
        if self.busy:
            self.after(300, self._offer_advice)
            return
        tips = [t for t in getattr(self, "_last_tips", []) if t.changes]
        if not tips:
            return
        win = tk.Toplevel(self)
        win.title("Vorschläge übernehmen")
        ttk.Label(win, text="Auswählen und übernehmen (rückgängig mit ↶):", padding=6).pack(anchor="w")
        vars_ = []
        for t in tips:
            v = tk.BooleanVar(value=t.priority == 1)
            cb = ttk.Checkbutton(win, text=f"[{t.category}] {t.title}", variable=v)
            cb.pack(anchor="w", padx=8)
            HoverTip(cb, lambda t=t: f"{t.reason}\n{t.effect}\n" + ", ".join(f"{k}={v!r}" for k, v in t.changes.items()))
            vars_.append((t, v))

        def apply() -> None:
            keys = [t.key for t, v in vars_ if v.get()]
            if keys:
                done = self.project.apply_advice(keys)
                self.log_write("✓ übernommen: " + ", ".join(a.title for a in done) + "\n")
                self.refresh_tree()
                self.select_path(self.current_path if self._exists(self.current_path) else "")
            win.destroy()

        ttk.Button(win, text="Übernehmen", command=apply).pack(pady=6)

    def inspector(self) -> None:
        from .inspector import InspectorWindow

        InspectorWindow(self, self.project)

    def import_model(self) -> None:
        from tkinter import simpledialog

        path = filedialog.askopenfilename(title="Modell importieren", filetypes=[
            ("PyTorch", "*.pt2 *.pt *.pth *.ts *.bin *.safetensors"), ("Alle", "*")])
        if not path:
            return
        shape = simpledialog.askstring("Beispiel-Eingabe", "Form einer Eingabe (z. B. 1,16 für Text oder "
                                       "1,3,224,224 für Bilder):", initialvalue="1,16", parent=self)
        dtype = "long" if messagebox.askyesno("Eingabetyp", "Sind die Eingaben Token-IDs (ganze Zahlen)?",
                                             parent=self) else "float32"
        try:
            from ..project import Project
            self.project = Project.import_model(path, input_shape=[int(v) for v in (shape or "1,16").split(",")],
                                                input_dtype=dtype)
            self.log_write(f"✓ importiert: {path} ({self.project.model.import_format})\n")
            self.refresh_tree()
            self.select_path("")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Import", str(exc), parent=self)

    def reduce_dialog(self) -> None:
        from ..analysis.targets import TARGETS

        win = tk.Toplevel(self)
        win.title("Reduzieren für ein Zielsystem")
        win.geometry("900x520")
        top = ttk.Frame(win, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Zielsystem:").pack(side="left")
        names = {p.title: k for k, p in TARGETS.items()}
        var = tk.StringVar(value=TARGETS["smartphone"].title)
        cb = ttk.Combobox(top, textvariable=var, values=list(names), width=50, state="readonly")
        cb.pack(side="left", padx=4)
        text = tk.Text(win, wrap="word", font=("TkFixedFont", 9))
        text.pack(fill="both", expand=True, padx=6, pady=6)

        def show(_e=None):
            text.delete("1.0", "end")
            try:
                text.insert("1.0", str(self.project.recommend_for(names[var.get()])) + "\n\n"
                            + self.project.deployment_overview(list(TARGETS)))
            except Exception as exc:  # noqa: BLE001
                text.insert("1.0", f"Empfehlung nicht möglich: {exc}")

        def apply():
            key = names[var.get()]

            def job():
                reduced = self.project.reduce_for(key)
                self.project = reduced
                rep = reduced.last_compression
                return str(rep) if rep is not None else "Keine Kompression nötig."

            win.destroy()
            self._run_bg(f"Reduzieren für {key}", job)

        def export():
            path = filedialog.asksaveasfilename(title="Exportieren", defaultextension=".pt2",
                                                filetypes=[("torch.export", "*.pt2"), ("TorchScript", "*.ts"),
                                                           ("ONNX", "*.onnx"), ("Gewichte", "*.pt")], parent=win)
            if not path:
                return
            fmt = {".pt2": "export", ".ts": "torchscript", ".onnx": "onnx", ".pt": "state_dict"}.get(
                path[path.rfind("."):], "export")
            self._run_bg("Export", lambda: f"Exportiert: {self.project.export(path, fmt)}")

        cb.bind("<<ComboboxSelected>>", show)
        btns = ttk.Frame(win, padding=6)
        btns.pack(fill="x")
        ttk.Button(btns, text="Empfehlung anwenden (verkleinern)", command=apply).pack(side="left", padx=4)
        b = ttk.Button(btns, text="Notfall-Ausführung…", command=lambda: self.fallback_dialog(names[var.get()]))
        b.pack(side="left", padx=4)
        HoverTip(b, lambda: "Passt das Modell trotz Verkleinern nicht? Dann stückweise ausführen: Schicht für Schicht "
                            "von der SSD, Blöcke im RAM, Teile auf der CPU, LISA, QLoRA …")
        ttk.Button(btns, text="Exportieren…", command=export).pack(side="left", padx=4)
        show()

    def fallback_dialog(self, target: str | None = None) -> None:
        """Notfall-Optionen (stückweise Ausführung) für ein Zielsystem bzw. Budget wählen und einschalten."""
        from ..analysis.targets import TARGETS

        win = tk.Toplevel(self)
        win.title("Notfall-Ausführung – wenn das Modell nicht passt")
        win.geometry("980x560")
        top = ttk.Frame(win, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Zielsystem:").pack(side="left")
        tvar = tk.StringVar(value=target or "laptop_gpu")
        ttk.Combobox(top, textvariable=tvar, values=list(TARGETS), width=20, state="readonly").pack(side="left", padx=4)
        ttk.Label(top, text="oder Budget (MB):").pack(side="left", padx=(8, 0))
        bvar = tk.StringVar(value="")
        ttk.Entry(top, textvariable=bvar, width=8).pack(side="left", padx=4)
        mvar = tk.StringVar(value="Inferenz")
        ttk.Combobox(top, textvariable=mvar, values=["Inferenz", "Training"], width=10, state="readonly").pack(
            side="left", padx=4)
        cols = ("passt", "geraet", "tempo", "titel")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=12)
        for c, w, t in zip(cols, (50, 110, 90, 640), ("passt", "Gerät (MB)", "langsamer", "Option")):
            tree.heading(c, text=t)
            tree.column(c, width=w, anchor="w")
        tree.pack(fill="both", expand=True, padx=6)
        detail = tk.Text(win, height=8, wrap="word", font=("TkFixedFont", 9))
        detail.pack(fill="x", padx=6, pady=4)
        options: dict[str, Any] = {}

        def params() -> tuple[str | None, float | None, bool]:
            budget = float(bvar.get().replace(",", ".")) if bvar.get().strip() else None
            return (None if budget is not None else tvar.get()), budget, mvar.get() == "Training"

        def refresh(_e=None):
            tree.delete(*tree.get_children())
            options.clear()
            try:
                tgt, budget, training = params()
                for o in self.project.fallback_options(tgt, budget, training):
                    options[o.key] = o
                    tree.insert("", "end", iid=o.key, values=("✓" if o.fits else "✗", f"{o.peak_mb:.1f}",
                                                             f"{o.slowdown:.1f}×", o.title))
            except Exception as exc:  # noqa: BLE001
                detail.delete("1.0", "end")
                detail.insert("1.0", f"Nicht möglich: {exc}")

        def select(_e=None):
            sel = tree.selection()
            if sel:
                detail.delete("1.0", "end")
                detail.insert("1.0", str(options[sel[0]]))

        def apply():
            sel = tree.selection()
            if not sel:
                return
            tgt, budget, training = params()
            try:
                chosen = self.project.apply_fallback(sel[0], tgt, budget, training)
            except ConfigError as exc:
                messagebox.showerror("Notfall-Ausführung", str(exc), parent=win)
                return
            self.log_write(f"✓ Notfall-Ausführung '{chosen.key}' eingeschaltet: {chosen.title}\n"
                           + (f"  Nutzung: {chosen.usage}\n" if chosen.usage else ""))
            self.refresh_tree()
            self.select_path(self.current_path if self._exists(self.current_path) else "")

        tree.bind("<<TreeviewSelect>>", select)
        bar = ttk.Frame(win, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Neu berechnen", command=refresh).pack(side="left", padx=4)
        ttk.Button(bar, text="Gewählte Option einschalten", command=apply).pack(side="left", padx=4)
        ttk.Label(bar, text="Zahlen sind grobe Schätzungen · Rückgängig mit ↶", foreground="#666").pack(side="left",
                                                                                                    padx=12)
        refresh()

    def undo(self) -> None:
        if self.project.undo():
            self.log_write("↶ rückgängig\n")
            self.refresh_tree()
            self.select_path(self.current_path if self._exists(self.current_path) else "")

    def redo(self) -> None:
        if self.project.redo():
            self.log_write("↷ wiederhergestellt\n")
            self.refresh_tree()
            self.select_path(self.current_path if self._exists(self.current_path) else "")

    def train(self) -> None:
        def job():
            self.project.train()
            if self.project.config.data.name == "anomaly":
                print("Anomalie-Erkennung:", self.project.anomaly_report())
            path = self.project.save()
            return f"Gespeichert in: {path}"

        self._run_bg("Training", job)


def main(source: str | None = None) -> None:
    app = BaukastenGUI(source)
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else None)
