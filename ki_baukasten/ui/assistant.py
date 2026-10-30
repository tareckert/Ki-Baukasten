"""
ki_baukasten.ui.assistant
=========================

Fenster der GUI für den **Ziel-Assistenten** und den **Techniken-Katalog**.

* ``GoalWizard`` – Zweck, wichtige Punkte und Ziele eingeben → empfohlene und abgeratene Modelle
  mit Begründung → neues Projekt erstellen *oder* die Ziele des aktuellen Projekts ändern.
* ``TechniqueWindow`` – alle Techniken nach Kategorien (Reiter) mit Häkchen zum An-/Abwählen,
  Status (● an · ○ aus · ◐ teilweise · – passt nicht), Suche und Tooltips (wer nutzt es, Hinweise).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from ..core import ConfigError


class _Scrollable(ttk.Frame):
    """Rahmen mit senkrechter Bildlaufleiste (für lange Listen)."""

    def __init__(self, parent: tk.Widget, **kw: Any):
        super().__init__(parent, **kw)
        canvas = tk.Canvas(self, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas, padding=6)
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")


# ===========================================================================
# Ziel-Assistent
# ===========================================================================

class GoalWizard(tk.Toplevel):
    """Zweck & Ziele → Empfehlungen; neues Projekt oder Ziele des aktuellen Projekts ändern."""

    def __init__(self, app: Any, check_current: bool = False):
        super().__init__(app)
        from ..analysis.goals import PRIORITIES, TASKS
        from ..analysis.targets import TARGETS

        self.app = app
        self.check_current = check_current
        self.title("Ziel-Assistent – wofür soll die KI genutzt werden?")
        self.geometry("1250x760")
        goals = app.project.config.goals if check_current else None
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8)
        right.pack(side="left", fill="both", expand=True)

        def row(label: str, widget: tk.Widget, tip: str = "") -> None:
            ttk.Label(left, text=label).pack(anchor="w", pady=(6, 0))
            widget.pack(anchor="w", fill="x")
            if tip:
                from .gui import HoverTip
                HoverTip(widget, lambda: tip)

        self.task_labels = {f"{v}  ({k})": k for k, v in TASKS.items()}
        current_task = goals.task if goals is not None else "unbekannt"
        self.task = tk.StringVar(value=next(lbl for lbl, k in self.task_labels.items() if k == current_task))
        row("Aufgabe", ttk.Combobox(left, textvariable=self.task, values=list(self.task_labels), width=58,
                                    state="readonly"), "Was soll die KI tun? 'noch offen' = aus der Beschreibung ableiten.")
        self.desc = tk.Text(left, height=4, width=60, wrap="word")
        if goals is not None and goals.description:
            self.desc.insert("1.0", goals.description)
        row("Beschreibung: wofür, wichtige Punkte, Einschränkungen", self.desc,
            "Stichwörter wie 'Handy', 'Echtzeit', 'Datenschutz', 'lange Texte', 'erklärbar' werden erkannt.")
        ttk.Label(left, text="Wichtig ist (Mehrfachauswahl):").pack(anchor="w", pady=(8, 0))
        grid = ttk.Frame(left)
        grid.pack(anchor="w")
        self.prio_vars: dict[str, tk.BooleanVar] = {}
        chosen = set(goals.priorities) if goals is not None else set()
        for i, (key, text) in enumerate(PRIORITIES.items()):
            v = tk.BooleanVar(value=key in chosen)
            ttk.Checkbutton(grid, text=text, variable=v).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 12))
            self.prio_vars[key] = v

        def combo(label: str, key: str, values: list[str], default: str, tip: str = "") -> tk.StringVar:
            var = tk.StringVar(value=str(getattr(goals, key)) if goals is not None and getattr(goals, key) is not None
                               else default)
            row(label, ttk.Combobox(left, textvariable=var, values=values, width=28, state="readonly"), tip)
            return var

        self.data_size = combo("Datenmenge", "data_size", ["winzig", "klein", "mittel", "gross"], "mittel",
                               "winzig < 1.000 · klein < 10.000 · mittel < 1 Mio. · gross darüber")
        self.labels = combo("Labels (richtige Antworten)", "labels", ["vorhanden", "wenige", "keine"], "vorhanden")
        self.seq = combo("Sequenzlänge (Text/Zeitreihe)", "sequence_length",
                         ["keine", "kurz", "mittel", "lang", "sehr_lang"], "keine")
        self.target = combo("Zielgerät", "target", ["(keins)"] + list(TARGETS), "(keins)",
                            "Wo die KI laufen soll – zu große Modelle werden abgewertet, Reduktion wird eingeplant.")
        self.hw = combo("Trainings-Hardware", "training_hardware",
                        ["cpu", "laptop_gpu", "gaming_gpu", "apple_silicon", "rechenzentrum_gpu", "cluster"],
                        "laptop_gpu")
        self.pretrained = tk.BooleanVar(value=bool(goals.pretrained) if goals is not None else False)
        ttk.Checkbutton(left, text="vortrainiertes Modell anpassen (Feintuning)", variable=self.pretrained).pack(
            anchor="w", pady=(8, 0))
        self.apply_tech = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="passende Techniken automatisch einschalten", variable=self.apply_tech).pack(
            anchor="w")
        btns = ttk.Frame(left)
        btns.pack(anchor="w", pady=10)
        ttk.Button(btns, text="Bewerten", command=self.evaluate).pack(side="left")
        if check_current:
            ttk.Button(btns, text="Ziele übernehmen", command=self.apply_to_current).pack(side="left", padx=4)
        ttk.Button(btns, text="Neues Projekt aus Empfehlung", command=self.create).pack(side="left", padx=4)

        ttk.Label(right, text="Empfehlungen (Auswahl für 'Neues Projekt'):", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w")
        self.rec_list = tk.Listbox(right, height=6, exportselection=False)
        self.rec_list.pack(fill="x")
        self.rec_list.bind("<<ListboxSelect>>", lambda e: self._show_selected())
        self.text = tk.Text(right, wrap="word", font=("TkFixedFont", 9))
        self.text.pack(fill="both", expand=True, pady=(6, 0))
        self.report = None
        self.evaluate()

    def goals(self) -> dict[str, Any]:
        target = self.target.get()
        return {"task": self.task_labels[self.task.get()], "description": self.desc.get("1.0", "end").strip(),
                "priorities": [k for k, v in self.prio_vars.items() if v.get()], "data_size": self.data_size.get(),
                "labels": self.labels.get(), "sequence_length": self.seq.get(),
                "target": None if target in ("(keins)", "None", "") else target,
                "training_hardware": self.hw.get(), "pretrained": self.pretrained.get(),
                "use_description": True}

    def evaluate(self) -> None:
        from ..analysis.goals import evaluate_goals

        try:
            self.report = evaluate_goals(self.goals(), project=self.app.project if self.check_current else None)
        except ConfigError as exc:
            messagebox.showerror("Ziele", str(exc), parent=self)
            return
        self.rec_list.delete(0, "end")
        for s in self.report.recommended:
            self.rec_list.insert("end", f"{s.preset}  ({s.score:+.1f})  – {s.why}")
        if self.report.recommended:
            self.rec_list.selection_set(0)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", str(self.report))

    def _show_selected(self) -> None:
        sel = self.rec_list.curselection()
        if sel and self.report is not None:
            s = self.report.recommended[sel[0]]
            self.text.delete("1.0", "end")
            self.text.insert("1.0", str(s) + "\n\n" + str(self.report))

    def create(self) -> None:
        from ..project import Project

        sel = self.rec_list.curselection()
        try:
            project = Project.from_goals(self.goals(), choice=sel[0] if sel else 0,
                                         apply_techniques=self.apply_tech.get())
        except ConfigError as exc:
            messagebox.showerror("Ziele", str(exc), parent=self)
            return
        self.app.set_project(project, f"Neues Projekt nach Zielen: Vorlage '{project.config.name}'")
        self.app.log_write(str(project.goal_report) + "\n")
        self.destroy()

    def apply_to_current(self) -> None:
        g = self.goals()
        g["use_description"] = False
        report = self.app.project.set_goals(**g)
        if self.apply_tech.get():
            done = self.app.project.apply_goal_techniques()
            if done:
                self.app.log_write(f"✓ Techniken eingeschaltet: {', '.join(done)}\n")
        self.app.log_write(f"✓ Ziele übernommen – aktuelles Modell: {report.current_verdict}\n")
        self.app.refresh_tree()
        self.evaluate()


# ===========================================================================
# Techniken-Katalog
# ===========================================================================

class TechniqueWindow(tk.Toplevel):
    """Alle Techniken nach Kategorien – an/abwählbar, mit Status und Erklärungen."""

    MARK = {"an": "●", "aus": "○", "teilweise": "◐", "nicht_anwendbar": "–"}
    SHORT = {"stabilitaet": "Stabilität", "generalisierung": "Auswendiglernen", "tempo": "Tempo",
             "grosse_modelle": "Große Modelle", "cluster": "Cluster", "automl": "AutoML", "schwarm": "Schwarm",
             "evolution": "Evolution", "daten": "Daten", "datenschutz": "Datenschutz", "ueberwachung": "Überwachung",
             "kompression": "Kompression", "experimentell": "Experimentell"}

    def __init__(self, app: Any):
        super().__init__(app)
        from ..techniques import CATEGORIES

        self.app = app
        self.title("Techniken – an- und abwählen")
        self.geometry("1320x720")
        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Suche:").pack(side="left")
        self.query = tk.StringVar()
        e = ttk.Entry(top, textvariable=self.query, width=30)
        e.pack(side="left", padx=4)
        e.bind("<KeyRelease>", lambda ev: self.refresh())
        self.only_active = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="nur eingeschaltete", variable=self.only_active, command=self.refresh).pack(
            side="left", padx=8)
        self.hide_na = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="unpassende ausblenden", variable=self.hide_na, command=self.refresh).pack(side="left")
        self.summary = ttk.Label(top, text="", foreground="#555")
        self.summary.pack(side="right")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.tabs: dict[str, _Scrollable] = {}
        for key, (title, desc) in CATEGORIES.items():
            frame = _Scrollable(self.notebook)
            self.notebook.add(frame, text=self.SHORT.get(key, title))
            self.tabs[key] = frame
        ttk.Label(self, text="● an · ○ aus · ◐ teilweise · – passt nicht zu diesem Modell · Maus über den Namen = "
                             "Erklärung · Rückgängig im Hauptfenster (↶)", foreground="#666", padding=6).pack(anchor="w")
        self.refresh()

    def refresh(self) -> None:
        from ..techniques import CATEGORIES
        from .gui import HoverTip

        rows = self.app.project.techniques()
        q = self.query.get().strip().lower()
        n_on = sum(r.status == "an" for r in rows)
        self.summary.configure(text=f"{n_on} von {len(rows)} Techniken eingeschaltet")
        for cat, frame in self.tabs.items():
            for w in frame.inner.winfo_children():
                w.destroy()
            ttk.Label(frame.inner, text=CATEGORIES[cat][1], foreground="#555").grid(row=0, column=0, columnspan=4,
                                                                                   sticky="w", pady=(0, 6))
            r_i = 1
            for r in rows:
                t = r.technique
                if not t.in_category(cat):
                    continue
                if self.hide_na.get() and r.status == "nicht_anwendbar":
                    continue
                if self.only_active.get() and r.status not in ("an", "teilweise"):
                    continue
                blob = f"{t.key} {t.title} {t.tooltip.what} {t.tooltip.why} {t.used_by}".lower()
                if q and not all(w in blob for w in q.split()):
                    continue
                var = tk.BooleanVar(value=r.status == "an")
                cb = ttk.Checkbutton(frame.inner, variable=var, command=lambda k=t.key, v=var: self.toggle(k, v.get()))
                if r.status == "nicht_anwendbar":
                    cb.state(["disabled"])
                elif r.status == "teilweise":
                    cb.state(["alternate"])
                cb.grid(row=r_i, column=0, sticky="w")
                ttk.Label(frame.inner, text=self.MARK[r.status]).grid(row=r_i, column=1, padx=4)
                name = ttk.Label(frame.inner, text=t.title, cursor="question_arrow",
                                 foreground="#999" if r.status == "nicht_anwendbar" else "")
                name.grid(row=r_i, column=2, sticky="w")
                HoverTip(name, lambda t=t: f"{t.key}\n\n{t.tooltip.format(width=80)}"
                         + (f"\n\nGenutzt von: {t.used_by}" if t.used_by else "")
                         + (f"\nHinweis: {t.hint}" if t.hint else "")
                         + "\n\nÄndert: " + (", ".join(t.changes) if isinstance(t.changes, dict) else "berechnet"))
                ttk.Label(frame.inner, text=t.used_by + (f"  · {t.hint}" if t.hint else ""), foreground="#777").grid(
                    row=r_i, column=3, sticky="w", padx=(12, 0))
                r_i += 1
            if r_i == 1:
                ttk.Label(frame.inner, text="(keine passenden Einträge)", foreground="#999").grid(row=1, column=0)

    def toggle(self, key: str, on: bool) -> None:
        try:
            changed = self.app.project.technique(key, "an" if on else "aus")
            self.app.log_write(f"✓ Technik '{key}' {'an' if on else 'aus'}: {', '.join(changed) or 'keine Änderung'}\n")
        except ConfigError as exc:
            messagebox.showerror("Technik", str(exc), parent=self)
        self.app.refresh_tree()
        self.app.select_path(self.app.current_path if self.app._exists(self.app.current_path) else "")
        self.refresh()
