"""
ki_baukasten.ui.docs
====================

Erzeugt die **vollständige Parameter-Referenz** als Markdown – direkt aus den
Tooltips im Code. Dadurch ist die Dokumentation immer aktuell: wer eine
Einstellung hinzufügt, dokumentiert sie automatisch mit.

Aufruf: ``python -m ki_baukasten doku -o docs/PARAMETER_REFERENZ.md``
"""

from __future__ import annotations

from pathlib import Path

from ..core import BaseConfig, Level, all_registries


def _esc(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _tooltip_cell(tip) -> str:
    parts = [f"**Was:** {tip.what}"]
    if tip.why:
        parts.append(f"**Nutzen:** {tip.why}")
    if tip.use_for:
        parts.append(f"**Wofür:** {tip.use_for}")
    if tip.tip:
        parts.append(f"**Tipp:** {tip.tip}")
    return _esc("<br>".join(parts))


def config_table(cfg: BaseConfig, prefix: str = "") -> list[str]:
    """Tabelle aller Einstellungen eines Konfigurationsobjekts (rekursiv)."""
    lines = ["| Pfad | Typ | Standard | Regeln | Stufe | Beschreibung |", "|---|---|---|---|---|---|"]
    for ref in cfg.walk(prefix=prefix):
        p = ref.param
        kind = getattr(p, "kind", "value")
        if kind in ("nested",):
            default = f"⟶ {type(ref.value).__name__}" if ref.value is not None else "None"
        elif kind == "component":
            default = f"`{ref.value.name}`" if ref.value is not None else "None"
        elif kind in ("component_list", "config_list"):
            default = f"{len(ref.value)} Elemente"
        else:
            default = f"`{_esc(repr(ref.value))[:60]}`"
        lines.append(f"| `{ref.path}` | {_esc(p.type_label())} | {default} | {_esc(p.constraint_label())[:160]} | "
                     f"{ref.level.label} | {_tooltip_cell(p.tooltip)} |")
    return lines


def export_markdown(path: str | Path = "docs/PARAMETER_REFERENZ.md") -> Path:
    from ..project import ProjectConfig

    out: list[str] = [
        "# KI-Baukasten – Parameter-Referenz",
        "",
        "Automatisch erzeugt aus den Tooltips im Quellcode (`python -m ki_baukasten doku`).",
        "Stufen: **Basis** → **Fortgeschritten** → **Experte** → **Experimentell** (nur Orientierung – alles ist änderbar).",
        "",
        "## Inhalt",
        "",
        "1. [Projekt-Konfiguration](#projekt-konfiguration)",
        "2. [Registries und Komponenten](#registries-und-komponenten)",
        "",
        "## Projekt-Konfiguration",
        "",
        "Standardwerte eines leeren Projekts (`ProjectConfig()`). Modell- und Komponenten-Argumente",
        "hängen von der gewählten Komponente ab – siehe Abschnitt *Registries und Komponenten*.",
        "",
    ]
    out += config_table(ProjectConfig())
    out += ["", "## Registries und Komponenten", ""]
    regs = all_registries()
    for kind, reg in regs.items():
        out.append(f"- [{reg.title} (`{kind}`)](#registry-{kind.replace('_', '-')})")
    for kind, reg in regs.items():
        out += ["", f"### Registry `{kind}`", "", f"<a id=\"registry-{kind.replace('_', '-')}\"></a>", "",
                f"**{reg.title}** – {reg.tooltip.what} {reg.tooltip.why}", ""]
        if reg.context_doc:
            out += [f"Fabrik-Signatur: `{reg.context_doc}`", ""]
        for comp in reg.components(Level.EXPERIMENTAL):
            alias = f" (Aliase: {', '.join(f'`{x}`' for x in comp.aliases)})" if comp.aliases else ""
            out += [f"#### `{kind}:{comp.name}`{alias} — Stufe {comp.level.label}", "",
                    _esc(f"**Was:** {comp.tooltip.what}"), ""]
            if comp.tooltip.why:
                out += [_esc(f"**Nutzen:** {comp.tooltip.why}"), ""]
            if comp.tooltip.use_for:
                out += [_esc(f"**Wofür:** {comp.tooltip.use_for}"), ""]
            if comp.tooltip.tip:
                out += [_esc(f"**Tipp:** {comp.tooltip.tip}"), ""]
            try:
                args = comp.config_cls()
            except Exception:
                args = None
            if args is not None and type(args)._params:
                out += config_table(args)
                out.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
