"""
ki_baukasten.analysis.architecture
==================================

**Komplette Auflistung und Darstellung** einer gebauten KI – zum Verstehen,
Integrieren und gezielten Anpassen.

Was erfasst wird
----------------
* **Jedes Modul** mit Name (= Pfad für Modul-/Tensor-Regeln), Typ, eigenen und
  enthaltenen Parametern, Trainierbarkeit, Datentyp, Gerät, Kenngrößen
  (z. B. ``in_features``, ``num_heads``, ``d_state``) und Markierungen
  (eingefroren, LoRA, quantisiert, Checkpointing, Streaming, Hooks).
* **Formen & Reihenfolge**: Ein Probe-Forward mit Beispiel-Eingabe zeichnet
  Ein-/Ausgabeformen und die tatsächliche Ausführungsreihenfolge auf.
* **Schichttabelle**: pro Block Mixer/FFN/Norm, Parameter, Ausgabeform und die
  Abweichungen seiner Konfiguration vom Basis-Block – samt der Pfade, über die
  man genau diese Schicht ändert (``model.layer_overrides.<i>…``, Modul-Regel).
* **Tensoren**: alle Parameter-Namen mit Form (Grundlage für Tensor-Regeln).

Ausgabeformate
--------------
``text`` (Konsole), ``markdown`` (Dokumentation, inkl. Mermaid-Diagramm),
``mermaid`` (nur Diagramm, z. B. für GitHub/Obsidian), ``json`` (maschinenlesbar,
zum Integrieren in eigene Werkzeuge).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from ..core import BaseConfig, format_number

_ATTRS = ("in_features", "out_features", "num_embeddings", "embedding_dim", "in_channels", "out_channels",
          "kernel_size", "stride", "padding", "dilation", "groups", "num_heads", "num_kv_heads", "head_dim",
          "hidden_size", "num_layers", "bidirectional", "p", "eps", "normalized_shape", "num_features", "e", "n",
          "fmt", "group_size", "k", "num_experts", "top_k", "drop_prob")


@dataclass
class ModuleInfo:
    name: str
    type: str
    depth: int
    params_own: int
    params_total: int
    trainable_total: int
    dtypes: list[str]
    device: str
    attrs: dict[str, Any] = field(default_factory=dict)
    markers: list[str] = field(default_factory=list)
    in_shapes: list[list[int]] = field(default_factory=list)
    out_shapes: list[list[int]] = field(default_factory=list)
    order: int | None = None
    children: list[str] = field(default_factory=list)


def _shapes(obj: Any) -> list[list[int]]:
    if isinstance(obj, torch.Tensor):
        return [list(obj.shape)]
    if isinstance(obj, (tuple, list)):
        return [s for o in obj for s in _shapes(o)][:4]
    if isinstance(obj, dict):
        return [s for o in obj.values() for s in _shapes(o)][:4]
    return []


def _plain(v: Any) -> Any:
    if isinstance(v, torch.Size):
        return list(v)
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if isinstance(v, (tuple, list)) and all(isinstance(x, (int, float, bool, str)) for x in v):
        return list(v)
    return str(v)


def _markers(module: nn.Module) -> list[str]:
    out = []
    params = list(module.parameters())
    if params:
        frozen = sum(not p.requires_grad for p in params)
        if frozen == len(params):
            out.append("eingefroren")
        elif frozen:
            out.append("teilweise eingefroren")
    if getattr(module, "_ki_lora", False):
        out.append(f"LoRA r={module.lora_A.shape[0]}" + (" (DoRA)" if module.lora_magnitude is not None else ""))
    if type(module).__name__ == "QuantLinear":
        out.append(f"quantisiert {module.fmt}")
    if getattr(module, "_ki_checkpointed", False):
        out.append("Checkpointing")
    if getattr(module, "_ki_streamed", False):
        out.append("RAM-Streaming")
    if module._forward_hooks:
        out.append(f"{len(module._forward_hooks)} Hook(s)")
    if getattr(module, "supports_cache", False):
        out.append("Cache")
    return out


@dataclass
class ArchitectureInfo:
    model_type: str
    total: int
    trainable: int
    weight_bytes: int
    modules: dict[str, ModuleInfo]
    blocks: list[str]
    layer_rows: list[dict[str, Any]]
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]

    # ------------------------------------------------------------ Hilfen
    def _visible(self, max_depth: int) -> list[ModuleInfo]:
        return [m for m in self.modules.values() if 0 < m.depth <= max_depth]

    def _execution(self, max_depth: int) -> list[ModuleInfo]:
        mods = self._visible(max_depth)
        ran = [m for m in mods if m.order is not None]
        return sorted(ran, key=lambda m: m.order) if ran else mods

    # ------------------------------------------------------------ Formate
    def to_dict(self) -> dict[str, Any]:
        def tree(name: str) -> dict[str, Any]:
            info = asdict(self.modules[name])
            info["children"] = [tree(c) for c in self.modules[name].children]
            return info
        return {"model_type": self.model_type, "parameters": self.total, "trainable": self.trainable,
                "weight_bytes": self.weight_bytes, "input_shapes": self.input_shapes,
                "output_shapes": self.output_shapes, "blocks": self.blocks, "layers": self.layer_rows,
                "tree": tree("")}

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_text(self, max_depth: int = 3) -> str:
        lines = [f"{self.model_type}: {format_number(self.total)} Parameter, {format_number(self.trainable)} "
                 f"trainierbar, Gewichte {format_number(self.weight_bytes / 2**20, 2)} MiB",
                 f"Eingabe {self.input_shapes or '?'} → Ausgabe {self.output_shapes or '?'}", ""]
        lines.append(f"{'Modul (Pfad für Regeln)':<46} {'Typ':<22} {'Param.':>10} {'Ausgabe':<18} Details")
        lines.append("─" * 120)
        for m in self._visible(max_depth):
            indent = "  " * (m.depth - 1)
            label = (indent + (m.name.split(".")[-1] if m.depth > 1 else m.name))[:45]
            out = "×".join(map(str, m.out_shapes[0])) if m.out_shapes else ""
            details = ", ".join(f"{k}={v}" for k, v in list(m.attrs.items())[:6] if not k.startswith("_"))
            if m.markers:
                details += ("  " if details else "") + "[" + ", ".join(m.markers) + "]"
            lines.append(f"{label:<46} {m.type[:21]:<22} {format_number(m.params_total):>10} {out[:17]:<18} {details}")
        if self.layer_rows:
            lines += ["", "Schichten (Blöcke) – Abweichungen von der Basis und Anpassungspfade:"]
            for row in self.layer_rows:
                diff = ", ".join(f"{k}={v}" for k, v in row["abweichungen"].items()) or "wie Basis"
                lines.append(f"  [{row['index']:>2}] {row['name']:<18} {row['typ']:<18} "
                             f"{format_number(row['parameter']):>10}  {row.get('aufbau', '')}  | {diff}")
            first = self.layer_rows[0]
            lines.append(f"  → einzelne Schicht ändern: {first['anpassen'].replace('.0.', '.<index>.')}  "
                         f"bzw. Modul-Regel {{\"target\": \"{first['name'].rsplit('.', 1)[0]}.<index>\", "
                         f"\"match\": \"exact\", …}}")
        return "\n".join(lines)

    def to_mermaid(self, max_depth: int = 2) -> str:
        """Datenfluss-Diagramm (Ausführungsreihenfolge, Unterblöcke als Subgraphen)."""
        def nid(name: str) -> str:
            return "m_" + name.replace(".", "_").replace("-", "_") if name else "eingabe"

        def label(m: ModuleInfo) -> str:
            shape = ("<br/>" + "×".join(map(str, m.out_shapes[0]))) if m.out_shapes else ""
            return f"{m.name.split('.')[-1]}<br/><i>{m.type}</i><br/>{format_number(m.params_total)} P.{shape}"

        lines = ["flowchart TD", f'  eingabe(["Eingabe {self.input_shapes[0] if self.input_shapes else ""}"])']
        order = self._execution(max_depth)
        leaves = [m for m in order if not any(c in self.modules and self.modules[c].depth <= max_depth
                                              for c in m.children)]
        top = [m for m in self._visible(1)]
        for t in top:
            inner = [m for m in leaves if m.name == t.name or m.name.startswith(t.name + ".")]
            if len(inner) > 1 or (inner and inner[0].name != t.name):
                lines.append(f'  subgraph {nid(t.name)}_g["{t.name} · {t.type}"]')
                lines += [f'    {nid(m.name)}["{label(m)}"]' for m in inner]
                lines.append("  end")
            elif inner:
                lines.append(f'  {nid(t.name)}["{label(t)}"]')
        prev = "eingabe"
        for m in leaves:
            lines.append(f"  {prev} --> {nid(m.name)}")
            prev = nid(m.name)
        lines.append(f'  ausgabe(["Ausgabe {self.output_shapes[0] if self.output_shapes else ""}"])')
        lines.append(f"  {prev} --> ausgabe")
        return "\n".join(lines)

    def to_markdown(self, max_depth: int = 3) -> str:
        md = [f"# Aufbau: {self.model_type}", "",
              f"* Parameter: **{format_number(self.total)}** ({format_number(self.trainable)} trainierbar)",
              f"* Gewichtsspeicher: {format_number(self.weight_bytes / 2**20, 2)} MiB",
              f"* Eingabe: `{self.input_shapes}` → Ausgabe: `{self.output_shapes}`", "",
              "## Datenfluss", "", "```mermaid", self.to_mermaid(min(max_depth, 2)), "```", "",
              "## Module", "", "| Pfad | Typ | Parameter | Ausgabe | Details |", "|---|---|---:|---|---|"]
        for m in self._visible(max_depth):
            out = "×".join(map(str, m.out_shapes[0])) if m.out_shapes else ""
            details = ", ".join(f"{k}={v}" for k, v in m.attrs.items() if not k.startswith("_"))
            if m.markers:
                details += " **[" + ", ".join(m.markers) + "]**"
            md.append(f"| `{m.name}` | {m.type} | {format_number(m.params_total)} | {out} | {details} |")
        if self.layer_rows:
            md += ["", "## Schichten", "", "| # | Modul | Typ | Aufbau | Parameter | Abweichungen von der Basis | "
                   "Anpassen über |", "|---:|---|---|---|---:|---|---|"]
            for r in self.layer_rows:
                diff = ", ".join(f"`{k}={v}`" for k, v in r["abweichungen"].items()) or "–"
                md.append(f"| {r['index']} | `{r['name']}` | {r['typ']} | {r.get('aufbau', '')} | "
                          f"{format_number(r['parameter'])} | {diff} | `{r['anpassen']}` |")
        md += ["", "## Tensoren", "", "| Name | Form | Anzahl | trainierbar |", "|---|---|---:|:---:|"]
        for m in self.modules.values():
            for pname, shape, n, tr in m.attrs.get("_params", []):
                md.append(f"| `{pname}` | {shape} | {format_number(n)} | {'✓' if tr else '–'} |")
        md += ["", "## Anpassen", "",
               "* **Ganze Schichtart / alle Schichten**: Basis-Block `model.block.…` ändern.",
               "* **Bestimmte Schichten**: `model.layer_overrides` mit Selektor (`\"0\"`, `\"-1\"`, `\"::2\"`) – "
               "Übernahme von anderen Schichten mit `@von`, `@nur`, `@ausser`.",
               "* **Gebaute Module jeder Art**: `module_rules` mit `target`/`module_type`, `layers` und `scope` "
               "(alle / nur_auswahl / nur_einer / alle_ausser).",
               "* **Einzelne Tensoren oder Ausschnitte**: `tensor_rules` (Namen siehe Tabelle oben)."]
        return "\n".join(md)


def _layer_rows(model: nn.Module, blocks: list[str], modules: dict[str, ModuleInfo]) -> list[dict[str, Any]]:
    rows = []
    base_cfg = getattr(getattr(model, "config", None), "block", None)
    has_base = isinstance(base_cfg, BaseConfig)
    base_full = _flat(base_cfg.to_dict()) if has_base else {}
    base_changed = set(_flat(base_cfg.to_dict(only_changed=True))) if has_base else set()
    for i, name in enumerate(blocks):
        block = model.get_submodule(name)
        info = modules[name]
        row: dict[str, Any] = {"index": i, "name": name, "typ": type(block).__name__, "parameter": info.params_total,
                               "ausgabe": info.out_shapes[0] if info.out_shapes else None, "abweichungen": {}}
        parts = []
        for part in ("mixer", "ffn", "norm1", "cross", "attn"):
            sub = getattr(block, part, None)
            if isinstance(sub, nn.Module):
                kind = type(sub).__name__
                if part == "mixer" and hasattr(sub, "attn"):
                    a = sub.attn
                    kind += f"(H={a.num_heads}, KV={a.num_kv_heads}, hd={a.head_dim})"
                parts.append(f"{part}={kind}")
        row["aufbau"] = ", ".join(parts)
        cfg = getattr(block, "cfg", None)
        if isinstance(cfg, BaseConfig) and has_base and type(cfg) is type(base_cfg):
            # Nur Einstellungen, die bei dieser Schicht ODER der Basis vom Standard abweichen – und verschieden sind.
            mine_full = _flat(cfg.to_dict())
            candidates = list(_flat(cfg.to_dict(only_changed=True))) + sorted(base_changed)
            row["abweichungen"] = {k: mine_full.get(k) for k in dict.fromkeys(candidates)
                                   if mine_full.get(k) != base_full.get(k) and not k.endswith("drop_path")}
        is_encoder = name.startswith("encoder_blocks")
        idx = int(name.split(".")[-1]) if name.split(".")[-1].isdigit() else i
        row["anpassen"] = f"model.{'encoder_' if is_encoder else ''}layer_overrides.{idx}.<einstellung>"
        rows.append(row)
    return rows


def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        if "name" in d and set(d) <= {"name", "args"}:  # Komponente: Name + Argumente
            out[prefix.rstrip(".")] = d["name"]
            out.update(_flat(d.get("args", {}), prefix))
            return out
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    out[prefix.rstrip(".")] = d if not isinstance(d, list) else tuple(d)
    return out


def inspect_model(model: nn.Module, example: Any = None, run: bool = True) -> ArchitectureInfo:
    """Erfasst den kompletten Aufbau (optional mit Probe-Forward für Formen/Reihenfolge)."""
    modules: dict[str, ModuleInfo] = {}
    for name, module in model.named_modules():
        own = list(module.parameters(recurse=False))
        allp = list(module.parameters())
        attrs = {}
        for a in _ATTRS:
            v = getattr(module, a, None)
            if v is not None and not isinstance(v, (nn.Module, torch.Tensor)) and not callable(v):
                attrs[a] = _plain(v)
        if own:
            attrs["_params"] = [(f"{name}.{n}" if name else n, tuple(p.shape), p.numel(), p.requires_grad)
                                for n, p in module.named_parameters(recurse=False)]
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in allp}
                        | {str(b.dtype).replace("torch.", "") for b in module.buffers() if b.dtype in (torch.int8, torch.uint8)})
        dev = str(allp[0].device) if allp else "-"
        modules[name] = ModuleInfo(name, type(module).__name__, name.count(".") + 1 if name else 0,
                                   sum(p.numel() for p in own), sum(p.numel() for p in allp),
                                   sum(p.numel() for p in allp if p.requires_grad), dtypes, dev, attrs,
                                   _markers(module) if name else [],
                                   children=[f"{name}.{c}" if name else c for c, _ in module.named_children()])
    in_shapes: list[list[int]] = []
    out_shapes: list[list[int]] = []
    if run:
        if example is None and hasattr(model, "example_input"):
            try:
                example = model.example_input(1)
            except Exception:  # noqa: BLE001 - Darstellung soll nie am Beispiel scheitern
                example = None
        if example is not None:
            counter = [0]
            handles = []
            names = {id(m): n for n, m in model.named_modules()}

            def hook(module, args, output):
                info = modules[names[id(module)]]
                if info.order is None:
                    info.order = counter[0]
                    counter[0] += 1
                    info.in_shapes, info.out_shapes = _shapes(args), _shapes(output)

            for m in model.modules():
                if m is not model:
                    handles.append(m.register_forward_hook(hook))
            was = model.training
            model.eval()
            try:
                from ..training.paradigms import call_model
                dev = next(model.parameters()).device
                from ..training.data_utils import to_device
                with torch.no_grad():
                    out = call_model(model, to_device(example, dev))
                in_shapes, out_shapes = _shapes(example), _shapes(out)
            except Exception:  # noqa: BLE001
                pass
            finally:
                for h in handles:
                    h.remove()
                model.train(was)
    blocks = []
    if hasattr(model, "checkpointable_blocks"):
        by_id = {id(m): n for n, m in model.named_modules()}
        blocks = [by_id[id(b)] for b in model.checkpointable_blocks() if id(b) in by_id]
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers() if b.dtype in (torch.int8, torch.uint8))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ArchitectureInfo(type(model).__name__, total, trainable, weight_bytes, modules, blocks,
                            _layer_rows(model, blocks, modules), in_shapes, out_shapes)


def architecture_report(model: nn.Module, fmt: str = "text", max_depth: int = 3, example: Any = None) -> str:
    """Aufbau-Darstellung im Format ``text`` | ``markdown`` | ``mermaid`` | ``json``."""
    info = inspect_model(model, example)
    if fmt == "text":
        return info.to_text(max_depth)
    if fmt in ("markdown", "md"):
        return info.to_markdown(max_depth)
    if fmt == "mermaid":
        return info.to_mermaid(min(max_depth, 3))
    if fmt == "json":
        return info.to_json()
    raise ValueError(f"Unbekanntes Format '{fmt}' (text, markdown, mermaid, json).")
