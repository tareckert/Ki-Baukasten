"""
ki_baukasten.tensor.modules
===========================

**Modul-Chirurgie** – Eingriffe in ganze Schichten/Module *jeder Art* nach dem
Bauen des Modells (Ergänzung zur Tensor-Chirurgie, die einzelne Parameter trifft).

Eine Regel (``ModuleRuleConfig``)
---------------------------------
* **Auswahl**: Modulname (``target`` + ``match``), Modultyp (``module_type``,
  z. B. ``"Linear"``, ``"Dropout"``, ``"LayerNorm,RMSNorm"``), Blöcke (``layers``)
* **Geltungsbereich**: ``scope`` = alle / nur_auswahl / nur_einer / alle_ausser
  mit ``pick`` (Positionen in der Trefferliste) – siehe ``tensor.selection``
* **Operation** (Registry ``module_op``)

Beispiele::

    "module_rules": [
      {"module_type": "Dropout", "op": {"name": "set_attr", "attribute": "p", "value": 0.0}},
      {"target": "ffn", "layers": "-1", "match": "full", "op": {"name": "lora", "rank": 16}},
      {"module_type": "GELU", "scope": "nur_einer", "pick": "0", "op": {"name": "replace_activation",
                                                                        "activation": "silu"}},
      {"target": "blocks.3", "match": "exact", "op": {"name": "copy_from", "source": "blocks.0"}},
      {"module_type": "RMSNorm", "op": {"name": "dtype", "dtype": "float32"}},
      {"target": "mixer", "layers": "::2", "op": {"name": "scale_output", "factor": 0.5}}
    ]

Operationen
-----------
=====================  ===========================================================
Struktur               replace_activation, replace_norm, identity, quantize, lora,
                       spectral_norm, weight_norm, checkpoint, dtype
Werte                  init, reset, copy_from (Übernahme von einem anderen Modul),
                       set_attr (beliebiges Attribut, z. B. Dropout.p, eps)
Training               freeze, unfreeze
Datenfluss (Hooks)     scale_output, output_noise, clamp_output, stop_gradient, record
=====================  ===========================================================

Die Vorschau (``ModuleSurgeon.preview``) zeigt vorab, welche Module jede Regel trifft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from ..components.activations import build_activation
from ..components.initializers import apply_initializer
from ..components.normalization import build_norm
from ..core import BaseConfig, ComponentParam, ConfigError, Level, Param, T, create_registry
from .selection import apply_scope, layer_filter, name_matcher, scope_params

MODULE_OPS = create_registry(
    "module_op",
    "Modul-Operationen",
    T(
        "Eingriffe in ganze Schichten/Module (ersetzen, umbauen, einfrieren, Datenfluss verändern …).",
        "Jede Schicht jeder Art einzeln anpassen – auch nach dem Bau, ohne Code.",
        "Ablationen, Feintuning (LoRA einzelner Schichten), Präzision einzelner Teile, Analyse.",
    ),
    context_doc="factory(args) -> callable(model, name, module, ctx)",
)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16, "float64": torch.float64}


# ===========================================================================
# Hilfen
# ===========================================================================

def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, attr = name.rpartition(".")
    owner = root.get_submodule(parent) if parent else root
    if isinstance(owner, (nn.Sequential, nn.ModuleList)) and attr.isdigit():
        owner[int(attr)] = new
    else:
        setattr(owner, attr, new)


def _feature_dim(module: nn.Module) -> int:
    """Merkmalsbreite eines (Norm-)Moduls für den Ersatz."""
    for attr in ("normalized_shape", "num_features", "dim", "num_channels", "d_model"):
        v = getattr(module, attr, None)
        if isinstance(v, (tuple, list, torch.Size)) and len(v):
            return int(v[-1])
        if isinstance(v, int):
            return v
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor) and w.dim() == 1:
        return w.shape[0]
    raise ConfigError(f"Merkmalsbreite von {type(module).__name__} nicht erkennbar – 'num_features' angeben.")


class PassThrough(nn.Module):
    """Identität, die beliebige Zusatzargumente akzeptiert (Ersatz für entfernte Blöcke/Mixer)."""

    def forward(self, x, *args, **kwargs):
        return x


def _type_names(module: nn.Module) -> set[str]:
    return {c.__name__ for c in type(module).__mro__}


def _linears(module: nn.Module, name: str) -> list[str]:
    """Namen aller nn.Linear im Modul (inklusive des Moduls selbst)."""
    return [f"{name}.{n}" if n else name for n, m in module.named_modules() if type(m) is nn.Linear]


# ===========================================================================
# Operationen
# ===========================================================================

class ReplaceActivationConfig(BaseConfig):
    activation = ComponentParam("activation", "gelu", tooltip=T(
        "Neue Aktivierung.", "Austausch der Nichtlinearität einzelner Schichten.", "silu, gelu, relu …"))


class ReplaceNormConfig(BaseConfig):
    norm = ComponentParam("normalization", "rmsnorm", tooltip=T(
        "Neue Normalisierung.", "Austausch einzelner Normschichten.", "rmsnorm, layernorm …"))
    num_features = Param(None, type=int, min=1, nullable=True, tooltip=T(
        "Merkmalsbreite (None = vom alten Modul übernehmen).", "Nur nötig, wenn nicht erkennbar.", "None."),
        level=Level.EXPERT)


class SetAttrConfig(BaseConfig):
    attribute = Param("p", tooltip=T(
        "Attributname (auch Punkt-Pfad, z. B. 'attn.scale').",
        "Verändert Eigenschaften gebauter Module, die im Forward gelesen werden.",
        "Dropout 'p', Norm 'eps', Attention 'scale', LeakyReLU 'negative_slope'."))
    value = Param(0.0, type=(int, float, bool, str), nullable=True, tooltip=T(
        "Neuer Wert.", "Wird unverändert gesetzt.", "Je nach Attribut."))
    must_exist = Param(True, tooltip=T(
        "Fehler, wenn das Attribut nicht existiert.", "Schützt vor Tippfehlern.", "an."), level=Level.EXPERT)


class FactorConfig(BaseConfig):
    factor = Param(1.0, tooltip=T("Faktor auf die Ausgabe.", "Dämpft/verstärkt den Beitrag des Moduls.", "0.5."))


class NoiseConfig(BaseConfig):
    std = Param(0.01, min=0.0, tooltip=T("Standardabweichung.", "Stärke des Rauschens.", "0.01–0.1."))
    only_training = Param(True, tooltip=T("Nur im Training.", "Inferenz bleibt deterministisch.", "an."))


class ClampConfig(BaseConfig):
    min = Param(None, type=float, nullable=True, tooltip=T("Untergrenze.", "Begrenzt Aktivierungen.", "-10."))
    max = Param(None, type=float, nullable=True, tooltip=T("Obergrenze.", "Begrenzt Ausreißer.", "10."))


class DtypeConfig(BaseConfig):
    dtype = Param("float32", choices=list(_DTYPES), tooltip=T(
        "Datentyp der Parameter dieses Moduls.",
        "Empfindliche Teile (Normen, Router) in fp32 halten, Rest in bf16.", "float32 für Normen."))


class QuantizeConfig(BaseConfig):
    format = Param("int8", choices=["int8", "int4", "nf4", "pal4"], tooltip=T(
        "Quantisierungsformat (eigene Implementierung).", "Speicher dieses Moduls verkleinern.", "int8."))
    group_size = Param(128, type=int, min=8, nullable=True, tooltip=T(
        "Gruppengröße (None = pro Kanal).", "Genauigkeit vs. Skalen-Speicher.", "128."))


class LoraOpConfig(BaseConfig):
    rank = Param(8, min=1, tooltip=T("Rang.", "Adapter-Kapazität.", "8–32."))
    alpha = Param(16.0, min=0.0, tooltip=T("Skalierung α.", "Faktor α/r.", "2·rank."))
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T("Dropout.", "Regularisierung.", "0–0.1."))
    use_dora = Param(False, tooltip=T("DoRA.", "Betrag + Richtung.", "Experimente."), level=Level.EXPERIMENTAL)
    freeze_module = Param(True, tooltip=T(
        "Basisgewichte dieses Moduls einfrieren.", "Nur die Adapter lernen.", "an."))


class InitOpConfig(BaseConfig):
    weight = ComponentParam("initializer", "xavier_uniform", tooltip=T(
        "Initialisierer für Gewichte (≥ 2 Dimensionen).", "Neue Startwerte.", "xavier_uniform, orthogonal …"))
    bias = ComponentParam("initializer", "zeros", tooltip=T(
        "Initialisierer für Biases/1D-Parameter (None = unverändert).", "Startwerte.", "zeros."), nullable=True)


class CopyFromConfig(BaseConfig):
    source = Param("", tooltip=T(
        "Name des Quellmoduls mit gleicher Struktur (z. B. 'blocks.0').",
        "Übernimmt alle Gewichte und Puffer eines anderen Moduls.", "Schichten klonen, Warm-Start."))
    mode = Param("replace", choices={"replace": "kopieren", "interpolate": "(1-α)·eigen + α·Quelle"}, tooltip=T(
        "Art der Übernahme.", "Vollständig oder anteilig.", "replace."))
    alpha = Param(0.5, min=0.0, max=1.0, tooltip=T("Anteil der Quelle (interpolate).", "Mischung.", "0.5."))


@dataclass
class _Hook:
    fn: Callable


def _output_hook(transform: Callable[[torch.Tensor, nn.Module], torch.Tensor]):
    """Forward-Hook, der jede Tensor-Ausgabe (auch in Tupeln) transformiert."""
    def hook(module, args, output):
        if isinstance(output, torch.Tensor):
            return transform(output, module)
        if isinstance(output, tuple):
            return tuple(transform(o, module) if isinstance(o, torch.Tensor) else o for o in output)
        return output
    return _Hook(hook)


def _record(out: torch.Tensor, module: nn.Module) -> torch.Tensor:
    x = out.detach().float()
    module._ki_stats = {"mean": float(x.mean()), "std": float(x.std()) if x.numel() > 1 else 0.0,
                        "absmax": float(x.abs().max()), "shape": tuple(out.shape)}
    return out


def _freeze(flag: bool):
    def fn(model, name, module, ctx):
        for p in module.parameters():
            p.requires_grad_(flag)
    return fn


def _replace_activation(a: ReplaceActivationConfig):
    def fn(model, name, module, ctx):
        _set_module(model, name, build_activation(a.activation))
    return fn


def _replace_norm(a: ReplaceNormConfig):
    def fn(model, name, module, ctx):
        dim = a.num_features or _feature_dim(module)
        new = build_norm(a.norm, dim)
        ref = next(module.parameters(), None)
        _set_module(model, name, new.to(ref.device) if ref is not None else new)
    return fn


def _set_attr(a: SetAttrConfig):
    def fn(model, name, module, ctx):
        *path, attr = a.attribute.split(".")
        obj = module
        for seg in path:
            obj = getattr(obj, seg)
        if a.must_exist and not hasattr(obj, attr):
            raise ConfigError(f"set_attr: {name or type(module).__name__} hat kein Attribut '{a.attribute}'.")
        setattr(obj, attr, a.value)
    return fn


def _dtype(a: DtypeConfig):
    def fn(model, name, module, ctx):
        module.to(_DTYPES[a.dtype])
    return fn


def _quantize(a: QuantizeConfig):
    def fn(model, name, module, ctx):
        from ..memory.quant import QuantLinear
        targets = _linears(module, name)
        if not targets:
            raise ConfigError(f"quantize: '{name}' enthält keine lineare Schicht.")
        for n in targets:
            _set_module(model, n, QuantLinear.from_linear(model.get_submodule(n), a.format, a.group_size))
    return fn


def _lora(a: LoraOpConfig):
    def fn(model, name, module, ctx):
        from ..memory.lora import LoRAConfig, _attach
        from ..memory.quant import QuantLinear
        cfg = LoRAConfig(enabled=True, rank=a.rank, alpha=a.alpha, dropout=a.dropout, use_dora=a.use_dora)
        targets = [n for n, m in module.named_modules() if isinstance(m, (nn.Linear, QuantLinear))
                   and not getattr(m, "_ki_lora", False)]
        if not targets:
            raise ConfigError(f"lora: '{name}' enthält keine lineare Schicht.")
        for n in targets:
            sub = module.get_submodule(n) if n else module
            if a.freeze_module:
                for p in sub.parameters():
                    p.requires_grad_(False)
            _attach(sub, cfg)
    return fn


def _parametrize(kind: str):
    def fn(model, name, module, ctx):
        from torch.nn.utils import parametrizations
        if not isinstance(getattr(module, "weight", None), torch.Tensor):
            raise ConfigError(f"{kind}: '{name}' hat kein Gewicht 'weight'.")
        getattr(parametrizations, kind)(module)
    return fn


def _checkpoint(model, name, module, ctx):
    from ..memory.manager import _checkpointed_class
    if not getattr(module, "_ki_checkpointed", False):
        module.__class__ = _checkpointed_class(type(module))


def _init(a: InitOpConfig):
    def fn(model, name, module, ctx):
        with torch.no_grad():
            for p in module.parameters():
                if p.dim() >= 2:
                    apply_initializer(a.weight, p)
                elif a.bias is not None:
                    apply_initializer(a.bias, p)
    return fn


def _reset(model, name, module, ctx):
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()


def _copy_from(a: CopyFromConfig):
    def fn(model, name, module, ctx):
        try:
            src = model.get_submodule(a.source)
        except AttributeError:
            raise ConfigError(f"copy_from: Quellmodul '{a.source}' existiert nicht.") from None
        if src is module:
            return
        mine, theirs = module.state_dict(), src.state_dict()
        diff = sorted(set(mine) ^ set(theirs)) + [k for k in mine if k in theirs and mine[k].shape != theirs[k].shape]
        if diff:
            raise ConfigError(f"copy_from: '{name}' und '{a.source}' haben unterschiedliche Struktur "
                              f"(z. B. {', '.join(diff[:4])}). Regeln, die die Struktur ändern, ggf. danach anwenden.")
        with torch.no_grad():
            for k, v in mine.items():
                s = theirs[k].to(v.device, v.dtype)
                if a.mode == "replace" or not v.is_floating_point():
                    v.copy_(s)
                else:
                    v.mul_(1 - a.alpha).add_(s, alpha=a.alpha)
    return fn


R = MODULE_OPS.register
R("freeze", lambda a: _freeze(False), tooltip=T(
    "Alle Parameter des Moduls einfrieren.", "Schützt Wissen, spart Gradienten.", "Transfer-Learning."))
R("unfreeze", lambda a: _freeze(True), tooltip=T(
    "Alle Parameter des Moduls auftauen.", "Hebt freeze auf.", "'Alles einfrieren außer …'."))
R("replace_activation", lambda a: _replace_activation(a), config=ReplaceActivationConfig, tooltip=T(
    "Aktivierungsmodul ersetzen.", "Andere Nichtlinearität in einzelnen Schichten.", "Ablationen, Experimente."))
R("replace_norm", lambda a: _replace_norm(a), config=ReplaceNormConfig, tooltip=T(
    "Normalisierungsmodul ersetzen.", "Andere Norm in einzelnen Schichten.", "LayerNorm → RMSNorm."))
R("identity", lambda a: (lambda model, name, module, ctx: _set_module(model, name, PassThrough())),
  aliases=("entfernen",), tooltip=T(
      "Modul durch Identität ersetzen (entfernen).", "Ablationen: Was passiert ohne diese Schicht?",
      "Form muss erhalten bleiben (z. B. Dropout, Norm, Aktivierung, ganzer Residual-Block)."))
R("set_attr", lambda a: _set_attr(a), config=SetAttrConfig, tooltip=T(
    "Beliebiges Attribut setzen.", "Feinste Eingriffe in gebaute Module.", "Dropout.p, eps, scale …"))
R("dtype", lambda a: _dtype(a), config=DtypeConfig, level=Level.ADVANCED, tooltip=T(
    "Datentyp des Moduls ändern.", "Gemischte Präzision pro Schicht.", "Normen/Router in fp32."))
R("quantize", lambda a: _quantize(a), config=QuantizeConfig, level=Level.ADVANCED, tooltip=T(
    "Lineare Schichten in diesem Modul quantisieren.", "Gezielte Kompression einzelner Teile.", "Große FFNs."))
R("lora", lambda a: _lora(a), config=LoraOpConfig, level=Level.ADVANCED, tooltip=T(
    "LoRA-Adapter für die linearen Schichten dieses Moduls.", "Adapter nur dort, wo gewünscht.",
    "Feintuning ausgewählter Schichten."))
R("spectral_norm", lambda a: _parametrize("spectral_norm"), level=Level.ADVANCED, tooltip=T(
    "Spektralnormierung des Gewichts.", "Begrenzt die Lipschitz-Konstante → stabile GANs/Diskriminatoren.",
    "GAN-Diskriminatoren."))
R("weight_norm", lambda a: _parametrize("weight_norm"), level=Level.ADVANCED, tooltip=T(
    "Gewichtsnormierung (Richtung + Betrag).", "Schnellere, stabilere Optimierung.", "Audio-/RL-Netze."))
R("checkpoint", lambda a: _checkpoint, level=Level.ADVANCED, tooltip=T(
    "Gradient-Checkpointing für dieses Modul.", "Spart Aktivierungsspeicher genau hier.", "Speicherintensive Teile."))
R("init", lambda a: _init(a), config=InitOpConfig, tooltip=T(
    "Alle Parameter des Moduls neu initialisieren.", "Gezielte Startwerte.", "Neuer Kopf, Re-Init."))
R("reset", lambda a: _reset, tooltip=T(
    "PyTorch-Standard-Initialisierung (reset_parameters).", "Zurück auf Anfang.", "Re-Init einzelner Schichten."))
R("copy_from", lambda a: _copy_from(a), config=CopyFromConfig, aliases=("uebernehmen",), tooltip=T(
    "Gewichte eines anderen Moduls gleicher Struktur übernehmen.",
    "Übernahme ganzer Schichten (alle Tensoren auf einmal).", "Schichten klonen, Block-Wiederholung."))
R("scale_output", lambda a: _output_hook(lambda o, m: o * a.factor), config=FactorConfig, tooltip=T(
    "Ausgabe mit Faktor multiplizieren (Hook).", "Beitrag einzelner Module dämpfen/verstärken.", "Ablationen."))
R("output_noise", lambda a: _output_hook(
    lambda o, m: o + torch.randn_like(o) * a.std if (m.training or not a.only_training) else o),
  config=NoiseConfig, level=Level.ADVANCED, tooltip=T(
      "Rauschen auf die Ausgabe (Hook).", "Robustheit, Regularisierung.", "Experimente."))
R("clamp_output", lambda a: _output_hook(lambda o, m: o.clamp(a.min, a.max)), config=ClampConfig,
  level=Level.ADVANCED, tooltip=T("Ausgabe begrenzen (Hook).", "Gegen Aktivierungsausreißer.", "Stabilisierung."))
R("stop_gradient", lambda a: _output_hook(lambda o, m: o.detach()), level=Level.ADVANCED, tooltip=T(
    "Gradient an der Ausgabe abschneiden (Hook).", "Alles davor lernt nicht über diesen Pfad.", "Experimente."))
R("record", lambda a: _output_hook(_record), tooltip=T(
    "Statistik der Ausgabe aufzeichnen (Mittelwert, Std, Max) → ModuleSurgeon.activation_report().",
    "Analyse: Wo explodieren oder verschwinden Aktivierungen?", "Debugging, Interpretierbarkeit."))


# ===========================================================================
# Regeln & Chirurg
# ===========================================================================

class ModuleRuleConfig(BaseConfig):
    __title__ = "Modul-Regel"
    __tooltip__ = T(
        "Eine Regel der Modul-Chirurgie: Auswahl von Schichten + Geltungsbereich + Operation.",
        "Jede Schicht jeder Art lässt sich nach dem Bau ersetzen, umbauen, einfrieren oder beobachten.",
        "Ablationen, gezieltes Feintuning, Präzision pro Schicht, Analyse.",
    )
    label = Param("", tooltip=T("Beschreibung (frei).", "Übersicht.", "z. B. 'Dropout aus'."))
    target = Param(".*", tooltip=T(
        "Regex (oder exakter Name, siehe match) auf den Modulnamen.",
        "Wählt Module, z. B. 'ffn', 'mixer\\\\.attn', 'blocks\\\\.3$'.",
        "Namen zeigt 'python -m ki_baukasten aufbau' bzw. Project.architecture().",
    ))
    match, layers, scope, pick = scope_params()
    module_type = Param(None, type=str, nullable=True, tooltip=T(
        "Nur Module dieser Klasse(n), kommagetrennt (auch Basisklassen), z. B. 'Linear' oder 'LayerNorm,RMSNorm'.",
        "Auswahl nach Schichtart.", "Alle Dropouts: 'Dropout'."))
    exclude = Param(None, type=str, nullable=True, tooltip=T(
        "Regex für Ausnahmen.", "Schließt Module wieder aus.", "'lm_head'."), level=Level.ADVANCED)
    op = ComponentParam("module_op", "freeze", tooltip=T(
        "Operation (Registry 'module_op').", "Was mit den Modulen passiert.", "Siehe Optionen."))
    enabled = Param(True, tooltip=T("Regel aktiv.", "Schnell an/aus.", "Experimente."))


class ModuleSurgeon:
    """Wendet eine Liste von ``ModuleRuleConfig`` auf ein Modell an."""

    def __init__(self, model: nn.Module, rules: list[ModuleRuleConfig]):
        self.model = model
        self.rules = [r for r in rules if r.enabled]
        self.applied: list[tuple[str, str]] = []
        self.handles: list = []

    def matches(self, rule: ModuleRuleConfig) -> list[tuple[str, nn.Module]]:
        hit = name_matcher(rule.target, rule.match)
        in_layers = layer_filter(self.model, rule.layers)
        ex = re.compile(rule.exclude) if rule.exclude else None
        types = {t.strip() for t in rule.module_type.split(",")} if rule.module_type else None
        found = []
        for name, module in self.model.named_modules():
            if not name or not hit(name) or not in_layers(name) or (ex and ex.search(name)):
                continue
            if types and not (types & _type_names(module)):
                continue
            found.append((name, module))
        if rule.match != "exact" and not types:
            # Ohne Typfilter: nur die äußersten Treffer (sonst würde 'ffn' auch 'ffn.w1' treffen).
            outer = []
            for name, module in found:
                if not any(name.startswith(o + ".") for o, _ in outer):
                    outer.append((name, module))
            found = outer
        return apply_scope(found, rule.scope, rule.pick, f"Modul-Regel '{rule.label or rule.target}'")

    def preview(self) -> str:
        lines = []
        for i, rule in enumerate(self.rules):
            hits = self.matches(rule)
            lines.append(f"{rule.label or f'Modul-Regel {i}'}: op={rule.op.name} → {len(hits)} Module")
            lines += [f"    {n}  ({type(m).__name__})" for n, m in hits[:12]]
            if len(hits) > 12:
                lines.append(f"    … und {len(hits) - 12} weitere")
        return "\n".join(lines) or "(keine Modul-Regeln)"

    def apply(self) -> "ModuleSurgeon":
        for rule in self.rules:
            hits = self.matches(rule)
            if not hits:
                raise ConfigError(f"Modul-Regel '{rule.label or rule.target}' trifft kein Modul.")
            op = rule.op.build()
            for name, module in hits:
                if isinstance(op, _Hook):
                    self.handles.append(module.register_forward_hook(op.fn))
                else:
                    op(self.model, name, module, {})
                self.applied.append((rule.op.name, name))
        return self

    def remove_hooks(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def activation_report(self) -> str:
        """Aufgezeichnete Ausgabestatistiken (Operation ``record``)."""
        rows = [(n, m._ki_stats) for n, m in self.model.named_modules() if hasattr(m, "_ki_stats")]
        if not rows:
            return "(keine Aufzeichnungen – Regel mit op 'record' anlegen und einen Forward ausführen)"
        lines = [f"{'Modul':<44} {'Mittel':>10} {'Std':>10} {'|max|':>10}  Form"]
        lines += [f"{n:<44} {s['mean']:>10.4g} {s['std']:>10.4g} {s['absmax']:>10.4g}  {s['shape']}" for n, s in rows]
        return "\n".join(lines)

    def report(self) -> str:
        lines = [f"Modul-Chirurgie: {len(self.rules)} Regeln, {len(self.applied)} Eingriffe"]
        lines += [f"  {op:<18} {name}" for op, name in self.applied[:40]]
        if len(self.applied) > 40:
            lines.append(f"  … {len(self.applied) - 40} weitere")
        return "\n".join(lines)
