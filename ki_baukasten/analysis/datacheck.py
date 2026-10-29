"""
ki_baukasten.analysis.datacheck
===============================

**Datenprüfung** – die meisten KI-Probleme sind Datenprobleme. ``check_data`` (bzw.
``Project.data_check()``) untersucht Trainings- und Validierungsdaten *vor* dem Training und
nennt zu jedem Befund eine konkrete Abhilfe (Einstellung oder Technik zum Einschalten).

======================  =============================================================================
Prüfung                 Warum
======================  =============================================================================
Umfang                  zu wenige Beispiele für die Modellgröße → Auswendiglernen
NaN/Inf                 zerstören das Training sofort
konstante Merkmale      tragen keine Information
Skalen                  stark unterschiedliche Wertebereiche → langsames, instabiles Lernen
Klassenverteilung       seltene Klassen werden ignoriert, Genauigkeit wird irreführend
Klassen mit wenigen     zu wenige Beispiele, um sie zu lernen bzw. fair zu bewerten
Beispielen
Duplikate               verfälschen Training und Bewertung
Überlappung Train/Val   geschönte Validierung (Datenleck)
Verdacht auf Leck       ein einzelnes Merkmal sagt das Ziel (fast) perfekt voraus
Verschiebung Train/Val  Validierung misst etwas anderes als das Training
abgeschnittene Texte    wichtige Information geht verloren (seq_len zu klein)
unbekannte Tokens       Tokenizer/Vokabular passt nicht
======================  =============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

LEVELS = {"ok": "✓", "hinweis": "ℹ", "warnung": "⚠", "fehler": "✗"}


@dataclass
class Finding:
    """Ein Befund der Datenprüfung."""

    check: str
    level: str                       # ok | hinweis | warnung | fehler
    text: str
    fix: str = ""                    # Abhilfe in Worten
    settings: dict[str, Any] = field(default_factory=dict)   # direkt anwendbare Einstellungen
    techniques: list[str] = field(default_factory=list)       # passende Techniken (Project.enable)


@dataclass
class DataReport:
    """Ergebnis von ``check_data`` – ``str(report)`` für die Übersicht."""

    findings: list[Finding]
    stats: dict[str, Any]

    def problems(self, min_level: str = "warnung") -> list[Finding]:
        order = list(LEVELS)
        return [f for f in self.findings if order.index(f.level) >= order.index(min_level)]

    @property
    def ok(self) -> bool:
        return not self.problems("warnung")

    def settings(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in self.problems("hinweis"):
            out.update(f.settings)
        return out

    def techniques(self) -> list[str]:
        seen: list[str] = []
        for f in self.problems("hinweis"):
            seen += [t for t in f.techniques if t not in seen]
        return seen

    def __str__(self) -> str:
        s = self.stats
        head = f"Datenprüfung: {s.get('n_train', '?')} Trainings-, {s.get('n_val', 0)} Validierungsbeispiele"
        if s.get("features"):
            head += f", {s['features']} Merkmale"
        if s.get("task"):
            head += f", Aufgabe: {s['task']}"
        lines = [head, "=" * len(head)]
        order = list(LEVELS)
        for f in sorted(self.findings, key=lambda f: -order.index(f.level)):
            lines.append(f"{LEVELS[f.level]} [{f.check}] {f.text}")
            if f.fix:
                lines.append(f"    → {f.fix}")
        n = len(self.problems("warnung"))
        lines.append("")
        lines.append("Keine ernsten Probleme gefunden." if n == 0 else
                     f"{n} Problem(e). Vorschläge übernehmen: project.apply_data_fixes()")
        return "\n".join(lines)


# ===========================================================================
# Hilfen
# ===========================================================================

def _tensors(data: Any) -> tuple[Any, Any, Any]:
    """(inputs, targets, rest) aus dict/Tupel/Dataset – oder (None, None, data), wenn nicht greifbar."""
    from ..training.data_utils import DictDataset

    if isinstance(data, DictDataset):
        data = data.data
    if isinstance(data, torch.utils.data.Subset):
        x, y, _ = _tensors(data.dataset)
        idx = torch.tensor(list(data.indices), dtype=torch.long)
        pick = (lambda t: t[idx] if isinstance(t, torch.Tensor) else None)
        return pick(x), pick(y), data
    if isinstance(data, dict):
        return data.get("inputs"), data.get("targets"), data
    if isinstance(data, (tuple, list)) and len(data) == 2 and all(isinstance(t, torch.Tensor) for t in data):
        return data[0], data[1], data
    if isinstance(data, torch.Tensor):
        return data, None, data
    return None, None, data


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _row_keys(x: torch.Tensor, limit: int = 20000) -> list[int]:
    flat = _rows(x[:limit]).float()
    return [hash(tuple(r)) for r in flat.round(decimals=6).tolist()]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float() - a.float().mean(), b.float() - b.float().mean()
    den = float(a.norm() * b.norm())
    return float((a * b).sum()) / den if den > 0 else 0.0


# ===========================================================================
# Prüfung
# ===========================================================================

def check_data(train: Any, val: Any = None, info: dict | None = None, n_params: int | None = None,
               task: str = "auto") -> DataReport:
    """Prüft Trainings- (und Validierungs-)Daten und liefert Befunde mit Abhilfen."""
    info = info or {}
    F: list[Finding] = []
    x, y, raw = _tensors(train)
    vx, vy, _ = _tensors(val) if val is not None else (None, None, None)
    stats: dict[str, Any] = {"n_train": int(x.shape[0]) if isinstance(x, torch.Tensor) else _len(train),
                             "n_val": int(vx.shape[0]) if isinstance(vx, torch.Tensor) else _len(val)}
    if not isinstance(x, torch.Tensor):
        F.append(Finding("format", "hinweis", "Daten liegen nicht als Tensoren vor (z. B. Graphen, eigenes Dataset) "
                                              "– nur der Umfang wurde geprüft."))
        return DataReport(F, stats)

    is_tokens = not x.is_floating_point() and x.dim() == 2
    is_image = x.is_floating_point() and x.dim() == 4
    is_cls = isinstance(y, torch.Tensor) and not y.is_floating_point() and y.dim() == 1
    if task == "auto":
        task = "klassifikation" if is_cls else ("regression" if isinstance(y, torch.Tensor) else
                                                ("sprache" if is_tokens else "ohne Ziele"))
    stats["task"] = task
    n = stats["n_train"]
    if x.is_floating_point() and x.dim() == 2:
        stats["features"] = int(x.shape[1])

    # --- Umfang -------------------------------------------------------------------
    if n < 50:
        F.append(Finding("umfang", "warnung", f"Nur {n} Trainingsbeispiele.",
                         "Mehr Daten sammeln, Augmentierung einschalten oder ein vortrainiertes Modell feintunen.",
                         techniques=["bild_augmentierung" if is_image else "eingaberauschen", "label_smoothing"]
                         if not is_tokens else ["token_rauschen"]))
    if n_params and n:
        ratio = n_params / n
        stats["parameter_pro_beispiel"] = round(ratio, 1)
        if ratio > 2000 and not is_tokens:
            F.append(Finding("umfang", "warnung", f"{ratio:,.0f} Parameter pro Trainingsbeispiel – hohe Gefahr des "
                                                  "Auswendiglernens.".replace(",", "."),
                             "Kleineres Modell, Regularisierung (Dropout, Weight Decay), Early Stopping, Augmentierung.",
                             techniques=["dropout", "weight_decay", "early_stopping"]))
    if vx is None and val is None:
        F.append(Finding("validierung", "hinweis", "Keine eigene Validierungsmenge – sie wird automatisch abgeteilt "
                                                   "(training.validation_split).",
                         "Für eine ehrliche Endbewertung zusätzlich eine Testmenge zurückhalten.",
                         settings={"training.test_split": 0.15}))

    # --- NaN / Inf --------------------------------------------------------------------
    for name, t in (("Eingaben", x), ("Ziele", y)):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            bad = int((~torch.isfinite(t)).sum())
            if bad:
                F.append(Finding("nan", "fehler", f"{bad} NaN/Inf-Werte in den {name}.",
                                 "Fehlende Werte ersetzen (CSV: data.missing = 'median') oder Zeilen entfernen."))

    # --- Merkmale (Tabellen) ------------------------------------------------------------
    if x.is_floating_point() and x.dim() == 2 and n > 1:
        finite = torch.where(torch.isfinite(x), x, torch.zeros_like(x)).float()
        std = finite.std(0)
        const = int((std < 1e-12).sum())
        if const:
            F.append(Finding("konstant", "hinweis", f"{const} Merkmal(e) sind konstant und tragen keine Information.",
                             "Spalten entfernen (CSV: data.drop_columns)."))
        scale = finite.abs().max(0).values
        spread = float(scale.max() / scale[scale > 0].min()) if (scale > 0).any() else 1.0
        stats["skalen_verhaeltnis"] = round(spread, 1)
        if spread > 100:
            F.append(Finding("skalen", "warnung", f"Wertebereiche der Merkmale unterscheiden sich um Faktor "
                                                  f"{spread:,.0f}.".replace(",", "."),
                             "Merkmale standardisieren (CSV: data.scaling = 'standard' oder 'robust').",
                             settings={"data.scaling": "standard"} if info.get("preprocessor") is not None else {}))
        # Datenleck: ein Merkmal sagt das Ziel fast perfekt voraus
        if isinstance(y, torch.Tensor) and y.shape[0] == n and y.numel() == n:
            target = y.reshape(-1).float()
            if float(target.std()) > 0:
                names = info.get("features") or [f"Merkmal {j}" for j in range(x.shape[1])]
                for j in range(min(x.shape[1], 500)):
                    r = abs(_corr(finite[:, j], target))
                    if r > 0.995:
                        F.append(Finding("leck", "warnung", f"'{names[j] if j < len(names) else j}' korreliert zu "
                                                            f"{r:.3f} mit dem Ziel – Verdacht auf Datenleck.",
                                         "Prüfen, ob dieses Merkmal zur Vorhersagezeit wirklich bekannt ist; sonst "
                                         "entfernen."))
                        break

    # --- Klassen ------------------------------------------------------------------------
    if is_cls:
        counts = torch.bincount(y.clamp_min(0))
        present = counts[counts > 0]
        stats["klassen"] = {int(k): int(v) for k, v in enumerate(counts.tolist()) if v}
        names = info.get("label_names")
        if len(present) == 1:
            F.append(Finding("klassen", "fehler", "Alle Trainingsbeispiele haben dieselbe Klasse.",
                             "Daten mit mehreren Klassen verwenden."))
        elif len(present) > 1:
            ratio = float(present.max() / present.min())
            stats["unausgewogenheit"] = round(ratio, 1)
            if ratio > 3:
                F.append(Finding("klassen", "warnung", f"Unausgewogene Klassen (häufigste : seltenste = {ratio:.1f} : 1)"
                                                       " – Genauigkeit allein ist irreführend.",
                                 "Klassengewichte oder ausgewogenes Ziehen einschalten, nach f1/balanced_accuracy "
                                 "bewerten, stratifiziert aufteilen.",
                                 settings={"training.class_weights": "wurzel" if ratio < 10 else "ausgeglichen",
                                           "training.validation_strategy": "stratifiziert"},
                                 techniques=["klassengewichte"]))
            rare = [k for k, v in enumerate(counts.tolist()) if 0 < v < 10]
            if rare:
                label = ", ".join(str(names[k]) if names and k < len(names) else str(k) for k in rare[:5])
                F.append(Finding("klassen", "warnung", f"Klassen mit weniger als 10 Beispielen: {label}.",
                                 "Mehr Beispiele sammeln oder Klassen zusammenlegen; stratifiziert aufteilen.",
                                 settings={"training.validation_strategy": "stratifiziert"}))
            if isinstance(vy, torch.Tensor) and not vy.is_floating_point():
                missing = sorted(set(stats["klassen"]) - set(vy.unique().tolist()))
                if missing:
                    F.append(Finding("klassen", "hinweis", f"{len(missing)} Klasse(n) fehlen in der Validierung.",
                                     "Stratifiziert aufteilen.", settings={"training.validation_strategy": "stratifiziert"}))

    # --- Duplikate & Überlappung ---------------------------------------------------------
    if n > 1 and x.numel() / max(1, n) <= 4096:
        keys = _row_keys(x)
        dup = len(keys) - len(set(keys))
        stats["duplikate"] = dup
        # Textfenster eines Sprachmodells wiederholen sich leicht (nicht aber Frage-Antwort-Paare)
        windows = is_tokens and y is None and not (isinstance(raw, dict) and "loss_mask" in raw)
        if dup / len(keys) > 0.05:
            F.append(Finding("duplikate", "warnung" if dup / len(keys) > 0.2 and not windows else "hinweis",
                             f"{dup} doppelte Trainingsbeispiele ({dup / len(keys):.0%}).",
                             "Mehr bzw. abwechslungsreicheren Text verwenden." if windows else
                             "Duplikate entfernen – sie verzerren das Training zu ihren Gunsten."))
        if isinstance(vx, torch.Tensor) and vx.shape[1:] == x.shape[1:]:
            overlap = len(set(_row_keys(vx)) & set(keys))
            stats["ueberlappung"] = overlap
            if overlap and not windows:
                F.append(Finding("ueberlappung", "warnung", f"{overlap} Validierungsbeispiele kommen auch im Training "
                                                            "vor – die Validierung ist geschönt.",
                                 "Duplikate vor dem Aufteilen entfernen; bei Zeitreihen zeitlich aufteilen."))

    # --- Verschiebung Train/Val ------------------------------------------------------------
    if isinstance(vx, torch.Tensor) and x.is_floating_point() and x.dim() == 2 and vx.shape[1:] == x.shape[1:] \
            and vx.shape[0] > 5:
        from .drift import feature_drift

        drift = feature_drift(x, vx)
        strong = [d for d in drift if d["psi"] > 0.25]
        if strong:
            F.append(Finding("verschiebung", "hinweis", f"{len(strong)} Merkmal(e) sind in der Validierung anders "
                                                        "verteilt als im Training (PSI > 0.25).",
                             "Bei zeitlicher Aufteilung normal (Drift) – sonst zufällig/stratifiziert aufteilen."))

    # --- Texte / Tokens --------------------------------------------------------------------
    if is_tokens:
        trunc = info.get("truncated_fraction")
        if trunc is not None and trunc > 0.1:
            F.append(Finding("texte", "warnung", f"{trunc:.0%} der Texte werden abgeschnitten.",
                             "data.seq_len erhöhen oder einen kompakteren Tokenizer wählen (bpe).",
                             settings={"data.seq_len": int(info.get("seq_len", 64) * 2)}))
        unk = float((x == 1).float().mean()) if info.get("tokenizer") is not None else 0.0
        if unk > 0.05:
            F.append(Finding("texte", "warnung", f"{unk:.0%} unbekannte Tokens (<unk>).",
                             "Größeres Vokabular oder BPE-Tokenizer verwenden.", settings={"data.tokenizer": "bpe"}))
        stats["vokabular_genutzt"] = int(x.unique().numel())

    # --- Bilder --------------------------------------------------------------------------
    if is_image:
        mean = x.float().mean((0, 2, 3))
        std = x.float().std((0, 2, 3))
        stats["kanal_mittel"] = [round(float(v), 3) for v in mean]
        stats["kanal_std"] = [round(float(v), 3) for v in std]
        if float(mean.abs().max()) > 1.5 or float(std.max()) > 5:
            F.append(Finding("bilder", "warnung", "Bildwerte sind nicht normalisiert.",
                             "Bildordner: data.normalize = 'datensatz' (oder durch 255 teilen)."))

    if not any(f.level in ("warnung", "fehler") for f in F):
        F.append(Finding("gesamt", "ok", "Keine ernsten Datenprobleme gefunden."))
    return DataReport(F, stats)


def _len(data: Any) -> int:
    if data is None:
        return 0
    try:
        return len(data)
    except TypeError:
        return 0
