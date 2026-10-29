"""
ki_baukasten.memory.config
==========================

``MemoryConfig`` – alle Weichen für Speicher, Präzision und Geschwindigkeit.

Wo geht der Speicher hin? (Training)
------------------------------------
=====================  ========================================  =============================
Posten                 Größe                                     Stellschraube
=====================  ========================================  =============================
Parameter              N · Bytes/Wert                            ``param_dtype``, Quantisierung
Gradienten             N · Bytes/Wert                            ``param_dtype``, Einfrieren
Optimierer-Zustände    0–2 · N · 4 Bytes (Adam: 2 Kopien)        Optimierer (lion/adafactor)
Aktivierungen          ∝ Batch · Sequenz · Breite · Tiefe        ``mixed_precision``,
                                                                 ``gradient_checkpointing``,
                                                                 ``activation_offload``,
                                                                 Batch + Akkumulation
=====================  ========================================  =============================

Für sehr große Modelle (größer als der GPU-Speicher) gibt es **Auslagerung**
(``offload``): Blöcke liegen im CPU-RAM oder auf der Festplatte und werden nur
für ihre eigene Berechnung auf das Rechengerät geladen – ``cpu``/``disk`` für
Inferenz, ``stream`` auch für das **Training** (Gewichte, Gradienten und
Optimierer-Zustände der Blöcke bleiben im RAM).
"""

from __future__ import annotations

from ..core import BaseConfig, Level, NestedParam, Param, T
from .lora import LoRAConfig

_DTYPES = {
    "float32": "32-Bit (Standard, genau)",
    "bfloat16": "16-Bit Brain-Float (großer Wertebereich, ideal für Training auf modernen GPUs)",
    "float16": "16-Bit (kleiner Wertebereich – nur mit Vorsicht/Loss-Scaling)",
    "float64": "64-Bit (sehr genau, langsam – Wissenschaft/Debugging)",
}


class OffloadConfig(BaseConfig):
    __title__ = "Offloading / Layer-Swapping"
    __tooltip__ = T(
        "Auslagern von Modellblöcken auf CPU-RAM oder Festplatte (Swapping).",
        "Ermöglicht Modelle, die größer als der Speicher des Rechengeräts sind.",
        "Inferenz sehr großer Modelle auf kleiner GPU / wenig RAM.",
    )
    mode = Param("none", choices={
        "none": "alles bleibt auf dem Rechengerät",
        "cpu": "Blöcke liegen im CPU-RAM, werden pro Forward auf das Gerät geladen (Inferenz/eingefroren)",
        "disk": "Blöcke liegen auf der Festplatte (Memory-Map), werden pro Forward geladen (Inferenz/eingefroren)",
        "stream": "TRAINIERBAR: Gewichte, Gradienten und Optimierer-Zustände im RAM, Blöcke werden in "
                  "Forward und Backward gestreamt",
        "split": "Blöcke aufteilen: die gewählten Blöcke (layers) rechnen auf split_device (z. B. CPU), der Rest "
                 "auf dem Hauptgerät – trainierbar, wie 'device_map' bei großen Modellen",
    }, tooltip=T(
        "Auslagerungsmodus.",
        "Tauscht Geschwindigkeit gegen Speicher: disk spart am meisten, ist aber am langsamsten; stream erlaubt "
        "Training von Modellen, deren Gewichte + Optimierer nicht auf die GPU passen.",
        "Inferenz großer Modelle: cpu/disk. Training großer Modelle auf kleiner GPU: stream.",
        "stream rechnet jeden Block im Backward neu (wie Checkpointing) und überträgt die Gewichte zweimal pro "
        "Schritt – typischerweise 1.3–3× langsamer.",
    ))
    targets = Param("auto", tooltip=T(
        "Welche Module ausgelagert werden: 'auto' = Hauptblöcke des Modells, sonst Regex auf Modulnamen.",
        "Kleine Teile (Embeddings, Kopf) bleiben dauerhaft auf dem Gerät.",
        "Meist 'auto'; z. B. 'blocks\\.(1[0-9]|2[0-9])' für nur hintere Blöcke.",
    ), level=Level.ADVANCED)
    layers = Param("*", tooltip=T(
        "Welche der automatisch gefundenen Blöcke ausgelagert werden (Selektor wie bei layer_overrides).",
        "Feiner Kompromiss: nur so viele Blöcke auslagern, wie nötig – der Rest rechnet ohne Übertragung.",
        "'*' = alle, '0:12' = die ersten 12, '::2' = jeder zweite. Nur mit targets='auto'.",
    ), level=Level.ADVANCED)
    storage = Param("ram", choices={
        "ram": "eingefrorene Gewichte der gestreamten Blöcke im CPU-RAM",
        "disk": "eingefrorene Gewichte/Puffer auf der Festplatte (Memory-Map) – für Modelle größer als der RAM",
    }, tooltip=T(
        "Ablageort der eingefrorenen Gewichte im Modus 'stream'.",
        "Mit 'disk' passt auch ein Modell, das größer als der RAM ist: z. B. QLoRA/LoRA-Feintuning, bei dem nur "
        "die kleinen Adapter im RAM liegen und die Basis Block für Block von der SSD gelesen wird.",
        "Sehr große Modelle auf kleinen Rechnern. Schnelle SSD empfohlen (directory).",
        "Trainierbare Gewichte, Gradienten und Optimierer-Zustände bleiben immer im RAM.",
    ), level=Level.ADVANCED)
    split_device = Param("cpu", tooltip=T(
        "Zweites Rechengerät für den Modus 'split' (z. B. 'cpu', 'cuda:1').",
        "Blöcke, die nicht auf das Hauptgerät passen, rechnen dort – langsamer, aber es läuft.",
        "GPU zu klein: hintere Blöcke auf der CPU; zwei GPUs: Modell aufteilen.",
    ), level=Level.ADVANCED)
    prefetch = Param(False, tooltip=T(
        "Nächsten Block auf einem eigenen CUDA-Stream vorladen, während der aktuelle rechnet (nur stream + CUDA).",
        "Versteckt Übertragungszeit hinter Rechenzeit.", "Experimentell; bei Problemen ausschalten.",
    ), level=Level.EXPERIMENTAL)
    directory = Param("offload_cache", tooltip=T(
        "Ordner für ausgelagerte Gewichte (disk).", "Schnelle SSD empfohlen.", "Lokale SSD."))
    pin_memory = Param(True, tooltip=T(
        "CPU-Kopien in gepinntem Speicher halten (cpu-Modus, nur CUDA).",
        "Deutlich schnellere Übertragung CPU → GPU.", "CUDA."), level=Level.EXPERT)
    keep_resident = Param(1, min=0, tooltip=T(
        "Wie viele zuletzt genutzte Blöcke auf dem Gerät bleiben dürfen.",
        "Mehr = weniger Nachladen, mehr Speicher.", "1–2."), level=Level.ADVANCED)


class MemoryConfig(BaseConfig):
    __title__ = "Speicher & Präzision"
    __tooltip__ = T(
        "Einstellungen für Speicherbedarf, numerische Präzision und Geschwindigkeit.",
        "Entscheiden, ob ein Modell überhaupt in den Speicher passt und wie schnell es läuft.",
        "Große Modelle, kleine GPUs, CPU-Training, Inferenz-Optimierung.",
    )

    device = Param("auto", tooltip=T(
        "Rechengerät: 'auto', 'cpu', 'cuda', 'cuda:1', 'mps' (Apple).",
        "GPUs sind für Training meist 10–100× schneller.", "'auto' wählt die beste verfügbare Option."),
        group="Gerät")
    param_dtype = Param("float32", choices=_DTYPES, group="Präzision", tooltip=T(
        "Datentyp der Gewichte.",
        "16 Bit halbiert den Speicher für Gewichte und Gradienten.",
        "Reines bf16-Training großer Modelle; sonst float32 + mixed_precision.",
    ))
    mixed_precision = Param("none", choices={
        "none": "alles in param_dtype",
        "bf16": "Rechnen in bfloat16, Gewichte in param_dtype (robust, kein Loss-Scaling nötig)",
        "fp16": "Rechnen in float16 (mit automatischem Loss-Scaling)",
    }, group="Präzision", tooltip=T(
        "Gemischte Präzision (autocast).",
        "Halbiert Aktivierungsspeicher und beschleunigt auf Tensor-Core-GPUs deutlich.",
        "GPU-Training: bf16 (Ampere+), fp16 (ältere GPUs).",
    ))
    grad_scaler = Param("auto", choices={"auto": "an bei fp16", "on": "immer", "off": "nie"}, group="Präzision",
                        level=Level.EXPERT, tooltip=T(
        "Dynamisches Loss-Scaling.", "Verhindert Unterlauf kleiner fp16-Gradienten.", "auto."))
    gradient_checkpointing = Param(False, group="Aktivierungen", tooltip=T(
        "Aktivierungen der Blöcke nicht speichern, sondern im Backward neu berechnen.",
        "Spart bei tiefen Netzen oft 60–80 % Aktivierungsspeicher – kostet ~30 % mehr Rechenzeit.",
        "Große Modelle / lange Sequenzen, wenn der Speicher knapp ist.",
    ))
    checkpoint_layers = Param("*", group="Aktivierungen", level=Level.ADVANCED, tooltip=T(
        "Welche Blöcke neu berechnet werden (Selektor wie bei layer_overrides).",
        "Feiner Kompromiss Speicher vs. Zeit (z. B. nur jeden zweiten Block).", "'*', '::2', '0:8'."))
    activation_offload = Param(False, group="Aktivierungen", level=Level.ADVANCED, tooltip=T(
        "Für den Backward gespeicherte Aktivierungen in den CPU-RAM auslagern.",
        "Spart GPU-Speicher ohne Neuberechnung, kostet Übertragungszeit.", "Lange Sequenzen auf kleiner GPU."))
    offload = NestedParam(OffloadConfig, group="Auslagerung", level=Level.ADVANCED)
    quantization = Param("none", choices={
        "none": "keine",
        "int8": "eigene Gewichts-Quantisierung 8 Bit (alle Geräte; Basis eingefroren, LoRA trainierbar)",
        "int4": "eigene Gewichts-Quantisierung 4 Bit, gruppenweise (halber int8-Speicher)",
        "nf4": "4-Bit NormalFloat wie QLoRA (beste 4-Bit-Qualität für normalverteilte Gewichte)",
        "pal4": "4-Bit-Palette: 16 per k-means gelernte Werte je Matrix (Apple-artige 'Palettization')",
        "fake_quant": "Gewichte im Forward auf n Bit runden (Quantization-Aware Training, voll trainierbar)",
        "torchao_int8": "int8 über das Paket torchao (optimierte Kernel, falls installiert)",
        "torchao_int4": "int4 über das Paket torchao (optimierte Kernel, falls installiert)",
        "dynamic_int8": "VERALTET: torch.ao dynamic int8 (nur CPU-Inferenz; PyTorch entfernt diese API)",
    }, group="Quantisierung", level=Level.ADVANCED, tooltip=T(
        "Quantisierung der Gewichte linearer Schichten.",
        "int8 braucht ¼, int4/nf4 ⅛ des float32-Gewichtsspeichers – Inferenz großer Modelle und "
        "QLoRA-Feintuning; fake_quant bereitet ein Modell auf echte Quantisierung vor.",
        "Inferenz/Edge: int8; Feintuning großer Modelle: nf4 + memory.lora (QLoRA).",
        "int8/int4/nf4 sind eine eigene Implementierung in reinem PyTorch – unabhängig von torch.ao.",
    ))
    quant_bits = Param(8, min=2, max=16, group="Quantisierung", level=Level.ADVANCED, tooltip=T(
        "Bitbreite für fake_quant.", "Weniger Bits = kleiner, ungenauer.", "8 (sicher), 4 (aggressiv)."))
    quant_group_size = Param(128, type=int, min=8, nullable=True, group="Quantisierung", level=Level.ADVANCED,
                             tooltip=T(
        "Gruppengröße: so viele Gewichte einer Zeile teilen sich eine Skala (None = eine Skala pro Ausgabekanal).",
        "Kleinere Gruppen = genauer (wichtig bei 4 Bit), minimal mehr Speicher für Skalen.",
        "int8: None oder 128; int4/nf4: 32–128."))
    quant_skip = Param("lm_head", type=str, nullable=True, group="Quantisierung", level=Level.ADVANCED, tooltip=T(
        "Regex: Module, die NICHT quantisiert werden.",
        "Ausgabeschicht und empfindliche Teile in voller Genauigkeit halten.", "'lm_head', 'lm_head|router'."))
    quant_only = Param(None, type=str, nullable=True, group="Quantisierung", level=Level.EXPERT, tooltip=T(
        "Regex: nur diese Module quantisieren (None = alle linearen Schichten).",
        "Gezielt nur große Schichten (z. B. FFN) komprimieren.", "'ffn', 'blocks\\.(1[0-9])'."))
    quant_compute_dtype = Param("auto", choices={
        "auto": "wie die Eingabe bzw. autocast", "float32": "32 Bit", "bfloat16": "bf16", "float16": "fp16"},
        group="Quantisierung", level=Level.EXPERT, tooltip=T(
        "Datentyp, in den die Gewichte zum Rechnen entpackt werden.", "Geschwindigkeit vs. Genauigkeit.", "auto."))
    quant_targets = Param("linear", choices={"linear": "nur lineare Schichten", "linear_conv": "linear + Faltungen"},
                          group="Quantisierung", level=Level.EXPERT, tooltip=T(
        "Welche Schichten fake_quant betrifft.", "Faltungen sind empfindlicher.", "linear."))
    lora = NestedParam(LoRAConfig, group="Quantisierung", level=Level.ADVANCED)
    channels_last = Param(False, group="Geschwindigkeit", level=Level.EXPERT, tooltip=T(
        "Speicherlayout NHWC für Faltungsnetze.", "Schneller auf Tensor-Core-GPUs.", "CNN-Training auf GPU."))
    compile = Param(False, group="Geschwindigkeit", level=Level.ADVANCED, tooltip=T(
        "Modell mit torch.compile übersetzen.", "Oft 1.3–2× schneller nach Aufwärmzeit.", "Lange Trainings auf GPU."))
    compile_mode = Param("default", choices=["default", "reduce-overhead", "max-autotune"], group="Geschwindigkeit",
                         level=Level.EXPERT, tooltip=T("Modus von torch.compile.", "Mehr Optimierung = längeres Kompilieren.",
                                                       "default."))
    matmul_precision = Param("highest", choices={
        "highest": "volle fp32-Genauigkeit", "high": "TF32 erlaubt (schneller auf Ampere+)",
        "medium": "bf16-Zwischenergebnisse erlaubt (am schnellsten)",
    }, group="Geschwindigkeit", level=Level.EXPERT, tooltip=T(
        "Genauigkeit von fp32-Matrixmultiplikationen.", "Geschwindigkeit vs. Genauigkeit.", "Training auf A100/H100: high."))
    cpu_threads = Param(None, type=int, min=1, group="Gerät", level=Level.ADVANCED, tooltip=T(
        "Anzahl CPU-Threads für PyTorch (None = automatisch).", "Begrenzen auf geteilten Rechnern.", "Kernanzahl."))
    max_memory_fraction = Param(None, type=float, min=0.05, max=1.0, group="Gerät", level=Level.EXPERT, tooltip=T(
        "Maximaler Anteil des GPU-Speichers für diesen Prozess.", "Mehrere Prozesse auf einer GPU.", "0.5."))
    empty_cache_every_n_steps = Param(None, type=int, min=1, group="Gerät", level=Level.EXPERT, tooltip=T(
        "GPU-Cache alle n Schritte leeren.", "Gegen Fragmentierung bei variablen Formen.", "Selten nötig."))
    oom_strategy = Param("batch_teilen", choices={
        "fehler": "bei Speichermangel abbrechen (klassisch)",
        "batch_teilen": "Batch automatisch halbieren und mit Gradientenakkumulation weiterrechnen (gleiches Ergebnis)",
    }, group="Planung", tooltip=T(
        "Was bei Speichermangel (Out-of-Memory) während Training/Vorhersage passiert.",
        "'batch_teilen' teilt den Batch so oft wie nötig in kleinere Stücke; durch Akkumulation bleibt die "
        "effektive Batchgröße – und damit das Training – gleich. Das Training bricht nicht mehr mitten ab.",
        "Unbekannte Hardware, schwankende Sequenzlängen, Notfall-Ausführung.",
        "Erkennt CUDA-/MPS-/CPU-Speicherfehler. Mehr Teilstücke = langsamer.",
    ))
    oom_max_splits = Param(64, min=1, group="Planung", level=Level.ADVANCED, tooltip=T(
        "Höchstens so viele Teilstücke pro Batch.", "Begrenzt die Verlangsamung; danach wird der Fehler gemeldet.",
        "16–64."))
    memory_budget_gb = Param(None, type=float, min=0.0, group="Planung", tooltip=T(
        "Verfügbarer Speicher in GB für die Speicherschätzung (None = automatisch erkennen).",
        "Die Schätzung prüft, ob das Training hineinpasst, und schlägt Einstellungen vor.",
        "Planung vor dem Training großer Modelle.",
    ))

    def check(self) -> list[str]:
        problems = []
        if self.param_dtype == "bfloat16" and self.mixed_precision == "fp16":
            problems.append("param_dtype=bfloat16 mit mixed_precision=fp16 ist widersprüchlich.")
        return problems
