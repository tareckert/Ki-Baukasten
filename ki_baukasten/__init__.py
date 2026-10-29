"""
KI-Baukasten
============

Ein Python-Baukasten, mit dem sich KIs und ganze KI-Systeme **bis ins
kleinste Detail** konfigurieren, kombinieren, trainieren und speichern lassen.

Leitprinzipien
--------------
1. **Jede Einstellung ist erklärt.** Jeder Parameter trägt einen Tooltip
   (Was? Warum/Nutzen? Wofür wichtig? Tipp) – abrufbar per
   ``Project.explain(pfad)``, CLI (``python -m ki_baukasten erklaere …``) oder GUI.
2. **Alles ist änderbar.** Auch experimentelle, exotische oder scheinbar
   unnötige Optionen – Stufen (Basis … Experimentell) dienen nur der Orientierung.
3. **Alles ist kombinierbar.** Komponenten (Aktivierungen, Normen, Attention-
   Kernel, Mixer, FFNs, Modelle, Verluste, Belohnungen …) stecken in Registries
   und sind überall einsetzbar – auch eigene.
4. **Jede Schicht darf anders sein.** Schichtweise Overrides und Tensor-
   Chirurgie bis auf einzelne Ausschnitte von Gewichtstensoren.
5. **Speicher ist planbar.** Präzision, Checkpointing, trainierbares RAM-Streaming,
   Offloading/Swapping, eigene int8/int4/nf4-Quantisierung, LoRA/QLoRA, 8-Bit-AdamW
   und eine Speicherschätzung mit Empfehlungen.
6. **Alles ist durchschaubar.** Komplette Aufbau-Darstellung, Ratgeber mit direkt
   übernehmbaren Vorschlägen, Rückgängig/Wiederholen, Export.

Schnellstart
------------
::

    from ki_baukasten import Project
    p = Project.from_preset("tabellen_klassifikation")
    p.set("model.hidden_dims", [256, 256])
    p.train()

Paketübersicht
--------------
``core``        Parameter-/Tooltip-System, Registries, Selektoren
``components``  Aktivierungen, Normen, Init, Attention, Positionen, FFN/MoE, Mixer, Schichten
``models``      MLP, CNN, RNN, Transformer, SSM, Hybrid, AE/VAE, GAN, Diffusion, GNN, RL-Netze,
                freier Graph, Ensemble, Pipeline, Fusion, Mixture-of-Models, Multi-Task
``training``    Trainer, Paradigmen, Optimierer, Zeitpläne, Verluste, Callbacks, RL
``memory``      Präzision, Checkpointing, Streaming/Offloading, Quantisierung, LoRA, Speicherschätzung
``tensor``      Tensor- und Modul-Chirurgie, Geltungsbereiche
``compress``    Modelle eindampfen: Pruning, Low-Rank, Sparsity, Quantisierung, Heilen
``analysis``    Aufbau-Darstellung, Ratgeber, Export, Messung
``data``        Datensätze (synthetisch + eigene Dateien)
``presets``     Vorlagen je Anwendungsgebiet, Speicherprofile
``project``     ``Project``-Fassade
``ui``          Kommandozeile, grafische Oberfläche, Doku-Export
"""

from __future__ import annotations

__version__ = "1.2.0"

from .core import (BaseConfig, BuildError, ComponentListParam, ComponentParam, ComponentSpec, ConfigError,
                   ConfigListParam, FreeConfig, KIBaukastenError, Level, NestedParam, Param, T, Tooltip,
                   TrainingError, all_registries, register, registry, select_indices)
from . import components, models, training, memory, tensor, compress, data, presets, analysis  # noqa: E402,F401
from .memory import MemoryConfig, estimate_memory  # noqa: E402
from .models import BaseModel, ModelConfig, build_model  # noqa: E402
from .presets import DOMAINS, MEMORY_PROFILES, apply_memory_profile, get_preset, presets_by_domain  # noqa: E402
from .project import Project, ProjectConfig  # noqa: E402
from .tensor import ModuleRuleConfig, ModuleSurgeon, TensorRuleConfig, TensorSurgeon  # noqa: E402
from .training import Trainer, TrainingConfig  # noqa: E402

__all__ = [
    "__version__", "Project", "ProjectConfig", "Trainer", "TrainingConfig", "MemoryConfig", "TensorRuleConfig",
    "TensorSurgeon", "ModuleRuleConfig", "ModuleSurgeon", "BaseModel", "ModelConfig", "build_model", "estimate_memory",
    "BaseConfig", "Param", "NestedParam", "ConfigListParam", "ComponentParam", "ComponentListParam", "ComponentSpec",
    "FreeConfig", "Level", "T", "Tooltip", "registry", "register", "all_registries", "select_indices",
    "get_preset", "presets_by_domain", "apply_memory_profile", "DOMAINS", "MEMORY_PROFILES",
    "KIBaukastenError", "ConfigError", "BuildError", "TrainingError",
]
