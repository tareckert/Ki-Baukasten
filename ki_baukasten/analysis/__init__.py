"""
ki_baukasten.analysis
=====================

Werkzeuge, um eine KI zu **verstehen** und gezielt zu **verbessern**:

* ``architecture`` – komplette Auflistung/Darstellung des Aufbaus (Text, Markdown, Mermaid, JSON)
* ``advisor``      – Optimierungs- und Speicher-Ratgeber mit direkt anwendbaren Vorschlägen
* ``export``       – Export (torch.export, TorchScript, ONNX, state_dict)
"""

from .advisor import Advice, advise, apply_advice
from .architecture import ArchitectureInfo, ModuleInfo, architecture_report, inspect_model
from .export import FORMATS as EXPORT_FORMATS, export_model

__all__ = ["ArchitectureInfo", "ModuleInfo", "architecture_report", "inspect_model", "Advice", "advise",
           "apply_advice", "export_model", "EXPORT_FORMATS"]
