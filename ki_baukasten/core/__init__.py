"""
ki_baukasten.core
=================

Das Fundament, auf dem alles andere aufbaut:

* ``tooltip``    – ``Tooltip``/``T`` (Erklärungen) und ``Level`` (Stufen)
* ``config``     – ``Param``-Arten und ``BaseConfig``
* ``registry``   – Kataloge austauschbarer Komponenten
* ``component``  – ``ComponentSpec``/``ComponentParam`` (Komponenten in Configs)
* ``selectors``  – Schicht-Selektoren und schichtweise Overrides
* ``errors``     – verständliche Fehlerklassen

Dieses Paket hängt **nicht** von PyTorch ab – das Konfigurationssystem kann
also auch ohne installiertes PyTorch genutzt werden (z. B. in Oberflächen).
"""

from .component import ComponentListParam, ComponentParam, ComponentSpec
from .config import (
    BaseConfig,
    ConfigListParam,
    EmptyConfig,
    FreeConfig,
    NestedParam,
    Param,
    ParamRef,
    flatten_dict,
)
from .errors import (
    BuildError,
    ConfigError,
    KIBaukastenError,
    RegistryError,
    TrainingError,
    UnknownParameterError,
)
from .registry import Component, Registry, all_registries, create_registry, register, registry
from .selectors import matches, resolve_layer_config, select_indices, validate_overrides
from .tooltip import Level, T, Tooltip

__all__ = [
    "BaseConfig", "Param", "NestedParam", "ConfigListParam", "ComponentParam", "ComponentListParam",
    "ComponentSpec", "EmptyConfig", "FreeConfig", "ParamRef", "flatten_dict",
    "Component", "Registry", "registry", "create_registry", "register", "all_registries",
    "Level", "T", "Tooltip",
    "matches", "select_indices", "resolve_layer_config", "validate_overrides",
    "KIBaukastenError", "ConfigError", "UnknownParameterError", "RegistryError", "BuildError", "TrainingError",
]
