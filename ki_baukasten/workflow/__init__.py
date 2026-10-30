"""
ki_baukasten.workflow
=====================

Der **Ablauf** einer KI (``Project.run``) und die Verfahren, die über ein einzelnes Training hinausgehen:

* ``config``    – ``WorkflowConfig`` (Suche, föderiert, Model Soup, Kreuzvalidierung, Kompression, Export)
* ``search``    – Zufall, Gitter, genetischer Algorithmus, Partikelschwarm, Population Based Training
* ``federated`` – föderiertes Lernen (FedAvg/FedProx) und Model Soups
"""

from .config import FederatedConfig, SearchConfig, WorkflowConfig
from .federated import average_state_dicts, federated_training, model_soup, split_clients
from .search import SearchResult, default_space, parse_space, run_search

__all__ = ["WorkflowConfig", "SearchConfig", "FederatedConfig", "run_search", "SearchResult", "default_space",
           "parse_space", "federated_training", "model_soup", "average_state_dicts", "split_clients"]
