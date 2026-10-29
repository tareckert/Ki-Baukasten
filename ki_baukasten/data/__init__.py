"""
ki_baukasten.data
=================

Datensätze (Registry ``dataset``): synthetische Beispielaufgaben für jede
Modellart sowie Lader für eigene CSV-, Text- und Tensor-Dateien.
"""

from .datasets import BEISPIELTEXT, DATASETS, CharTokenizer, DatasetBundle, GraphDataset, sample_distribution

__all__ = ["DATASETS", "DatasetBundle", "CharTokenizer", "GraphDataset", "BEISPIELTEXT", "sample_distribution"]
