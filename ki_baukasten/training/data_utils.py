"""
ki_baukasten.training.data_utils
================================

Alles, was Daten in Batches verwandelt.

Akzeptierte Datenformen (für ``Trainer.fit`` und ``Trainer.evaluate``)
--------------------------------------------------------------------
* ``(X, y)``-Tupel aus Tensoren               → ``{"inputs": X, "targets": y}``
* ``dict`` aus Tensoren gleicher Länge        → Batches mit denselben Schlüsseln
* ``torch.utils.data.Dataset``                → Elemente als dict oder (x, y)
* ``torch.utils.data.DataLoader``             → wird unverändert genutzt
* einzelner Tensor ``X``                      → ``{"inputs": X}`` (unüberwacht)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, default_collate


class DictDataset(Dataset):
    """Dataset aus einem ``dict`` von Tensoren (oder verschachtelten dicts) gleicher Länge."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.n = _length(data)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Any]:
        return _index(self.data, i)


def _length(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.shape[0]
    if isinstance(obj, dict):
        lengths = {_length(v) for v in obj.values()}
        if len(lengths) != 1:
            raise ValueError(f"Alle Einträge müssen gleich lang sein, gefunden: {sorted(lengths)}")
        return lengths.pop()
    if isinstance(obj, (list, tuple)):
        return len(obj)
    raise TypeError(f"Nicht unterstützter Datentyp {type(obj).__name__}")


def _index(obj: Any, i: int) -> Any:
    if isinstance(obj, dict):
        return {k: _index(v, i) for k, v in obj.items()}
    return obj[i]


def collate(items: list[Any]) -> dict[str, Any]:
    """Fasst Elemente zu einem Batch-dict zusammen; (x, y)-Tupel → inputs/targets."""
    first = items[0]
    if isinstance(first, dict):
        return default_collate(items)
    if isinstance(first, (tuple, list)):
        cols = default_collate(items)
        return {"inputs": cols[0], "targets": cols[1]} if len(cols) >= 2 else {"inputs": cols[0]}
    return {"inputs": default_collate(items)}


def as_dataset(data: Any) -> Dataset:
    """Wandelt alle unterstützten Datenformen in ein ``Dataset``."""
    if isinstance(data, Dataset):
        return data
    if isinstance(data, dict):
        return DictDataset(data)
    if isinstance(data, torch.Tensor):
        return DictDataset({"inputs": data})
    if isinstance(data, (tuple, list)) and len(data) == 2 and all(isinstance(d, (torch.Tensor, dict)) for d in data):
        return DictDataset({"inputs": data[0], "targets": data[1]})
    raise TypeError(
        "Daten müssen Dataset, DataLoader, dict von Tensoren, Tensor oder (X, y)-Tupel sein – "
        f"erhalten: {type(data).__name__}."
    )


def split_dataset(ds: Dataset, fraction: float, seed: int | None, strategy: str = "zufall"
                  ) -> tuple[Dataset, Dataset | None]:
    """Teilt einen Validierungsanteil ab: ``zufall`` | ``stratifiziert`` (Klassenanteile gleich) | ``zeitlich``
    (die letzten Beispiele – keine Zukunftsinformation im Training)."""
    from ..data.files import split_indices

    n = len(ds)  # type: ignore[arg-type]
    if strategy == "zufall":
        n_val = int(round(n * fraction))
        if fraction <= 0 or n_val == 0 or n_val >= n:
            return ds, None
        g = torch.Generator().manual_seed(seed if seed is not None else 0)
        perm = torch.randperm(n, generator=g).tolist()
        return Subset(ds, perm[n_val:]), Subset(ds, perm[:n_val])
    labels = class_labels(ds) if strategy == "stratifiziert" else None
    train, val = split_indices(n, fraction, strategy, labels.tolist() if labels is not None else None,
                               seed if seed is not None else 0)
    if not val:
        return ds, None
    return Subset(ds, train), Subset(ds, val)


def class_labels(ds: Any) -> torch.Tensor | None:
    """Ganzzahlige Klassen-Ziele eines (Teil-)Datensatzes oder ``None``."""
    if isinstance(ds, Subset):
        inner = class_labels(ds.dataset)
        return inner[list(ds.indices)] if inner is not None else None
    data = getattr(ds, "data", None) if isinstance(ds, DictDataset) else None
    targets = data.get("targets") if isinstance(data, dict) else None
    if not isinstance(targets, torch.Tensor) or targets.is_floating_point() or targets.dim() != 1:
        return None
    return targets


def class_weight_tensor(ds: Any, mode: str) -> torch.Tensor | None:
    """Klassengewichte aus den Häufigkeiten: ``ausgeglichen`` = N / (K · n_k), ``wurzel`` = deren Wurzel (milder)."""
    if mode in (None, "keine"):
        return None
    labels = class_labels(ds)
    if labels is None:
        return None
    labels = labels[labels >= 0].long()
    counts = torch.bincount(labels).float()
    present = counts > 0
    weights = torch.ones_like(counts)
    weights[present] = len(labels) / (int(present.sum()) * counts[present])
    if mode == "wurzel":
        weights = weights.sqrt()
    return weights


def to_device(obj: Any, device: torch.device, non_blocking: bool = False) -> Any:
    """Verschiebt Tensoren in (verschachtelten) dicts/Listen auf ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device, non_blocking) for v in obj)
    return obj


class CurriculumSampler(Sampler[int]):
    """Liefert pro Epoche nur den 'leichtesten' Anteil der Daten (wachsend mit dem Fortschritt).

    Args:
        difficulties: Schwierigkeit je Element (kleiner = leichter).
        progress:     Funktion → Trainingsfortschritt in [0, 1].
    """

    def __init__(self, difficulties: Sequence[float], start_fraction: float, full_after: float,
                 progress: Callable[[], float], shuffle: bool = True, seed: int = 0):
        self.order = sorted(range(len(difficulties)), key=lambda i: difficulties[i])
        self.start, self.full_after, self.progress = start_fraction, full_after, progress
        self.shuffle, self.epoch, self.seed = shuffle, 0, seed

    def fraction(self) -> float:
        p = self.progress()
        if self.full_after <= 0 or p >= self.full_after:
            return 1.0
        return self.start + (1 - self.start) * p / self.full_after

    def __iter__(self) -> Iterator[int]:
        k = max(1, int(math.ceil(len(self.order) * self.fraction())))
        chosen = self.order[:k]
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            chosen = [chosen[i] for i in torch.randperm(len(chosen), generator=g).tolist()]
        self.epoch += 1
        return iter(chosen)

    def __len__(self) -> int:
        return len(self.order)


def difficulty_of(item: Any, strategy: str) -> float:
    """Schwierigkeit eines Elements für das Curriculum."""
    if strategy == "key":
        if not isinstance(item, dict) or "difficulty" not in item:
            raise KeyError("Curriculum 'key' braucht ein Feld 'difficulty' in jedem Datenelement.")
        return float(item["difficulty"])
    x = item["inputs"] if isinstance(item, dict) else item[0]
    if isinstance(x, torch.Tensor):
        if not x.is_floating_point() and x.dim() >= 1:
            return float((x != 0).sum())  # Tokens ≠ 0 (Padding) zählen
        return float(x.shape[0]) if x.dim() >= 1 else 1.0
    return float(len(x))


def find_collate(ds: Any) -> Callable | None:
    """Sucht eine eigene ``collate_fn`` am Dataset (auch durch ``Subset``-Hüllen hindurch)."""
    while ds is not None:
        fn = getattr(ds, "collate_fn", None)
        if fn is not None:
            return fn
        ds = getattr(ds, "dataset", None)
    return None


def _tensor_source(ds: Any) -> tuple[dict, torch.Tensor | None] | None:
    """(Tensor-dict, Index-Abbildung) für ``DictDataset`` und (verschachtelte) ``Subset`` davon – sonst ``None``."""
    mapping = None
    while isinstance(ds, Subset):
        idx = torch.as_tensor(list(ds.indices), dtype=torch.long)
        mapping = idx if mapping is None else idx[mapping]
        ds = ds.dataset
    if not isinstance(ds, DictDataset) or getattr(ds, "collate_fn", None) is not None:
        return None

    def all_tensors(obj: Any) -> bool:
        if isinstance(obj, dict):
            return all(all_tensors(v) for v in obj.values())
        return isinstance(obj, torch.Tensor) and obj.dim() > 0
    return (ds.data, mapping) if all_tensors(ds.data) else None


def _take(obj: Any, idx: torch.Tensor) -> Any:
    if isinstance(obj, dict):
        return {k: _take(v, idx) for k, v in obj.items()}
    return obj.index_select(0, idx) if idx.device == obj.device else obj[idx.to(obj.device)]


class TensorBatchLoader(DataLoader):
    """DataLoader für Daten, die als Tensoren im Speicher liegen: holt jeden Batch mit **einem** Index-Zugriff
    je Tensor statt Beispiel für Beispiel zu lesen und wieder zusammenzufügen (bei kleinen Modellen oft der
    größte Zeitfresser). Reihenfolge, Sampler, ``drop_last`` und Länge verhalten sich exakt wie beim
    normalen ``DataLoader``; die Batches sind identisch zu ``default_collate``."""

    def __init__(self, dataset: Dataset, data: dict, mapping: torch.Tensor | None, batch_size: int, shuffle: bool,
                 drop_last: bool, sampler: Sampler | None, pin_memory: bool):
        self._data, self._mapping, self._pin = data, mapping, pin_memory and torch.cuda.is_available()
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, drop_last=drop_last,
                         sampler=sampler, collate_fn=collate)

    def __iter__(self):
        batches = iter(self.batch_sampler)
        # wie DataLoader: einen Basis-Seed ziehen → gleicher Zufallsverlauf (Reihenfolge, Dropout …) wie bisher
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        for indices in batches:
            idx = torch.as_tensor(indices, dtype=torch.long)
            if self._mapping is not None:
                idx = self._mapping[idx]
            batch = _take(self._data, idx)
            if self._pin:
                batch = pin(batch)
            yield batch


def pin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: pin(v) for k, v in obj.items()}
    return obj.pin_memory() if isinstance(obj, torch.Tensor) and obj.device.type == "cpu" else obj


def make_loader(data: Any, batch_size: int, shuffle: bool, drop_last: bool = False, num_workers: int = 0,
                pin_memory: bool = False, sampler: Sampler | None = None,
                collate_fn: Callable | None = None) -> DataLoader:
    """Baut einen DataLoader (ein vorhandener DataLoader wird unverändert zurückgegeben).

    Liegen die Daten als Tensoren im Speicher (Standardfall) und laufen keine Hintergrundprozesse, entsteht ein
    ``TensorBatchLoader`` (gleiche Batches, viel weniger Aufwand je Batch).
    """
    if isinstance(data, DataLoader):
        return data
    ds = as_dataset(data)
    source = _tensor_source(ds) if collate_fn is None and num_workers == 0 else None
    if source is not None:
        return TensorBatchLoader(ds, source[0], source[1], batch_size, shuffle, drop_last, sampler, pin_memory)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=pin_memory, sampler=sampler,
                      collate_fn=collate_fn or find_collate(ds) or collate)
