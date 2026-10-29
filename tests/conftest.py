"""Gemeinsame Test-Einstellungen: deterministisch, ruhig, klein."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def quiet(cfg):
    """Schaltet die Konsolenausgabe eines Projekts/Trainings ab."""
    cfg.training.verbose = False
    return cfg
