"""
ki_baukasten.training
=====================

Alles rund ums Lernen:

* ``config``      – ``TrainingConfig`` (+ Regularisierung, Clipping, EMA, Curriculum)
* ``optimizers``  – Registry ``optimizer`` (sgd, adamw, lion, muon …)
* ``schedulers``  – Registry ``scheduler`` (warmup_cosine, wsd, one_cycle, plateau …)
* ``losses``      – Registry ``loss`` (frei kombinierbar, gewichtet)
* ``metrics``     – Registry ``metric`` (f1, auroc, ece, rmse, r2, bleu, iou …)
* ``paradigms``   – Registry ``paradigm`` (supervised, causal_lm, gan, diffusion, dpo …)
* ``callbacks``   – Registry ``callback`` (early_stopping, checkpoint, logger …)
* ``trainer``     – der allgemeine ``Trainer``
* ``rl``          – Reinforcement Learning (Umgebungen, Belohnungen, Algorithmen)
"""

from . import callbacks, losses, metrics, optimizers, paradigms, schedulers  # noqa: F401
from .callbacks import CALLBACKS, Callback
from .config import (CurriculumConfig, EMAConfig, GradientClippingConfig, RegularizationConfig, TrainingConfig)
from .losses import LOSSES
from .metrics import METRICS, compute_metrics
from .optimizers import OPTIMIZERS
from .paradigms import PARADIGMS, Paradigm, call_model, main_output
from .schedulers import SCHEDULERS
from .trainer import EMA, Trainer, set_seed
from . import rl  # noqa: F401,E402  (registriert das Paradigma 'reinforcement')
from .rl import RLConfig  # noqa: E402

__all__ = [
    "TrainingConfig", "RegularizationConfig", "GradientClippingConfig", "EMAConfig", "CurriculumConfig",
    "Trainer", "EMA", "set_seed", "Paradigm", "Callback", "call_model", "main_output", "RLConfig",
    "OPTIMIZERS", "SCHEDULERS", "LOSSES", "PARADIGMS", "CALLBACKS", "METRICS", "compute_metrics",
]
