"""
ki_baukasten.components
=======================

Alle austauschbaren Bausteine, aus denen Modelle zusammengesetzt werden.
Das Importieren dieses Pakets füllt die folgenden Registries:

=====================  ======================================================
Registry               Inhalt
=====================  ======================================================
``activation``         Aktivierungsfunktionen (relu, gelu, silu, snake ...)
``normalization``      Normalisierungen (layernorm, rmsnorm, batchnorm, dyt ...)
``initializer``        Initialisierer (xavier, kaiming, orthogonal, siren ...)
``positional``         Positionskodierungen (rope, alibi, learned, t5 ...)
``attention_kernel``   Attention-Kernel (softmax, sigmoid, linear, cosine ...)
``feedforward``        FFN-Varianten (mlp, glu/SwiGLU, moe)
``sequence_mixer``     Mixer eines Blocks (attention, ssm/Mamba, rnn, conv, fourier)
``layer``              Universelle Schichten für freie Netze
``merge``              Zusammenführungen (concat, add, gated ...)
``function``           Parameterfreie Tensor-Funktionen
=====================  ======================================================
"""

from . import activations, attention, feedforward, initializers, layers, mixers, normalization, positional  # noqa: F401
from .activations import ACTIVATIONS, build_activation
from .attention import AttentionConfig, MaskInfo, MultiHeadAttention
from .feedforward import FEEDFORWARDS
from .initializers import INITIALIZERS, apply_initializer
from .layers import FUNCTIONS, LAYERS, MERGES
from .mixers import MIXERS, SSMConfig, SelectiveSSM
from .normalization import NORMALIZATIONS, build_norm
from .positional import POSITIONALS, PositionalEncoding

__all__ = [
    "ACTIVATIONS", "NORMALIZATIONS", "INITIALIZERS", "POSITIONALS", "FEEDFORWARDS", "LAYERS", "MERGES", "FUNCTIONS",
    "MIXERS", "SSMConfig", "SelectiveSSM",
    "AttentionConfig", "MultiHeadAttention", "MaskInfo", "PositionalEncoding",
    "build_activation", "build_norm", "apply_initializer",
]
