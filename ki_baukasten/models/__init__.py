"""
ki_baukasten.models
===================

Alle vorgefertigten Modelle. Jedes Modell besteht aus einer ``*Config``-Klasse
(alle Einstellungen mit Tooltips) und einer ``nn.Module``-Klasse und ist in
der Registry ``model`` eingetragen – damit überall wählbar und kombinierbar.

==================  ==========================================================
Registry-Name       Modell
==================  ==========================================================
mlp                 Mehrschichtiges Perzeptron
cnn                 Faltungsnetz 1D/2D/3D (plain, ResNet, MobileNet, ConvNeXt …)
rnn                 LSTM/GRU/RNN/minGRU, schichtweise kombinierbar
transformer         Decoder / Encoder / Encoder-Decoder, Text/Zeitreihen/Bilder
ssm                 Mamba-artiges State-Space-Modell
hybrid              Hybrid aus SSM- und Attention-Schichten (Jamba-artig)
autoencoder         AE, VAE, Denoising-, Sparse-, VQ-VAE
gan                 Generative Adversarial Network
diffusion           Diffusionsmodell (DDPM/DDIM)
gnn                 Graph-Netze (GCN, GraphSAGE, GIN, GAT)
actor_critic        Policy + Value für Reinforcement Learning
q_network           Q-Netz (Dueling, Noisy) für DQN
graph               freies Netz aus beliebigen Schichten
ensemble            Kombination: Ensemble
pipeline            Kombination: Hintereinanderschaltung
fusion              Kombination: multimodale Fusion
mixture_of_models   Kombination: Router über ganze Modelle
multitask           Kombination: gemeinsamer Rumpf, mehrere Köpfe
==================  ==========================================================
"""

from . import autoencoder, base, cnn, composite, generative, gnn, graph, mlp, rl_nets, rnn, transformer  # noqa: F401
from .autoencoder import Autoencoder, AutoencoderConfig
from .base import MODELS, BaseModel, InitConfig, ModelConfig, build_model
from .cnn import CNN, CNNConfig
from .composite import Ensemble, Fusion, MixtureOfModels, MultiTask, Pipeline
from .generative import GAN, DiffusionConfig, DiffusionModel, GANConfig
from .gnn import GNN, GNNConfig
from .graph import GraphModel, GraphModelConfig
from .mlp import MLP, MLPConfig
from .rl_nets import ActorCritic, ActorCriticConfig, QNetwork, QNetworkConfig
from .rnn import RecurrentConfig, RecurrentNet
from .transformer import GenerationConfig, Transformer, TransformerBlockConfig, TransformerConfig

__all__ = [
    "MODELS", "BaseModel", "ModelConfig", "InitConfig", "build_model",
    "MLP", "MLPConfig", "CNN", "CNNConfig", "RecurrentNet", "RecurrentConfig",
    "Transformer", "TransformerConfig", "TransformerBlockConfig", "GenerationConfig",
    "Autoencoder", "AutoencoderConfig", "GAN", "GANConfig", "DiffusionModel", "DiffusionConfig",
    "GNN", "GNNConfig", "ActorCritic", "ActorCriticConfig", "QNetwork", "QNetworkConfig",
    "GraphModel", "GraphModelConfig", "Ensemble", "Pipeline", "Fusion", "MixtureOfModels", "MultiTask",
]
