"""
ki_baukasten.presets
====================

**Vorlagen (Presets) je Anwendungsgebiet** und **Speicherprofile**.

Ein Preset ist eine vollständige, sinnvoll vorbelegte ``ProjectConfig`` –
der Startpunkt, von dem aus man jede Einstellung weiter verändern kann.
Alle Presets laufen ohne Downloads (eingebaute synthetische Daten) und
passen sich über ``auto_fit_dims`` auch an eigene Daten an.

Nutzung::

    from ki_baukasten import Project
    p = Project.from_preset("bildklassifikation_cnn")
    # oder in der Kommandozeile:
    #   python -m ki_baukasten presets
    #   python -m ki_baukasten vorlage bildklassifikation_cnn -o meine_ki.json

Registry ``preset``
-------------------
Fabrik-Signatur: ``factory(args) -> dict`` (Konfiguration als dict).
Eigene Vorlagen einfach registrieren::

    from ki_baukasten import registry, T
    registry("preset").register("meine_vorlage", lambda a: {...}, tags=("bild",),
                                tooltip=T("Meine Vorlage", "...", "..."))
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from ..core import T, create_registry

PRESETS = create_registry(
    "preset",
    "Vorlagen (Presets)",
    T(
        "Vollständig vorkonfigurierte KI-Projekte je Anwendungsgebiet.",
        "Sinnvolle Standardwerte als Startpunkt – alles bleibt änderbar.",
        "Schneller Einstieg, Vergleichsbasis für Experimente.",
    ),
    context_doc="factory(args) -> dict (ProjectConfig)",
)

#: Anwendungsgebiete mit Beschreibung (Tags der Presets).
DOMAINS: dict[str, str] = {
    "sprache": "Sprachmodelle, Textverständnis, Übersetzung",
    "bild": "Bildklassifikation, Bildgenerierung",
    "zeitreihen": "Prognosen, Sensordaten, Signale",
    "tabellen": "Tabellendaten: Klassifikation und Regression",
    "generativ": "Daten erzeugen: GAN, Diffusion, VAE",
    "graph": "Graphen: Moleküle, Netzwerke",
    "rl": "Reinforcement Learning mit frei kombinierbaren Belohnungen",
    "praeferenz": "Lernen aus Präferenzen (Reward-Modelle, RLHF-Bausteine)",
    "anomalie": "Anomalie- und Ausreißererkennung",
    "kombination": "Kombinationen mehrerer Modelle (Ensemble, Fusion, Mixture)",
    "forschung": "Experimentelle und exotische Konfigurationen",
}


def _project(name: str, domain: str, description: str, model: dict, data: Any, training: dict,
             memory: dict | None = None, tensor_rules: list | None = None) -> dict:
    out = {"name": name, "domain": domain, "description": description, "model": model, "data": data,
           "training": training}
    if memory:
        out["memory"] = memory
    if tensor_rules:
        out["tensor_rules"] = tensor_rules
    return out


_LM_TRAIN = {
    "paradigm": "causal_lm", "epochs": 10, "batch_size": 32,
    "optimizer": {"name": "adamw", "args": {"lr": 3e-3, "weight_decay": 0.1, "betas": [0.9, 0.95]}},
    "scheduler": {"name": "warmup_cosine", "args": {"warmup_ratio": 0.05, "min_lr_ratio": 0.1}},
    "gradient_clipping": {"mode": "norm", "max_norm": 1.0}, "log_every_steps": 20,
}
_TEXT = {"name": "text", "args": {"seq_len": 64, "stride": 4, "repeat": 8}}


def _lm_block(heads: int = 4, **extra: Any) -> dict:
    block = {"mixer": {"name": "attention", "args": {"num_heads": heads}}, "ffn": "glu", "norm": "rmsnorm",
             "dropout": 0.1}
    block.update(extra)
    return block


# ===========================================================================
# Sprache
# ===========================================================================

def sprachmodell_mini(a=None) -> dict:
    return _project(
        "sprachmodell_mini", "sprache",
        "Kleines GPT-artiges Sprachmodell (RoPE, SwiGLU, RMSNorm, Pre-Norm) auf Zeichenebene.",
        {"name": "transformer", "args": {"d_model": 128, "n_layers": 4, "max_seq_len": 128, "block": _lm_block(4),
                                         "positional": "rope"}},
        _TEXT, copy.deepcopy(_LM_TRAIN))


def sprachmodell_moe(a=None) -> dict:
    p = sprachmodell_mini()
    p.update(name="sprachmodell_moe", description="Sprachmodell mit Mixture-of-Experts in jeder zweiten Schicht.")
    p["model"]["args"]["layer_overrides"] = {"1::2": {"ffn": {"name": "moe", "args": {
        "num_experts": 4, "top_k": 2, "aux_loss_coef": 0.01}}}}
    return p


def sprachmodell_mamba(a=None) -> dict:
    return _project("sprachmodell_mamba", "sprache", "Mamba-artiges State-Space-Sprachmodell (ohne Attention).",
                    {"name": "ssm", "args": {"d_model": 128, "n_layers": 4, "max_seq_len": 128}},
                    _TEXT, copy.deepcopy(_LM_TRAIN))


def sprachmodell_hybrid(a=None) -> dict:
    return _project("sprachmodell_hybrid", "sprache", "Hybrid aus SSM- und Attention-Schichten (Jamba-Prinzip).",
                    {"name": "hybrid", "args": {"d_model": 128, "n_layers": 4, "max_seq_len": 128}},
                    _TEXT, copy.deepcopy(_LM_TRAIN))


def sprachmodell_speichersparend(a=None) -> dict:
    t = copy.deepcopy(_LM_TRAIN)
    t.update(batch_size=8, gradient_accumulation_steps=4, max_steps=300)
    return _project(
        "sprachmodell_speichersparend", "sprache",
        "Größeres Sprachmodell mit allen Speicher-Weichen: GQA, Gradient-Checkpointing, bf16-Autocast, Akkumulation.",
        {"name": "transformer", "args": {"d_model": 384, "n_layers": 8, "max_seq_len": 256,
                                         "block": _lm_block(8, mixer={"name": "attention",
                                                                      "args": {"num_heads": 8, "num_kv_heads": 2}}),
                                         "positional": "rope"}},
        _TEXT, t, memory={"mixed_precision": "bf16", "gradient_checkpointing": True})


def text_klassifikation(a=None) -> dict:
    return _project(
        "text_klassifikation", "sprache", "Encoder-Transformer (BERT-artig) mit CLS-Token zur Sequenzklassifikation.",
        {"name": "transformer", "args": {"architecture": "encoder_only", "d_model": 64, "n_layers": 2, "pooling": "cls",
                                         "output_dim": 2, "max_seq_len": 64, "positional": "learned",
                                         "block": {"mixer": {"name": "attention", "args": {"num_heads": 4}},
                                                   "norm": "layernorm", "ffn": "mlp", "dropout": 0.1}}},
        {"name": "sequence_classification", "args": {"task": "contains", "seq_len": 24}},
        {"epochs": 15, "optimizer": {"name": "adamw", "args": {"lr": 1e-3}},
         "callbacks": [{"name": "early_stopping", "args": {"patience": 4}}]})


def bert_vortraining(a=None) -> dict:
    return _project(
        "bert_vortraining", "sprache", "Maskiertes Sprachmodell-Vortraining (BERT-Stil).",
        {"name": "transformer", "args": {"architecture": "encoder_only", "d_model": 128, "n_layers": 4,
                                         "max_seq_len": 128, "positional": "learned",
                                         "block": {"mixer": {"name": "attention", "args": {"num_heads": 4}},
                                                   "norm": "layernorm", "ffn": "mlp"}}},
        _TEXT, {**copy.deepcopy(_LM_TRAIN), "paradigm": {"name": "masked_lm", "args": {"mask_prob": 0.15}}})


def uebersetzung_seq2seq(a=None) -> dict:
    return _project(
        "uebersetzung_seq2seq", "sprache", "Encoder-Decoder-Transformer für Sequenz-zu-Sequenz (hier: Folgen umkehren).",
        {"name": "transformer", "args": {"architecture": "encoder_decoder", "d_model": 64, "n_layers": 2,
                                         "max_seq_len": 32, "positional": "sinusoidal",
                                         "block": {"mixer": {"name": "attention", "args": {"num_heads": 4}},
                                                   "norm": "layernorm", "ffn": "mlp"}}},
        {"name": "seq2seq_toy", "args": {"task": "reverse"}},
        {"paradigm": {"name": "seq2seq", "args": {"label_smoothing": 0.1}}, "epochs": 20,
         "optimizer": {"name": "adamw", "args": {"lr": 2e-3}}})


# ===========================================================================
# Bild
# ===========================================================================

def bildklassifikation_cnn(a=None) -> dict:
    return _project(
        "bildklassifikation_cnn", "bild", "ResNet-artiges CNN mit Label Smoothing und Mixup.",
        {"name": "cnn", "args": {"stem": {"channels": 16}, "stages": [
            {"channels": 16, "num_blocks": 1, "stride": 1}, {"channels": 32, "num_blocks": 2, "stride": 2},
            {"channels": 64, "num_blocks": 2, "stride": 2}], "head_dropout": 0.1}},
        {"name": "shapes", "args": {"n_samples": 1500}},
        {"epochs": 8, "optimizer": {"name": "adamw", "args": {"lr": 3e-3, "weight_decay": 0.05}},
         "losses": [{"name": "cross_entropy", "args": {"label_smoothing": 0.1}}],
         "regularization": {"mixup_alpha": 0.2}})


def bildklassifikation_vit(a=None) -> dict:
    return _project(
        "bildklassifikation_vit", "bild", "Vision Transformer (Patches, CLS-Token, Stochastic Depth).",
        {"name": "transformer", "args": {"architecture": "encoder_only", "input_type": "patches", "patch_size": 4,
                                         "image_size": [16, 16], "image_channels": 1, "d_model": 64, "n_layers": 4,
                                         "pooling": "cls", "output_dim": 3, "positional": "learned",
                                         "block": {"mixer": {"name": "attention", "args": {"num_heads": 4}},
                                                   "norm": "layernorm", "ffn": "mlp", "drop_path": 0.1}}},
        {"name": "shapes", "args": {"n_samples": 1500}},
        {"epochs": 12, "optimizer": {"name": "adamw", "args": {"lr": 2e-3, "weight_decay": 0.05}}})


def bildklassifikation_mobil(a=None) -> dict:
    return _project(
        "bildklassifikation_mobil", "bild", "Effizientes MobileNet/EfficientNet-artiges CNN (Inverted Residuals + SE).",
        {"name": "cnn", "args": {"stem": {"channels": 16, "activation": "hardswish"}, "stages": [
            {"channels": 16, "num_blocks": 1, "stride": 1, "block_type": "inverted_residual", "expansion": 4.0,
             "se_ratio": 4.0, "activation": "hardswish"},
            {"channels": 32, "num_blocks": 2, "stride": 2, "block_type": "inverted_residual", "expansion": 4.0,
             "se_ratio": 4.0, "activation": "hardswish"}], "head_dropout": 0.2}},
        {"name": "shapes", "args": {"n_samples": 1500}},
        {"epochs": 8, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}})


def vae_bilder(a=None) -> dict:
    return _project(
        "vae_bilder", "generativ", "Faltungs-VAE für Bilder (β-VAE mit KL-Annealing).",
        {"name": "autoencoder", "args": {"variant": "vae", "backbone": "conv", "in_channels": 1, "image_size": [16, 16],
                                         "hidden_dims": [16, 32], "latent_dim": 8, "beta": 1.0}},
        {"name": "shapes", "args": {"n_samples": 1500, "noise": 0.0}},
        {"paradigm": {"name": "autoencoder", "args": {"reconstruction_loss": "mse", "kl_warmup_steps": 200}},
         "epochs": 10, "optimizer": {"name": "adamw", "args": {"lr": 2e-3}}})


# ===========================================================================
# Zeitreihen
# ===========================================================================

def zeitreihen_lstm(a=None) -> dict:
    return _project(
        "zeitreihen_lstm", "zeitreihen", "LSTM-Prognose mehrerer Zukunftsschritte (Huber-Verlust).",
        {"name": "rnn", "args": {"hidden_sizes": [64, 64], "pooling": "last", "layer": {"cell": "lstm", "dropout": 0.1}}},
        {"name": "time_series"},
        {"epochs": 15, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}, "losses": [{"name": "huber"}]})


def zeitreihen_transformer(a=None) -> dict:
    return _project(
        "zeitreihen_transformer", "zeitreihen", "Transformer-Encoder für kontinuierliche Zeitreihen.",
        {"name": "transformer", "args": {"architecture": "encoder_only", "input_type": "continuous", "d_model": 64,
                                         "n_layers": 2, "pooling": "mean", "positional": "sinusoidal", "max_seq_len": 64,
                                         "block": {"mixer": {"name": "attention", "args": {"num_heads": 4}}}}},
        {"name": "time_series"},
        {"epochs": 15, "optimizer": {"name": "adamw", "args": {"lr": 1e-3}}, "losses": [{"name": "mse"}]})


def sequenz_gedaechtnis_rnn(a=None) -> dict:
    return _project(
        "sequenz_gedaechtnis_rnn", "zeitreihen",
        "Gemischter RNN-Stapel (LSTM → GRU → minGRU) auf einer Gedächtnisaufgabe.",
        {"name": "rnn", "args": {"input_type": "tokens", "hidden_sizes": [64, 64, 64], "pooling": "attention",
                                 "layer": {"cell": "lstm", "norm": "layernorm"},
                                 "layer_overrides": {"1": {"cell": "gru"}, "2": {"cell": "mingru", "residual": True}}}},
        {"name": "sequence_classification", "args": {"task": "majority"}},
        {"epochs": 15, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}})


# ===========================================================================
# Tabellen
# ===========================================================================

def tabellen_klassifikation(a=None) -> dict:
    return _project(
        "tabellen_klassifikation", "tabellen", "Tiefes Residual-MLP mit LayerNorm, GELU und Dropout.",
        {"name": "mlp", "args": {"hidden_dims": [128, 128, 64], "layer": {
            "activation": "gelu", "norm": "layernorm", "dropout": 0.1, "residual": "projection"}}},
        {"name": "classification", "args": {"kind": "spirals"}},
        {"epochs": 60, "batch_size": 64, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}},
         "callbacks": [{"name": "early_stopping", "args": {"patience": 10, "monitor": "val_accuracy", "mode": "max"}}]})


def tabellen_regression(a=None) -> dict:
    return _project(
        "tabellen_regression", "tabellen", "MLP-Regression (Friedman-Benchmark) mit robustem Huber-Verlust.",
        {"name": "mlp", "args": {"hidden_dims": [128, 64], "layer": {"activation": "silu"}}},
        {"name": "regression", "args": {"function": "friedman"}},
        {"epochs": 40, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}, "losses": [{"name": "huber"}]})


# ===========================================================================
# Generativ
# ===========================================================================

def gan_2d(a=None) -> dict:
    return _project(
        "gan_2d", "generativ", "GAN auf einer 2D-Verteilung (Hinge-Verlust, Spectral-Norm, TTUR).",
        {"name": "gan", "args": {"data_dim": 2, "latent_dim": 8}},
        {"name": "distribution_2d", "args": {"kind": "moons", "n_samples": 2000}},
        {"paradigm": {"name": "gan", "args": {"loss": "hinge"}}, "epochs": 30, "batch_size": 64, "validation_split": 0.1,
         "optimizer": {"name": "adam", "args": {"lr": 2e-4, "betas": [0.5, 0.999]}}, "scheduler": "constant",
         "optimizer_overrides": {"discriminator": {"name": "adam", "args": {"lr": 4e-4, "betas": [0.5, 0.999]}}}})


def diffusion_2d(a=None) -> dict:
    return _project(
        "diffusion_2d", "generativ", "Diffusionsmodell auf 2D-Gauß-Ring (v-Vorhersage, Min-SNR, EMA, DDIM).",
        {"name": "diffusion", "args": {"timesteps": 200, "prediction": "v", "hidden_dim": 128, "num_blocks": 3}},
        {"name": "distribution_2d", "args": {"kind": "gaussians", "n_samples": 3000}},
        {"paradigm": {"name": "diffusion", "args": {"weighting": "min_snr"}}, "epochs": 40, "batch_size": 128,
         "optimizer": {"name": "adamw", "args": {"lr": 2e-3}}, "ema": {"enabled": True, "decay": 0.995}})


# ===========================================================================
# Graph, Anomalie, Präferenz
# ===========================================================================

def graph_klassifikation(a=None) -> dict:
    return _project(
        "graph_klassifikation", "graph", "Graph Isomorphism Network (GIN) zur Unterscheidung von Ringen und Bäumen.",
        {"name": "gnn", "args": {"hidden_dims": [64, 64, 64], "readout": "sum", "jumping_knowledge": "concat",
                                 "layer": {"conv": "gin", "aggregation": "sum"}}},
        {"name": "graphs"},
        {"epochs": 25, "batch_size": 32, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}})


def anomalie_autoencoder(a=None) -> dict:
    return _project(
        "anomalie_autoencoder", "anomalie", "Denoising-Autoencoder: hoher Rekonstruktionsfehler = Anomalie.",
        {"name": "autoencoder", "args": {"variant": "denoising", "hidden_dims": [32, 16], "latent_dim": 3,
                                         "corruption_level": 0.1}},
        {"name": "anomaly"},
        {"paradigm": "autoencoder", "epochs": 30, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}})


def reward_modell(a=None) -> dict:
    return _project(
        "reward_modell", "praeferenz", "Reward-Modell aus Präferenzpaaren (Bradley-Terry) – Baustein für RLHF.",
        {"name": "mlp", "args": {"hidden_dims": [64, 64], "output_dim": 1}},
        {"name": "preferences"},
        {"paradigm": "preference", "epochs": 15, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}})


# ===========================================================================
# Reinforcement Learning
# ===========================================================================

def _rl_train(rl: dict, lr: float = 3e-3) -> dict:
    return {"paradigm": {"name": "reinforcement", "args": rl}, "scheduler": "constant",
            "optimizer": {"name": "adamw", "args": {"lr": lr, "weight_decay": 0.0}},
            "gradient_clipping": {"mode": "norm", "max_norm": 0.5}}


def rl_cartpole_ppo(a=None) -> dict:
    return _project(
        "rl_cartpole_ppo", "rl", "PPO balanciert einen Stab (CartPole).",
        {"name": "actor_critic", "args": {"hidden_dims": [64, 64]}}, "none",
        _rl_train({"algorithm": {"name": "ppo", "args": {"n_steps": 128}}, "environment": "cartpole",
                   "total_timesteps": 30000, "log_every_env_steps": 5000}))


def rl_cartpole_dqn(a=None) -> dict:
    return _project(
        "rl_cartpole_dqn", "rl", "Double-DQN mit n-Schritt-Erträgen und priorisiertem Replay (CartPole).",
        {"name": "q_network", "args": {"hidden_dims": [128, 128], "dueling": True}}, "none",
        _rl_train({"algorithm": {"name": "dqn", "args": {"learning_starts": 500, "target_update_interval": 200,
                                                         "n_step": 3, "prioritized": True, "train_freq": 1}},
                   "environment": "cartpole", "num_envs": 1, "total_timesteps": 15000, "log_every_env_steps": 5000},
                  lr=1e-3))


def rl_gitterwelt_belohnungen(a=None) -> dict:
    return _project(
        "rl_gitterwelt_belohnungen", "rl",
        "Gitterwelt mit kombinierter Belohnung: Ziel + Zeitstrafe + Potential-Formung + Wandstrafe + Neuheitsbonus.",
        {"name": "actor_critic", "args": {"hidden_dims": [64, 64]}}, "none",
        _rl_train({"algorithm": {"name": "ppo", "args": {"n_steps": 64}},
                   "environment": {"name": "gridworld", "args": {"size": 6, "wall_density": 0.15, "slip_prob": 0.05}},
                   "total_timesteps": 40000, "eval_deterministic": False, "log_every_env_steps": 10000,
                   "rewards": [
                       {"name": "env", "args": {"kind": "extrinsisch"}},
                       {"name": "constant", "args": {"value": -0.01, "kind": "strafe"}},
                       {"name": "potential", "args": {"key": "distance", "weight": 0.2, "kind": "formend"}},
                       {"name": "event", "args": {"key": "hit_wall", "value": -0.05, "kind": "strafe"}},
                       {"name": "count", "args": {"weight": 0.02, "anneal": "linear_decay", "anneal_steps": 20000}},
                   ]}))


def rl_pendel_kontinuierlich(a=None) -> dict:
    return _project(
        "rl_pendel_kontinuierlich", "rl", "PPO mit Gauß-Policy für kontinuierliche Steuerung (Pendel) + Energiestrafe.",
        {"name": "actor_critic", "args": {"hidden_dims": [64, 64], "log_std_init": -0.5}}, "none",
        _rl_train({"algorithm": {"name": "ppo", "args": {"n_steps": 512, "n_epochs": 10, "gae_lambda": 0.95,
                                                         "entropy_coef": 0.0}},
                   "environment": "pendulum", "total_timesteps": 100000, "reward_normalization": "return",
                   "observation_normalization": True, "log_every_env_steps": 20000,
                   "rewards": ["env", {"name": "action_penalty", "args": {"weight": 0.001}}]}, lr=1e-3))


# ===========================================================================
# Kombinationen & Forschung
# ===========================================================================

def ensemble_tabellen(a=None) -> dict:
    return _project(
        "ensemble_tabellen", "kombination", "Ensemble dreier unterschiedlicher MLPs (Wahrscheinlichkeits-Mittel).",
        {"name": "ensemble", "args": {"combine": "softmax_mean", "members": [
            {"name": "mlp", "args": {"input_dim": 2, "output_dim": 3, "hidden_dims": [64, 64], "layer": {"activation": "relu"}}},
            {"name": "mlp", "args": {"input_dim": 2, "output_dim": 3, "hidden_dims": [128], "layer": {"activation": "gelu"}}},
            {"name": "mlp", "args": {"input_dim": 2, "output_dim": 3, "hidden_dims": [32, 32, 32],
                                     "layer": {"activation": "tanh", "norm": "layernorm"}}}]}},
        {"name": "classification", "args": {"kind": "spirals"}},
        {"epochs": 40, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}},
         "losses": [{"name": "nll"}]})


def multimodal_fusion(a=None) -> dict:
    return _project(
        "multimodal_fusion", "kombination", "Fusion von Bild (CNN) und Metadaten (MLP) per Attention.",
        {"name": "fusion", "args": {"fusion": "attention", "fusion_dim": 32, "output_dim": 3, "branches": [
            {"name": "bild", "input_key": "bild", "model": {"name": "cnn", "args": {
                "in_channels": 1, "input_size": [16, 16], "output_dim": 32,
                "stages": [{"channels": 16, "num_blocks": 1, "stride": 2}, {"channels": 32, "num_blocks": 1, "stride": 2}]}}},
            {"name": "meta", "input_key": "meta", "model": {"name": "mlp", "args": {"input_dim": 4, "output_dim": 32}}}]}},
        {"name": "multimodal", "args": {"n_samples": 1500}},
        {"epochs": 10, "optimizer": {"name": "adamw", "args": {"lr": 2e-3}}})


def mixture_of_models(a=None) -> dict:
    return _project(
        "mixture_of_models", "kombination", "Router wählt pro Beispiel zwischen verschiedenen Experten-Modellen.",
        {"name": "mixture_of_models", "args": {"top_k": 2, "experts": [
            {"name": "mlp", "args": {"input_dim": 5, "hidden_dims": [64], "layer": {"activation": "relu"}}},
            {"name": "mlp", "args": {"input_dim": 5, "hidden_dims": [64], "layer": {"activation": "sine"}}},
            {"name": "mlp", "args": {"input_dim": 5, "hidden_dims": [32, 32], "layer": {"activation": "tanh"}}}]}},
        {"name": "regression", "args": {"function": "sin"}},
        {"epochs": 30, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}})


def freies_netz(a=None) -> dict:
    return _project(
        "freies_netz", "forschung", "Frei definierter Netzgraph mit Verzweigung, Gating und Residual.",
        {"name": "graph", "args": {
            "inputs": [{"name": "x", "shape": [2]}],
            "nodes": [
                {"name": "a", "layer": {"name": "linear", "args": {"out_features": 64}}, "inputs": ["x"]},
                {"name": "a_act", "layer": {"name": "activation", "args": {"activation": "gelu"}}},
                {"name": "b", "layer": {"name": "linear", "args": {"out_features": 64}}, "inputs": ["x"]},
                {"name": "b_act", "layer": {"name": "activation", "args": {"activation": "sine"}}},
                {"name": "mix", "layer": "identity", "inputs": ["a_act", "b_act"], "merge": "gated"},
                {"name": "deep", "layer": {"name": "mlp", "args": {"hidden_dims": [64], "output_dim": 64}}},
                {"name": "res", "layer": {"name": "normalization", "args": {"norm": "layernorm"}},
                 "inputs": ["mix", "deep"], "merge": "add"},
                {"name": "out", "layer": {"name": "linear", "args": {"out_features": 3}}},
            ]}},
        {"name": "classification", "args": {"kind": "spirals"}},
        {"epochs": 60, "optimizer": {"name": "adamw", "args": {"lr": 3e-3}}})


def kontrastiv_vortraining(a=None) -> dict:
    return _project(
        "kontrastiv_vortraining", "forschung", "Selbstüberwachtes kontrastives Lernen (SimCLR-artig) ohne Labels.",
        {"name": "mlp", "args": {"hidden_dims": [128, 128], "output_dim": 32, "layer": {"norm": "batchnorm"}}},
        {"name": "classification", "args": {"n_features": 16, "kind": "blobs"}},
        {"paradigm": {"name": "contrastive", "args": {"temperature": 0.2}}, "epochs": 10, "batch_size": 128,
         "optimizer": {"name": "adamw", "args": {"lr": 1e-3}}})


def forschung_experimentell(a=None) -> dict:
    return _project(
        "forschung_experimentell", "forschung",
        "Exotischer Transformer: Sigmoid-Attention, QK-Norm, Output-Gate, DyT statt Norm, Sandwich, Muon + WSD.",
        {"name": "transformer", "args": {"d_model": 128, "n_layers": 4, "max_seq_len": 128, "positional": "alibi",
                                         "block": {"mixer": {"name": "attention", "args": {
                                             "num_heads": 4, "kernel": "sigmoid", "qk_norm": "rmsnorm",
                                             "output_gate": True}},
                                             "ffn": {"name": "glu", "args": {"activation": "squared_relu"}},
                                             "norm": "dyt", "norm_position": "sandwich", "layer_scale_init": 0.1},
                                         "layer_overrides": {"-1": {"mixer": {"name": "attention",
                                                                              "args": {"num_heads": 8}}}}}},
        _TEXT, {**copy.deepcopy(_LM_TRAIN), "optimizer": {"name": "muon", "args": {"lr": 0.02, "adamw_lr": 3e-3}},
                "scheduler": {"name": "wsd", "args": {"decay_ratio": 0.2}}})


_ALL = [
    sprachmodell_mini, sprachmodell_moe, sprachmodell_mamba, sprachmodell_hybrid, sprachmodell_speichersparend,
    text_klassifikation, bert_vortraining, uebersetzung_seq2seq,
    bildklassifikation_cnn, bildklassifikation_vit, bildklassifikation_mobil, vae_bilder,
    zeitreihen_lstm, zeitreihen_transformer, sequenz_gedaechtnis_rnn,
    tabellen_klassifikation, tabellen_regression,
    gan_2d, diffusion_2d, graph_klassifikation, anomalie_autoencoder, reward_modell,
    rl_cartpole_ppo, rl_cartpole_dqn, rl_gitterwelt_belohnungen, rl_pendel_kontinuierlich,
    ensemble_tabellen, multimodal_fusion, mixture_of_models, freies_netz, kontrastiv_vortraining,
    forschung_experimentell,
]
for _fn in _ALL:
    _d = _fn()
    PRESETS.register(_fn.__name__, _fn, tags=(_d["domain"],),
                     tooltip=T(_d["description"], f"Anwendungsgebiet: {DOMAINS[_d['domain']]}.",
                               f"Modell '{_d['model']['name']}', Daten "
                               f"'{_d['data'] if isinstance(_d['data'], str) else _d['data']['name']}'."))


def get_preset(name: str) -> Any:
    """Liefert die ``ProjectConfig`` eines Presets (jedes Mal eine frische Kopie)."""
    from ..project import ProjectConfig

    return ProjectConfig.from_dict(PRESETS.get(name).factory(None))


def presets_by_domain() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {d: [] for d in DOMAINS}
    for comp in PRESETS:
        for tag in comp.tags:
            out.setdefault(tag, []).append(comp.name)
    return {k: v for k, v in out.items() if v}


# ===========================================================================
# Speicherprofile
# ===========================================================================

MEMORY_PROFILES: dict[str, str] = {
    "standard": "keine Änderung (fp32, alles auf dem Gerät)",
    "sparsam": "bf16-Autocast + Gradient-Checkpointing + 4× Gradientenakkumulation (¼ Batch)",
    "extrem_sparsam": "wie sparsam + reines bf16 + Aktivierungs-Offload + Adafactor/Lion + 8× Akkumulation",
    "schnell": "TF32-Matmul, bf16-Autocast auf GPU, torch.compile auf GPU",
}


def apply_memory_profile(cfg: Any, name: str) -> Any:
    """Wendet ein Speicherprofil auf eine ``ProjectConfig`` an (siehe ``MEMORY_PROFILES``)."""
    if name not in MEMORY_PROFILES:
        raise KeyError(f"Unbekanntes Speicherprofil '{name}'. Verfügbar: {', '.join(MEMORY_PROFILES)}")
    t, m = cfg.training, cfg.memory
    if name in ("sparsam", "extrem_sparsam"):
        factor = 4 if name == "sparsam" else 8
        m.mixed_precision = "bf16"
        m.gradient_checkpointing = True
        t.gradient_accumulation_steps = t.gradient_accumulation_steps * factor
        t.batch_size = max(1, t.batch_size // factor)
    if name == "extrem_sparsam":
        m.param_dtype = "bfloat16"
        m.activation_offload = True
        from ..core import registry

        lr = getattr(t.optimizer.args, "lr", 1e-3)
        if "adafactor" in registry("optimizer"):
            t.optimizer = {"name": "adafactor", "args": {"lr": lr}}
        else:
            t.optimizer = {"name": "lion", "args": {"lr": lr / 3}}
    if name == "schnell":
        m.matmul_precision = "high"
        if torch.cuda.is_available():
            m.mixed_precision = "bf16"
            m.compile = True
    return cfg


__all__ = ["PRESETS", "DOMAINS", "MEMORY_PROFILES", "get_preset", "presets_by_domain", "apply_memory_profile"]
