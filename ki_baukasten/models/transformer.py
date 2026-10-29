"""
ki_baukasten.models.transformer
===============================

Der Transformer in allen Ausprägungen – plus Mamba-artige und hybride Modelle.

Architekturen (``architecture``)
--------------------------------
* ``decoder_only``    – GPT/LLaMA-artig: kausal, generiert Text Token für Token.
* ``encoder_only``    – BERT/ViT-artig: bidirektional, für Klassifikation,
  Tagging, Embeddings.
* ``encoder_decoder`` – T5/Original-Transformer: Encoder liest die Eingabe,
  Decoder erzeugt die Ausgabe mit Cross-Attention (Übersetzung,
  Zusammenfassung, Sequenz-zu-Sequenz-Prognosen).

Eingabearten (``input_type``)
-----------------------------
* ``tokens``     – Ganzzahl-IDs (Text, diskrete Symbole) → Embedding-Tabelle
* ``continuous`` – Vektoren pro Zeitschritt (Zeitreihen, Sensoren) → lineare Projektion
* ``patches``    – Bilder ``[B, C, H, W]`` → Patch-Embedding (Vision Transformer)

Ein Block (``TransformerBlockConfig``)
--------------------------------------
::

    x ─┬─► Norm ─► Mixer (Attention/SSM/…) ─► Dropout ─► LayerScale ─► DropPath ─┐
       └──────────────────────────────────── skip ─────────────────────────────── + ─► …
                                    (optional Cross-Attention-Zweig)
    … ─┬─► Norm ─► FFN (MLP/GLU/MoE) ─► … ─┐
       └────────────── skip ─────────────── + ─► Ausgabe

Alles ist einstellbar: Mixer und FFN (Registries), Normalisierung und ihre
Position (pre/post/sandwich/none), parallele Zweige (GPT-J), Residual-
Skalierungen (DeepNet), LayerScale (CaiT), Stochastic Depth, Dropout.
Über ``layer_overrides`` kann **jede einzelne Schicht** abweichen – z. B.
andere Kopfzahl, Sliding Window nur in manchen Schichten, MoE nur jede zweite
Schicht oder SSM statt Attention (Hybrid).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..components.attention import AttentionConfig, MultiHeadAttention
from ..components.layers import LAYERS, DropPath
from ..components.normalization import build_norm
from ..core import (BaseConfig, ComponentParam, Level, NestedParam, Param, T, resolve_layer_config,
                    validate_overrides)
from .base import MODELS, BaseModel, InitConfig, ModelConfig, mark_residual_projections


# ===========================================================================
# Block
# ===========================================================================

class TransformerBlockConfig(BaseConfig):
    __title__ = "Transformer-Block"
    __tooltip__ = T(
        "Bauplan eines Transformer-Blocks (Mixer + FFN mit Residualverbindungen).",
        "Gilt für alle Schichten; einzelne Schichten weichen über layer_overrides ab.",
        "Jede Transformer-Variante.",
    )

    mixer = ComponentParam("sequence_mixer", "attention", tooltip=T(
        "Mixer: tauscht Information zwischen Positionen aus (Attention, SSM, RNN, Faltung ...).",
        "Bestimmt Kosten und Reichweite; pro Schicht wählbar → Hybride.",
        "Standard: attention. Lange Sequenzen: ssm. Hybride über layer_overrides.",
        "Bei 'attention' liegen die Attention-Einstellungen direkt unter mixer.* (z. B. mixer.num_heads).",
    ))
    cross_attention = NestedParam(AttentionConfig, level=Level.ADVANCED, tooltip=T(
        "Cross-Attention der Decoder-Blöcke (nur encoder_decoder).",
        "Lässt den Decoder auf die Encoder-Ausgabe schauen.",
        "Übersetzung, Zusammenfassung, Seq2Seq.",
    ))
    ffn = ComponentParam("feedforward", "glu", tooltip=T(
        "Feed-Forward-Teil (mlp, glu/SwiGLU, moe, none).",
        "Positionsweise Verarbeitung – hier steckt die meiste Kapazität.",
        "glu (SwiGLU) ist moderner Standard; moe für sehr große Kapazität.",
    ))
    norm = ComponentParam("normalization", "rmsnorm", tooltip=T(
        "Normalisierung in den Residual-Zweigen.", "Stabilisiert das Training.", "rmsnorm (modern), layernorm (klassisch)."))
    norm_position = Param("pre", choices={
        "pre": "Norm vor dem Zweig (Pre-LN): stabil, Standard moderner Modelle",
        "post": "Norm nach der Addition (Post-LN, Original-Transformer): oft besser, aber instabil ohne Warmup",
        "sandwich": "Norm vor UND nach dem Zweig (Gemma 2, CogView): sehr stabil",
        "none": "keine Normalisierung (nur mit vorsichtiger Init/Skalierung)",
    }, tooltip=T(
        "Position der Normalisierung relativ zur Residualverbindung.",
        "Großer Einfluss auf Trainingsstabilität und Endqualität.",
        "pre für tiefe Modelle; post für kleine Modelle mit Warmup.",
    ), level=Level.ADVANCED)
    parallel = Param(False, tooltip=T(
        "Mixer und FFN parallel statt nacheinander (x + Mixer(n(x)) + FFN(n(x))).",
        "~15 % schneller bei großen Modellen, minimal andere Qualität.",
        "GPT-J, PaLM.",
    ), level=Level.ADVANCED)
    dropout = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Dropout auf die Ausgabe jedes Zweigs (Residual-Dropout).",
        "Regularisierung.", "Kleine Datensätze: 0.1; große LLMs: 0."), group="Regularisierung")
    drop_path = Param(0.0, min=0.0, max=1.0, tooltip=T(
        "Stochastic Depth: Wahrscheinlichkeit, einen ganzen Zweig auszulassen.",
        "Starke Regularisierung, besonders für Vision-Transformer.",
        "ViT/DeiT: 0.1–0.3 (mit linearem Anstieg über die Tiefe).",
    ), level=Level.ADVANCED, group="Regularisierung")
    layer_scale_init = Param(None, type=float, min=0.0, tooltip=T(
        "LayerScale: lernbarer Faktor pro Kanal auf jedem Zweig (Startwert; None = aus).",
        "Kleiner Start (1e-5…1e-1) lässt tiefe Netze zunächst wie flache trainieren.",
        "Sehr tiefe Transformer (CaiT, DINOv2).",
    ), level=Level.EXPERT)
    branch_scale = Param(1.0, tooltip=T(
        "Fester Faktor auf jeder Zweig-Ausgabe.", "Dämpft Residual-Beiträge in tiefen Netzen.",
        "Experimente; z. B. 1/√(2N)."), level=Level.EXPERT)
    skip_scale = Param(1.0, tooltip=T(
        "Fester Faktor auf dem Skip-Pfad (DeepNet: α > 1).",
        "Stärkerer Skip stabilisiert Post-LN-Netze mit hunderten Schichten.",
        "DeepNorm-Experimente.",
    ), level=Level.EXPERT)


class TransformerBlock(nn.Module):
    """Ein Block gemäß ``TransformerBlockConfig``."""

    def __init__(self, d_model: int, cfg: TransformerBlockConfig, cross: bool = False):
        super().__init__()
        self.cfg = cfg
        pos = cfg.norm_position

        def norm() -> nn.Module:
            return build_norm(cfg.norm, d_model) if pos != "none" else nn.Identity()

        def post_norm() -> nn.Module:
            return build_norm(cfg.norm, d_model) if pos == "sandwich" else nn.Identity()

        def scale() -> nn.Parameter | None:
            return nn.Parameter(torch.full((d_model,), cfg.layer_scale_init)) if cfg.layer_scale_init is not None else None

        self.mixer = cfg.mixer.build(d_model)
        self.ffn = cfg.ffn.build(d_model)
        self.norm1, self.norm1_post, self.ls1 = norm(), post_norm(), scale()
        self.norm2, self.norm2_post, self.ls2 = norm(), post_norm(), scale()
        self.cross = MultiHeadAttention(d_model, cfg.cross_attention) if cross else None
        if cross:
            self.norm_c, self.norm_c_post, self.ls_c = norm(), post_norm(), scale()
        self.drop = nn.Dropout(cfg.dropout)
        self.drop_path = DropPath(cfg.drop_path)
        mark_residual_projections(self)

    def _finish_branch(self, h: torch.Tensor, ls: nn.Parameter | None) -> torch.Tensor:
        h = self.drop(h)
        if ls is not None:
            h = h * ls
        return self.drop_path(h) * self.cfg.branch_scale

    def _branch(self, x, fn, pre, post, ls):
        pos = self.cfg.norm_position
        h = fn(pre(x) if pos in ("pre", "sandwich") else x)
        h = self._finish_branch(post(h), ls)
        out = x * self.cfg.skip_scale + h
        return pre(out) if pos == "post" else out

    def forward(self, x, key_padding_mask=None, positional=None, positions=None, causal=False,
                memory=None, memory_mask=None):
        def mix(h):
            return self.mixer(h, key_padding_mask=key_padding_mask, positional=positional, positions=positions,
                              causal=causal)

        if self.cfg.parallel:
            pos = self.cfg.norm_position
            n = self.norm1(x) if pos in ("pre", "sandwich") else x
            out = x * self.cfg.skip_scale + self._finish_branch(self.norm1_post(mix(n)), self.ls1) \
                + self._finish_branch(self.norm2_post(self.ffn(n)), self.ls2)
            if self.cross is not None and memory is not None:
                out = self._branch(out, lambda h: self.cross(h, kv=memory, key_padding_mask=memory_mask),
                                   self.norm_c, self.norm_c_post, self.ls_c)
            return self.norm1(out) if pos == "post" else out

        x = self._branch(x, mix, self.norm1, self.norm1_post, self.ls1)
        if self.cross is not None and memory is not None:
            x = self._branch(x, lambda h: self.cross(h, kv=memory, key_padding_mask=memory_mask),
                             self.norm_c, self.norm_c_post, self.ls_c)
        return self._branch(x, self.ffn, self.norm2, self.norm2_post, self.ls2)


# ===========================================================================
# Generierung
# ===========================================================================

class GenerationConfig(BaseConfig):
    __title__ = "Generierung"
    __tooltip__ = T(
        "Einstellungen für das Erzeugen von Text/Token-Folgen.",
        "Steuern Kreativität vs. Verlässlichkeit der Ausgabe.",
        "Sprachmodelle (decoder_only mit tokens).",
    )
    max_new_tokens = Param(50, min=1, tooltip=T(
        "Maximale Anzahl neu erzeugter Tokens.", "Begrenzt Länge und Laufzeit.", "Je nach Anwendung."))
    temperature = Param(1.0, min=0.0, tooltip=T(
        "Sampling-Temperatur (0 = immer das wahrscheinlichste Token).",
        "Höher = kreativer/zufälliger, niedriger = fokussierter.",
        "Fakten: 0–0.3; kreatives Schreiben: 0.7–1.2.",
    ))
    top_k = Param(None, type=int, min=1, tooltip=T(
        "Nur aus den k wahrscheinlichsten Tokens ziehen.", "Schneidet unwahrscheinlichen Unsinn ab.", "Typisch 40–100."))
    top_p = Param(None, type=float, min=0.0, max=1.0, tooltip=T(
        "Nucleus-Sampling: kleinste Menge mit kumulierter Wahrscheinlichkeit ≥ p.",
        "Passt die Auswahlgröße dynamisch an die Unsicherheit an.", "Typisch 0.9–0.95."))
    min_p = Param(None, type=float, min=0.0, max=1.0, tooltip=T(
        "Nur Tokens mit p ≥ min_p · p_max zulassen.", "Robust bei hohen Temperaturen.", "0.05–0.1."),
        level=Level.ADVANCED)
    repetition_penalty = Param(1.0, min=0.0, tooltip=T(
        "Strafe für bereits vorgekommene Tokens (> 1 = weniger Wiederholungen).",
        "Verhindert Endlosschleifen.", "1.1–1.3 bei Wiederholungsproblemen."), level=Level.ADVANCED)
    eos_token_id = Param(None, type=int, tooltip=T(
        "Token, das die Generierung beendet.", "Stoppt, sobald alle Sequenzen es erzeugt haben.", "Wenn vorhanden."))


def _filter_logits(logits: torch.Tensor, g: GenerationConfig) -> torch.Tensor:
    if g.top_k is not None:
        kth = torch.topk(logits, min(g.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if g.min_p is not None:
        probs = logits.softmax(-1)
        logits = logits.masked_fill(probs < g.min_p * probs.amax(-1, keepdim=True), float("-inf"))
    if g.top_p is not None:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > g.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    return logits


# ===========================================================================
# Transformer-Modell
# ===========================================================================

class TransformerConfig(ModelConfig):
    __title__ = "Transformer"
    __tooltip__ = T(
        "Transformer-Modell (Decoder, Encoder oder Encoder-Decoder).",
        "Die vielseitigste Architektur der modernen KI – Sprache, Bilder, Zeitreihen, Audio, Multimodal.",
        "Sprachmodelle, Klassifikation, Übersetzung, Vision Transformer, Prognosen.",
    )

    init = NestedParam(InitConfig, level=Level.ADVANCED, group="Initialisierung", default_factory=lambda: InitConfig(
        embedding={"name": "normal", "std": 0.02}, linear_bias="zeros"), tooltip=T(
        "Initialisierung (Transformer-Standard: Embeddings ~ N(0, 0.02²), Biases 0).",
        "Kleine Embeddings halten die Logits gebundener Ausgabegewichte anfangs klein → stabiler Start.",
        "GPT-2-Schema zusätzlich: linear_weight normal(0.02) + residual_projection scaled_normal.",
    ))

    # --- Grundarchitektur --------------------------------------------------
    architecture = Param("decoder_only", choices={
        "decoder_only": "kausal, generativ (GPT, LLaMA)",
        "encoder_only": "bidirektional, verstehend (BERT, ViT)",
        "encoder_decoder": "Eingabe lesen, Ausgabe erzeugen (T5, Übersetzung)",
    }, tooltip=T("Grundarchitektur.", "Bestimmt Maskierung, Datenfluss und Einsatzzweck.", "Siehe Optionen."),
        group="Architektur")
    input_type = Param("tokens", choices={
        "tokens": "Ganzzahl-IDs (Text) → Embedding",
        "continuous": "Vektoren pro Schritt → lineare Projektion",
        "patches": "Bilder → Patch-Embedding (ViT)",
    }, tooltip=T("Art der Eingabedaten.", "Bestimmt die Eingangsschicht.", "Text: tokens; Zeitreihen: continuous."),
        group="Architektur")
    vocab_size = Param(1000, min=1, tooltip=T(
        "Vokabulargröße (nur tokens).", "Anzahl unterscheidbarer Tokens.", "Zeichen-Modelle: ~100; BPE: 32k–256k."),
        group="Architektur")
    input_dim = Param(8, min=1, tooltip=T(
        "Features pro Zeitschritt (nur continuous).", "Breite der Eingabevektoren.", "Anzahl Sensoren/Variablen."),
        group="Architektur")
    image_channels = Param(3, min=1, tooltip=T("Bildkanäle (nur patches).", "RGB = 3.", "ViT."), group="Architektur")
    image_size = Param([32, 32], item_type=int, min=1, length=2, tooltip=T(
        "Bildgröße [H, W] (nur patches).", "Bestimmt Anzahl der Patches.", "ViT."), group="Architektur")
    patch_size = Param(4, min=1, tooltip=T(
        "Kantenlänge eines Patches (nur patches).",
        "Kleinere Patches = mehr Tokens = feiner, aber quadratisch teurer.",
        "ViT-B/16: 16; kleine Bilder: 4.",
    ), group="Architektur")
    output_dim = Param(None, type=int, min=1, tooltip=T(
        "Ausgabegröße. None = vocab_size (Sprachmodell) bzw. Pflicht bei anderen Aufgaben.",
        "Klassen bei Klassifikation, Zielgrößen bei Regression.",
        "Klassifikation mit pooling ≠ none: Klassenzahl.",
    ), group="Architektur")
    d_model = Param(128, min=1, tooltip=T(
        "Modellbreite (Größe der Repräsentation jedes Tokens).",
        "Hauptregler der Kapazität (Parameter wachsen ~quadratisch).",
        "Klein 128–512, mittel 768–2048, groß 4096+.",
    ), group="Größe")
    n_layers = Param(4, min=1, tooltip=T(
        "Anzahl Blöcke (beim encoder_decoder: Decoder-Blöcke).",
        "Mehr Tiefe = abstraktere Verarbeitung.",
        "Klein 2–6, mittel 12–24, groß 32–80+.",
    ), group="Größe")
    n_encoder_layers = Param(None, type=int, min=1, tooltip=T(
        "Encoder-Blöcke (nur encoder_decoder; None = n_layers).", "Getrennte Tiefe für Lesen/Schreiben.", "T5: gleich."),
        group="Größe")
    max_seq_len = Param(256, min=1, tooltip=T(
        "Maximale Sequenzlänge.", "Größe von Positionstabellen; längere Eingaben nicht möglich (außer RoPE/ALiBi).",
        "Kontextlänge der Anwendung.",
    ), group="Größe")

    # --- Blöcke ------------------------------------------------------------
    block = NestedParam(TransformerBlockConfig, group="Blöcke")
    encoder_block = NestedParam(TransformerBlockConfig, default_none=True, level=Level.ADVANCED, group="Blöcke",
                                tooltip=T("Abweichender Block-Bauplan für den Encoder (None = wie block).",
                                          "Encoder und Decoder können völlig verschieden sein.", "encoder_decoder."))
    layer_overrides = Param({}, level=Level.ADVANCED, group="Blöcke", tooltip=T(
        "Schichtweise Abweichungen des (Decoder-)Stapels: {selektor: {einstellung: wert}}.",
        "Jede einzelne Schicht kann anders gebaut sein.",
        "Hybride, lokale/globale Attention im Wechsel, MoE nur in manchen Schichten.",
        "Beispiel: {'1::2': {'ffn': 'moe'}, '-1': {'mixer.num_heads': 16}}",
    ))
    encoder_layer_overrides = Param({}, level=Level.ADVANCED, group="Blöcke", tooltip=T(
        "Schichtweise Abweichungen des Encoders.", "Wie layer_overrides.", "encoder_decoder."))
    drop_path_schedule = Param("linear", choices={"constant": "überall gleich", "linear": "von 0 linear ansteigend"},
                               level=Level.EXPERT, group="Blöcke", tooltip=T(
        "Verteilung von drop_path über die Tiefe.", "Frühe Schichten seltener auslassen.", "Bei drop_path > 0."))

    # --- Positionen & Embeddings ------------------------------------------
    positional = ComponentParam("positional", "rope", group="Positionen", tooltip=T(
        "Positionskodierung.", "Gibt dem Modell Reihenfolge-Information.", "rope (Text), learned (ViT), sinusoidal."))
    embedding_dropout = Param(0.0, min=0.0, max=1.0, group="Regularisierung", tooltip=T(
        "Dropout nach dem Embedding.", "Regularisiert Eingaberepräsentationen.", "Kleine Datensätze."))
    embedding_scale = Param(None, type=bool, level=Level.EXPERT, group="Positionen", tooltip=T(
        "Embeddings mit √d_model multiplizieren (None = automatisch: an bei 'sinusoidal').",
        "Balanciert Token- gegen Positionssignal: ohne Skalierung gehen kleine Token-Embeddings im festen "
        "Sinus-Signal (Amplitude 1) unter (Original-Transformer skaliert deshalb).",
        "Sinusoidale Positionen, Gemma.",
    ))
    embedding_norm = ComponentParam("normalization", None, level=Level.EXPERT, group="Positionen", tooltip=T(
        "Normalisierung direkt nach dem Embedding.", "Stabilisiert sehr große Vokabulare (BLOOM).", "Selten."))
    tie_embeddings = Param(True, level=Level.ADVANCED, group="Ausgabe", tooltip=T(
        "Ausgabe-Gewichte = Embedding-Gewichte (Weight Tying).",
        "Spart vocab·d Parameter und verbessert oft kleine Sprachmodelle.",
        "Sprachmodelle (nur wenn output_dim = vocab_size und kein Kopf-MLP).",
    ))
    pad_token_id = Param(None, type=int, level=Level.ADVANCED, group="Architektur", tooltip=T(
        "Padding-Token: daraus wird automatisch die Padding-Maske erzeugt.",
        "Padding wird von Attention ignoriert.", "Batches mit unterschiedlich langen Texten."))
    num_register_tokens = Param(0, min=0, level=Level.EXPERIMENTAL, group="Architektur", tooltip=T(
        "Zusätzliche lernbare 'Register'-Tokens am Sequenzanfang.",
        "Geben dem Modell Notizplatz; reduziert Artefakte in Attention-Karten (ViT-Register).",
        "Vision Transformer, Forschung.",
    ))

    # --- Ausgabe -----------------------------------------------------------
    final_norm = Param(True, level=Level.ADVANCED, group="Ausgabe", tooltip=T(
        "Normalisierung nach dem letzten Block.", "Bei Pre-Norm nötig, damit die Ausgabe skaliert ist.", "Pre-Norm: an."))
    pooling = Param("none", choices={
        "none": "eine Ausgabe pro Position [B, T, out] (Sprachmodell, Tagging)",
        "cls": "lernbares CLS-Token am Anfang (BERT/ViT-Klassifikation)",
        "mean": "Mittelwert über gültige Positionen",
        "max": "Maximum über Positionen",
        "last": "letzte gültige Position (für kausale Modelle)",
    }, group="Ausgabe", tooltip=T("Verdichtung zu einem Vektor pro Sequenz.", "Für sequenzweise Vorhersagen.",
                                  "Klassifikation: cls/mean; Decoder: last."))
    head_hidden_dims = Param([], item_type=int, min=1, level=Level.ADVANCED, group="Ausgabe", tooltip=T(
        "Versteckte Schichten des Ausgabekopfs.", "Mehr Kapazität im Kopf.", "Meist leer."))
    head_activation = ComponentParam("activation", "gelu", level=Level.ADVANCED, group="Ausgabe", tooltip=T(
        "Aktivierung im Kopf-MLP.", "Nichtlinearität.", "Nur mit head_hidden_dims."))
    head_bias = Param(False, level=Level.EXPERT, group="Ausgabe", tooltip=T(
        "Bias der Ausgabeschicht.", "Sprachmodelle meist ohne.", "Klassifikation: an."))
    logit_softcap = Param(None, type=float, min=1e-6, level=Level.EXPERT, group="Ausgabe", tooltip=T(
        "Weiche Obergrenze der Ausgabe-Logits: cap·tanh(x/cap).",
        "Verhindert überkonfidente Vorhersagen und Instabilität.", "Gemma 2: 30."))

    def resolved_output_dim(self) -> int:
        if self.output_dim is not None:
            return self.output_dim
        if self.input_type == "tokens" and self.pooling == "none":
            return self.vocab_size
        if self.input_type == "continuous" and self.pooling == "none":
            return self.input_dim
        raise ValueError("output_dim muss gesetzt sein (z. B. Klassenzahl).")

    def num_patches(self) -> int:
        h, w = self.image_size
        return (h // self.patch_size) * (w // self.patch_size)

    def check(self) -> list[str]:
        problems = validate_overrides(self.block, self.layer_overrides, self.n_layers)
        if self.architecture == "encoder_decoder":
            enc = self.encoder_block or self.block
            problems += validate_overrides(enc, self.encoder_layer_overrides, self.n_encoder_layers or self.n_layers)
        if self.input_type == "patches":
            h, w = self.image_size
            if h % self.patch_size or w % self.patch_size:
                problems.append("image_size muss durch patch_size teilbar sein.")
            if self.architecture == "encoder_decoder":
                problems.append("patches wird nur für decoder_only/encoder_only unterstützt.")
        if self.pooling == "cls" and self.architecture == "decoder_only":
            problems.append("pooling='cls' ist bei kausalen Decodern wirkungslos (CLS sieht nur sich) – nutze 'last'.")
        if self.architecture == "encoder_decoder" and self.pooling != "none":
            problems.append("encoder_decoder unterstützt nur pooling='none'.")
        return problems


class Transformer(BaseModel):
    """Transformer gemäß ``TransformerConfig``.

    Aufruf:
        * decoder_only / encoder_only: ``model(inputs, attention_mask=None)``
        * encoder_decoder: ``model(inputs, decoder_inputs=..., attention_mask=None, decoder_attention_mask=None)``

    ``attention_mask``: ``[B, T]`` mit ``True``/1 = gültig, ``False``/0 = Padding.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        config.validate()
        c = config
        d = c.d_model
        self.out_dim = c.resolved_output_dim()
        self.n_prefix = c.num_register_tokens + (1 if c.pooling == "cls" else 0)
        seq_len = (c.num_patches() if c.input_type == "patches" else c.max_seq_len) + self.n_prefix

        # --- Eingangsschicht ---
        if c.input_type == "tokens":
            self.embed = nn.Embedding(c.vocab_size, d, padding_idx=c.pad_token_id)
        elif c.input_type == "continuous":
            self.embed = nn.Linear(c.input_dim, d)
        else:
            self.embed = nn.Conv2d(c.image_channels, d, c.patch_size, stride=c.patch_size)
        self.prefix = nn.Parameter(torch.randn(1, self.n_prefix, d) * 0.02) if self.n_prefix else None
        self.embed_norm = build_norm(c.embedding_norm, d) if c.embedding_norm is not None else nn.Identity()
        self.embed_drop = nn.Dropout(c.embedding_dropout)

        # --- Schicht-Konfigurationen auflösen (Basis + Overrides) ---
        is_ed = c.architecture == "encoder_decoder"
        dec_cfgs = self._resolve(c.block, c.layer_overrides, c.n_layers)
        enc_cfgs = self._resolve(c.encoder_block or c.block, c.encoder_layer_overrides,
                                 c.n_encoder_layers or c.n_layers) if is_ed else []

        # --- Positionen (ein Modul für alle Schichten; Positionsinfo wird geteilt) ---
        heads = max([getattr(b.mixer.args, "num_heads", 1) for b in dec_cfgs + enc_cfgs] or [1])
        self.positional = c.positional.build(d, d // max(1, heads), heads, seq_len)

        # --- Blockstapel ---
        self.blocks = nn.ModuleList([TransformerBlock(d, b, cross=is_ed) for b in dec_cfgs])
        if is_ed:
            self.encoder_blocks = nn.ModuleList([TransformerBlock(d, b) for b in enc_cfgs])
            self.encoder_norm = build_norm(c.block.norm, d) if c.final_norm else nn.Identity()
            if c.input_type == "tokens":
                self.decoder_embed = self.embed  # gemeinsames Vokabular
            else:
                self.decoder_embed = nn.Linear(self.out_dim, d)
        self.final_norm = build_norm(c.block.norm, d) if c.final_norm else nn.Identity()

        # --- Kopf ---
        head: list[nn.Module] = []
        dims = [d] + list(c.head_hidden_dims)
        for i in range(len(c.head_hidden_dims)):
            head += [nn.Linear(dims[i], dims[i + 1]), c.head_activation.build()]
        self.head_mlp = nn.Sequential(*head)
        self.lm_head = nn.Linear(dims[-1], self.out_dim, bias=c.head_bias)
        if (c.tie_embeddings and c.input_type == "tokens" and self.out_dim == c.vocab_size
                and not c.head_hidden_dims):
            self.lm_head.weight = self.embed.weight
        self.finish()

    # ------------------------------------------------------------ Aufbau-Hilfen
    def _resolve(self, base: TransformerBlockConfig, overrides: dict, n: int) -> list[TransformerBlockConfig]:
        """Konfiguration jeder einzelnen Schicht: Basis + passende Overrides (+ Drop-Path-Verlauf)."""
        cfgs = []
        for i in range(n):
            cfg = resolve_layer_config(base, overrides, i, n)
            if self.config.drop_path_schedule == "linear" and n > 1:
                cfg.drop_path = cfg.drop_path * i / (n - 1)
            cfgs.append(cfg)
        return cfgs

    def depth(self) -> int:
        return len(self.blocks) + len(getattr(self, "encoder_blocks", []))

    def checkpointable_blocks(self) -> list[nn.Module]:
        return list(getattr(self, "encoder_blocks", [])) + list(self.blocks)

    # ------------------------------------------------------------ Vorwärts
    def _embed(self, x: torch.Tensor, embed: nn.Module, mask: torch.Tensor | None, with_prefix: bool = True):
        c = self.config
        if embed is self.embed and c.input_type == "patches":
            h = embed(x).flatten(2).transpose(1, 2)
        else:
            h = embed(x)
        scale = c.embedding_scale if c.embedding_scale is not None else c.positional.name == "sinusoidal"
        if scale:
            h = h * (c.d_model ** 0.5)
        if self.prefix is not None and with_prefix:
            h = torch.cat([self.prefix.expand(h.shape[0], -1, -1).to(h.dtype), h], dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(mask.shape[0], self.n_prefix, dtype=mask.dtype, device=mask.device), mask], 1)
        positions = torch.arange(h.shape[1], device=h.device)
        h = self.positional.embed(h, positions)
        return self.embed_drop(self.embed_norm(h)), mask, positions

    def _auto_mask(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is not None:
            return mask.bool()
        if self.config.pad_token_id is not None and self.config.input_type == "tokens":
            return x != self.config.pad_token_id
        return None

    def encode(self, inputs: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, Any]:
        """Läuft durch Embedding + (Encoder-)Blöcke; liefert Repräsentation und Maske."""
        c = self.config
        mask = self._auto_mask(inputs, attention_mask)
        h, mask, positions = self._embed(inputs, self.embed, mask)
        if c.architecture == "encoder_decoder":
            for block in self.encoder_blocks:
                h = block(h, key_padding_mask=mask, positional=self.positional, positions=positions, causal=False)
            return self.encoder_norm(h), mask
        causal = c.architecture == "decoder_only"
        for block in self.blocks:
            h = block(h, key_padding_mask=mask, positional=self.positional, positions=positions, causal=causal)
        return self.final_norm(h), mask

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor | None = None,
                decoder_inputs: torch.Tensor | None = None,
                decoder_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        c = self.config
        if c.architecture == "encoder_decoder":
            if decoder_inputs is None:
                raise ValueError("encoder_decoder braucht decoder_inputs.")
            memory, mem_mask = self.encode(inputs, attention_mask)
            dmask = self._auto_mask(decoder_inputs, decoder_attention_mask)
            h, dmask, positions = self._embed(decoder_inputs, self.decoder_embed, dmask, with_prefix=False)
            for block in self.blocks:
                h = block(h, key_padding_mask=dmask, positional=self.positional, positions=positions, causal=True,
                          memory=memory, memory_mask=mem_mask)
            h = self.final_norm(h)
            return self._head(h)

        h, mask = self.encode(inputs, attention_mask)
        return self._head(self._pool(h, mask))

    def _pool(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        p = self.config.pooling
        if p == "none":
            return h[:, self.n_prefix:] if self.n_prefix else h
        if p == "cls":
            return h[:, self.n_prefix - 1]
        body = h[:, self.n_prefix:]
        m = mask[:, self.n_prefix:] if mask is not None else None
        if p == "mean":
            if m is None:
                return body.mean(1)
            w = m.to(body.dtype).unsqueeze(-1)
            return (body * w).sum(1) / w.sum(1).clamp_min(1.0)
        if p == "max":
            if m is not None:
                body = body.masked_fill(~m.unsqueeze(-1), torch.finfo(body.dtype).min)
            return body.amax(1)
        # last: letzte gültige Position
        if m is None:
            return body[:, -1]
        idx = (m.long().cumsum(1).argmax(1)).clamp_min(0)
        return body[torch.arange(body.shape[0], device=body.device), idx]

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(self.head_mlp(h))
        cap = self.config.logit_softcap
        return cap * torch.tanh(logits / cap) if cap is not None else logits

    # ------------------------------------------------------------ Generierung
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, generation: GenerationConfig | None = None, **overrides: Any) -> torch.Tensor:
        """Autoregressive Generierung (decoder_only + tokens)."""
        c = self.config
        if c.architecture != "decoder_only" or c.input_type != "tokens" or c.pooling != "none":
            raise ValueError("generate() erfordert decoder_only, input_type='tokens' und pooling='none'.")
        g = (generation or GenerationConfig()).replace(**overrides)
        was_training = self.training
        self.eval()
        ids = input_ids
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        for _ in range(g.max_new_tokens):
            logits = self(ids[:, -c.max_seq_len:])[:, -1, :].float()
            if g.repetition_penalty != 1.0:
                prev = torch.gather(logits, 1, ids)
                prev = torch.where(prev > 0, prev / g.repetition_penalty, prev * g.repetition_penalty)
                logits = logits.scatter(1, ids, prev)
            if g.temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = _filter_logits(logits / g.temperature, g).softmax(-1)
                nxt = torch.multinomial(probs, 1)
            if g.eos_token_id is not None:
                nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, g.eos_token_id), nxt)
                finished |= nxt.squeeze(1) == g.eos_token_id
            ids = torch.cat([ids, nxt], dim=1)
            if g.eos_token_id is not None and bool(finished.all()):
                break
        self.train(was_training)
        return ids

    def example_input(self, batch_size: int = 2) -> Any:
        c = self.config
        t = min(16, c.max_seq_len)
        if c.input_type == "tokens":
            x = torch.randint(0, c.vocab_size, (batch_size, t))
        elif c.input_type == "continuous":
            x = torch.randn(batch_size, t, c.input_dim)
        else:
            x = torch.randn(batch_size, c.image_channels, *c.image_size)
        if c.architecture == "encoder_decoder":
            dec = torch.randint(0, c.vocab_size, (batch_size, t)) if c.input_type == "tokens" \
                else torch.randn(batch_size, t, self.out_dim)
            return {"inputs": x, "decoder_inputs": dec}
        return x


MODELS.register("transformer", Transformer, config=TransformerConfig, aliases=("gpt", "bert", "vit"),
                tags=("nlp", "vision", "zeitreihen"), tooltip=T(
    "Transformer (Decoder / Encoder / Encoder-Decoder) für Text, Bilder und Zeitreihen.",
    "Extrem vielseitig; jede Schicht einzeln konfigurierbar, Mixer austauschbar.",
    "Sprachmodelle, Klassifikation, Übersetzung, ViT, Prognosen.",
))


# ===========================================================================
# Mamba-artige und hybride Modelle (spezielle Standardwerte)
# ===========================================================================

class SSMModelConfig(TransformerConfig):
    __title__ = "State-Space-Modell (Mamba-artig)"
    __tooltip__ = T(
        "Stapel selektiver Zustandsraum-Blöcke (Mamba-Prinzip) statt Attention.",
        "Lineare Kosten in der Sequenzlänge, konstanter Speicher pro Generierungsschritt.",
        "Sehr lange Sequenzen: Audio, Genomik, lange Dokumente, Sensordaten.",
    )
    block = NestedParam(TransformerBlockConfig, group="Blöcke",
                        default_factory=lambda: TransformerBlockConfig(mixer="ssm", ffn="none"))
    positional = ComponentParam("positional", "none", group="Positionen", tooltip=T(
        "Positionskodierung (SSMs brauchen keine – Reihenfolge steckt im Scan).",
        "Meist 'none'.", "SSM-Modelle."))


class HybridConfig(TransformerConfig):
    __title__ = "Hybrid (Attention + SSM)"
    __tooltip__ = T(
        "Hybrider Stapel: überwiegend SSM-Blöcke, dazwischen Attention-Blöcke (Jamba-Prinzip).",
        "Verbindet die Effizienz von SSMs mit der präzisen Abrufstärke von Attention.",
        "Lange Kontexte mit hohem Qualitätsanspruch.",
    )
    block = NestedParam(TransformerBlockConfig, group="Blöcke",
                        default_factory=lambda: TransformerBlockConfig(mixer="ssm"))
    layer_overrides = Param({"1::2": {"mixer": "attention"}}, level=Level.ADVANCED, group="Blöcke", tooltip=T(
        "Schichtweise Abweichungen; Standard: jede zweite Schicht (1, 3, 5 ...) nutzt Attention.",
        "Steuert das Mischungsverhältnis der Mixer.",
        "Jamba: 1 Attention auf 7 Mamba → {'7::8': {'mixer': 'attention'}}.",
    ))


MODELS.register("ssm", Transformer, config=SSMModelConfig, aliases=("mamba",), level=Level.ADVANCED,
                tags=("nlp", "zeitreihen"), tooltip=T(
    "Mamba-artiges State-Space-Modell.", "Linear skalierende Sequenzverarbeitung.", "Lange Sequenzen."))
MODELS.register("hybrid", Transformer, config=HybridConfig, aliases=("jamba",), level=Level.ADVANCED,
                tags=("nlp",), tooltip=T(
    "Hybrid aus SSM- und Attention-Schichten.", "Effizienz + Abrufpräzision.", "Lange Kontexte."))


# ===========================================================================
# Transformer-Block als Schicht für freie Graph-Modelle
# ===========================================================================

class BlockLayerConfig(TransformerBlockConfig):
    __title__ = "Transformer-Block (als Schicht)"
    causal = Param(False, tooltip=T("Kausale Maske.", "Nur Vergangenheit sichtbar.", "Generative Graphen."))
    positional = ComponentParam("positional", "none", tooltip=T(
        "Positionskodierung innerhalb des Blocks (rope/alibi wirken hier).", "Positionsinfo.", "Graph-Transformer."))


class _BlockLayer(nn.Module):
    def __init__(self, d_model: int, a: BlockLayerConfig, max_len: int):
        super().__init__()
        cfg = TransformerBlockConfig.from_dict({k: v for k, v in a.to_dict().items() if k not in ("causal", "positional")})
        self.block = TransformerBlock(d_model, cfg)
        heads = getattr(cfg.mixer.args, "num_heads", 1)
        self.positional = a.positional.build(d_model, d_model // max(1, heads), heads, max_len)
        self.causal = a.causal

    def forward(self, x):
        return self.block(x, positional=self.positional, causal=self.causal)


LAYERS.register("transformer_block", lambda a, in_shape: _BlockLayer(in_shape[-1], a, max(4096, in_shape[0])),
                config=BlockLayerConfig, tooltip=T(
    "Vollständiger Transformer-Block auf [B, T, D].", "Baustein für eigene Transformer-Hybride.", "Graph-Modelle."))
