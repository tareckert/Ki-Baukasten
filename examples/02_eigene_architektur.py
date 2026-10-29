"""
Beispiel 02 – Eine eigene Transformer-Architektur bis ins Detail
================================================================

Baut ein Sprachmodell, bei dem fast jede Schicht anders ist:

* Grundbauplan: Attention mit Grouped-Query-Attention, QK-Norm, SwiGLU, RMSNorm
* Schicht 0: weniger Köpfe
* jede zweite Schicht: Mixture-of-Experts statt normalem FFN
* Schichten 1 und 3: lokales Sliding-Window
* letzte Schicht: Mamba-artiges SSM statt Attention (Hybrid!)

Ausführen:  python examples/02_eigene_architektur.py
"""

from ki_baukasten import Project

projekt = Project.from_preset("sprachmodell_mini")
projekt.update({
    "model.d_model": 128,
    "model.n_layers": 6,
    "model.block": {
        "mixer": {"name": "attention", "args": {"num_heads": 8, "num_kv_heads": 2, "qk_norm": "rmsnorm"}},
        "ffn": {"name": "glu", "args": {"activation": "silu", "multiple_of": 32}},
        "norm": "rmsnorm",
        "norm_position": "pre",
        "drop_path": 0.05,
    },
    "model.layer_overrides": {
        "0": {"mixer.num_heads": 4, "mixer.num_kv_heads": 1},             # Multi-Query in Schicht 0
        "1::2": {"ffn": {"name": "moe", "args": {"num_experts": 4, "top_k": 2}}},
        "1,3": {"mixer.window_size": 16},                                  # lokale Attention
        "-1": {"mixer": "ssm", "ffn": "none"},                             # Mamba-Schicht am Ende
    },
    "training.max_steps": 40,
})

modell = projekt.build()
print(projekt.summary(max_depth=3))
for i, block in enumerate(modell.blocks):
    print(f"Schicht {i}: Mixer={type(block.mixer).__name__:<16} FFN={type(block.ffn).__name__}")

projekt.train()
print(projekt.generate("Die ", max_new_tokens=60, temperature=0.8, top_k=10))
