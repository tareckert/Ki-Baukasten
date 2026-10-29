"""
Beispiel 12 – Modellblöcke trainierbar in den RAM auslagern
===========================================================

Mit ``memory.offload.mode = "stream"`` liegen Gewichte, Gradienten und Optimierer-Zustände der Blöcke
im CPU-RAM; auf dem Rechengerät ist immer nur der Block, der gerade rechnet. Das Training ist dabei
bitgenau dasselbe wie ohne Auslagerung – das zeigt dieses Beispiel.

Ausführen:  python examples/12_ram_streaming.py
"""

import torch

from ki_baukasten import MemoryConfig, Trainer, TrainingConfig, build_model
from ki_baukasten.memory import estimate_memory

spec = {"name": "transformer", "vocab_size": 50, "d_model": 128, "n_layers": 6,
        "block": {"dropout": 0.1, "mixer": {"name": "attention", "num_heads": 4}}}
daten = {"inputs": torch.randint(0, 50, (64, 32))}

ergebnisse = {}
for name, speicher in {"normal": MemoryConfig(),
                       "RAM-Streaming": MemoryConfig(offload={"mode": "stream"}),
                       "Streaming nur Blöcke 0–3": MemoryConfig(offload={"mode": "stream", "layers": "0:4"})}.items():
    torch.manual_seed(0)
    trainer = Trainer(build_model(spec), TrainingConfig(paradigm="causal_lm", epochs=1, verbose=False, seed=0), speicher)
    trainer.fit(daten)
    ergebnisse[name] = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
    streams = trainer.mem.streams
    print(f"{name:<26} Übertragungen pro Block: {[s.transfers for s in streams] or '-'}")

ref = ergebnisse["normal"]
for name, werte in ergebnisse.items():
    print(f"{name:<26} max. Abweichung der Gewichte nach dem Training: {(werte - ref).abs().max().item():.2e}")

# Speicherschätzung: was liegt auf dem Gerät, was im RAM?
modell = build_model({**spec, "d_model": 512, "n_layers": 12})
print()
print(estimate_memory(modell, batch_size=8, memory=MemoryConfig(offload={"mode": "stream"})))
