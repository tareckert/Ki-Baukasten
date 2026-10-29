"""
Beispiel 10 – Schnelles Mamba-Training und Generierung mit Cache
================================================================

* Der Zustands-Scan des SSM läuft als eigene Autograd-Funktion (fusioniert, Neuberechnung im Backward).
  Hier wird er gegen die alte, aufgezeichnete Schleife gemessen – Zeit und gespeicherte Aktivierungen.
* Generierung mit KV-/Zustands-Cache: identisches Ergebnis wie ohne Cache, aber nur ein Schritt pro Token.
* Cache-Speicher: float32 vs. int8, begrenztes Fenster mit Attention-Sinks.

Ausführen:  python examples/10_mamba_schnell_und_cache.py
"""

import time

import torch

from ki_baukasten import build_model

torch.manual_seed(0)

# ---------------------------------------------------------------- 1. Scan-Varianten im Vergleich
modell = build_model({"name": "ssm", "vocab_size": 64, "d_model": 128, "n_layers": 4})
x = torch.randint(0, 64, (16, 128))


def messen(mode: str, backward: str) -> tuple[float, float]:
    for block in modell.blocks:                       # dieselbe Einstellung für jede Schicht
        block.mixer.cfg.scan_mode, block.mixer.cfg.scan_backward = mode, backward
    gespeichert = [0]

    def pack(t):                                      # zählt, was Autograd für den Backward aufhebt
        gespeichert[0] += t.numel() * t.element_size()
        return t

    start = time.time()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        verlust = modell(x).square().mean()
    verlust.backward()
    modell.zero_grad()
    return time.time() - start, gespeichert[0] / 2**20


for mode, backward in [("sequential", "autograd"), ("auto", "scan"), ("auto", "fused")]:
    sek, mib = messen(mode, backward)
    print(f"scan_mode={mode:<10} scan_backward={backward:<8}  {sek:6.2f} s   {mib:8.1f} MiB Aktivierungen")

# ---------------------------------------------------------------- 2. Generierung mit Cache
for name, spec in {
    "Mamba (SSM)": {"name": "ssm", "vocab_size": 64, "d_model": 64, "n_layers": 2},
    "Hybrid (SSM + Attention)": {"name": "hybrid", "vocab_size": 64, "d_model": 64, "n_layers": 4},
    "GPT mit GQA": {"name": "transformer", "vocab_size": 64, "d_model": 64, "n_layers": 4, "max_seq_len": 512,
                    "block": {"mixer": {"name": "attention", "num_heads": 8, "num_kv_heads": 2}}},
}.items():
    m = build_model(spec).eval()
    prompt = torch.randint(0, 64, (4, 8))
    t0 = time.time()
    ohne = m.generate(prompt, max_new_tokens=120, temperature=0, use_cache=False)
    t1 = time.time()
    mit = m.generate(prompt, max_new_tokens=120, temperature=0)          # use_cache=True ist Standard
    t2 = time.time()
    print(f"{name:<26} ohne Cache {t1 - t0:5.2f} s · mit Cache {t2 - t1:5.2f} s · identisch: {torch.equal(ohne, mit)}")

# ---------------------------------------------------------------- 3. Cache-Speicher
gpt = build_model({"name": "transformer", "vocab_size": 64, "d_model": 128, "n_layers": 4, "max_seq_len": 1024}).eval()
text = torch.randint(0, 64, (8, 512))
for einstellung in [{}, {"dtype": "float16"}, {"dtype": "int8"}, {"max_length": 128, "sink_tokens": 4}]:
    cache = gpt.new_cache(einstellung)
    with torch.no_grad():
        gpt(text, cache=cache)
    print(f"Cache {str(einstellung or 'float32'):<40} {cache.nbytes() / 2**20:7.2f} MiB")
