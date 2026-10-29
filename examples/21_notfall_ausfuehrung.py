"""
Beispiel 21 – Notfall-Ausführung: wenn ein Modell trotz Verkleinern nicht passt
==============================================================================

Statt aufzugeben wird stückweise gerechnet: Schicht für Schicht von der SSD, Blöcke im RAM, Teile
auf der CPU, nur wenige Blöcke gleichzeitig trainieren (LISA), eingefrorene Basis auf der SSD
mit LoRA-Adaptern im RAM, Batches automatisch teilen, lange Prompts in Stücken …

GUI: "Reduzieren / Export" → "Notfall-Ausführung…".  CLI:  python -m ki_baukasten notfall …

Ausführen:  python examples/21_notfall_ausfuehrung.py
"""

import tempfile
import warnings

import torch

from ki_baukasten import Project
from ki_baukasten.analysis.fallback import options_text

warnings.simplefilter("ignore")

p = Project.from_preset("sprachmodell_mini", **{"training.max_steps": 15, "training.verbose": False,
                                                 "deployment.show_after_training": False})

# 1) Empfehlung für ein winziges Gerät: Verkleinern reicht nicht → Notfall-Optionen werden mitgeliefert
rec = p.recommend_for("mikrocontroller")
print(rec)

# 2) Alle Optionen für ein festes Budget – Inferenz und Training
print("\n" + options_text(p.fallback_options(budget_mb=3, training=False), 3, limit=4))
print("\n" + options_text(p.fallback_options(budget_mb=8, training=True), 8, limit=4))

# 3) Einschalten: eingefrorene Basis liegt auf der SSD, nur LoRA-Adapter im RAM – und trotzdem trainieren
with tempfile.TemporaryDirectory() as ordner:
    p.update({"memory.offload.directory": ordner})
    gewaehlt = p.apply_fallback("stream_disk_lora", budget_mb=8, training=True)
    print("\nEingeschaltet:", gewaehlt.title)
    hist = p.train()
    print("Training mit Basis auf der SSD ok, Verlust:", round(next(h["loss"] for h in reversed(hist) if "loss" in h), 3))
    p.undo()

# 4) Inferenz stückweise + lange Prompts in Stücken (Ergebnis identisch)
p.build()
ids = torch.randint(0, p.model.config.vocab_size, (8, 64))
voll, stueck = p.predict(ids), p.predict(ids, chunk_size=2)
print("predict(chunk_size=2) identisch:", torch.allclose(voll, stueck, atol=1e-5))
a = p.model.generate(ids[:1], max_new_tokens=8, temperature=0)
b = p.model.generate(ids[:1], max_new_tokens=8, temperature=0, prefill_chunk=16)
print("Chunked Prefill identisch:", torch.equal(a, b))
print("Bei Speichermangel im Training wird der Batch automatisch geteilt: memory.oom_strategy =",
      p.get("memory.oom_strategy"))
