"""
Beispiel 06 – Speicher planen und optimieren
============================================

* Speicherschätzung mit gemessenen Aktivierungen und Empfehlungen
* Speicherprofile ('sparsam', 'extrem_sparsam', 'schnell')
* Layer-Swapping für Inferenz großer Modelle (Blöcke liegen auf der Festplatte)

Ausführen:  python examples/06_speicher.py
"""

import torch

from ki_baukasten import Project
from ki_baukasten.memory import MemoryConfig, MemoryManager

projekt = Project.from_preset("sprachmodell_speichersparend")
projekt.set("memory.memory_budget_gb", 0.25)      # angenommenes Budget: 256 MB
print(projekt.memory_report(batch_size=32))

print("\n--- mit Profil 'extrem_sparsam' ---")
projekt.apply_memory_profile("extrem_sparsam")
print(projekt.memory_report(batch_size=4))

# Layer-Swapping (Inferenz): Blöcke werden erst bei Bedarf von der Festplatte geladen.
modell = Project.from_preset("sprachmodell_mini").build().eval()
x = torch.randint(1, 40, (1, 16))
referenz = modell(x)
mgr = MemoryManager(MemoryConfig(offload={"mode": "disk", "directory": "ausgaben/offload_cache"}))
modell = mgr.prepare_model(modell, training=False)
print("\nParameter eines ausgelagerten Blocks im RAM:", sum(p.numel() for p in modell.blocks[0].parameters()))
with torch.no_grad():
    print("Abweichung mit Swapping:", (modell(x) - referenz).abs().max().item())
