"""
Beispiel 11 – QLoRA-Feintuning: 4-Bit-Basis + LoRA-Adapter + 8-Bit-AdamW
=======================================================================

* Eigene Gewichts-Quantisierung (nf4) – ohne torch.ao, auf jedem Gerät
* Nur kleine LoRA-Adapter (und auf Wunsch Normen) werden trainiert
* 8-Bit-AdamW für die Optimierer-Zustände
* Speichern/Laden stellt die quantisierte Struktur automatisch wieder her; Export rechnet LoRA ein

Ausführen:  python examples/11_qlora_feintuning.py
"""

import warnings

import torch

from ki_baukasten import Project
from ki_baukasten.memory import lora_parameter_count, quantized_bytes

warnings.simplefilter("ignore")

projekt = Project.from_preset("sprachmodell_qlora")       # nf4 + LoRA(r=16) + adamw8bit sind schon eingestellt
projekt.set("training.max_steps", 60)
projekt.set("memory.lora.targets", "q_proj|k_proj|v_proj|o_proj|gate|up|down")   # welche Schichten Adapter bekommen
projekt.set("memory.lora.layers", "*")                    # in allen Blöcken ('-2:' = nur die letzten zwei)
projekt.train()

adapter, gesamt = lora_parameter_count(projekt.model)
q, voll = quantized_bytes(projekt.model)
print(f"\nTrainierbar: {adapter:,} von {gesamt:,} Parametern ({100 * adapter / gesamt:.1f} %)")
print(f"Quantisierte Gewichte: {q / 2**20:.2f} MiB statt {voll / 2**20:.2f} MiB in float32")
print(projekt.generate("Der ", max_new_tokens=60, temperature=0.7))

ziel = projekt.save("ausgaben/qlora_demo")
wieder = Project.load(ziel)                               # Struktur (nf4 + LoRA) wird automatisch hergestellt
x = torch.randint(0, 40, (1, 12))
projekt.model.cpu().eval()
wieder.model.eval()                                       # Dropout aus – sonst unterscheiden sich die Ausgaben
with torch.no_grad():
    print("Gleich nach Laden:", torch.allclose(projekt.model(x), wieder.model(x)))
print("Export:", wieder.export("ausgaben/qlora_demo/modell", "export"))   # LoRA eingerechnet
