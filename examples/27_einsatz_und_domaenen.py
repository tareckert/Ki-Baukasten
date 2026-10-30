"""
Beispiel 27 – Einsatz und weitere Anwendungen
==============================================

* Speichern mit Modellkarte, Umgebung und Vorverarbeitung → Laden **ohne** Trainingsdaten → Rohdaten vorhersagen
* HTTP-Dienst (Standardbibliothek): /vorhersage, /frage, /generieren, /info
* Segmentierung mit U-Net (IoU/Dice), Audio-Klassifikation auf Log-Mel-Spektrogrammen
* Instruction-Tuning: Verlust nur auf den Antworten, danach ``ask``; Beam Search und Wiederholungssperre

Ausführen:  python examples/27_einsatz_und_domaenen.py
"""

import json
import tempfile
import threading
import urllib.request
import warnings
from pathlib import Path

from ki_baukasten import Project
from ki_baukasten.ui.serve import create_server

warnings.simplefilter("ignore")
ruhig = {"training.verbose": False, "deployment.show_after_training": False}
ordner = Path(tempfile.mkdtemp())

# --- Speichern & Laden ohne Daten -------------------------------------------------------------
p = Project.from_preset("tabellen_klassifikation", **ruhig, **{"training.max_steps": 200,
                                                                "training.test_split": 0.15})
p.train()
p.test()
ziel = p.save(ordner / "tabelle")
print("Gespeichert:", sorted(f.name for f in ziel.iterdir()))
print((ziel / "MODELLKARTE.md").read_text(encoding="utf-8")[:700], "…\n")
q = Project.load(ziel)
print("Geladen, Vorhersage:", q.predict_raw(p.data().train["inputs"][:3])["vorhersage"])

# --- HTTP-Dienst ---------------------------------------------------------------------------------
server = create_server(q, port=0)                     # Port 0 = freier Port
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}"
anfrage = urllib.request.Request(url + "/vorhersage", data=json.dumps(
    {"daten": p.data().train["inputs"][:2].tolist(), "kompakt": True}).encode())
print("HTTP:", urllib.request.urlopen(anfrage).read().decode())
server.shutdown()

# --- Segmentierung -----------------------------------------------------------------------------
s = Project.from_preset("segmentierung_unet", **ruhig, **{"training.max_steps": 150})
s.enable("bild_augmentierung")                        # Spiegeln/Ausschnitt – die Masken werden mitbewegt
s.train()
print("Segmentierung:", {k: round(v, 3) for k, v in s.evaluate().items()})

# --- Audio ---------------------------------------------------------------------------------------
a = Project.from_preset("audio_klassifikation", **ruhig, **{"training.max_steps": 100})
a.train()
print("Audio:", {k: round(v, 3) for k, v in a.evaluate().items()}, a.data().info["label_names"])

# --- Instruction-Tuning ------------------------------------------------------------------------
lm = Project.from_preset("sprachmodell_sft", **ruhig, **{"training.max_steps": 500})
lm.train()
for frage in ["Was ist 12 plus 7?", "Nachfolger von 99?", "Zähle bis 4."]:
    print(f"{frage} → {lm.ask(frage)}  (Beam Search: {lm.ask(frage, num_beams=3)})")
lm.enable("keine_wiederholungen")                     # Standard für generate/ask/HTTP: project.generation
print("Generierungs-Standard:", lm.config.generation.to_dict(only_changed=True))
print(lm.generate("Frage: Was ist 3 plus 5?\nAntwort: ", max_new_tokens=6, temperature=0))
