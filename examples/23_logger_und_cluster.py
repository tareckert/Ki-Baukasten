"""
Beispiel 23 – Logger (Umfang, Ziele, eigene Logger) und Cluster-Training
=======================================================================

* ``training.logging``: Stufe still … debug, Ziele Konsole/Datei/JSONL/CSV/Python-logging/TensorBoard/W&B/MLflow,
  eigene Stufe je Ziel, Gewichts-/Gradientenstatistiken, Rotation.
* ``Project.add_logger``: eigenen Logger (z. B. Firmen-``logging.Logger``) einbinden.
* ``training.distributed``: DDP, ZeRO-1, FSDP – hier mit 2 CPU-Prozessen über torchrun vorgeführt.

Ausführen:  python examples/23_logger_und_cluster.py
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from ki_baukasten import Project
from ki_baukasten.training.distributed import launch_command

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO, format="[firma] %(levelname)s %(message)s")

with tempfile.TemporaryDirectory() as ordner:
    p = Project.from_preset("tabellen_klassifikation", **{
        "training.max_steps": 30, "training.log_every_steps": 10, "deployment.show_after_training": False,
        "training.logging": {
            "level": "minimal",                                   # Konsole knapp …
            "directory": ordner, "run_name": "demo",
            "backends": ["console", {"name": "jsonl", "args": {"level": "normal"}},   # … Dateien ausführlich
                         {"name": "csv", "args": {"level": "normal"}}, "datei"],
            "weights_every_n_steps": 15, "max_file_mb": 5, "keep_files": 2}})

    class Zaehler:                                                 # beliebiges Objekt als Logger
        def __init__(self):
            self.n = 0

        def metrics(self, step, werte, phase):
            self.n += 1

    z = Zaehler()
    p.add_logger(z).add_logger(logging.getLogger("firma.ki"))
    p.train()
    lauf = Path(ordner) / "demo"
    zeilen = [json.loads(x) for x in (lauf / "protokoll.jsonl").read_text().splitlines()]
    print("Dateien:", sorted(f.name for f in lauf.iterdir()))
    print("JSONL-Einträge:", len(zeilen), "Phasen:", sorted({r.get("phase") for r in zeilen if r.get("phase")}))
    print("Eigener Logger hat", z.n, "Metrik-Pakete erhalten")

    # Cluster: 2 Prozesse (gloo/CPU) – auf GPUs genauso, nur mit --prozesse = Anzahl GPUs
    cmd = launch_command("tabellen_klassifikation", 2, "ddp",
                         extra=["--setze", "training.max_steps=20", "--setze", "training.verbose=true",
                                "--setze", "deployment.show_after_training=false", "--speichern", f"{ordner}/verteilt"])
    print("\nStarte verteilt:", " ".join(cmd[2:]))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    print("\n".join(line for line in res.stdout.splitlines() if "Training" in line or "Gespeichert" in line))
    print("Rückgabewert:", res.returncode, "| gespeichert:", (Path(ordner) / "verteilt" / "modell.pt").exists())
    sys.stdout.flush()
