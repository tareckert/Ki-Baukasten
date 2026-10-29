"""Logger: Stufen, Ziele (Konsole, Datei, JSONL, CSV, Python-logging, eigene), Rotation, Statistiken."""

import json
import logging
import re

from ki_baukasten import Project
from ki_baukasten.training.logger import LoggingConfig, RunLogger, _Rotating


def _project(tmp_path, **logging_cfg):
    return Project.from_preset("tabellen_klassifikation", **{
        "training.max_steps": 6, "training.log_every_steps": 2, "training.callbacks": [],
        "training.logging": {"directory": str(tmp_path), "run_name": "lauf", **logging_cfg}})


def test_levels_control_console(tmp_path, capsys):
    _project(tmp_path, level="still").train()
    assert capsys.readouterr().out.strip() == ""
    _project(tmp_path, level="minimal").train()
    out = capsys.readouterr().out
    progress = re.compile(r"^\s+Schritt\s+\d", re.M)
    assert "▶ Training" in out and not progress.search(out)
    _project(tmp_path, level="ausfuehrlich").train()
    out = capsys.readouterr().out
    assert progress.search(out) and "beispiele_pro_s" in out


def test_file_backends_write_everything(tmp_path):
    full = {"level": "normal"}                     # Konsole knapp (minimal), Dateien vollständig
    p = _project(tmp_path, backends=["console", {"name": "datei", "args": full}, {"name": "jsonl", "args": full},
                                     {"name": "csv", "args": full}], level="minimal",
                 weights_every_n_steps=3, gradients_every_n_steps=3)
    p.train()
    run = tmp_path / "lauf"
    rows = [json.loads(line) for line in (run / "protokoll.jsonl").read_text().splitlines()]
    kinds = {r["typ"] for r in rows}
    assert {"start", "meldung", "metriken"} <= kinds
    assert rows[0]["config"]["model"]["name"] == "mlp"             # ganze Projekt-Konfiguration
    phases = {r.get("phase") for r in rows if r["typ"] == "metriken"}
    assert {"train", "gewichte", "gradientennorm"} <= phases
    csv_lines = (run / "metriken.csv").read_text().splitlines()
    assert csv_lines[0] == "schritt,phase,name,wert" and len(csv_lines) > 5
    assert "Training beendet" in (run / "training.log").read_text()


def test_backend_own_level_and_python_logging(tmp_path, caplog):
    p = _project(tmp_path, level="minimal", backends=[{"name": "python", "args": {"level": "debug"}}])
    with caplog.at_level(logging.DEBUG, logger="ki_baukasten"):
        p.train()
    text = caplog.text
    assert "Schritt" in text and "train schritt=" in text            # Stufe debug nur für dieses Ziel


def test_custom_logger_object_and_std_logger(tmp_path):
    class Sammler:
        def __init__(self):
            self.metrics_seen, self.messages = [], []

        def message(self, kind, text):
            self.messages.append(kind)

        def metrics(self, step, values, phase):
            self.metrics_seen.append((step, phase, dict(values)))

    s = Sammler()
    std = logging.getLogger("firma.ki")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    std.addHandler(handler)
    std.setLevel(logging.INFO)
    p = _project(tmp_path, level="normal")
    p.add_logger(s).add_logger(std)
    p.train()
    assert any(phase == "train" and "loss" in v for _, phase, v in s.metrics_seen)
    assert "info" in s.messages and records


def test_rotation_keeps_n_files(tmp_path):
    f = _Rotating(tmp_path / "x.log", max_mb=0.0001, keep=2, flush_every=1)
    for i in range(200):
        f.write(f"zeile {i:05d} " + "x" * 40 + "\n")
    f.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["x.log", "x.log.1", "x.log.2"]


def test_missing_optional_backend_does_not_break(tmp_path):
    # TensorBoard/W&B/MLflow sind in der Testumgebung evtl. nicht installiert – das Training läuft trotzdem.
    logger = RunLogger(LoggingConfig(backends=["wandb", "mlflow"], directory=str(tmp_path)), verbose=False)
    logger.start("t", {})
    logger.metrics(1, {"loss": 1.0})
    logger.close()
