"""Tests für Modellkarte, Umgebung, gespeicherte Vorverarbeitung (Laden ohne Daten) und HTTP-Server."""

import base64
import csv
import io
import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from ki_baukasten import Project
from ki_baukasten.data.tokenizers import BPETokenizer, WordTokenizer, tokenizer_from_dict
from ki_baukasten.ui.serve import Service, create_server


def test_tokenizer_roundtrip_through_json():
    text = "Hallo Welt, hallo Baukasten! Die Welt ist groß."
    for tok in (BPETokenizer(text, 60), WordTokenizer(text, 30)):
        again = tokenizer_from_dict(json.loads(json.dumps(tok.to_dict())))
        assert again.encode(text) == tok.encode(text) and again.decode(tok.encode(text)) == tok.decode(tok.encode(text))


@pytest.fixture()
def csv_project(tmp_path):
    path = tmp_path / "kunden.csv"
    g = torch.Generator().manual_seed(0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alter", "stadt", "einkommen", "kauft"])
        for i in range(240):
            inc = 2000 + int(torch.randint(0, 4000, (1,), generator=g))
            city = ["Berlin", "Hamburg", "Köln"][i % 3]
            w.writerow([20 + i % 50, city, "" if i % 13 == 0 else inc,
                        "ja" if inc > 4000 or city == "Köln" else "nein"])
    p = Project.from_preset("tabellen_klassifikation")
    p.set("data", {"name": "csv", "path": str(path), "target": "kauft"})
    p.set("training.max_steps", 120)
    p.set("training.verbose", False)
    p.set("training.test_split", 0.15)
    p.train()
    p.test()
    return p, path


def test_save_load_without_data_and_model_card(csv_project, tmp_path):
    p, path = csv_project
    out = p.save(tmp_path / "modell")
    names = {f.name for f in out.iterdir()}
    assert {"projekt.json", "modell.pt", "vorverarbeitung.json", "umgebung.json", "MODELLKARTE.md"} <= names
    card = (out / "MODELLKARTE.md").read_text(encoding="utf-8")
    assert "Test (nie im Training benutzt)" in card and "test_accuracy" in card and "Prüfsumme" in card
    rows = [{"alter": 30, "stadt": "Köln", "einkommen": 2500}, {"alter": 50, "stadt": "Berlin", "einkommen": 5500}]
    before = p.predict_raw(rows)["vorhersage"]
    path.unlink()                                                      # Originaldaten weg
    q = Project.load(out)
    after = q.predict_raw(rows)
    assert after["vorhersage"] == before and set(after["vorhersage"]) <= {"ja", "nein"}
    assert q.predict_raw([[30, "Köln", 2500]])["vorhersage"][0] == before[0]       # Liste in Spaltenreihenfolge


def test_service_and_http(csv_project):
    p, _ = csv_project
    svc = Service(p)
    assert svc.describe()["spalten"] == ["alter", "stadt", "einkommen"]
    assert svc.predict({"daten": {"alter": 30, "stadt": "Köln", "einkommen": 2500}})["vorhersage"]
    with pytest.raises(ValueError):
        svc.predict({})
    server = create_server(p, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert json.loads(urllib.request.urlopen(base + "/gesundheit").read())["status"] == "ok"
        body = json.dumps({"daten": [{"alter": 30, "stadt": "Köln", "einkommen": 2500}]}).encode()
        res = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/vorhersage", data=body)).read())
        assert res["vorhersage"][0] in ("ja", "nein") and 0 <= res["wahrscheinlichkeit"][0] <= 1
        assert b"<html" in urllib.request.urlopen(base + "/").read()
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(urllib.request.Request(base + "/vorhersage", data=b"[1]"))
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_server_requires_key_on_network_and_checks_it(csv_project):
    p, _ = csv_project
    with pytest.raises(ValueError):
        create_server(p, host="0.0.0.0", port=0)
    server = create_server(p, port=0, api_key="geheim")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/info")
        assert err.value.code == 401
        ok = urllib.request.Request(f"http://127.0.0.1:{port}/info", headers={"X-API-Key": "geheim"})
        assert json.loads(urllib.request.urlopen(ok).read())["eingabe"] == "tabelle"
    finally:
        server.shutdown()
        server.server_close()


def test_image_service_rejects_paths_and_accepts_base64(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    for cls, color in (("rot", (255, 0, 0)), ("blau", (0, 0, 255))):
        (tmp_path / cls).mkdir()
        for i in range(4):
            PIL.new("RGB", (8, 8), color).save(tmp_path / cls / f"{i}.png")
    p = Project.from_preset("bildklassifikation_cnn")
    p.set("data", {"name": "image_folder", "path": str(tmp_path), "size": [8, 8], "val_fraction": 0.25})
    p.set("training.max_steps", 3)
    p.set("training.verbose", False)
    p.train()
    svc = Service(p)
    with pytest.raises(ValueError):
        svc.predict({"daten": [str(tmp_path / "rot" / "0.png")]})
    buf = io.BytesIO()
    PIL.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    out = svc.predict({"daten": [{"base64": base64.b64encode(buf.getvalue()).decode()}]})
    assert out["vorhersage"][0] in ("rot", "blau")
    assert p.predict_raw([str(tmp_path / "rot" / "0.png")])["vorhersage"][0] in ("rot", "blau")   # lokal erlaubt
