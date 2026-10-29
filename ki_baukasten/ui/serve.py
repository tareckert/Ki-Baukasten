"""
ki_baukasten.ui.serve
=====================

**Bereitstellung per HTTP** – ein kleiner Web-Server aus der Standardbibliothek (``http.server``),
damit Apps, Webseiten und andere Programme die KI nutzen können::

    python -m ki_baukasten bereitstellen ausgaben/meine_ki --port 8000

=================  =====  ======================================================================
Pfad               Art    Inhalt
=================  =====  ======================================================================
/                  GET    kleine Testseite im Browser
/info              GET    Name, Aufgabe, erwartete Eingaben (Spalten, Klassen …)
/gesundheit        GET    {"status": "ok"} (für Überwachung/Load-Balancer)
/vorhersage        POST   {"daten": [...]} → Vorhersagen mit Wahrscheinlichkeit/Unsicherheit
/frage             POST   {"frage": "..."} → {"antwort": "..."} (Instruction-Modelle)
/generieren        POST   {"prompt": "...", "max_new_tokens": 50} → {"text": "..."} (Sprachmodelle)
=================  =====  ======================================================================

``daten`` je nach Modell: Tabellenzeilen (dicts mit Spaltennamen), Texte, Bilder/WAV als Base64
(``{"base64": "..."}``) oder fertige Zahlenlisten (Tensoren). Dateipfade werden aus
Sicherheitsgründen über HTTP **nicht** angenommen. Standard: nur lokal erreichbar
(``127.0.0.1``); im Netzwerk nur mit ``api_key`` (Kopfzeile ``X-API-Key``).
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

_PAGE = """<!doctype html><html lang="de"><meta charset="utf-8"><title>{name}</title>
<body style="font-family:sans-serif;max-width:48rem;margin:2rem auto;padding:0 1rem">
<h1>{name}</h1><p>{desc}</p><p>Eingabe: <b>{kind}</b>. Beispiel für <code>POST /{route}</code>:</p>
<textarea id="t" rows="8" style="width:100%">{example}</textarea><br>
<button onclick="go()">Senden</button><pre id="o"></pre>
<script>
async function go(){{const r=await fetch('/{route}',{{method:'POST',headers:{{'Content-Type':'application/json'}},
body:document.getElementById('t').value}});document.getElementById('o').textContent=await r.text();}}
</script></body></html>"""


def jsonable(obj: Any) -> Any:
    """Tensoren & Co. in JSON-Werte umwandeln."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def _decode_items(items: list[Any], kind: str) -> list[Any]:
    out = []
    for it in items:
        if isinstance(it, dict) and "base64" in it:
            out.append(base64.b64decode(it["base64"]))
        elif kind in ("bild", "audio") and isinstance(it, str):
            raise ValueError("Bilder/Audio bitte als {\"base64\": \"…\"} senden – Dateipfade sind über HTTP gesperrt.")
        else:
            out.append(it)
    return out


class Service:
    """Die Logik hinter dem Server (ohne HTTP – auch direkt testbar)."""

    def __init__(self, project: Any, max_batch: int = 256):
        from ..data.artifacts import kind_of

        self.project = project
        self.max_batch = max_batch
        self.info = project._info()
        self.kind = kind_of(self.info)
        self.lock = threading.Lock()          # ein Modell, viele Anfragen: nacheinander rechnen

    def describe(self) -> dict[str, Any]:
        cfg = self.project.config
        prep = self.info.get("preprocessor")
        return jsonable({"name": cfg.name, "beschreibung": cfg.description, "modell": cfg.model.name,
                         "eingabe": self.kind, "aufgabe": self.info.get("task"),
                         "klassen": self.info.get("label_names"),
                         "spalten": prep.input_columns if prep is not None else None,
                         "vorhersage_einstellungen": cfg.inference.to_dict(only_changed=True)})

    def predict(self, body: dict[str, Any]) -> dict[str, Any]:
        items = body.get("daten", body.get("inputs"))
        if items is None:
            raise ValueError("Feld 'daten' fehlt.")
        if not isinstance(items, list):
            items = [items]
        if len(items) > self.max_batch:
            raise ValueError(f"Zu viele Beispiele ({len(items)} > {self.max_batch}).")
        numeric = bool(items) and all(isinstance(i, list) for i in items) and self.kind not in ("tabelle",)
        if self.kind == "tensor" or numeric:                 # fertige Zahlenlisten
            tokens = self.info.get("vocab_size") is not None and self.kind != "bild"
            x: Any = torch.tensor(items, dtype=torch.long if tokens else torch.float32)
        else:
            x = _decode_items(items, self.kind)
        with self.lock:
            out = self.project.predict_raw(x)
        if body.get("kompakt"):
            out.pop("wahrscheinlichkeiten", None)
        return jsonable(out)

    def ask(self, body: dict[str, Any]) -> dict[str, Any]:
        q = body.get("frage")
        if not isinstance(q, str):
            raise ValueError("Feld 'frage' (Text) fehlt.")
        opts = {k: body[k] for k in ("max_new_tokens", "temperature", "num_beams") if k in body}
        with self.lock:
            return {"antwort": self.project.ask(q, **opts)}

    def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Feld 'prompt' (Text) fehlt.")
        opts = {k: body[k] for k in ("max_new_tokens", "temperature", "top_k", "top_p", "num_beams",
                                     "no_repeat_ngram_size") if k in body}
        opts["max_new_tokens"] = min(int(opts.get("max_new_tokens", 50)), 2048)
        with self.lock:
            return {"text": self.project.generate(prompt, **opts)}

    def example(self) -> tuple[str, str]:
        if self.kind == "anweisung":
            return "frage", json.dumps({"frage": "Was ist 2 plus 3?"}, ensure_ascii=False)
        if self.kind == "sprachmodell":
            return "generieren", json.dumps({"prompt": "Der", "max_new_tokens": 40}, ensure_ascii=False)
        if self.kind == "tabelle":
            cols = self.info["preprocessor"].input_columns
            return "vorhersage", json.dumps({"daten": [{c: "" for c in cols}]}, ensure_ascii=False, indent=1)
        if self.kind == "text":
            return "vorhersage", json.dumps({"daten": ["Ein Beispieltext."]}, ensure_ascii=False)
        return "vorhersage", json.dumps({"daten": [{"base64": "…"}]} if self.kind in ("bild", "audio") else
                                        {"daten": [[0.0]]})


def make_handler(service: Service, api_key: str | None = None, max_body: int = 16 * 2**20) -> type:
    routes = {"vorhersage": service.predict, "predict": service.predict, "frage": service.ask,
              "generieren": service.generate, "generate": service.generate}

    class Handler(BaseHTTPRequestHandler):
        server_version = "KI-Baukasten"

        def log_message(self, fmt: str, *args: Any) -> None:     # ruhig (Protokoll über den Logger der KI)
            pass

        def _send(self, code: int, payload: Any, content_type: str = "application/json") -> None:
            data = payload.encode() if isinstance(payload, str) else json.dumps(
                payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _allowed(self) -> bool:
            if api_key is None or self.headers.get("X-API-Key") == api_key:
                return True
            self._send(401, {"fehler": "API-Schlüssel fehlt oder ist falsch (Kopfzeile X-API-Key)."})
            return False

        def do_GET(self) -> None:  # noqa: N802 - http.server-Schnittstelle
            path = self.path.split("?", 1)[0].strip("/")
            if path in ("gesundheit", "health"):
                return self._send(200, {"status": "ok"})
            if not self._allowed():
                return None
            if path == "info":
                return self._send(200, service.describe())
            if path == "":
                route, example = service.example()
                cfg = service.project.config
                return self._send(200, _PAGE.format(name=cfg.name, desc=cfg.description or "", kind=service.kind,
                                                    route=route, example=example), "text/html")
            return self._send(404, {"fehler": f"Unbekannter Pfad '/{path}'."})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].strip("/")
            if not self._allowed():
                return None
            if path not in routes:
                return self._send(404, {"fehler": f"Unbekannter Pfad '/{path}'. Erlaubt: {', '.join(routes)}."})
            length = int(self.headers.get("Content-Length") or 0)
            if length > max_body:
                return self._send(413, {"fehler": "Anfrage zu groß."})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("Erwartet wird ein JSON-Objekt.")
                return self._send(200, routes[path](body))
            except (ValueError, KeyError, TypeError) as exc:
                return self._send(400, {"fehler": str(exc)})
            except Exception as exc:  # noqa: BLE001 - Server bleibt am Leben
                return self._send(500, {"fehler": f"{type(exc).__name__}: {exc}"})

    return Handler


def create_server(project: Any, host: str | None = None, port: int | None = None, api_key: str | None = None,
                  max_batch: int | None = None) -> ThreadingHTTPServer:
    """Server erzeugen (Werte ohne Angabe aus ``deployment.server``); ``serve_forever()`` startet ihn."""
    sc = project.config.deployment.server
    host = host or sc.host
    api_key = api_key if api_key is not None else sc.api_key
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        raise ValueError("Im Netzwerk erreichbarer Server nur mit API-Schlüssel (deployment.server.api_key).")
    service = Service(project, max_batch or sc.max_batch)
    handler = make_handler(service, api_key, int(sc.max_body_mb * 2**20))
    return ThreadingHTTPServer((host, sc.port if port is None else port), handler)


def serve(project: Any, **kw: Any) -> None:
    server = create_server(project, **kw)
    host, port = server.server_address[:2]
    print(f"KI '{project.config.name}' bereit: http://{host}:{port}/  (Beenden: Strg+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
