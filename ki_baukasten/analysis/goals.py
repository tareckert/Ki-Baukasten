"""
ki_baukasten.analysis.goals
===========================

**Ziel-Assistent**: Wofür soll die KI genutzt werden – und was ist wichtig?

Aus Aufgabe, Datenmenge, Sequenzlänge, Prioritäten, Zielgerät, Trainings-Hardware und freiem
Text (Stichwörter wie "Handy", "Echtzeit", "Datenschutz") bewertet ``evaluate_goals`` alle Vorlagen:

* **Empfohlen** – passende Modelle/Varianten mit Begründung (✓ Stärken, ✗ Schwächen),
* **Abgeraten** – Modelle, die zur Aufgabe gehören, aber zu den Zielen schlecht passen (mit Grund),
* **Techniken** – passende Optimierungen aus dem Katalog (z. B. Datenschutz → DP-SGD, föderiert),
* **Hinweise** – z. B. Weiterlernen, Generierungseinstellungen, Reduktion für das Zielgerät.

Die Ziele werden im Projekt gespeichert (``goals``) und können **jederzeit geändert** werden;
``Project.goal_check()`` bewertet das aktuelle Modell danach neu.

Die Bewertung ist ein regelbasiertes Expertensystem (bewusst nachvollziehbar, jede Begründung
wird angezeigt) – keine Garantie, aber ein fundierter Startpunkt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core import BaseConfig, Level, Param, T

TASKS: dict[str, str] = {
    "unbekannt": "noch offen – aus der Beschreibung ableiten",
    "text_erzeugen": "Text erzeugen (Chatbot, Vervollständigung, Code)",
    "text_klassifizieren": "Text einordnen (Stimmung, Spam, Thema)",
    "uebersetzen": "Text umwandeln (Übersetzung, Zusammenfassung)",
    "bild_klassifizieren": "Bilder erkennen/einordnen",
    "bild_segmentieren": "Bilder pixelgenau einteilen (Segmentierung: Medizin, Satellit, Prüfung)",
    "bild_erzeugen": "Bilder/Daten erzeugen",
    "audio_klassifizieren": "Geräusche/Sprachbefehle erkennen",
    "assistent": "Anweisungen befolgen / Fragen beantworten (Assistent, Instruction-Tuning)",
    "zeitreihe_prognose": "Zeitreihen vorhersagen (Sensoren, Verkäufe, Energie)",
    "tabelle_klassifizieren": "Tabellendaten einordnen",
    "tabelle_regression": "Zahlen aus Tabellendaten vorhersagen",
    "anomalien_finden": "Auffälligkeiten erkennen (Betrug, Defekte)",
    "graph_analyse": "Graphen analysieren (Moleküle, Netzwerke)",
    "steuerung_rl": "Agenten steuern (Spiele, Robotik, Regelung)",
    "bewertung_praeferenz": "Antworten bewerten / Präferenzen lernen (RLHF)",
    "aehnlichkeit_embeddings": "Ähnlichkeit / Suche / Vortraining ohne Labels",
    "multimodal": "mehrere Datenarten kombinieren",
}

PRIORITIES: dict[str, str] = {
    "genauigkeit": "möglichst gute Ergebnisse",
    "geschwindigkeit": "schnelle Vorhersagen",
    "wenig_speicher": "wenig Speicher / kleines Modell",
    "schnelles_training": "kurze Trainingszeit",
    "erklaerbarkeit": "nachvollziehbare Entscheidungen",
    "robustheit": "stabil und robust",
    "datenschutz": "sensible Daten schützen",
    "energie": "wenig Energie / Akku",
    "einfachheit": "einfach zu verstehen und zu warten",
    "lange_kontexte": "lange Texte/Sequenzen",
    "echtzeit": "Echtzeit / Streaming",
    "weiterlernen": "später mit neuen Daten weiterlernen",
    "kreativitaet": "vielfältige, kreative Ausgaben",
}

_RANK = {"winzig": 0, "klein": 1, "mittel": 2, "gross": 3}

#: Aufwendige Techniken: werden empfohlen, aber nicht automatisch eingeschaltet (verändern Ablauf/Rechenzeit stark).
HEAVY = {"dp_sgd", "foederiert", "fedprox", "partikelschwarm", "genetisch", "pbt", "model_soup", "kreuzvalidierung",
         "ddp", "fsdp", "zero1", "evolution_strategies"}
_SEQ = {"keine": 0, "kurz": 1, "mittel": 2, "lang": 3, "sehr_lang": 4}


class GoalsConfig(BaseConfig):
    __title__ = "Ziele & Zweck"
    __tooltip__ = T(
        "Wofür die KI genutzt werden soll, was wichtig ist und wo sie laufen soll.",
        "Daraus empfiehlt der Assistent passende Modelle, rät von unpassenden ab und schlägt Techniken vor – "
        "jederzeit änderbar.",
        "Start eines neuen Projekts (Project.from_goals) und spätere Prüfung (Project.goal_check).",
    )
    task = Param("unbekannt", choices=TASKS, tooltip=T(
        "Hauptaufgabe der KI.", "Bestimmt, welche Modellfamilien überhaupt in Frage kommen.", "Möglichst konkret."))
    description = Param("", tooltip=T(
        "Freie Beschreibung: wofür, wichtige Punkte, Einschränkungen.",
        "Stichwörter (z. B. 'Handy', 'Echtzeit', 'Datenschutz', 'lange Texte', 'erklärbar') werden erkannt.",
        "Ein, zwei Sätze genügen."))
    priorities = Param([], item_type=str, validator=lambda v: None if all(x in PRIORITIES for x in v) else
                       f"unbekannte Priorität(en) {[x for x in v if x not in PRIORITIES]}; erlaubt: "
                       f"{', '.join(PRIORITIES)}", tooltip=T(
        "Was besonders wichtig ist (Mehrfachauswahl): " + ", ".join(PRIORITIES) + ".",
        "Gewichtet Stärken und Schwächen der Modelle.", "2–4 Punkte wählen."))
    data_size = Param("mittel", choices={"winzig": "unter 1.000 Beispiele", "klein": "1.000–10.000",
                                         "mittel": "10.000–1 Mio.", "gross": "über 1 Mio."}, tooltip=T(
        "Wie viele Trainingsbeispiele vorhanden sind.", "Große Modelle brauchen viele Daten.", "Grob schätzen."))
    labels = Param("vorhanden", choices={"vorhanden": "Ziele/Labels vorhanden", "wenige": "nur wenige Labels",
                                         "keine": "keine Labels (selbstüberwacht)"}, tooltip=T(
        "Gibt es zu den Daten die richtigen Antworten?", "Ohne Labels: selbstüberwachte Verfahren.", "vorhanden."))
    sequence_length = Param("keine", choices={"keine": "keine Sequenzen", "kurz": "< 128", "mittel": "128–1.000",
                                              "lang": "1.000–16.000", "sehr_lang": "> 16.000"}, tooltip=T(
        "Typische Länge von Texten/Sequenzen.", "Lange Sequenzen sprengen klassische Attention.", "Bei Text/Zeitreihen."))
    target = Param(None, type=str, nullable=True, tooltip=T(
        "Zielgerät (Profil aus deployment, z. B. smartphone, raspberry_pi, laptop_cpu, gaming_gpu).",
        "Modelle, die dort nicht laufen, werden abgewertet; Reduktion wird eingeplant.", "Wo läuft die KI?"))
    training_hardware = Param("laptop_gpu", choices={
        "cpu": "nur CPU", "laptop_gpu": "kleine GPU", "gaming_gpu": "große GPU (8–24 GB)", "apple_silicon": "Apple M",
        "rechenzentrum_gpu": "Rechenzentrums-GPU", "cluster": "mehrere GPUs/Rechner"}, tooltip=T(
        "Womit trainiert wird.", "Bestimmt, wie groß das Modell sein darf und ob verteilt trainiert wird.", ""))
    pretrained = Param(False, tooltip=T(
        "Ein vortrainiertes Modell soll angepasst werden (Feintuning).", "Dann sind LoRA/QLoRA erste Wahl.",
        "Mit Project.import_model laden."))
    use_description = Param(True, tooltip=T(
        "Stichwörter der Beschreibung bei der Bewertung auswerten (ergänzt Aufgabe, Prioritäten, Zielgerät).",
        "Bequem beim ersten Anlegen. Project.from_goals übernimmt die erkannten Angaben fest und schaltet das danach "
        "aus – spätere Änderungen gelten dann genau so, wie sie eingestellt sind.", "an für neue Ziele."),
        level=Level.ADVANCED)
    avoid = Param([], item_type=str, tooltip=T(
        "Vorlagen/Modelle, die nicht vorgeschlagen werden sollen.", "Eigene Ausschlüsse.", "z. B. ['gan_2d']."),
        level=Level.ADVANCED)


# ===========================================================================
# Profile der Vorlagen (Stärken/Schwächen je Priorität)
# ===========================================================================

@dataclass(frozen=True)
class Profile:
    tasks: tuple[str, ...]
    related: tuple[str, ...] = ()
    strengths: tuple[str, ...] = ()
    weaknesses: tuple[str, ...] = ()
    min_data: str = "winzig"
    size: str = "klein"                  # winzig | klein | mittel | gross (für Zielgeräte)
    max_seq: str = "keine"
    needs_labels: bool = True
    train_cost: str = "niedrig"          # niedrig | mittel | hoch
    why: str = ""
    caveat: str = ""


PROFILES: dict[str, Profile] = {
    "sprachmodell_mini": Profile(("text_erzeugen",), ("uebersetzen", "text_klassifizieren"),
                                 ("genauigkeit", "kreativitaet", "einfachheit"), ("lange_kontexte", "energie"),
                                 "mittel", "mittel", "mittel", False, "mittel",
                                 "GPT-artiger Transformer – der bewährte Standard für Textgenerierung.",
                                 "Attention wächst quadratisch mit der Länge."),
    "sprachmodell_moe": Profile(("text_erzeugen",), (), ("genauigkeit",), ("wenig_speicher", "einfachheit", "energie"),
                                "gross", "gross", "mittel", False, "hoch",
                                "Mixture-of-Experts: mehr Kapazität bei gleicher Rechenzeit pro Token.",
                                "Alle Experten müssen in den Speicher – ungeeignet für kleine Geräte."),
    "sprachmodell_mamba": Profile(("text_erzeugen",), ("zeitreihe_prognose",),
                                  ("lange_kontexte", "echtzeit", "geschwindigkeit", "wenig_speicher"),
                                  ("erklaerbarkeit", "einfachheit"), "mittel", "mittel", "sehr_lang", False, "mittel",
                                  "Mamba/SSM: linearer Aufwand und konstanter Zustand – ideal für lange Sequenzen und "
                                  "Streaming."),
    "sprachmodell_hybrid": Profile(("text_erzeugen",), (), ("lange_kontexte", "genauigkeit"), ("einfachheit",),
                                   "mittel", "mittel", "sehr_lang", False, "mittel",
                                   "Hybrid aus SSM und Attention: lange Kontexte plus präzises Nachschlagen."),
    "sprachmodell_speichersparend": Profile(("text_erzeugen",), (), ("wenig_speicher", "energie"), ("schnelles_training",),
                                            "mittel", "klein", "mittel", False, "mittel",
                                            "Sprachmodell mit allen Speicher-Spartechniken (bf16, Checkpointing, GQA …)."),
    "sprachmodell_qlora": Profile(("text_erzeugen",), ("text_klassifizieren",), ("wenig_speicher",), ("einfachheit",),
                                  "winzig", "mittel", "mittel", False, "niedrig",
                                  "QLoRA-Feintuning: vortrainiertes Modell mit 4-Bit-Basis anpassen."),
    "sprachmodell_ram_streaming": Profile(("text_erzeugen",), (), ("wenig_speicher",), ("schnelles_training",),
                                          "mittel", "gross", "mittel", False, "hoch",
                                          "Großes Modell auf kleiner GPU: Blöcke werden aus dem RAM gestreamt."),
    "text_klassifikation": Profile(("text_klassifizieren",), (), ("genauigkeit", "geschwindigkeit"),
                                   ("erklaerbarkeit",), "klein", "klein", "mittel", True, "niedrig",
                                   "Kompakter Transformer-Encoder mit Pooling zum Einordnen von Texten."),
    "bert_vortraining": Profile(("text_klassifizieren",), ("aehnlichkeit_embeddings",), ("genauigkeit",),
                                ("schnelles_training",), "mittel", "mittel", "mittel", False, "mittel",
                                "BERT-artiges Vortraining (maskierte Wörter) – stark, wenn es nur wenige Labels gibt."),
    "uebersetzung_seq2seq": Profile(("uebersetzen",), ("text_erzeugen",), ("genauigkeit",), ("lange_kontexte",),
                                    "mittel", "mittel", "mittel", True, "mittel",
                                    "Encoder-Decoder-Transformer für Übersetzung und Zusammenfassung."),
    "bildklassifikation_cnn": Profile(("bild_klassifizieren",), (), ("geschwindigkeit", "robustheit",
                                                                      "schnelles_training", "einfachheit"),
                                      (), "klein", "klein", "keine", True, "niedrig",
                                      "Faltungsnetz – robust und dateneffizient für Bilder."),
    "bildklassifikation_vit": Profile(("bild_klassifizieren",), (), ("genauigkeit",), ("schnelles_training", "energie"),
                                      "gross", "mittel", "keine", True, "hoch",
                                      "Vision Transformer – sehr stark mit vielen Daten.",
                                      "Mit wenigen Daten schlechter als CNNs (keine eingebauten Bild-Annahmen)."),
    "bildklassifikation_mobil": Profile(("bild_klassifizieren",), (), ("geschwindigkeit", "wenig_speicher", "energie",
                                                                        "echtzeit"), ("genauigkeit",),
                                        "klein", "winzig", "keine", True, "niedrig",
                                        "Mobiles CNN (depthwise) – für Smartphone, Raspberry Pi und Echtzeit."),
    "vae_bilder": Profile(("bild_erzeugen",), ("anomalien_finden", "aehnlichkeit_embeddings"),
                          ("einfachheit", "schnelles_training", "robustheit"), ("genauigkeit", "kreativitaet"),
                          "klein", "klein", "keine", False, "niedrig",
                          "Variational Autoencoder: stabiles Training, glatter latenter Raum.",
                          "Erzeugte Bilder sind oft unscharf."),
    "zeitreihen_lstm": Profile(("zeitreihe_prognose",), (), ("echtzeit", "wenig_speicher", "schnelles_training",
                                                              "einfachheit"), ("lange_kontexte",),
                               "klein", "winzig", "mittel", True, "niedrig",
                               "LSTM: bewährt für Zeitreihen und Streaming mit wenig Daten."),
    "zeitreihen_transformer": Profile(("zeitreihe_prognose",), (), ("genauigkeit",), ("echtzeit", "wenig_speicher"),
                                      "mittel", "klein", "mittel", True, "mittel",
                                      "Transformer für Zeitreihen – erkennt weitreichende Muster."),
    "sequenz_gedaechtnis_rnn": Profile(("text_klassifizieren", "zeitreihe_prognose"), (), ("echtzeit", "wenig_speicher"),
                                       ("genauigkeit",), "klein", "winzig", "mittel", True, "niedrig",
                                       "RNN mit Gedächtnis für Sequenzen – klein und streamingfähig."),
    "tabellen_klassifikation": Profile(("tabelle_klassifizieren",), (), ("einfachheit", "schnelles_training",
                                                                          "geschwindigkeit", "wenig_speicher"), (),
                                       "winzig", "winzig", "keine", True, "niedrig",
                                       "Residual-MLP – schneller, starker Standard für Tabellen."),
    "tabellen_regression": Profile(("tabelle_regression",), (), ("einfachheit", "schnelles_training", "geschwindigkeit",
                                                                  "wenig_speicher"), (), "winzig", "winzig", "keine",
                                   True, "niedrig", "MLP für Zahlen-Vorhersagen aus Tabellen."),
    "ensemble_tabellen": Profile(("tabelle_klassifizieren",), ("tabelle_regression",), ("genauigkeit", "robustheit"),
                                 ("geschwindigkeit", "wenig_speicher", "einfachheit"), "klein", "klein", "keine", True,
                                 "mittel", "Ensemble mehrerer Modelle – robuster und genauer, aber größer."),
    "gan_2d": Profile(("bild_erzeugen",), (), ("geschwindigkeit", "kreativitaet"), ("robustheit", "einfachheit"),
                      "klein", "klein", "keine", False, "mittel", "GAN: sehr schnelle Erzeugung.",
                      "Instabiles Training, Modus-Kollaps möglich."),
    "diffusion_2d": Profile(("bild_erzeugen",), (), ("genauigkeit", "robustheit", "kreativitaet"),
                            ("geschwindigkeit", "energie", "echtzeit"), "klein", "klein", "keine", False, "mittel",
                            "Diffusionsmodell: stabilste und hochwertigste Generierung.",
                            "Erzeugung braucht viele Schritte (langsam)."),
    "graph_klassifikation": Profile(("graph_analyse",), (), ("genauigkeit",), (), "klein", "klein", "keine", True,
                                    "niedrig", "Graph-Neuronales Netz für Moleküle und Netzwerke."),
    "anomalie_autoencoder": Profile(("anomalien_finden",), (), ("einfachheit", "schnelles_training", "erklaerbarkeit"),
                                    (), "klein", "winzig", "keine", False, "niedrig",
                                    "Autoencoder: lernt das Normale, Auffälliges hat hohen Rekonstruktionsfehler."),
    "reward_modell": Profile(("bewertung_praeferenz",), (), ("einfachheit",), (), "klein", "klein", "keine", True,
                             "niedrig", "Reward-Modell aus Paarvergleichen (Bradley-Terry) für RLHF."),
    "rl_cartpole_ppo": Profile(("steuerung_rl",), (), ("robustheit", "einfachheit"), ("schnelles_training",),
                               "winzig", "winzig", "keine", False, "mittel",
                               "PPO: robuster Standard-Algorithmus für Steuerungsaufgaben."),
    "rl_cartpole_dqn": Profile(("steuerung_rl",), (), ("schnelles_training",), ("robustheit",), "winzig", "winzig",
                               "keine", False, "niedrig", "DQN: effizient für wenige, diskrete Aktionen."),
    "rl_gitterwelt_belohnungen": Profile(("steuerung_rl",), (), ("erklaerbarkeit",), (), "winzig", "winzig", "keine",
                                         False, "niedrig", "RL mit frei kombinierbaren Belohnungen."),
    "rl_pendel_kontinuierlich": Profile(("steuerung_rl",), (), ("robustheit",), ("schnelles_training",), "winzig",
                                        "winzig", "keine", False, "mittel",
                                        "PPO für kontinuierliche Aktionen (Robotik, Regelung)."),
    "multimodal_fusion": Profile(("multimodal",), (), ("genauigkeit",), ("einfachheit",), "klein", "klein", "keine",
                                 True, "mittel", "Fusion mehrerer Eingänge (z. B. Bild + Tabelle)."),
    "mixture_of_models": Profile(("tabelle_regression",), ("tabelle_klassifizieren", "multimodal"), ("genauigkeit",),
                                 ("einfachheit", "wenig_speicher"), "klein", "klein", "keine", True, "mittel",
                                 "Mixture-of-Models: ein Router wählt Spezialisten."),
    "kontrastiv_vortraining": Profile(("aehnlichkeit_embeddings",), ("tabelle_klassifizieren",), ("genauigkeit",),
                                      ("einfachheit",), "klein", "klein", "keine", False, "mittel",
                                      "Kontrastives Lernen: gute Repräsentationen ohne Labels."),
    "sprachmodell_sft": Profile(("assistent", "text_erzeugen"), ("uebersetzen",),
                                ("genauigkeit", "einfachheit"), ("lange_kontexte", "energie"),
                                "klein", "mittel", "kurz", True, "mittel",
                                "Instruction-Tuning: Antworten auf Anweisungen, Verlust nur auf den Antworten.",
                                "Für echte Assistenten ein vortrainiertes Modell importieren und mit LoRA feintunen."),
    "segmentierung_unet": Profile(("bild_segmentieren",), ("bild_klassifizieren",),
                                  ("genauigkeit", "robustheit", "erklaerbarkeit"), ("energie",),
                                  "klein", "klein", "keine", True, "mittel",
                                  "U-Net – Standard für Segmentierung, sehr dateneffizient durch Skip-Verbindungen."),
    "audio_klassifikation": Profile(("audio_klassifizieren",), ("zeitreihe_prognose",),
                                    ("geschwindigkeit", "wenig_speicher", "robustheit"), (),
                                    "klein", "klein", "keine", True, "niedrig",
                                    "CNN auf Log-Mel-Spektrogrammen – bewährt für Geräusche und Sprachbefehle."),
    "freies_netz": Profile((), ("tabelle_klassifizieren",), ("erklaerbarkeit",), ("einfachheit",), "winzig", "winzig",
                           "keine", True, "niedrig", "Frei verdrahteter Netzgraph für eigene Architekturen."),
    "forschung_experimentell": Profile((), ("text_erzeugen",), ("genauigkeit",), ("einfachheit", "robustheit"),
                                       "mittel", "mittel", "lang", False, "mittel",
                                       "Experimentelle Konfiguration mit neuesten Bausteinen."),
}

#: Zielgerät → größte sinnvolle Modellklasse.
_TARGET_SIZE = {"mikrocontroller": "winzig", "raspberry_pi": "klein", "browser": "klein", "smartphone": "klein",
                "laptop_cpu": "mittel", "laptop_gpu": "mittel", "apple_silicon": "mittel", "server_cpu": "gross",
                "gaming_gpu": "gross", "rechenzentrum_gpu": "gross"}
_HW_SIZE = {"cpu": "klein", "laptop_gpu": "mittel", "apple_silicon": "mittel", "gaming_gpu": "gross",
            "rechenzentrum_gpu": "gross", "cluster": "gross"}

#: Stichwörter im Freitext → (Aufgabe | None, Prioritäten, Zielgerät | None)
_KEYWORDS: list[tuple[str, str | None, tuple[str, ...], str | None]] = [
    (r"handy|smartphone|mobil|app\b", None, ("wenig_speicher", "energie"), "smartphone"),
    (r"raspberry", None, ("wenig_speicher",), "raspberry_pi"),
    (r"mikrocontroller|arduino|esp32", None, ("wenig_speicher", "energie"), "mikrocontroller"),
    (r"browser|webseite|web-app", None, ("wenig_speicher",), "browser"),
    (r"echtzeit|live|stream", None, ("echtzeit", "geschwindigkeit"), None),
    (r"lange? (texte|dokumente|kontexte?|bücher)|langer kontext", None, ("lange_kontexte",), None),
    (r"datenschutz|dsgvo|medizin|patient|vertraulich|sensibel", None, ("datenschutz",), None),
    (r"erklär|nachvollzieh|transparen|begründ", None, ("erklaerbarkeit",), None),
    (r"schnell", None, ("geschwindigkeit",), None),
    (r"genau|präzis|beste qualität|höchste qualität", None, ("genauigkeit",), None),
    (r"stabil|robust|zuverlässig", None, ("robustheit",), None),
    (r"akku|energie|strom", None, ("energie",), None),
    (r"einfach|wartbar", None, ("einfachheit",), None),
    (r"weiterlern|neue daten|nachtrainier|kontinuierlich", None, ("weiterlernen",), None),
    (r"kreativ|geschichten|vielfalt", None, ("kreativitaet",), None),
    (r"assistent|anweisung|instruction|frage.{0,12}antwort|beantwort", "assistent", (), None),
    (r"chat|chatbot|text (erzeug|generier|schreib)|autovervollständ|code", "text_erzeugen", (), None),
    (r"segment|pixelgenau|maske|umriss|tumor|organ", "bild_segmentieren", (), None),
    (r"audio|geräusch|klang|sprachbefehl|mikrofon|ton\b|töne|akustik", "audio_klassifizieren", (), None),
    (r"übersetz|zusammenfass", "uebersetzen", (), None),
    (r"spam|stimmung|sentiment|thema|kategori", "text_klassifizieren", (), None),
    (r"(bild|foto|kamera).*(erzeug|generier|mal)", "bild_erzeugen", (), None),
    (r"(bild|foto|kamera|röntgen)", "bild_klassifizieren", (), None),
    (r"prognose|zeitreihe|sensor|vorhersage der|absatz|verbrauch", "zeitreihe_prognose", (), None),
    (r"betrug|anomal|ausreißer|defekt|fehlererkennung", "anomalien_finden", (), None),
    (r"spiel|roboter|agent|steuer|regel", "steuerung_rl", (), None),
    (r"molekül|graph|netzwerk", "graph_analyse", (), None),
    (r"tabelle|excel|csv", "tabelle_klassifizieren", (), None),
    (r"preis|umsatz|wert schätz", "tabelle_regression", (), None),
]


@dataclass
class Suggestion:
    preset: str
    score: float
    reasons: list[str] = field(default_factory=list)
    model: str = ""
    why: str = ""

    def __str__(self) -> str:
        head = f"{self.preset}  (Punkte {self.score:+.1f}, Modell {self.model})"
        lines = [head, f"      {self.why}"] if self.why else [head]
        lines += [f"      {r}" for r in self.reasons]
        return "\n".join(lines)


@dataclass
class GoalReport:
    goals: dict[str, Any]
    recommended: list[Suggestion] = field(default_factory=list)
    discouraged: list[Suggestion] = field(default_factory=list)
    techniques: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    current: Suggestion | None = None
    current_verdict: str = ""

    def __str__(self) -> str:
        g = self.goals
        lines = [f"Ziel-Assistent: Aufgabe '{g['task']}' – {TASKS.get(g['task'], '')}",
                 f"  Prioritäten: {', '.join(g['priorities']) or '–'} · Daten: {g['data_size']} · Labels: {g['labels']}"
                 + (f" · Sequenzen: {g['sequence_length']}" if g['sequence_length'] != "keine" else "")
                 + (f" · Zielgerät: {g['target']}" if g.get("target") else "")
                 + f" · Training: {g['training_hardware']}"]
        if self.current is not None:
            lines.append(f"\nAktuelles Modell: {self.current_verdict}")
            lines += [f"    {r}" for r in self.current.reasons]
        lines.append("\nEmpfohlen:")
        for i, s in enumerate(self.recommended, 1):
            lines.append(f"  {i}. {s}")
        if not self.recommended:
            lines.append("  (keine passende Vorlage – Aufgabe genauer angeben)")
        if self.discouraged:
            lines.append("\nAbgeraten:")
            for s in self.discouraged:
                lines.append(f"  ✗ {s}")
        if self.techniques:
            lines.append("\nPassende Techniken (projekt.enable(...); ◇ = aufwendig, nicht automatisch eingeschaltet):")
            lines += [f"  {'◇' if k in HEAVY else '•'} {k:<26} {why}" for k, why in self.techniques]
        if self.notes:
            lines.append("\nHinweise:")
            lines += [f"  – {n}" for n in self.notes]
        lines.append("\nStarten: Project.from_goals(ziele) · Ziele ändern: projekt.set_goals(...) · "
                     "prüfen: projekt.goal_check()")
        return "\n".join(lines)


# ===========================================================================
# Bewertung
# ===========================================================================

def _as_goals(goals: Any) -> GoalsConfig:
    if isinstance(goals, GoalsConfig):
        return goals.copy()
    return GoalsConfig.from_dict(goals or {})


def interpret(goals: GoalsConfig) -> GoalsConfig:
    """Stichwörter aus der Beschreibung übernehmen (Aufgabe, Prioritäten, Zielgerät) – sofern ``use_description``."""
    g = goals.copy()
    if not g.use_description:
        return g
    text = (g.description or "").lower()
    prios = list(g.priorities)
    for rx, task, add, target in _KEYWORDS:
        if not text or not re.search(rx, text):
            continue
        if task and g.task == "unbekannt":
            g.task = task
        for p in add:
            if p not in prios:
                prios.append(p)
        if target and g.target is None:
            g.target = target
    g.priorities = prios
    return g


def _score(name: str, prof: Profile, g: GoalsConfig) -> Suggestion | None:
    if name in g.avoid:
        return None
    if g.task in prof.tasks:
        score, reasons = 10.0, [f"✓ gemacht für '{g.task}'"]
    elif g.task in prof.related:
        score, reasons = 4.0, [f"○ auch für '{g.task}' geeignet (nicht die Hauptaufgabe)"]
    else:
        return None
    weak = False
    for p in g.priorities:
        if p in prof.strengths:
            score += 2.5
            reasons.append(f"✓ Stärke: {PRIORITIES[p]}")
        if p in prof.weaknesses:
            score -= 3.5
            weak = True
            reasons.append(f"✗ Schwäche: {PRIORITIES[p]}")
    if weak and prof.caveat:
        reasons.append(f"⚠ {prof.caveat}")
    need = _RANK[prof.min_data] - _RANK[g.data_size]
    if need > 0:
        score -= 2.0 * need
        reasons.append(f"✗ braucht mehr Daten (ab '{prof.min_data}', vorhanden '{g.data_size}')")
    elif _RANK[g.data_size] >= 2 and prof.min_data in ("winzig",) and "genauigkeit" in g.priorities \
            and prof.size == "winzig":
        reasons.append("○ bei vielen Daten evtl. zu klein – Breite/Tiefe erhöhen")
    seq_gap = _SEQ[g.sequence_length] - _SEQ[prof.max_seq]
    if g.sequence_length != "keine" and prof.max_seq != "keine":
        if seq_gap > 0:
            score -= 3.0 * seq_gap
            reasons.append(f"✗ Sequenzen '{g.sequence_length}' sind für dieses Modell lang (gut bis '{prof.max_seq}')")
        elif prof.max_seq == "sehr_lang" and _SEQ[g.sequence_length] >= 3:
            score += 2.0
            reasons.append("✓ lange Sequenzen mit linearem Aufwand")
    if g.labels == "keine" and prof.needs_labels:
        score -= 6.0
        reasons.append("✗ braucht Labels – es gibt keine")
    elif g.labels in ("keine", "wenige") and not prof.needs_labels:
        score += 1.5
        reasons.append("✓ lernt ohne (viele) Labels")
    if g.target:
        allowed = _TARGET_SIZE.get(g.target, "gross")
        gap = _RANK[prof.size] - _RANK[allowed]
        if gap > 0:
            score -= 2.5 * gap
            reasons.append(f"✗ für '{g.target}' zu groß – nur mit Reduktion/Schüler-Modell")
        elif _RANK[prof.size] <= _RANK[allowed] and g.target in ("smartphone", "raspberry_pi", "mikrocontroller",
                                                                  "browser"):
            score += 1.0
            reasons.append(f"✓ passt auf '{g.target}'")
    hw_gap = _RANK[prof.size] - _RANK[_HW_SIZE[g.training_hardware]]
    if hw_gap > 0 or (g.training_hardware == "cpu" and prof.train_cost == "hoch"):
        score -= 2.0 * max(1, hw_gap)
        reasons.append(f"✗ Training auf '{g.training_hardware}' aufwendig")
    if g.pretrained and name == "sprachmodell_qlora":
        score += 4.0
        reasons.append("✓ ideal zum Feintuning eines vortrainierten Modells")
    return Suggestion(name, score, reasons, "", prof.why)


def _technique_suggestions(g: GoalsConfig) -> tuple[list[tuple[str, str]], list[str]]:
    tech: list[tuple[str, str]] = []
    notes: list[str] = []
    add = (lambda key, why: tech.append((key, why)) if key not in [k for k, _ in tech] else None)
    p = set(g.priorities)
    if "genauigkeit" in p:
        add("ema", "geglättete Gewichte sind meist genauer")
        add("partikelschwarm", "Hyperparameter automatisch optimieren")
        add("model_soup", "mehrere Läufe mitteln")
    if "robustheit" in p:
        add("gradient_clipping", "keine Ausreißer-Schritte")
        add("spike_schutz", "Verlust-Spitzen abfangen")
        add("oom_schutz", "kein Abbruch bei Speichermangel")
    if "wenig_speicher" in p or "energie" in p:
        add("checkpointing", "weniger Aktivierungsspeicher")
        add("adamw8bit", "Optimierer-Speicher ÷ 4")
        add("bf16", "halber Aktivierungsspeicher")
        if g.pretrained:
            add("qlora", "Feintuning mit 4-Bit-Basis")
    if "geschwindigkeit" in p or "echtzeit" in p or "energie" in p or g.target:
        add("komprimieren_nach_training", f"für {'das Zielgerät' if g.target else 'schnelle Inferenz'} verkleinern")
    if "schnelles_training" in p:
        add("bf16", "schneller rechnen")
        add("daten_parallel_laden", "GPU wartet nicht auf Daten")
    if "datenschutz" in p:
        add("dp_sgd", "Einzelpersonen nicht rekonstruierbar")
        add("foederiert", "Daten bleiben bei den Teilnehmern")
    if "erklaerbarkeit" in p:
        add("l1_sparsity", "spärliche, besser deutbare Gewichte")
        add("gewichts_ueberwachung", "nachvollziehen, welche Schichten lernen")
        notes.append("Erklärbarkeit: kleine Modelle bevorzugen; der Probelauf (GUI 'Probelauf & Gewichte') zeigt "
                     "Aktivierungen, Gewichte und Attention-Karten.")
    if "weiterlernen" in p:
        add("checkpoints", "Stände sichern")
        notes.append("Weiterlernen ohne Vergessen: projekt.continue_training(neue_daten, method='lwf' | 'ewc' | "
                     "'replay').")
    if "kreativitaet" in p and g.task in ("text_erzeugen",):
        notes.append("Kreative Texte: generate(..., temperature=0.9, top_p=0.95); für Fakten temperature 0–0.3.")
    if "lange_kontexte" in p or g.sequence_length in ("lang", "sehr_lang"):
        add("checkpointing", "lange Sequenzen brauchen viel Aktivierungsspeicher")
        notes.append("Lange Kontexte: generate(..., prefill_chunk=512, cache={'dtype': 'int8'}) spart Speicher.")
    if g.data_size in ("winzig", "klein"):
        add("early_stopping", "wenig Daten → früh stoppen")
        add("weight_decay", "gegen Auswendiglernen")
        add("dropout", "gegen Auswendiglernen")
        add("kreuzvalidierung", "ehrliche Einschätzung trotz wenig Daten")
        if g.task in ("bild_klassifizieren",):
            add("bild_augmentierung", "mehr Vielfalt aus wenigen Bildern")
        if g.task in ("tabelle_klassifizieren", "bild_klassifizieren"):
            add("mixup", "glattere Entscheidungsgrenzen")
    if g.data_size == "gross":
        add("checkpoints", "lange Läufe absichern")
        add("log_jsonl", "Verlauf auswerten")
    if g.training_hardware == "cluster":
        add("ddp", "Daten parallel auf mehreren GPUs")
        add("fsdp", "Modell über GPUs verteilen, falls es nicht auf eine passt")
    if g.labels == "wenige":
        notes.append("Wenige Labels: erst selbstüberwacht vortrainieren (bert_vortraining / kontrastiv_vortraining), "
                     "dann feintunen.")
    if g.pretrained:
        notes.append("Vortrainiertes Modell: Project.import_model(pfad) – dann LoRA/QLoRA oder schichtweise Lernraten.")
    if g.target:
        notes.append(f"Nach dem Training: projekt.recommend_for('{g.target}') / reduce_for('{g.target}') – passt es "
                     "trotzdem nicht, schaltet reduce_for eine Notfall-Ausführung ein.")
    return tech, notes


def evaluate_goals(goals: Any, top: int = 4, project: Any = None) -> GoalReport:
    """Bewertet alle Vorlagen für die Ziele (``GoalsConfig`` oder dict); optional das aktuelle Projekt."""
    from ..presets import get_preset

    g = interpret(_as_goals(goals))
    rows = []
    for name, prof in PROFILES.items():
        s = _score(name, prof, g)
        if s is not None:
            try:
                s.model = get_preset(name).model.name
            except Exception:  # noqa: BLE001 - nicht registrierte Vorlage
                continue
            rows.append(s)
    rows.sort(key=lambda s: -s.score)
    recommended = [s for s in rows if s.score >= 6.0][:top]
    discouraged = [s for s in rows if s not in recommended and (s.score < 4.0 or
                                                               sum(r.startswith("✗") for r in s.reasons) >= 2)]
    discouraged.sort(key=lambda s: s.score)
    tech, notes = _technique_suggestions(g)
    report = GoalReport(g.to_dict(), recommended, discouraged, tech, notes)
    if project is not None:
        name = project.config.name
        current = next((s for s in rows if s.preset == name), None)
        if current is None:
            prof_name = _closest_profile(project)
            current = next((s for s in rows if s.preset == prof_name), None)
        if current is None:
            report.current = Suggestion(name, -10.0, [f"✗ Modell '{project.config.model.name}' passt nicht zur "
                                                      f"Aufgabe '{g.task}'"], project.config.model.name)
            report.current_verdict = "passt nicht zur Aufgabe – Wechsel empfohlen"
        else:
            report.current = current
            rank = [s.preset for s in recommended]
            best = recommended[0].preset if recommended else None
            if current.preset == best:
                report.current_verdict = "beste Wahl für diese Ziele ✓"
            elif current.preset in rank:
                report.current_verdict = (f"empfohlen (Platz {rank.index(current.preset) + 1}) – noch besser passt "
                                          f"'{best}' (Project.from_goals(projekt.config.goals))")
            elif current in discouraged:
                report.current_verdict = f"abgeraten ✗ – besser: '{best}'" if best else "abgeraten ✗"
            else:
                report.current_verdict = f"möglich, aber nicht erste Wahl – besser: '{best}'"
    return report


def _closest_profile(project: Any) -> str | None:
    """Vorlage mit gleichem Modell und Paradigma wie das Projekt (für eigene Projekte)."""
    from ..presets import get_preset

    cfg = project.config
    for name in PROFILES:
        try:
            pc = get_preset(name)
        except Exception:  # noqa: BLE001
            continue
        if pc.model.name == cfg.model.name and pc.training.paradigm.name == cfg.training.paradigm.name:
            return name
    return None
