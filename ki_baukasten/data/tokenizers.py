"""
ki_baukasten.data.tokenizers
============================

Tokenizer für Textdaten – ohne Zusatzpakete.

=========  ===================================================================================
Art        Eigenschaften
=========  ===================================================================================
zeichen    jedes Zeichen ein Token – winziges Vokabular, lange Folgen, kein "unbekanntes Wort"
wort       Wörter (und Satzzeichen) – kurze Folgen, aber unbekannte Wörter werden ``<unk>``
bpe        Byte-Pair-Encoding wie bei GPT/LLaMA: häufige Zeichenfolgen werden zu einem Token
           zusammengefasst – guter Kompromiss aus Vokabulargröße und Folgenlänge
=========  ===================================================================================

IDs: 0 = ``<pad>``, 1 = ``<unk>`` (bei zeichen/wort/bpe), danach das gelernte Vokabular.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

PAD, UNK = 0, 1
_SPACE = "▁"   # markiert ein vorangestelltes Leerzeichen (wie SentencePiece)


class _Base:
    kind = "basis"

    def __init__(self) -> None:
        self.itos: list[str] = ["<pad>", "<unk>"]
        self.stoi: dict[str, int] = {}

    def _finish(self) -> None:
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def decode(self, ids: Any) -> str:
        return "".join(self.itos[int(i)] for i in ids if 1 < int(i) < len(self.itos))

    def __len__(self) -> int:
        return self.vocab_size

    def to_dict(self) -> dict[str, Any]:
        """Gelernter Zustand als JSON-fähiges dict (zum Speichern mit dem Modell)."""
        return {"kind": self.kind, "itos": list(self.itos)}

    def _restore(self, d: dict[str, Any]) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.kind}: {self.vocab_size} Tokens>"


class CharTokenizer(_Base):
    """Zeichen-Tokenizer (ID 0 = Padding, 1 = unbekanntes Zeichen)."""

    kind = "zeichen"

    def __init__(self, text: str):
        super().__init__()
        self.chars = sorted(set(text))
        self.itos += self.chars
        self._finish()

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in text]

    def _restore(self, d: dict[str, Any]) -> None:
        self.chars = self.itos[2:]


class WordTokenizer(_Base):
    """Wort-Tokenizer: Wörter und Satzzeichen, häufigste ``vocab_size`` Einträge."""

    kind = "wort"
    _RX = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, text: str, vocab_size: int = 5000, lowercase: bool = True):
        super().__init__()
        self.lowercase = lowercase
        counts = Counter(self._split(text))
        self.itos += [w for w, _ in counts.most_common(max(0, vocab_size - 2))]
        self._finish()

    def _split(self, text: str) -> list[str]:
        return self._RX.findall(text.lower() if self.lowercase else text)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(w, UNK) for w in self._split(text)]

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "lowercase": self.lowercase}

    def _restore(self, d: dict[str, Any]) -> None:
        self.lowercase = bool(d.get("lowercase", True))

    def decode(self, ids: Any) -> str:
        words = [self.itos[int(i)] for i in ids if 0 < int(i) < len(self.itos)]   # <unk> bleibt sichtbar
        out = ""
        for w in words:
            out += w if (not out or re.fullmatch(r"[^\w\s]", w)) else " " + w
        return out


class BPETokenizer(_Base):
    """Byte-Pair-Encoding auf Zeichenebene (wie GPT/LLaMA-Tokenizer, vereinfacht).

    Training: alle Wörter in Zeichen zerlegen und wiederholt das häufigste benachbarte Paar zu einem
    neuen Token verschmelzen, bis ``vocab_size`` erreicht ist.
    """

    kind = "bpe"

    def __init__(self, text: str, vocab_size: int = 1000, max_train_chars: int = 2_000_000):
        super().__init__()
        words = Counter(self._pretokenize(text[:max_train_chars]))
        alphabet = sorted({c for w in words for c in w})
        self.itos += alphabet
        splits = {w: list(w) for w in words}
        self.merges: dict[tuple[str, str], int] = {}
        while len(self.itos) < vocab_size:
            pairs: Counter = Counter()
            for w, freq in words.items():
                s = splits[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            self.merges[(a, b)] = len(self.merges)
            self.itos.append(a + b)
            for w, s in splits.items():
                i, out = 0, []
                while i < len(s):
                    if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                        out.append(a + b)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                splits[w] = out
        self._finish()
        self._cache: dict[str, list[int]] = {}

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return [m.replace(" ", _SPACE) for m in re.findall(r" ?[^\s]+|\s+", text)]

    def _encode_word(self, word: str) -> list[int]:
        if word in self._cache:
            return self._cache[word]
        s = list(word)
        while len(s) > 1:
            ranked = [(self.merges.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(s, s[1:]))]
            rank, i = min(ranked)
            if rank == 1 << 30:
                break
            s = s[:i] + [s[i] + s[i + 1]] + s[i + 2:]
        ids = [self.stoi.get(t, UNK) for t in s]
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for w in self._pretokenize(text) for i in self._encode_word(w)]

    def decode(self, ids: Any) -> str:
        return super().decode(ids).replace(_SPACE, " ")

    def to_dict(self) -> dict[str, Any]:
        order = sorted(self.merges, key=self.merges.get)
        return {**super().to_dict(), "merges": [list(m) for m in order]}

    def _restore(self, d: dict[str, Any]) -> None:
        self.merges = {(a, b): i for i, (a, b) in enumerate(d.get("merges", []))}
        self._cache = {}


_CLASSES = {"zeichen": CharTokenizer, "wort": WordTokenizer, "bpe": BPETokenizer}


def tokenizer_from_dict(d: dict[str, Any]) -> Any:
    """Gespeicherten Tokenizer (``to_dict``) ohne erneutes Training wiederherstellen."""
    cls = _CLASSES[d["kind"]]
    tok = cls.__new__(cls)
    _Base.__init__(tok)
    tok.itos = list(d["itos"])
    tok._finish()
    tok._restore(d)
    return tok


KINDS = {"zeichen": "jedes Zeichen ein Token", "wort": "Wörter und Satzzeichen",
         "bpe": "Byte-Pair-Encoding (häufige Zeichenfolgen als Token, wie GPT/LLaMA)"}


def make_tokenizer(kind: str, text: str, vocab_size: int = 1000) -> Any:
    """Tokenizer der Art ``zeichen`` | ``wort`` | ``bpe`` auf ``text`` trainieren."""
    if kind == "zeichen":
        return CharTokenizer(text)
    if kind == "wort":
        return WordTokenizer(text, vocab_size)
    if kind == "bpe":
        return BPETokenizer(text, vocab_size)
    raise ValueError(f"Unbekannter Tokenizer '{kind}'. Erlaubt: {', '.join(KINDS)}.")
