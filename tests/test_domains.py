"""Tests für Segmentierung (U-Net), Audio, Instruction-Tuning (SFT) und Beam Search."""

import math
import struct
import wave

import torch

from ki_baukasten import Project, build_model, registry
from ki_baukasten.data.domains import log_mel_spectrogram, mel_filterbank, read_wav


def test_unet_shapes_and_padding():
    m = build_model({"name": "unet", "args": {"in_channels": 1, "output_dim": 3, "depth": 2, "base_channels": 4}})
    assert m(torch.randn(2, 1, 30, 22)).shape == (2, 3, 30, 22)
    assert len(m.checkpointable_blocks()) == 5


def test_segmentation_preset_learns():
    p = Project.from_preset("segmentierung_unet")
    p.set("data.n_samples", 200)
    p.set("model.base_channels", 8)
    p.set("training.max_steps", 80)
    p.set("training.verbose", False)
    p.train()
    out = p.evaluate()
    assert out["val_iou"] > 0.5 and "val_dice" in out
    pred = p.predict_with_uncertainty(p.data().val["inputs"][:2])
    assert pred["vorhersage"].shape == (2, 32, 32)
    assert "temperatur" in p.calibrate()


def test_mel_and_wav_roundtrip(tmp_path):
    fb = mel_filterbank(129, 20, 8000)
    assert fb.shape == (20, 129) and float(fb.max()) <= 1.0
    sr = 8000
    tone = [math.sin(2 * math.pi * 440 * i / sr) for i in range(sr // 4)]
    path = tmp_path / "ton.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(b"".join(struct.pack("<h", int(v * 30000)) for v in tone))
    x = read_wav(path, 16000)
    assert abs(len(x) - 4000) <= 1 and float(x.abs().max()) > 0.8
    spec = log_mel_spectrogram(x, 16000, n_mels=32)
    assert spec.shape[0] == 32


def test_audio_dataset_and_preset(tmp_path):
    b = registry("dataset").build({"name": "audio", "n_samples": 80})
    assert b.train["inputs"].dim() == 4 and b.info["output_dim"] == 4
    p = Project.from_preset("audio_klassifikation")
    p.set("data.n_samples", 160)
    p.set("training.max_steps", 40)
    p.set("training.verbose", False)
    p.train()
    assert p.evaluate()["val_accuracy"] > 0.6


def test_instructions_loss_mask_and_files(tmp_path):
    path = tmp_path / "daten.jsonl"
    path.write_text('{"prompt": "Hallo?", "response": "Hi."}\n{"instruction": "Zwei?", "output": "2."}\n',
                    encoding="utf-8")
    b = registry("dataset").build({"name": "instructions", "path": str(path), "val_fraction": 0.0})
    tok = b.info["tokenizer"]
    row, mask = b.train["inputs"][0], b.train["loss_mask"][0]
    text = tok.decode(row[mask].tolist())
    assert text in ("Hi.␄", "2.␄")                                      # nur die Antwort zählt im Verlust
    full = registry("dataset").build({"name": "instructions", "path": str(path), "val_fraction": 0.0,
                                      "train_on_prompt": True})
    assert int(full.train["loss_mask"].sum()) > int(b.train["loss_mask"].sum())


def test_sft_ignores_prompt_tokens_in_loss():
    from ki_baukasten import Trainer, TrainingConfig
    from ki_baukasten.training.metrics import Collector

    b = registry("dataset").build({"name": "instructions", "n_samples": 64})
    m = build_model({"name": "transformer", "args": {"vocab_size": b.info["vocab_size"], "d_model": 32, "n_layers": 1,
                                                       "max_seq_len": 64, "block": {"mixer.num_heads": 2}}})
    tr = Trainer(m, TrainingConfig(paradigm="causal_lm", epochs=1, verbose=False, validation_split=0.0))
    batch = {k: v[:4] for k, v in b.train.items()}
    tr._collector = Collector()
    tr.paradigm.compute(tr, batch, train=False)
    targets = tr._collector.targets[0]
    counted = (batch["loss_mask"][:, 1:] & batch["attention_mask"][:, 1:]).sum()
    assert int((targets != -100).sum()) == int(counted)               # nur Antwort-Tokens sind Ziele


def test_ask_and_beam_search():
    p = Project.from_preset("sprachmodell_sft")
    p.set("model.d_model", 32)
    p.set("model.n_layers", 1)
    p.set("training.max_steps", 5)
    p.set("training.verbose", False)
    p.train()
    answer = p.ask("Was ist 1 plus 1?", max_new_tokens=6)
    assert isinstance(answer, str) and "␄" not in answer
    model = p.model
    ids = torch.tensor([[3, 4, 5]])
    greedy = model.generate(ids, max_new_tokens=5, temperature=0)
    beams = model.generate(ids, max_new_tokens=5, num_beams=3)
    assert greedy.shape == beams.shape == (1, 8)
    no_rep = model.generate(ids, max_new_tokens=12, temperature=0, no_repeat_ngram_size=2)
    seq = no_rep[0].tolist()
    bigrams = list(zip(seq, seq[1:]))
    assert len(bigrams) == len(set(bigrams))
