"""Notfall-Ausführung: stückweise rechnen, wenn das Modell nicht passt."""

import copy

import pytest
import torch

from ki_baukasten import MemoryConfig, Project, Trainer, TrainingConfig, build_model
from ki_baukasten.analysis.fallback import best_option, execution_options
from ki_baukasten.memory import MemoryManager
from ki_baukasten.memory.execution import concat_outputs, is_oom, run_chunked, split_batch

SPEC = {"name": "transformer", "args": {"vocab_size": 40, "d_model": 32, "n_layers": 3}}
DATA = {"inputs": torch.randint(0, 40, (32, 10))}


def _train(mem: MemoryConfig, steps: int = 4, callbacks=None, model=None):
    torch.manual_seed(0)
    model = model if model is not None else build_model(SPEC)
    tr = Trainer(model, TrainingConfig(paradigm="causal_lm", max_steps=steps, batch_size=8, verbose=False, seed=0,
                                       callbacks=callbacks or []), mem)
    tr.fit(DATA)
    return tr


def test_stream_with_frozen_weights_on_disk(tmp_path):
    mem_disk = MemoryConfig(lora={"enabled": True, "rank": 4}, offload={"mode": "stream", "storage": "disk",
                                                                         "directory": str(tmp_path)})
    mem_ram = MemoryConfig(lora={"enabled": True, "rank": 4}, offload={"mode": "stream"})
    a, b = _train(mem_disk), _train(mem_ram)
    sa = a.model.state_dict()
    sb = b.model.state_dict()
    assert list(sa) == list(sb)
    for k in sa:                                        # identisches Training, vollständiger state_dict
        assert sa[k].shape == sb[k].shape, k
        assert torch.allclose(sa[k].float(), sb[k].float(), atol=1e-5), k
    frozen_in_ram = [p for blk in a.model.checkpointable_blocks() for p in blk.parameters() if not p.requires_grad]
    assert frozen_in_ram and all(p.numel() == 0 for p in frozen_in_ram)    # Basis liegt wirklich auf der SSD
    assert list(tmp_path.glob("stream_block_*.pt"))


def test_disk_storage_refuses_training_frozen_weights(tmp_path):
    from ki_baukasten.memory import enable_block_streaming

    m = build_model(SPEC)
    for p in m.parameters():
        p.requires_grad_(False)
    enable_block_streaming(m, m.checkpointable_blocks(), torch.device("cpu"), storage="disk", directory=tmp_path)
    for p in m.checkpointable_blocks()[0].parameters():
        p.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Festplatte"):
        m(torch.randint(0, 40, (2, 6))).sum().backward()


def test_split_device_map_is_exact_and_trainable():
    m = build_model(SPEC)
    ref = copy.deepcopy(m)
    mm = MemoryManager(MemoryConfig(offload={"mode": "split", "split_device": "cpu", "layers": "-2:"}))
    m = mm.prepare_model(m)
    x = torch.randint(0, 40, (2, 8))
    assert torch.allclose(m(x), ref(x), atol=1e-6)
    assert len(mm.offloader.placement()) == 2
    _train(MemoryConfig(offload={"mode": "split", "layers": "-2:"}), steps=2)


def test_oom_splits_batch_and_keeps_result():
    calls = []

    def run(mem):
        torch.manual_seed(0)
        model = build_model(SPEC)
        tr = Trainer(model, TrainingConfig(paradigm="causal_lm", max_steps=3, batch_size=8, verbose=False, seed=0,
                                           shuffle=False), mem)
        original = tr.paradigm.training_step

        def limited(trainer, batch):
            if batch["inputs"].shape[0] > 2:
                raise torch.OutOfMemoryError("CUDA out of memory (simuliert)")
            calls.append(batch["inputs"].shape[0])
            return original(trainer, batch)

        tr.paradigm.training_step = limited
        tr.fit(DATA)
        return tr

    tr = run(MemoryConfig())
    assert tr._split == 4 and tr.global_step == 3 and set(calls) == {2}
    with pytest.raises(torch.OutOfMemoryError):
        run(MemoryConfig(oom_strategy="fehler"))


def test_execution_helpers():
    batch = {"inputs": torch.arange(10).reshape(10, 1), "meta": "x"}
    parts = split_batch(batch, 3)
    assert [p["inputs"].shape[0] for p in parts] == [4, 4, 2] and parts[0]["meta"] == "x"
    assert torch.equal(concat_outputs([p["inputs"] for p in parts]), batch["inputs"])
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and not is_oom(ValueError("x"))

    def fn(x):
        if x.shape[0] > 3:
            raise RuntimeError("DefaultCPUAllocator: can't allocate memory")
        return x * 2

    out, n = run_chunked(fn, torch.ones(10, 2))
    assert n == 4 and torch.equal(out, torch.full((10, 2), 2.0))


def test_lisa_trains_only_active_blocks():
    torch.manual_seed(0)
    model = build_model(SPEC)
    before = [copy.deepcopy(b.state_dict()) for b in model.checkpointable_blocks()]
    tr = _train(MemoryConfig(), steps=3, model=model,
                callbacks=[{"name": "lisa", "args": {"active_blocks": 1, "every_n_steps": 100}}])
    lisa = tr.callbacks[0]
    changed = [i for i, (b, old) in enumerate(zip(tr.model.checkpointable_blocks(), before))
               if any(not torch.equal(v, old[k]) for k, v in b.state_dict().items())]
    assert changed == lisa.active                       # nur der gezogene Block hat gelernt
    assert all(p.requires_grad for p in tr.model.parameters())   # danach wieder alles trainierbar


def test_chunked_prefill_and_chunked_predict_are_exact(tmp_path):
    p = Project.from_preset("sprachmodell_mini", **{"training.verbose": False})
    m = p.build().eval()
    ids = torch.randint(0, m.config.vocab_size, (2, 40))
    a = m.generate(ids, max_new_tokens=5, temperature=0)
    b = m.generate(ids, max_new_tokens=5, temperature=0, prefill_chunk=8)
    assert torch.equal(a, b)
    full = p.predict(ids)
    assert torch.allclose(p.predict(ids, chunk_size=1), full, atol=1e-6)
    d = p.save(tmp_path / "lm")
    q = Project.load(d, mmap=True)
    assert torch.allclose(q.predict(ids), full, atol=1e-6)


def test_options_and_apply_fallback():
    model = build_model({"name": "transformer", "args": {"vocab_size": 64, "d_model": 64, "n_layers": 6}})
    size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    opts = execution_options(model, size * 0.5, training=False)
    keys = [o.key for o in opts]
    assert {"split", "ram_offload", "disk_offload", "layerweise_stapel", "micro_batch"} <= set(keys)
    best = best_option(opts)
    assert best is not None and best.fits and not best.quality_loss
    train_opts = execution_options(model, size * 1.5, training=True, activations_mb=1.0, batch_size=8)
    assert any(o.key in ("lora", "lisa", "stream") and o.fits for o in train_opts)

    p = Project.from_preset("sprachmodell_mini", **{"training.verbose": False})
    chosen = p.apply_fallback("disk_offload", budget_mb=0.5)
    assert chosen.key == "disk_offload" and p.get("memory.offload.mode") == "disk"
    p.undo()
    assert p.get("memory.offload.mode") == "none"
    opt = p.apply_fallback("auto", budget_mb=8.0, training=True)
    assert opt.fits


def test_recommendation_lists_fallbacks_when_reduction_is_not_enough():
    p = Project.from_preset("sprachmodell_mini", **{"training.verbose": False, "model.d_model": 128})
    rec = p.recommend_for("mikrocontroller")
    assert not rec.fits_after and rec.fallbacks
    assert "Notfall" in str(rec)
