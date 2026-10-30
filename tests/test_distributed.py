"""Verteiltes Training: 2 CPU-Prozesse (gloo) über torchrun – Ergebnis wie ein Prozess mit doppeltem Batch."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from ki_baukasten import MemoryConfig, Trainer, TrainingConfig, build_model
from ki_baukasten.training.distributed import launch_command

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed fehlt")

WORKER = textwrap.dedent('''
    import os, sys
    import torch
    from ki_baukasten import MemoryConfig, Project, Trainer, TrainingConfig, build_model
    strategy, out = sys.argv[1], sys.argv[2]
    rank = int(os.environ["RANK"])
    if strategy == "gan":
        p = Project.from_preset("gan_2d", **{"training.max_steps": 3, "training.verbose": False,
                                            "training.distributed.strategy": "ddp"})
        p.train()
        torch.save({k: v.detach() for k, v in p.model.state_dict().items()}, f"{out}_{rank}.pt")
        p.save(f"{out}_projekt")
        sys.exit(0)
    torch.manual_seed(0)
    x = torch.randn(64, 6); y = (x.sum(1) > 0).long()
    model = build_model({"name": "mlp", "args": {"input_dim": 6, "output_dim": 2, "hidden_dims": [16]}})
    tr = Trainer(model, TrainingConfig(max_steps=5, batch_size=8, verbose=False, seed=0, shuffle=False,
                                       validation_split=0.0, distributed={"strategy": strategy},
                                       optimizer={"name": "adamw", "args": {"lr": 0.01}}), MemoryConfig())
    tr.fit({"inputs": x, "targets": y})
    state = tr.dist.model_state(tr.model)
    torch.save({k: v.detach().cpu() for k, v in state.items()}, f"{out}_{rank}.pt")
    tr.save_checkpoint(f"{out}_ckpt.pt")
''')


def _run(tmp_path: Path, strategy: str) -> Path:
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    out = tmp_path / strategy
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "OMP_NUM_THREADS": "1"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(script),
           strategy, str(out)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-4000:]
    return out


def _single_process_reference() -> dict:
    torch.manual_seed(0)
    x = torch.randn(64, 6)
    y = (x.sum(1) > 0).long()
    model = build_model({"name": "mlp", "args": {"input_dim": 6, "output_dim": 2, "hidden_dims": [16]}})
    tr = Trainer(model, TrainingConfig(max_steps=5, batch_size=16, verbose=False, seed=0, shuffle=False,
                                       validation_split=0.0, optimizer={"name": "adamw", "args": {"lr": 0.01}}),
                 MemoryConfig())
    tr.fit({"inputs": x, "targets": y})
    return tr.model.state_dict()


@pytest.mark.parametrize("strategy", ["ddp", "zero1", "fsdp"])
def test_two_processes_match_single_process(tmp_path, strategy):
    out = _run(tmp_path, strategy)
    ref = _single_process_reference()
    rank0 = torch.load(f"{out}_0.pt")
    for k, v in ref.items():
        assert torch.allclose(rank0[k], v, atol=1e-5), k
    if strategy != "fsdp":          # FSDP liefert den vollständigen state_dict nur auf Prozess 0
        rank1 = torch.load(f"{out}_1.pt")
        assert all(torch.equal(rank0[k], rank1[k]) for k in rank0)
    ckpt = torch.load(f"{out}_ckpt.pt", weights_only=False)
    assert set(ckpt["model"]) == set(ref) and ckpt["optimizers"]["main"]["state"]


def test_gan_uses_gradient_allreduce(tmp_path):
    out = _run(tmp_path, "gan")
    a, b = torch.load(f"{out}_0.pt"), torch.load(f"{out}_1.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert (Path(f"{out}_projekt") / "modell.pt").exists()


def test_launch_command():
    cmd = launch_command("projekt.json", 4, "fsdp", nodes=2, node_rank=1, master="10.0.0.1:1234",
                         extra=["--speichern", "aus"])
    assert "--nproc_per_node=4" in cmd and "--nnodes=2" in cmd and "--master_addr=10.0.0.1" in cmd
    assert "training.distributed.strategy=fsdp" in cmd and cmd[-2:] == ["--speichern", "aus"]
    assert "--standalone" in launch_command("p.json", 2)
