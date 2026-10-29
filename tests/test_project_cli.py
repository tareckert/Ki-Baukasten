"""Tests: Projekt-Fassade, Presets, automatische Dimensionen, Speichern/Laden, CLI, Doku-Export."""

import json

import pytest
import torch

from ki_baukasten import Project, ProjectConfig, registry
from ki_baukasten.ui.cli import main as cli


@pytest.mark.parametrize("name", registry("preset").names())
def test_every_preset_builds_and_roundtrips(name):
    p = Project.from_preset(name)
    p.config.training.verbose = False
    model = p.build()
    assert model.num_parameters() > 0
    assert ProjectConfig.from_dict(p.config.to_dict(only_changed=True)) == p.config


@pytest.mark.parametrize("name", ["tabellen_klassifikation", "bildklassifikation_cnn", "sprachmodell_mini",
                                  "graph_klassifikation", "multimodal_fusion", "anomalie_autoencoder",
                                  "rl_cartpole_ppo", "gan_2d", "diffusion_2d"])
def test_presets_train_briefly(name):
    p = Project.from_preset(name)
    t = p.config.training
    t.verbose = False
    if t.paradigm.name == "reinforcement":
        t.paradigm.args.total_timesteps = 600
    else:
        t.max_steps = 3
    assert p.train()


def test_auto_fit_dims_follow_data():
    p = Project.from_preset("tabellen_klassifikation")
    p.set("data", {"name": "classification", "args": {"n_features": 7, "n_classes": 5}})
    p.config.training.verbose = False
    p.build()
    assert (p.config.model.args.input_dim, p.config.model.args.output_dim) == (7, 5)
    p = Project.from_preset("rl_cartpole_ppo")
    p.set("training.paradigm.environment", "pendulum")
    p.config.training.verbose = False
    p.build()
    assert p.config.model.args.action_type == "continuous" and p.config.model.args.obs_dim == 3


def test_save_load_generate_sample(tmp_path):
    p = Project.from_preset("sprachmodell_mini")
    p.config.training.verbose = False
    p.config.training.max_steps = 3
    p.train()
    d = p.save(tmp_path / "lm")
    q = Project.load(d)
    x = torch.randint(1, 10, (1, 5))
    assert torch.allclose(p.predict(x), q.predict(x), atol=1e-5)
    assert isinstance(q.generate("Der", max_new_tokens=5), str)
    g = Project.from_preset("diffusion_2d")
    assert g.sample(4, steps=3).shape == (4, 2)


def test_memory_profiles():
    p = Project.from_preset("sprachmodell_mini").apply_memory_profile("sparsam")
    assert p.config.memory.gradient_checkpointing and p.config.training.gradient_accumulation_steps == 4


def test_cli_commands(tmp_path, capsys):
    cli(["presets"])
    cli(["komponenten", "reward"])
    cli(["erklaere", "model.block.norm_position", "--quelle", "sprachmodell_mini"])
    cli(["erklaere", "optimizer:lion"])
    out = tmp_path / "k.json"
    cli(["vorlage", "tabellen_regression", "-o", str(out), "--setze", "model.hidden_dims=[32]", "--nur-geaendert"])
    assert json.loads(out.read_text())["model"]["args"]["hidden_dims"] == [32]
    cli(["pruefen", str(out)])
    cli(["zeige", str(out), "--geaendert"])
    cli(["tensoren", str(out), "--filter", "layers.0"])
    text = capsys.readouterr().out
    assert "Pre-LN" in text and "layers.0.ops.linear.weight" in text


def test_docs_export(tmp_path):
    from ki_baukasten.ui.docs import export_markdown

    path = export_markdown(tmp_path / "ref.md")
    text = path.read_text(encoding="utf-8")
    assert "Registry `reward`" in text and "model.init" in text
