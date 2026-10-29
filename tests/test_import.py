"""Tests: fremde Modelle importieren, ändern, speichern/laden und neu exportieren."""

import warnings

import pytest
import torch

from ki_baukasten import ConfigError, Project

warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("quelle")
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.set("training.max_steps", 3)
    p.train()
    p.model.cpu().eval()
    files = {fmt: p.export(d / "lm", fmt) for fmt in ("export", "torchscript", "state_dict")}
    return p, files


@pytest.mark.parametrize("fmt", ["export", "torchscript", "state_dict"])
def test_import_matches_original(trained, fmt):
    src, files = trained
    kwargs = {"architecture": src.config.model.to_dict()} if fmt == "state_dict" else {"input_shape": [1, 16],
                                                                                      "input_high": 40}
    p = Project.import_model(files[fmt], **kwargs)
    assert p.model.import_format == fmt
    x = torch.randint(0, 40, (3, 20))  # andere Batchgröße/Länge als beim Export (dynamische Formen)
    with torch.no_grad():
        assert torch.allclose(p.model.eval()(x), src.model(x), atol=1e-5)
    assert "ExternalModel" in p.architecture("text", max_depth=1)


def test_import_modify_save_load_reexport(trained, tmp_path):
    src, files = trained
    p = Project.import_model(files["state_dict"], architecture=src.config.model.to_dict())
    p.config.training.verbose = False
    report = p.compress([{"name": "prune_ffn", "args": {"amount": 0.5, "importance": "magnitude"}},
                         {"name": "quantize", "args": {"format": "int8"}}])
    assert report.ratio("size_bytes") > 3
    p.save(tmp_path / "proj")
    q = Project.load(tmp_path / "proj")
    x = torch.randint(0, 40, (2, 12))
    with torch.no_grad():
        assert torch.allclose(q.model.eval()(x), p.model.eval()(x))
    out = p.export(tmp_path / "neu", "export")
    assert torch.export.load(str(out)).module()(x).shape[:2] == (2, 12)


def test_finetune_imported_export_with_rules(trained):
    _, files = trained
    p = Project.import_model(files["export"], input_shape=[1, 16], input_high=40, data="text")
    p.set("training.paradigm", "causal_lm")
    p.set("training.max_steps", 2)
    p.set("training.verbose", False)
    p.set("tensor_rules", [{"target": "embed", "op": "freeze"}])
    p.train()
    assert p.history
    assert [n for n, t in p.model.named_parameters() if not t.requires_grad] == ["inner.embed.weight"]


def test_import_key_map_and_strict(trained, tmp_path):
    src, _ = trained
    state = {"modell." + k: v for k, v in src.model.state_dict().items()}
    torch.save(state, tmp_path / "fremd.pt")
    with pytest.raises(ConfigError):
        Project.import_model(tmp_path / "fremd.pt", format="state_dict", architecture=src.config.model.to_dict())
    p = Project.import_model(tmp_path / "fremd.pt", format="state_dict", architecture=src.config.model.to_dict(),
                             key_map={"^modell\\.": ""})
    assert p.model.import_report["loaded"] == len(state)


def test_pickle_requires_permission(trained, tmp_path):
    src, _ = trained
    torch.save(src.model, tmp_path / "ganz.pt")
    with pytest.raises(ConfigError, match="allow_pickle"):
        Project.import_model(tmp_path / "ganz.pt", format="module")
    p = Project.import_model(tmp_path / "ganz.pt", format="module", allow_pickle=True)
    assert p.model.depth() == len(src.model.blocks)
