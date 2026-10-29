"""Tests: Übernahme in layer_overrides, Geltungsbereiche, Modul-Chirurgie, Platzhalter-Pfade."""

import pytest
import torch

from ki_baukasten import ConfigError, Project, TensorRuleConfig, TensorSurgeon, build_model
from ki_baukasten.tensor import ModuleRuleConfig, ModuleSurgeon, apply_scope

BASE = {"name": "transformer", "vocab_size": 30, "d_model": 32, "n_layers": 6}


def test_layer_override_inheritance():
    m = build_model({**BASE, "layer_overrides": {
        "0": {"mixer.num_heads": 2, "ffn": "mlp", "dropout": 0.1},
        "-1": {"@von": 0},
        "2": {"@von": 0, "@nur": ["mixer.num_heads"]},
        "3": {"@from": 0, "@except": ["ffn"], "dropout": 0.3},
        "4": {"@von": "-1", "dropout": 0.0},
    }})
    heads = [b.mixer.attn.num_heads for b in m.blocks]
    ffns = [type(b.ffn).__name__ for b in m.blocks]
    drops = [b.cfg.dropout for b in m.blocks]
    assert heads == [2, 8, 2, 2, 2, 2]
    assert ffns == ["MLPFeedForward", "GLUFeedForward", "GLUFeedForward", "GLUFeedForward", "MLPFeedForward",
                    "MLPFeedForward"]
    assert drops == [0.1, 0.0, 0.0, 0.3, 0.0, 0.1]


def test_layer_override_cycle_detected():
    with pytest.raises(ConfigError, match="Kreis"):
        build_model({**BASE, "layer_overrides": {"0": {"@von": 1}, "1": {"@von": 0}}})


def test_apply_scope_variants():
    items = list("abcde")
    assert apply_scope(items, "alle", None) == items
    assert apply_scope(items, "nur_auswahl", "0,-1") == ["a", "e"]
    assert apply_scope(items, "nur_einer", "2") == ["c"]
    assert apply_scope(items, "alle_ausser", "::2") == ["b", "d"]
    with pytest.raises(ConfigError):
        apply_scope(items, "nur_einer", "0:2")


def test_tensor_rules_layers_scope_and_copy():
    m = build_model(BASE)
    rules = [
        TensorRuleConfig(target="q_proj.weight", op="freeze", layers="0,-1"),
        TensorRuleConfig(target="o_proj.weight", op={"name": "set", "value": 0.0}, scope="nur_einer", pick="-1"),
        TensorRuleConfig(target="blocks.2.mixer.attn.v_proj.weight", match="exact",
                         op={"name": "copy", "source": "blocks.0.mixer.attn.v_proj.weight"}),
        TensorRuleConfig(target="v_proj.weight", layers="3", index="0:4, :",
                         op={"name": "copy", "source": "blocks.1.mixer.attn.v_proj.weight", "source_index": "4:8, :"}),
    ]
    TensorSurgeon(m, rules).apply_build()
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert frozen == ["blocks.0.mixer.attn.q_proj.weight", "blocks.5.mixer.attn.q_proj.weight"]
    assert not m.blocks[5].mixer.attn.o_proj.weight.any() and m.blocks[4].mixer.attn.o_proj.weight.any()
    assert torch.equal(m.blocks[2].mixer.attn.v_proj.weight, m.blocks[0].mixer.attn.v_proj.weight)
    assert torch.equal(m.blocks[3].mixer.attn.v_proj.weight[:4], m.blocks[1].mixer.attn.v_proj.weight[4:8])


def test_module_rules():
    m = build_model({**BASE, "block": {"dropout": 0.2}})
    rules = [
        ModuleRuleConfig(target="blocks.1", match="exact", op={"name": "copy_from", "source": "blocks.0"}),
        ModuleRuleConfig(module_type="Dropout", op={"name": "set_attr", "attribute": "p", "value": 0.0}),
        ModuleRuleConfig(module_type="RMSNorm", scope="nur_einer", pick="0",
                         op={"name": "replace_norm", "norm": "layernorm"}),
        ModuleRuleConfig(target="ffn", layers="-1", op={"name": "lora", "rank": 4}),
        ModuleRuleConfig(target="q_proj", layers="0", op={"name": "quantize", "format": "int8"}),
        ModuleRuleConfig(target="mixer", layers="::2", op={"name": "scale_output", "factor": 0.0}),
        ModuleRuleConfig(target=r"blocks\.\d+$", match="full", op="record"),
        ModuleRuleConfig(target="blocks.4", match="exact", op="identity"),
    ]
    s = ModuleSurgeon(m, rules).apply()
    assert torch.equal(m.blocks[1].mixer.attn.k_proj.weight, m.blocks[0].mixer.attn.k_proj.weight)
    assert all(mod.p == 0.0 for mod in m.modules() if isinstance(mod, torch.nn.Dropout))
    assert type(m.blocks[0].norm1).__name__ == "LayerNorm" and type(m.blocks[0].norm2).__name__ != "LayerNorm"
    assert any(".lora_" in n for n, p in m.blocks[5].named_parameters() if p.requires_grad)
    assert type(m.blocks[0].mixer.attn.q_proj).__name__ == "QuantLinear"
    assert type(m.blocks[4]).__name__ == "PassThrough"
    out = m(torch.randint(0, 30, (2, 7)))
    out.sum().backward()
    assert "blocks.0" in s.activation_report() and len(s.applied) > 8


def test_wildcard_set_all_and_search():
    p = Project.from_preset("sprachmodell_mini")
    assert "model.block.dropout" in p.find("dropout")
    changed = p.set_all("dropout", 0.15, scope="alle_ausser", pick="-1")
    assert p.get("model.block.dropout") == 0.15 and p.get("memory.lora.dropout") == 0.0
    assert len(changed) == len(p.find("dropout")) - 1
    p.set("model.**.num_heads", 2)
    assert p.get("model.block.mixer.num_heads") == 2
    with pytest.raises(ConfigError):
        p.set_all("dropout", 3.0)  # alles oder nichts
    assert p.get("model.block.dropout") == 0.15
    assert any("kv_compression_dim" in path for path, _ in p.search("latent"))


def test_project_module_rules_train():
    p = Project.from_preset("sprachmodell_mini")
    p.set("training.verbose", False)
    p.set("training.max_steps", 2)
    p.set("module_rules", [{"module_type": "Dropout", "op": {"name": "set_attr", "attribute": "p", "value": 0.0}},
                           {"target": "blocks", "layers": "0", "match": "search", "op": "freeze"}])
    p.train()
    assert p.module_surgeon is not None
    assert not any(t.requires_grad for t in p.model.blocks[0].parameters())
