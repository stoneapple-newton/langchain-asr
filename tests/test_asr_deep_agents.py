from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "deep_research"
    / "asr"
    / "stage_07_deep_agents"
    / "01_asr_quality_swarm.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_asr_quality_swarm", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asr_deep_agent_blueprints_count_and_uniqueness():
    module = _load_module()

    blueprints = module.ASR_DEEP_AGENT_BLUEPRINTS
    assert len(blueprints) == 20
    names = [blueprint["name"] for blueprint in blueprints]
    assert len(names) == len(set(names))


def test_build_subagents_returns_all_specialists():
    module = _load_module()

    shared_tools = module.build_shared_tools()
    subagents = module.build_subagents(shared_tools)

    assert len(subagents) == 20
    assert subagents[0]["tools"] == shared_tools
    assert all(subagent["name"].endswith("_agent") for subagent in subagents)


def test_build_swarm_requires_deepagents_install():
    module = _load_module()

    try:
        module.build_asr_quality_swarm()
    except RuntimeError as exc:
        assert "uv add deepagents" in str(exc)
    else:
        raise AssertionError("Expected build_asr_quality_swarm to require deepagents")
