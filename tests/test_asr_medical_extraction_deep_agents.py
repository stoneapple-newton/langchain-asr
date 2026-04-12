from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "deep_research"
    / "asr"
    / "stage_07_deep_agents"
    / "02_medical_term_extraction_swarm.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_asr_medical_swarm", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_medical_blueprints_count_and_uniqueness():
    module = _load_module()

    blueprints = module.MEDICAL_EXTRACTION_BLUEPRINTS
    assert len(blueprints) == 5
    names = [blueprint["name"] for blueprint in blueprints]
    assert len(names) == len(set(names))


def test_build_subagents_returns_all_medical_specialists():
    module = _load_module()

    shared_tools = module.build_shared_tools()
    subagents = module.build_subagents(shared_tools)

    assert len(subagents) == 5
    assert subagents[0]["tools"] == shared_tools
    assert all(subagent["name"].endswith("_agent") for subagent in subagents)


def test_build_medical_extraction_swarm_requires_deepagents_install():
    module = _load_module()

    try:
        module.build_medical_extraction_swarm()
    except RuntimeError as exc:
        assert "uv add deepagents" in str(exc)
    else:
        raise AssertionError(
            "Expected build_medical_extraction_swarm to require deepagents"
        )


def test_candidate_tools_return_strings_with_sparse_input(tmp_path: Path):
    module = _load_module()

    transcript_path = tmp_path / "medical.json"
    transcript_path.write_text(
        '{"segments": [{"text": "Patient was prescribed amoxicillin 500 mg for sinusitis.", "words": []}]}',
        encoding="utf-8",
    )

    medical = module.medical_term_candidates.invoke(
        {"path": str(transcript_path), "min_occurrences": 1}
    )
    meds = module.medication_name_candidates.invoke(
        {"path": str(transcript_path), "min_occurrences": 1}
    )

    assert isinstance(medical, str)
    assert isinstance(meds, str)
    assert "amoxicillin" in meds.lower()
