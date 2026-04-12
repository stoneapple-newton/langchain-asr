from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TRANSLATION_ROOT = ROOT / "deep_research" / "asr-v2" / "translation"
if str(TRANSLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSLATION_ROOT))

from translation_shared import finalize_translation_result, load_dataset


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_translation_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_modules_smoke_import():
    paths = [
        TRANSLATION_ROOT / "stage_01_basics" / "01_one_shot_translation.py",
        TRANSLATION_ROOT / "stage_02_agents" / "01_langchain_translation_agent.py",
        TRANSLATION_ROOT / "stage_03_langgraph" / "01_translation_workflow.py",
        TRANSLATION_ROOT / "stage_04_deep_agents" / "01_translation_deep_agent.py",
        TRANSLATION_ROOT / "stage_05_evaluation" / "01_benchmark_translation_variants.py",
        TRANSLATION_ROOT / "stage_06_production" / "01_langsmith_translation_observability.py",
    ]
    modules = [_load_module(path) for path in paths]
    assert all(hasattr(module, "__file__") for module in modules)


def test_one_shot_contract(monkeypatch):
    example = load_dataset()[0]
    module = _load_module(TRANSLATION_ROOT / "stage_01_basics" / "01_one_shot_translation.py")

    monkeypatch.setattr(
        module,
        "_run_one_shot_translation",
        lambda _example: {"translated_segments": example["reference_segments"], "notes": ["stub"]},
    )

    result = module.translate_example(example)
    assert result["variant"] == "one_shot_translation"
    assert len(result["translated_segments"]) == len(example["reference_segments"])
    assert result["translated_segments"][0]["speaker"] == example["source_document"]["segments"][0]["speaker"]


def test_langchain_contract(monkeypatch):
    example = load_dataset()[0]
    module = _load_module(TRANSLATION_ROOT / "stage_02_agents" / "01_langchain_translation_agent.py")

    monkeypatch.setattr(
        module,
        "_run_langchain_agent",
        lambda _example: {"translated_segments": example["reference_segments"], "notes": ["stub"]},
    )

    result = module.translate_example(example)
    assert result["variant"] == "langchain_agent"
    assert len(result["translated_segments"]) == len(example["reference_segments"])


def test_langgraph_contract(monkeypatch):
    example = load_dataset()[0]
    module = _load_module(TRANSLATION_ROOT / "stage_03_langgraph" / "01_translation_workflow.py")

    monkeypatch.setattr(
        module,
        "_run_langgraph_workflow",
        lambda _example: finalize_translation_result(
            example,
            example["reference_segments"],
            variant="langgraph_workflow",
            notes=["stub"],
        ),
    )

    result = module.translate_example(example)
    assert result["variant"] == "langgraph_workflow"
    assert len(result["translated_segments"]) == len(example["reference_segments"])


def test_deep_agents_contract(monkeypatch):
    example = load_dataset()[0]
    module = _load_module(TRANSLATION_ROOT / "stage_04_deep_agents" / "01_translation_deep_agent.py")

    monkeypatch.setattr(
        module,
        "_run_deep_agents_translation",
        lambda _example: {"translated_segments": example["reference_segments"], "notes": ["stub"]},
    )

    result = module.translate_example(example)
    assert result["variant"] == "deep_agents"
    assert len(result["translated_segments"]) == len(example["reference_segments"])
