from __future__ import annotations

import importlib.util
from pathlib import Path

from deep_research.pii_redaction.shared.pii_utils import (
    apply_redaction,
    dataset_path,
    evaluate_example,
    load_dataset,
    normalize_entities,
    regex_candidate_entities,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_pii_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_loads_and_has_control_case():
    dataset = load_dataset(dataset_path())
    assert len(dataset) >= 10
    assert any(example["id"] == "no_pii_control" for example in dataset)


def test_normalize_entities_and_apply_redaction():
    text = "Patient Jane Miller called from 415-555-0199."
    entities = normalize_entities(
        text,
        [
            {"label": "person", "value": "Jane Miller"},
            {"label": "phone", "value": "415-555-0199"},
        ],
    )
    assert [entity.label for entity in entities] == ["PERSON", "PHONE"]
    assert apply_redaction(text, entities) == "Patient [PERSON] called from [PHONE]."


def test_regex_candidates_find_email_and_phone():
    text = "Contact Robert Chen at robert.chen@acme.io or (212) 555-7788."
    candidates = regex_candidate_entities(text)
    labels = {candidate["label"] for candidate in candidates}
    assert "EMAIL" in labels
    assert "PHONE" in labels


def test_evaluate_example_scores_exact_match():
    example = load_dataset(dataset_path())[0]
    prediction = {
        "variant": "test",
        "entities": example["entities"],
        "redacted_text": example["expected_redacted_text"],
    }
    metrics = evaluate_example(example, prediction)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["exact_redaction"] is True


def test_stage_modules_smoke_import():
    paths = [
        ROOT / "deep_research" / "pii_redaction" / "stage_01_basics" / "01_one_shot_prompt_redactor.py",
        ROOT / "deep_research" / "pii_redaction" / "stage_02_agents" / "01_langchain_pii_agent.py",
        ROOT / "deep_research" / "pii_redaction" / "stage_03_langgraph" / "01_pii_redaction_workflow.py",
        ROOT / "deep_research" / "pii_redaction" / "stage_04_deep_agents" / "01_pii_redaction_deep_agent.py",
        ROOT / "deep_research" / "pii_redaction" / "stage_05_evaluation" / "01_benchmark_redaction_variants.py",
    ]
    modules = [_load_module(path) for path in paths]
    assert all(hasattr(module, "__file__") for module in modules)
