from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TRANSLATION_ROOT = ROOT / "deep_research" / "asr-v2" / "translation"
if str(TRANSLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSLATION_ROOT))

from translation_shared import finalize_translation_result, load_dataset


BENCHMARK_PATH = (
    TRANSLATION_ROOT / "stage_05_evaluation" / "01_benchmark_translation_variants.py"
)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_translation_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_summary_and_details(monkeypatch, tmp_path):
    module = _load_module(BENCHMARK_PATH)
    example = load_dataset()[0]

    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        module,
        "run_variant",
        lambda variant, _example: finalize_translation_result(
            _example,
            _example["reference_segments"],
            variant=variant,
            notes=["stub"],
        ),
    )

    module.main(["--variant", "one_shot_translation", "--limit", "1"])

    summary_path = tmp_path / "benchmark_summary.md"
    detail_path = tmp_path / "benchmark_details.json"
    assert summary_path.exists()
    assert detail_path.exists()
    assert "one_shot_translation" in summary_path.read_text(encoding="utf-8")
    detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
    assert detail_payload["dataset_id"] == example["dataset_id"]


def test_run_benchmark_skips_deep_agents_when_unavailable(monkeypatch):
    module = _load_module(BENCHMARK_PATH)
    examples = load_dataset()[:1]

    def fake_run_variant(variant, example):
        if variant == "deep_agents":
            raise RuntimeError("deepagents is not installed. Install it with `uv add deepagents`.")
        return finalize_translation_result(
            example,
            example["reference_segments"],
            variant=variant,
            notes=["stub"],
        )

    monkeypatch.setattr(module, "run_variant", fake_run_variant)

    summary_rows, detail_payload = module.run_benchmark(
        ["deep_agents", "one_shot_translation"],
        examples,
    )

    deep_row = next(row for row in summary_rows if row["variant"] == "deep_agents")
    one_shot_row = next(row for row in summary_rows if row["variant"] == "one_shot_translation")

    assert deep_row["status"].startswith("skipped")
    assert one_shot_row["status"] == "ok"
    assert detail_payload["variants"]["deep_agents"]["failures"][0]["status"] == "skipped"


def test_benchmark_reports_failures_without_aborting(monkeypatch):
    module = _load_module(BENCHMARK_PATH)
    examples = load_dataset()[:2]

    def fake_run_variant(variant, example):
        if variant == "langchain_agent" and example["id"] == examples[0]["id"]:
            raise ValueError("synthetic failure")
        return finalize_translation_result(
            example,
            example["reference_segments"],
            variant=variant,
            notes=["stub"],
        )

    monkeypatch.setattr(module, "run_variant", fake_run_variant)

    summary_rows, detail_payload = module.run_benchmark(
        ["langchain_agent", "one_shot_translation"],
        examples,
    )

    langchain_row = next(row for row in summary_rows if row["variant"] == "langchain_agent")
    one_shot_row = next(row for row in summary_rows if row["variant"] == "one_shot_translation")

    assert langchain_row["status"].startswith("partial")
    assert langchain_row["failed_cases"] == 1
    assert one_shot_row["status"] == "ok"
    assert detail_payload["variants"]["langchain_agent"]["failures"][0]["error"] == "synthetic failure"
