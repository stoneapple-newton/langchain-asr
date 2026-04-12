"""
Stage 5: Benchmark PII Redaction Variants
=========================================
CONCEPT: Run the same dataset through each implementation and compare entity
precision / recall / F1 plus exact-redaction rate.

Run this file:
  uv run deep_research/pii_redaction/stage_05_evaluation/01_benchmark_redaction_variants.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.pii_redaction.shared.pii_utils import (
    dataset_path,
    evaluate_example,
    finalize_prediction,
    load_dataset,
    regex_candidate_entities,
    summarize_results,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

VARIANT_MODULES = {
    "one_shot_prompt": ROOT / "stage_01_basics" / "01_one_shot_prompt_redactor.py",
    "langchain_agent": ROOT / "stage_02_agents" / "01_langchain_pii_agent.py",
    "langgraph_workflow": ROOT / "stage_03_langgraph" / "01_pii_redaction_workflow.py",
    "deep_agents": ROOT / "stage_04_deep_agents" / "01_pii_redaction_deep_agent.py",
}


def _load_module(module_path: Path):
    module_name = f"_benchmark_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_regex_baseline(text: str) -> dict[str, Any]:
    return finalize_prediction(
        text,
        regex_candidate_entities(text),
        variant="regex_baseline",
        notes=["Deterministic baseline"],
    )


def run_variant(variant: str, text: str) -> dict[str, Any]:
    if variant == "regex_baseline":
        return run_regex_baseline(text)
    module = _load_module(VARIANT_MODULES[variant])
    return module.redact_text(text)


def benchmark_variant(
    variant: str,
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    results: list[dict[str, Any]] = []
    try:
        for example in examples:
            prediction = run_variant(variant, example["text"])
            metrics = evaluate_example(example, prediction)
            results.append(metrics)
    except Exception as exc:
        return [], str(exc)
    return results, None


def render_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Cases | Precision | Recall | F1 | Exact Redaction | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['cases']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['exact_redaction_rate']:.3f} | "
            f"{row['status']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant to run. Repeat the flag for multiple variants.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional dataset limit.")
    args = parser.parse_args()

    variants = args.variant or [
        "regex_baseline",
        "one_shot_prompt",
        "langchain_agent",
        "langgraph_workflow",
        "deep_agents",
    ]
    examples = load_dataset(dataset_path())
    if args.limit:
        examples = examples[: args.limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    detailed_rows: dict[str, list[dict[str, Any]]] = {}

    for variant in variants:
        case_results, error = benchmark_variant(variant, examples)
        if error:
            summary_rows.append(
                {
                    "variant": variant,
                    "cases": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "exact_redaction_rate": 0.0,
                    "status": f"skipped: {error}",
                }
            )
            continue

        summary = summarize_results(case_results)
        summary_rows.append(
            {
                "variant": variant,
                **summary,
                "status": "ok",
            }
        )
        detailed_rows[variant] = case_results

    summary_markdown = render_summary_table(summary_rows)
    summary_path = OUTPUT_DIR / "benchmark_summary.md"
    detail_path = OUTPUT_DIR / "benchmark_details.json"
    summary_path.write_text(summary_markdown + "\n", encoding="utf-8")
    detail_path.write_text(json.dumps(detailed_rows, indent=2), encoding="utf-8")

    print("=" * 72)
    print("PII REDACTION BENCHMARK")
    print("=" * 72)
    print(summary_markdown)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved details: {detail_path}")


if __name__ == "__main__":
    main()
