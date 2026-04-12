"""
Stage 5: Benchmark Translation Variants
=======================================
CONCEPT: Evaluate one-shot prompting, LangChain, LangGraph, Deep Agents, and a
deterministic baseline against the same labeled zh -> en dataset.

Run this file:
  uv run deep_research/asr-v2/translation/stage_05_evaluation/01_benchmark_translation_variants.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_settings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_shared import (
    copy_source_baseline,
    evaluate_translation,
    load_dataset,
    load_dataset_bundle,
    save_translation_outputs,
    summarize_results,
)


settings = get_settings()
for key, value in settings.tracing.to_langsmith_env(
    default_enabled=True,
    default_project="asr-v2-translation-benchmark",
).items():
    os.environ.setdefault(key, value)


OUTPUT_DIR = ROOT / "outputs"

VARIANT_MODULES: dict[str, Path] = {
    "one_shot_translation": ROOT / "stage_01_basics" / "01_one_shot_translation.py",
    "langchain_agent": ROOT / "stage_02_agents" / "01_langchain_translation_agent.py",
    "langgraph_workflow": ROOT / "stage_03_langgraph" / "01_translation_workflow.py",
    "deep_agents": ROOT / "stage_04_deep_agents" / "01_translation_deep_agent.py",
}


def _load_module(module_path: Path):
    module_name = f"_translation_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_variant(variant: str, example: dict[str, Any]) -> dict[str, Any]:
    if variant == "copy_source_baseline":
        return copy_source_baseline(example)
    module = _load_module(VARIANT_MODULES[variant])
    return module.translate_example(example)


def benchmark_variant(
    variant: str,
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for example in examples:
        try:
            prediction = run_variant(variant, example)
            results.append(evaluate_translation(example, prediction))
            save_translation_outputs(example, prediction, OUTPUT_DIR / "examples")
        except RuntimeError as exc:
            message = str(exc)
            if variant == "deep_agents" and "uv add deepagents" in message:
                failures.append(
                    {
                        "id": example["id"],
                        "variant": variant,
                        "status": "skipped",
                        "error": message,
                    }
                )
                return [], failures, "skipped"
            failures.append(
                {
                    "id": example["id"],
                    "variant": variant,
                    "status": "error",
                    "error": message,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "id": example["id"],
                    "variant": variant,
                    "status": "error",
                    "error": str(exc),
                }
            )

    if failures and results:
        return results, failures, "partial"
    if failures and not results:
        return results, failures, "error"
    return results, failures, "ok"


def render_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Cases | Seg Exact | Full Exact | Token P | Token R | Token F1 | Structure | Failed | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['cases']} | "
            f"{row.get('segment_exact_match_rate', 0.0):.3f} | "
            f"{row.get('full_transcript_exact_match_rate', 0.0):.3f} | "
            f"{row.get('token_precision', 0.0):.3f} | "
            f"{row.get('token_recall', 0.0):.3f} | "
            f"{row.get('token_f1', 0.0):.3f} | "
            f"{row.get('structure_preserved_rate', 0.0):.3f} | "
            f"{row.get('failed_cases', 0)} | {row['status']} |"
        )
    return "\n".join(lines)


def run_benchmark(
    variants: list[str],
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    detail_payload: dict[str, Any] = {"variants": {}}

    for variant in variants:
        case_results, failures, status = benchmark_variant(variant, examples)
        summary = summarize_results(case_results)

        if status == "skipped":
            status_text = "skipped: deepagents unavailable"
        elif status == "partial":
            status_text = f"partial: {len(failures)} failure(s)"
        elif status == "error":
            status_text = f"error: {len(failures)} failure(s)"
        else:
            status_text = "ok"

        summary_rows.append(
            {
                "variant": variant,
                **summary,
                "failed_cases": len(failures),
                "status": status_text,
            }
        )
        detail_payload["variants"][variant] = {
            "results": case_results,
            "failures": failures,
        }

    return summary_rows, detail_payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark zh -> en translation variants.")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant to run. Repeat for multiple variants.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional dataset limit.")
    args = parser.parse_args(argv)

    bundle = load_dataset_bundle()
    examples = load_dataset()
    if args.limit:
        examples = examples[: args.limit]

    variants = args.variant or [
        "copy_source_baseline",
        "one_shot_translation",
        "langchain_agent",
        "langgraph_workflow",
        "deep_agents",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, detail_payload = run_benchmark(variants, examples)
    detail_payload["dataset_id"] = bundle["dataset_id"]

    summary_markdown = render_summary_table(summary_rows)
    summary_path = OUTPUT_DIR / "benchmark_summary.md"
    detail_path = OUTPUT_DIR / "benchmark_details.json"
    summary_path.write_text(summary_markdown + "\n", encoding="utf-8")
    detail_path.write_text(json.dumps(detail_payload, indent=2), encoding="utf-8")

    print("=" * 80)
    print("ASR-V2 TRANSLATION BENCHMARK")
    print("=" * 80)
    print(f"Dataset: {bundle['dataset_id']}")
    print(summary_markdown)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved details: {detail_path}")


if __name__ == "__main__":
    main()
