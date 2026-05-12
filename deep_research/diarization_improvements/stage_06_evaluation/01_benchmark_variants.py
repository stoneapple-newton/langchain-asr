"""
Stage 6: Benchmark Diarization Variants
========================================
CONCEPT: Run all 30 dataset examples through each implementation and compare
speaker accuracy, F1, defect detection rate, and text preservation across
one-shot prompts, LangChain tools, LangGraph, and Deep Agents.

Run this file:
  uv run deep_research/diarization_improvements/stage_06_evaluation/01_benchmark_variants.py

Flags:
  --variant VARIANT    Run only this variant (repeatable). Default: all.
  --defect  TYPE       Filter examples by defect type (run_on|head_attached|tail_attached).
  --limit   N          Use only the first N examples per defect type.
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

from deep_research.diarization_improvements.shared.diarization_utils import (
    dataset_path,
    evaluate_correction,
    load_dataset,
    summarize_results,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

VARIANT_MODULES: dict[str, Path] = {
    "one_shot_run_on":     ROOT / "stage_02_one_shot" / "01_fix_run_on.py",
    "one_shot_head":       ROOT / "stage_02_one_shot" / "02_fix_head_attached.py",
    "one_shot_tail":       ROOT / "stage_02_one_shot" / "03_fix_tail_attached.py",
    "cot_all_types":       ROOT / "stage_03b_chain_of_thought" / "01_fix_all_cot.py",
    "langchain_tools":     ROOT / "stage_03_langchain_tools" / "01_tool_agent.py",
    "langgraph_pipeline":  ROOT / "stage_04_langgraph" / "01_correction_graph.py",
    "deep_agents_swarm":   ROOT / "stage_05_deep_agents" / "01_diarization_swarm.py",
}

# Which defect type(s) each variant targets (None = handles all types)
VARIANT_DEFECT_FILTER: dict[str, str | None] = {
    "one_shot_run_on":    "run_on",
    "one_shot_head":      "head_attached",
    "one_shot_tail":      "tail_attached",
    "cot_all_types":      None,   # unified prompt handles all three types
    "langchain_tools":    None,
    "langgraph_pipeline": None,
    "deep_agents_swarm":  None,
}


def _load_module(module_path: Path):
    module_name = f"_benchmark_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_variant_on_example(variant: str, example) -> dict[str, Any]:
    """Load the variant module and call correct_transcript on one example."""
    module = _load_module(VARIANT_MODULES[variant])
    result = module.correct_transcript(
        example.input_transcript,
        example.ground_truth_transcript,
    )
    return evaluate_correction(example, result)


def benchmark_variant(
    variant: str,
    examples: list,
) -> tuple[list[dict[str, Any]], str | None]:
    """Run a variant on a list of examples; return (results, error_or_None)."""
    defect_filter = VARIANT_DEFECT_FILTER.get(variant)
    filtered = [ex for ex in examples if defect_filter is None or ex.defect_type == defect_filter]

    results: list[dict[str, Any]] = []
    try:
        for example in filtered:
            metrics = run_variant_on_example(variant, example)
            results.append(metrics)
    except Exception as exc:
        return [], str(exc)
    return results, None


def render_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Variant | Cases | Spk Accuracy | F1 | Detection Rate | Text Preserved | Spurious Removed | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} "
            f"| {row['cases']} "
            f"| {row.get('speaker_accuracy', 0.0):.3f} "
            f"| {row.get('f1', 0.0):.3f} "
            f"| {row.get('defect_detection_rate', 0.0):.3f} "
            f"| {row.get('text_preserved_rate', 0.0):.3f} "
            f"| {row.get('spurious_labels_removed_rate', 0.0):.3f} "
            f"| {row['status']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark diarization correction variants.")
    parser.add_argument(
        "--variant", action="append", default=[],
        help="Variant to run (repeatable). Default: all.",
    )
    parser.add_argument(
        "--defect", choices=["run_on", "head_attached", "tail_attached"],
        default=None, help="Filter examples by defect type.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max examples per defect type.",
    )
    args = parser.parse_args()

    variants = args.variant or list(VARIANT_MODULES.keys())
    examples = load_dataset(dataset_path())

    # Filter by defect type if requested
    if args.defect:
        examples = [ex for ex in examples if ex.defect_type == args.defect]

    # Apply per-type limit
    if args.limit:
        by_type: dict[str, list] = {}
        for ex in examples:
            by_type.setdefault(ex.defect_type, []).append(ex)
        examples = [ex for segs in by_type.values() for ex in segs[:args.limit]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    detailed_rows: dict[str, list[dict[str, Any]]] = {}

    print("=" * 72)
    print("DIARIZATION IMPROVEMENTS BENCHMARK")
    print("=" * 72)
    print(f"Total examples: {len(examples)}")
    print(f"Variants: {', '.join(variants)}")
    print()

    for variant in variants:
        print(f"Running: {variant} ...", end=" ", flush=True)
        case_results, error = benchmark_variant(variant, examples)
        if error:
            print(f"FAILED: {error[:80]}")
            summary_rows.append({
                "variant": variant,
                "cases": 0,
                "speaker_accuracy": 0.0,
                "f1": 0.0,
                "defect_detection_rate": 0.0,
                "text_preserved_rate": 0.0,
                "spurious_labels_removed_rate": 0.0,
                "status": f"error: {error[:60]}",
            })
            continue

        summary = summarize_results(case_results)
        print(f"OK ({summary['cases']} cases, F1={summary['f1']:.3f})")
        summary_rows.append({"variant": variant, **summary, "status": "ok"})
        detailed_rows[variant] = case_results

    print()
    table = render_summary_table(summary_rows)
    print(table)

    summary_path = OUTPUT_DIR / "benchmark_summary.md"
    detail_path = OUTPUT_DIR / "benchmark_details.json"
    summary_path.write_text(table + "\n", encoding="utf-8")
    detail_path.write_text(json.dumps(detailed_rows, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved details: {detail_path}")


if __name__ == "__main__":
    main()
