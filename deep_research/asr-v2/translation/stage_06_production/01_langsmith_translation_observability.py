"""
Stage 6: LangSmith Translation Observability
============================================
CONCEPT: Trace translation runs with LangSmith while keeping the local JSON
dataset as the source of truth for evaluation.

Run this file:
  uv run deep_research/asr-v2/translation/stage_06_production/01_langsmith_translation_observability.py
"""

from __future__ import annotations

import importlib.util
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

from translation_shared import evaluate_translation, load_dataset, save_translation_outputs


settings = get_settings()
for key, value in settings.tracing.to_langsmith_env(
    default_enabled=True,
    default_project="asr-v2-translation-observability",
).items():
    os.environ.setdefault(key, value)


VARIANT_MODULES: dict[str, Path] = {
    "one_shot_translation": ROOT / "stage_01_basics" / "01_one_shot_translation.py",
    "langgraph_workflow": ROOT / "stage_03_langgraph" / "01_translation_workflow.py",
}


def _load_module(module_path: Path):
    module_name = f"_translation_obs_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from langsmith import Client, traceable

    @traceable(name="translation_observability_demo", tags=["translation", "observability"])
    def compare_variants(example: dict[str, Any]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for variant, path in VARIANT_MODULES.items():
            module = _load_module(path)
            prediction = module.translate_example(example)
            metrics = evaluate_translation(example, prediction)
            artifacts = save_translation_outputs(example, prediction, ROOT / "outputs" / "observability")
            outputs[variant] = {"metrics": metrics, "artifacts": artifacts}
        return outputs

    example = load_dataset()[0]
    outputs = compare_variants(example)

    print("=" * 80)
    print("LANGSMITH TRANSLATION OBSERVABILITY")
    print("=" * 80)
    print(f"LANGSMITH_TRACING={os.environ.get('LANGSMITH_TRACING', 'not set')}")
    print(f"LANGSMITH_PROJECT={os.environ.get('LANGSMITH_PROJECT', 'not set')}")
    print(f"Dataset example: {example['id']}")
    print()

    for variant, payload in outputs.items():
        metrics = payload["metrics"]
        print(
            f"{variant}: token_f1={metrics['token_f1']:.3f}, "
            f"segment_exact={metrics['segment_exact_match_rate']:.3f}, "
            f"structure={metrics['structure_preserved']}"
        )
        print(f"  JSON: {payload['artifacts']['json_path']}")
        print(f"  MD:   {payload['artifacts']['markdown_path']}")

    if settings.tracing.api_key:
        client = Client()
        print("\nLangSmith tracing is enabled. Check the configured project in the LangSmith UI.")
        print(f"Client type: {type(client).__name__}")
    else:
        print(
            "\nSet TRACING__API_KEY or LANGSMITH_API_KEY to view traces in LangSmith. "
            "The local dataset remains the source of truth for scoring."
        )


if __name__ == "__main__":
    main()
