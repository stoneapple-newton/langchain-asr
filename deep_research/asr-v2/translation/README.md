# ASR-v2 Chinese to English Translation Track

This track adds a parallel, segment-preserving translation workflow under
`deep_research/asr-v2/translation/` without changing the existing cleanup or
readability stages.

## Scope

- Source language: Chinese (`zh`)
- Target language: English (`en`)
- Output shape: preserve segment count, timestamps, and speaker labels exactly
- Dataset source of truth: local JSON in `dataset/translation_dataset.json`
- Tracing and comparisons: LangSmith when tracing env vars are configured

## Environment

Install project dependencies first:

```powershell
uv sync
```

Provider requirements:

- The one-shot, LangChain, and LangGraph stages use `create_chat_model("asr_v2", ...)`
- Configure `.env` so the `asr_v2` chat profile points at a reachable provider
- `deepagents` is optional for stage 4:

```powershell
uv add deepagents
```

LangSmith tracing is optional but supported in the benchmark and observability
examples. Set either the nested tracing vars or the legacy aliases:

```powershell
TRACING__ENABLED=true
TRACING__API_KEY=<your_langsmith_key>
TRACING__PROJECT=asr-v2-translation
```

## Run Order

```powershell
uv run deep_research/asr-v2/translation/stage_01_basics/01_one_shot_translation.py
uv run deep_research/asr-v2/translation/stage_02_agents/01_langchain_translation_agent.py
uv run deep_research/asr-v2/translation/stage_03_langgraph/01_translation_workflow.py
uv run deep_research/asr-v2/translation/stage_04_deep_agents/01_translation_deep_agent.py
uv run deep_research/asr-v2/translation/stage_05_evaluation/01_benchmark_translation_variants.py
uv run deep_research/asr-v2/translation/stage_06_production/01_langsmith_translation_observability.py
```

## Benchmark

Run all variants plus the deterministic baseline:

```powershell
uv run deep_research/asr-v2/translation/stage_05_evaluation/01_benchmark_translation_variants.py
```

Useful flags:

```powershell
uv run deep_research/asr-v2/translation/stage_05_evaluation/01_benchmark_translation_variants.py --limit 2
uv run deep_research/asr-v2/translation/stage_05_evaluation/01_benchmark_translation_variants.py --variant one_shot_translation --variant langgraph_workflow
```

Outputs are written under `translation/outputs/`:

- `benchmark_summary.md`
- `benchmark_details.json`
- per-example translated JSON / Markdown artifacts

## What Is Scored

Local deterministic metrics:

- segment exact-match rate
- full-transcript exact-match rate
- normalized token precision / recall / F1
- structure-preservation checks for segment count, speakers, and timestamps

The benchmark includes a `copy_source_baseline` row so model-backed variants can
be compared against a deterministic floor even when an LLM stage fails.

## Tests

```powershell
uv run --with pytest pytest tests/test_asr_translation_utils.py tests/test_asr_translation_variants.py tests/test_asr_translation_benchmark.py tests/test_asr_translation_deep_agents.py -q
```

These tests cover dataset loading, scoring, output validation, variant
contracts, deepagents optional handling, benchmark behavior, and smoke imports.
