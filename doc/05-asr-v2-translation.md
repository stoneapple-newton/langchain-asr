# ASR-v2 Translation Feature

## Feature Summary

The ASR-v2 translation track translates Mandarin Chinese ASR transcripts into
English while preserving segment count, speaker labels, and timestamps. It
compares one-shot, LangChain agent, LangGraph, Deep Agents, evaluation, and
LangSmith observability variants.

## Business Outcome

Developers can build translation workflows that keep transcript structure
stable enough for downstream subtitles, speaker attribution, audit, and
benchmarking.

## Scope

Primary folder:

- `deep_research/asr-v2/translation/`

Important files:

- `shared/translation_utils.py`
- `dataset/translation_dataset.json`
- `stage_01_basics/01_one_shot_translation.py`
- `stage_02_agents/01_langchain_translation_agent.py`
- `stage_03_langgraph/01_translation_workflow.py`
- `stage_04_deep_agents/01_translation_deep_agent.py`
- `stage_05_evaluation/01_benchmark_translation_variants.py`
- `stage_06_production/01_langsmith_translation_observability.py`

## Current Architecture

`translation_utils.py` is the central helper module. It dynamically loads the
ASR-v2 transcript dataclasses, loads dataset examples, builds segment-preserving
prompts, validates output structure, finalizes predictions, evaluates against
reference translations, summarizes benchmark rows, and writes outputs.

Stages implement the same contract through different framework layers:

- One-shot structured output.
- LangChain agent with inspection tools.
- LangGraph prepare, draft, review, finalize workflow.
- Deep Agents translation specialist.
- Benchmark runner across variants.
- LangSmith tracing configuration.

## Key APIs and Data Contracts

- `TranslationDraftModel`: list of translated segment strings plus notes.
- `load_dataset() -> list[dict]`
- `load_source_document(payload) -> TranscriptDocument`
- `translation_rules() -> str`
- `format_segment_preserving_prompt(doc) -> str`
- `validate_segment_preservation(source_document, translated_segments) -> dict`
- `finalize_translation_result(example, translated_texts, variant, notes=None)`
- `evaluate_translation(example, prediction) -> dict`
- `summarize_results(rows) -> dict`
- `save_translation_outputs(example, prediction, output_dir) -> dict`

Prediction contract:

- `variant`
- `translated_segments`
- `notes`
- optional `trace_metadata`

Each translated segment preserves `id`, `start`, `end`, and `speaker` from the
source segment.

## Dependencies

- ASR-v2 transcript utilities.
- Shared config.
- LangChain agents and tools.
- LangGraph.
- Optional Deep Agents package.
- LangSmith for tracing and observability examples.

## Nonfunctional Requirements

- Segment count must be preserved exactly.
- Speaker labels and timestamps must be preserved exactly.
- Evaluation should report both translation quality proxies and structure
  preservation.
- Benchmarks should continue running when a variant fails and record the
  failure.

## Acceptance Criteria

- A translation variant exposes `translate_example(example) -> dict`.
- `finalize_translation_result()` validates structure before returning output.
- Benchmark output includes summary and details.
- Tests cover dataset loading, prompt formatting, structural validation,
  scoring, output writing, and variant contracts.

## Risks and Open Questions

- Token overlap and exact match are useful deterministic signals, but they are
  weak translation-quality metrics.
- Dynamic import of ASR-v2 utilities avoids package naming issues with
  `asr-v2`, but it is less idiomatic than a package-safe module name.
- Deep Agents support is optional and may not be installed by default.

## Development Guidance

When adding a translation variant, implement the same `translate_example`
contract and route final output through `finalize_translation_result()`. Do not
write custom structure-preservation logic in each stage. Add benchmark coverage
for failure behavior if the variant calls optional packages or external models.
