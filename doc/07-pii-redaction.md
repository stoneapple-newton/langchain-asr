# PII Redaction Feature

## Feature Summary

The PII redaction track identifies sensitive entities and replaces them with
standard placeholders. It demonstrates one-shot prompting, LangChain agents,
LangGraph, Deep Agents, and benchmark evaluation over a labeled dataset.

## Business Outcome

Developers can compare redaction implementation variants while preserving one
canonical entity contract, redaction function, and evaluation path.

## Scope

Primary folder:

- `deep_research/pii_redaction/`

Important files:

- `shared/pii_utils.py`
- `dataset/pii_redaction_dataset.json`
- `stage_01_basics/01_one_shot_prompt_redactor.py`
- `stage_02_agents/01_langchain_pii_agent.py`
- `stage_03_langgraph/01_pii_redaction_workflow.py`
- `stage_04_deep_agents/01_pii_redaction_deep_agent.py`
- `stage_05_evaluation/01_benchmark_redaction_variants.py`
- `tests/test_pii_redaction_utils.py`

## Current Architecture

`pii_utils.py` defines supported labels, label aliases, Pydantic output models,
entity normalization, redaction application, deterministic regex candidates,
example evaluation, and summary scoring.

The redaction stages should call shared finalization utilities so every
variant emits the same prediction shape.

## Key APIs and Data Contracts

- `SUPPORTED_LABELS`: maps labels to placeholders.
- `PiiEntityModel`: label, exact source value, justification.
- `PiiPredictionModel`: entity list plus notes.
- `ResolvedEntity`: normalized entity with source span.
- `normalize_label(label) -> str | None`
- `normalize_entities(text, raw_entities) -> list[ResolvedEntity]`
- `apply_redaction(text, entities) -> str`
- `finalize_prediction(text, raw_entities, variant, notes=None) -> dict`
- `regex_candidate_entities(text) -> list[dict]`
- `evaluate_example(example, prediction) -> dict`
- `summarize_results(results) -> dict`

## Dependencies

- Shared config for LLM variants.
- Pydantic for structured output.
- LangChain, LangGraph, and optional Deep Agents for variant stages.

## Nonfunctional Requirements

- Entity values must be exact substrings of the source text before redaction.
- Overlapping spans must not corrupt text replacement.
- Redaction should be deterministic after entity normalization.
- Benchmarks should report precision, recall, F1, and exact redaction rate.

## Acceptance Criteria

- New variants call `finalize_prediction()` before returning.
- Unknown labels are normalized through aliases or dropped.
- Duplicate entities are removed.
- Tests cover labels, span resolution, regex candidates, scoring, and imports.
- Benchmark details identify failures without aborting the whole run.

## Risks and Open Questions

- Regex detection is intentionally limited and should be treated as a candidate
  generator, not a complete redaction engine.
- Exact substring matching can miss paraphrased or reformatted entities.
- PII work has higher safety requirements than tutorial code; production usage
  would need stronger validation, audit, and privacy controls.

## Development Guidance

Keep the shared placeholder contract stable. Add new labels only with dataset
examples and tests. Prefer deterministic candidate generation plus LLM
classification over free-form LLM redaction when precision matters.
