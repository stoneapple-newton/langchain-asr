# Diarization Improvements Feature

## Feature Summary

The diarization improvements track focuses on common speaker-attribution
defects: run-on speaker turns, head-attached text, tail-attached text, graph
correction, Deep Agents swarms, and benchmarking.

## Business Outcome

Developers can improve speaker segmentation quality while measuring whether a
new repair strategy actually detects and fixes known diarization defect types.

## Scope

Primary folder:

- `deep_research/diarization_improvements/`

Important folders:

- `stage_01_dataset_generator/`
- `stage_02_one_shot/`
- `stage_03_langchain_tools/`
- `stage_04_langgraph/`
- `stage_05_deep_agents/`
- `stage_06_evaluation/`
- `shared/`
- `dataset/`
- `tests/`

## Current Architecture

The track starts by generating or loading a defect dataset. One-shot scripts
show direct correction prompts for specific defect types. Tool-agent and
LangGraph stages add explicit analysis, correction, and quality scoring.
Deep Agents stages split diarization work across specialists. Evaluation
compares variants on dataset cases.

Shared utilities in `shared/diarization_utils.py` support dataset examples,
defect analysis, repair helpers, and scoring.

## Key APIs and Data Contracts

- Dataset examples include input transcript, expected output, defect type, and
  metadata.
- Defect types include run-on, head-attached, and tail-attached cases.
- Graph state includes input transcript, candidate repairs, attempts, quality
  score, notes, and final segments.
- Benchmark output includes per-variant summary and detailed results.

## Dependencies

- Shared config for LLM-backed variants.
- LangChain tools for tool-agent stage.
- LangGraph for correction graph stage.
- Optional Deep Agents package.
- Local dataset JSON.

## Nonfunctional Requirements

- Speaker labels should only change when justified by a defect.
- Segment count and text boundaries may change, but changes must be auditable.
- Graph correction loops should have attempt limits.
- Benchmarks should produce comparable metrics across variants.

## Acceptance Criteria

- New repair variants identify supported defect types.
- Dataset cases include expected outputs.
- Evaluation computes detection and correction quality.
- Tests cover shared utilities, dataset generator behavior, one-shot utilities,
  and LangGraph correction behavior.

## Risks and Open Questions

- Some examples catch broad exceptions to keep demos moving; production code
  would need narrower failure handling.
- LLM correction quality may vary across providers and prompts.
- Dataset coverage determines how meaningful benchmark scores are.

## Development Guidance

Use this track for speaker-attribution experiments. Keep new defect categories
small, add synthetic dataset cases, and update benchmark scoring before adding
LLM variants.
