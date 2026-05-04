# ASR Quality Feature

## Feature Summary

The ASR quality track demonstrates progressive improvement of automatic speech
recognition transcripts. It covers loading, metrics, readability cleanup,
speaker and diarization repair, LangGraph workflows, RAG grounding, production
agents, and Deep Agents swarms.

## Business Outcome

Developers can understand the end-to-end ASR improvement problem and choose the
right implementation level for a new correction feature: deterministic utility,
LLM chain, graph, RAG workflow, or multi-agent system.

## Scope

Primary folder:

- `deep_research/asr/`

Important subfolders:

- `stage_01_basics`
- `stage_02_llm_enhancement`
- `stage_03_diarization`
- `stage_04_langgraph_pipeline`
- `stage_05_rag_context`
- `stage_06_production_agent`
- `stage_07_deep_agents`
- `sample_data`
- `tests`

## Current Architecture

The track begins with transcript parsing and metrics, then layers in cleanup
and graph workflows. Stage 4 introduces graph state and iterative quality loops.
Stage 5 uses contextual retrieval to ground corrections. Stage 6 builds a
production-style agent with quality targets and exports. Stage 7 shows Deep
Agents specialists for ASR quality review.

The deterministic helper patterns are tested in `deep_research/asr/tests`.
These tests load stage modules and verify functions such as quality metrics,
diarization windowing, suspicious segment detection, threshold routing, and
output application.

## Key APIs and Data Contracts

- Transcript segments are represented as dictionaries in older ASR examples.
- Quality metrics include average confidence, filler rate, punctuation score,
  capitalization score, segment counts, and word counts.
- Graph states are `TypedDict` objects with segment lists, quality reports,
  rounds, and output paths.
- Deep Agents stages define specialist blueprints and tool wrappers.

## Dependencies

- Shared config layer for LLM calls.
- LangGraph for stateful stages.
- LangChain prompts, parsers, and tools.
- RAG dependencies for contextual correction examples.
- Optional Deep Agents package for swarm examples.

## Nonfunctional Requirements

- Deterministic metrics should be testable without model calls.
- Cleanup should preserve timestamps and speaker labels unless explicitly
  repairing diarization.
- Iterative workflows should have round limits.
- Exports should make before and after quality visible.

## Acceptance Criteria

- A new ASR quality stage identifies its input and output transcript shape.
- Corrections are auditable through logs or output metadata.
- Quality thresholds are explicit and tested where deterministic.
- RAG-backed corrections include evidence or retrieved context.
- Optional Deep Agents examples fail gracefully when the package is unavailable.

## Risks and Open Questions

- The original ASR track predates ASR-v2 and uses looser dictionary contracts.
- Some scripts execute model setup at import time.
- There is duplicated transcript handling between ASR and ASR-v2.
- Quality scores are useful teaching signals but not calibrated production
  metrics.

## Development Guidance

Prefer ASR-v2 shared dataclasses for new reusable transcript logic. Use this
original ASR track for examples of staged teaching, graph loops, and quality
heuristics. If adding a production-like correction, write deterministic tests
for parsing, routing, thresholding, and output application before adding LLM
behavior.
