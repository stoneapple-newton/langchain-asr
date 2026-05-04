# Codebase Overview

## Epic Summary

This repository is an educational LangChain, LangGraph, and Deep Agents
workspace. It demonstrates how to build LLM systems progressively: basic
chains, tools, graphs, RAG, memory, evaluation, and multi-agent orchestration.
The applied domains include deep research, ASR cleanup, transcript comparison,
translation, summarization, PII redaction, diarization repair, and medical ASR
verification.

## Business Outcome

The expected outcome is developer enablement. A developer should be able to
copy a local pattern, choose the right framework layer, and extend a feature
track without rewriting provider setup or relearning transcript data contracts.

## Scope

In scope:

- Shared provider-aware configuration in `config/`.
- Staged tutorials under `deep_research/`.
- Deterministic helpers for transcript parsing, evaluation, redaction,
  comparison, and correction.
- LangChain chains, tool agents, LangGraph workflows, and Deep Agents examples.
- Local sample datasets and benchmark runners.

Out of scope:

- A single deployable production application.
- Unified packaging for every stage module.
- Stable external API compatibility guarantees.
- Full end-to-end testing for every LLM-backed script.

## Current Architecture

The repository is organized around feature tracks:

- `config/`: shared settings and provider factories.
- `deep_research/deep_research_agent/`: general LangChain learning path.
- `deep_research/asr/`: original ASR quality progression.
- `deep_research/asr-v2/`: cleaner ASR post-processing track.
- `deep_research/asr-v2/translation/`: segment-preserving translation track.
- `deep_research/summary_agents/`: transcript summarization progression.
- `deep_research/pii_redaction/`: PII redaction variants and benchmark.
- `deep_research/diarization_improvements/`: focused diarization repair.
- `deep_research/asr_comparison/`: ASR transcript diffing.
- `deep_research/asr_medical_verification/`: medical ASR verification helpers.
- `deep_research/transcription_correction_agent/`: LangGraph correction agent.
- `deep_research/medical_asr_context_system/`: medical context storage and
  retrieval experiments.
- `deep_research/medical_term_extraction/`: medical term extraction examples.
- `tests/` and track-local `tests/`: deterministic and smoke test coverage.

Most scripts are runnable examples. Shared helpers are the stronger reuse
points and should be preferred for new code.

## Key Design Principles

- Provider-agnostic scripts use `create_chat_model()` and `create_embeddings()`.
- Deterministic utilities are tested more thoroughly than LLM calls.
- Stages compare complexity levels for the same task.
- LangGraph is used when state, routing, fan-out, or iterative refinement
  matter.
- Deep Agents examples are optional and guarded in tests where possible.

## Dependencies

Core runtime dependencies are declared in `pyproject.toml` and installed with
`uv sync`. Important packages include LangChain, LangGraph, LangSmith,
LangChain provider integrations, Chroma, FastAPI, Uvicorn, and Pydantic
Settings.

The repository targets Python 3.13 or newer. Some examples require external
services or credentials: Ollama, OpenAI, OpenRouter, Azure OpenAI, Tavily, and
LangSmith.

## Nonfunctional Requirements

- Examples should remain understandable as teaching material.
- Shared helpers should remain deterministic and testable.
- Provider selection should stay in configuration, not in tutorial scripts.
- Generated outputs should be written under local `outputs/` folders.
- Tests should avoid real network and real model calls unless explicitly marked.

## Acceptance Criteria

- New feature tracks have staged scripts and a shared helper layer where reuse
  is expected.
- New provider usage goes through `config/`.
- New data contracts are represented by dataclasses, Pydantic models, or typed
  dictionaries.
- New deterministic helpers include focused tests.
- Documentation identifies run commands, dependencies, and extension points.

## Risks and Open Questions

- Several scripts mutate `sys.path` to import sibling shared modules. This is
  pragmatic for tutorials, but fragile for reuse.
- Some feature tracks duplicate transcript models and helper logic.
- Some LLM chains are created at module import time, which can make imports
  fail without local provider services.
- Optional packages are sometimes handled by runtime fallbacks instead of formal
  dependency extras.
- There is no single CLI or package namespace that unifies all tracks.

## Development Guidance

Use the shared `config/` package first. For transcript work, prefer the ASR-v2
data model unless a feature-specific model is required. For graph work, keep
state schemas explicit and isolate pure helpers so tests can run without model
calls. For benchmark work, persist both summary and detail outputs so variant
regressions are explainable.
