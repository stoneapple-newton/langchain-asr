# Transcription Correction Agent Feature

## Feature Summary

The transcription correction agent is a LangGraph workflow that loads a
WhisperX-style transcript, detects general transcription errors and medical
term candidates, fans out segment workers, retranscribes or text-corrects
segments, links medical terms, synthesizes corrections, and writes corrected
JSON and Markdown outputs.

## Business Outcome

This feature demonstrates a production-shaped correction agent with explicit
state, worker fan-out, correction records, and human-readable output artifacts.

## Scope

Primary folder:

- `deep_research/transcription_correction_agent/`

Important files:

- `agent.py`
- `run.py`
- `state.py`
- `tools/audio_extraction.py`
- `tools/retranscription.py`
- `tools/medical_terms.py`
- `data/sample_transcript.json`
- `data/medical_glossary.json`
- `README.md`

## Current Architecture

`agent.py` builds a singleton compiled graph named `correction_graph`.

Graph nodes:

- `load`
- `detect_errors`
- `detect_medical`
- `error_worker`
- `medical_worker`
- `synthesize`
- `save_output`

The graph uses one conditional fan-out from `detect_medical` so both error and
medical workers are dispatched together and `synthesize` is reached once after
workers finish.

`run.py` provides the CLI wrapper, sample transcript creation, and output
summary printing.

`state.py` defines `TypedDict` contracts for main state, worker state,
candidates, linked medical terms, and correction records.

## Key APIs and Data Contracts

- `AgentState`: transcript path, audio path, method, segments, glossary,
  candidates, corrections, corrected segments, stage log, output paths.
- `ErrorCandidate`: segment ID, timing, text, speaker, issue type, reason,
  context.
- `MedicalCandidate`: segment ID, timing, text, speaker, suspected term,
  specialty, context.
- `CorrectionRecord`: segment ID, original text, corrected text, method,
  correction type, confidence, medical terms, linked terms.
- `build_correction_graph()`
- `correction_graph.invoke(state)`

## Dependencies

- Shared config default chat profile.
- LangGraph and `Send`.
- LangChain prompts and string parser.
- ffmpeg or equivalent audio extraction support.
- Optional faster-whisper and OpenAI audio provider depending on method.
- Local medical glossary data.

## Nonfunctional Requirements

- Worker outputs must accumulate through LangGraph reducers.
- Corrections must be traceable in JSON and Markdown.
- Audio unavailable paths should fall back to text correction.
- Medical-term corrections should override general corrections for the same
  segment.

## Acceptance Criteria

- CLI can run against a transcript path and optional audio path.
- Graph produces corrected JSON and Markdown under an `outputs/` folder.
- Stage log records load, detection, synthesis, and save actions.
- Correction records contain enough metadata to audit applied changes.

## Risks and Open Questions

- `_llm` is created at module import time in `agent.py`, so import can fail if
  local provider configuration is unavailable.
- The agent imports sibling modules through modified `sys.path` and unqualified
  imports such as `from state import ...`, which can collide with other modules
  in larger applications.
- Detection uses free-form LLM JSON strings and manual cleanup rather than the
  shared structured output helper.
- Confidence values from text-only fallback are model-derived and should not be
  treated as calibrated probabilities.

## Development Guidance

For new work, avoid creating model clients at import time. Prefer lazy graph
construction that accepts an LLM or provider profile. If this agent becomes a
library component, convert local imports to package-relative imports and add
tests around graph construction, fan-out, synthesis precedence, and output
paths.
