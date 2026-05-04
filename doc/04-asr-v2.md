# ASR-v2 Feature

## Feature Summary

ASR-v2 is a cleaner transcript post-processing track for WhisperX-like JSON. It
normalizes transcripts into dataclasses, performs deterministic diarization and
readability cleanup, adds LLM-backed readability editing, builds LangGraph
workflows, and provides a batch runner.

## Business Outcome

ASR-v2 should be the preferred foundation for new transcript cleanup code. It
has clearer data contracts, shared utilities, tests, and output conventions.

## Scope

Primary folder:

- `deep_research/asr-v2/`

Important files:

- `shared/transcript_utils.py`
- `shared/pipelines.py`
- `shared/transcript_diff.py`
- `shared/comparison_utils.py`
- `stage_01_basics/*`
- `stage_02_tools/*`
- `stage_03_langgraph/*`
- `stage_04_agent/*`
- `stage_05_production/01_batch_asr_quality_runner.py`
- `tests/*`

## Current Architecture

`shared/transcript_utils.py` defines the canonical ASR-v2 transcript data model:

- `WordToken`
- `TranscriptSegment`
- `TranscriptDocument`

It also provides loading, analysis, diarization repair, readability cleanup,
Markdown rendering, output writing, chunking, and document rebuilding.

`shared/pipelines.py` provides a small reusable cleanup graph and a rule-based
cleanup function.

`shared/transcript_diff.py` compares two transcript documents by segment ID and
categorizes differences.

`shared/comparison_utils.py` supports IoU-based alignment, WER, deterministic
diff classification, document summaries, and Markdown reports. It is also used
by `deep_research/asr_comparison`.

## Key APIs and Data Contracts

- `load_transcript(path) -> TranscriptDocument`
- `analyze_transcript(doc) -> dict`
- `repair_diarization(doc, max_gap_seconds=0.75) -> TranscriptDocument`
- `improve_readability(doc) -> TranscriptDocument`
- `save_enhanced_outputs(doc, source_path, output_dir) -> dict[str, str]`
- `chunk_segments(doc, max_chars=900) -> list[list[TranscriptSegment]]`
- `run_rule_based_cleanup(input_path, output_dir) -> dict`
- `build_diarization_cleanup_graph()`
- `build_comparison_report(reference_path, candidate_path) -> dict`

The output convention is `outputs/<input-stem>/<input-stem>.enhanced.json` and
`outputs/<input-stem>/<input-stem>.transcript.md`.

## Dependencies

- Shared config for LLM-backed stages.
- LangGraph for stage 3.
- Pydantic for structured LLM output in readability editing.
- ASR-v2 sample data and tests.

## Nonfunctional Requirements

- Preserve source path, language, metadata, timestamps, and word tokens unless
  intentionally changing the segment structure.
- Use deterministic repair where possible before invoking an LLM.
- Keep LLM edits segment-count preserving.
- Save both machine-readable JSON and human-readable Markdown.

## Acceptance Criteria

- New cleanup logic uses `TranscriptDocument` and `TranscriptSegment`.
- Structural changes are reflected in analysis before and after.
- Tests cover deterministic helpers.
- Output paths are returned to callers rather than only printed.
- LLM-backed stages have deterministic fallback behavior.

## Risks and Open Questions

- `repair_diarization()` mutates merged segment objects while building the new
  document. It returns a new document, but callers should still treat input
  segment objects as not fully isolated.
- `transcript_diff.py` compares by segment ID, while `comparison_utils.py`
  aligns by time IoU. Choose the correct tool based on whether IDs are stable.
- Some ASR-v2 stage scripts rely on `sys.path` injection for `shared` imports.

## Development Guidance

For new ASR cleanup features, start in `shared/transcript_utils.py` only if the
logic is broadly reusable. Otherwise keep feature-specific logic in a stage file
and add tests before promoting it to shared. For comparisons, use ID-based diff
for same-lineage transcripts and IoU alignment for transcripts from different
ASR systems.
