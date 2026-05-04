# Medical ASR Verification Feature

## Feature Summary

Medical ASR verification detects potential transcription errors and medical
term issues, optionally extracts audio clips, retranscribes questionable
segments, links recognized medical terms to a multilingual glossary, applies
accepted corrections, and writes an audit sidecar.

## Business Outcome

Developers can build medical-domain ASR correction workflows with traceable
term linking and confidence-gated correction, rather than unstructured free-form
transcript rewriting.

## Scope

Primary folder:

- `deep_research/asr_medical_verification/`

Important files:

- `workflow.py`
- `state.py`
- `shared/transcript_utils.py`
- `shared/detection.py`
- `shared/medical_glossary.py`
- `shared/retranscription.py`
- `shared/audio_utils.py`
- `stage_01_error_detection/*`
- `stage_02_audio_extraction/*`
- `stage_03_transcription_verification/*`
- `stage_04_medical_terms/*`
- `stage_05_deep_agent/*`
- `tests/test_asr_medical_verification.py`

Related folders:

- `deep_research/medical_asr_context_system/`
- `deep_research/medical_term_extraction/`

## Current Architecture

`workflow.py` builds a LangGraph with these nodes:

- `load`
- `detect_candidates`
- `error_worker`
- `medical_worker`
- `synthesize`
- `save`

`fan_out_candidates()` dispatches error and medical candidates to worker nodes
using LangGraph `Send`. Workers call retranscription with audio when available
and text fallback otherwise. `_should_apply()` gates corrections by confidence,
with lower threshold for linked medical terms.

Shared utilities provide transcript loading, low-confidence detection,
homophone and repetition candidates, glossary loading, multilingual term
linking, audio extraction, retranscription, correction application, and sidecar
writing.

## Key APIs and Data Contracts

- `AgentState` and `WorkerState` in `state.py`.
- `load_transcript(path, audio_path=None) -> TranscriptDocument`
- `analyze_transcript(doc) -> dict`
- `identify_error_candidates(doc, glossary, confidence_threshold) -> dict`
- `link_medical_terms(text, glossary, specialty=None) -> list[dict]`
- `retranscribe_segment(...) -> tuple[str, float, str]`
- `apply_corrections(doc, corrections, min_confidence) -> tuple`
- `write_correction_sidecar(...) -> Path`
- `build_asr_medical_verification_graph()`
- `run_asr_medical_verification(...) -> dict`

Correction record fields include segment ID, original text, corrected text,
confidence, method, correction type, linked terms, issue reason, and accepted
status.

## Dependencies

- Shared config profile `asr_v2`.
- LangGraph.
- Audio tools and optional ffmpeg behavior through helper modules.
- Optional faster-whisper or OpenAI audio paths depending on retranscription
  method.
- Local multilingual medical glossary JSON.

## Nonfunctional Requirements

- Corrections must be confidence-gated.
- Audit sidecars must preserve what was detected and applied.
- Missing audio should not prevent text-only fallback.
- Medical terms should link to canonical names and multilingual metadata.
- Original transcript updates must be deliberate because the workflow saves
  corrected output in place.

## Acceptance Criteria

- Error and medical candidates are detected from sample transcripts.
- Accepted high-confidence corrections are applied.
- Low-confidence or unchanged corrections are not applied.
- Workflow writes corrected transcript and sidecar.
- Tests cover glossary linking, detection, correction application, and workflow
  output behavior.

## Risks and Open Questions

- `save_node()` writes the corrected transcript back to the input transcript
  path. This is useful for the workflow test, but risky for production usage
  unless callers pass a copy or add explicit output path support.
- Broad audio fallback behavior can hide extraction failures unless stage logs
  are reviewed.
- Medical correction is high-stakes. Production use would require stronger
  human review, audit, and clinical validation.

## Development Guidance

For new medical ASR features, keep detection, retranscription, term linking,
and correction application separate. Add tests around confidence thresholds and
sidecar contents. Avoid applying changes in place for new production-oriented
commands unless the caller explicitly asks for that behavior.
