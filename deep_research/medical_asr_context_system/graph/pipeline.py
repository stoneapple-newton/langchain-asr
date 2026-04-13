"""
Medical ASR Context System — main LangGraph pipeline.

Graph structure
---------------

    START
      │
      ▼
    retrieve_context          ← Query Chroma store; build Whisper initial_prompt
      │
      ▼
    transcribe                ← Run Whisper (or load existing JSON) with prompt injected
      │
      ▼
    fan_out_segments ─────────┬──────────────────┐
      (Send per chunk)        │                  │
                              ▼                  ▼
                        extract_worker    extract_worker   … (parallel)
                              │                  │
                              └────────┬─────────┘
                                       ▼  (operator.add accumulates)
                               validate_terms
                                       │
                                       ▼
                                update_store          ← Upsert into Chroma
                                       │
                                       ▼
                               annotate_transcript    ← Tag segments with medical terms
                                       │
                                       ▼
                                 save_output          ← Write JSON + audit sidecar
                                       │
                                       ▼
                                      END

The feedback loop is automatic: ``retrieve_context`` always queries the same
Chroma store that ``update_store`` writes to, so each session enriches the
context available to future sessions.

Public API
----------
    from deep_research.medical_asr_context_system.graph.pipeline import (
        build_pipeline,
        run_pipeline,
    )

    # Run end-to-end
    result = run_pipeline(
        audio_path="path/to/recording.wav",
        transcript_path="path/to/output.json",
        language="es",
        specialty="cardiology",
    )
    print(result["stage_log"])
    print(result["terms_added"], "new terms stored")
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.graph import END, START, StateGraph

from ..agents.extraction import extract_worker_node, fan_out_segments_node
from ..agents.update import annotate_transcript_node, update_store_node, validate_terms_node
from ..context_store.retriever import build_whisper_prompt
from ..context_store.store import TermContextStore
from ..state import ExtractWorkerState, MedicalASRContextState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_dir() -> str:
    """Resolve context store directory from env or default."""
    return os.environ.get(
        "MEDICAL_CONTEXT_STORE_DIR",
        str(REPO_ROOT / "data" / "medical_context_store"),
    )


def _transcribe_audio(audio_path: str, initial_prompt: str, language: str) -> dict[str, Any]:
    """
    Transcribe ``audio_path`` with faster-whisper, injecting ``initial_prompt``.

    Falls back gracefully when faster-whisper is not installed: returns an
    empty transcript dict so the rest of the pipeline can still run (useful
    when the caller provides an existing JSON transcript instead).
    """
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
        model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_iter, info = model.transcribe(
            audio_path,
            language=language if language != "en" else None,
            initial_prompt=initial_prompt or None,
            word_timestamps=True,
        )
        segments: list[dict] = []
        for seg in segments_iter:
            words = [
                {"word": w.word, "start": w.start, "end": w.end, "score": w.probability}
                for w in (seg.words or [])
            ]
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words,
            })
        return {
            "language": info.language,
            "duration": segments[-1]["end"] if segments else 0.0,
            "segments": segments,
        }
    except ImportError:
        return {"language": language, "duration": 0.0, "segments": []}
    except Exception as exc:
        return {"language": language, "duration": 0.0, "segments": [], "_error": str(exc)}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def retrieve_context_node(state: MedicalASRContextState) -> dict[str, Any]:
    """
    Step 1: Query the Chroma context store and build the Whisper initial_prompt.

    Also bootstraps the store from the project glossary on the very first run
    (i.e. when the Chroma collection is empty).
    """
    language = state.get("language", "en")
    specialty = state.get("specialty", "general")

    store = TermContextStore(persist_dir=_store_dir())

    # Seed from glossary if the store is empty
    if store.count() == 0:
        n = store.bootstrap_from_glossary()
        log_msg = f"Bootstrapped context store with {n} terms from glossary"
    else:
        log_msg = f"Context store has {store.count()} term(s)"

    retrieved = store.query(specialty, language, top_k=40)
    whisper_prompt = build_whisper_prompt(retrieved, language=language, specialty=specialty)

    # whisper_prompt — token-limited string for Whisper's initial_prompt parameter
    # The same string is also reused as a compact term reference in GPT-4.1 prompts
    # (fan_out_segments_node passes it as glossary_hint to each extract_worker).
    return {
        "retrieved_terms": retrieved,
        "whisper_prompt": whisper_prompt,
        "stage_log": [
            log_msg,
            f"Retrieved {len(retrieved)} context term(s) for {specialty}/{language}",
            f"Whisper prompt ({len(whisper_prompt.split())} words): {whisper_prompt[:120]}{'…' if len(whisper_prompt) > 120 else ''}",
        ],
    }


def transcribe_node(state: MedicalASRContextState) -> dict[str, Any]:
    """
    Step 2: Transcribe the audio file.

    - If ``transcript_path`` already exists as a JSON file, loads it directly
      (useful for re-running extraction on an existing transcript).
    - Otherwise calls faster-whisper with ``whisper_prompt`` injected as
      ``initial_prompt`` to bias vocabulary toward known medical terms.
    """
    transcript_path = state.get("transcript_path", "")
    audio_path = state.get("audio_path", "")
    language = state.get("language", "en")
    whisper_prompt = state.get("whisper_prompt", "")

    # Load existing transcript if available
    if transcript_path and Path(transcript_path).exists():
        try:
            raw = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
            seg_count = len(raw.get("segments", []))
            return {
                "raw_transcript": raw,
                "stage_log": [f"Loaded existing transcript: {seg_count} segment(s)"],
            }
        except Exception as exc:
            pass  # fall through to transcription

    # Transcribe with Whisper
    if not audio_path or not Path(audio_path).exists():
        return {
            "raw_transcript": {"language": language, "duration": 0.0, "segments": []},
            "stage_log": ["No audio file found — using empty transcript"],
        }

    raw = _transcribe_audio(audio_path, whisper_prompt, language)
    seg_count = len(raw.get("segments", []))

    # Save to transcript_path for downstream nodes and re-runs
    if transcript_path:
        out = Path(transcript_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "raw_transcript": raw,
        "stage_log": [
            f"Transcribed {audio_path}: {seg_count} segment(s), "
            f"lang={raw.get('language', language)}, "
            f"duration={raw.get('duration', 0):.1f}s"
        ],
    }


def save_output_node(state: MedicalASRContextState) -> dict[str, Any]:
    """
    Step 6: Write the annotated transcript and an audit sidecar JSON.

    Output files:
      ``<transcript_path>``                   — annotated transcript (overwritten)
      ``<transcript_path>.medical_audit.json`` — session audit trail
    """
    corrected = state.get("corrected_transcript") or state.get("raw_transcript") or {}
    transcript_path = state.get("transcript_path", "")

    if not transcript_path:
        return {
            "output_paths": [],
            "stage_log": ["No transcript_path set — skipping save"],
        }

    out_path = Path(transcript_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(corrected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    audit = {
        "session_id": state.get("session_id", ""),
        "language": state.get("language", "en"),
        "specialty": state.get("specialty", "general"),
        "terms_extracted": len(state.get("validated_terms") or []),
        "terms_added_to_store": state.get("terms_added", 0),
        "terms_updated_in_store": state.get("terms_updated", 0),
        "whisper_prompt": state.get("whisper_prompt", ""),
        "retrieved_terms_count": len(state.get("retrieved_terms") or []),
        "stage_log": state.get("stage_log") or [],
        "validated_terms": [
            {
                "canonical": t.get("canonical") or t.get("canonical_en", ""),
                "category": t.get("category", ""),
                "specialty": t.get("specialty", ""),
                "confidence": t.get("confidence", ""),
            }
            for t in (state.get("validated_terms") or [])
        ],
    }
    audit_path = out_path.parent / (out_path.stem + ".medical_audit.json")
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "output_paths": [str(out_path), str(audit_path)],
        "stage_log": [
            f"Saved annotated transcript: {out_path.name}",
            f"Saved audit sidecar: {audit_path.name}",
        ],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pipeline() -> Any:
    """
    Compile and return the Medical ASR Context System LangGraph.

    Fan-out pattern (same as asr_medical_verification/workflow.py):
    - ``transcribe`` produces ``raw_transcript``
    - ``fan_out_segments_node`` is used as the *routing function* in
      ``add_conditional_edges``; it reads ``raw_transcript``, builds chunks,
      and returns ``list[Send("extract_worker", ...)]``
    - Each ``extract_worker`` invocation runs in parallel; results accumulate
      in ``extracted_terms`` via ``operator.add``
    - All workers edge to ``validate_terms`` which merges the results
    """
    builder = StateGraph(MedicalASRContextState)

    # Register nodes
    builder.add_node("retrieve_context", retrieve_context_node)
    builder.add_node("transcribe", transcribe_node)
    builder.add_node("extract_worker", extract_worker_node)
    builder.add_node("validate_terms", validate_terms_node)
    builder.add_node("update_store", update_store_node)
    builder.add_node("annotate_transcript", annotate_transcript_node)
    builder.add_node("save_output", save_output_node)

    # Linear backbone up to transcription
    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "transcribe")

    # Fan-out: fan_out_segments_node is the routing function — reads
    # raw_transcript and returns list[Send] to dispatch parallel workers.
    # If transcript is empty it returns Send("validate_terms", ...) directly.
    builder.add_conditional_edges(
        "transcribe",
        fan_out_segments_node,
        ["extract_worker", "validate_terms"],
    )
    # Workers converge back (operator.add merges extracted_terms)
    builder.add_edge("extract_worker", "validate_terms")

    # Continue pipeline
    builder.add_edge("validate_terms", "update_store")
    builder.add_edge("update_store", "annotate_transcript")
    builder.add_edge("annotate_transcript", "save_output")
    builder.add_edge("save_output", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# High-level run function
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    audio_path: str = "",
    transcript_path: str = "",
    language: str = "en",
    specialty: str = "general",
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the complete Medical ASR Context System pipeline.

    At minimum, supply either ``audio_path`` (to transcribe) or
    ``transcript_path`` pointing to an existing WhisperX JSON (to re-run
    extraction without re-transcribing).

    Parameters
    ----------
    audio_path      : path to the audio file (.wav, .mp3, .flac, …)
    transcript_path : path where the output JSON will be written (or read from
                      if it already exists)
    language        : BCP-47 code of the session language (default "en")
    specialty       : medical specialty hint (default "general")
    session_id      : unique identifier for provenance; auto-generated if omitted

    Returns
    -------
    dict : final LangGraph state
    """
    graph = build_pipeline()

    initial_state: MedicalASRContextState = {
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "language": language,
        "specialty": specialty,
        "session_id": session_id or str(uuid.uuid4())[:8],
        "retrieved_terms": [],
        "whisper_prompt": "",
        "raw_transcript": {},
        "extracted_terms": [],
        "validated_terms": [],
        "terms_added": 0,
        "terms_updated": 0,
        "corrected_transcript": {},
        "output_paths": [],
        "stage_log": [],
    }

    return graph.invoke(initial_state)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Medical ASR Context System — transcribe and extract medical terms"
    )
    parser.add_argument("--audio", default="", help="Path to audio file")
    parser.add_argument("--transcript", default="", help="Path to output/existing transcript JSON")
    parser.add_argument("--language", default="en", help="BCP-47 language code (default: en)")
    parser.add_argument("--specialty", default="general", help="Medical specialty (default: general)")
    args = parser.parse_args()

    print("=" * 60)
    print("Medical ASR Context System")
    print("=" * 60)

    result = run_pipeline(
        audio_path=args.audio,
        transcript_path=args.transcript,
        language=args.language,
        specialty=args.specialty,
    )

    print("\nStage log:")
    for line in result.get("stage_log", []):
        print(f"  {line}")

    print(f"\nNew terms stored : {result.get('terms_added', 0)}")
    print(f"Terms updated    : {result.get('terms_updated', 0)}")
    print(f"Output files     : {result.get('output_paths', [])}")
