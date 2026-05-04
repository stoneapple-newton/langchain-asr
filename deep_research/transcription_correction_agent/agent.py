"""
Transcription Correction Agent
================================
A LangGraph StateGraph that:

  1. Loads a WhisperX-style JSON transcript + its source audio
  2. Identifies transcription errors (low-confidence, incoherent, garbled)
  3. Identifies segments that may contain medical terms
  4. For each flagged segment:
       a. Extracts the corresponding audio clip with ffmpeg
       b. Re-transcribes via faster-whisper OR multimodal LLM (auto-fallback)
       c. If neither audio tool is available, falls back to text-only LLM fix
  5. Links confirmed medical terms to the multilingual glossary
  6. Writes a corrected JSON transcript + annotated markdown

Graph structure (linear backbone + Send fan-out for parallel segment workers):

  START
    ↓
  load ──────────────────────────────────────────────────────────
    ↓                                                            │
  detect_errors ──→ detect_medical                              │
    ↓                    ↓                                       │
  [Send fan-out]    [Send fan-out]                              │
  error_worker(*)   medical_worker(*)                           │
    ↓                    ↓                                       │
  synthesize ←──────────────────────────────────────────────────
    ↓
  save_output
    ↓
  END

Each *_worker uses conditional edges to route:
  extract_audio → choose_method → [faster_whisper | multimodal_llm | llm_text]
                                        ↓
                                  record_correction

Run with:
  uv run deep_research/transcription_correction_agent/run.py
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Literal

import operator
from typing_extensions import TypedDict

# Resolve repo root so we can import `config`
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import create_chat_model  # type: ignore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from state import (
    AgentState,
    ErrorWorkerState,
    MedicalWorkerState,
    CorrectionRecord,
)
from tools.audio_extraction import extract_segment, wav_to_base64
from tools.retranscription import retranscribe
from tools.medical_terms import (
    load_glossary,
    link_terms,
    build_llm_hint,
    has_medical_keywords,
)

# ---------------------------------------------------------------------------
# LLM setup (text-only; used for detection + fallback correction)
# ---------------------------------------------------------------------------

_llm = create_chat_model(temperature=0, max_tokens=4096)

# ---------------------------------------------------------------------------
# Helper: build surrounding context string for a segment
# ---------------------------------------------------------------------------

def _context_window(segments: list[dict], segment_id: str, window: int = 3) -> str:
    """Return up to *window* segments before and after *segment_id* as text."""
    ids = [s.get("id", str(i)) for i, s in enumerate(segments)]
    try:
        idx = ids.index(segment_id)
    except ValueError:
        return ""
    lo = max(0, idx - window)
    hi = min(len(segments), idx + window + 1)
    lines: list[str] = []
    for i, seg in enumerate(segments[lo:hi], start=lo):
        marker = ">>>" if i == idx else "   "
        spk = seg.get("speaker", "?")
        lines.append(f"{marker} [{seg['start']:.1f}s] {spk}: {seg['text']}")
    return "\n".join(lines)


# ===========================================================================
# NODE: load
# ===========================================================================

def load_node(state: AgentState) -> dict:
    """Load transcript JSON + medical glossary; initialise working segments."""
    path = state["transcript_path"]
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    segs = copy.deepcopy(raw.get("segments", []))
    # Ensure each segment has an id field
    for i, seg in enumerate(segs):
        seg.setdefault("id", str(i))

    glossary = load_glossary()
    n = len(segs)
    dur = raw.get("duration", (segs[-1]["end"] if segs else 0))

    print(f"  [load] {n} segments, {dur:.0f}s audio — {path}")
    return {
        "segments": segs,
        "medical_glossary": glossary,
        "corrections": [],
        "stage_log": [f"Loaded {n} segments from {Path(path).name}"],
        "output_paths": [],
    }


# ===========================================================================
# NODE: detect_errors
# ===========================================================================

_ERROR_DETECT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an ASR quality reviewer. Identify transcription segments that "
        "likely contain errors: garbled words, incoherent sentences, repetitions, "
        "or text that doesn't fit the surrounding context.\n\n"
        "Return a JSON array of objects with keys:\n"
        "  segment_id (string), issue_type (string), reason (string)\n"
        "issue_type must be one of: low_confidence, incoherence, repetition, garbled\n"
        "Return an empty array [] if no issues found. "
        "Return ONLY valid JSON, no markdown, no explanation.",
    ),
    (
        "human",
        "Transcript segments (each line: [id] [start]s speaker: text):\n\n{transcript}",
    ),
])
_error_detect_chain = _ERROR_DETECT_PROMPT | _llm | StrOutputParser()


def detect_errors_node(state: AgentState) -> dict:
    """Ask the LLM to flag segments with possible transcription errors."""
    segs = state["segments"]

    # Build compact transcript text
    lines = [
        f"[{seg.get('id', i)}] {seg['start']:.1f}s "
        f"{seg.get('speaker', '?')}: {seg['text']}"
        for i, seg in enumerate(segs)
    ]
    transcript_text = "\n".join(lines)

    raw = _error_detect_chain.invoke({"transcript": transcript_text})

    # Parse JSON robustly
    candidates: list[dict] = []
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            candidates = parsed
    except Exception as exc:
        print(f"  [detect_errors] JSON parse failed: {exc!r} — skipping")

    # Enrich candidates with segment data + context
    seg_by_id = {seg.get("id", str(i)): seg for i, seg in enumerate(segs)}
    enriched: list[dict] = []
    for c in candidates:
        sid = str(c.get("segment_id", ""))
        seg = seg_by_id.get(sid)
        if seg is None:
            continue
        enriched.append({
            "segment_id": sid,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": seg.get("speaker"),
            "issue_type": c.get("issue_type", "garbled"),
            "reason": c.get("reason", ""),
            "context": _context_window(segs, sid),
        })

    print(f"  [detect_errors] {len(enriched)} error candidates flagged")
    return {
        "error_candidates": enriched,
        "stage_log": [f"Error detection: {len(enriched)} candidates flagged"],
    }


# ===========================================================================
# NODE: detect_medical
# ===========================================================================

_MEDICAL_DETECT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical terminology expert reviewing an ASR transcript. "
        "Identify segments where the speaker is likely using medical terminology "
        "that may have been transcribed incorrectly or phonetically mangled.\n\n"
        "Return a JSON array of objects with keys:\n"
        "  segment_id (string), suspected_term (string), specialty (string)\n"
        "specialty must be one of: cardiology, oncology, neurology, pulmonology, "
        "gastroenterology, nephrology, orthopedics, pharmacy, critical_care, general\n"
        "Return an empty array [] if no medical terms found. "
        "Return ONLY valid JSON, no markdown, no explanation.",
    ),
    (
        "human",
        "Transcript segments:\n\n{transcript}\n\n"
        "Known medical term hints (multilingual):\n{hints}",
    ),
])
_medical_detect_chain = _MEDICAL_DETECT_PROMPT | _llm | StrOutputParser()


def detect_medical_node(state: AgentState) -> dict:
    """Identify segments that likely contain medical terminology."""
    segs = state["segments"]
    glossary = state["medical_glossary"]

    # Pre-filter with regex to reduce LLM calls
    candidate_segs = [
        seg for seg in segs
        if has_medical_keywords(seg["text"])
    ]
    # Always run LLM even if no regex hits — it catches contextual cues
    if not candidate_segs:
        candidate_segs = segs

    lines = [
        f"[{seg.get('id', i)}] {seg['start']:.1f}s "
        f"{seg.get('speaker', '?')}: {seg['text']}"
        for i, seg in enumerate(candidate_segs)
    ]
    transcript_text = "\n".join(lines)
    hints = build_llm_hint("general", glossary, max_terms=10)

    raw = _medical_detect_chain.invoke({
        "transcript": transcript_text,
        "hints": hints,
    })

    candidates: list[dict] = []
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            candidates = parsed
    except Exception as exc:
        print(f"  [detect_medical] JSON parse failed: {exc!r} — skipping")

    seg_by_id = {seg.get("id", str(i)): seg for i, seg in enumerate(segs)}
    enriched: list[dict] = []
    for c in candidates:
        sid = str(c.get("segment_id", ""))
        seg = seg_by_id.get(sid)
        if seg is None:
            continue
        enriched.append({
            "segment_id": sid,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": seg.get("speaker"),
            "suspected_term": c.get("suspected_term", ""),
            "specialty": c.get("specialty", "general"),
            "context": _context_window(segs, sid),
        })

    print(f"  [detect_medical] {len(enriched)} medical term candidates flagged")
    return {
        "medical_candidates": enriched,
        "stage_log": [f"Medical detection: {len(enriched)} candidates flagged"],
    }


# ===========================================================================
# Fan-out: dispatch ALL workers (error + medical) in one conditional edge
# ===========================================================================

def fan_out_all_workers(state: AgentState) -> list[Send]:
    """
    Dispatch error workers and medical workers in parallel.

    Using a single conditional edge avoids the double-invoke problem that
    arises when two conditional edges from the same node both fall back to
    Send("synthesize", {}).
    """
    sends: list[Send] = []

    worker_args_base = {
        "audio_path": state["audio_path"],
        "retranscription_method": state["retranscription_method"],
        "segments": state["segments"],
        "medical_glossary": state["medical_glossary"],
    }

    for c in state.get("error_candidates") or []:
        sends.append(Send("error_worker", {**worker_args_base, "candidate": c}))

    for c in state.get("medical_candidates") or []:
        sends.append(Send("medical_worker", {**worker_args_base, "candidate": c}))

    # If nothing to do, jump straight to synthesis
    if not sends:
        return [Send("synthesize", {})]

    return sends


# ===========================================================================
# NODE: error_worker  (runs once per error candidate, in parallel)
# ===========================================================================

def error_worker_node(state: ErrorWorkerState) -> dict:
    """
    For one error candidate:
      1. Extract audio segment
      2. Retranscribe (faster-whisper → multimodal LLM → text LLM)
      3. Return a CorrectionRecord
    """
    c = state["candidate"]
    sid = c["segment_id"]
    original = c["text"]
    context = c["context"]

    print(f"  [error_worker] seg={sid} ({c['issue_type']}): {original[:60]!r}")

    corrected_text = original
    method_used = "unchanged"
    confidence = 1.0

    audio_path = state["audio_path"]
    if audio_path and Path(audio_path).exists():
        try:
            wav = extract_segment(audio_path, c["start"], c["end"])
            corrected_text, confidence, method_used = retranscribe(
                wav_path=wav,
                method=state["retranscription_method"],
                context_text=context,
                issue_reason=c.get("reason", ""),
                original_text=original,
                llm=_llm,
            )
        except Exception as exc:
            print(f"  [error_worker] audio extraction failed: {exc!r} — text fallback")

    # If audio unavailable or retranscription unchanged, do text-only correction
    if method_used == "unchanged" or corrected_text == original:
        from tools.retranscription import correct_with_text_llm
        corrected_text, confidence = correct_with_text_llm(
            original_text=original,
            context_text=context,
            issue_reason=c.get("reason", ""),
            llm=_llm,
        )
        method_used = "llm_text"

    linked = link_terms(corrected_text, state["medical_glossary"])

    record: CorrectionRecord = {
        "segment_id": sid,
        "original_text": original,
        "corrected_text": corrected_text,
        "method": method_used,
        "correction_type": "error",
        "confidence": confidence,
        "medical_terms": [t["canonical_en"] for t in linked],
        "linked_terms": linked,
    }

    print(f"  [error_worker] seg={sid} → {corrected_text[:60]!r} [{method_used}]")
    return {"corrections": [record]}


# ===========================================================================
# NODE: medical_worker  (runs once per medical candidate, in parallel)
# ===========================================================================

def medical_worker_node(state: MedicalWorkerState) -> dict:
    """
    For one medical term candidate:
      1. Extract audio segment
      2. Retranscribe with specialty-aware prompt
      3. Link the corrected text to the multilingual glossary
    """
    c = state["candidate"]
    sid = c["segment_id"]
    original = c["text"]
    specialty = c.get("specialty", "general")
    context = c["context"]
    glossary = state["medical_glossary"]

    print(f"  [medical_worker] seg={sid} ({specialty}): {original[:60]!r}")

    corrected_text = original
    method_used = "unchanged"
    confidence = 1.0

    # Build a glossary hint for the specialty to improve retranscription
    specialty_hint = build_llm_hint(specialty, glossary, max_terms=5)

    audio_path = state["audio_path"]
    if audio_path and Path(audio_path).exists():
        try:
            wav = extract_segment(audio_path, c["start"], c["end"])
            corrected_text, confidence, method_used = retranscribe(
                wav_path=wav,
                method=state["retranscription_method"],
                context_text=context,
                specialty_hint=specialty_hint,
                issue_reason=f"Possible medical term: {c.get('suspected_term', '')}",
                original_text=original,
                llm=_llm,
            )
        except Exception as exc:
            print(f"  [medical_worker] audio extraction failed: {exc!r} — text fallback")

    if method_used == "unchanged" or corrected_text == original:
        # Text-only: ask LLM to correct the suspected medical term
        from tools.retranscription import correct_with_text_llm
        corrected_text, confidence = correct_with_text_llm(
            original_text=original,
            context_text=context,
            issue_reason=f"Suspected medical term: {c.get('suspected_term', '')}",
            medical_context=specialty_hint,
            llm=_llm,
        )
        method_used = "llm_text"

    linked = link_terms(corrected_text, glossary)

    record: CorrectionRecord = {
        "segment_id": sid,
        "original_text": original,
        "corrected_text": corrected_text,
        "method": method_used,
        "correction_type": "medical_term",
        "confidence": confidence,
        "medical_terms": [t["canonical_en"] for t in linked],
        "linked_terms": linked,
    }

    print(f"  [medical_worker] seg={sid} → terms found: {record['medical_terms']}")
    return {"corrections": [record]}


# ===========================================================================
# NODE: synthesize  (runs after all workers complete)
# ===========================================================================

def synthesize_node(state: AgentState) -> dict:
    """
    Merge all CorrectionRecords into the working segment list.

    Later corrections (medical_term type) win over earlier ones if they
    cover the same segment, since they are more specific.
    """
    corrections = state.get("corrections") or []
    segs = copy.deepcopy(state["segments"])
    seg_by_id = {seg.get("id", str(i)): seg for i, seg in enumerate(segs)}

    # Build a correction map; medical_term corrections take precedence
    correction_map: dict[str, dict] = {}
    for rec in corrections:
        sid = rec["segment_id"]
        existing = correction_map.get(sid)
        if existing is None or rec["correction_type"] == "medical_term":
            correction_map[sid] = rec

    applied = 0
    for sid, rec in correction_map.items():
        seg = seg_by_id.get(sid)
        if seg is None:
            continue
        if rec["corrected_text"] and rec["corrected_text"] != rec["original_text"]:
            seg["text"] = rec["corrected_text"]
            seg.setdefault("corrections", []).append({
                "method": rec["method"],
                "correction_type": rec["correction_type"],
                "confidence": rec["confidence"],
                "medical_terms": rec["medical_terms"],
            })
            applied += 1

    print(f"  [synthesize] {applied}/{len(correction_map)} corrections applied to segments")
    return {
        "corrected_segments": segs,
        "stage_log": [
            f"Synthesis: {len(corrections)} records → {applied} segments updated"
        ],
    }


# ===========================================================================
# NODE: save_output
# ===========================================================================

def save_output_node(state: AgentState) -> dict:
    """Write the corrected transcript to JSON + Markdown."""
    segs = state.get("corrected_segments") or state["segments"]
    corrections = state.get("corrections") or []

    # Derive output dir from transcript path
    src = Path(state["transcript_path"])
    out_dir = src.parent.parent / "outputs" / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────
    payload = {
        "source": str(src),
        "audio": state.get("audio_path", ""),
        "language": "en",
        "retranscription_method": state.get("retranscription_method", "auto"),
        "segments": segs,
        "correction_summary": [
            {
                "segment_id": r["segment_id"],
                "type": r["correction_type"],
                "method": r["method"],
                "medical_terms": r["medical_terms"],
            }
            for r in corrections
        ],
        "pipeline_log": state.get("stage_log", []),
    }
    json_path = out_dir / f"{src.stem}.corrected.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Markdown ───────────────────────────────────────────────────────────
    md_lines = [
        "# Corrected Transcript",
        "",
        f"- Source: `{src.name}`",
        f"- Audio: `{state.get('audio_path', 'N/A')}`",
        f"- Corrections applied: {len([r for r in corrections if r['corrected_text'] != r['original_text']])}",
        "",
        "## Segments",
        "",
    ]
    prev_spk = None
    for seg in segs:
        spk = seg.get("speaker", "UNKNOWN")
        if spk != prev_spk:
            md_lines.append(f"\n**{spk}**")
            prev_spk = spk
        corr_flag = " *(corrected)*" if seg.get("corrections") else ""
        md_lines.append(
            f"- `[{seg['start']:.1f}s–{seg['end']:.1f}s]` "
            f"{seg['text']}{corr_flag}"
        )
        for c in seg.get("corrections", []):
            if c.get("medical_terms"):
                md_lines.append(
                    f"  - Medical terms: **{', '.join(c['medical_terms'])}**"
                )

    # ── Medical term glossary annex ────────────────────────────────────────
    all_linked: list[dict] = []
    for rec in corrections:
        all_linked.extend(rec.get("linked_terms", []))
    # Deduplicate by canonical_en
    seen: set[str] = set()
    unique_linked = []
    for t in all_linked:
        if t["canonical_en"] not in seen:
            unique_linked.append(t)
            seen.add(t["canonical_en"])

    if unique_linked:
        md_lines += ["", "## Medical Terms Identified", ""]
        for t in unique_linked:
            icd = f" *(ICD-10: {t['icd_hint']})*" if t.get("icd_hint") else ""
            md_lines.append(f"### {t['canonical_en']}{icd}")
            md_lines.append(f"- Specialty: {t.get('specialty', 'general')}")
            trans = t.get("translations", {})
            if any(trans.values()):
                md_lines.append("- Translations:")
                lang_names = {
                    "es": "Spanish", "fr": "French", "de": "German",
                    "pt": "Portuguese", "ar": "Arabic (romanized)",
                    "zh_pinyin": "Mandarin (Pinyin)", "ja_romaji": "Japanese (Romaji)",
                }
                for lang, name in lang_names.items():
                    val = trans.get(lang, "")
                    if val:
                        md_lines.append(f"  - {name}: {val}")
            variants = t.get("phonetic_variants", [])[:4]
            if variants:
                md_lines.append(f"- Common ASR variants: {', '.join(variants)}")
            md_lines.append("")

    md_path = out_dir / f"{src.stem}.corrected.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"  [save] → {json_path}")
    print(f"  [save] → {md_path}")
    return {
        "output_paths": [str(json_path), str(md_path)],
        "stage_log": [f"Saved output to {out_dir}"],
    }


# ===========================================================================
# Graph assembly
# ===========================================================================

def build_correction_graph():
    """Build and compile the transcription correction agent graph."""
    builder = StateGraph(AgentState)

    # Backbone nodes
    builder.add_node("load", load_node)
    builder.add_node("detect_errors", detect_errors_node)
    builder.add_node("detect_medical", detect_medical_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("save_output", save_output_node)

    # Worker nodes (invoked via Send fan-out)
    builder.add_node("error_worker", error_worker_node)
    builder.add_node("medical_worker", medical_worker_node)

    # ── Linear backbone ─────────────────────────────────────────────────
    builder.add_edge(START, "load")
    builder.add_edge("load", "detect_errors")
    builder.add_edge("detect_errors", "detect_medical")

    # ── Fan-out after detection ──────────────────────────────────────────
    # Single conditional edge dispatches BOTH error and medical workers,
    # so "synthesize" is only ever reached once (after all workers complete).
    builder.add_conditional_edges(
        "detect_medical",
        fan_out_all_workers,
        ["error_worker", "medical_worker", "synthesize"],
    )

    # Workers return their CorrectionRecord → accumulate via reducer → synthesize
    builder.add_edge("error_worker", "synthesize")
    builder.add_edge("medical_worker", "synthesize")

    # ── After synthesis ──────────────────────────────────────────────────
    builder.add_edge("synthesize", "save_output")
    builder.add_edge("save_output", END)

    return builder.compile()


# Singleton graph instance (imported by run.py)
correction_graph = build_correction_graph()
