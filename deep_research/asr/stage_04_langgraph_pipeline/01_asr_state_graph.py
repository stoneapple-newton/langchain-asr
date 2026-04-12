"""
ASR Stage 4, File 1: ASR Pipeline as a LangGraph StateGraph
=============================================================
CONCEPT: Orchestrating all enhancement steps as a stateful graph.

In Stages 1–3 we built isolated tools:
  • Quality metrics
  • Punctuation & cleanup
  • Speaker naming
  • Error correction
  • Diarization correction

Now we wire them into a single StateGraph that runs the full pipeline
automatically, passing results between stages through shared state.

The graph captures the full provenance of every change:
  • What was the quality before?
  • Which stages ran?
  • What changed?
  • What is the quality now?

Graph structure:
  ┌─────────┐   ┌────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐
  │  load   │──▶│ assess │──▶│  cleanup  │──▶│  diariz  │──▶│ report │
  └─────────┘   └────────┘   └───────────┘   └──────────┘   └────────┘

Run this file:
  uv run deep_research/asr/stage_04_langgraph_pipeline/01_asr_state_graph.py
"""

import os
import json
import re
import copy
from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langgraph.graph import StateGraph, START, END


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

llm = create_chat_model(
    temperature=0,
    max_tokens=512,
)

FILLERS = {"uh", "um", "you know", "i mean", "like"}
CONTRACTION_MAP = {
    r"\barent\b": "aren't", r"\bcant\b": "can't", r"\bwont\b": "won't",
    r"\bdont\b": "don't",   r"\bisnt\b": "isn't", r"\bills\b": "I'll",
    r"\bill\b": "I'll",     r"\bthats\b": "that's", r"\bits\b": "it's",
    r"\blets\b": "let's",   r"\bweve\b": "we've", r"\bim\b": "I'm",
    r"\bive\b": "I've",     r"\bitll\b": "it'll",
}


# ---------------------------------------------------------------------------
# Shared State Schema
# ---------------------------------------------------------------------------

class ASRPipelineState(TypedDict):
    # Input
    transcript_path: str

    # Raw data
    raw_transcript: dict          # original JSON
    segments: list[dict]          # working copy of segments

    # Quality tracking
    quality_before: dict          # QualityReport as dict
    quality_after: dict           # QualityReport after all stages

    # Outputs from each stage
    speaker_names: dict[str, str]  # {SPEAKER_XX: display_name}
    stage_log: list[str]           # log of what each stage did

    # Final outputs
    improved_transcript: dict
    output_paths: list[str]


# ---------------------------------------------------------------------------
# Helper: compute quality metrics (from Stage 1, File 2)
# ---------------------------------------------------------------------------

def compute_quality(segments: list[dict]) -> dict:
    words = [w for s in segments for w in s.get("words", [])]
    total_words = len(words)

    avg_conf = sum(w.get("score", 1.0) for w in words) / max(total_words, 1)
    filler_count = sum(
        1 for s in segments for w in s.get("words", [])
        if w["word"].lower().strip(".,?!") in FILLERS
    )
    has_punct = sum(1 for s in segments if re.search(r"[.!?]\s*$", s["text"].strip()))

    return {
        "avg_confidence": round(avg_conf, 3),
        "filler_rate": round(filler_count / max(total_words, 1), 3),
        "punctuation_score": round(has_punct / max(len(segments), 1), 3),
        "total_segments": len(segments),
        "total_words": total_words,
    }


# ---------------------------------------------------------------------------
# Node 1: load
# ---------------------------------------------------------------------------

def load_node(state: ASRPipelineState) -> dict:
    """Load the raw transcript JSON from disk."""
    path = state["transcript_path"]
    with open(path) as f:
        raw = json.load(f)
    segs = copy.deepcopy(raw["segments"])
    print(f"  [load]    {len(segs)} segments, {raw['duration']:.0f}s audio")
    return {
        "raw_transcript": raw,
        "segments": segs,
        "stage_log": [f"Loaded {len(segs)} segments from {Path(path).name}"],
        "output_paths": [],
    }


# ---------------------------------------------------------------------------
# Node 2: assess
# ---------------------------------------------------------------------------

def assess_node(state: ASRPipelineState) -> dict:
    """Compute baseline quality metrics before any changes."""
    q = compute_quality(state["segments"])
    print(f"  [assess]  conf={q['avg_confidence']:.2f}, "
          f"filler={q['filler_rate']:.1%}, "
          f"punct={q['punctuation_score']:.1%}")
    return {
        "quality_before": q,
        "stage_log": state["stage_log"] + [
            f"Baseline quality: conf={q['avg_confidence']:.2f}, "
            f"filler={q['filler_rate']:.1%}, punct={q['punctuation_score']:.1%}"
        ],
    }


# ---------------------------------------------------------------------------
# Node 3: cleanup
# ---------------------------------------------------------------------------

def _rule_cleanup(text: str) -> str:
    t = text.strip()
    # Remove fillers
    t = re.sub(r"\b(uh|um|you know|i mean)\s*", "", t, flags=re.IGNORECASE)
    # Fix contractions
    for pat, rep in CONTRACTION_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    # Collapse duplicate words
    t = re.sub(r"\b(\w+) \1\b", r"\1", t, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", t).strip()


punct_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Add correct punctuation and capitalise the first word and proper nouns. "
     "Do NOT change any content words. Return only the corrected text, nothing else."),
    ("human", "Context: {context}\nText: {text}"),
])
punct_chain = punct_prompt | llm | StrOutputParser()


def cleanup_node(state: ASRPipelineState) -> dict:
    """Rule-based cleanup + LLM punctuation on every segment."""
    segs = copy.deepcopy(state["segments"])
    changes = 0

    # Rule-based pass (free)
    for seg in segs:
        original = seg["text"]
        seg["text"] = " " + _rule_cleanup(seg["text"])
        if seg["text"].strip() != original.strip():
            changes += 1

    # LLM punctuation in blocks of 6
    block_size = 6
    for start in range(0, len(segs), block_size):
        block = segs[start:start + block_size]
        block_text = "\n".join(
            f"[{i}] {s['text'].strip()}"
            for i, s in enumerate(block, start)
        )
        ctx = segs[start-1]["text"].strip() if start > 0 else ""
        result = punct_chain.invoke({"context": ctx, "text": block_text})
        # Parse the result back into individual lines
        for line in result.strip().split("\n"):
            m = re.match(r"^\[(\d+)\]\s*(.*)", line)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(segs):
                    segs[idx]["text"] = " " + m.group(2).strip()

    print(f"  [cleanup] {changes} rule-based changes, LLM punctuation applied")
    return {
        "segments": segs,
        "stage_log": state["stage_log"] + [f"Cleanup: {changes} rule-based + LLM punctuation"],
    }


# ---------------------------------------------------------------------------
# Node 4: diarize
# ---------------------------------------------------------------------------

name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Assign a role-based display name to each speaker based on their text sample. "
     "Reply with only a JSON object mapping SPEAKER_XX to a name, e.g. "
     '{"SPEAKER_00": "Facilitator", "SPEAKER_01": "Engineer"}. '
     "No other text."),
    ("human", "{samples}"),
])
name_chain = name_prompt | llm | StrOutputParser()


def diarize_node(state: ASRPipelineState) -> dict:
    """Infer speaker names and merge short backchannel segments."""
    segs = copy.deepcopy(state["segments"])

    # Build speaker samples
    samples: dict[str, str] = {}
    for seg in segs:
        spk = seg.get("speaker")
        if spk and len(samples.get(spk, "")) < 600:
            samples[spk] = samples.get(spk, "") + " " + seg["text"].strip()

    prompt_text = "\n\n".join(
        f"[{spk}]: {text[:300]}" for spk, text in samples.items()
    )

    raw_result = name_chain.invoke({"samples": prompt_text})

    # Parse JSON response
    names: dict[str, str] = {}
    try:
        names = json.loads(raw_result)
    except Exception:
        # Try extracting JSON from response
        m = re.search(r"\{[^}]+\}", raw_result, re.DOTALL)
        if m:
            try:
                names = json.loads(m.group())
            except Exception:
                names = {}

    # Merge backchannels (≤2 words, same speaker as previous)
    merged: list[dict] = []
    for seg in segs:
        wc = len(seg.get("words", []))
        dur = seg["end"] - seg["start"]
        if merged and wc <= 2 and dur < 2.0 and merged[-1].get("speaker") == seg.get("speaker"):
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"].strip()
        else:
            merged.append(seg)

    print(f"  [diarize] Names: {names}  |  Merged {len(segs)-len(merged)} backchannels")
    return {
        "segments": merged,
        "speaker_names": names,
        "stage_log": state["stage_log"] + [
            f"Diarization: {len(names)} speakers named, {len(segs)-len(merged)} segments merged"
        ],
    }


# ---------------------------------------------------------------------------
# Node 5: report
# ---------------------------------------------------------------------------

def report_node(state: ASRPipelineState) -> dict:
    """Compute final quality, assemble output, save files."""
    segs = state["segments"]
    q_after = compute_quality(segs)
    q_before = state["quality_before"]
    names = state.get("speaker_names", {})

    # Build final transcript JSON
    improved = copy.deepcopy(state["raw_transcript"])
    improved["segments"] = segs
    improved["speaker_names"] = names
    improved["pipeline_log"] = state["stage_log"]

    # Save JSON
    json_path = OUT_DIR / "transcript_final.json"
    with open(json_path, "w") as f:
        json.dump(improved, f, indent=2)

    # Save readable TXT
    txt_path = OUT_DIR / "transcript_final.txt"
    prev_spk = None
    lines = []
    for seg in segs:
        spk = seg.get("speaker", "UNKNOWN")
        label = names.get(spk, spk)
        if spk != prev_spk:
            lines.append(f"\n{label}:")
            prev_spk = spk
        lines.append(f"  {seg['text'].strip()}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # Print quality delta
    delta_conf = q_after["avg_confidence"] - q_before["avg_confidence"]
    delta_punct = q_after["punctuation_score"] - q_before["punctuation_score"]
    delta_filler = q_before["filler_rate"] - q_after["filler_rate"]

    print(f"  [report]  Quality delta:")
    print(f"            confidence : {q_before['avg_confidence']:.3f} → {q_after['avg_confidence']:.3f} ({delta_conf:+.3f})")
    print(f"            punctuation: {q_before['punctuation_score']:.1%} → {q_after['punctuation_score']:.1%} ({delta_punct:+.1%})")
    print(f"            filler rate: {q_before['filler_rate']:.1%} → {q_after['filler_rate']:.1%} ({-delta_filler:+.1%} removed)")
    print(f"            segments   : {q_before['total_segments']} → {q_after['total_segments']}")

    return {
        "quality_after": q_after,
        "improved_transcript": improved,
        "output_paths": [str(json_path), str(txt_path)],
        "stage_log": state["stage_log"] + [
            f"Final quality: conf={q_after['avg_confidence']:.2f}, "
            f"punct={q_after['punctuation_score']:.1%}"
        ],
    }


# ---------------------------------------------------------------------------
# Build and run the graph
# ---------------------------------------------------------------------------

builder = StateGraph(ASRPipelineState)
builder.add_node("load",    load_node)
builder.add_node("assess",  assess_node)
builder.add_node("cleanup", cleanup_node)
builder.add_node("diarize", diarize_node)
builder.add_node("report",  report_node)

builder.add_edge(START,     "load")
builder.add_edge("load",    "assess")
builder.add_edge("assess",  "cleanup")
builder.add_edge("cleanup", "diarize")
builder.add_edge("diarize", "report")
builder.add_edge("report",  END)

pipeline = builder.compile()

print("=" * 60)
print("  ASR ENHANCEMENT PIPELINE")
print("=" * 60)

final_state = pipeline.invoke({
    "transcript_path": str(TRANSCRIPT_PATH),
    "raw_transcript": {},
    "segments": [],
    "quality_before": {},
    "quality_after": {},
    "speaker_names": {},
    "stage_log": [],
    "improved_transcript": {},
    "output_paths": [],
})

print()
print("  Stage log:")
for entry in final_state["stage_log"]:
    print(f"    • {entry}")
print()
print("  Output files:")
for path in final_state["output_paths"]:
    print(f"    • {path}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ StateGraph makes the pipeline's data flow explicit and inspectable
# ✅ Each node receives full state, returns only what it changed
# ✅ Quality metrics run before AND after — proves the pipeline is helping
# ✅ stage_log gives a full audit trail of every change made
# ✅ Separating concerns into nodes makes it easy to add/remove/reorder stages
# ✅ The graph compiles once; you can invoke it on any transcript file
