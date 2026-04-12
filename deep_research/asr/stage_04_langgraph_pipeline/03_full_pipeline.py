"""
ASR Stage 4, File 3: Full Multi-Stage ASR Pipeline
====================================================
CONCEPT: Combining all stages into one production-ready graph with
         human review, quality loops, and multiple output formats.

This is the complete pipeline:

  load → assess → [quality gate]
                       │ pass
                  cleanup_loop ← (iterates until thresholds met)
                       │
                  diarize
                       │
                  error_correct
                       │
                  human_review ← interrupt() — human approves before saving
                       │ approved
                  export → END

New elements vs. Files 1 & 2:
  - error_correct node  : flags and fixes low-confidence words
  - human_review node   : interrupt() for human approval before writing output
  - export node         : writes JSON, TXT, SRT, and MD simultaneously
  - skip_improvement    : fast path when quality is already good

Run this file:
  uv run deep_research/asr/stage_04_langgraph_pipeline/03_full_pipeline.py
"""

import os
import json
import re
import copy
from pathlib import Path
import sys
from typing_extensions import TypedDict
from datetime import timedelta

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

llm = create_chat_model(
    temperature=0,
    max_tokens=512,
)

FILLERS = {"uh", "um", "you know", "i mean"}
CONTRACTION_MAP = {
    r"\barent\b": "aren't", r"\bcant\b": "can't", r"\bdont\b": "don't",
    r"\bills\b": "I'll",    r"\bill\b": "I'll",    r"\bthats\b": "that's",
    r"\blets\b": "let's",   r"\bwere\b": "we're",  r"\bim\b": "I'm",
    r"\bive\b": "I've",     r"\bits\b": "it's",
}
MAX_LOOP = 2
QUALITY_TARGETS = {"punctuation_score": 0.65, "filler_rate": 0.05, "capitalisation_score": 0.65}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FullPipelineState(TypedDict):
    transcript_path: str
    raw: dict
    segments: list[dict]
    speaker_names: dict
    quality: dict
    iteration: int
    review_approved: bool
    review_notes: str
    output_paths: list[str]
    log: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality(segments: list[dict]) -> dict:
    words = [w for s in segments for w in s.get("words", [])]
    n = max(len(words), 1)
    ns = max(len(segments), 1)
    return {
        "avg_confidence": round(sum(w.get("score", 1) for w in words) / n, 3),
        "filler_rate": round(sum(1 for s in segments for w in s.get("words", [])
                                  if w["word"].lower().strip(".,?!") in FILLERS) / n, 3),
        "punctuation_score": round(sum(1 for s in segments
                                        if re.search(r"[.!?]$", s["text"].strip())) / ns, 3),
        "capitalisation_score": round(sum(1 for s in segments
                                           if s["text"].strip() and s["text"].strip()[0].isupper()) / ns, 3),
    }


def _rule_clean(text: str) -> str:
    t = re.sub(r"\b(uh|um|you know|i mean)\s*", "", text.strip(), flags=re.IGNORECASE)
    for pat, rep in CONTRACTION_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", t).strip()


def _fmt_srt(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    h, rem = divmod(int(td.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_node(state: FullPipelineState) -> dict:
    with open(state["transcript_path"]) as f:
        raw = json.load(f)
    segs = copy.deepcopy(raw["segments"])
    q = _quality(segs)
    print(f"  [load]   {len(segs)} segs | conf={q['avg_confidence']:.2f} "
          f"punct={q['punctuation_score']:.0%} filler={q['filler_rate']:.1%}")
    return {"raw": raw, "segments": segs, "quality": q,
            "log": [f"Loaded {len(segs)} segments"]}


def should_improve(state: FullPipelineState) -> str:
    q = state["quality"]
    if state.get("iteration", 0) >= MAX_LOOP:
        return "diarize"
    failing = [k for k, threshold in QUALITY_TARGETS.items()
               if (q.get(k, 0) < threshold if k != "filler_rate" else q.get(k, 0) > threshold)]
    return "cleanup" if failing else "diarize"


cleanup_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Fix punctuation, capitalisation, remove fillers (uh, um), fix contractions. "
     "Return each line as [N] corrected text. Same number of lines as input."),
    ("human", "{block}"),
])
cleanup_chain = cleanup_prompt | llm | StrOutputParser()


def cleanup_node(state: FullPipelineState) -> dict:
    segs = copy.deepcopy(state["segments"])
    # Rule pass
    for seg in segs:
        seg["text"] = " " + _rule_clean(seg["text"])

    # LLM pass in blocks
    block_size = 8
    for start in range(0, len(segs), block_size):
        block = segs[start:start + block_size]
        block_text = "\n".join(f"[{start+i}] {s['text'].strip()}" for i, s in enumerate(block))
        result = cleanup_chain.invoke({"block": block_text})
        for line in result.strip().split("\n"):
            m = re.match(r"^\[(\d+)\]\s*(.*)", line)
            if m and 0 <= int(m.group(1)) < len(segs):
                segs[int(m.group(1))]["text"] = " " + m.group(2).strip()

    q = _quality(segs)
    it = state.get("iteration", 0) + 1
    print(f"  [cleanup iter={it}] punct={q['punctuation_score']:.0%} filler={q['filler_rate']:.1%}")
    return {"segments": segs, "quality": q, "iteration": it,
            "log": state["log"] + [f"Cleanup pass {it} complete"]}


def diarize_node(state: FullPipelineState) -> dict:
    segs = copy.deepcopy(state["segments"])
    # Sample text per speaker
    samples: dict[str, str] = {}
    for seg in segs:
        spk = seg.get("speaker")
        if spk and len(samples.get(spk, "")) < 500:
            samples[spk] = samples.get(spk, "") + " " + seg["text"].strip()

    name_prompt = ChatPromptTemplate.from_messages([
        ("system",
         'Assign a role-based name to each speaker. Reply ONLY with JSON: '
         '{"SPEAKER_XX": "Name", ...}'),
        ("human", "{samples}"),
    ])
    name_chain = name_prompt | llm | StrOutputParser()

    prompt_text = "\n\n".join(f"[{spk}]: {t[:300]}" for spk, t in samples.items())
    raw_result = name_chain.invoke({"samples": prompt_text})

    names: dict[str, str] = {}
    try:
        names = json.loads(raw_result)
    except Exception:
        m = re.search(r"\{[^}]+\}", raw_result, re.DOTALL)
        if m:
            try:
                names = json.loads(m.group())
            except Exception:
                pass

    # Merge backchannels
    merged = []
    for seg in segs:
        wc = len(seg.get("words", []))
        if merged and wc <= 2 and merged[-1].get("speaker") == seg.get("speaker"):
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"].strip()
        else:
            merged.append(seg)

    print(f"  [diarize] {names} | merged {len(segs)-len(merged)} segs")
    return {"segments": merged, "speaker_names": names,
            "log": state["log"] + [f"Speakers named: {names}"]}


error_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Fix likely transcription errors (low-confidence, technical terms). "
     "Domain terms: JWT, authentication, devops, Node.js, analytics, dashboard. "
     "Return each line as [N] corrected text."),
    ("human", "{block}"),
])
error_chain = error_prompt | llm | StrOutputParser()


def error_correct_node(state: FullPipelineState) -> dict:
    segs = copy.deepcopy(state["segments"])
    # Only process segments with low-confidence words
    suspect_indices = [
        i for i, seg in enumerate(segs)
        if any(w.get("score", 1.0) < 0.72 for w in seg.get("words", []))
    ]

    if suspect_indices:
        block_text = "\n".join(
            f"[{i}] {segs[i]['text'].strip()}" for i in suspect_indices[:12]
        )
        result = error_chain.invoke({"block": block_text})
        for line in result.strip().split("\n"):
            m = re.match(r"^\[(\d+)\]\s*(.*)", line)
            if m and 0 <= int(m.group(1)) < len(segs):
                segs[int(m.group(1))]["text"] = " " + m.group(2).strip()

    print(f"  [errors]  reviewed {len(suspect_indices)} suspect segments")
    return {"segments": segs, "log": state["log"] + [f"Error correction: {len(suspect_indices)} segments reviewed"]}


def human_review_node(state: FullPipelineState) -> dict:
    """Pause for human approval before writing output files."""
    # Build a preview of the improved transcript
    segs = state["segments"]
    names = state.get("speaker_names", {})
    preview_lines = []
    prev_spk = None
    for seg in segs[:8]:
        spk = seg.get("speaker", "UNKNOWN")
        label = names.get(spk, spk)
        if spk != prev_spk:
            preview_lines.append(f"\n{label}:")
            prev_spk = spk
        preview_lines.append(f"  {seg['text'].strip()}")
    preview = "\n".join(preview_lines)

    print(f"  [review]  Pausing for human approval...")

    # interrupt() surfaces the preview to the caller
    decision = interrupt({
        "preview": preview,
        "quality": state["quality"],
        "instruction": "Type 'approve', 'reject', or write notes for revision",
    })

    approved = str(decision).strip().lower() in {"approve", "yes", "y", ""}
    print(f"  [review]  Decision: {'approved' if approved else 'rejected'}")
    return {
        "review_approved": approved,
        "review_notes": str(decision),
        "log": state["log"] + [f"Human review: {'approved' if approved else 'rejected'}"],
    }


def after_review(state: FullPipelineState) -> str:
    return "export" if state.get("review_approved") else END


def export_node(state: FullPipelineState) -> dict:
    """Write all output formats."""
    segs = state["segments"]
    names = state.get("speaker_names", {})
    raw = state["raw"]
    paths = []

    # JSON
    out = copy.deepcopy(raw)
    out["segments"] = segs
    out["speaker_names"] = names
    out["pipeline_log"] = state["log"]
    p = OUT_DIR / "transcript_pipeline_final.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    paths.append(str(p))

    # Plain text
    prev_spk = None
    lines = []
    for seg in segs:
        spk = seg.get("speaker", "UNKNOWN")
        label = names.get(spk, spk)
        if spk != prev_spk:
            lines.append(f"\n{label}:")
            prev_spk = spk
        lines.append(f"  {seg['text'].strip()}")
    p = OUT_DIR / "transcript_pipeline_final.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    paths.append(str(p))

    # SRT
    srt_lines = []
    for i, seg in enumerate(segs, 1):
        spk = seg.get("speaker", "UNKNOWN")
        label = names.get(spk, spk)
        srt_lines.append(f"{i}\n{_fmt_srt(seg['start'])} --> {_fmt_srt(seg['end'])}\n{label}: {seg['text'].strip()}")
    p = OUT_DIR / "transcript_pipeline_final.srt"
    p.write_text("\n\n".join(srt_lines), encoding="utf-8")
    paths.append(str(p))

    print(f"  [export]  Wrote {len(paths)} files")
    return {"output_paths": paths, "log": state["log"] + [f"Exported {len(paths)} files"]}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(FullPipelineState)
for name, fn in [("load", load_node), ("cleanup", cleanup_node),
                  ("diarize", diarize_node), ("error_correct", error_correct_node),
                  ("human_review", human_review_node), ("export", export_node)]:
    builder.add_node(name, fn)

builder.add_edge(START, "load")
builder.add_conditional_edges("load", should_improve, {"cleanup": "cleanup", "diarize": "diarize"})
builder.add_conditional_edges("cleanup", should_improve, {"cleanup": "cleanup", "diarize": "diarize"})
builder.add_edge("diarize", "error_correct")
builder.add_edge("error_correct", "human_review")
builder.add_conditional_edges("human_review", after_review, {"export": "export", END: END})
builder.add_edge("export", END)

pipeline = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("=" * 60)
print("  FULL ASR PIPELINE (with human review)")
print("=" * 60)

config = {"configurable": {"thread_id": "full_pipeline"}}

init_state = {
    "transcript_path": str(TRANSCRIPT_PATH),
    "raw": {}, "segments": [], "speaker_names": {},
    "quality": {}, "iteration": 0,
    "review_approved": False, "review_notes": "",
    "output_paths": [], "log": [],
}

# Run until interrupt
result = pipeline.invoke(init_state, config=config)

# Human review
print("\n  === PREVIEW ===")
if "__interrupt__" in str(result):
    print("  (Graph paused at human_review node)")

user_input = input("\n  Approve this transcript? (approve/reject) [approve]: ").strip() or "approve"

# Resume
final = pipeline.invoke(Command(resume=user_input), config=config)

print("\n  Pipeline complete!")
print("  Output files:")
for p in final.get("output_paths", []):
    print(f"    • {p}")
print("\n  Log:")
for entry in final.get("log", []):
    print(f"    • {entry}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Conditional entry: skip improvement loop if quality is already good
# ✅ error_correct runs after diarize so it sees speaker-labelled text
# ✅ human_review as interrupt() — human sees a preview before files are written
# ✅ export writes multiple formats in one node — JSON, TXT, SRT
# ✅ MemorySaver checkpoints every node — full audit trail preserved
# ✅ This is the template for a production ASR post-processing service
