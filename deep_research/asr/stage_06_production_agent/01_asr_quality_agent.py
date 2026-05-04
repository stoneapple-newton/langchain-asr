"""
ASR Stage 6: Production ASR Quality Agent
==========================================
CONCEPT: A fully autonomous multi-agent system that takes a raw WhisperX
         transcript from 0 to production-ready output.

This is the capstone of the ASR series. It combines everything:
  Stage 1 → Data parsing and quality metrics
  Stage 2 → LLM-powered punctuation, naming, error correction
  Stage 3 → Diarization analysis and correction
  Stage 4 → LangGraph pipeline with quality loop and HITL
  Stage 5 → RAG-based domain-aware correction

Architecture: supervisor + three specialist workers

  ┌─────────────────────────────────────────────────────┐
  │                    SUPERVISOR                       │
  │  Plans work, routes to workers, checks quality,    │
  │  decides when to stop iterating                     │
  └─────────────┬────────────┬──────────────────────────┘
                │            │                │
    ┌───────────▼──┐  ┌──────▼───────┐  ┌───▼──────────┐
    │   CLEANER    │  │  DIARIZER    │  │   EXPORTER   │
    │ Punctuation  │  │  Speaker     │  │ JSON/TXT/SRT │
    │ Fillers      │  │  naming      │  │ output files │
    │ Contractions │  │  Backchannel │  │              │
    └──────────────┘  │  merge       │  └──────────────┘
                      └──────────────┘

Quality gates run between every worker call.
The supervisor iterates until targets are met or MAX_ROUNDS is reached.

Run this file:
  uv run deep_research/asr/stage_06_production_agent/01_asr_quality_agent.py
"""

import os
import json
import re
import copy
from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict
from datetime import timedelta

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, create_embeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

llm = create_chat_model(
    "asr",
    temperature=0,
    max_tokens=4096,
)
embeddings = create_embeddings("asr")

MAX_ROUNDS = 3
QUALITY_TARGETS = {
    "punctuation_score": 0.65,
    "filler_rate_max": 0.05,
    "capitalisation_score": 0.65,
}

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

FILLERS = {"uh", "um", "you know", "i mean"}
CONTRACTION_MAP = {
    r"\barent\b": "aren't", r"\bcant\b": "can't", r"\bdont\b": "don't",
    r"\bills\b": "I'll",    r"\bill\b": "I'll",    r"\bthats\b": "that's",
    r"\blets\b": "let's",   r"\bwere\b": "we're",  r"\bim\b": "I'm",
    r"\bive\b": "I've",     r"\bits\b": "it's",    r"\bitll\b": "it'll",
}

CONTEXT_DOCS = [
    Document(page_content="JWT: JSON Web Token. DevOps (one word, capitalised). Node.js version 18.",
             metadata={"cat": "terms"}),
    Document(page_content="Participants: Alice Chen (Product Manager, SPEAKER_00), "
             "Bob Nakamura (Engineer, SPEAKER_01), Carol Rivera (Designer, SPEAKER_02).",
             metadata={"cat": "participants"}),
    Document(page_content="Projects: Dashboard v2 (deadline Oct 15), Auth Service JWT bug fix, "
             "Analytics event tracking spec, Node 16→18 upgrade via DevOps.",
             metadata={"cat": "projects"}),
]
ctx_store = InMemoryVectorStore.from_documents(CONTEXT_DOCS, embeddings)
ctx_retriever = ctx_store.as_retriever(search_kwargs={"k": 2})


def _quality(segments: list[dict]) -> dict:
    words = [w for s in segments for w in s.get("words", [])]
    n = max(len(words), 1)
    ns = max(len(segments), 1)
    filler_c = sum(1 for s in segments for w in s.get("words", [])
                   if w["word"].lower().strip(".,?!") in FILLERS)
    has_punct = sum(1 for s in segments if re.search(r"[.!?]$", s["text"].strip()))
    capped = sum(1 for s in segments if s["text"].strip() and s["text"].strip()[0].isupper())
    return {
        "avg_confidence": round(sum(w.get("score", 1) for w in words) / n, 3),
        "filler_rate": round(filler_c / n, 3),
        "punctuation_score": round(has_punct / ns, 3),
        "capitalisation_score": round(capped / ns, 3),
        "total_segments": ns,
    }


def _passes_targets(q: dict) -> bool:
    return (
        q["punctuation_score"] >= QUALITY_TARGETS["punctuation_score"]
        and q["filler_rate"] <= QUALITY_TARGETS["filler_rate_max"]
        and q["capitalisation_score"] >= QUALITY_TARGETS["capitalisation_score"]
    )


def _fmt_srt(sec: float) -> str:
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Data
    raw: dict
    segments: list[dict]
    speaker_names: dict[str, str]

    # Progress tracking
    round: int
    quality_before: dict
    quality_now: dict
    quality_history: list[dict]
    done: bool

    # Output
    output_paths: list[str]
    log: list[str]


# ---------------------------------------------------------------------------
# Worker nodes
# ---------------------------------------------------------------------------

# -- CLEANER --

_cleanup_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Fix this block of transcription lines:\n"
     "  1. Add punctuation (. , ? !)\n"
     "  2. Capitalise sentence starts and proper nouns\n"
     "  3. Remove fillers (uh, um, you know, i mean)\n"
     "  4. Fix contractions (arent→aren't, cant→can't, lets→let's, ill→I'll, etc.)\n"
     "Context: {context}\n\n"
     "Return each line as [N] corrected text. Same count as input."),
    ("human", "{block}"),
])
_cleanup_chain = _cleanup_prompt | llm | StrOutputParser()


def _rule_clean(text: str) -> str:
    t = re.sub(r"\b(uh|um|you know|i mean)\s*", "", text.strip(), flags=re.IGNORECASE)
    for pat, rep in CONTRACTION_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", t).strip()


def _apply_block(segs: list[dict], raw_result: str) -> list[dict]:
    for line in raw_result.strip().split("\n"):
        m = re.match(r"^\[(\d+)\]\s*(.*)", line.strip())
        if m and 0 <= int(m.group(1)) < len(segs):
            segs[int(m.group(1))]["text"] = " " + m.group(2).strip()
    return segs


def cleaner_node(state: AgentState) -> dict:
    segs = copy.deepcopy(state["segments"])
    for seg in segs:
        seg["text"] = " " + _rule_clean(seg["text"])

    block_size = 8
    for start in range(0, len(segs), block_size):
        block = segs[start:start + block_size]
        block_text = "\n".join(f"[{start+i}] {s['text'].strip()}" for i, s in enumerate(block))
        ctx_query = " ".join(s["text"].strip().split()[:8])
        ctx_docs = ctx_retriever.invoke(ctx_query)
        ctx_text = " | ".join(d.page_content[:100] for d in ctx_docs)
        result = _cleanup_chain.invoke({"context": ctx_text, "block": block_text})
        segs = _apply_block(segs, result)

    q = _quality(segs)
    rnd = state.get("round", 0)
    print(f"  [cleaner  r={rnd}] punct={q['punctuation_score']:.0%}  "
          f"filler={q['filler_rate']:.1%}  cap={q['capitalisation_score']:.0%}")
    return {
        "segments": segs,
        "quality_now": q,
        "quality_history": state.get("quality_history", []) + [q],
        "log": state.get("log", []) + [
            f"Round {rnd} cleaner: punct={q['punctuation_score']:.0%} filler={q['filler_rate']:.1%}"
        ],
    }


# -- DIARIZER --

_name_prompt = ChatPromptTemplate.from_messages([
    ("system",
     'Assign display names to speakers. Reply ONLY with JSON: '
     '{"SPEAKER_00": "Name", ...}. Use context for names if available.'),
    ("human", "Context: {context}\n\nSamples:\n{samples}"),
])
_name_chain = _name_prompt | llm | StrOutputParser()


def diarizer_node(state: AgentState) -> dict:
    segs = copy.deepcopy(state["segments"])

    # Build samples
    samples: dict[str, str] = {}
    for seg in segs:
        spk = seg.get("speaker")
        if spk and len(samples.get(spk, "")) < 400:
            samples[spk] = samples.get(spk, "") + " " + seg["text"].strip()

    ctx_docs = ctx_retriever.invoke("participants names roles")
    ctx_text = " | ".join(d.page_content[:150] for d in ctx_docs)
    sample_text = "\n".join(f"[{k}]: {v[:250]}" for k, v in samples.items())

    raw_r = _name_chain.invoke({"context": ctx_text, "samples": sample_text})
    names: dict[str, str] = {}
    try:
        names = json.loads(raw_r)
    except Exception:
        m = re.search(r"\{[^}]+\}", raw_r, re.DOTALL)
        if m:
            try:
                names = json.loads(m.group())
            except Exception:
                pass

    # Merge backchannels
    merged: list[dict] = []
    for seg in segs:
        wc = len(seg.get("words", []))
        if merged and wc <= 2 and seg["end"] - seg["start"] < 2.0 and merged[-1].get("speaker") == seg.get("speaker"):
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"].strip()
        else:
            merged.append(seg)

    rnd = state.get("round", 0)
    print(f"  [diarizer r={rnd}] names={names} | merged {len(segs)-len(merged)} segs")
    return {
        "segments": merged,
        "speaker_names": names,
        "log": state.get("log", []) + [f"Round {rnd} diarizer: {names}"],
    }


# -- EXPORTER --

def exporter_node(state: AgentState) -> dict:
    segs = state["segments"]
    names = state.get("speaker_names", {})
    raw = state["raw"]
    paths = []

    # JSON
    out = copy.deepcopy(raw)
    out["segments"] = segs
    out["speaker_names"] = names
    out["quality_history"] = state.get("quality_history", [])
    out["pipeline_log"] = state.get("log", [])
    p = OUT_DIR / "asr_agent_output.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    paths.append(str(p))

    # TXT
    prev_spk = None
    lines = []
    for seg in segs:
        spk = seg.get("speaker", "UNKNOWN")
        label = names.get(spk, spk)
        if spk != prev_spk:
            lines.append(f"\n{label}:")
            prev_spk = spk
        lines.append(f"  {seg['text'].strip()}")
    p = OUT_DIR / "asr_agent_output.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    paths.append(str(p))

    # SRT
    srt_blocks = []
    for i, seg in enumerate(segs, 1):
        spk = names.get(seg.get("speaker", ""), seg.get("speaker", ""))
        srt_blocks.append(
            f"{i}\n{_fmt_srt(seg['start'])} --> {_fmt_srt(seg['end'])}\n{spk}: {seg['text'].strip()}"
        )
    p = OUT_DIR / "asr_agent_output.srt"
    p.write_text("\n\n".join(srt_blocks), encoding="utf-8")
    paths.append(str(p))

    # Quality summary
    q_before = state.get("quality_before", {})
    q_now = state.get("quality_now", {})
    summary = [
        "# ASR Quality Agent — Output Summary",
        f"Rounds completed : {state.get('round', 0)}",
        f"Segments         : {len(segs)}",
        "",
        "Quality (before → after):",
        f"  Confidence     : {q_before.get('avg_confidence', 0):.3f} → {q_now.get('avg_confidence', 0):.3f}",
        f"  Punctuation    : {q_before.get('punctuation_score', 0):.0%} → {q_now.get('punctuation_score', 0):.0%}",
        f"  Filler rate    : {q_before.get('filler_rate', 0):.1%} → {q_now.get('filler_rate', 0):.1%}",
        f"  Capitalisation : {q_before.get('capitalisation_score', 0):.0%} → {q_now.get('capitalisation_score', 0):.0%}",
        "",
        "Speaker names:",
        *[f"  {k} → {v}" for k, v in names.items()],
        "",
        "Stage log:",
        *[f"  • {e}" for e in state.get("log", [])],
    ]
    p = OUT_DIR / "asr_agent_summary.md"
    p.write_text("\n".join(summary), encoding="utf-8")
    paths.append(str(p))

    print(f"  [exporter]  Wrote {len(paths)} output files")
    return {"output_paths": paths, "log": state.get("log", []) + [f"Exported {len(paths)} files"]}


# ---------------------------------------------------------------------------
# Supervisor router
# ---------------------------------------------------------------------------

def supervisor(state: AgentState) -> str:
    """Decide what to do next based on quality and round number."""
    rnd = state.get("round", 0)
    q = state.get("quality_now", {})
    done = state.get("done", False)

    if done:
        return "export"

    if rnd == 0:
        # First pass: always clean and diarize
        return "cleaner"

    if not _passes_targets(q) and rnd < MAX_ROUNDS:
        return "cleaner"

    # Quality acceptable or max rounds reached → diarize then export
    return "diarize_then_export"


def after_cleaner(state: AgentState) -> str:
    """After cleaning: either loop or move to diarizer."""
    q = state.get("quality_now", {})
    rnd = state.get("round", 0) + 1
    if _passes_targets(q) or rnd >= MAX_ROUNDS:
        return "diarize"
    return "cleaner"


def increment_round(state: AgentState) -> dict:
    """Increment round counter after each clean pass."""
    return {"round": state.get("round", 0) + 1}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("cleaner",  cleaner_node)
builder.add_node("inc_round", increment_round)
builder.add_node("diarizer", diarizer_node)
builder.add_node("exporter", exporter_node)

builder.add_edge(START, "cleaner")
builder.add_edge("cleaner", "inc_round")
builder.add_conditional_edges(
    "inc_round", after_cleaner, {"cleaner": "cleaner", "diarize": "diarizer"}
)
builder.add_edge("diarizer", "exporter")
builder.add_edge("exporter", END)

agent = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("=" * 60)
print("  ASR QUALITY AGENT")
print(f"  Targets: punct≥{QUALITY_TARGETS['punctuation_score']:.0%}  "
      f"filler≤{QUALITY_TARGETS['filler_rate_max']:.0%}  "
      f"cap≥{QUALITY_TARGETS['capitalisation_score']:.0%}")
print("=" * 60)

with open(TRANSCRIPT_PATH) as f:
    raw_data = json.load(f)

initial_segs = copy.deepcopy(raw_data["segments"])
q_init = _quality(initial_segs)

print(f"\n  Baseline: conf={q_init['avg_confidence']:.3f}  "
      f"punct={q_init['punctuation_score']:.0%}  "
      f"filler={q_init['filler_rate']:.1%}  "
      f"cap={q_init['capitalisation_score']:.0%}\n")

config = {"configurable": {"thread_id": "asr_agent_run"}}
final = agent.invoke({
    "raw": raw_data,
    "segments": initial_segs,
    "speaker_names": {},
    "round": 0,
    "quality_before": q_init,
    "quality_now": q_init,
    "quality_history": [q_init],
    "done": False,
    "output_paths": [],
    "log": ["Agent started"],
}, config=config)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  AGENT COMPLETE")
print("=" * 60)

q_b = final["quality_before"]
q_a = final["quality_now"]
print(f"\n  {'Metric':<22} {'Before':>8} {'After':>8} {'Delta':>8}")
print("  " + "-" * 50)
for key, label in [
    ("punctuation_score", "Punctuation"),
    ("filler_rate", "Filler rate"),
    ("capitalisation_score", "Capitalisation"),
    ("avg_confidence", "Avg confidence"),
]:
    b, a = q_b.get(key, 0), q_a.get(key, 0)
    print(f"  {label:<22} {b:>8.1%} {a:>8.1%} {a-b:>+8.1%}")

print(f"\n  Rounds used: {final['round']} / {MAX_ROUNDS}")
print(f"  Speaker map: {final['speaker_names']}")
print("\n  Output files:")
for p in final["output_paths"]:
    print(f"    • {p}")
print("\n  Full log:")
for entry in final["log"]:
    print(f"    • {entry}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS — THE FULL ASR PIPELINE
# ---------------------------------------------------------------------------
# ✅ Stage 1: Parse + quality metrics = your baseline and the work queue
# ✅ Stage 2: LLM punctuation, naming, error correction = the core improvements
# ✅ Stage 3: Diarization analysis + correction = speaker accuracy
# ✅ Stage 4: LangGraph orchestration = reliable, inspectable, loopable
# ✅ Stage 5: RAG context = domain accuracy for technical/proper terms
# ✅ Stage 6: Supervisor + workers = autonomous quality improvement
#
# Production extensions to build next:
#   • Async FastAPI endpoint — POST a JSON file, get improved transcript back
#   • LangSmith tracing — debug every LLM call with full context
#   • Chroma speaker profile store — accumulate knowledge across meetings
#   • Confidence threshold auto-tuning — learn from human corrections
#   • Multi-language support — detect language per segment and route accordingly
