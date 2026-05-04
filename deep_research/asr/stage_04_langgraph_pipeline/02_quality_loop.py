"""
ASR Stage 4, File 2: Quality-Gated Improvement Loop
=====================================================
CONCEPT: Running the pipeline repeatedly until quality targets are met.

Some transcripts need multiple passes to reach acceptable quality.
A single cleanup run may not be enough if:
  • The first pass introduced new errors (over-correction)
  • Punctuation added in pass 1 reveals new filler patterns in pass 2
  • Speaker assignment in pass 1 was wrong and needs revision

Solution: a conditional loop graph.
  assess → check thresholds → if failing: improve → reassess → loop
                                         if passing: report → END

This builds directly on File 1's graph, adding:
  - Quality threshold checking
  - Conditional edge to loop or exit
  - Iteration counter as a safety guard
  - Per-iteration delta tracking

Run this file:
  uv run deep_research/asr/stage_04_langgraph_pipeline/02_quality_loop.py
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"

llm = create_chat_model(
    "asr",
    temperature=0,
    max_tokens=4096,
)

# ---------------------------------------------------------------------------
# Quality thresholds (targets for the loop to reach)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "punctuation_score": 0.70,    # ≥70% of segments end with punctuation
    "filler_rate": 0.04,           # ≤4% of words are fillers
    "capitalisation_score": 0.70,  # ≥70% of segments start with capital
}
MAX_ITERATIONS = 3
FILLERS = {"uh", "um", "you know", "i mean", "like"}
CONTRACTION_MAP = {
    r"\barent\b": "aren't", r"\bcant\b": "can't", r"\bdont\b": "don't",
    r"\bills\b": "I'll",    r"\bill\b": "I'll",    r"\bthats\b": "that's",
    r"\blets\b": "let's",   r"\bwere\b": "we're",  r"\bim\b": "I'm",
    r"\bive\b": "I've",     r"\bits\b": "it's",    r"\bitll\b": "it'll",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class LoopState(TypedDict):
    segments: list[dict]
    iteration: int
    quality_history: list[dict]      # one entry per iteration
    failing_dimensions: list[str]    # which thresholds are not yet met
    done: bool
    stage_log: list[str]


# ---------------------------------------------------------------------------
# Quality computation
# ---------------------------------------------------------------------------

def compute_quality(segments: list[dict]) -> dict:
    words = [w for s in segments for w in s.get("words", [])]
    total_w = max(len(words), 1)
    total_s = max(len(segments), 1)

    avg_conf = sum(w.get("score", 1.0) for w in words) / total_w
    filler_c = sum(
        1 for s in segments for w in s.get("words", [])
        if w["word"].lower().strip(".,?!") in FILLERS
    )
    has_punct = sum(1 for s in segments if re.search(r"[.!?]$", s["text"].strip()))
    capped = sum(1 for s in segments if s["text"].strip() and s["text"].strip()[0].isupper())

    return {
        "avg_confidence": round(avg_conf, 3),
        "filler_rate": round(filler_c / total_w, 3),
        "punctuation_score": round(has_punct / total_s, 3),
        "capitalisation_score": round(capped / total_s, 3),
    }


def check_thresholds(quality: dict) -> list[str]:
    """Return list of failing dimension names."""
    failing = []
    if quality["punctuation_score"] < THRESHOLDS["punctuation_score"]:
        failing.append("punctuation")
    if quality["filler_rate"] > THRESHOLDS["filler_rate"]:
        failing.append("fillers")
    if quality["capitalisation_score"] < THRESHOLDS["capitalisation_score"]:
        failing.append("capitalisation")
    return failing


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def assess_node(state: LoopState) -> dict:
    """Measure quality and decide whether to loop."""
    q = compute_quality(state["segments"])
    failing = check_thresholds(q)
    history = state.get("quality_history", []) + [q]
    iteration = state.get("iteration", 0)

    print(f"  [assess iter={iteration}] "
          f"punct={q['punctuation_score']:.0%}  "
          f"filler={q['filler_rate']:.1%}  "
          f"cap={q['capitalisation_score']:.0%}  "
          f"failing={failing or 'none'}")

    return {
        "quality_history": history,
        "failing_dimensions": failing,
        "done": len(failing) == 0 or iteration >= MAX_ITERATIONS,
        "stage_log": state.get("stage_log", []) + [
            f"Iteration {iteration}: punct={q['punctuation_score']:.0%}, "
            f"filler={q['filler_rate']:.1%}, cap={q['capitalisation_score']:.0%}, "
            f"failing={failing}"
        ],
    }


# Targeted improvement prompts for each failing dimension
PUNCT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Add sentence-ending punctuation (. ! ?) and capitalise sentence starts. "
     "Return each line prefixed with its index [N], e.g. [0] Corrected text. "
     "Return only the corrected lines."),
    ("human", "{block}"),
])

FILLER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Remove filler words (uh, um, like, you know, I mean) from each line. "
     "Return each line prefixed with its index [N]. Return only the corrected lines."),
    ("human", "{block}"),
])

CAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Capitalise the first letter of each line and all proper nouns. "
     "Return each line prefixed with its index [N]. Return only the corrected lines."),
    ("human", "{block}"),
])

punct_chain = PUNCT_PROMPT | llm | StrOutputParser()
filler_chain = FILLER_PROMPT | llm | StrOutputParser()
cap_chain = CAP_PROMPT | llm | StrOutputParser()


def _apply_block_result(segs: list[dict], raw_result: str) -> list[dict]:
    """Parse [N] prefixed lines back into the segments list."""
    for line in raw_result.strip().split("\n"):
        m = re.match(r"^\[(\d+)\]\s*(.*)", line.strip())
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            if 0 <= idx < len(segs):
                segs[idx]["text"] = " " + text
    return segs


def improve_node(state: LoopState) -> dict:
    """Run targeted improvements only on failing dimensions."""
    segs = copy.deepcopy(state["segments"])
    failing = state.get("failing_dimensions", [])
    block_size = 8

    for start in range(0, len(segs), block_size):
        block = segs[start:start + block_size]
        block_text = "\n".join(f"[{start+i}] {s['text'].strip()}" for i, s in enumerate(block))

        if "punctuation" in failing or "capitalisation" in failing:
            result = punct_chain.invoke({"block": block_text})
            segs = _apply_block_result(segs, result)

        if "fillers" in failing:
            # Re-build block after punct pass
            block_text = "\n".join(f"[{start+i}] {segs[start+i]['text'].strip()}" for i in range(len(block)))
            result = filler_chain.invoke({"block": block_text})
            segs = _apply_block_result(segs, result)

        if "capitalisation" in failing:
            block_text = "\n".join(f"[{start+i}] {segs[start+i]['text'].strip()}" for i in range(len(block)))
            result = cap_chain.invoke({"block": block_text})
            segs = _apply_block_result(segs, result)

    print(f"  [improve] Fixed: {failing}")
    return {
        "segments": segs,
        "iteration": state.get("iteration", 0) + 1,
        "stage_log": state.get("stage_log", []) + [f"Improvement pass on: {failing}"],
    }


def report_node(state: LoopState) -> dict:
    """Summarise the quality progression across all iterations."""
    history = state["quality_history"]
    print(f"\n  [report] Completed in {len(history)} iteration(s)")
    print(f"  {'Iter':<5} {'Punct':>7} {'Filler':>7} {'Cap':>7}")
    print("  " + "-" * 30)
    for i, q in enumerate(history):
        print(f"  {i:<5} {q['punctuation_score']:>7.1%} {q['filler_rate']:>7.1%} "
              f"{q['capitalisation_score']:>7.1%}")

    first, last = history[0], history[-1]
    print()
    print(f"  Punctuation improvement : {last['punctuation_score'] - first['punctuation_score']:+.1%}")
    print(f"  Filler reduction        : {first['filler_rate'] - last['filler_rate']:+.1%}")
    print(f"  Capitalisation gain     : {last['capitalisation_score'] - first['capitalisation_score']:+.1%}")
    return {}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def should_continue(state: LoopState) -> str:
    if state.get("done"):
        return "report"
    return "improve"


# ---------------------------------------------------------------------------
# Build the loop graph
# ---------------------------------------------------------------------------

builder = StateGraph(LoopState)
builder.add_node("assess",  assess_node)
builder.add_node("improve", improve_node)
builder.add_node("report",  report_node)

builder.add_edge(START, "assess")
builder.add_conditional_edges("assess", should_continue, {"improve": "improve", "report": "report"})
builder.add_edge("improve", "assess")   # ← the loop
builder.add_edge("report", END)

loop_pipeline = builder.compile(checkpointer=MemorySaver())

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("=" * 60)
print("  QUALITY-GATED IMPROVEMENT LOOP")
print(f"  Targets: punct≥{THRESHOLDS['punctuation_score']:.0%}  "
      f"filler≤{THRESHOLDS['filler_rate']:.0%}  "
      f"cap≥{THRESHOLDS['capitalisation_score']:.0%}")
print("=" * 60)

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)

config = {"configurable": {"thread_id": "asr_loop_demo"}}
final = loop_pipeline.invoke({
    "segments": copy.deepcopy(raw["segments"]),
    "iteration": 0,
    "quality_history": [],
    "failing_dimensions": [],
    "done": False,
    "stage_log": [],
}, config=config)

print()
print("  Stage log:")
for entry in final["stage_log"]:
    print(f"    • {entry}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Quality-gated loops prevent premature exit — run until targets are met
# ✅ Targeted improvement: fix only failing dimensions, leave passing ones alone
# ✅ MAX_ITERATIONS is a hard safety cap — never run indefinitely
# ✅ quality_history tracks progress — proves the loop is converging
# ✅ MemorySaver checkpoints every iteration — you can inspect any step
# ✅ The loop pattern is reusable: swap the thresholds dict to change targets
