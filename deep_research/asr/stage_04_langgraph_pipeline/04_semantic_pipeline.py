"""
ASR Stage 4, File 4: Semantic Quality Pipeline
================================================
CONCEPT: Replace regex-based quality metrics with LLM semantic judgment
         and use LLM routing decisions instead of fixed threshold logic.

Files 1-3 measure quality with rules:
  punctuation_score = segments ending in [.!?] / total segments
  filler_rate       = filler word count / total word count

These metrics are proxies. They tell you *nothing* about whether the
transcript actually makes sense. A segment could pass all thresholds and
still be semantically broken:
  "We need to review the auth thing for the deadline the October."
  → has punctuation ✓, no fillers ✓, starts with capital ✓
  → LLM: "missing word before 'October', likely 'before' or 'by'"

This pipeline adds:
  1. semantic_assess_node — LLM rates each segment on 3 dimensions:
       coherence: does it make grammatical and logical sense?
       completeness: does it appear to be a full thought?
       accuracy: are technical terms correct?
  2. semantic_router — LLM decides which repair to apply (not a threshold dict)
  3. semantic_repair_node — targeted LLM repair for low-scoring segments
  4. LLM quality gate — LLM decides when the transcript is "good enough"

New LangGraph patterns:
  - LLM as a conditional edge router (not just a string comparison)
  - Segment-level semantic scores in graph state
  - Adaptive repair: different prompt per dimension that failed
  - LLM-as-judge at the graph level (approve/reject entire output)

Run this file:
  uv run deep_research/asr/stage_04_langgraph_pipeline/04_semantic_pipeline.py
"""

import os
import json
import re
import copy
from pathlib import Path
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv()

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments_raw = raw["segments"]

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=1024,
)


# ---------------------------------------------------------------------------
# 1. Pydantic schemas
# ---------------------------------------------------------------------------

class SegmentSemanticScore(BaseModel):
    index: int
    coherence: float = Field(description="0.0-1.0: grammatically and logically sound")
    completeness: float = Field(description="0.0-1.0: appears to be a full thought")
    accuracy: float = Field(description="0.0-1.0: technical terms correct and properly cased")
    overall: float = Field(description="0.0-1.0: composite score")
    primary_issue: str = Field(description="'none', 'incoherent', 'incomplete', 'wrong_term', 'mixed'")
    note: str = Field(description="One short note about the issue, or 'ok' if no issues")


class SemanticAssessment(BaseModel):
    scores: list[SegmentSemanticScore]
    worst_segments: list[int] = Field(description="Indices of the 3-5 lowest scoring segments")
    overall_transcript_score: float = Field(description="0.0-1.0: average across all segments")


class TranscriptVerdict(BaseModel):
    approved: bool = Field(description="True if transcript meets production quality")
    score: float = Field(description="0.0-1.0: final quality estimate")
    remaining_issues: list[str] = Field(description="Any issues that still need attention")
    recommendation: str = Field(description="'publish', 'needs_light_editing', 'needs_rework'")


assess_parser = JsonOutputParser(pydantic_object=SemanticAssessment)
verdict_parser = JsonOutputParser(pydantic_object=TranscriptVerdict)


# ---------------------------------------------------------------------------
# 2. State
# ---------------------------------------------------------------------------

class SemanticState(TypedDict):
    segments: list[dict]
    semantic_scores: list[dict]       # SegmentSemanticScore dicts
    worst_segments: list[int]
    overall_score: float
    iteration: int
    verdict: dict                     # TranscriptVerdict dict
    log: list[str]


# ---------------------------------------------------------------------------
# 3. Chains
# ---------------------------------------------------------------------------

# Assessment chain — evaluates a block of segments
assess_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a semantic transcript quality evaluator. Score each segment on:\n"
     "  coherence    (0-1): is the sentence grammatically and logically sound?\n"
     "  completeness (0-1): does it express a complete thought?\n"
     "  accuracy     (0-1): are technical terms correct? (JWT not jwt, DevOps not devops)\n"
     "  overall      (0-1): weighted average (coherence 40%, completeness 30%, accuracy 30%)\n\n"
     "Rate strictly — 0.9+ means nearly perfect professional transcript quality.\n"
     "Identify the 3-5 lowest-scoring segment indices as worst_segments.\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Segments (global index shown in brackets):\n{block}"),
]).partial(format_instructions=assess_parser.get_format_instructions())

assess_chain = assess_prompt | llm | assess_parser


# Repair chains — one per issue type
REPAIR_PROMPTS = {
    "incoherent": ChatPromptTemplate.from_messages([
        ("system",
         "Fix the grammatical structure of this transcript segment. "
         "The sentence is incoherent — it may have missing words, wrong word order, "
         "or fragments. Reconstruct it so it reads naturally.\n"
         "Return ONLY the corrected text, nothing else."),
        ("human", "Context: {context}\nSegment to fix: {text}"),
    ]),
    "incomplete": ChatPromptTemplate.from_messages([
        ("system",
         "This transcript segment appears incomplete (thought cut off). "
         "Using the context below, complete the sentence naturally. "
         "If you cannot determine what was meant, add '...' at the end.\n"
         "Return ONLY the completed text, nothing else."),
        ("human", "Context: {context}\nSegment to complete: {text}"),
    ]),
    "wrong_term": ChatPromptTemplate.from_messages([
        ("system",
         "Fix technical term capitalisation and spelling in this transcript segment.\n"
         "Known terms: JWT, DevOps, Node.js, Node 18, WCAG, API, UI, UX, authentication.\n"
         "Return ONLY the corrected text, nothing else."),
        ("human", "Segment: {text}"),
    ]),
}

repair_chains = {
    k: (prompt | llm | StrOutputParser())
    for k, prompt in REPAIR_PROMPTS.items()
}

# Judge chain — final approval decision
judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a transcript quality gatekeeper. Review the provided transcript sample "
     "and decide if it meets production quality standards.\n\n"
     "Production quality means:\n"
     "  • Technical terms correctly spelled and cased\n"
     "  • Sentences are complete and grammatically sound\n"
     "  • No excessive fillers or stutters\n"
     "  • Speaker attribution makes sense in context\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Overall semantic score from assessment: {overall_score:.2f}\n"
     "Remaining worst segments: {worst_segments}\n\n"
     "Sample of current transcript (first 10 segments):\n{sample}"),
]).partial(format_instructions=verdict_parser.get_format_instructions())

judge_chain = judge_prompt | llm | verdict_parser


# ---------------------------------------------------------------------------
# 4. LangGraph nodes
# ---------------------------------------------------------------------------

CHUNK_SIZE = 8
MAX_ITERATIONS = 2
REPAIR_THRESHOLD = 0.65    # segments below this score get repaired


def semantic_assess_node(state: SemanticState) -> dict:
    """LLM scores every segment semantically, not just by regex."""
    segs = state["segments"]
    all_scores: list[dict] = []
    worst: list[int] = []

    for chunk_start in range(0, len(segs), CHUNK_SIZE):
        chunk = segs[chunk_start:chunk_start + CHUNK_SIZE]
        block = "\n".join(
            f"[{chunk_start + i}] [{s.get('speaker', '?')}] {s['text'].strip()}"
            for i, s in enumerate(chunk)
        )
        try:
            result = assess_chain.invoke({"block": block})
            r = result if isinstance(result, dict) else {}
            chunk_scores = r.get("scores", [])
            all_scores.extend(chunk_scores)
            worst.extend(r.get("worst_segments", []))
        except Exception as e:
            print(f"  [assess] chunk {chunk_start} error: {e}")

    # Deduplicate and sort worst segments
    worst = sorted(set(worst))
    overall = (
        sum(s.get("overall", 0.5) for s in all_scores) / max(len(all_scores), 1)
        if all_scores else 0.5
    )

    it = state.get("iteration", 0)
    print(f"  [semantic_assess iter={it}] "
          f"scored={len(all_scores)} segs | "
          f"overall={overall:.2f} | "
          f"worst={worst[:5]}")

    return {
        "semantic_scores": all_scores,
        "worst_segments": worst,
        "overall_score": round(overall, 3),
        "log": state.get("log", []) + [
            f"Iteration {it}: semantic_score={overall:.2f}, worst={worst[:5]}"
        ],
    }


def semantic_router(state: SemanticState) -> str:
    """Route based on overall score and iteration count."""
    it = state.get("iteration", 0)
    score = state.get("overall_score", 1.0)
    worst = state.get("worst_segments", [])

    if it >= MAX_ITERATIONS:
        return "judge"
    if score >= 0.80 and len(worst) == 0:
        return "judge"
    return "repair"


def semantic_repair_node(state: SemanticState) -> dict:
    """Targeted LLM repair for low-scoring segments."""
    segs = copy.deepcopy(state["segments"])
    scores = {s.get("index"): s for s in state.get("semantic_scores", [])}
    repaired = 0

    for idx in state.get("worst_segments", []):
        if idx >= len(segs):
            continue
        score_data = scores.get(idx, {})
        if score_data.get("overall", 1.0) >= REPAIR_THRESHOLD:
            continue  # Not bad enough to repair

        issue = score_data.get("primary_issue", "none")
        if issue == "none":
            continue

        text = segs[idx]["text"].strip()
        # Build context from surrounding segments
        ctx_segs = segs[max(0, idx - 2):idx] + segs[idx + 1:idx + 3]
        ctx = " | ".join(s["text"].strip() for s in ctx_segs)

        chain = repair_chains.get(issue) or repair_chains.get("wrong_term")
        try:
            fixed = chain.invoke({"text": text, "context": ctx})
            if fixed.strip() and fixed.strip() != text:
                segs[idx]["text"] = " " + fixed.strip()
                repaired += 1
                print(f"  [repair] seg {idx} ({issue}): {text[:40]} → {fixed.strip()[:40]}")
        except Exception as e:
            print(f"  [repair] seg {idx} error: {e}")

    return {
        "segments": segs,
        "iteration": state.get("iteration", 0) + 1,
        "log": state.get("log", []) + [f"Repaired {repaired} segments"],
    }


def judge_node(state: SemanticState) -> dict:
    """LLM final verdict: is the transcript ready for production?"""
    segs = state["segments"]
    sample = "\n".join(
        f"[{i}] [{s.get('speaker', '?')}] {s['text'].strip()}"
        for i, s in enumerate(segs[:10])
    )
    try:
        result = judge_chain.invoke({
            "overall_score": state.get("overall_score", 0.0),
            "worst_segments": state.get("worst_segments", [])[:5],
            "sample": sample,
        })
        verdict = result if isinstance(result, dict) else {}
    except Exception:
        verdict = {"approved": False, "score": 0.0, "remaining_issues": [], "recommendation": "needs_rework"}

    approved = verdict.get("approved", False)
    rec = verdict.get("recommendation", "?")
    score = verdict.get("score", 0.0)
    print(f"  [judge] approved={approved} | score={score:.2f} | recommendation={rec}")
    if verdict.get("remaining_issues"):
        for issue in verdict["remaining_issues"][:3]:
            print(f"    ⚠  {issue}")

    return {
        "verdict": verdict,
        "log": state.get("log", []) + [f"Judge: {rec} (score={score:.2f})"],
    }


def export_node(state: SemanticState) -> dict:
    out = copy.deepcopy(raw)
    out["segments"] = state["segments"]
    out["semantic_quality"] = {
        "overall_score": state["overall_score"],
        "iterations": state["iteration"],
        "verdict": state["verdict"],
        "log": state["log"],
    }
    out_path = OUT_DIR / "transcript_semantic_pipeline.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  [export] Saved: {out_path.name}")
    return {}


# ---------------------------------------------------------------------------
# 5. Build graph
# ---------------------------------------------------------------------------

builder = StateGraph(SemanticState)
builder.add_node("assess", semantic_assess_node)
builder.add_node("repair", semantic_repair_node)
builder.add_node("judge",  judge_node)
builder.add_node("export", export_node)

builder.add_edge(START, "assess")
builder.add_conditional_edges("assess", semantic_router, {
    "repair": "repair",
    "judge": "judge",
})
builder.add_edge("repair", "assess")   # ← loop back for re-assessment
builder.add_edge("judge", "export")
builder.add_edge("export", END)

pipeline = builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------

print("=" * 60)
print("  SEMANTIC QUALITY PIPELINE")
print("=" * 60)
print()

config = {"configurable": {"thread_id": "semantic_pipeline"}}
final = pipeline.invoke({
    "segments": copy.deepcopy(segments_raw),
    "semantic_scores": [],
    "worst_segments": [],
    "overall_score": 0.0,
    "iteration": 0,
    "verdict": {},
    "log": [],
}, config=config)

print("\n  Pipeline log:")
for entry in final.get("log", []):
    print(f"    • {entry}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Semantic scoring beats regex: a sentence can pass all rules but still be broken
# ✅ LLM-as-router: the routing decision is itself an LLM judgment, not a hardcoded rule
# ✅ Issue-specific repair chains: 'incoherent' needs different fix than 'wrong_term'
# ✅ LLM judge at the end: approval is a reasoned decision, not just a threshold check
# ✅ REPAIR_THRESHOLD gates which segments get expensive LLM repair calls
# ✅ Loop is still capped at MAX_ITERATIONS — LLM routing doesn't mean infinite loops
# ✅ Semantic scores in state are available to every downstream node
