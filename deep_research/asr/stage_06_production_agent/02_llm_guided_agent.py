"""
ASR Stage 6, File 2: LLM-Guided Production Agent
==================================================
CONCEPT: LLM is not just a worker — it's the planner and supervisor.

Stage 6 File 1 runs a fixed pipeline: clean → diarize → export.
The LLM executes tasks but doesn't decide what to do.

In this file the LLM takes on two new roles:

  PLANNER (start of pipeline):
    Reads the transcript and outputs a structured RemediationPlan:
      • Which problem categories are present? (technical terms, speaker confusion, fluency)
      • Which segments need attention?
      • What sequence of workers should run?
    This plan drives all downstream routing decisions.

  SUPERVISOR (between workers):
    After each worker completes a pass, the Supervisor LLM evaluates
    the output quality and decides:
      • "continue_cleaning" — more cleanup needed
      • "run_technical"    — technical term worker needed
      • "run_fluency"      — fluency worker needed
      • "run_diarize"      — diarization correction needed
      • "export"           — quality is sufficient, export now

Three specialist workers (each has a targeted LLM prompt):
  • cleaner_node    — punctuation, fillers, contractions (broad)
  • technical_node  — domain terminology and proper nouns (precise)
  • fluency_node    — sentence structure and disfluency (reconstructive)

New LangGraph patterns:
  - Plan-first architecture: planner node produces structured plan
  - LLM as supervisor/router: routing decision comes from LLM, not thresholds
  - Worker pool: multiple specialist nodes the supervisor can select from
  - Plan field in state: all nodes read the plan to focus their work
  - Supervisor decision history: prevents infinite loops by tracking past decisions

Run this file:
  uv run deep_research/asr/stage_06_production_agent/02_llm_guided_agent.py
"""

import os
import json
import re
import copy
from pathlib import Path
from datetime import timedelta
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv()

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=1024,
)
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

# ---------------------------------------------------------------------------
# 1. RAG context store (terminology + participants)
# ---------------------------------------------------------------------------

CONTEXT_DOCS = [
    Document(page_content="JWT = JSON Web Token. Always uppercase. Auth service has JWT token invalidation bug.",
             metadata={"id": "jwt"}),
    Document(page_content="DevOps: one capitalised word. Node 16→18 upgrade needs DevOps approval.",
             metadata={"id": "devops"}),
    Document(page_content="Alice Chen: Product Manager. Owns roadmap. Analytics Integration project.",
             metadata={"id": "alice"}),
    Document(page_content="Bob Nakamura: Senior Engineer. Authentication service, backend.",
             metadata={"id": "bob"}),
    Document(page_content="Carol Rivera: UX Designer. Dashboard, accessibility, WCAG, breakpoints.",
             metadata={"id": "carol"}),
    Document(page_content="Dashboard v2: analytics dashboard. Hard deadline October 15th. Carol+Bob.",
             metadata={"id": "dashboard"}),
    Document(page_content="Staging: pre-production test environment. WCAG: Web Content Accessibility Guidelines.",
             metadata={"id": "terms"}),
]
context_store = InMemoryVectorStore.from_documents(CONTEXT_DOCS, embeddings)
context_retriever = context_store.as_retriever(search_kwargs={"k": 3})


# ---------------------------------------------------------------------------
# 2. Pydantic schemas
# ---------------------------------------------------------------------------

class RemediationPlan(BaseModel):
    problem_categories: list[str] = Field(
        description="Categories present: technical_terms, speaker_confusion, disfluency, punctuation, incomplete_sentences"
    )
    high_priority_segments: list[int] = Field(
        description="Indices of segments needing most attention (max 10)"
    )
    recommended_worker_sequence: list[str] = Field(
        description="Ordered list of workers to run: cleaner, technical, fluency, diarize"
    )
    overall_severity: str = Field(description="low, medium, or high")
    notes: str = Field(description="Any specific observations for the workers")


class SupervisorDecision(BaseModel):
    quality_assessment: str = Field(
        description="One sentence: current quality state of the transcript"
    )
    next_action: str = Field(
        description="One of: continue_cleaning, run_technical, run_fluency, run_diarize, export"
    )
    reasoning: str = Field(description="Why this action was chosen")
    estimated_remaining_issues: int = Field(
        description="Rough count of segments still needing work"
    )


plan_parser = JsonOutputParser(pydantic_object=RemediationPlan)
supervisor_parser = JsonOutputParser(pydantic_object=SupervisorDecision)


# ---------------------------------------------------------------------------
# 3. State
# ---------------------------------------------------------------------------

class GuidedAgentState(TypedDict):
    segments: list[dict]
    speaker_names: dict
    plan: dict                          # RemediationPlan dict
    supervisor_history: list[str]       # past next_action decisions
    round: int
    quality_notes: list[str]
    output_paths: list[str]
    log: list[str]


MAX_ROUNDS = 4
FILLERS = {"uh", "um", "you know", "i mean"}
CONTRACTION_MAP = {
    r"\barent\b": "aren't", r"\bcant\b": "can't", r"\bdont\b": "don't",
    r"\bills\b": "I'll",    r"\bill\b": "I'll",    r"\bthats\b": "that's",
    r"\blets\b": "let's",   r"\bwere\b": "we're",  r"\bim\b": "I'm",
    r"\bive\b": "I've",     r"\bits\b": "it's",    r"\bitll\b": "it'll",
}


# ---------------------------------------------------------------------------
# 4. Utility helpers
# ---------------------------------------------------------------------------

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


def _get_context(text: str) -> str:
    docs = context_retriever.invoke(text)
    return "\n".join(f"• {d.page_content}" for d in docs)


def _transcript_sample(segs: list[dict], n: int = 8) -> str:
    return "\n".join(
        f"[{i}] [{s.get('speaker', '?')}] {s['text'].strip()}"
        for i, s in enumerate(segs[:n])
    )


# ---------------------------------------------------------------------------
# 5. Chains
# ---------------------------------------------------------------------------

# --- Planner ---
planner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an ASR post-processing strategist. Analyze the transcript and produce "
     "a concrete remediation plan that will guide specialist workers.\n\n"
     "Workers available: cleaner (punctuation/fillers), technical (terms/names), "
     "fluency (sentence structure), diarize (speaker attribution).\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Transcript sample ({n_segs} total segments):\n{sample}\n\n"
     "Meeting metadata: {metadata}"),
]).partial(format_instructions=plan_parser.get_format_instructions())

planner_chain = planner_prompt | llm | plan_parser

# --- Supervisor ---
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a transcript quality supervisor. After reviewing the current transcript "
     "state, decide what the next action should be.\n\n"
     "Available actions:\n"
     "  continue_cleaning — run another cleanup pass (punctuation, fillers)\n"
     "  run_technical     — fix technical terms, proper nouns, acronym casing\n"
     "  run_fluency       — rewrite disfluent or incomplete sentences\n"
     "  run_diarize       — fix speaker attribution\n"
     "  export            — transcript is ready, no more processing needed\n\n"
     "Actions already taken this session: {history}\n"
     "Round: {round} of {max_rounds}\n"
     "Original plan: {plan_categories}\n\n"
     "IMPORTANT: If round >= {max_rounds} or all plan categories addressed, choose 'export'.\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Current transcript sample:\n{sample}\n\n"
     "Quality notes so far:\n{quality_notes}"),
]).partial(
    format_instructions=supervisor_parser.get_format_instructions(),
    max_rounds=str(MAX_ROUNDS),
)

supervisor_chain = supervisor_prompt | llm | supervisor_parser

# --- Cleaner ---
cleaner_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Fix punctuation, capitalisation, and remove filler words. "
     "Return each line as [N] corrected text. Same number of lines as input."),
    ("human", "{block}"),
])
cleaner_chain = cleaner_prompt | llm | StrOutputParser()

# --- Technical worker ---
technical_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Fix technical term capitalisation and domain proper nouns using the provided context.\n"
     "Known corrections: jwt→JWT, devops→DevOps, node→Node.js, wcag→WCAG, auth→authentication.\n"
     "Also fix participant names if identifiable from context.\n"
     "Return each line as [N] corrected text.\n\n"
     "Context:\n{context}"),
    ("human", "{block}"),
])
technical_chain = technical_prompt | llm | StrOutputParser()

# --- Fluency worker ---
fluency_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite heavily disfluent or incomplete sentences for readability. "
     "Remove stutters, false starts, and repeated phrases. "
     "Complete clearly interrupted sentences using surrounding context. "
     "Do NOT add new information. "
     "Return each line as [N] corrected text."),
    ("human", "Full context:\n{full_context}\n\nSegments to fix:\n{block}"),
])
fluency_chain = fluency_prompt | llm | StrOutputParser()

# --- Diarize ---
diarize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     'Infer speaker names from their speech style and role. '
     'Reply ONLY with JSON: {"SPEAKER_XX": "Name (Role)", ...}'),
    ("human", "{samples}"),
])
diarize_chain = diarize_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 6. Shared block processor
# ---------------------------------------------------------------------------

def _apply_block_output(segs: list[dict], raw_result: str) -> list[dict]:
    for line in raw_result.strip().split("\n"):
        m = re.match(r"^\[(\d+)\]\s*(.*)", line.strip())
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            if 0 <= idx < len(segs) and text:
                segs[idx]["text"] = " " + text
    return segs


# ---------------------------------------------------------------------------
# 7. Nodes
# ---------------------------------------------------------------------------

def planner_node(state: GuidedAgentState) -> dict:
    """LLM analyzes transcript and produces a remediation plan."""
    segs = state["segments"]
    sample = _transcript_sample(segs, n=10)
    metadata = raw.get("meeting_metadata", {})
    metadata_str = json.dumps(metadata, indent=2) if metadata else "none"

    try:
        result = planner_chain.invoke({
            "sample": sample,
            "n_segs": len(segs),
            "metadata": metadata_str,
        })
        plan = result if isinstance(result, dict) else {}
    except Exception as e:
        plan = {
            "problem_categories": ["punctuation", "technical_terms"],
            "high_priority_segments": [],
            "recommended_worker_sequence": ["cleaner", "technical"],
            "overall_severity": "medium",
            "notes": f"Plan generation failed: {e}",
        }

    print(f"  [planner] severity={plan.get('overall_severity')} | "
          f"categories={plan.get('problem_categories', [])}")
    print(f"           sequence={plan.get('recommended_worker_sequence', [])}")
    print(f"           notes: {plan.get('notes', '')[:80]}")

    return {
        "plan": plan,
        "log": state.get("log", []) + [f"Plan: {plan.get('problem_categories', [])}"],
    }


def supervisor_node(state: GuidedAgentState) -> dict:
    """LLM evaluates current quality and decides next action."""
    segs = state["segments"]
    sample = _transcript_sample(segs, n=8)
    history = state.get("supervisor_history", [])
    plan = state.get("plan", {})
    round_num = state.get("round", 0)
    notes = state.get("quality_notes", [])

    # Hard exit if max rounds reached
    if round_num >= MAX_ROUNDS:
        print(f"  [supervisor] max rounds reached — forcing export")
        return {
            "supervisor_history": history + ["export (forced)"],
            "log": state.get("log", []) + ["Supervisor: max rounds, exporting"],
        }

    try:
        result = supervisor_chain.invoke({
            "sample": sample,
            "history": ", ".join(history) if history else "none",
            "round": round_num,
            "plan_categories": plan.get("problem_categories", []),
            "quality_notes": "\n".join(notes[-5:]) if notes else "none",
        })
        decision = result if isinstance(result, dict) else {}
    except Exception:
        decision = {"next_action": "export", "reasoning": "supervisor error", "estimated_remaining_issues": 0}

    action = decision.get("next_action", "export")
    reasoning = decision.get("reasoning", "")[:70]
    remaining = decision.get("estimated_remaining_issues", 0)

    print(f"  [supervisor round={round_num}] action={action} | remaining≈{remaining}")
    print(f"           reasoning: {reasoning}")

    return {
        "supervisor_history": history + [action],
        "log": state.get("log", []) + [
            f"Round {round_num}: supervisor→{action} (≈{remaining} issues left)"
        ],
    }


def supervisor_router(state: GuidedAgentState) -> str:
    """Route based on supervisor's last decision."""
    history = state.get("supervisor_history", [])
    if not history:
        return "cleaner"
    action = history[-1]
    route_map = {
        "continue_cleaning": "cleaner",
        "run_technical": "technical",
        "run_fluency": "fluency",
        "run_diarize": "diarize",
        "export": "export",
        "export (forced)": "export",
    }
    return route_map.get(action, "export")


def cleaner_node(state: GuidedAgentState) -> dict:
    segs = copy.deepcopy(state["segments"])
    # Rule pass first
    for seg in segs:
        seg["text"] = " " + _rule_clean(seg["text"])

    # LLM block pass
    BLOCK = 8
    for start in range(0, len(segs), BLOCK):
        chunk = segs[start:start + BLOCK]
        block_text = "\n".join(f"[{start+i}] {s['text'].strip()}" for i, s in enumerate(chunk))
        try:
            result = cleaner_chain.invoke({"block": block_text})
            segs = _apply_block_output(segs, result)
        except Exception:
            pass

    r = state.get("round", 0) + 1
    print(f"  [cleaner round={r}] processed {len(segs)} segments")
    return {
        "segments": segs,
        "round": r,
        "quality_notes": state.get("quality_notes", []) + [f"Round {r}: cleaner pass done"],
        "log": state.get("log", []) + [f"Cleaner pass {r}"],
    }


def technical_node(state: GuidedAgentState) -> dict:
    segs = copy.deepcopy(state["segments"])
    BLOCK = 8
    for start in range(0, len(segs), BLOCK):
        chunk = segs[start:start + BLOCK]
        block_text = "\n".join(f"[{start+i}] {s['text'].strip()}" for i, s in enumerate(chunk))
        # Retrieve context for the block
        combined_text = " ".join(s["text"].strip() for s in chunk)
        ctx = _get_context(combined_text)
        try:
            result = technical_chain.invoke({"block": block_text, "context": ctx})
            segs = _apply_block_output(segs, result)
        except Exception:
            pass

    r = state.get("round", 0) + 1
    print(f"  [technical round={r}] domain term correction done")
    return {
        "segments": segs,
        "round": r,
        "quality_notes": state.get("quality_notes", []) + [f"Round {r}: technical pass done"],
        "log": state.get("log", []) + [f"Technical pass {r}"],
    }


def fluency_node(state: GuidedAgentState) -> dict:
    segs = copy.deepcopy(state["segments"])
    plan = state.get("plan", {})
    priority_indices = set(plan.get("high_priority_segments", []))

    # Build full context string for the fluency prompt
    full_ctx = "\n".join(
        f"[{i}] [{s.get('speaker', '?')}] {s['text'].strip()}"
        for i, s in enumerate(segs[:20])
    )

    # Target high-priority segments only
    target = [i for i in priority_indices if i < len(segs)] or list(range(min(8, len(segs))))
    block_text = "\n".join(f"[{i}] {segs[i]['text'].strip()}" for i in sorted(target))

    try:
        result = fluency_chain.invoke({"block": block_text, "full_context": full_ctx})
        segs = _apply_block_output(segs, result)
    except Exception:
        pass

    r = state.get("round", 0) + 1
    print(f"  [fluency round={r}] rewrote {len(target)} priority segments")
    return {
        "segments": segs,
        "round": r,
        "quality_notes": state.get("quality_notes", []) + [f"Round {r}: fluency pass done"],
        "log": state.get("log", []) + [f"Fluency pass {r}"],
    }


def diarize_node(state: GuidedAgentState) -> dict:
    segs = copy.deepcopy(state["segments"])
    samples: dict[str, str] = {}
    for seg in segs:
        spk = seg.get("speaker", "UNKNOWN")
        if spk and len(samples.get(spk, "")) < 400:
            samples[spk] = samples.get(spk, "") + " " + seg["text"].strip()

    prompt_text = "\n\n".join(f"[{spk}]: {t[:300]}" for spk, t in samples.items())
    names: dict[str, str] = {}
    try:
        raw_result = diarize_chain.invoke({"samples": prompt_text})
        names = json.loads(raw_result)
    except Exception:
        m = re.search(r"\{[^}]+\}", raw_result if isinstance(raw_result, str) else "", re.DOTALL)
        if m:
            try:
                names = json.loads(m.group())
            except Exception:
                pass

    # Merge adjacent same-speaker short backchannels
    merged = []
    for seg in segs:
        wc = len(seg.get("words", []))
        if merged and wc <= 2 and merged[-1].get("speaker") == seg.get("speaker"):
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"].strip()
        else:
            merged.append(seg)

    r = state.get("round", 0) + 1
    print(f"  [diarize round={r}] names={names} | merged {len(segs)-len(merged)} segments")
    return {
        "segments": merged,
        "speaker_names": names,
        "round": r,
        "quality_notes": state.get("quality_notes", []) + [f"Round {r}: diarize pass done"],
        "log": state.get("log", []) + [f"Diarize pass {r}: {names}"],
    }


def export_node(state: GuidedAgentState) -> dict:
    segs = state["segments"]
    names = state.get("speaker_names", {})
    paths = []

    # JSON
    out = copy.deepcopy(raw)
    out["segments"] = segs
    out["speaker_names"] = names
    out["plan"] = state.get("plan", {})
    out["supervisor_history"] = state.get("supervisor_history", [])
    out["log"] = state.get("log", [])
    p = OUT_DIR / "asr_llm_guided_output.json"
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
    p = OUT_DIR / "asr_llm_guided_output.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    paths.append(str(p))

    # SRT
    srt_lines = []
    for i, seg in enumerate(segs, 1):
        label = names.get(seg.get("speaker", "UNKNOWN"), seg.get("speaker", "UNKNOWN"))
        srt_lines.append(
            f"{i}\n{_fmt_srt(seg['start'])} --> {_fmt_srt(seg['end'])}\n{label}: {seg['text'].strip()}"
        )
    p = OUT_DIR / "asr_llm_guided_output.srt"
    p.write_text("\n\n".join(srt_lines), encoding="utf-8")
    paths.append(str(p))

    print(f"  [export] wrote {len(paths)} files")
    return {"output_paths": paths, "log": state.get("log", []) + [f"Exported {len(paths)} files"]}


# ---------------------------------------------------------------------------
# 8. Build and run graph
# ---------------------------------------------------------------------------

builder = StateGraph(GuidedAgentState)
for name, fn in [
    ("planner", planner_node),
    ("supervisor", supervisor_node),
    ("cleaner", cleaner_node),
    ("technical", technical_node),
    ("fluency", fluency_node),
    ("diarize", diarize_node),
    ("export", export_node),
]:
    builder.add_node(name, fn)

# Planner always runs first, then supervisor
builder.add_edge(START, "planner")
builder.add_edge("planner", "supervisor")

# Supervisor routes to any worker or export
builder.add_conditional_edges("supervisor", supervisor_router, {
    "cleaner": "cleaner",
    "technical": "technical",
    "fluency": "fluency",
    "diarize": "diarize",
    "export": "export",
})

# Every worker goes back to supervisor for the next decision
for worker in ["cleaner", "technical", "fluency", "diarize"]:
    builder.add_edge(worker, "supervisor")

builder.add_edge("export", END)

pipeline = builder.compile(checkpointer=MemorySaver())

print("=" * 60)
print("  LLM-GUIDED ASR PRODUCTION AGENT")
print("=" * 60)
print()

config = {"configurable": {"thread_id": "llm_guided_agent"}}
final = pipeline.invoke({
    "segments": copy.deepcopy(raw["segments"]),
    "speaker_names": {},
    "plan": {},
    "supervisor_history": [],
    "round": 0,
    "quality_notes": [],
    "output_paths": [],
    "log": [],
}, config=config)

print("\n  Supervisor decision sequence:")
for i, action in enumerate(final.get("supervisor_history", []), 1):
    print(f"    {i}. {action}")

print("\n  Output files:")
for p in final.get("output_paths", []):
    print(f"    • {p}")

print("\n  Full log:")
for entry in final.get("log", []):
    print(f"    • {entry}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Plan-first: LLM reads the transcript once and produces a strategy for all workers
# ✅ Supervisor pattern: LLM decides the next action after every worker — not a fixed sequence
# ✅ Specialist workers: each has a targeted prompt, not one generic "fix everything" prompt
# ✅ supervisor_history prevents the LLM from looping on the same action repeatedly
# ✅ Plan is in state: workers read it to focus on high-priority segments
# ✅ MAX_ROUNDS is still a hard cap — LLM supervisor doesn't mean infinite loops
# ✅ RAG context is fetched per-block in technical_node — most relevant docs per segment
# ✅ This is the production template: plug in better LLMs, richer RAG, more workers
