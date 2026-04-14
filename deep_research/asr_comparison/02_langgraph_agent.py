"""
02_langgraph_agent.py
=====================
CONCEPT: Full LangGraph comparison agent with LLM-powered resolution of
ambiguous diffs.

Graph topology:
  START → load → align → rule_diff
                              ↓
                route_after_rule_diff
                ↙ (has ambiguous)  ↘ (none)
          llm_resolve             report
                ↘                   ↙
                report → save → END

Run this file:
  uv run deep_research/asr_comparison/02_langgraph_agent.py
"""

import json
import sys
from pathlib import Path
from typing_extensions import TypedDict

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ASR_V2_ROOT = Path(__file__).resolve().parents[1] / "asr-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]

for p in (REPO_ROOT, ASR_V2_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# ── imports ───────────────────────────────────────────────────────────────────
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import create_chat_model
from shared.transcript_utils import load_transcript  # asr-v2/shared
from shared.comparison_utils import (  # asr-v2/shared
    AlignedPair,
    DiffCategory,
    align_segments,
    categorise_all,
    compute_document_summary,
    render_report_markdown,
)

# ── LLM setup ─────────────────────────────────────────────────────────────────

class ResolvedDiff(BaseModel):
    pair_id: str = Field(description="The pair_id being classified")
    category: str = Field(
        description="One of: substitution, readability, filler_word, speaker_mismatch"
    )
    rationale: str = Field(description="One-sentence explanation")


class ResolvedBatch(BaseModel):
    results: list[ResolvedDiff] = Field(description="Resolved diffs for each ambiguous pair")


_parser = JsonOutputParser(pydantic_object=ResolvedBatch)
_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an ASR transcript diff analyser.\n"
        "For each pair in the input, classify the difference into exactly one of:\n"
        "  substitution  — one or more words changed (potential transcription error)\n"
        "  readability   — only punctuation, capitalisation, or contraction style differs\n"
        "  filler_word   — only filler tokens (um, uh, hmm, yeah, like) added or removed\n"
        "  speaker_mismatch — same words, but attributed to different speakers\n\n"
        "Return a JSON object with a 'results' array. Each element must have:\n"
        "  pair_id, category (one of the four above), rationale (one sentence).\n"
        "Do not invent new categories. Do not change pair_id values.\n"
        "{format_instructions}",
    ),
    ("human", "{batch_json}"),
]).partial(format_instructions=_parser.get_format_instructions())

_llm = create_chat_model("asr_v2", temperature=0, max_tokens=512)
_chain = _prompt | _llm | _parser

# ── State ─────────────────────────────────────────────────────────────────────

class ComparisonState(TypedDict):
    ref_path: str
    hyp_path: str
    label_a: str
    label_b: str
    output_dir: str
    ref_doc: object
    hyp_doc: object
    aligned_pairs: list[dict]
    rule_diffs: list[dict]      # may contain AMBIGUOUS entries
    resolved_diffs: list[dict]  # AMBIGUOUS replaced by LLM
    summary: dict
    output_paths: dict


# ── Nodes ─────────────────────────────────────────────────────────────────────

def load_node(state: ComparisonState) -> dict:
    ref_doc = load_transcript(state["ref_path"])
    hyp_doc = load_transcript(state["hyp_path"])
    print(f"[load] ref={len(ref_doc.segments)} segments  hyp={len(hyp_doc.segments)} segments")
    return {"ref_doc": ref_doc, "hyp_doc": hyp_doc}


def align_node(state: ComparisonState) -> dict:
    pairs = align_segments(state["ref_doc"], state["hyp_doc"], iou_threshold=0.4)
    print(f"[align] {len(pairs)} aligned pairs")
    return {"aligned_pairs": [p.to_dict() for p in pairs]}


def _pair_from_dict(d: dict) -> AlignedPair:
    d = dict(d)
    d["category"] = DiffCategory(d["category"])
    d.setdefault("metadata", {})
    return AlignedPair(**d)


def rule_diff_node(state: ComparisonState) -> dict:
    pairs = [_pair_from_dict(d) for d in state["aligned_pairs"]]
    diffs = categorise_all(pairs)
    ambiguous = sum(1 for d in diffs if d.category == DiffCategory.AMBIGUOUS)
    print(f"[rule_diff] {len(diffs)} diffs  ({ambiguous} ambiguous)")
    return {"rule_diffs": [d.to_dict() for d in diffs]}


def llm_resolve_node(state: ComparisonState) -> dict:
    ambiguous = [
        d for d in state["rule_diffs"]
        if d["category"] == DiffCategory.AMBIGUOUS.value
    ]
    print(f"[llm_resolve] sending {len(ambiguous)} ambiguous pairs to LLM")

    batch_payload = [
        {
            "pair_id": d["pair_id"],
            "ref_text": d["ref_text"],
            "hyp_text": d["hyp_text"],
            "ref_speaker": d["ref_speaker"],
            "hyp_speaker": d["hyp_speaker"],
        }
        for d in ambiguous
    ]

    try:
        result = _chain.invoke({"batch_json": json.dumps(batch_payload, indent=2)})
        resolutions: dict[str, str] = {
            r["pair_id"]: r["category"] for r in result["results"]
        }
    except Exception as exc:
        print(f"[llm_resolve] LLM error, falling back to substitution: {exc}")
        resolutions = {d["pair_id"]: "substitution" for d in ambiguous}

    resolved = []
    for diff in state["rule_diffs"]:
        if diff["category"] == DiffCategory.AMBIGUOUS.value:
            diff = dict(diff)
            diff["category"] = resolutions.get(diff["pair_id"], "substitution")
            diff.setdefault("metadata", {})["llm_resolved"] = True
        resolved.append(diff)

    return {"resolved_diffs": resolved}


def report_node(state: ComparisonState) -> dict:
    diffs_source = state.get("resolved_diffs") or state["rule_diffs"]
    pairs = [_pair_from_dict(d) for d in diffs_source]
    summary = compute_document_summary(
        pairs, state["label_a"], state["label_b"]
    )

    print("\n[report] ── Category breakdown ──────────────────────────")
    for cat, count in summary["category_counts"].items():
        if count > 0:
            print(f"  {cat:<20}  {count}")
    print(f"\n  Overall WER: {summary['overall_wer']:.1%}")
    errs = summary["word_errors"]
    print(f"  S={errs['substitutions']}  I={errs['insertions']}  "
          f"D={errs['deletions']}  N={errs['ref_word_count']}")

    return {"summary": summary}


def save_node(state: ComparisonState) -> dict:
    diffs_source = state.get("resolved_diffs") or state["rule_diffs"]
    pairs = [_pair_from_dict(d) for d in diffs_source]

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = output_dir / "comparison_report.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": state["summary"],
                "diffs": [d.to_dict() for d in pairs],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    # Markdown report
    md_path = output_dir / "comparison_report.md"
    md_path.write_text(
        render_report_markdown(pairs, state["summary"], state["label_a"], state["label_b"]),
        encoding="utf-8",
    )

    print(f"[save] json={json_path}")
    print(f"[save] markdown={md_path}")
    return {"output_paths": {"json": str(json_path), "markdown": str(md_path)}}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_rule_diff(state: ComparisonState) -> str:
    has_ambiguous = any(
        d["category"] == DiffCategory.AMBIGUOUS.value
        for d in state["rule_diffs"]
    )
    return "llm_resolve" if has_ambiguous else "report"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_comparison_graph():
    builder = StateGraph(ComparisonState)
    builder.add_node("load", load_node)
    builder.add_node("align", align_node)
    builder.add_node("rule_diff", rule_diff_node)
    builder.add_node("llm_resolve", llm_resolve_node)
    builder.add_node("report", report_node)
    builder.add_node("save", save_node)

    builder.add_edge(START, "load")
    builder.add_edge("load", "align")
    builder.add_edge("align", "rule_diff")
    builder.add_conditional_edges("rule_diff", route_after_rule_diff)
    builder.add_edge("llm_resolve", "report")
    builder.add_edge("report", "save")
    builder.add_edge("save", END)

    return builder.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_comparison_graph()

    sample_a = ROOT / "sample_data" / "transcript_a.json"
    sample_b = ROOT / "sample_data" / "transcript_b.json"
    output_dir = ROOT / "outputs"

    print("=" * 60)
    print("  ASR Transcript Comparison — LangGraph Agent")
    print("=" * 60)

    result = app.invoke({
        "ref_path": str(sample_a),
        "hyp_path": str(sample_b),
        "label_a": "raw_asr",
        "label_b": "enhanced_asr",
        "output_dir": str(output_dir),
    })

    print("\n" + "=" * 60)
    print("  Done")
    print("=" * 60)
    paths = result.get("output_paths", {})
    for key, path in paths.items():
        print(f"  {key}: {path}")
