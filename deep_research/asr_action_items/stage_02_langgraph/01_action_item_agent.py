"""
Stage 2, File 1: Action Item Extraction LangGraph Agent
========================================================
CONCEPT: A four-node LangGraph that loads a transcript, detects action item
candidates deterministically, classifies them with the LLM into typed action
items, then saves a JSON and Markdown report.

Graph: load → detect_candidates → llm_classify → save

Run this file:
  uv run deep_research/asr_action_items/stage_02_langgraph/01_action_item_agent.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACK_ROOT = Path(__file__).resolve().parents[2]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

ASR_V2_SHARED = REPO_ROOT / "deep_research" / "asr-v2" / "shared"
if str(ASR_V2_SHARED) not in sys.path:
    sys.path.insert(0, str(ASR_V2_SHARED))

from config import create_chat_model, structured_output_chain
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from shared.action_item_utils import (
    ActionItem,
    CandidateItem,
    detect_candidates,
    format_candidates_for_llm,
    render_action_items_report,
)
from transcript_utils import TranscriptDocument, load_transcript
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# LLM output contract
# ---------------------------------------------------------------------------

class ExtractedItem(BaseModel):
    item_id: str = Field(description="Unique ID, e.g. AI-001")
    item_type: Literal["task", "decision", "open_question", "commitment"] = Field(
        description="Category of the action item"
    )
    text: str = Field(description="Concise restatement of the action item")
    segment_id: str = Field(description="Source segment ID")
    owner: str | None = Field(default=None, description="Owner name or null")
    due_context: str | None = Field(
        default=None, description="Time reference (e.g. 'by noon', 'next Monday') or null"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence 0-1")


class ExtractedItemList(BaseModel):
    items: list[ExtractedItem] = Field(
        description="All action items extracted from the candidates"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ActionItemState(TypedDict):
    input_path: str
    output_dir: str
    doc: Any
    candidates: list[Any]
    action_items: list[Any]
    output_paths: dict[str, str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_node(state: ActionItemState) -> dict:
    doc = load_transcript(state["input_path"])
    print(f"[load] {len(doc.segments)} segments from {Path(state['input_path']).name}")
    return {"doc": doc}


def detect_candidates_node(state: ActionItemState) -> dict:
    doc: TranscriptDocument = state["doc"]
    candidates = detect_candidates(doc.segments)
    print(f"[detect_candidates] {len(candidates)} candidate segments")
    for c in candidates:
        print(f"  seg={c.segment_id} count={c.match_count} types={c.candidate_types}")
    return {"candidates": candidates}


def llm_classify_node(state: ActionItemState) -> dict:
    candidates: list[CandidateItem] = state["candidates"]

    if not candidates:
        print("[llm_classify] no candidates — skipping LLM call")
        return {"action_items": []}

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a meeting assistant. Extract structured action items from "
            "the candidate transcript segments. Classify each item as task, "
            "decision, open_question, or commitment. Assign a short item_id "
            "(AI-001, AI-002, …). Preserve the segment_id from the input. "
            "Only extract genuine action items — skip conversational filler.",
        ),
        ("human", "{candidates}"),
    ])

    llm = create_chat_model(temperature=0, max_tokens=2048)
    chain = structured_output_chain(llm, prompt, ExtractedItemList)

    try:
        result = chain.invoke({"candidates": format_candidates_for_llm(candidates)})
        raw_items = result.get("items", [])
    except Exception as exc:
        print(f"[llm_classify] fallback: {exc}")
        raw_items = []

    action_items: list[ActionItem] = []
    seg_text_map = {c.segment_id: c.text for c in candidates}

    for raw in raw_items:
        action_items.append(
            ActionItem(
                item_id=raw["item_id"],
                item_type=raw["item_type"],
                text=raw["text"],
                raw_segment_text=seg_text_map.get(raw["segment_id"], ""),
                segment_id=raw["segment_id"],
                owner=raw.get("owner"),
                due_context=raw.get("due_context"),
                confidence=raw.get("confidence", 0.8),
            )
        )

    print(f"[llm_classify] extracted {len(action_items)} action items")
    return {"action_items": action_items}


def save_node(state: ActionItemState) -> dict:
    items: list[ActionItem] = state["action_items"]
    source = Path(state["input_path"])
    out_dir = Path(state["output_dir"]) / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{source.stem}.action_items.md"
    json_path = out_dir / f"{source.stem}.action_items.json"

    md_path.write_text(render_action_items_report(items), encoding="utf-8")

    payload = {
        "source": str(source),
        "item_count": len(items),
        "items": [
            {
                "item_id": item.item_id,
                "item_type": item.item_type,
                "text": item.text,
                "segment_id": item.segment_id,
                "owner": item.owner,
                "due_context": item.due_context,
                "confidence": item.confidence,
            }
            for item in items
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[save] markdown → {md_path}")
    print(f"[save] json → {json_path}")
    return {"output_paths": {"markdown": str(md_path), "json": str(json_path)}}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_action_item_graph():
    builder = StateGraph(ActionItemState)
    builder.add_node("load", load_node)
    builder.add_node("detect_candidates", detect_candidates_node)
    builder.add_node("llm_classify", llm_classify_node)
    builder.add_node("save", save_node)
    builder.add_edge(START, "load")
    builder.add_edge("load", "detect_candidates")
    builder.add_edge("detect_candidates", "llm_classify")
    builder.add_edge("llm_classify", "save")
    builder.add_edge("save", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_action_item_graph()

    # reuse the team_meeting_sample from topic segmentation track
    meeting_sample = (
        REPO_ROOT
        / "deep_research"
        / "asr_topic_segmentation"
        / "sample_data"
        / "team_meeting_sample.json"
    )

    result = graph.invoke({
        "input_path": str(meeting_sample),
        "output_dir": str(TRACK_ROOT / "outputs"),
    })

    print("\n=== Action Items ===")
    for item in result["action_items"]:
        owner = f" [{item.owner}]" if item.owner else ""
        print(f"  {item.item_id} ({item.item_type}){owner}: {item.text}")
