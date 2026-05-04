"""
Stage 2, File 1: Topic Segmentation LangGraph Agent
====================================================
CONCEPT: A five-node LangGraph that loads a transcript, detects topic
boundaries deterministically, uses the LLM to label each topic with a title
and summary, then saves a JSON and Markdown report.

Graph: load → detect_signals → group_topics → llm_label → save

Run this file:
  uv run deep_research/asr_topic_segmentation/stage_02_langgraph/01_topic_segmentation_agent.py
"""

import json
import sys
from pathlib import Path
from typing import Any

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
from shared.segmentation_utils import (
    TopicBoundarySignal,
    TopicSegment,
    detect_boundary_signals,
    format_topic_for_llm,
    group_into_topic_segments,
    render_segmentation_report,
)
from transcript_utils import TranscriptDocument, load_transcript
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# LLM output contract
# ---------------------------------------------------------------------------

class TopicLabel(BaseModel):
    topic_id: str = Field(description="Matches input topic_id, e.g. topic_01")
    title: str = Field(description="Short topic title, max 8 words")
    summary: str = Field(description="One-sentence summary of what was discussed")


class TopicLabelBatch(BaseModel):
    topics: list[TopicLabel] = Field(description="One entry per input topic block")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SegmentationState(TypedDict):
    input_path: str
    output_dir: str
    doc: Any
    signals: list[Any]
    raw_topics: list[Any]
    labelled_topics: list[Any]
    output_paths: dict[str, str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_node(state: SegmentationState) -> dict:
    doc = load_transcript(state["input_path"])
    print(f"[load] {len(doc.segments)} segments from {Path(state['input_path']).name}")
    return {"doc": doc}


def detect_signals_node(state: SegmentationState) -> dict:
    doc: TranscriptDocument = state["doc"]
    signals = detect_boundary_signals(doc.segments)
    high_signal = sum(1 for s in signals if s.signal_score >= 0.5)
    print(f"[detect_signals] {len(signals)} signals, {high_signal} above threshold")
    return {"signals": signals}


def group_topics_node(state: SegmentationState) -> dict:
    doc: TranscriptDocument = state["doc"]
    signals: list[TopicBoundarySignal] = state["signals"]
    topics = group_into_topic_segments(doc.segments, signals, threshold=0.5)
    print(f"[group_topics] {len(topics)} topic segments detected")
    return {"raw_topics": topics}


def llm_label_node(state: SegmentationState) -> dict:
    doc: TranscriptDocument = state["doc"]
    raw_topics: list[TopicSegment] = state["raw_topics"]

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a meeting analyst. For each topic block, assign a short "
            "title (max 8 words) and a one-sentence summary. "
            "Return one entry per input topic_id, preserving the exact topic_id.",
        ),
        ("human", "{topic_blocks}"),
    ])

    llm = create_chat_model(temperature=0, max_tokens=1024)
    chain = structured_output_chain(llm, prompt, TopicLabelBatch)

    topic_blocks = "\n\n".join(
        format_topic_for_llm(topic, doc.segments) for topic in raw_topics
    )

    try:
        result = chain.invoke({"topic_blocks": topic_blocks})
        labels_by_id = {lbl["topic_id"]: lbl for lbl in result.get("topics", [])}
    except Exception as exc:
        print(f"[llm_label] fallback: {exc}")
        labels_by_id = {}

    labelled: list[TopicSegment] = []
    for topic in raw_topics:
        lbl = labels_by_id.get(topic.topic_id, {})
        labelled.append(
            TopicSegment(
                topic_id=topic.topic_id,
                title=lbl.get("title", topic.title),
                start_time=topic.start_time,
                end_time=topic.end_time,
                segment_ids=topic.segment_ids,
                summary=lbl.get("summary", ""),
            )
        )

    print(f"[llm_label] labelled {len(labelled)} topics")
    return {"labelled_topics": labelled}


def save_node(state: SegmentationState) -> dict:
    labelled: list[TopicSegment] = state["labelled_topics"]
    source = Path(state["input_path"])
    out_dir = Path(state["output_dir"]) / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{source.stem}.topics.md"
    json_path = out_dir / f"{source.stem}.topics.json"

    md_path.write_text(render_segmentation_report(labelled), encoding="utf-8")

    payload = {
        "source": str(source),
        "topics": [
            {
                "topic_id": t.topic_id,
                "title": t.title,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "duration_seconds": round(t.end_time - t.start_time, 2),
                "segment_count": len(t.segment_ids),
                "segment_ids": t.segment_ids,
                "summary": t.summary,
            }
            for t in labelled
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[save] markdown → {md_path}")
    print(f"[save] json → {json_path}")
    return {"output_paths": {"markdown": str(md_path), "json": str(json_path)}}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_segmentation_graph():
    builder = StateGraph(SegmentationState)
    builder.add_node("load", load_node)
    builder.add_node("detect_signals", detect_signals_node)
    builder.add_node("group_topics", group_topics_node)
    builder.add_node("llm_label", llm_label_node)
    builder.add_node("save", save_node)
    builder.add_edge(START, "load")
    builder.add_edge("load", "detect_signals")
    builder.add_edge("detect_signals", "group_topics")
    builder.add_edge("group_topics", "llm_label")
    builder.add_edge("llm_label", "save")
    builder.add_edge("save", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_segmentation_graph()
    result = graph.invoke({
        "input_path": str(TRACK_ROOT / "sample_data" / "team_meeting_sample.json"),
        "output_dir": str(TRACK_ROOT / "outputs"),
    })
    print("\n=== Topic Segments ===")
    for topic in result["labelled_topics"]:
        print(
            f"  {topic.topic_id}: {topic.title} "
            f"({topic.start_time:.1f}s – {topic.end_time:.1f}s)"
        )
        if topic.summary:
            print(f"    {topic.summary}")
