"""
Stage 2, File 1: Sentiment Analysis LangGraph Agent
====================================================
CONCEPT: A four-node LangGraph that loads a transcript, scores segments
deterministically, refines labels with an LLM, aggregates by speaker, then
saves a JSON sidecar and Markdown report.

Graph: load → score_naive → llm_refine → aggregate → save

Run this file:
  uv run deep_research/asr_sentiment_analysis/stage_02_langgraph/01_sentiment_agent.py
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
from shared.sentiment_utils import (
    SegmentSentimentLabel,
    SpeakerSentimentProfile,
    aggregate_speaker_profiles,
    format_labels_for_llm,
    render_sentiment_report,
    score_segment_naive,
)
from transcript_utils import TranscriptDocument, chunk_segments, load_transcript
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# LLM output contract
# ---------------------------------------------------------------------------

class RefinedLabel(BaseModel):
    segment_id: str = Field(description="Segment ID matching input")
    label: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Refined sentiment label"
    )
    intensity: float = Field(ge=0.0, le=1.0, description="Emotional intensity 0-1")
    key_phrases: list[str] = Field(
        default_factory=list,
        description="Up to 3 key phrases driving the label",
    )


class RefinedLabelBatch(BaseModel):
    refined: list[RefinedLabel] = Field(description="One entry per input segment")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SentimentState(TypedDict):
    input_path: str
    output_dir: str
    doc: Any
    naive_labels: list[Any]
    refined_labels: list[Any]
    speaker_profiles: dict[str, Any]
    output_paths: dict[str, str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_node(state: SentimentState) -> dict:
    doc = load_transcript(state["input_path"])
    print(f"[load] {len(doc.segments)} segments from {Path(state['input_path']).name}")
    return {"doc": doc}


def score_naive_node(state: SentimentState) -> dict:
    doc: TranscriptDocument = state["doc"]
    labels = [
        score_segment_naive(
            segment_id=seg.segment_id,
            speaker=seg.speaker,
            start=seg.start,
            end=seg.end,
            text=seg.text,
        )
        for seg in doc.segments
    ]
    counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    for lbl in labels:
        counts[lbl.naive_label] = counts.get(lbl.naive_label, 0) + 1
    print(f"[score_naive] {counts}")
    return {"naive_labels": labels}


def llm_refine_node(state: SentimentState) -> dict:
    labels: list[SegmentSentimentLabel] = state["naive_labels"]
    doc: TranscriptDocument = state["doc"]

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a sentiment analysis model. For each labeled transcript "
            "segment, refine the sentiment label and rate emotional intensity. "
            "Return exactly one entry per input segment, preserving segment_id. "
            "Consider context across segments when labelling.",
        ),
        ("human", "{chunk_text}"),
    ])

    llm = create_chat_model(temperature=0, max_tokens=2048)
    chain = structured_output_chain(llm, prompt, RefinedLabelBatch)

    refined_labels: list[RefinedLabel] = []
    for chunk in chunk_segments(doc, max_chars=800):
        chunk_ids = {seg.segment_id for seg in chunk}
        chunk_labels = [lbl for lbl in labels if lbl.segment_id in chunk_ids]
        try:
            result = chain.invoke({"chunk_text": format_labels_for_llm(chunk_labels)})
            refined_labels.extend(result.get("refined", []))
        except Exception as exc:
            print(f"[llm_refine] fallback for chunk: {exc}")
            for lbl in chunk_labels:
                refined_labels.append(
                    RefinedLabel(
                        segment_id=lbl.segment_id,
                        label=lbl.naive_label,
                        intensity=0.5,
                        key_phrases=[],
                    )
                )

    print(f"[llm_refine] refined {len(refined_labels)} segments")
    return {"refined_labels": refined_labels}


def aggregate_node(state: SentimentState) -> dict:
    naive: list[SegmentSentimentLabel] = state["naive_labels"]
    refined: list[RefinedLabel] = state["refined_labels"]

    refined_by_id = {r["segment_id"]: r for r in refined}
    merged_labels = []
    for lbl in naive:
        refined_entry = refined_by_id.get(lbl.segment_id)
        if refined_entry:
            lbl.naive_label = refined_entry["label"]
        merged_labels.append(lbl)

    profiles = aggregate_speaker_profiles(merged_labels)
    print(f"[aggregate] speakers: {list(profiles.keys())}")
    return {"naive_labels": merged_labels, "speaker_profiles": profiles}


def save_node(state: SentimentState) -> dict:
    labels: list[SegmentSentimentLabel] = state["naive_labels"]
    profiles: dict[str, SpeakerSentimentProfile] = state["speaker_profiles"]
    refined: list[RefinedLabel] = state["refined_labels"]

    source = Path(state["input_path"])
    out_dir = Path(state["output_dir"]) / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{source.stem}.sentiment.md"
    json_path = out_dir / f"{source.stem}.sentiment.json"

    md_path.write_text(render_sentiment_report(labels, profiles), encoding="utf-8")

    payload = {
        "source": str(source),
        "speaker_profiles": {
            speaker: {
                "dominant_label": profile.dominant_label,
                "positive_ratio": profile.positive_ratio,
                "negative_ratio": profile.negative_ratio,
                "total_fillers": profile.total_fillers,
                "segment_count": profile.segment_count,
            }
            for speaker, profile in profiles.items()
        },
        "segments": [
            {
                "segment_id": lbl.segment_id,
                "speaker": lbl.speaker,
                "start": lbl.start,
                "end": lbl.end,
                "label": lbl.naive_label,
                "text": lbl.text,
            }
            for lbl in labels
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[save] markdown → {md_path}")
    print(f"[save] json → {json_path}")
    return {"output_paths": {"markdown": str(md_path), "json": str(json_path)}}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_sentiment_graph():
    builder = StateGraph(SentimentState)
    builder.add_node("load", load_node)
    builder.add_node("score_naive", score_naive_node)
    builder.add_node("llm_refine", llm_refine_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("save", save_node)
    builder.add_edge(START, "load")
    builder.add_edge("load", "score_naive")
    builder.add_edge("score_naive", "llm_refine")
    builder.add_edge("llm_refine", "aggregate")
    builder.add_edge("aggregate", "save")
    builder.add_edge("save", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_sentiment_graph()
    result = graph.invoke({
        "input_path": str(TRACK_ROOT / "sample_data" / "call_center_sample.json"),
        "output_dir": str(TRACK_ROOT / "outputs"),
    })
    print("\n=== Speaker Profiles ===")
    for speaker, profile in result["speaker_profiles"].items():
        print(
            f"  {speaker}: {profile['dominant_label'].upper()} "
            f"(+{profile['positive_ratio']:.0%} / -{profile['negative_ratio']:.0%})"
        )
