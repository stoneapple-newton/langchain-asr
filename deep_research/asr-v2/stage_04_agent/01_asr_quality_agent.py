"""
Stage 4, File 1: ASR Quality Agent
==================================
CONCEPT: Combine deterministic repair, readability revision, and quality
checking into an iterative ASR improvement workflow.

Run this file:
  uv run deep_research/asr-v2/stage_04_agent/01_asr_quality_agent.py
"""

import os
from pathlib import Path
import sys
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.transcript_utils import (
    TranscriptSegment,
    analyze_transcript,
    chunk_segments,
    improve_readability,
    load_transcript,
    rebuild_document_from_segments,
    repair_diarization,
    save_enhanced_outputs,
)

load_dotenv()


class EditedChunk(BaseModel):
    edited_lines: list[str] = Field(description="Edited transcript lines in the same order as input")


class QualityState(TypedDict):
    input_path: str
    output_dir: str
    doc: object
    iteration: int
    quality_notes: list[str]
    output_paths: dict


parser = JsonOutputParser(pydantic_object=EditedChunk)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Improve transcript readability without changing speaker order, timestamps, or meaning. "
        "Keep the line count identical and return JSON with edited_lines.\n{format_instructions}",
    ),
    ("human", "{chunk_text}"),
]).partial(format_instructions=parser.get_format_instructions())

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=768,
)
chain = prompt | llm | parser


def load_node(state: QualityState) -> dict:
    return {
        "doc": load_transcript(state["input_path"]),
        "iteration": 0,
        "quality_notes": [],
    }


def diarization_node(state: QualityState) -> dict:
    repaired = repair_diarization(state["doc"])
    stats = analyze_transcript(repaired)
    note = (
        f"iteration {state['iteration'] + 1}: "
        f"missing speaker rows={stats['missing_speaker_segments']}, "
        f"segments={stats['segment_count']}"
    )
    print(f"[diarization] {note}")
    return {
        "doc": repaired,
        "iteration": state["iteration"] + 1,
        "quality_notes": state["quality_notes"] + [note],
    }


def readability_node(state: QualityState) -> dict:
    base_doc = improve_readability(state["doc"])
    edited_segments: list[TranscriptSegment] = []
    for chunk in chunk_segments(base_doc, max_chars=420):
        chunk_lines = [
            f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.speaker or 'UNKNOWN'}: {segment.text}"
            for segment in chunk
        ]
        try:
            result = chain.invoke({"chunk_text": "\n".join(chunk_lines)})
            edited_lines = result["edited_lines"]
        except Exception as exc:
            print(f"[readability] LLM fallback: {exc}")
            edited_lines = chunk_lines

        if len(edited_lines) != len(chunk_lines):
            edited_lines = chunk_lines

        for segment, edited_line in zip(chunk, edited_lines):
            _, _, edited_text = edited_line.partition(":")
            edited_segments.append(
                TranscriptSegment(
                    segment_id=segment.segment_id,
                    start=segment.start,
                    end=segment.end,
                    text=edited_text.strip() or segment.text,
                    speaker=segment.speaker,
                    words=segment.words,
                    metadata=dict(segment.metadata),
                )
            )

    return {"doc": rebuild_document_from_segments(base_doc, edited_segments)}


def quality_node(state: QualityState) -> dict:
    stats = analyze_transcript(state["doc"])
    pass_quality = stats["missing_speaker_segments"] == 0 or state["iteration"] >= 2
    note = f"quality gate: pass={pass_quality}, speakers={stats['speaker_count']}, changes={stats['speaker_changes']}"
    print(f"[quality] {note}")
    return {"quality_notes": state["quality_notes"] + [note]}


def route_after_quality(state: QualityState) -> str:
    stats = analyze_transcript(state["doc"])
    if stats["missing_speaker_segments"] == 0 or state["iteration"] >= 2:
        return "save"
    return "diarization"


def save_node(state: QualityState) -> dict:
    paths = save_enhanced_outputs(state["doc"], state["input_path"], state["output_dir"])
    print(f"[save] markdown={paths['markdown_path']}")
    print(f"[save] json={paths['json_path']}")
    return {"output_paths": paths}


builder = StateGraph(QualityState)
builder.add_node("load", load_node)
builder.add_node("diarization", diarization_node)
builder.add_node("readability", readability_node)
builder.add_node("quality", quality_node)
builder.add_node("save", save_node)
builder.add_edge(START, "load")
builder.add_edge("load", "diarization")
builder.add_edge("diarization", "readability")
builder.add_edge("readability", "quality")
builder.add_conditional_edges("quality", route_after_quality)
builder.add_edge("save", END)
app = builder.compile()

sample_path = ROOT / "sample_data" / "meeting_sample.json"
output_dir = ROOT / "outputs"

result = app.invoke({
    "input_path": str(sample_path),
    "output_dir": str(output_dir),
})

print("=" * 60)
print("  Final quality notes")
print("=" * 60)
for note in result["quality_notes"]:
    print(f"  - {note}")
