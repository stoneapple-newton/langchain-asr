"""
Stage 3, File 2: Readability Revision Graph
===========================================
CONCEPT: Combine deterministic cleanup and LLM revision inside a guarded graph.

Run this file:
  uv run deep_research/asr-v2/stage_03_langgraph/02_readability_revision_graph.py
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
    chunk_segments,
    improve_readability,
    load_transcript,
    rebuild_document_from_segments,
    repair_diarization,
)

load_dotenv()


class EditedChunk(BaseModel):
    edited_lines: list[str] = Field(description="Edited transcript lines in the same order as input")


class RevisionState(TypedDict):
    input_path: str
    doc: object
    revised_doc: object


parser = JsonOutputParser(pydantic_object=EditedChunk)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite transcript lines for readability while preserving speaker and order. "
        "Return valid JSON with edited_lines and keep the line count unchanged.\n{format_instructions}",
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


def prepare_node(state: RevisionState) -> dict:
    base = improve_readability(repair_diarization(load_transcript(state["input_path"])))
    return {"doc": base}


def revise_node(state: RevisionState) -> dict:
    edited_segments: list[TranscriptSegment] = []
    for chunk in chunk_segments(state["doc"], max_chars=420):
        chunk_lines = [
            f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.speaker or 'UNKNOWN'}: {segment.text}"
            for segment in chunk
        ]
        try:
            result = chain.invoke({"chunk_text": "\n".join(chunk_lines)})
            edited_lines = result["edited_lines"]
        except Exception:
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
                    text=(edited_text.strip() or segment.text),
                    speaker=segment.speaker,
                    words=segment.words,
                    metadata=dict(segment.metadata),
                )
            )

    return {"revised_doc": rebuild_document_from_segments(state["doc"], edited_segments)}


builder = StateGraph(RevisionState)
builder.add_node("prepare", prepare_node)
builder.add_node("revise", revise_node)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "revise")
builder.add_edge("revise", END)
app = builder.compile()

sample_path = ROOT / "sample_data" / "meeting_sample.json"
result = app.invoke({"input_path": str(sample_path)})

print("=" * 60)
print("  Readability revision graph")
print("=" * 60)
for segment in result["revised_doc"].segments[:5]:
    print(f"  {segment.speaker or 'UNKNOWN'}: {segment.text}")
