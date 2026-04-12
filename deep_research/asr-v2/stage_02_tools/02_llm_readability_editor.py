"""
Stage 2, File 2: LLM Readability Editor
=======================================
CONCEPT: Use an LLM only for wording cleanup after deterministic structural
repairs. The LLM should not invent timestamps, reorder speakers, or add facts.

Run this file:
  uv run deep_research/asr-v2/stage_02_tools/02_llm_readability_editor.py
"""

import os
from pathlib import Path
import sys

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
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



class EditedChunk(BaseModel):
    edited_lines: list[str] = Field(description="Edited transcript lines in the same order as input")


parser = JsonOutputParser(pydantic_object=EditedChunk)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You edit ASR transcript lines for readability.\n"
        "Rules:\n"
        "- Keep the number of lines exactly the same.\n"
        "- Keep each line's speaker and timestamp prefix unchanged.\n"
        "- Only improve punctuation, spacing, capitalization, and light phrasing cleanup.\n"
        "- Do not add information.\n"
        "Return JSON matching:\n{format_instructions}",
    ),
    ("human", "{chunk_text}"),
]).partial(format_instructions=parser.get_format_instructions())

llm = create_chat_model(
    "asr_v2",
    temperature=0,
    max_tokens=768,
)

chain = prompt | llm | parser

sample_path = ROOT / "sample_data" / "meeting_sample.json"
base_doc = improve_readability(repair_diarization(load_transcript(sample_path)))
chunks = chunk_segments(base_doc, max_chars=420)
edited_segments: list[TranscriptSegment] = []

print("=" * 60)
print("  LLM readability editor")
print("=" * 60)

for chunk_index, chunk in enumerate(chunks, start=1):
    chunk_lines = [
        f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.speaker or 'UNKNOWN'}: {segment.text}"
        for segment in chunk
    ]
    chunk_text = "\n".join(chunk_lines)
    print(f"\n[chunk {chunk_index}] input lines: {len(chunk_lines)}")

    try:
        result = chain.invoke({"chunk_text": chunk_text})
        edited_lines = result["edited_lines"]
    except Exception as exc:
        print(f"  LLM edit failed, keeping deterministic text: {exc}")
        edited_lines = chunk_lines

    if len(edited_lines) != len(chunk_lines):
        print("  Invalid line count from LLM, falling back to deterministic text")
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

edited_doc = rebuild_document_from_segments(base_doc, edited_segments)
for segment in edited_doc.segments[:5]:
    print(f"  {segment.speaker or 'UNKNOWN'}: {segment.text}")
