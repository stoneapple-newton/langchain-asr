"""
Stage 1, File 1: One-Shot Topic Segmentation
=============================================
CONCEPT: Detect topic boundaries deterministically, then send all detected
topics to the LLM in a single call to generate titles and summaries.

Run this file:
  uv run deep_research/asr_topic_segmentation/stage_01_basics/01_one_shot_segmentation.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACK_ROOT = Path(__file__).resolve().parents[2]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

ASR_V2_SHARED = REPO_ROOT / "deep_research" / "asr-v2" / "shared"
if str(ASR_V2_SHARED) not in sys.path:
    sys.path.insert(0, str(ASR_V2_SHARED))

from config import create_chat_model
from langchain_core.prompts import ChatPromptTemplate
from shared.segmentation_utils import (
    detect_boundary_signals,
    format_topic_for_llm,
    group_into_topic_segments,
    render_segmentation_report,
)
from transcript_utils import load_transcript

SAMPLE = TRACK_ROOT / "sample_data" / "team_meeting_sample.json"

# -- Step 1: deterministic boundary detection --------------------------------

doc = load_transcript(SAMPLE)
signals = detect_boundary_signals(doc.segments)
topics = group_into_topic_segments(doc.segments, signals, threshold=0.5)

print(f"Detected {len(topics)} topic segments (before LLM labelling):")
for t in topics:
    print(f"  {t.topic_id}: {t.start_time:.1f}s – {t.end_time:.1f}s ({len(t.segment_ids)} segs)")

# -- Step 2: one-shot LLM labelling ------------------------------------------

all_topic_texts = "\n\n".join(
    format_topic_for_llm(topic, doc.segments) for topic in topics
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a meeting analyst. For each topic block below, provide a short "
        "title (max 8 words) and a one-sentence summary of what was discussed. "
        "Respond in plain text, one block per topic, using the format:\n"
        "[topic_id] Title: <title>\nSummary: <summary>",
    ),
    ("human", "{topic_blocks}"),
])

llm = create_chat_model(temperature=0, max_tokens=1024)
chain = prompt | llm
response = chain.invoke({"topic_blocks": all_topic_texts})
print("\n--- LLM Topic Labels ---")
print(response.content)

# -- Step 3: render naive report ----------------------------------------------
print("\n--- Deterministic Report ---")
print(render_segmentation_report(topics))
