"""
Stage 1, File 1: One-Shot Sentiment Analysis
=============================================
CONCEPT: Run a deterministic keyword-based sentiment pass over a transcript,
then send the full result to the LLM for a refined summary.

This is the simplest approach: no graph, no tools, one LLM call.

Run this file:
  uv run deep_research/asr_sentiment_analysis/stage_01_basics/01_one_shot_sentiment.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRACK_ROOT = Path(__file__).resolve().parents[2]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from config import create_chat_model
from langchain_core.prompts import ChatPromptTemplate
from shared.sentiment_utils import (
    aggregate_speaker_profiles,
    format_labels_for_llm,
    render_sentiment_report,
    score_segment_naive,
)

ASR_V2_SHARED = REPO_ROOT / "deep_research" / "asr-v2" / "shared"
if str(ASR_V2_SHARED) not in sys.path:
    sys.path.insert(0, str(ASR_V2_SHARED))

from transcript_utils import load_transcript

SAMPLE = TRACK_ROOT / "sample_data" / "call_center_sample.json"

# -- Step 1: load and score naively ------------------------------------------

doc = load_transcript(SAMPLE)
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
profiles = aggregate_speaker_profiles(labels)

print(render_sentiment_report(labels, profiles))

# -- Step 2: one-shot LLM refinement -----------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sentiment analyst. Review the per-segment labels below "
        "(naive keyword scores) and provide a concise overall assessment of "
        "the emotional arc and any notable turning points. "
        "Limit your response to 5 bullet points.",
    ),
    ("human", "{labeled_segments}"),
])

def build_llm():
    return create_chat_model(temperature=0, max_tokens=512)

chain = prompt | build_llm()
response = chain.invoke({"labeled_segments": format_labels_for_llm(labels)})
print("\n--- LLM Sentiment Summary ---")
print(response.content)
