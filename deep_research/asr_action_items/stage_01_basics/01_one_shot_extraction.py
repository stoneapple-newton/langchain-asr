"""
Stage 1, File 1: One-Shot Action Item Extraction
=================================================
CONCEPT: Detect candidate segments deterministically with regex, then send
all candidates to the LLM in a single call for structured classification.

Run this file:
  uv run deep_research/asr_action_items/stage_01_basics/01_one_shot_extraction.py
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
from shared.action_item_utils import (
    detect_candidates,
    format_candidates_for_llm,
    render_action_items_report,
)
from transcript_utils import load_transcript

SAMPLE = TRACK_ROOT / "sample_data" / "team_meeting_sample.json"

# -- Step 1: deterministic candidate detection --------------------------------

doc = load_transcript(SAMPLE)
candidates = detect_candidates(doc.segments)

print(f"Found {len(candidates)} candidate segments:")
for c in candidates:
    print(f"  seg={c.segment_id} match_count={c.match_count} types={c.candidate_types}")
    print(f"  \"{c.text}\"")

# -- Step 2: one-shot LLM classification -------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a meeting assistant. Review the candidate transcript segments "
        "below and extract structured action items. For each item output:\n"
        "TYPE: task | decision | open_question | commitment\n"
        "TEXT: concise restatement of the item\n"
        "OWNER: speaker name or UNKNOWN\n"
        "DUE: time reference or NONE\n"
        "CONFIDENCE: 0.0-1.0\n\n"
        "Output one block per action item, separated by ---",
    ),
    ("human", "{candidates}"),
])

llm = create_chat_model(temperature=0, max_tokens=1024)
chain = prompt | llm
response = chain.invoke({"candidates": format_candidates_for_llm(candidates)})
print("\n--- LLM Action Items ---")
print(response.content)
