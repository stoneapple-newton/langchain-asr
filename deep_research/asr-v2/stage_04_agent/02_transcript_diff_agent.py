"""
Stage 4, File 2: Transcript Difference Agent
============================================
CONCEPT: Create a tool-calling LangChain agent that compares two ASR transcripts
and categorizes differences (wording, speaker, timing, insertion/deletion, etc).

Run this file:
  uv run deep_research/asr-v2/stage_04_agent/02_transcript_diff_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.transcript_diff import build_comparison_report, format_report_markdown


@tool
def compare_asr_transcripts(reference_path: str, candidate_path: str, include_unchanged: bool = False) -> str:
    """Compare and categorize differences between two WhisperX-like transcript JSON files.

    Args:
        reference_path: File path to the baseline/reference transcript JSON.
        candidate_path: File path to the candidate/new transcript JSON.
        include_unchanged: Whether to include unchanged segments in the markdown output.
    """

    report = build_comparison_report(reference_path, candidate_path)
    markdown = format_report_markdown(report, include_unchanged=include_unchanged)
    return json.dumps({"report": report, "markdown": markdown}, ensure_ascii=True)


SYSTEM_PROMPT = """You are an ASR transcript QA assistant.
Use compare_asr_transcripts when a user asks to compare transcript versions.
When responding:
1) Start with a category summary.
2) Highlight high-risk differences first (speaker_change, semantic_shift, deletion, insertion).
3) Give concise recommendations for manual review.
4) If the user asks for raw output, include the JSON report.
"""


def run_demo() -> None:
    agent = create_agent(
        model=create_chat_model("asr_v2", temperature=0),
        tools=[compare_asr_transcripts],
        system_prompt=SYSTEM_PROMPT,
    )

    reference = ROOT / "sample_data" / "meeting_sample.json"
    candidate = ROOT / "sample_data" / "meeting_sample_variant.json"

    user_prompt = (
        "Compare these two ASR transcripts and categorize the differences. "
        "Call the comparison tool and return a short review summary.\n"
        f"reference_path={reference}\n"
        f"candidate_path={candidate}"
    )

    result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    run_demo()
