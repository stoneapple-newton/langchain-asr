"""
Stage 4: Deep Agents PII Redactor
=================================
CONCEPT: A Deep Agents supervisor delegates to specialist subagents such as a
detector, reviewer, and formatter.

Run this file after installing deepagents:
  uv add deepagents
  uv run deep_research/pii_redaction/stage_04_deep_agents/01_pii_redaction_deep_agent.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys
from typing import Any

from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from deep_research.pii_redaction.shared.pii_utils import (
    dataset_path,
    finalize_prediction,
    format_supported_labels,
    load_dataset,
    regex_candidate_entities,
)


VARIANT_NAME = "deep_agents"


@tool
def regex_candidate_report(text: str) -> str:
    """Return deterministic regex candidates for PII spans."""
    return json.dumps(regex_candidate_entities(text), indent=2)


@tool
def supported_labels() -> str:
    """Return supported labels and placeholders for canonical redaction."""
    return format_supported_labels()


def build_agent():
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:
        raise RuntimeError(
            "deepagents is not installed. Install it with `uv add deepagents`."
        ) from exc

    return create_deep_agent(
        name="pii-redactor",
        model=create_chat_model(temperature=0, max_tokens=4096),
        tools=[regex_candidate_report, supported_labels],
        system_prompt=(
            "You redact direct PII from text.\n"
            "Delegate to specialists when helpful.\n"
            "Final answer must be valid JSON with keys entities and notes.\n"
            "Every entity must include label, value, and justification.\n"
            "Values must be exact substrings from the user text."
        ),
        subagents=[
            {
                "name": "detector",
                "description": "Find likely PII spans using the text and regex candidates.",
                "system_prompt": "Detect direct PII spans. Use regex_candidate_report first.",
                "tools": [regex_candidate_report, supported_labels],
            },
            {
                "name": "reviewer",
                "description": "Review a draft list of PII spans for misses and false positives.",
                "system_prompt": "Review a draft extraction. Remove weak guesses and add obvious misses.",
                "tools": [supported_labels],
            },
            {
                "name": "formatter",
                "description": "Prepare the final JSON payload for the parent agent.",
                "system_prompt": "Return clean JSON only. Do not add prose before or after the JSON.",
                "tools": [supported_labels],
            },
        ],
        backend=FilesystemBackend(root_dir=str(REPO_ROOT), virtual_mode=True),
        checkpointer=MemorySaver(),
    )


def _extract_json_block(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        joined = "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        joined = str(content)

    match = re.search(r"\{.*\}", joined, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse JSON from Deep Agents output: {joined}")
    return json.loads(match.group(0))


def redact_text(text: str) -> dict:
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"Redact PII from this text:\n{text}"}]},
        config={"configurable": {"thread_id": "pii-redaction-demo"}},
    )
    last_message = result["messages"][-1]
    payload = _extract_json_block(getattr(last_message, "content", last_message))
    return finalize_prediction(
        text,
        payload.get("entities", []),
        variant=VARIANT_NAME,
        notes=payload.get("notes", []),
    )


if __name__ == "__main__":
    example = load_dataset(dataset_path())[3]
    try:
        result = redact_text(example["text"])
    except RuntimeError as exc:
        print(exc)
    else:
        print("=" * 60)
        print("  DEEP AGENTS PII REDACTOR")
        print("=" * 60)
        print(f"Input:    {example['text']}")
        print(f"Output:   {result['redacted_text']}")
        print(f"Entities: {result['entities']}")
