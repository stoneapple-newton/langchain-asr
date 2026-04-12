"""
Stage 2: LangChain PII Agent
============================
CONCEPT: A LangChain agent uses tools to inspect deterministic candidates,
then returns a structured PII prediction.

Run this file:
  uv run deep_research/pii_redaction/stage_02_agents/01_langchain_pii_agent.py
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
from deep_research.pii_redaction.shared.pii_utils import (
    PiiPredictionModel,
    dataset_path,
    finalize_prediction,
    format_supported_labels,
    load_dataset,
    regex_candidate_entities,
)


VARIANT_NAME = "langchain_agent"


@tool
def regex_candidate_report(text: str) -> str:
    """Return deterministic regex-based PII candidates from the text."""
    return json.dumps(regex_candidate_entities(text), indent=2)


@tool
def supported_redaction_labels() -> str:
    """Return the supported PII labels and canonical placeholders."""
    return format_supported_labels()


def build_agent():
    return create_agent(
        model=create_chat_model(temperature=0, max_tokens=768),
        tools=[regex_candidate_report, supported_redaction_labels],
        system_prompt=(
            "You are a structured PII-redaction agent.\n"
            "Use tools when helpful, especially regex_candidate_report.\n"
            "Return only PII spans copied exactly from the input text.\n"
            "Prefer precision over recall when unsure."
        ),
        response_format=PiiPredictionModel,
    )


def redact_text(text: str) -> dict:
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"Redact PII from this text:\n{text}"}]},
        config={"recursion_limit": 8},
    )
    structured = result["structured_response"].model_dump()
    return finalize_prediction(
        text,
        structured.get("entities", []),
        variant=VARIANT_NAME,
        notes=structured.get("notes", []),
    )


if __name__ == "__main__":
    example = load_dataset(dataset_path())[1]
    result = redact_text(example["text"])
    print("=" * 60)
    print("  LANGCHAIN PII AGENT")
    print("=" * 60)
    print(f"Input:    {example['text']}")
    print(f"Output:   {result['redacted_text']}")
    print(f"Entities: {result['entities']}")
