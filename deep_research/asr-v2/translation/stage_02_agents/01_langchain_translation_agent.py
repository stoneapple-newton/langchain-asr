"""
Stage 2: LangChain Translation Agent
====================================
CONCEPT: Wrap the same translation task in a LangChain agent with tools for
viewing segment structure and terminology notes.

Run this file:
  uv run deep_research/asr-v2/translation/stage_02_agents/01_langchain_translation_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_shared import (
    TranslationDraftModel,
    build_langsmith_config,
    finalize_translation_result,
    format_example_notes,
    format_segment_preserving_prompt,
    load_dataset,
    load_source_document,
    save_translation_outputs,
    translation_rules,
)


VARIANT_NAME = "langchain_agent"


@tool
def inspect_source_segments(source_document_json: str) -> str:
    """Render the source transcript as a numbered segment block."""
    source_document = json.loads(source_document_json)
    doc = load_source_document(source_document)
    return format_segment_preserving_prompt(doc)


@tool
def inspect_translation_rules() -> str:
    """Return the translation constraints for this task."""
    return translation_rules()


@tool
def inspect_terminology_notes(notes_text: str) -> str:
    """Return any example-specific terminology notes."""
    return notes_text


def build_translation_agent():
    return create_agent(
        model=create_chat_model("asr_v2", temperature=0, max_tokens=4096),
        tools=[inspect_source_segments, inspect_translation_rules, inspect_terminology_notes],
        system_prompt=(
            "You are a structured transcript-translation agent.\n"
            "Use the tools when helpful, especially to inspect the source segments.\n"
            "Return English translations aligned 1:1 with the input segments.\n"
            "Do not merge or split segments."
        ),
        response_format=TranslationDraftModel,
    )


def _run_langchain_agent(example: dict) -> dict:
    agent = build_translation_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate this Chinese transcript into segment-preserving English.\n\n"
                        f"Example ID: {example['id']}\n"
                        f"Terminology notes:\n{format_example_notes(example)}\n\n"
                        f"Source document JSON:\n{json.dumps(example['source_document'], ensure_ascii=False, indent=2)}"
                    ),
                }
            ]
        },
        config=build_langsmith_config(
            VARIANT_NAME,
            example,
            extra_tags=["stage_02_agents"],
        ),
    )
    structured = result["structured_response"]
    return structured.model_dump() if hasattr(structured, "model_dump") else dict(structured)


def translate_example(example: dict) -> dict:
    raw = _run_langchain_agent(example)
    return finalize_translation_result(
        example,
        raw.get("translated_segments", []),
        variant=VARIANT_NAME,
        notes=raw.get("notes", []),
        trace_metadata={"variant": VARIANT_NAME, "example_id": example["id"]},
    )


if __name__ == "__main__":
    example = load_dataset()[0]
    result = translate_example(example)
    paths = save_translation_outputs(example, result, ROOT / "outputs")

    print("=" * 72)
    print("LANGCHAIN TRANSLATION AGENT")
    print("=" * 72)
    for segment in result["translated_segments"]:
        print(f"{segment['speaker']}: {segment['text']}")
    print(f"\nSaved JSON: {paths['json_path']}")
    print(f"Saved Markdown: {paths['markdown_path']}")
