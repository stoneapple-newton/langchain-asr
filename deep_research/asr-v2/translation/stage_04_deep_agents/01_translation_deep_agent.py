"""
Stage 4: Deep Agents Translation Supervisor
===========================================
CONCEPT: Use a Deep Agents supervisor with translator and reviewer subagents to
produce aligned English transcript segments.

Install deepagents first:
  uv add deepagents

Run this file:
  uv run deep_research/asr-v2/translation/stage_04_deep_agents/01_translation_deep_agent.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys
from typing import Any

from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_shared import (
    build_langsmith_config,
    finalize_translation_result,
    format_example_notes,
    format_segment_preserving_prompt,
    load_dataset,
    load_source_document,
    save_translation_outputs,
    translation_rules,
)


VARIANT_NAME = "deep_agents"


@tool
def inspect_translation_rules() -> str:
    """Return the translation constraints for segment-preserving output."""
    return translation_rules()


@tool
def inspect_source_segments(source_document_json: str) -> str:
    """Render the source transcript as a numbered segment block."""
    source_document = json.loads(source_document_json)
    doc = load_source_document(source_document)
    return format_segment_preserving_prompt(doc)


@tool
def inspect_terminology_notes(notes_text: str) -> str:
    """Return example-specific notes about terminology or ambiguity."""
    return notes_text


def build_translation_deep_agent():
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:
        raise RuntimeError(
            "deepagents is not installed. Install it with `uv add deepagents`."
        ) from exc

    return create_deep_agent(
        name="translation-supervisor",
        model=create_chat_model("asr_v2", temperature=0, max_tokens=4096),
        tools=[inspect_translation_rules, inspect_source_segments, inspect_terminology_notes],
        system_prompt=(
            "You supervise a segment-preserving translation workflow.\n"
            "Final answer must be JSON only with keys translated_segments and notes.\n"
            "translated_segments must be a list of English strings aligned 1:1 with the source segments."
        ),
        subagents=[
            {
                "name": "translator",
                "description": "Produce a first-pass English translation for each source segment.",
                "system_prompt": (
                    "Translate each Chinese source segment into natural English.\n"
                    "Keep the segment order unchanged.\n"
                    "Return JSON only with translated_segments and notes."
                ),
                "tools": [inspect_translation_rules, inspect_source_segments, inspect_terminology_notes],
            },
            {
                "name": "reviewer",
                "description": "Review the draft translation for fluency and terminology consistency.",
                "system_prompt": (
                    "Review a draft segment translation.\n"
                    "Do not change the number of segments.\n"
                    "Return JSON only with translated_segments and notes."
                ),
                "tools": [inspect_translation_rules, inspect_terminology_notes],
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


def _run_deep_agents_translation(example: dict) -> dict:
    agent = build_translation_deep_agent()
    doc = load_source_document(example["source_document"], source_path=f"{example['id']}.json")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate this Chinese transcript into English.\n"
                        "Preserve the number of segments exactly.\n"
                        "Return JSON only with translated_segments and notes.\n\n"
                        f"Example ID: {example['id']}\n"
                        f"Notes:\n{format_example_notes(example)}\n\n"
                        f"Segments:\n{format_segment_preserving_prompt(doc)}"
                    ),
                }
            ]
        },
        config=build_langsmith_config(
            VARIANT_NAME,
            example,
            extra_tags=["stage_04_deep_agents"],
            thread_id=f"{example['id']}-deepagents",
        ),
    )
    last_message = result["messages"][-1]
    return _extract_json_block(getattr(last_message, "content", last_message))


def translate_example(example: dict) -> dict:
    raw = _run_deep_agents_translation(example)
    return finalize_translation_result(
        example,
        raw.get("translated_segments", []),
        variant=VARIANT_NAME,
        notes=raw.get("notes", []),
        trace_metadata={"variant": VARIANT_NAME, "example_id": example["id"]},
    )


if __name__ == "__main__":
    example = load_dataset()[0]
    try:
        result = translate_example(example)
    except RuntimeError as exc:
        print(exc)
    else:
        paths = save_translation_outputs(example, result, ROOT / "outputs")
        print("=" * 72)
        print("DEEP AGENTS TRANSLATION")
        print("=" * 72)
        for segment in result["translated_segments"]:
            print(f"{segment['speaker']}: {segment['text']}")
        print(f"\nSaved JSON: {paths['json_path']}")
        print(f"Saved Markdown: {paths['markdown_path']}")
