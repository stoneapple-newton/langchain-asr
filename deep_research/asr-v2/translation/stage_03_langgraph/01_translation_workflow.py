"""
Stage 3: LangGraph Translation Workflow
=======================================
CONCEPT: Translate, review, and finalize segment-preserving output in a small
state graph with explicit validation.

Run this file:
  uv run deep_research/asr-v2/translation/stage_03_langgraph/01_translation_workflow.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, structured_output_chain

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


VARIANT_NAME = "langgraph_workflow"


class TranslationState(TypedDict):
    example: dict[str, Any]
    doc: object
    draft_payload: dict[str, Any]
    reviewed_payload: dict[str, Any]
    result: dict[str, Any]


parser = JsonOutputParser()
draft_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Draft an English translation for each Chinese transcript segment.\n"
            "{rules}\n\n"
            "Return JSON only.\n{format_instructions}",
        ),
        (
            "human",
            "Example ID: {example_id}\n"
            "Terminology notes:\n{notes}\n\n"
            "Segments:\n{segment_block}",
        ),
    ]
).partial(
    rules=translation_rules(),
    format_instructions=parser.get_format_instructions(),
)

review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Review the draft translation.\n"
            "Keep the number of segments identical and preserve the original order.\n"
            "Return JSON only.\n{format_instructions}",
        ),
        (
            "human",
            "Source segments:\n{segment_block}\n\n"
            "Draft translation:\n{draft_payload}",
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())


def prepare_node(state: TranslationState) -> dict[str, Any]:
    doc = load_source_document(
        state["example"]["source_document"],
        source_path=f"{state['example']['id']}.json",
    )
    return {"doc": doc}


def draft_node(state: TranslationState) -> dict[str, Any]:
    llm = create_chat_model("asr_v2", temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, draft_prompt, TranslationDraftModel)
    payload = chain.invoke(
        {
            "example_id": state["example"]["id"],
            "notes": format_example_notes(state["example"]),
            "segment_block": format_segment_preserving_prompt(state["doc"]),
        },
        config=build_langsmith_config(
            VARIANT_NAME,
            state["example"],
            extra_tags=["stage_03_langgraph", "draft"],
            thread_id=f"{state['example']['id']}-draft",
        ),
    )
    return {"draft_payload": payload}


def review_node(state: TranslationState) -> dict[str, Any]:
    llm = create_chat_model("asr_v2", temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, review_prompt, TranslationDraftModel)
    reviewed = chain.invoke(
        {
            "segment_block": format_segment_preserving_prompt(state["doc"]),
            "draft_payload": json.dumps(state["draft_payload"], ensure_ascii=False, indent=2),
        },
        config=build_langsmith_config(
            VARIANT_NAME,
            state["example"],
            extra_tags=["stage_03_langgraph", "review"],
            thread_id=f"{state['example']['id']}-review",
        ),
    )
    return {"reviewed_payload": reviewed}


def finalize_node(state: TranslationState) -> dict[str, Any]:
    reviewed = state["reviewed_payload"]
    result = finalize_translation_result(
        state["example"],
        reviewed.get("translated_segments", []),
        variant=VARIANT_NAME,
        notes=reviewed.get("notes", []),
        trace_metadata={"variant": VARIANT_NAME, "example_id": state["example"]["id"]},
    )
    return {"result": result}


def build_graph():
    builder = StateGraph(TranslationState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("draft", draft_node)
    builder.add_node("review", review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _run_langgraph_workflow(example: dict) -> dict:
    graph = build_graph()
    final_state = graph.invoke(
        {
            "example": example,
            "doc": None,
            "draft_payload": {},
            "reviewed_payload": {},
            "result": {},
        },
        config=build_langsmith_config(
            VARIANT_NAME,
            example,
            extra_tags=["stage_03_langgraph", "graph"],
            thread_id=f"{example['id']}-graph",
        ),
    )
    return final_state["result"]


def translate_example(example: dict) -> dict:
    return _run_langgraph_workflow(example)


if __name__ == "__main__":
    example = load_dataset()[0]
    result = translate_example(example)
    paths = save_translation_outputs(example, result, ROOT / "outputs")

    print("=" * 72)
    print("LANGGRAPH TRANSLATION WORKFLOW")
    print("=" * 72)
    for segment in result["translated_segments"]:
        print(f"{segment['speaker']}: {segment['text']}")
    print(f"\nSaved JSON: {paths['json_path']}")
    print(f"Saved Markdown: {paths['markdown_path']}")
