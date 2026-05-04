"""
Stage 3: LangGraph PII Workflow
===============================
CONCEPT: A small state graph pre-screens candidates, runs LLM extraction, then
reviews the draft before producing a canonical redacted output.

Run this file:
  uv run deep_research/pii_redaction/stage_03_langgraph/01_pii_redaction_workflow.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing_extensions import TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, structured_output_chain
from deep_research.pii_redaction.shared.pii_utils import (
    PiiPredictionModel,
    dataset_path,
    finalize_prediction,
    format_supported_labels,
    load_dataset,
    regex_candidate_entities,
)


VARIANT_NAME = "langgraph_workflow"


class RedactionState(TypedDict):
    text: str
    regex_candidates: list[dict]
    draft_entities: list[dict]
    reviewed_entities: list[dict]
    result: dict


parser = JsonOutputParser()
detect_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract direct PII spans from the input.\n"
            "Use only supported labels:\n{supported_labels}\n"
            "Return JSON only.\n{format_instructions}",
        ),
        (
            "human",
            "Text:\n{text}\n\nRegex candidates:\n{regex_candidates}",
        ),
    ]
).partial(
    supported_labels=format_supported_labels(),
    format_instructions=parser.get_format_instructions(),
)

review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Review a draft PII extraction. Remove false positives and add any obvious misses.\n"
            "All values must be exact substrings from the original text.\n"
            "Return JSON only.\n{format_instructions}",
        ),
        (
            "human",
            "Text:\n{text}\n\nDraft entities:\n{draft_entities}",
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())


def candidate_node(state: RedactionState) -> dict:
    return {"regex_candidates": regex_candidate_entities(state["text"])}


def detect_node(state: RedactionState) -> dict:
    llm = create_chat_model(temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, detect_prompt, PiiPredictionModel)
    raw = chain.invoke(
        {
            "text": state["text"],
            "regex_candidates": json.dumps(state["regex_candidates"], indent=2),
        }
    )
    return {"draft_entities": raw.get("entities", [])}


def review_node(state: RedactionState) -> dict:
    llm = create_chat_model(temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, review_prompt, PiiPredictionModel)
    raw = chain.invoke(
        {
            "text": state["text"],
            "draft_entities": json.dumps(state["draft_entities"], indent=2),
        }
    )
    return {"reviewed_entities": raw.get("entities", [])}


def finalize_node(state: RedactionState) -> dict:
    result = finalize_prediction(
        state["text"],
        state["reviewed_entities"],
        variant=VARIANT_NAME,
        notes=["candidate -> detect -> review workflow"],
    )
    return {"result": result}


def build_graph():
    builder = StateGraph(RedactionState)
    builder.add_node("candidate", candidate_node)
    builder.add_node("detect", detect_node)
    builder.add_node("review", review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "candidate")
    builder.add_edge("candidate", "detect")
    builder.add_edge("detect", "review")
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def redact_text(text: str) -> dict:
    graph = build_graph()
    final_state = graph.invoke(
        {
            "text": text,
            "regex_candidates": [],
            "draft_entities": [],
            "reviewed_entities": [],
            "result": {},
        }
    )
    return final_state["result"]


if __name__ == "__main__":
    example = load_dataset(dataset_path())[2]
    result = redact_text(example["text"])
    print("=" * 60)
    print("  LANGGRAPH PII WORKFLOW")
    print("=" * 60)
    print(f"Input:    {example['text']}")
    print(f"Output:   {result['redacted_text']}")
    print(f"Entities: {result['entities']}")
