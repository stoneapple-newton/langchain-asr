"""
Stage 3, File 2: Conditional Edges & Routing
=============================================
CONCEPT: Branching and looping — the heart of dynamic agents.

Fixed edges (A → B always) are too rigid for agents that need to decide what
to do next. Conditional edges let a router function inspect the current state
and return a string that selects the next node.

Key patterns introduced:
  - add_conditional_edges()  : register a router function for a source node
  - Router function          : takes state, returns a string key (node name)
  - Loops                    : edges that route back to a previous node
  - Termination guard        : a counter or flag to prevent infinite loops

Scenario: a "quality-check" pipeline that:
  1. Generates an answer
  2. Grades it (good / needs_revision)
  3. Either returns the answer OR revises it (up to MAX_REVISIONS times)

Run this file:
  uv run deep_research/deep_research_agent/stage_03_langgraph/02_conditional_edges.py
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


llm = create_chat_model(
    temperature=0,
    max_tokens=512,
)

MAX_REVISIONS = 2   # safety guard against infinite loops


# ---------------------------------------------------------------------------
# 1. State schema
# ---------------------------------------------------------------------------

class QAState(TypedDict):
    question: str               # the original user question
    answer: str                 # most recent generated answer
    feedback: str               # critique from the grader
    revision_count: int         # how many revisions have been made
    messages: Annotated[list, add_messages]  # full conversation history


# ---------------------------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------------------------

def generate_answer(state: QAState) -> dict:
    """Generate (or revise) an answer to the question."""
    prompt_parts = [
        SystemMessage(content="You are a helpful assistant. Give a clear, accurate answer."),
    ]

    if state.get("feedback"):
        # On revision rounds, include prior feedback
        prompt_parts.append(SystemMessage(
            content=f"Previous feedback on your answer: {state['feedback']}\n"
                    f"Please revise your answer to address this feedback."
        ))

    prompt_parts.append(HumanMessage(content=state["question"]))

    response = llm.invoke(prompt_parts)
    count = state.get("revision_count", 0)
    label = "initial" if count == 0 else f"revision #{count}"
    print(f"  [generate] ({label}) → {response.content[:80]}...")
    return {
        "answer": response.content,
        "messages": [response],
    }


def grade_answer(state: QAState) -> dict:
    """Grade the answer. Return feedback and a pass/fail verdict in the feedback string."""
    grader_prompt = [
        SystemMessage(content=(
            "You are a strict answer grader. Evaluate the answer for:\n"
            "  • Accuracy\n"
            "  • Completeness (at least 2 key points covered)\n"
            "  • Clarity\n\n"
            "Reply with EXACTLY one of:\n"
            "  PASS\n"
            "  NEEDS_REVISION: <brief reason>\n"
            "Nothing else."
        )),
        HumanMessage(content=f"Question: {state['question']}\n\nAnswer: {state['answer']}"),
    ]
    verdict = llm.invoke(grader_prompt).content.strip()
    print(f"  [grade]    → {verdict[:80]}")
    return {"feedback": verdict}


def revise_answer(state: QAState) -> dict:
    """Increment the revision counter (actual revision happens in generate_answer)."""
    new_count = state.get("revision_count", 0) + 1
    print(f"  [revise]   → scheduling revision #{new_count}")
    return {"revision_count": new_count}


def finalize(state: QAState) -> dict:
    """Package the final accepted answer."""
    print(f"  [finalize] → answer accepted after {state.get('revision_count', 0)} revision(s)")
    return {}   # no state changes needed; just a terminal node


# ---------------------------------------------------------------------------
# 3. Router function — the key to conditional edges
# ---------------------------------------------------------------------------
# This function is called AFTER grade_answer completes.
# It inspects the state and returns a string that selects the next node.

def route_after_grade(state: QAState) -> str:
    """Return 'revise' if the answer needs work, 'finalize' if it passes."""
    feedback = state.get("feedback", "")
    revision_count = state.get("revision_count", 0)

    # Safety: never exceed MAX_REVISIONS
    if revision_count >= MAX_REVISIONS:
        print(f"  [router]   → max revisions reached, forcing finalize")
        return "finalize"

    if feedback.startswith("PASS"):
        return "finalize"
    else:
        return "revise"


# ---------------------------------------------------------------------------
# 4. Build the graph with conditional edges
# ---------------------------------------------------------------------------

builder = StateGraph(QAState)

builder.add_node("generate", generate_answer)
builder.add_node("grade", grade_answer)
builder.add_node("revise", revise_answer)
builder.add_node("finalize", finalize)

# Fixed edges
builder.add_edge(START, "generate")
builder.add_edge("generate", "grade")
builder.add_edge("revise", "generate")    # ← the LOOP: revise → regenerate
builder.add_edge("finalize", END)

# Conditional edge: after grading, call route_after_grade to decide
#   "revise"   → revise node (which loops back to generate)
#   "finalize" → finalize node (which exits)
builder.add_conditional_edges(
    "grade",                         # source node
    route_after_grade,               # router function
    {                                # mapping: returned string → node name
        "revise": "revise",
        "finalize": "finalize",
    }
)

app = builder.compile()


# ---------------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Example 1: Simple question (likely passes first time)")
print("=" * 60)

result = app.invoke({
    "question": "What is the capital of France and why is it historically significant?",
    "answer": "",
    "feedback": "",
    "revision_count": 0,
    "messages": [],
})
print(f"\nFinal answer: {result['answer'][:200]}")
print(f"Revisions made: {result['revision_count']}")
print()

print("=" * 60)
print("  Example 2: Deliberately vague question (may trigger revision)")
print("=" * 60)

result2 = app.invoke({
    "question": "Explain machine learning",
    "answer": "",
    "feedback": "",
    "revision_count": 0,
    "messages": [],
})
print(f"\nFinal answer preview: {result2['answer'][:200]}")
print(f"Revisions made: {result2['revision_count']}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ add_conditional_edges(source, router_fn, mapping) — core branching primitive
# ✅ Router functions are plain Python: inspect state, return a string key
# ✅ Loops are just edges that point backwards (revise → generate)
# ✅ Always include a termination guard (counter, flag) to prevent infinite loops
# ✅ The mapping dict lets you alias router keys to node names
