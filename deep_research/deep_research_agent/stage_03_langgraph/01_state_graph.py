"""
Stage 3, File 1: LangGraph StateGraph Fundamentals
====================================================
CONCEPT: Stateful, cyclical computation graphs.

LCEL chains (Stage 1) are linear DAGs — data flows one way and can't loop.
LangGraph adds three new capabilities:
  1. Shared state  — a typed dict passed into every node and returned as updates
  2. Cycles        — nodes can route back to earlier nodes (essential for agents)
  3. Persistence   — state can be checkpointed and replayed (covered in Stage 5)

Key classes introduced:
  - StateGraph         : the graph builder; parameterised by a state schema
  - TypedDict          : defines the shape (keys + types) of the shared state
  - Annotated          : attaches a "reducer" to a field (how updates are merged)
  - add_messages       : built-in reducer that APPENDS to a message list
  - START / END        : built-in sentinel nodes marking entry and exit
  - .add_node()        : registers a Python function as a graph node
  - .add_edge()        : declares a fixed transition A → B
  - .compile()         : validates the graph and returns a runnable

Run this file:
  uv run deep_research/deep_research_agent/stage_03_langgraph/01_state_graph.py
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
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# PART A — Graph with a plain TypedDict state (no messages)
# ---------------------------------------------------------------------------
# The state is just a Python dict with typed keys. Every node receives the
# current state and returns a dict of keys to update (partial updates are OK).

print("=" * 60)
print("  PART A: Plain TypedDict State")
print("=" * 60)


class TextPipelineState(TypedDict):
    """State shared across every node in this graph."""
    raw_text: str          # input provided by the user
    word_count: int        # filled by the count_words node
    summary: str           # filled by the summarise node
    report: str            # filled by the format_report node


# --- Nodes ----------------------------------------------------------------
# Each node is a plain Python function that:
#   • receives the full current state as its argument
#   • returns a dict with ONLY the keys it wants to update

def count_words(state: TextPipelineState) -> dict:
    """Node 1: count words in raw_text."""
    count = len(state["raw_text"].split())
    print(f"  [count_words]  → {count} words")
    return {"word_count": count}


def summarise(state: TextPipelineState) -> dict:
    """Node 2: ask the LLM for a one-sentence summary."""
    response = llm.invoke([
        SystemMessage(content="Summarise the following text in exactly one sentence."),
        HumanMessage(content=state["raw_text"]),
    ])
    print(f"  [summarise]    → {response.content[:80]}...")
    return {"summary": response.content}


def format_report(state: TextPipelineState) -> dict:
    """Node 3: combine counts + summary into a final report."""
    report = (
        f"Words: {state['word_count']}\n"
        f"Summary: {state['summary']}"
    )
    print(f"  [format_report] → report assembled")
    return {"report": report}


# --- Build the graph -------------------------------------------------------
builder = StateGraph(TextPipelineState)

builder.add_node("count_words", count_words)
builder.add_node("summarise", summarise)
builder.add_node("format_report", format_report)

# Fixed edges: START → count_words → summarise → format_report → END
builder.add_edge(START, "count_words")
builder.add_edge("count_words", "summarise")
builder.add_edge("summarise", "format_report")
builder.add_edge("format_report", END)

graph_a = builder.compile()

# --- Invoke ----------------------------------------------------------------
sample_text = (
    "LangGraph is a library for building stateful, multi-actor applications "
    "with LLMs. It extends LangChain's capabilities by adding support for "
    "cyclic computation graphs, which are essential for creating agent loops."
)

final_state = graph_a.invoke({"raw_text": sample_text})
print("\nFinal state keys:", list(final_state.keys()))
print(final_state["report"])
print()


# ---------------------------------------------------------------------------
# PART B — Streaming through nodes step by step
# ---------------------------------------------------------------------------
# .stream() yields one dict per node as it completes.
# This is how you follow an agent's "thinking" in real time.

print("=" * 60)
print("  PART B: Streaming Node-by-Node")
print("=" * 60)

for step in graph_a.stream({"raw_text": "LangChain makes it easy to build LLM applications."}):
    node_name = list(step.keys())[0]
    updates = step[node_name]
    print(f"  ✓ {node_name}: {list(updates.keys())}")
print()


# ---------------------------------------------------------------------------
# PART C — Message-based state (the standard pattern for chat agents)
# ---------------------------------------------------------------------------
# Instead of ad-hoc keys, most LangGraph agents store ALL conversation turns
# in a single `messages` list. The `add_messages` reducer APPENDS new messages
# rather than replacing the whole list — critical for multi-turn memory.

print("=" * 60)
print("  PART C: Message-Based State (the Agent Pattern)")
print("=" * 60)


class ChatState(TypedDict):
    # Annotated[list, add_messages] means:
    #   • the field type is a list of BaseMessage
    #   • when a node returns {"messages": [new_msg]}, it is APPENDED (not replaced)
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    """A single LLM turn — reads all messages so far, returns the reply."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}   # add_messages will append this


chat_graph = StateGraph(ChatState)
chat_graph.add_node("chat", chat_node)
chat_graph.add_edge(START, "chat")
chat_graph.add_edge("chat", END)
chat_app = chat_graph.compile()

result = chat_app.invoke({
    "messages": [
        SystemMessage(content="You are a concise assistant. Reply in one sentence."),
        HumanMessage(content="What is a StateGraph?"),
    ]
})

print("Messages in final state:")
for msg in result["messages"]:
    role = type(msg).__name__
    print(f"  [{role}] {msg.content[:100]}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ StateGraph replaces LCEL when you need cycles or shared mutable state
# ✅ Each node receives the full state, returns only the keys it changes
# ✅ add_messages reducer APPENDS — without it, each node would overwrite the list
# ✅ .stream() lets you observe each node's output as the graph runs
# ✅ START / END are sentinels — every graph must have a path from START to END
