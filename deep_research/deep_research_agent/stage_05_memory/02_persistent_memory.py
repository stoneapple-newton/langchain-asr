"""
Stage 5, File 2: Persistent Memory & Time Travel
==================================================
CONCEPT: State that survives process restarts + rewinding to earlier states.

MemorySaver (File 1) is in-process only — restart the script and history is
gone. For real applications you need disk persistence. SqliteSaver writes every
checkpoint to a SQLite database file, so conversations survive restarts.

Additionally, LangGraph stores EVERY checkpoint in history. This enables
"time travel" — you can replay the graph from any prior state. Useful for:
  • Debugging agent decisions at each step
  • Forking a conversation from a past point
  • Audit trails

Key classes introduced:
  - SqliteSaver          : checkpointer that persists state to a SQLite file
  - get_state_history()  : iterate all checkpoints for a thread (newest first)
  - update_state()       : manually overwrite a checkpoint (for corrections)
  - Forking              : run from a past checkpoint_id to create a new branch

Run this file:
  uv run deep_research/deep_research_agent/stage_05_memory/02_persistent_memory.py
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from pathlib import Path
import sys

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


llm = create_chat_model(
    temperature=0.3,
    max_tokens=256,
)

# ---------------------------------------------------------------------------
# 1. Set up SqliteSaver
# ---------------------------------------------------------------------------
# The database file is created automatically in the project root.
# Re-run this script and the conversation history will still be there!

DB_PATH = Path(__file__).parent / "chat_history.db"
print(f"Using SQLite database: {DB_PATH}")

SYSTEM_PROMPT = SystemMessage(
    content="You are a concise assistant. Reply in 1-2 sentences."
)


# ---------------------------------------------------------------------------
# 2. Build the same graph as File 1 — only the checkpointer changes
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: ChatState) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_PROMPT] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}


# SqliteSaver is used as a context manager so the connection closes cleanly
with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    chatbot = builder.compile(checkpointer=checkpointer)

    def chat(thread_id: str, msg: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config=config,
        )
        return result["messages"][-1].content

    # -----------------------------------------------------------------------
    # 3. Persistent multi-turn conversation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Persistent conversation (thread: persistent_demo)")
    print("  Re-run this script — history will still be there!")
    print("=" * 60)

    thread = "persistent_demo"
    turns = [
        "My name is Sam. I am learning LangChain.",
        "What is the most important concept in LangGraph?",
        "Can you summarise what I've told you about myself?",
    ]

    for msg in turns:
        print(f"\n[User] {msg}")
        reply = chat(thread, msg)
        print(f"[Bot]  {reply}")

    # -----------------------------------------------------------------------
    # 4. Inspect checkpoint history (time travel source material)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Checkpoint history for this thread")
    print("=" * 60)

    config = {"configurable": {"thread_id": thread}}
    checkpoints = list(chatbot.get_state_history(config))

    print(f"Total checkpoints stored: {len(checkpoints)}")
    for i, snapshot in enumerate(checkpoints):
        msg_count = len(snapshot.values.get("messages", []))
        ts = snapshot.config["configurable"].get("checkpoint_id", "N/A")[:20]
        print(f"  [{i}] checkpoint_id: {ts}...  messages: {msg_count}")

    # -----------------------------------------------------------------------
    # 5. Time travel: replay from a past checkpoint
    # -----------------------------------------------------------------------
    # Pick the checkpoint BEFORE the last user message (index 1 = second-to-last)
    if len(checkpoints) >= 2:
        print("\n" + "=" * 60)
        print("  Time travel: forking from an earlier checkpoint")
        print("=" * 60)

        # The second checkpoint (index 1) has fewer messages
        past_snapshot = checkpoints[1]
        past_config = past_snapshot.config

        print(f"Rewinding to checkpoint with {len(past_snapshot.values['messages'])} message(s)")

        # Invoke from the past checkpoint — this creates a NEW branch
        # (the original thread is unchanged)
        fork_config = {
            "configurable": {
                "thread_id": "fork_from_past",
                "checkpoint_id": past_config["configurable"]["checkpoint_id"],
            }
        }

        # Copy the past state into the new thread
        chatbot.update_state(
            {"configurable": {"thread_id": "fork_from_past"}},
            past_snapshot.values,
        )

        fork_reply = chat("fork_from_past", "What is RAG?")
        print(f"\n[User (forked)] What is RAG?")
        print(f"[Bot]           {fork_reply}")

    # -----------------------------------------------------------------------
    # 6. Manual state correction with update_state()
    # -----------------------------------------------------------------------
    # update_state() lets you inject facts or correct mistakes in the history.
    # Useful for:
    #   • Injecting a human approval step
    #   • Correcting a wrong tool result
    #   • Seeding a conversation with facts before it starts

    print("\n" + "=" * 60)
    print("  Manual state update (injecting a correction)")
    print("=" * 60)

    correction_thread = "correction_demo"
    correction_config = {"configurable": {"thread_id": correction_thread}}

    # Seed the conversation with a factual correction
    chatbot.update_state(
        correction_config,
        {"messages": [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="My favourite colour is blue."),
            AIMessage(content="Got it! Your favourite colour is blue."),
        ]},
    )

    # Now ask a follow-up — the injected history provides the context
    print("[Injected] Human said their favourite colour is blue")
    follow_up = chat(correction_thread, "What colour did I just say I liked?")
    print(f"[User] What colour did I just say I liked?")
    print(f"[Bot]  {follow_up}")
    print()

print(f"\nDatabase file preserved at: {DB_PATH}")
print("Re-run this script to verify the conversation persists across restarts.")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ SqliteSaver.from_conn_string(path) — swap in for MemorySaver, zero other changes
# ✅ Use as a context manager (with ...) so the connection closes cleanly
# ✅ Every .invoke() creates a checkpoint — full history is always available
# ✅ get_state_history(config) lists all checkpoints for a thread (newest first)
# ✅ update_state() lets you inject or correct state without running nodes
# ✅ Forking: copy a past state to a new thread_id to explore alternative paths
