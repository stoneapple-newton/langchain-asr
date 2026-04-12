"""
Stage 5, File 1: Conversation Memory with LangGraph
=====================================================
CONCEPT: Persistent state across multiple turns — without passing history manually.

In Stage 3 and 4 we built graphs and RAG chains. But there was a problem:
every call to .invoke() started from scratch. For a chatbot, you want the
agent to remember everything said in the current session.

LangGraph solves this with checkpointers:
  - A checkpointer saves a snapshot of the full graph state after every node
  - On subsequent calls with the SAME thread_id, the state is automatically
    loaded from the checkpoint and merged with new input
  - Different thread_ids = completely isolated conversation histories

Key classes introduced:
  - MemorySaver                   : in-process checkpointer (lives until process ends)
  - {"configurable": {"thread_id"}} : config dict passed to .invoke() / .stream()
  - get_state()                   : inspect the current saved state for a thread
  - get_state_history()           : list all checkpoints (for time-travel in File 2)

Run this file:
  uv run deep_research/deep_research_agent/stage_05_memory/01_conversation_memory.py
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


llm = create_chat_model(
    temperature=0.3,
    max_tokens=256,
)

SYSTEM_PROMPT = SystemMessage(
    content="You are a helpful, friendly assistant. Keep answers concise (2-3 sentences)."
)


# ---------------------------------------------------------------------------
# 1. The simplest persistent chatbot
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: ChatState) -> dict:
    """Single LLM turn — reads full history, appends reply."""
    messages = state["messages"]

    # Prepend system prompt if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_PROMPT] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


# --- Build with a checkpointer --------------------------------------------
# The only difference from Stage 3: pass checkpointer=MemorySaver() to compile()
checkpointer = MemorySaver()

builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# ← this single argument enables full persistence
chatbot = builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 2. Multi-turn conversation using thread_id
# ---------------------------------------------------------------------------
# Every .invoke() call receives a config dict with a thread_id.
# LangGraph uses the thread_id to find the right checkpoint and resume from it.

def chat(thread_id: str, user_message: str) -> str:
    """Send one message; the graph remembers the full history internally."""
    config = {"configurable": {"thread_id": thread_id}}
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content


print("=" * 60)
print("  Thread A: a multi-turn conversation")
print("=" * 60)

turns_a = [
    "Hi! My name is Alex.",
    "What is a vector embedding?",
    "Can you give me a short example?",
    "What was the first thing I told you in this conversation?",  # tests memory
]

for msg in turns_a:
    print(f"\n[User]  {msg}")
    reply = chat("thread_A", msg)
    print(f"[Bot]   {reply}")


# ---------------------------------------------------------------------------
# 3. Thread isolation
# ---------------------------------------------------------------------------
# Starting a new thread_id gives a fresh conversation — no shared history.

print("\n" + "=" * 60)
print("  Thread B: an isolated conversation (no knowledge of Thread A)")
print("=" * 60)

turns_b = [
    "Hello! My name is Jordan.",
    "Do you know anyone named Alex?",   # should NOT recall Thread A
]

for msg in turns_b:
    print(f"\n[User]  {msg}")
    reply = chat("thread_B", msg)
    print(f"[Bot]   {reply}")


# ---------------------------------------------------------------------------
# 4. Inspect saved state
# ---------------------------------------------------------------------------
# get_state() returns the latest checkpoint for a given thread.

print("\n" + "=" * 60)
print("  Inspecting saved state for Thread A")
print("=" * 60)

config_a = {"configurable": {"thread_id": "thread_A"}}
state_snapshot = chatbot.get_state(config_a)

print(f"Number of messages saved : {len(state_snapshot.values['messages'])}")
print(f"Next node (if any)       : {state_snapshot.next}")
print("\nFull message history:")
for msg in state_snapshot.values["messages"]:
    role = type(msg).__name__.replace("Message", "")
    print(f"  [{role:6s}] {msg.content[:80]}")
print()


# ---------------------------------------------------------------------------
# 5. Message trimming — prevent context window overflow
# ---------------------------------------------------------------------------
# Long conversations grow the message list indefinitely. trim_messages() lets
# you keep only the N most recent tokens / messages before passing to the LLM.

print("=" * 60)
print("  Trimming messages to stay within context limits")
print("=" * 60)


def chatbot_with_trim(state: ChatState) -> dict:
    """Like chatbot_node, but trims old messages to fit the context window."""
    messages = state["messages"]

    # Keep the 6 most recent messages (excluding system) — adjust for your model
    trimmed = trim_messages(
        messages,
        max_tokens=6,           # here: max 6 *messages* (token_counter=len)
        strategy="last",        # keep the most recent
        token_counter=len,      # use message count instead of token count
        include_system=True,    # always keep the system prompt
        allow_partial=False,
    )

    if not any(isinstance(m, SystemMessage) for m in trimmed):
        trimmed = [SYSTEM_PROMPT] + trimmed

    response = llm.invoke(trimmed)
    return {"messages": [response]}


trimmed_checkpointer = MemorySaver()
trimmed_builder = StateGraph(ChatState)
trimmed_builder.add_node("chatbot", chatbot_with_trim)
trimmed_builder.add_edge(START, "chatbot")
trimmed_builder.add_edge("chatbot", END)
trimmed_chatbot = trimmed_builder.compile(checkpointer=trimmed_checkpointer)

def chat_trimmed(thread_id: str, msg: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = trimmed_chatbot.invoke(
        {"messages": [HumanMessage(content=msg)]},
        config=config,
    )
    return result["messages"][-1].content

print("Sending 4 messages — the bot only 'sees' the last 6 messages in context:")
for msg in ["My favourite language is Python.", "I love data science.", "What is 2+2?", "What language did I say I liked?"]:
    print(f"\n[User] {msg}")
    print(f"[Bot]  {chat_trimmed('trim_thread', msg)}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ checkpointer=MemorySaver() in compile() enables full conversation persistence
# ✅ config={"configurable": {"thread_id": id}} selects the conversation slot
# ✅ Each thread_id is a completely independent conversation
# ✅ get_state(config) inspects the saved state at any time
# ✅ trim_messages() prevents context window overflow in long conversations
