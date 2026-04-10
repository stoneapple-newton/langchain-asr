"""
Stage 1, File 1: Hello LangChain
=================================
CONCEPT: The LLM wrapper — how LangChain talks to language models.

LangChain's core abstraction is the "chat model": a callable that takes
a list of messages and returns an AI message. Every model (OpenAI, Anthropic,
Gemini, local Ollama) exposes the same interface, so swapping providers is
a one-line change.

Key classes introduced:
  - ChatOpenAI        : wraps OpenAI's chat API
  - HumanMessage      : a message from the user
  - SystemMessage     : instructions that set the model's behavior
  - AIMessage         : the model's response (what you get back)

Run this file:
  python 01_hello_langchain.py
"""

import os
from dotenv import load_dotenv

# langchain_core holds the fundamental building blocks (messages, etc.)
from langchain_core.messages import HumanMessage, SystemMessage

# langchain_openai is the OpenAI-specific integration package
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()  # reads OPENAI_API_KEY from your .env file

# ---------------------------------------------------------------------------
# 1. Instantiate the model
# ---------------------------------------------------------------------------
# temperature=0  → deterministic / factual output (great for research)
# temperature=1  → more creative / varied output
# max_tokens     → cap the response length (saves money during dev)
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    max_tokens=512,
)

# ---------------------------------------------------------------------------
# 2. The simplest possible call: a single string
# ---------------------------------------------------------------------------
print("=== Simple string call ===")
response = llm.invoke("What is LangChain in one sentence?")
print(response.content)   # AIMessage has a .content attribute
print()

# ---------------------------------------------------------------------------
# 3. Using messages for more control
# ---------------------------------------------------------------------------
# SystemMessage shapes how the model responds throughout a conversation.
# HumanMessage is the user's turn.
print("=== Structured message call ===")
messages = [
    SystemMessage(content="You are a concise technical tutor. Keep answers under 3 bullet points."),
    HumanMessage(content="What are the three main components of LangChain?"),
]
response = llm.invoke(messages)
print(response.content)
print()

# ---------------------------------------------------------------------------
# 4. Inspect the AIMessage object
# ---------------------------------------------------------------------------
print("=== AIMessage metadata ===")
print(f"Type:             {type(response)}")
print(f"Content:          {response.content[:80]}...")
print(f"Response metadata: {response.response_metadata}")
print()

# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ llm.invoke()  → synchronous, returns one AIMessage
# ✅ llm.stream()  → streams tokens as they arrive (try it below!)
# ✅ All LangChain chat models share this same interface

# BONUS: Streaming (uncomment to try)
# print("=== Streaming ===")
# for chunk in llm.stream("Tell me 3 fun facts about LangChain:"):
#     print(chunk.content, end="", flush=True)
# print()
