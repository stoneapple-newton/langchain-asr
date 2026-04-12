"""
Stage 1, File 2: Prompt Templates
===================================
CONCEPT: Reusable, parameterized prompts — like f-strings but smarter.

Hard-coding prompts inside llm.invoke() doesn't scale. LangChain's
PromptTemplate lets you define a prompt once with {placeholders}, then
fill them in at call-time.

Key classes introduced:
  - ChatPromptTemplate        : multi-turn prompt (system + human + ai)
  - HumanMessagePromptTemplate: a human turn that accepts variables
  - PromptTemplate            : for simpler single-string prompts
  - .format_messages()        : renders the template into actual messages
  - .invoke()                 : renders AND calls the LLM (when piped)

Run this file:
  python 02_prompt_templates.py
"""

import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model

llm = create_chat_model(temperature=0)

# ---------------------------------------------------------------------------
# 1. Basic ChatPromptTemplate
# ---------------------------------------------------------------------------
# Variables are denoted by {curly_braces}
print("=== 1. Basic ChatPromptTemplate ===")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant specializing in {domain}."),
    ("human", "Give me a brief overview of: {topic}"),
])

# .format_messages() renders the template — useful for inspecting before sending
messages = prompt.format_messages(domain="machine learning", topic="transformer architecture")
for m in messages:
    print(f"[{m.type}] {m.content}")
print()

# ---------------------------------------------------------------------------
# 2. Directly invoking a prompt (returns formatted messages)
# ---------------------------------------------------------------------------
print("=== 2. prompt.invoke() ===")
prompt_value = prompt.invoke({"domain": "neuroscience", "topic": "neuroplasticity"})
print(prompt_value.messages)  # list of BaseMessage objects
print()

# ---------------------------------------------------------------------------
# 3. LCEL: Chaining prompt → llm with the pipe operator
# ---------------------------------------------------------------------------
# The | operator is LangChain Expression Language (LCEL).
# It creates a Runnable chain: input → prompt renders → llm calls → output
print("=== 3. LCEL Pipe: prompt | llm ===")

chain = prompt | llm   # This is a RunnableSequence

response = chain.invoke({"domain": "quantum computing", "topic": "quantum entanglement"})
print(response.content)
print()

# ---------------------------------------------------------------------------
# 4. Few-shot prompting (teach the model with examples)
# ---------------------------------------------------------------------------
print("=== 4. Few-shot prompt ===")

few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at writing research questions. Study these examples:"),
    ("human", "Topic: climate change"),
    ("ai", "Research question: How do feedback loops in the Arctic amplify global warming rates?"),
    ("human", "Topic: antibiotic resistance"),
    ("ai", "Research question: What evolutionary pressures drive the spread of resistance genes across bacterial populations?"),
    ("human", "Topic: {topic}"),   # <-- this is the actual variable
])

chain2 = few_shot_prompt | llm
result = chain2.invoke({"topic": "large language models"})
print(result.content)
print()

# ---------------------------------------------------------------------------
# 5. PromptTemplate (non-chat, single string — useful for sub-chains)
# ---------------------------------------------------------------------------
print("=== 5. PromptTemplate (simple string) ===")

simple_prompt = PromptTemplate.from_template(
    "Summarize the following text in exactly {num_sentences} sentences:\n\n{text}"
)
rendered = simple_prompt.format(
    num_sentences=2,
    text="LangChain is a framework for building applications powered by language models. It provides a standard interface for chains, integrations with various tools and data sources, and end-to-end chains for common use cases like question answering and summarization."
)
print(rendered)
print()

# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ ChatPromptTemplate.from_messages([("role", "content"), ...])
# ✅ Variables = {curly_braces}
# ✅ | operator chains runnables together (LCEL)
# ✅ Few-shot = include (human, ai) example turns before the real question
