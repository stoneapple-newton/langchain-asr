"""
Stage 2, File 1: Building Custom Tools
========================================
CONCEPT: Tools are functions the LLM can decide to call.

An "agent" differs from a plain chain in one key way: it can USE TOOLS.
The model sees the tool's name and description, decides whether to call it,
and provides the arguments. LangChain executes the function and feeds the
result back to the model.

Key patterns introduced:
  - @tool decorator        : simplest way to create a tool
  - StructuredTool         : tool with a Pydantic schema for inputs
  - Tool.from_function()   : wrap an existing function
  - tool.invoke()          : call a tool directly (for testing)
  - tool.name / .description / .args_schema : introspection

Run this file:
  python 01_custom_tools.py
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# 1. @tool decorator — the quickest way to define a tool
# ---------------------------------------------------------------------------
# The docstring becomes the tool's description — the LLM reads this to decide
# when to use the tool. Write clear, specific docstrings!

@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Use this when the user asks about
    the current time, today's date, or any time-sensitive information."""
    return datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.
    Input must be a valid Python math expression, e.g. '2 ** 10' or 'math.sqrt(144)'.
    Do not use for any non-math operations."""
    try:
        # Safe eval: only expose math module
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def word_counter(text: str) -> dict:
    """Counts words, sentences, and characters in the provided text.
    Returns a dictionary with counts. Use this for text analysis tasks."""
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    return {
        "word_count": len(words),
        "sentence_count": max(1, sentences),
        "character_count": len(text),
        "avg_word_length": round(sum(len(w) for w in words) / max(1, len(words)), 2),
    }


# ---------------------------------------------------------------------------
# 2. StructuredTool — for tools with multiple typed inputs
# ---------------------------------------------------------------------------

class SearchFilterInput(BaseModel):
    """Input schema for the filtered search tool."""
    query: str = Field(description="The search query string")
    max_results: int = Field(default=5, description="Maximum number of results to return (1-20)")
    language: str = Field(default="en", description="Language code, e.g. 'en', 'es', 'fr'")


def _mock_filtered_search(query: str, max_results: int = 5, language: str = "en") -> list[dict]:
    """Simulated search — replace with real search API in stage_02/02."""
    return [
        {"title": f"Result {i}: {query}", "url": f"https://example.com/{i}", "language": language}
        for i in range(1, max_results + 1)
    ]


filtered_search = StructuredTool.from_function(
    func=_mock_filtered_search,
    name="filtered_web_search",
    description="Searches the web with optional filters for result count and language. "
                "Use this to find current information about any topic.",
    args_schema=SearchFilterInput,
)


# ---------------------------------------------------------------------------
# 3. Inspect tools (what the LLM sees)
# ---------------------------------------------------------------------------
print("=== Tool Metadata ===")
for t in [get_current_datetime, calculate, word_counter, filtered_search]:
    print(f"\n📌 {t.name}")
    print(f"   Description: {t.description[:80]}...")
    if hasattr(t, 'args_schema') and t.args_schema:
        schema = t.args_schema.model_json_schema()
        print(f"   Args schema: {list(schema.get('properties', {}).keys())}")
print()


# ---------------------------------------------------------------------------
# 4. Call tools directly (bypass the LLM — great for unit testing)
# ---------------------------------------------------------------------------
print("=== Direct Tool Invocation ===")

# Single-arg tools can be invoked with a string
print(get_current_datetime.invoke({}))

# Multi-arg tools take a dict
print(calculate.invoke({"expression": "math.sqrt(2) ** 10"}))
print(word_counter.invoke({"text": "LangChain tools are callable functions the LLM can use."}))

results = filtered_search.invoke({"query": "LangGraph tutorial", "max_results": 3})
for r in results:
    print(r)
print()


# ---------------------------------------------------------------------------
# 5. Binding tools to the LLM (preview — used fully in 03_react_agent.py)
# ---------------------------------------------------------------------------
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
)

# .bind_tools() tells the LLM about available tools.
# The model can now respond with tool_calls instead of plain text.
llm_with_tools = llm.bind_tools([get_current_datetime, calculate, word_counter])

print("=== LLM with tools bound ===")
response = llm_with_tools.invoke("What is 2 to the power of 32?")
print("Response type:", type(response))
print("Tool calls:", response.tool_calls)   # The LLM wants to call calculate()
print()

# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ @tool turns any function into a LangChain tool
# ✅ The docstring IS the tool description — write it for the LLM, not humans
# ✅ StructuredTool + Pydantic gives you typed, validated multi-arg tools
# ✅ .bind_tools() lets the LLM know which tools are available
# ✅ The LLM returns tool_calls (it doesn't execute them — the framework does)
