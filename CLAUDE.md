# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational LangChain learning progression using a local Ollama model (no OpenAI costs). Covers 6 planned stages from LangChain basics to a full deep research agent.

## Setup & Prerequisites

```bash
# Install dependencies
uv sync

# Ollama must be running locally before executing any scripts
# Default model: gemma4:e2b (configurable via OLLAMA_MODEL env var)
```

Environment variables (in `.env`):
- `OLLAMA_MODEL` — model name (default: `gemma4:e2b`)
- `OLLAMA_BASE_URL` — default: `http://localhost:11434`
- `TAVILY_API_KEY` — optional; falls back to DuckDuckGo if absent
- `LANGSMITH_API_KEY` — optional; for LangSmith tracing

## Running Scripts

```bash
# Run any tutorial script
uv run deep_research/deep_research_agent/stage_01_basics/01_hello_langchain.py
uv run deep_research/deep_research_agent/stage_02_tools/02_web_search.py
```

There are no automated tests — all scripts are standalone tutorials meant to be run directly.

## Architecture

### Stage Progression

```
deep_research/deep_research_agent/
├── stage_01_basics/      # DONE — LLM basics, prompt templates, LCEL chains
├── stage_02_tools/       # DONE — custom tools, web search integration
├── stage_03_langgraph/   # PLANNED — stateful agent graphs
├── stage_04_rag/         # PLANNED — retrieval-augmented generation
├── stage_05_memory/      # PLANNED — persistence and memory
└── stage_06_deep_research_agent/  # PLANNED — full agent
```

### Core Patterns

**LLM instantiation** (all scripts):
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"))
```

**LCEL composition** (chains, parsers):
```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"var": value})
```

**Tool definition**:
```python
@tool
def my_tool(arg: str) -> str:
    """Docstring becomes the tool description for the LLM."""
    ...

llm_with_tools = llm.bind_tools([my_tool])
```

**Search fallback pattern** (stage 02):
- Prefers `TavilySearch` if `TAVILY_API_KEY` is set
- Falls back to `DuckDuckGoSearchRun` otherwise

### Key Design Principles

- **Composability**: Everything implements the `Runnable` interface and chains with `|`
- **LLM-agnostic**: Scripts work with any `ChatOllama`-compatible model; swapping to OpenAI requires only changing the LLM class
- **Tool separation**: LLM decides *when* to call tools; the framework executes them

## Skills

LangChain-specific skills are linked from `langchain-ai/langchain-skills` and available in `.agents/skills/` and `.claude/skills/`. Invoke them for guidance on:
- `framework-selection` — choosing between LangChain, LangGraph, Deep Agents
- `langgraph-fundamentals` — StateGraph, nodes, edges before writing any LangGraph code
- `langchain-rag` — RAG systems (document loaders, splitters, vector stores)
- `langchain-fundamentals` — agents, tools, middleware
