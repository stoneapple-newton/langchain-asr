# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational LangChain learning progression with a shared provider-aware config
layer. Ollama is still the default runtime, but the repo now supports OpenAI,
OpenRouter, and Azure OpenAI through the same `config/` factories.

## Setup & Prerequisites

```bash
# Install dependencies
uv sync

# If you use Ollama, it must be running locally before executing scripts
# Default chat profile: "default" -> ollama / gemma4:e2b
# ASR v2 chat profile: "asr_v2" -> ollama / gemma4:e4b
```

Environment variables (in `.env`):
- Canonical settings use nested env vars such as `CHAT__PROFILES__...`,
  `EMBEDDINGS__PROFILES__...`, and `PROVIDERS__...`
- See `.env.example` for the supported structure
- Legacy flat vars like `OLLAMA_MODEL`, `OPENAI_API_KEY`, `TAVILY_API_KEY`,
  `LANGSMITH_API_KEY`, and `LANGCHAIN_API_KEY` are still accepted

## Running Scripts

```bash
# Run any tutorial script
uv run deep_research/deep_research_agent/stage_01_basics/01_hello_langchain.py
uv run deep_research/deep_research_agent/stage_02_tools/02_web_search.py
```

Automated tests now cover both the ASR helpers and the shared config layer.

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
from config import create_chat_model, create_embeddings

llm = create_chat_model()
llm_asr = create_chat_model("asr_v2", max_tokens=768)
embeddings = create_embeddings()
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
- Prefers `TavilySearch` if `SEARCH__TAVILY_API_KEY` or legacy `TAVILY_API_KEY` is set
- Falls back to `DuckDuckGoSearchRun` otherwise

### Key Design Principles

- **Composability**: Everything implements the `Runnable` interface and chains with `|`
- **LLM-agnostic**: Scripts use shared factories, so provider swaps live in `.env`, not in tutorial code
- **Tool separation**: LLM decides *when* to call tools; the framework executes them

## Skills

LangChain-specific skills are linked from `langchain-ai/langchain-skills` and available in `.agents/skills/` and `.claude/skills/`. Invoke them for guidance on:
- `framework-selection` — choosing between LangChain, LangGraph, Deep Agents
- `langgraph-fundamentals` — StateGraph, nodes, edges before writing any LangGraph code
- `langchain-rag` — RAG systems (document loaders, splitters, vector stores)
- `langchain-fundamentals` — agents, tools, middleware
