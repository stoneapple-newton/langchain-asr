# Deep Research Agent Feature

## Feature Summary

The deep research agent track is the general learning path for LangChain and
LangGraph. It starts with basic chat calls and progresses through tools,
graphs, RAG, memory, a research agent, and production-oriented examples.

## Business Outcome

This track gives developers a baseline implementation pattern before they work
on more specialized tracks. It is the best onboarding path for understanding
the repository's framework progression.

## Scope

Primary folder:

- `deep_research/deep_research_agent/`

Stages:

- `stage_01_basics`: chat calls, prompt templates, LCEL chains.
- `stage_02_tools`: custom tools and web search.
- `stage_03_langgraph`: state graphs, conditional edges, ReAct-style graph.
- `stage_04_rag`: embeddings and retrieval chains.
- `stage_05_memory`: conversation memory and persistent memory.
- `stage_06_deep_research_agent`: full research workflow.
- `stage_07_production`: observability, HITL, advanced RAG, async serving,
  multi-agent examples.

## Current Architecture

This track is script-oriented. Most stages are self-contained files with
top-level runnable examples. The common architectural pattern is:

1. Create provider-agnostic model objects with `config.create_chat_model()`.
2. Compose prompts, models, and parsers with LCEL.
3. Add tools when the LLM needs external actions.
4. Move to LangGraph when the workflow needs explicit state or routing.
5. Add RAG, memory, tracing, or human approval as separate stage concerns.

## Key APIs and Data Contracts

- LCEL chains: `prompt | llm | parser`.
- LangChain tools: functions decorated with `@tool`.
- LangGraph states: `TypedDict` schemas passed between graph nodes.
- RAG examples: embeddings, retrievers, text splitters, and document objects.
- Production examples: Runnable config, LangSmith tracing metadata, async
  invocation patterns, and multi-agent coordination patterns.

## Dependencies

- Shared config layer.
- LangChain core, community tools, text splitters, and provider integrations.
- LangGraph.
- LangSmith for observability examples.
- Search providers such as Tavily or DuckDuckGo fallback patterns.

## Nonfunctional Requirements

- Examples should remain readable and minimal.
- Each stage should show one main concept clearly.
- Scripts should avoid hardcoding provider classes.
- External services should be configured through `.env`.

## Acceptance Criteria

- New examples use `create_chat_model()` instead of direct provider imports.
- Stage scripts can be run independently.
- Tool examples document the tool contract through function docstrings.
- Graph examples keep node functions small and state updates explicit.
- RAG examples isolate loading, splitting, embedding, retrieval, and answering.

## Risks and Open Questions

- Many examples run logic at import time, which is acceptable for tutorials but
  less convenient for automated tests.
- Some production examples may require credentials or optional services.
- The track is broad, so patterns can drift if new examples skip shared config.

## Development Guidance

When adding new learning stages, make the progression explicit. If a concept
already exists in another track, link to it rather than duplicating the full
implementation. Keep reusable helpers out of top-level scripts when the logic
will be tested or reused.
