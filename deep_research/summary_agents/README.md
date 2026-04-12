# Summary Agents: A Progressive Learning Series

This module provides a progressive learning series for ASR transcription summarization using LangChain frameworks. It demonstrates how different implementations—from simple one-shot prompts to sophisticated Deep Agents—can improve summarization quality.

## Overview

The series progresses through 7 stages, each building on the previous:

| Stage | Topic | Files | Key Concepts |
|-------|-------|-------|--------------|
| 1 | One-Shot Prompts | 3 | LCEL chains, structured output, chunking |
| 2 | LangChain Agents | 3 | ReAct agents, tools, context-aware |
| 3 | LangGraph | 4 | State graphs, iteration, parallelism, routing |
| 4 | RAG-Enhanced | 3 | Historical context, entity grounding, personas |
| 5 | Memory & Persistence | 3 | Thread memory, persistent store, follow-up |
| 6 | Deep Agents | 3 | Deep agent harness, subagent swarms, HITL |
| 7 | Evaluation | 4 | Datasets, benchmarks, tracing, analysis |

## Quick Start

```bash
# Run any stage
uv run deep_research/summary_agents/stage_01_one_shot/01_basic_summary.py
uv run deep_research/summary_agents/stage_03_langgraph/01_summary_state_graph.py

# Create evaluation dataset
uv run deep_research/summary_agents/stage_07_evaluation/01_create_dataset.py

# Run benchmarks
uv run deep_research/summary_agents/stage_07_evaluation/02_run_benchmarks.py
```

## Directory Structure

```
summary_agents/
├── shared/                          # Common utilities
│   ├── transcript_loader.py         # ASR transcript loading
│   ├── evaluation.py                # Evaluation metrics
│   └── sample_data/
│       └── meeting_sample.json      # Sample transcript
│
├── stage_01_one_shot/               # Simple prompting
│   ├── 01_basic_summary.py
│   ├── 02_structured_output.py
│   └── 03_chunked_summary.py
│
├── stage_02_langchain_agents/       # Agent-based
│   ├── 01_react_summary_agent.py
│   ├── 02_multi_stage_summary.py
│   └── 03_summary_with_context.py
│
├── stage_03_langgraph/              # Stateful workflows
│   ├── 01_summary_state_graph.py
│   ├── 02_iterative_refinement.py
│   ├── 03_multi_aspect_summary.py
│   └── 04_conditional_routing.py
│
├── stage_04_rag_enhanced/           # Knowledge-augmented
│   ├── 01_historical_context.py
│   ├── 02_entity_grounding.py
│   └── 03_persona_aware_summary.py
│
├── stage_05_memory_persistent/      # Long-term memory
│   ├── 01_thread_memory.py
│   ├── 02_persistent_store.py
│   └── 03_follow_up_aware.py
│
├── stage_06_deep_agents/            # Production orchestration
│   ├── 01_summary_deep_agent.py
│   ├── 02_subagent_swarm.py
│   └── 03_quality_review_loop.py
│
└── stage_07_evaluation/             # Quality measurement
    ├── 01_create_dataset.py
    ├── 02_run_benchmarks.py
    ├── 03_langsmith_tracing.py
    └── 04_compare_quality.py
```

## Stage-by-Stage Learning Path

### Stage 1: One-Shot Prompts (Foundation)

Learn the basics of LLM summarization:

- **01_basic_summary.py**: Simple prompt templates with LCEL
- **02_structured_output.py**: Pydantic models for reliable parsing
- **03_chunked_summary.py**: Map-reduce for long transcripts

**Key Takeaway**: Start simple, get baseline quality metrics.

### Stage 2: LangChain Agents

Introduce agentic behavior with tools:

- **01_react_summary_agent.py**: Agent decides which tools to use
- **02_multi_stage_summary.py**: Explicit pipeline control
- **03_summary_with_context.py**: Knowledge-augmented summaries

**Key Takeaway**: Agents add flexibility but increase complexity.

### Stage 3: LangGraph

Build stateful, cyclical workflows:

- **01_summary_state_graph.py**: Basic graph with state passing
- **02_iterative_refinement.py**: Quality gates with loops
- **03_multi_aspect_summary.py**: Parallel extraction
- **04_conditional_routing.py**: Dynamic routing by content type

**Key Takeaway**: Explicit workflows offer more control than agents.

### Stage 4: RAG-Enhanced

Add retrieval augmentation:

- **01_historical_context.py**: Context from past meetings
- **02_entity_grounding.py**: Verify names and terms
- **03_persona_aware_summary.py**: Different formats per audience

**Key Takeaway**: Context significantly improves summary quality.

### Stage 5: Memory & Persistence

Enable long-term knowledge:

- **01_thread_memory.py**: Checkpoint and resume workflows
- **02_persistent_store.py**: Cross-session knowledge
- **03_follow_up_aware.py**: Track action items across meetings

**Key Takeaway**: Memory creates continuity across meeting series.

### Stage 6: Deep Agents

Production-grade orchestration:

- **01_summary_deep_agent.py**: Deep agent harness with filesystem
- **02_subagent_swarm.py**: Specialist subagents
- **03_quality_review_loop.py**: Human-in-the-loop approval

**Key Takeaway**: Deep agents provide built-in production features.

### Stage 7: Evaluation

Measure and compare quality:

- **01_create_dataset.py**: Build evaluation dataset
- **02_run_benchmarks.py**: Run all implementations
- **03_langsmith_tracing.py**: Comprehensive observability
- **04_compare_quality.py**: Analyze improvements

**Key Takeaway**: Data-driven decisions about complexity trade-offs.

## Quality Progression

Expected quality improvements across stages:

```
Stage 1.1 (Basic):         ████████░░░░░░░░░░░░  42%
Stage 1.2 (Structured):    ██████████░░░░░░░░░░  55%
Stage 2 (Agents):          ███████████░░░░░░░░░  62%
Stage 3 (LangGraph):       █████████████░░░░░░░  71%
Stage 4 (RAG):             ██████████████░░░░░░  78%
Stage 5+ (Memory/Deep):    ████████████████░░░░  85%+
```

*Scores are composite metrics (ROUGE + LLM-as-judge + element coverage)*

## Key Design Principles

1. **Educational Progression**: Each stage builds on previous concepts
2. **Measurable Quality**: Benchmarks at every stage
3. **ASR-Specific**: Handles speaker attribution, confidence scores
4. **Production-Ready**: Tracing, evaluation, quality gates

## Dependencies

Core (already in project):
- langchain-core
- langchain-community
- langgraph

Optional:
- `deepagents` (for Stage 6)
- `rouge-score` (for evaluation)
- `langsmith` (for tracing)

## Running Examples

```bash
# Stage 1: Basic one-shot
uv run deep_research/summary_agents/stage_01_one_shot/01_basic_summary.py

# Stage 3: LangGraph with iteration
uv run deep_research/summary_agents/stage_03_langgraph/02_iterative_refinement.py

# Stage 4: RAG-enhanced
uv run deep_research/summary_agents/stage_04_rag_enhanced/01_historical_context.py

# Stage 7: Run all benchmarks
uv run deep_research/summary_agents/stage_07_evaluation/02_run_benchmarks.py
```

## Best Practices Learned

1. **Start Simple**: Begin with Stage 1.2 (structured output)
2. **Measure First**: Establish baseline before adding complexity
3. **Add Gradually**: Only add complexity driven by quality gaps
4. **Trace Everything**: Enable LangSmith from day one
5. **Quality Gates**: Human review for high-stakes summaries

## Production Recommendations

| Use Case | Recommended Stage | Rationale |
|----------|------------------|-----------|
| Daily standups | 1.2 | Fast, cheap, good enough |
| Team meetings | 3.1 + 4.1 | Good quality, reasonable cost |
| Board meetings | 4+5+6 | Quality gates + human review |
| Meeting series | 4+5 | Context + follow-up tracking |
| Enterprise | 6 | Deep agents for orchestration |

## Contributing

This is a learning resource. To add new techniques:

1. Add files to appropriate stage directory
2. Update this README
3. Add to evaluation dataset
4. Run benchmarks to measure improvement

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/)
- [Deep Agents Guide](https://docs.deepagents.com/)
