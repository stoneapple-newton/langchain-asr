# test-langchain

An educational LangChain and LangGraph workspace organized as a set of
progressive, runnable examples. The repository starts with simple prompt and
tool patterns, then builds toward graph-based workflows, retrieval, memory,
evaluation, and Deep Agents across several applied problem domains.

The project also includes a shared `config/` package so the same tutorial code
can run against Ollama, OpenAI, OpenRouter, or Azure OpenAI without rewriting
each script.

## What This Repository Is

This is not a single app. It is a collection of learning tracks that show how
to implement the same kinds of tasks at increasing levels of sophistication:

- plain LangChain prompts and LCEL chains
- tool-using agents
- LangGraph state machines and iterative workflows
- RAG and context-grounding patterns
- memory and persistence
- evaluation and benchmarking
- Deep Agents orchestration for more production-like systems

Most folders under `deep_research/` are self-contained tracks with staged
examples, shared helpers, sample data, and sometimes benchmarks or tests.

## Core Design

- `config/` centralizes provider-aware model and embeddings construction
- tutorial code stays provider-agnostic and calls `create_chat_model(...)`
- Ollama is the default local runtime, but hosted providers are supported
- many tracks compare multiple implementation styles for the same task
- tests focus on deterministic utilities, config behavior, and benchmark logic

## Shared Model Factory

All tracks are expected to use the shared factories:

```python
from config import create_chat_model, create_embeddings

llm = create_chat_model()
llm_asr = create_chat_model("asr", max_tokens=768)
llm_asr_v2 = create_chat_model("asr_v2", max_tokens=768)
embeddings = create_embeddings()
```

The factory normalizes provider-specific arguments internally, so swapping
providers usually means updating `.env`, not changing tutorial code.

## Main Learning Tracks

### 1. Deep Research Agent

Location: [deep_research/deep_research_agent](./deep_research/deep_research_agent)

The general LangChain learning progression. It moves from basics to tools,
LangGraph, RAG, memory, a deep research agent, and production-oriented topics
such as observability, human-in-the-loop, advanced RAG, async serving, and
multi-agent patterns.

Representative stages:

- `stage_01_basics`: prompts, prompt templates, LCEL chains
- `stage_02_tools`: custom tools and web search
- `stage_03_langgraph`: state graphs and ReAct-style graph agents
- `stage_04_rag`: embeddings, retrieval chains, conversational RAG
- `stage_05_memory`: conversation and persistent memory
- `stage_06_deep_research_agent`: full deep research workflow
- `stage_07_production`: LangSmith, HITL, advanced RAG, async, multi-agent

### 2. ASR Quality Progression

Location: [deep_research/asr](./deep_research/asr)

An end-to-end progression for improving automatic speech recognition output.
This track covers transcript parsing, quality metrics, LLM cleanup,
speaker/diarization repair, LangGraph pipelines, RAG-based grounding,
production-style agents, and Deep Agents.

Highlights:

- staged learning path from transcript inspection to agent orchestration
- diarization-specific repair and analysis
- focused ASR tests in `deep_research/asr/tests`
- Deep Agents swarm stage with multiple ASR specialists

See also:
- [deep_research/asr/stage_07_deep_agents/README.md](./deep_research/asr/stage_07_deep_agents/README.md)
- [deep_research/ASR_QUALITY_AGENT_CATALOG.md](./deep_research/ASR_QUALITY_AGENT_CATALOG.md)

### 3. ASR-v2

Location: [deep_research/asr-v2](./deep_research/asr-v2)

A cleaner staged tutorial focused on post-processing WhisperX-like JSON into a
more readable, speaker-aware transcript. It mixes deterministic cleanup,
LLM-backed readability editing, LangGraph workflows, and a batch production
runner.

Outputs are written under `outputs/<input-stem>/` as enhanced JSON and Markdown
transcripts.

See:
- [deep_research/asr-v2/README.md](./deep_research/asr-v2/README.md)

### 4. ASR-v2 Translation

Location: [deep_research/asr-v2/translation](./deep_research/asr-v2/translation)

A parallel Chinese-to-English translation track built on top of ASR-v2. It
preserves segment counts, timestamps, and speaker labels while comparing
one-shot, agent, LangGraph, Deep Agents, evaluation, and observability
variants.

This track includes its own dataset, outputs, scoring utilities, and tests.

See:
- [deep_research/asr-v2/translation/README.md](./deep_research/asr-v2/translation/README.md)

### 5. Summary Agents

Location: [deep_research/summary_agents](./deep_research/summary_agents)

A staged series for meeting and transcript summarization. It compares one-shot
prompting, LangChain agents, LangGraph workflows, RAG-enhanced summaries,
memory, Deep Agents, and evaluation.

This is one of the clearest examples of the repository's teaching style:
multiple implementations of the same task with explicit trade-offs in quality,
complexity, and control.

See:
- [deep_research/summary_agents/README.md](./deep_research/summary_agents/README.md)

### 6. PII Redaction

Location: [deep_research/pii_redaction](./deep_research/pii_redaction)

A progression that solves the same redaction task across one-shot prompting,
LangChain agents, LangGraph, and Deep Agents, then benchmarks each approach on
a labeled dataset.

This track is useful if you want a compact example of "same task, different
framework layer" with deterministic scoring.

See:
- [deep_research/pii_redaction/README.md](./deep_research/pii_redaction/README.md)

### 7. Diarization Improvements

Location: [deep_research/diarization_improvements](./deep_research/diarization_improvements)

A focused track for fixing common diarization failures such as run-on segments,
speaker attachment errors, graph-based correction, Deep Agents swarms, and
benchmarking.

### 8. Transcription Correction Agent

Location: [deep_research/transcription_correction_agent](./deep_research/transcription_correction_agent)

A LangGraph-based agent that detects transcript errors, optionally extracts
audio, retranscribes difficult segments, and applies specialized medical-term
handling with glossary support.

See:
- [deep_research/transcription_correction_agent/README.md](./deep_research/transcription_correction_agent/README.md)

### 9. ASR Medical Verification

Location: [deep_research/asr_medical_verification](./deep_research/asr_medical_verification)

Supporting experiments and utilities for medical-term error detection,
audio-segment extraction, retranscription, and transcript verification in
medical ASR scenarios.

## Repository Layout

```text
config/                       Shared provider-aware model factory
deep_research/
  deep_research_agent/        General LangChain/LangGraph progression
  asr/                        Original ASR quality progression
  asr-v2/                     WhisperX-style cleanup and readability track
  summary_agents/             Summarization progression
  pii_redaction/              PII redaction progression + benchmarks
  diarization_improvements/   Focused diarization repair experiments
  asr_medical_verification/   Medical-ASR utilities and experiments
  transcription_correction_agent/
                              LangGraph correction agent
tests/                        Root tests for config and shared utilities
```

## Installation

Python 3.13+ is required.

```powershell
uv sync
```

If you use Ollama, start it locally and pull the default models:

```powershell
ollama serve
ollama pull gemma4:e2b
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

Some Deep Agents examples require an extra install:

```powershell
uv add deepagents
```

## Configuration

Copy `.env.example` to `.env` and configure only the providers you need.

Canonical nested settings example:

```env
CHAT__DEFAULT_PROFILE=default
CHAT__PROFILES__DEFAULT__PROVIDER=ollama
CHAT__PROFILES__DEFAULT__MODEL=gemma4:e2b

CHAT__PROFILES__ASR__PROVIDER=ollama
CHAT__PROFILES__ASR__MODEL=gemma4:e2b

CHAT__PROFILES__ASR_V2__PROVIDER=ollama
CHAT__PROFILES__ASR_V2__MODEL=gemma4:e4b

EMBEDDINGS__DEFAULT_PROFILE=default
EMBEDDINGS__PROFILES__DEFAULT__PROVIDER=ollama
EMBEDDINGS__PROFILES__DEFAULT__MODEL=nomic-embed-text

PROVIDERS__OLLAMA__BASE_URL=http://localhost:11434
SEARCH__TAVILY_API_KEY=
TRACING__API_KEY=
```

Legacy flat environment variables are still supported during migration,
including:

- `OLLAMA_MODEL`
- `OLLAMA_EMBEDDING_MODEL`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGCHAIN_API_KEY`

More detail is in [config/README.md](./config/README.md).

## Common Run Commands

General LangChain track:

```powershell
uv run deep_research/deep_research_agent/stage_01_basics/01_hello_langchain.py
uv run deep_research/deep_research_agent/stage_02_tools/02_web_search.py
uv run deep_research/deep_research_agent/stage_03_langgraph/01_state_graph.py
```

ASR-v2 track:

```powershell
uv run deep_research/asr-v2/stage_01_basics/01_inspect_whisperx_json.py
uv run deep_research/asr-v2/stage_02_tools/02_llm_readability_editor.py
uv run deep_research/asr-v2/stage_05_production/01_batch_asr_quality_runner.py
```

Summary agents:

```powershell
uv run deep_research/summary_agents/stage_01_one_shot/01_basic_summary.py
uv run deep_research/summary_agents/stage_03_langgraph/01_summary_state_graph.py
uv run deep_research/summary_agents/stage_07_evaluation/02_run_benchmarks.py
```

PII redaction:

```powershell
uv run deep_research/pii_redaction/stage_01_basics/01_one_shot_prompt_redactor.py
uv run deep_research/pii_redaction/stage_03_langgraph/01_pii_redaction_workflow.py
uv run deep_research/pii_redaction/stage_05_evaluation/01_benchmark_redaction_variants.py
```

Transcription correction agent:

```powershell
uv run deep_research/transcription_correction_agent/run.py
```

## Testing

The repository includes both root tests and track-specific tests.

Examples:

```powershell
uv run --with pytest pytest tests/test_config.py -q
uv run --with pytest pytest tests/test_pii_redaction_utils.py -q
uv run --with pytest pytest deep_research/asr-v2/tests -q
uv run --with pytest pytest deep_research/asr/tests -q
```

Current root coverage includes:

- shared config behavior
- PII redaction helpers
- ASR Deep Agents smoke coverage
- ASR-v2 translation utilities, variants, and benchmark logic

## How To Approach The Repo

If you are new to the codebase:

1. Start with `deep_research/deep_research_agent/stage_01_basics`.
2. Move to `stage_02_tools` and `stage_03_langgraph` to understand the core
   progression from chains to workflows.
3. Pick one applied track such as `summary_agents`, `pii_redaction`, or
   `asr-v2` to see how the same ideas map to a real task.
4. Use the evaluation folders to compare whether extra complexity actually
   improved quality.

## Related Docs

- [config/README.md](./config/README.md)
- [deep_research/summary_agents/README.md](./deep_research/summary_agents/README.md)
- [deep_research/asr-v2/README.md](./deep_research/asr-v2/README.md)
- [deep_research/asr-v2/translation/README.md](./deep_research/asr-v2/translation/README.md)
- [deep_research/pii_redaction/README.md](./deep_research/pii_redaction/README.md)
- [deep_research/transcription_correction_agent/README.md](./deep_research/transcription_correction_agent/README.md)
