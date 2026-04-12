# Provider-Aware LangChain Setup

This repo now uses a shared `config/` package to create LangChain chat models
and embeddings from `.env` settings. Ollama remains the default, but the same
structure supports OpenAI, OpenRouter, and Azure OpenAI.

## Install

```powershell
uv sync
```

If you use Ollama, start the local server and pull the default models:

```powershell
ollama serve
ollama pull gemma4:e2b
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

## Configure `.env`

Copy `.env.example` to `.env` and set only the providers you need.

Canonical profile-based settings:

```env
CHAT__DEFAULT_PROFILE=default
CHAT__PROFILES__DEFAULT__PROVIDER=ollama
CHAT__PROFILES__DEFAULT__MODEL=gemma4:e2b

CHAT__PROFILES__ASR_V2__PROVIDER=ollama
CHAT__PROFILES__ASR_V2__MODEL=gemma4:e4b

EMBEDDINGS__DEFAULT_PROFILE=default
EMBEDDINGS__PROFILES__DEFAULT__PROVIDER=ollama
EMBEDDINGS__PROFILES__DEFAULT__MODEL=nomic-embed-text

PROVIDERS__OLLAMA__BASE_URL=http://localhost:11434
SEARCH__TAVILY_API_KEY=
TRACING__API_KEY=
```

Legacy flat vars such as `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`,
`OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`, and
`LANGCHAIN_API_KEY` are still accepted for backward compatibility.

## Usage Pattern

```python
from config import create_chat_model, create_embeddings

llm = create_chat_model()
llm_asr = create_chat_model("asr_v2", max_tokens=768)
embeddings = create_embeddings()
```

`max_tokens` is normalized by the factory and mapped to the right provider
argument internally.

## Run Tutorials

```powershell
uv run .\deep_research\deep_research_agent\stage_01_basics\01_hello_langchain.py
uv run .\deep_research\deep_research_agent\stage_02_tools\02_web_search.py
uv run .\deep_research\asr-v2\stage_02_tools\02_llm_readability_editor.py
```

## Notes

- Tavily search still falls back to DuckDuckGo when no Tavily key is configured.
- The default chat profile is `default`; ASR v2 scripts use the `asr_v2` profile.
- Provider extension guidance lives in [config/README.md](/C:/Users/Newto/Documents/project/test-langchain/config/README.md).
