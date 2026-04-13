# Config Package

This package centralizes model-provider selection and `.env` loading for the
repository.

## Public API

```python
from config import create_chat_model, create_embeddings, get_settings

llm = create_chat_model()
llm_asr = create_chat_model("asr", max_tokens=768)
llm_asr_v2 = create_chat_model("asr_v2", max_tokens=768)
embeddings = create_embeddings()
settings = get_settings()
```

## Structure

- `settings.py` defines the Pydantic v2 settings schema and env compatibility.
- `providers.py` maps provider names to LangChain client constructors.
- `llm.py` exposes the public factory functions.

## Canonical `.env` pattern

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

EMBEDDINGS__PROFILES__ASR__PROVIDER=ollama
EMBEDDINGS__PROFILES__ASR__MODEL=nomic-embed-text

EMBEDDINGS__PROFILES__ASR_V2__PROVIDER=ollama
EMBEDDINGS__PROFILES__ASR_V2__MODEL=nomic-embed-text

PROVIDERS__OLLAMA__BASE_URL=http://localhost:11434
```

Legacy flat vars such as `OLLAMA_MODEL`, `OPENAI_API_KEY`, `TAVILY_API_KEY`,
`LANGSMITH_API_KEY`, and `LANGCHAIN_API_KEY` are still accepted so existing
local setups keep working while you migrate.

## Adding A New Provider

1. Add a provider settings model in [settings.py](./config/settings.py).
2. Add the provider field to `ProviderSettings`.
3. Add a constructor branch in [providers.py](./config/providers.py) for chat and, if supported, embeddings.
4. Add canonical `.env.example` entries for the new provider.
5. Add config tests for successful construction and missing-credential failures.

Use the existing `openai`, `openrouter`, and `azure_openai` branches as the
reference structure.
