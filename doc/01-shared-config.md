# Shared Config Feature

## Feature Summary

The shared config layer centralizes model provider selection, embedding
provider selection, search keys, and LangSmith tracing settings. It allows
tutorial scripts to call one factory API while switching providers through
`.env`.

## Business Outcome

Developers can run the same examples against Ollama, OpenAI, OpenRouter, or
Azure OpenAI without editing tutorial code. This reduces setup drift and makes
future examples easier to maintain.

## Scope

Feature files:

- `config/settings.py`
- `config/providers.py`
- `config/llm.py`
- `config/structured.py`
- `config/__init__.py`
- `config/README.md`
- `tests/test_config.py`

## Current Architecture

`settings.py` defines Pydantic Settings models for chat profiles, embedding
profiles, providers, search, and tracing. Defaults are merged into user
overrides so sparse `.env` files still produce usable local Ollama profiles.
Legacy flat environment variables are migrated into canonical nested settings.

`providers.py` maps logical provider names to LangChain client constructors.
It lazy-loads provider packages and raises `ProviderConfigError` for missing
dependencies, credentials, or unsupported providers.

`llm.py` exposes the public factories:

- `create_chat_model(profile="default", temperature=None, max_tokens=None,
  extra_kwargs=None)`
- `create_embeddings(profile="default", model=None, extra_kwargs=None)`

`structured.py` exposes `structured_output_chain()`, which tries native
structured output and falls back to prompt, LLM, and `JsonOutputParser`.

## Key APIs and Data Contracts

- `AppSettings`: root settings object.
- `ChatProfileSettings`: provider, model, temperature, max tokens, extra
  provider kwargs.
- `EmbeddingProfileSettings`: provider, model, extra kwargs.
- `ProviderSettings`: Ollama, OpenAI, OpenRouter, Azure OpenAI settings.
- `TracingSettings.to_langsmith_env()`: converts tracing settings into
  LangSmith environment values.
- `ProviderConfigError`: recoverable configuration failure.

## Dependencies

- `pydantic-settings`
- `langchain-ollama`
- `langchain-openai`
- `langchain-openrouter`
- `langchain-core`

Provider packages are lazy-loaded, so a provider branch only fails when used.

## Nonfunctional Requirements

- Provider errors must be actionable.
- Tests must not require real API credentials.
- Factory arguments must override profile defaults.
- Provider-specific token kwargs must be normalized.
- Legacy env aliases should continue to work during migration.

## Acceptance Criteria

- `create_chat_model()` builds the configured profile using the expected
  provider class.
- `create_embeddings()` builds supported embedding providers and rejects
  unsupported ones.
- Missing hosted-provider credentials raise `ProviderConfigError`.
- OpenRouter attribution headers merge with caller-supplied headers.
- Sparse local env configuration defaults to Ollama.

## Risks and Open Questions

- `get_settings()` is cached. Tests use direct `AppSettings(_env_file=None)` or
  monkeypatching, but application code that changes environment variables at
  runtime must clear the cache.
- OpenRouter embeddings are intentionally unsupported.
- The default model names are educational defaults and may not exist in every
  Ollama installation.

## Development Guidance

Add new providers in this order: settings model, provider field, constructor
branch, `.env.example`, config tests. Keep provider-specific compatibility
inside `providers.py`; tutorial code should only use the factory API.
