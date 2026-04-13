from __future__ import annotations

from importlib import import_module
from typing import Any

from .settings import (
    AppSettings,
    ChatProfileSettings,
    EmbeddingProfileSettings,
)


class ProviderConfigError(RuntimeError):
    """Raised when a provider client cannot be configured from settings."""


def _load_class(module_name: str, class_name: str, package_name: str):
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ProviderConfigError(
            f"Provider support requires '{package_name}'. Install dependencies and run uv sync."
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ProviderConfigError(
            f"Could not find '{class_name}' in '{module_name}'."
        ) from exc


def _require(value: Any, *, message: str) -> Any:
    if value in (None, ""):
        raise ProviderConfigError(message)
    return value


def _resolve_chat_kwargs(
    profile: ChatProfileSettings,
    *,
    temperature: float | None,
    max_tokens: int | None,
    extra_kwargs: dict[str, Any] | None,
    max_tokens_key: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if profile.temperature is not None:
        kwargs["temperature"] = profile.temperature
    if profile.max_tokens is not None:
        kwargs[max_tokens_key] = profile.max_tokens

    kwargs.update(profile.extra_kwargs)

    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs[max_tokens_key] = max_tokens
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def _resolve_embedding_kwargs(
    profile: EmbeddingProfileSettings,
    *,
    extra_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs = dict(profile.extra_kwargs)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return kwargs


def build_chat_model(
    settings: AppSettings,
    profile: ChatProfileSettings,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra_kwargs: dict[str, Any] | None = None,
):
    provider = _require(
        profile.provider,
        message="Chat profile is missing a provider.",
    )
    model_name = _require(
        profile.model,
        message="Chat profile is missing a model name.",
    )

    if provider == "ollama":
        ChatOllama = _load_class("langchain_ollama", "ChatOllama", "langchain-ollama")
        kwargs = _resolve_chat_kwargs(
            profile,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            max_tokens_key="num_predict",
        )
        return ChatOllama(
            model=model_name,
            base_url=settings.providers.ollama.base_url,
            **kwargs,
        )

    if provider == "openai":
        ChatOpenAI = _load_class("langchain_openai", "ChatOpenAI", "langchain-openai")
        kwargs = _resolve_chat_kwargs(
            profile,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            max_tokens_key="max_tokens",
        )
        return ChatOpenAI(
            model=model_name,
            api_key=_require(
                settings.providers.openai.api_key,
                message="OpenAI chat profiles require PROVIDERS__OPENAI__API_KEY or OPENAI_API_KEY.",
            ),
            base_url=settings.providers.openai.base_url,
            organization=settings.providers.openai.organization,
            **kwargs,
        )

    if provider == "openrouter":
        ChatOpenRouter = _load_class(
            "langchain_openrouter",
            "ChatOpenRouter",
            "langchain-openrouter",
        )
        kwargs = _resolve_chat_kwargs(
            profile,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            max_tokens_key="max_tokens",
        )
        headers: dict[str, str] = {}
        if settings.providers.openrouter.site_url:
            headers["HTTP-Referer"] = settings.providers.openrouter.site_url
        if settings.providers.openrouter.site_name:
            headers["X-OpenRouter-Title"] = settings.providers.openrouter.site_name
        if headers:
            existing_headers = kwargs.get("default_headers")
            if isinstance(existing_headers, dict):
                kwargs["default_headers"] = {**existing_headers, **headers}
            else:
                kwargs["default_headers"] = headers
        return ChatOpenRouter(
            model=model_name,
            api_key=_require(
                settings.providers.openrouter.api_key,
                message=(
                    "OpenRouter chat profiles require "
                    "PROVIDERS__OPENROUTER__API_KEY or OPENROUTER_API_KEY."
                ),
            ),
            base_url=settings.providers.openrouter.base_url,
            **kwargs,
        )

    if provider == "azure_openai":
        AzureChatOpenAI = _load_class(
            "langchain_openai",
            "AzureChatOpenAI",
            "langchain-openai",
        )
        kwargs = _resolve_chat_kwargs(
            profile,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_kwargs=extra_kwargs,
            max_tokens_key="max_tokens",
        )
        return AzureChatOpenAI(
            azure_deployment=model_name,
            azure_endpoint=_require(
                settings.providers.azure_openai.endpoint,
                message=(
                    "Azure OpenAI chat profiles require "
                    "PROVIDERS__AZURE_OPENAI__ENDPOINT or AZURE_OPENAI_ENDPOINT."
                ),
            ),
            api_key=_require(
                settings.providers.azure_openai.api_key,
                message=(
                    "Azure OpenAI chat profiles require "
                    "PROVIDERS__AZURE_OPENAI__API_KEY or AZURE_OPENAI_API_KEY."
                ),
            ),
            api_version=settings.providers.azure_openai.api_version,
            **kwargs,
        )

    raise ProviderConfigError(
        f"Unsupported chat provider '{provider}'. Add it to config/providers.py."
    )


def build_embeddings(
    settings: AppSettings,
    profile: EmbeddingProfileSettings,
    *,
    model: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
):
    provider = _require(
        profile.provider,
        message="Embedding profile is missing a provider.",
    )
    model_name = model or _require(
        profile.model,
        message="Embedding profile is missing a model name.",
    )
    kwargs = _resolve_embedding_kwargs(profile, extra_kwargs=extra_kwargs)

    if provider == "ollama":
        OllamaEmbeddings = _load_class(
            "langchain_ollama",
            "OllamaEmbeddings",
            "langchain-ollama",
        )
        return OllamaEmbeddings(
            model=model_name,
            base_url=settings.providers.ollama.base_url,
            **kwargs,
        )

    if provider == "openai":
        OpenAIEmbeddings = _load_class(
            "langchain_openai",
            "OpenAIEmbeddings",
            "langchain-openai",
        )
        return OpenAIEmbeddings(
            model=model_name,
            api_key=_require(
                settings.providers.openai.api_key,
                message=(
                    "OpenAI embedding profiles require "
                    "PROVIDERS__OPENAI__API_KEY or OPENAI_API_KEY."
                ),
            ),
            base_url=settings.providers.openai.base_url,
            organization=settings.providers.openai.organization,
            **kwargs,
        )

    if provider == "azure_openai":
        AzureOpenAIEmbeddings = _load_class(
            "langchain_openai",
            "AzureOpenAIEmbeddings",
            "langchain-openai",
        )
        return AzureOpenAIEmbeddings(
            model=model_name,
            azure_deployment=model_name,
            azure_endpoint=_require(
                settings.providers.azure_openai.endpoint,
                message=(
                    "Azure OpenAI embedding profiles require "
                    "PROVIDERS__AZURE_OPENAI__ENDPOINT or AZURE_OPENAI_ENDPOINT."
                ),
            ),
            api_key=_require(
                settings.providers.azure_openai.api_key,
                message=(
                    "Azure OpenAI embedding profiles require "
                    "PROVIDERS__AZURE_OPENAI__API_KEY or AZURE_OPENAI_API_KEY."
                ),
            ),
            api_version=settings.providers.azure_openai.api_version,
            **kwargs,
        )

    if provider == "openrouter":
        raise ProviderConfigError(
            "Embeddings are not supported for provider 'openrouter' in this config layer."
        )

    raise ProviderConfigError(
        f"Unsupported embedding provider '{provider}'. Add it to config/providers.py."
    )
