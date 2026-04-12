from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"


class BaseConfigModel(BaseModel):
    model_config = {"extra": "ignore"}


class ChatProfileSettings(BaseConfigModel):
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class EmbeddingProfileSettings(BaseConfigModel):
    provider: str | None = None
    model: str | None = None
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class ChatSettings(BaseConfigModel):
    default_profile: str = "default"
    profiles: dict[str, ChatProfileSettings] = Field(default_factory=dict)


class EmbeddingSettings(BaseConfigModel):
    default_profile: str = "default"
    profiles: dict[str, EmbeddingProfileSettings] = Field(default_factory=dict)


class OllamaProviderSettings(BaseConfigModel):
    base_url: str = "http://localhost:11434"


class OpenAIProviderSettings(BaseConfigModel):
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None


class OpenRouterProviderSettings(BaseConfigModel):
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    site_url: str | None = None
    site_name: str | None = None


class AzureOpenAIProviderSettings(BaseConfigModel):
    api_key: str | None = None
    endpoint: str | None = None
    api_version: str = "2024-10-21"


class ProviderSettings(BaseConfigModel):
    ollama: OllamaProviderSettings = Field(default_factory=OllamaProviderSettings)
    openai: OpenAIProviderSettings = Field(default_factory=OpenAIProviderSettings)
    openrouter: OpenRouterProviderSettings = Field(default_factory=OpenRouterProviderSettings)
    azure_openai: AzureOpenAIProviderSettings = Field(default_factory=AzureOpenAIProviderSettings)


class SearchSettings(BaseConfigModel):
    tavily_api_key: str | None = None

    @property
    def has_tavily_api_key(self) -> bool:
        return bool(self.tavily_api_key)


class TracingSettings(BaseConfigModel):
    enabled: bool | None = None
    api_key: str | None = None
    project: str | None = None
    endpoint: str | None = None

    def to_langsmith_env(
        self,
        *,
        default_enabled: bool | None = None,
        default_project: str | None = None,
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        enabled = self.enabled if self.enabled is not None else default_enabled
        project = self.project or default_project

        if enabled is not None:
            env["LANGSMITH_TRACING"] = "true" if enabled else "false"
        if self.api_key:
            env["LANGSMITH_API_KEY"] = self.api_key
        if project:
            env["LANGSMITH_PROJECT"] = project
        if self.endpoint:
            env["LANGSMITH_ENDPOINT"] = self.endpoint
        return env


def _default_chat_profiles() -> dict[str, ChatProfileSettings]:
    return {
        "default": ChatProfileSettings(
            provider="ollama",
            model="gemma4:e2b",
        ),
        "asr_v2": ChatProfileSettings(
            provider="ollama",
            model="gemma4:e4b",
        ),
    }


def _default_embedding_profiles() -> dict[str, EmbeddingProfileSettings]:
    return {
        "default": EmbeddingProfileSettings(
            provider="ollama",
            model="nomic-embed-text",
        ),
    }


def _merge_profile_maps[T: BaseConfigModel](
    defaults: dict[str, T],
    overrides: dict[str, T],
    model_cls: type[T],
) -> dict[str, T]:
    merged = {name: value.model_copy(deep=True) for name, value in defaults.items()}
    for name, override in overrides.items():
        base = merged.get(name, model_cls())
        merged[name] = _merge_models(base, override)
    return merged


def _merge_models[T: BaseConfigModel](base: T, override: T) -> T:
    data = base.model_dump()
    override_data = override.model_dump(exclude_none=True)

    base_extra = data.get("extra_kwargs", {})
    override_extra = override_data.pop("extra_kwargs", {})
    if base_extra or override_extra:
        data["extra_kwargs"] = {**base_extra, **override_extra}

    data.update(override_data)
    return type(base)(**data)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    chat: ChatSettings = Field(default_factory=ChatSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)

    legacy_ollama_model: str | None = Field(default=None, alias="OLLAMA_MODEL", exclude=True)
    legacy_ollama_embedding_model: str | None = Field(
        default=None,
        alias="OLLAMA_EMBEDDING_MODEL",
        exclude=True,
    )
    legacy_ollama_base_url: str | None = Field(default=None, alias="OLLAMA_BASE_URL", exclude=True)
    legacy_openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY", exclude=True)
    legacy_openrouter_api_key: str | None = Field(
        default=None,
        alias="OPENROUTER_API_KEY",
        exclude=True,
    )
    legacy_azure_openai_api_key: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_API_KEY",
        exclude=True,
    )
    legacy_azure_openai_endpoint: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_ENDPOINT",
        exclude=True,
    )
    legacy_tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY", exclude=True)
    legacy_langsmith_api_key: str | None = Field(
        default=None,
        alias="LANGSMITH_API_KEY",
        exclude=True,
    )
    legacy_langsmith_project: str | None = Field(
        default=None,
        alias="LANGSMITH_PROJECT",
        exclude=True,
    )
    legacy_langsmith_tracing: bool | None = Field(
        default=None,
        alias="LANGSMITH_TRACING",
        exclude=True,
    )
    legacy_langsmith_endpoint: str | None = Field(
        default=None,
        alias="LANGSMITH_ENDPOINT",
        exclude=True,
    )
    legacy_langchain_api_key: str | None = Field(
        default=None,
        alias="LANGCHAIN_API_KEY",
        exclude=True,
    )
    legacy_langchain_project: str | None = Field(
        default=None,
        alias="LANGCHAIN_PROJECT",
        exclude=True,
    )
    legacy_langchain_tracing_v2: bool | None = Field(
        default=None,
        alias="LANGCHAIN_TRACING_V2",
        exclude=True,
    )

    @model_validator(mode="after")
    def apply_defaults_and_legacy_aliases(self) -> AppSettings:
        self.chat.profiles = _merge_profile_maps(
            _default_chat_profiles(),
            self.chat.profiles,
            ChatProfileSettings,
        )
        self.embeddings.profiles = _merge_profile_maps(
            _default_embedding_profiles(),
            self.embeddings.profiles,
            EmbeddingProfileSettings,
        )

        if self.legacy_ollama_model:
            self.chat.profiles["default"].model = self.legacy_ollama_model
        if self.legacy_ollama_embedding_model:
            self.embeddings.profiles["default"].model = self.legacy_ollama_embedding_model
        if self.legacy_ollama_base_url and not self.providers.ollama.base_url:
            self.providers.ollama.base_url = self.legacy_ollama_base_url
        elif self.legacy_ollama_base_url:
            self.providers.ollama.base_url = self.legacy_ollama_base_url

        if self.legacy_openai_api_key and not self.providers.openai.api_key:
            self.providers.openai.api_key = self.legacy_openai_api_key
        if self.legacy_openrouter_api_key and not self.providers.openrouter.api_key:
            self.providers.openrouter.api_key = self.legacy_openrouter_api_key
        if self.legacy_azure_openai_api_key and not self.providers.azure_openai.api_key:
            self.providers.azure_openai.api_key = self.legacy_azure_openai_api_key
        if self.legacy_azure_openai_endpoint and not self.providers.azure_openai.endpoint:
            self.providers.azure_openai.endpoint = self.legacy_azure_openai_endpoint

        if self.legacy_tavily_api_key and not self.search.tavily_api_key:
            self.search.tavily_api_key = self.legacy_tavily_api_key

        if self.legacy_langsmith_api_key and not self.tracing.api_key:
            self.tracing.api_key = self.legacy_langsmith_api_key
        if self.legacy_langchain_api_key and not self.tracing.api_key:
            self.tracing.api_key = self.legacy_langchain_api_key

        if self.legacy_langsmith_project and not self.tracing.project:
            self.tracing.project = self.legacy_langsmith_project
        if self.legacy_langchain_project and not self.tracing.project:
            self.tracing.project = self.legacy_langchain_project

        if self.legacy_langsmith_endpoint and not self.tracing.endpoint:
            self.tracing.endpoint = self.legacy_langsmith_endpoint

        if self.legacy_langsmith_tracing is not None and self.tracing.enabled is None:
            self.tracing.enabled = self.legacy_langsmith_tracing
        if self.legacy_langchain_tracing_v2 is not None and self.tracing.enabled is None:
            self.tracing.enabled = self.legacy_langchain_tracing_v2

        return self

    def resolve_chat_profile_name(self, profile: str = "default") -> str:
        return self.chat.default_profile if profile == "default" else profile

    def resolve_embedding_profile_name(self, profile: str = "default") -> str:
        return self.embeddings.default_profile if profile == "default" else profile

    def get_chat_profile(self, profile: str = "default") -> ChatProfileSettings:
        profile_name = self.resolve_chat_profile_name(profile)
        try:
            return self.chat.profiles[profile_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.chat.profiles))
            raise KeyError(
                f"Unknown chat profile '{profile_name}'. Available profiles: {available}"
            ) from exc

    def get_embedding_profile(self, profile: str = "default") -> EmbeddingProfileSettings:
        profile_name = self.resolve_embedding_profile_name(profile)
        try:
            return self.embeddings.profiles[profile_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.embeddings.profiles))
            raise KeyError(
                f"Unknown embedding profile '{profile_name}'. Available profiles: {available}"
            ) from exc


def _clear_cached_settings() -> None:
    get_settings.cache_clear()


from functools import lru_cache


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
