from __future__ import annotations

import importlib.util
import io
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config.llm as llm_module
import config.providers as providers_module
from config.providers import ProviderConfigError
from config.settings import AppSettings

ROOT = Path(__file__).resolve().parents[1]


def test_app_settings_defaults_with_sparse_env(monkeypatch):
    for key in [
        "CHAT__DEFAULT_PROFILE",
        "CHAT__PROFILES__DEFAULT__PROVIDER",
        "CHAT__PROFILES__DEFAULT__MODEL",
        "EMBEDDINGS__DEFAULT_PROFILE",
        "PROVIDERS__OLLAMA__BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_EMBEDDING_MODEL",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.chat.default_profile == "default"
    assert settings.chat.profiles["default"].provider == "ollama"
    assert settings.chat.profiles["default"].model == "gemma4:e2b"
    assert settings.chat.profiles["asr_v2"].model == "gemma4:e4b"
    assert settings.embeddings.profiles["default"].model == "nomic-embed-text"
    assert settings.providers.ollama.base_url == "http://localhost:11434"


def test_app_settings_parse_nested_profiles(monkeypatch):
    monkeypatch.setenv("CHAT__DEFAULT_PROFILE", "asr_v2")
    monkeypatch.setenv("CHAT__PROFILES__ASR_V2__PROVIDER", "openai")
    monkeypatch.setenv("CHAT__PROFILES__ASR_V2__MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("PROVIDERS__OPENAI__API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDINGS__PROFILES__DEFAULT__PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS__PROFILES__DEFAULT__MODEL", "text-embedding-3-small")

    settings = AppSettings(_env_file=None)

    assert settings.chat.default_profile == "asr_v2"
    assert settings.get_chat_profile().provider == "openai"
    assert settings.get_chat_profile().model == "gpt-4.1-mini"
    assert settings.chat.profiles["default"].model == "gemma4:e2b"
    assert settings.embeddings.profiles["default"].provider == "openai"
    assert settings.embeddings.profiles["default"].model == "text-embedding-3-small"


def test_langsmith_legacy_aliases(monkeypatch):
    for key in [
        "TRACING__API_KEY",
        "TRACING__PROJECT",
        "TRACING__ENABLED",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGCHAIN_API_KEY", "legacy-key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "legacy-project")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    settings = AppSettings(_env_file=None)

    assert settings.tracing.api_key == "legacy-key"
    assert settings.tracing.project == "legacy-project"
    assert settings.tracing.enabled is True


def _fake_load_class(_module_name: str, class_name: str, _package_name: str):
    def factory(**kwargs):
        return {"class_name": class_name, "kwargs": kwargs}

    return factory


@pytest.mark.parametrize(
    ("provider", "model_name", "provider_config", "expected_class", "expected_keys"),
    [
        (
            "ollama",
            "gemma4:e2b",
            {"ollama": {"base_url": "http://localhost:11434"}},
            "ChatOllama",
            {"base_url": "http://localhost:11434", "num_predict": 321},
        ),
        (
            "openai",
            "gpt-4.1-mini",
            {"openai": {"api_key": "sk-openai"}},
            "ChatOpenAI",
            {"api_key": "sk-openai", "max_tokens": 321},
        ),
        (
            "openrouter",
            "openai/gpt-5-mini",
            {
                "openrouter": {
                    "api_key": "sk-openrouter",
                    "site_url": "https://example.com",
                    "site_name": "Test App",
                }
            },
            "ChatOpenRouter",
            {"api_key": "sk-openrouter", "max_tokens": 321},
        ),
        (
            "azure_openai",
            "azure-chat-deployment",
            {
                "azure_openai": {
                    "api_key": "azure-key",
                    "endpoint": "https://example.openai.azure.com/",
                    "api_version": "2024-10-21",
                }
            },
            "AzureChatOpenAI",
            {
                "api_key": "azure-key",
                "azure_endpoint": "https://example.openai.azure.com/",
                "azure_deployment": "azure-chat-deployment",
                "api_version": "2024-10-21",
                "max_tokens": 321,
            },
        ),
    ],
)
def test_create_chat_model_uses_expected_factory(
    monkeypatch,
    provider,
    model_name,
    provider_config,
    expected_class,
    expected_keys,
):
    settings = AppSettings(
        _env_file=None,
        chat={"profiles": {"default": {"provider": provider, "model": model_name}}},
        providers=provider_config,
    )

    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(providers_module, "_load_class", _fake_load_class)

    result = llm_module.create_chat_model(temperature=0.2, max_tokens=321)

    assert result["class_name"] == expected_class
    assert (
        result["kwargs"].get("model") == model_name
        or result["kwargs"].get("azure_deployment") == model_name
    )
    for key, value in expected_keys.items():
        assert result["kwargs"][key] == value
    if provider == "openrouter":
        assert result["kwargs"]["default_headers"] == {
            "HTTP-Referer": "https://example.com",
            "X-OpenRouter-Title": "Test App",
        }


@pytest.mark.parametrize(
    ("provider", "model_name", "provider_config", "expected_class"),
    [
        (
            "ollama",
            "nomic-embed-text",
            {"ollama": {"base_url": "http://localhost:11434"}},
            "OllamaEmbeddings",
        ),
        (
            "openai",
            "text-embedding-3-small",
            {"openai": {"api_key": "sk-openai"}},
            "OpenAIEmbeddings",
        ),
        (
            "azure_openai",
            "azure-embedding-deployment",
            {
                "azure_openai": {
                    "api_key": "azure-key",
                    "endpoint": "https://example.openai.azure.com/",
                }
            },
            "AzureOpenAIEmbeddings",
        ),
    ],
)
def test_create_embeddings_uses_expected_factory(
    monkeypatch,
    provider,
    model_name,
    provider_config,
    expected_class,
):
    settings = AppSettings(
        _env_file=None,
        embeddings={"profiles": {"default": {"provider": provider, "model": model_name}}},
        providers=provider_config,
    )

    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(providers_module, "_load_class", _fake_load_class)

    result = llm_module.create_embeddings()

    assert result["class_name"] == expected_class
    if provider == "azure_openai":
        assert result["kwargs"]["azure_deployment"] == model_name
        assert result["kwargs"]["azure_endpoint"] == "https://example.openai.azure.com/"
    else:
        assert result["kwargs"]["model"] == model_name


@pytest.mark.parametrize(
    ("provider", "providers"),
    [
        ("openai", {"openai": {}}),
        ("openrouter", {"openrouter": {}}),
        ("azure_openai", {"azure_openai": {"endpoint": "https://example.openai.azure.com/"}}),
    ],
)
def test_missing_credentials_raise_for_active_chat_provider(monkeypatch, provider, providers):
    for key in [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "PROVIDERS__OPENAI__API_KEY",
        "PROVIDERS__OPENROUTER__API_KEY",
        "PROVIDERS__AZURE_OPENAI__API_KEY",
        "PROVIDERS__AZURE_OPENAI__ENDPOINT",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = AppSettings(
        _env_file=None,
        chat={"profiles": {"default": {"provider": provider, "model": "test-model"}}},
        providers=providers,
    )

    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(providers_module, "_load_class", _fake_load_class)

    with pytest.raises(ProviderConfigError):
        llm_module.create_chat_model()


def test_unsupported_openrouter_embeddings_raise(monkeypatch):
    settings = AppSettings(
        _env_file=None,
        embeddings={"profiles": {"default": {"provider": "openrouter", "model": "embed-model"}}},
        providers={"openrouter": {"api_key": "sk-openrouter"}},
    )

    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(providers_module, "_load_class", _fake_load_class)

    with pytest.raises(ProviderConfigError, match="Embeddings are not supported"):
        llm_module.create_embeddings()


def _load_module(module_path: Path, mocked_modules: dict[str, object]):
    module_name = f"_test_{module_path.stem}_{abs(hash(module_path))}"
    original_modules = {name: sys.modules.get(name) for name in mocked_modules}
    try:
        sys.modules.update(mocked_modules)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        with patch("sys.stdout", new=io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


def test_smoke_import_deep_research_agent_script():
    fake_config = types.ModuleType("config")

    class FakeResponse:
        content = "stub"
        response_metadata = {}

    class FakeLLM:
        def invoke(self, *_args, **_kwargs):
            return FakeResponse()

        def stream(self, *_args, **_kwargs):
            return iter(())

    fake_config.create_chat_model = lambda *args, **kwargs: FakeLLM()
    fake_config.create_embeddings = lambda *args, **kwargs: MagicMock()
    fake_config.get_settings = lambda: MagicMock()

    module = _load_module(
        ROOT / "deep_research" / "deep_research_agent" / "stage_01_basics" / "01_hello_langchain.py",
        {"config": fake_config},
    )

    assert hasattr(module, "llm")


def test_smoke_import_asr_script():
    spec = importlib.util.spec_from_file_location(
        "_asr_test_conftest",
        ROOT / "deep_research" / "asr" / "tests" / "conftest.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loaded = module.load_asr_module("stage_01_basics/04_llm_diagnostics.py")
    assert hasattr(loaded, "llm")


def test_smoke_import_asr_v2_script():
    fake_config = types.ModuleType("config")
    fake_config.create_chat_model = lambda *args, **kwargs: MagicMock()
    fake_config.create_embeddings = lambda *args, **kwargs: MagicMock()
    fake_config.get_settings = lambda: MagicMock()

    fake_output_parsers = MagicMock()
    fake_prompts = MagicMock()

    @dataclass
    class TranscriptSegment:
        segment_id: str
        start: float
        end: float
        text: str
        speaker: str | None = None
        words: list[dict] = field(default_factory=list)
        metadata: dict = field(default_factory=dict)

    @dataclass
    class TranscriptDocument:
        segments: list[TranscriptSegment]

    fake_shared = types.ModuleType("shared.transcript_utils")
    fake_shared.TranscriptSegment = TranscriptSegment
    fake_shared.chunk_segments = lambda doc, max_chars=420: [doc.segments]
    fake_shared.improve_readability = lambda doc: doc
    fake_shared.load_transcript = lambda _path: TranscriptDocument(
        segments=[
            TranscriptSegment(
                segment_id="seg-1",
                start=0.0,
                end=1.0,
                text="hello world",
                speaker="SPEAKER_00",
            )
        ]
    )
    fake_shared.rebuild_document_from_segments = lambda _base, segments: TranscriptDocument(segments=segments)
    fake_shared.repair_diarization = lambda doc: doc

    module = _load_module(
        ROOT / "deep_research" / "asr-v2" / "stage_02_tools" / "02_llm_readability_editor.py",
        {
            "config": fake_config,
            "langchain_core": MagicMock(),
            "langchain_core.output_parsers": fake_output_parsers,
            "langchain_core.prompts": fake_prompts,
            "shared": types.ModuleType("shared"),
            "shared.transcript_utils": fake_shared,
        },
    )

    assert hasattr(module, "llm")
