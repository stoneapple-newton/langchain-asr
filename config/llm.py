from __future__ import annotations

from typing import Any

from .providers import build_chat_model, build_embeddings
from .settings import get_settings


def create_chat_model(
    profile: str = "default",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra_kwargs: dict[str, Any] | None = None,
):
    settings = get_settings()
    chat_profile = settings.get_chat_profile(profile)
    return build_chat_model(
        settings,
        chat_profile,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_kwargs=extra_kwargs,
    )


def create_embeddings(
    profile: str = "default",
    *,
    model: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
):
    settings = get_settings()
    embedding_profile = settings.get_embedding_profile(profile)
    return build_embeddings(
        settings,
        embedding_profile,
        model=model,
        extra_kwargs=extra_kwargs,
    )
