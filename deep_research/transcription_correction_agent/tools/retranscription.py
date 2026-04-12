"""
Retranscription Tools
=====================
Two strategies for re-transcribing a short audio segment:

1. faster_whisper  — local, fast, offline. Requires: pip install faster-whisper
2. multimodal_llm  — uses OpenAI gpt-4o-audio-preview (or any model with
                     audio input). Requires OPENAI_API_KEY.

If faster-whisper is not installed, the function raises ImportError so the
graph can route to the multimodal fallback.

If neither is available, a text-only LLM correction is applied as a last resort.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 1. faster-whisper
# ---------------------------------------------------------------------------

def transcribe_with_faster_whisper(
    wav_path: str,
    language: str = "en",
    model_size: str = "base",
    beam_size: int = 5,
    condition_on_previous_text: bool = False,
) -> tuple[str, float]:
    """
    Re-transcribe *wav_path* with faster-whisper.

    Returns:
        (transcribed_text, avg_confidence)

    Raises:
        ImportError: If faster-whisper is not installed.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. "
            "Run: pip install faster-whisper"
        ) from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
    )

    texts: list[str] = []
    confidences: list[float] = []

    for seg in segments:
        texts.append(seg.text.strip())
        for word in seg.words or []:
            confidences.append(word.probability)

    full_text = " ".join(texts).strip()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.8

    return full_text, avg_conf


# ---------------------------------------------------------------------------
# 2. multimodal LLM (OpenAI gpt-4o-audio-preview)
# ---------------------------------------------------------------------------

def transcribe_with_multimodal_llm(
    wav_path: str,
    context_text: str = "",
    language_hint: str = "en",
    specialty_hint: str = "",
) -> tuple[str, float]:
    """
    Re-transcribe *wav_path* by sending the audio to a multimodal LLM.

    The audio is base64-encoded and sent as an audio_url content block.
    Works with OpenAI gpt-4o-audio-preview.

    Returns:
        (transcribed_text, confidence_estimate)
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    wav_bytes = Path(wav_path).read_bytes()
    b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

    specialty_note = (
        f" The speaker is likely discussing {specialty_hint}." if specialty_hint else ""
    )
    context_note = (
        f"\n\nSurrounding context:\n{context_text}" if context_text else ""
    )

    system_prompt = (
        "You are a medical transcription specialist. "
        "Listen carefully to the audio and transcribe it verbatim. "
        "Pay special attention to medical terminology, drug names, and clinical language."
        f"{specialty_note}"
        " Return ONLY the transcribed text, nothing else."
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Please transcribe this audio segment accurately."
                            f"{context_note}"
                        ),
                    },
                ],
            },
        ],
        max_tokens=256,
    )

    transcribed = response.choices[0].message.content or ""
    transcribed = transcribed.strip().strip('"').strip("'")

    # Multimodal LLMs don't return token-level confidence; estimate high
    return transcribed, 0.85


# ---------------------------------------------------------------------------
# 3. Text-only LLM correction (always available as fallback)
# ---------------------------------------------------------------------------

def correct_with_text_llm(
    original_text: str,
    context_text: str,
    issue_reason: str,
    medical_context: str = "",
    llm=None,
) -> tuple[str, float]:
    """
    Use a text-only LLM to correct a segment based on surrounding context.

    This is the fallback when no audio retranscription is possible.

    Args:
        original_text: The garbled/suspect text.
        context_text: Surrounding transcript segments.
        issue_reason: Why this segment was flagged.
        medical_context: Medical glossary hints if available.
        llm: A LangChain chat model (create_chat_model()).

    Returns:
        (corrected_text, confidence_estimate)
    """
    if llm is None:
        # Lazy import to avoid circular dependency
        root = Path(__file__).resolve().parents[3]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from config import create_chat_model  # type: ignore
        llm = create_chat_model(temperature=0, max_tokens=256)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    medical_note = (
        f"\n\nMedical term hints (multilingual): {medical_context}"
        if medical_context
        else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert medical transcription editor. "
            "Fix ASR transcription errors using the surrounding context. "
            "Return ONLY the corrected text — no explanation, no quotes.{medical_note}"
        ),
        (
            "human",
            "Issue: {issue}\n\n"
            "Surrounding context:\n{context}\n\n"
            "Suspect segment to correct:\n{text}"
        ),
    ])

    chain = prompt | llm | StrOutputParser()
    corrected = chain.invoke({
        "medical_note": medical_note,
        "issue": issue_reason,
        "context": context_text,
        "text": original_text,
    }).strip()

    return corrected, 0.70


# ---------------------------------------------------------------------------
# Dispatcher: choose method, fall back automatically
# ---------------------------------------------------------------------------

def retranscribe(
    wav_path: str,
    method: str,
    context_text: str = "",
    language: str = "en",
    specialty_hint: str = "",
    issue_reason: str = "",
    original_text: str = "",
    llm=None,
) -> tuple[str, float, str]:
    """
    Retranscribe a segment using the requested *method*, with automatic fallback.

    Args:
        method: "faster_whisper" | "multimodal_llm" | "auto"
        ... other args passed to the specific implementation

    Returns:
        (corrected_text, confidence, actual_method_used)
    """
    attempts: list[tuple[str, callable]] = []

    if method == "faster_whisper":
        attempts = [("faster_whisper", _try_faster_whisper)]
    elif method == "multimodal_llm":
        attempts = [("multimodal_llm", _try_multimodal)]
    else:  # "auto": try fw first, then multimodal, then text fallback
        attempts = [
            ("faster_whisper", _try_faster_whisper),
            ("multimodal_llm", _try_multimodal),
        ]

    for method_name, fn in attempts:
        try:
            text, conf = fn(
                wav_path=wav_path,
                context_text=context_text,
                language=language,
                specialty_hint=specialty_hint,
            )
            return text, conf, method_name
        except (ImportError, RuntimeError, Exception) as exc:
            print(f"  [retranscribe] {method_name} failed: {exc!r} — trying fallback")

    # Last resort: text-only LLM
    text, conf = correct_with_text_llm(
        original_text=original_text,
        context_text=context_text,
        issue_reason=issue_reason,
        medical_context=specialty_hint,
        llm=llm,
    )
    return text, conf, "llm_text"


def _try_faster_whisper(wav_path, context_text, language, specialty_hint):
    text, conf = transcribe_with_faster_whisper(wav_path, language=language)
    return text, conf


def _try_multimodal(wav_path, context_text, language, specialty_hint):
    text, conf = transcribe_with_multimodal_llm(
        wav_path,
        context_text=context_text,
        language_hint=language,
        specialty_hint=specialty_hint,
    )
    return text, conf
