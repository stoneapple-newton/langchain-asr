"""
Retranscription helpers with faster-whisper, OpenAI audio, and text fallback.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import create_chat_model


def transcribe_with_faster_whisper(
    wav_path: str | Path,
    *,
    language: str = "en",
    model_size: str = "base",
    beam_size: int = 5,
) -> tuple[str, float]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError("faster-whisper is not installed.") from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    pieces: list[str] = []
    scores: list[float] = []
    for segment in segments:
        pieces.append(segment.text.strip())
        for word in segment.words or []:
            scores.append(word.probability)
    confidence = round(sum(scores) / len(scores), 3) if scores else 0.8
    return " ".join(item for item in pieces if item).strip(), confidence


def transcribe_with_openai_audio(
    wav_path: str | Path,
    *,
    context_text: str = "",
    language_hint: str = "en",
    specialty_hint: str = "",
) -> tuple[str, float]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is not installed.") from exc

    audio_bytes = Path(wav_path).read_bytes()
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    system_prompt = (
        "You are a medical transcription specialist. "
        "Transcribe the audio verbatim and prefer clinically plausible terminology. "
        "Return only the transcript text."
    )
    if specialty_hint:
        system_prompt += f" Specialty hints: {specialty_hint}."
    user_text = f"Language hint: {language_hint}."
    if context_text:
        user_text += f"\nSurrounding transcript context:\n{context_text}"

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "wav"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        max_tokens=256,
    )
    content = response.choices[0].message.content or ""
    return content.strip().strip('"').strip("'"), 0.85


def correct_with_text_llm(
    *,
    original_text: str,
    context_text: str,
    issue_reason: str,
    medical_context: str = "",
    llm=None,
) -> tuple[str, float]:
    model = llm or create_chat_model("asr_v2", temperature=0, max_tokens=256)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert ASR transcript editor. "
                "Fix transcription errors using the surrounding context. "
                "Prefer clinically plausible medical terms when hints are present. "
                "Return only the corrected segment text.",
            ),
            (
                "human",
                "Issue: {issue_reason}\n\n"
                "Medical hints: {medical_context}\n\n"
                "Context:\n{context_text}\n\n"
                "Original segment:\n{original_text}",
            ),
        ]
    )
    corrected = (prompt | model | StrOutputParser()).invoke(
        {
            "issue_reason": issue_reason,
            "medical_context": medical_context or "(none)",
            "context_text": context_text,
            "original_text": original_text,
        }
    )
    return corrected.strip(), 0.7


def retranscribe_segment(
    wav_path: str | Path,
    *,
    method: str,
    context_text: str,
    original_text: str,
    issue_reason: str,
    language: str = "en",
    specialty_hint: str = "",
    llm=None,
) -> tuple[str, float, str]:
    attempts: list[tuple[str, callable]] = []
    if method == "faster_whisper":
        attempts = [("faster_whisper", transcribe_with_faster_whisper)]
    elif method == "multimodal_llm":
        attempts = [("multimodal_llm", transcribe_with_openai_audio)]
    else:
        attempts = [
            ("faster_whisper", transcribe_with_faster_whisper),
            ("multimodal_llm", transcribe_with_openai_audio),
        ]

    for name, func in attempts:
        try:
            if name == "faster_whisper":
                text, confidence = func(wav_path, language=language)
            else:
                text, confidence = func(
                    wav_path,
                    context_text=context_text,
                    language_hint=language,
                    specialty_hint=specialty_hint,
                )
            if text.strip():
                return text.strip(), confidence, name
        except Exception:
            continue

    corrected, confidence = correct_with_text_llm(
        original_text=original_text,
        context_text=context_text,
        issue_reason=issue_reason,
        medical_context=specialty_hint,
        llm=llm,
    )
    return corrected, confidence, "llm_text"
