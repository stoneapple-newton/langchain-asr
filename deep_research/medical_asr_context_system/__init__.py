"""
Medical ASR Context System
==========================
A persistent, self-improving context pipeline that bridges Whisper's fixed
vocabulary with domain-specific medical terminology across multiple languages.

Workflow
--------
1. **Retrieve** — query the Chroma term store for the session's specialty/language
2. **Inject** — format retrieved terms as a Whisper ``initial_prompt``
3. **Transcribe** — run Whisper with the injected prompt
4. **Extract** — GPT-4.1 agents extract new medical terms from the transcript
5. **Store** — upsert extracted terms back into Chroma (feedback loop)
6. **Annotate** — tag transcript segments with matched medical terms

Quick start
-----------
    from deep_research.medical_asr_context_system import run_pipeline

    result = run_pipeline(
        audio_path="consultation.wav",
        transcript_path="output/consultation.json",
        language="es",
        specialty="cardiology",
    )
    print(result["stage_log"])

Components
----------
- ``context_store.store.TermContextStore`` — Chroma-backed term store
- ``context_store.retriever.build_whisper_prompt`` — prompt formatter
- ``agents.extraction`` — GPT-4.1 extraction nodes (fan-out)
- ``agents.update`` — validation + store update nodes
- ``graph.pipeline`` — full LangGraph StateGraph + ``run_pipeline()``
"""

from .graph.pipeline import build_pipeline, run_pipeline
from .context_store.store import TermContextStore
from .context_store.retriever import build_whisper_prompt, format_terms_for_llm_hint

__all__ = [
    "run_pipeline",
    "build_pipeline",
    "TermContextStore",
    "build_whisper_prompt",
    "format_terms_for_llm_hint",
]
