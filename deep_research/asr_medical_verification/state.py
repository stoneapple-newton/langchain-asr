"""
LangGraph state schema for the ASR medical verification workflow.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict


class AgentState(TypedDict):
    transcript_path: str
    audio_path: str
    glossary_path: str
    retranscription_method: str
    confidence_threshold: float
    transcript_doc: object
    glossary: object
    error_candidates: list[dict]
    medical_candidates: list[dict]
    corrections: Annotated[list[dict], operator.add]
    corrected_doc: object
    stage_log: Annotated[list[str], operator.add]
    output_paths: list[str]


class WorkerState(TypedDict):
    candidate: dict
    audio_path: str
    retranscription_method: str
    confidence_threshold: float
    transcript_doc: object
    glossary: object
