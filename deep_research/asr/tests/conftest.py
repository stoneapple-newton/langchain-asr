"""
Shared fixtures and import helper for the ASR test suite.

Every ASR script executes module-level code at import time:
  - opens sample_transcript.json
  - creates ChatOllama / OllamaEmbeddings instances
  - prints demo output

load_asr_module() works around this by:
  1. Replacing all LangChain/LangGraph packages with MagicMocks in sys.modules
  2. Patching builtins.open to return FAKE_JSON for any .json read
  3. Suppressing stdout
  4. Loading the script via importlib
  5. Restoring sys.modules in a finally block
"""

import builtins
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ASR_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# FAKE_JSON — used to satisfy every open() at import time.
# 8 segments: all with clean gaps (≥0.5s), all with ≥3 words, no stutters,
# so that demo loops in stage_02 don't hit index errors or mocked LLM chains.
# ---------------------------------------------------------------------------
FAKE_JSON = {
    "meeting_metadata": {"title": "Test Meeting", "date": "2024-01-01"},
    "duration": 80.0,
    "language": "en",
    "segments": [
        # All segments: non-overlapping, 0.5s gap, ≥5 words, proper speaker alternation
        # Gaps all ≥ 0.5s so no suspicious fast-switches (threshold < 0.15s)
        {
            "start": 0.0, "end": 5.0,
            "text": " Alright let us get started today.",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "Alright", "start": 0.0, "end": 0.5, "score": 0.98, "speaker": "SPEAKER_00"},
                {"word": "let",     "start": 0.6, "end": 0.8, "score": 0.97, "speaker": "SPEAKER_00"},
                {"word": "us",      "start": 0.9, "end": 1.0, "score": 0.99, "speaker": "SPEAKER_00"},
                {"word": "get",     "start": 1.1, "end": 1.3, "score": 0.98, "speaker": "SPEAKER_00"},
                {"word": "started", "start": 1.4, "end": 2.0, "score": 0.99, "speaker": "SPEAKER_00"},
                {"word": "today",   "start": 2.1, "end": 2.5, "score": 0.97, "speaker": "SPEAKER_00"},
            ],
        },
        {
            "start": 5.5, "end": 11.0,
            "text": " The main agenda is reviewing the roadmap.",
            "speaker": "SPEAKER_01",
            "words": [
                {"word": "The",       "start": 5.5,  "end": 5.7,  "score": 0.99, "speaker": "SPEAKER_01"},
                {"word": "main",      "start": 5.8,  "end": 6.1,  "score": 0.97, "speaker": "SPEAKER_01"},
                {"word": "agenda",    "start": 6.2,  "end": 6.6,  "score": 0.96, "speaker": "SPEAKER_01"},
                {"word": "is",        "start": 6.7,  "end": 6.85, "score": 0.99, "speaker": "SPEAKER_01"},
                {"word": "reviewing", "start": 6.9,  "end": 7.5,  "score": 0.95, "speaker": "SPEAKER_01"},
                {"word": "the",       "start": 7.6,  "end": 7.75, "score": 0.99, "speaker": "SPEAKER_01"},
                {"word": "roadmap",   "start": 7.8,  "end": 8.4,  "score": 0.94, "speaker": "SPEAKER_01"},
            ],
        },
        {
            "start": 11.5, "end": 17.0,
            "text": " Authentication service has been really flaky.",
            "speaker": "SPEAKER_02",
            "words": [
                {"word": "Authentication", "start": 11.5, "end": 12.3, "score": 0.91, "speaker": "SPEAKER_02"},
                {"word": "service",        "start": 12.4, "end": 12.9, "score": 0.95, "speaker": "SPEAKER_02"},
                {"word": "has",            "start": 13.0, "end": 13.2, "score": 0.99, "speaker": "SPEAKER_02"},
                {"word": "been",           "start": 13.3, "end": 13.5, "score": 0.98, "speaker": "SPEAKER_02"},
                {"word": "really",         "start": 13.6, "end": 13.9, "score": 0.97, "speaker": "SPEAKER_02"},
                {"word": "flaky",          "start": 14.0, "end": 14.5, "score": 0.93, "speaker": "SPEAKER_02"},
            ],
        },
        {
            "start": 17.5, "end": 23.0,
            "text": " The JWT library needs to be updated.",
            "speaker": "SPEAKER_01",
            "words": [
                {"word": "The",     "start": 17.5, "end": 17.7, "score": 0.99, "speaker": "SPEAKER_01"},
                {"word": "JWT",     "start": 17.8, "end": 18.1, "score": 0.88, "speaker": "SPEAKER_01"},
                {"word": "library", "start": 18.2, "end": 18.7, "score": 0.97, "speaker": "SPEAKER_01"},
                {"word": "needs",   "start": 18.8, "end": 19.1, "score": 0.98, "speaker": "SPEAKER_01"},
                {"word": "to",      "start": 19.2, "end": 19.3, "score": 0.99, "speaker": "SPEAKER_01"},
                {"word": "be",      "start": 19.4, "end": 19.5, "score": 0.99, "speaker": "SPEAKER_01"},
                {"word": "updated", "start": 19.6, "end": 20.2, "score": 0.96, "speaker": "SPEAKER_01"},
            ],
        },
        {
            "start": 23.5, "end": 29.0,
            "text": " DevOps needs to approve the node upgrade.",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "DevOps",  "start": 23.5, "end": 24.0, "score": 0.89, "speaker": "SPEAKER_00"},
                {"word": "needs",   "start": 24.1, "end": 24.4, "score": 0.98, "speaker": "SPEAKER_00"},
                {"word": "to",      "start": 24.5, "end": 24.6, "score": 0.99, "speaker": "SPEAKER_00"},
                {"word": "approve", "start": 24.7, "end": 25.2, "score": 0.97, "speaker": "SPEAKER_00"},
                {"word": "the",     "start": 25.3, "end": 25.4, "score": 0.99, "speaker": "SPEAKER_00"},
                {"word": "node",    "start": 25.5, "end": 25.8, "score": 0.96, "speaker": "SPEAKER_00"},
                {"word": "upgrade", "start": 25.9, "end": 26.5, "score": 0.95, "speaker": "SPEAKER_00"},
            ],
        },
        {
            "start": 29.5, "end": 35.0,
            "text": " Carol is handling the dashboard design.",
            "speaker": "SPEAKER_02",
            "words": [
                {"word": "Carol",     "start": 29.5, "end": 29.8, "score": 0.98, "speaker": "SPEAKER_02"},
                {"word": "is",        "start": 29.9, "end": 30.0, "score": 0.99, "speaker": "SPEAKER_02"},
                {"word": "handling",  "start": 30.1, "end": 30.6, "score": 0.97, "speaker": "SPEAKER_02"},
                {"word": "the",       "start": 30.7, "end": 30.8, "score": 0.99, "speaker": "SPEAKER_02"},
                {"word": "dashboard", "start": 30.9, "end": 31.5, "score": 0.96, "speaker": "SPEAKER_02"},
                {"word": "design",    "start": 31.6, "end": 32.1, "score": 0.97, "speaker": "SPEAKER_02"},
            ],
        },
        {
            "start": 35.5, "end": 41.0,
            "text": " WCAG contrast requirements must be satisfied.",
            "speaker": "SPEAKER_02",
            "words": [
                {"word": "WCAG",         "start": 35.5, "end": 35.9, "score": 0.87, "speaker": "SPEAKER_02"},
                {"word": "contrast",     "start": 36.0, "end": 36.5, "score": 0.96, "speaker": "SPEAKER_02"},
                {"word": "requirements", "start": 36.6, "end": 37.4, "score": 0.97, "speaker": "SPEAKER_02"},
                {"word": "must",         "start": 37.5, "end": 37.7, "score": 0.99, "speaker": "SPEAKER_02"},
                {"word": "be",           "start": 37.8, "end": 37.9, "score": 0.99, "speaker": "SPEAKER_02"},
                {"word": "satisfied",    "start": 38.0, "end": 38.7, "score": 0.98, "speaker": "SPEAKER_02"},
            ],
        },
        {
            "start": 41.5, "end": 47.0,
            "text": " The deadline is October the fifteenth.",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "The",       "start": 41.5, "end": 41.7, "score": 0.99, "speaker": "SPEAKER_00"},
                {"word": "deadline",  "start": 41.8, "end": 42.3, "score": 0.98, "speaker": "SPEAKER_00"},
                {"word": "is",        "start": 42.4, "end": 42.5, "score": 0.99, "speaker": "SPEAKER_00"},
                {"word": "October",   "start": 42.6, "end": 43.1, "score": 0.97, "speaker": "SPEAKER_00"},
                {"word": "the",       "start": 43.2, "end": 43.3, "score": 0.99, "speaker": "SPEAKER_00"},
                {"word": "fifteenth", "start": 43.4, "end": 44.1, "score": 0.96, "speaker": "SPEAKER_00"},
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# MINIMAL_SEGMENTS — engineered 5-segment fixture for deterministic assertions.
#
# Pre-computed values (used in tests):
#   total words          = 30
#   avg_confidence       ≈ 0.928  (sum of scores / 30)
#   filler_rate stage_01 = 4/30 = 0.133  (so, uh, um, right  — stage_01 FILLERS includes these)
#   filler_rate stage_04 = 2/30 = 0.067  (uh, um only — stage_04 FILLERS is {"uh","um","you know","i mean","like"})
#   punctuation_score    = 2/5 = 0.4   (segs 0, 4 end with .)
#   capitalisation_score = 2/5 = 0.4   (segs 0, 4 start with capital after strip)
#   diarization_coverage = 1.0  (all labelled)
#   speaker_count        = 3
#   low_confidence_words = 2 (authentication 0.68, JWT 0.71)  → rate = 2/30
#   duplicate_word_rate  = 1/30 (one stutter pair "The the" in seg 4) — depends on impl
#   build_turns          → 4 turns: SPEAKER_00, SPEAKER_01, SPEAKER_02, SPEAKER_01
#   detect_anomalies     → SHORT_TURN(seg3), FAST_SWITCH(seg2 gap=0.02)
# ---------------------------------------------------------------------------
MINIMAL_SEGMENTS = [
    # Seg 0: SPEAKER_00, clean, period, capitalised  (4 words)
    {
        "start": 0.0, "end": 4.5,
        "text": " Alright, let's get started.",
        "speaker": "SPEAKER_00",
        "words": [
            {"word": "Alright", "start": 0.0,  "end": 0.6,  "score": 0.96, "speaker": "SPEAKER_00"},
            {"word": "let's",   "start": 0.65, "end": 1.1,  "score": 0.94, "speaker": "SPEAKER_00"},
            {"word": "get",     "start": 1.15, "end": 1.35, "score": 0.98, "speaker": "SPEAKER_00"},
            {"word": "started", "start": 1.4,  "end": 2.0,  "score": 0.99, "speaker": "SPEAKER_00"},
        ],
    },
    # Seg 1: SPEAKER_00, filler words (so + uh + um), no period, lowercase  (10 words)
    {
        "start": 4.6, "end": 10.0,
        "text": " so uh the main agenda um is reviewing the roadmap",
        "speaker": "SPEAKER_00",
        "words": [
            {"word": "so",        "start": 4.6,  "end": 4.75, "score": 0.97, "speaker": "SPEAKER_00"},
            {"word": "uh",        "start": 4.8,  "end": 4.95, "score": 0.80, "speaker": "SPEAKER_00"},
            {"word": "the",       "start": 5.0,  "end": 5.15, "score": 0.99, "speaker": "SPEAKER_00"},
            {"word": "main",      "start": 5.2,  "end": 5.5,  "score": 0.97, "speaker": "SPEAKER_00"},
            {"word": "agenda",    "start": 5.55, "end": 6.0,  "score": 0.95, "speaker": "SPEAKER_00"},
            {"word": "um",        "start": 6.05, "end": 6.25, "score": 0.82, "speaker": "SPEAKER_00"},
            {"word": "is",        "start": 6.3,  "end": 6.45, "score": 0.99, "speaker": "SPEAKER_00"},
            {"word": "reviewing", "start": 6.5,  "end": 7.1,  "score": 0.95, "speaker": "SPEAKER_00"},
            {"word": "the",       "start": 7.15, "end": 7.25, "score": 0.99, "speaker": "SPEAKER_00"},
            {"word": "roadmap",   "start": 7.3,  "end": 7.85, "score": 0.94, "speaker": "SPEAKER_00"},
        ],
    },
    # Seg 2: SPEAKER_01, fast switch (gap = 0.02 < 0.1), 1 low-conf word  (8 words)
    {
        "start": 10.02, "end": 16.0,
        "text": " yeah the authentication service its been really flaky",
        "speaker": "SPEAKER_01",
        "words": [
            {"word": "yeah",           "start": 10.02, "end": 10.4,  "score": 0.91, "speaker": "SPEAKER_01"},
            {"word": "the",            "start": 10.5,  "end": 10.65, "score": 0.99, "speaker": "SPEAKER_01"},
            {"word": "authentication", "start": 10.7,  "end": 11.6,  "score": 0.68, "speaker": "SPEAKER_01"},
            {"word": "service",        "start": 11.7,  "end": 12.2,  "score": 0.93, "speaker": "SPEAKER_01"},
            {"word": "its",            "start": 12.3,  "end": 12.5,  "score": 0.87, "speaker": "SPEAKER_01"},
            {"word": "been",           "start": 12.55, "end": 12.8,  "score": 0.98, "speaker": "SPEAKER_01"},
            {"word": "really",         "start": 12.85, "end": 13.2,  "score": 0.97, "speaker": "SPEAKER_01"},
            {"word": "flaky",          "start": 13.25, "end": 13.8,  "score": 0.89, "speaker": "SPEAKER_01"},
        ],
    },
    # Seg 3: SPEAKER_02, 2-word backchannel, 1.0s → SHORT_TURN  (2 words)
    {
        "start": 16.2, "end": 17.2,
        "text": " right yeah",
        "speaker": "SPEAKER_02",
        "words": [
            {"word": "right", "start": 16.2, "end": 16.6, "score": 0.95, "speaker": "SPEAKER_02"},
            {"word": "yeah",  "start": 16.7, "end": 17.2, "score": 0.93, "speaker": "SPEAKER_02"},
        ],
    },
    # Seg 4: SPEAKER_01, duplicate stutter ("The the"), 1 low-conf word, period, capitalised  (6 words)
    {
        "start": 17.5, "end": 24.0,
        "text": " The the JWT library needs updating.",
        "speaker": "SPEAKER_01",
        "words": [
            {"word": "The",      "start": 17.5,  "end": 17.7,  "score": 0.97, "speaker": "SPEAKER_01"},
            {"word": "the",      "start": 17.72, "end": 17.9,  "score": 0.92, "speaker": "SPEAKER_01"},
            {"word": "JWT",      "start": 18.0,  "end": 18.4,  "score": 0.71, "speaker": "SPEAKER_01"},
            {"word": "library",  "start": 18.5,  "end": 19.1,  "score": 0.96, "speaker": "SPEAKER_01"},
            {"word": "needs",    "start": 19.2,  "end": 19.6,  "score": 0.98, "speaker": "SPEAKER_01"},
            {"word": "updating", "start": 19.7,  "end": 20.5,  "score": 0.97, "speaker": "SPEAKER_01"},
        ],
    },
]


# ---------------------------------------------------------------------------
# LangChain/LangGraph packages to mock when loading ASR scripts
# ---------------------------------------------------------------------------
MOCK_MODULE_NAMES = [
    "config",
    "langchain_ollama",
    "langchain_core",
    "langchain_core.documents",
    "langchain_core.output_parsers",
    "langchain_core.prompts",
    "langchain_core.runnables",
    "langchain_core.vectorstores",
    "langchain_text_splitters",
    "langgraph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "langgraph.graph",
    "langgraph.types",
    "pydantic",
]


def load_asr_module(rel_path: str):
    """
    Load an ASR script module with all LangChain/LangGraph deps mocked
    and file I/O pointing to FAKE_JSON. Returns the module object.
    """
    abs_path = ASR_ROOT / rel_path
    module_name = f"_asr_{abs_path.parent.name}_{abs_path.stem}"

    _real_open = builtins.open

    def fake_open(file, mode="r", *args, **kwargs):
        if str(file).endswith(".json") and "r" in str(mode):
            return io.StringIO(json.dumps(FAKE_JSON))
        if "w" in str(mode) or "a" in str(mode):
            return io.StringIO()
        return _real_open(file, mode, *args, **kwargs)

    mock_modules = {
        name: MagicMock()
        for name in MOCK_MODULE_NAMES
    }
    # Set sentinel values needed by scripts
    mock_modules["langgraph.graph"].START = "START"
    mock_modules["langgraph.graph"].END = "END"
    fake_llm = MagicMock()
    fake_embeddings = MagicMock()
    fake_settings = types.SimpleNamespace(
        search=types.SimpleNamespace(has_tavily_api_key=False),
        tracing=types.SimpleNamespace(
            api_key=None,
            to_langsmith_env=lambda **_: {},
        ),
        get_chat_profile=lambda *args, **kwargs: types.SimpleNamespace(
            provider="ollama",
            model="test-model",
        ),
        get_embedding_profile=lambda *args, **kwargs: types.SimpleNamespace(
            provider="ollama",
            model="test-embedding-model",
        ),
    )
    mock_modules["config"].create_chat_model = MagicMock(return_value=fake_llm)
    mock_modules["config"].create_embeddings = MagicMock(return_value=fake_embeddings)
    mock_modules["config"].get_settings = MagicMock(return_value=fake_settings)

    # pydantic: BaseModel and Field must be real so dataclasses/models work in pure-python fns.
    # We use the real pydantic instead of mocking it.
    del mock_modules["pydantic"]

    original_modules = {k: sys.modules.get(k) for k in MOCK_MODULE_NAMES if k != "pydantic"}
    try:
        sys.modules.update(mock_modules)
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        mod = importlib.util.module_from_spec(spec)
        with patch("builtins.open", side_effect=fake_open):
            with patch("sys.stdout", new=io.StringIO()):
                spec.loader.exec_module(mod)
    finally:
        for name, orig in original_modules.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
    return mod


# ---------------------------------------------------------------------------
# Module-scope fixtures — each ASR script loaded once per test file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage1_parse_mod():
    return load_asr_module("stage_01_basics/01_load_and_parse.py")


@pytest.fixture(scope="module")
def stage1_metrics_mod():
    return load_asr_module("stage_01_basics/02_quality_metrics.py")


@pytest.fixture(scope="module")
def stage2_cleanup_mod():
    return load_asr_module("stage_02_llm_enhancement/01_punctuation_cleanup.py")


@pytest.fixture(scope="module")
def stage2_errors_mod():
    return load_asr_module("stage_02_llm_enhancement/03_error_correction.py")


@pytest.fixture(scope="module")
def stage3_analysis_mod():
    return load_asr_module("stage_03_diarization/01_speaker_analysis.py")


@pytest.fixture(scope="module")
def stage3_diarize_mod():
    return load_asr_module("stage_03_diarization/02_diarization_correction.py")


@pytest.fixture(scope="module")
def stage4_loop_mod():
    return load_asr_module("stage_04_langgraph_pipeline/02_quality_loop.py")


@pytest.fixture(scope="module")
def stage4_graph_mod():
    return load_asr_module("stage_04_langgraph_pipeline/01_asr_state_graph.py")


@pytest.fixture(scope="module")
def stage5_grounded_mod():
    return load_asr_module("stage_05_rag_context/03_grounded_correction.py")


@pytest.fixture(scope="module")
def stage6_agent_mod():
    return load_asr_module("stage_06_production_agent/02_llm_guided_agent.py")
