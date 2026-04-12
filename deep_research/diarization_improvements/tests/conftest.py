"""
Shared fixtures and import helper for the diarization improvements test suite.

load_diarization_module() mirrors load_asr_module() from asr/tests/conftest.py:
  1. Mocks LangChain/LangGraph/config packages in sys.modules
  2. Patches builtins.open to return FAKE_JSON for any .json read
  3. Suppresses stdout
  4. Loads the script via importlib
  5. Restores sys.modules in a finally block
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

DIAG_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# CLEAN_SEGMENTS — 6 segments, 3 speakers, proper timing, ≥4 words each.
# No defects. Used as the base for injecting defects in tests.
# ---------------------------------------------------------------------------
CLEAN_SEGMENTS = [
    # Seg 0: SPEAKER_00
    {
        "start": 0.0, "end": 5.5,
        "text": "Alright let us get started today.",
        "speaker": "SPEAKER_00",
        "words": [
            {"word": "Alright", "start": 0.0, "end": 0.6, "score": 0.97, "speaker": "SPEAKER_00"},
            {"word": "let",     "start": 0.7, "end": 0.9, "score": 0.98, "speaker": "SPEAKER_00"},
            {"word": "us",      "start": 1.0, "end": 1.1, "score": 0.99, "speaker": "SPEAKER_00"},
            {"word": "get",     "start": 1.2, "end": 1.4, "score": 0.98, "speaker": "SPEAKER_00"},
            {"word": "started", "start": 1.5, "end": 2.2, "score": 0.99, "speaker": "SPEAKER_00"},
            {"word": "today",   "start": 2.3, "end": 3.0, "score": 0.97, "speaker": "SPEAKER_00"},
        ],
    },
    # Seg 1: SPEAKER_01
    {
        "start": 6.0, "end": 12.0,
        "text": "The main agenda is reviewing the roadmap.",
        "speaker": "SPEAKER_01",
        "words": [
            {"word": "The",       "start": 6.0,  "end": 6.2,  "score": 0.99, "speaker": "SPEAKER_01"},
            {"word": "main",      "start": 6.3,  "end": 6.6,  "score": 0.98, "speaker": "SPEAKER_01"},
            {"word": "agenda",    "start": 6.7,  "end": 7.1,  "score": 0.96, "speaker": "SPEAKER_01"},
            {"word": "is",        "start": 7.2,  "end": 7.35, "score": 0.99, "speaker": "SPEAKER_01"},
            {"word": "reviewing", "start": 7.4,  "end": 8.0,  "score": 0.95, "speaker": "SPEAKER_01"},
            {"word": "the",       "start": 8.1,  "end": 8.2,  "score": 0.99, "speaker": "SPEAKER_01"},
            {"word": "roadmap",   "start": 8.3,  "end": 9.0,  "score": 0.94, "speaker": "SPEAKER_01"},
        ],
    },
    # Seg 2: SPEAKER_02
    {
        "start": 12.5, "end": 18.0,
        "text": "Authentication service has been really flaky.",
        "speaker": "SPEAKER_02",
        "words": [
            {"word": "Authentication", "start": 12.5, "end": 13.3, "score": 0.91, "speaker": "SPEAKER_02"},
            {"word": "service",        "start": 13.4, "end": 13.9, "score": 0.95, "speaker": "SPEAKER_02"},
            {"word": "has",            "start": 14.0, "end": 14.2, "score": 0.99, "speaker": "SPEAKER_02"},
            {"word": "been",           "start": 14.3, "end": 14.5, "score": 0.98, "speaker": "SPEAKER_02"},
            {"word": "really",         "start": 14.6, "end": 14.9, "score": 0.97, "speaker": "SPEAKER_02"},
            {"word": "flaky",          "start": 15.0, "end": 15.6, "score": 0.93, "speaker": "SPEAKER_02"},
        ],
    },
    # Seg 3: SPEAKER_01
    {
        "start": 18.5, "end": 24.0,
        "text": "The JWT library needs to be updated.",
        "speaker": "SPEAKER_01",
        "words": [
            {"word": "The",     "start": 18.5, "end": 18.7, "score": 0.99, "speaker": "SPEAKER_01"},
            {"word": "JWT",     "start": 18.8, "end": 19.1, "score": 0.88, "speaker": "SPEAKER_01"},
            {"word": "library", "start": 19.2, "end": 19.7, "score": 0.97, "speaker": "SPEAKER_01"},
            {"word": "needs",   "start": 19.8, "end": 20.1, "score": 0.98, "speaker": "SPEAKER_01"},
            {"word": "to",      "start": 20.2, "end": 20.3, "score": 0.99, "speaker": "SPEAKER_01"},
            {"word": "be",      "start": 20.4, "end": 20.5, "score": 0.99, "speaker": "SPEAKER_01"},
            {"word": "updated", "start": 20.6, "end": 21.2, "score": 0.96, "speaker": "SPEAKER_01"},
        ],
    },
    # Seg 4: SPEAKER_00
    {
        "start": 24.5, "end": 30.0,
        "text": "DevOps needs to approve the node upgrade.",
        "speaker": "SPEAKER_00",
        "words": [
            {"word": "DevOps",  "start": 24.5, "end": 25.0, "score": 0.89, "speaker": "SPEAKER_00"},
            {"word": "needs",   "start": 25.1, "end": 25.4, "score": 0.98, "speaker": "SPEAKER_00"},
            {"word": "to",      "start": 25.5, "end": 25.6, "score": 0.99, "speaker": "SPEAKER_00"},
            {"word": "approve", "start": 25.7, "end": 26.2, "score": 0.97, "speaker": "SPEAKER_00"},
            {"word": "the",     "start": 26.3, "end": 26.4, "score": 0.99, "speaker": "SPEAKER_00"},
            {"word": "node",    "start": 26.5, "end": 26.8, "score": 0.96, "speaker": "SPEAKER_00"},
            {"word": "upgrade", "start": 26.9, "end": 27.5, "score": 0.95, "speaker": "SPEAKER_00"},
        ],
    },
    # Seg 5: SPEAKER_02
    {
        "start": 30.5, "end": 36.0,
        "text": "Carol is handling the dashboard design.",
        "speaker": "SPEAKER_02",
        "words": [
            {"word": "Carol",     "start": 30.5, "end": 30.8, "score": 0.98, "speaker": "SPEAKER_02"},
            {"word": "is",        "start": 30.9, "end": 31.0, "score": 0.99, "speaker": "SPEAKER_02"},
            {"word": "handling",  "start": 31.1, "end": 31.6, "score": 0.97, "speaker": "SPEAKER_02"},
            {"word": "the",       "start": 31.7, "end": 31.8, "score": 0.99, "speaker": "SPEAKER_02"},
            {"word": "dashboard", "start": 31.9, "end": 32.5, "score": 0.96, "speaker": "SPEAKER_02"},
            {"word": "design",    "start": 32.6, "end": 33.2, "score": 0.97, "speaker": "SPEAKER_02"},
        ],
    },
]


def _make_transcript(segments: list[dict]) -> dict:
    """Wrap segments in a minimal WhisperX transcript dict."""
    return {
        "meeting_metadata": {"title": "Test Meeting", "date": "2024-01-01"},
        "duration": segments[-1]["end"] if segments else 0.0,
        "language": "en",
        "segments": segments,
    }


# Seg 1 + 2 merged into one segment (run-on: SPEAKER_01 text merged into SPEAKER_02 segment)
import copy

_seg1 = copy.deepcopy(CLEAN_SEGMENTS[1])
_seg2 = copy.deepcopy(CLEAN_SEGMENTS[2])
_merged_words = _seg1["words"] + _seg2["words"]
_run_on_seg = {
    "start": _seg1["start"],
    "end": _seg2["end"],
    "text": _seg1["text"].rstrip(".") + " " + _seg2["text"],
    "speaker": _seg1["speaker"],   # only SPEAKER_01 label — SPEAKER_02 text merged in
    "words": _merged_words,
}
RUN_ON_SEGMENTS = (
    [CLEAN_SEGMENTS[0]]
    + [_run_on_seg]
    + CLEAN_SEGMENTS[3:]
)

# Seg 2 with SPEAKER_02: prepended to text; speaker field cleared
_ha_seg = copy.deepcopy(CLEAN_SEGMENTS[2])
_ha_seg["text"] = "SPEAKER_02: " + _ha_seg["text"]
_ha_seg["speaker"] = ""
HEAD_ATTACHED_SEGMENTS = (
    CLEAN_SEGMENTS[:2] + [_ha_seg] + CLEAN_SEGMENTS[3:]
)

# Seg 4 with SPEAKER_00 appended to text; speaker field cleared
_ta_seg = copy.deepcopy(CLEAN_SEGMENTS[4])
_ta_seg["text"] = _ta_seg["text"].rstrip(".") + " SPEAKER_00"
_ta_seg["speaker"] = ""
TAIL_ATTACHED_SEGMENTS = (
    CLEAN_SEGMENTS[:4] + [_ta_seg] + CLEAN_SEGMENTS[5:]
)

FAKE_TRANSCRIPT = _make_transcript(CLEAN_SEGMENTS)
RUN_ON_TRANSCRIPT = _make_transcript(RUN_ON_SEGMENTS)
HEAD_ATTACHED_TRANSCRIPT = _make_transcript(HEAD_ATTACHED_SEGMENTS)
TAIL_ATTACHED_TRANSCRIPT = _make_transcript(TAIL_ATTACHED_SEGMENTS)


# ---------------------------------------------------------------------------
# Mock module list
# ---------------------------------------------------------------------------
MOCK_MODULE_NAMES = [
    "config",
    "langchain",
    "langchain.agents",
    "langchain_core",
    "langchain_core.documents",
    "langchain_core.output_parsers",
    "langchain_core.prompts",
    "langchain_core.runnables",
    "langchain_core.tools",
    "langchain_core.vectorstores",
    "langchain_ollama",
    "langchain_text_splitters",
    "langgraph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "langgraph.graph",
    "langgraph.types",
    "deepagents",
    "deepagents.backends",
    "pydantic",
]


def load_diarization_module(rel_path: str):
    """
    Load a diarization_improvements script with all LangChain/LangGraph deps
    mocked and file I/O pointing to FAKE_TRANSCRIPT. Returns the module object.
    """
    abs_path = DIAG_ROOT / rel_path
    module_name = f"_diag_{abs_path.parent.name}_{abs_path.stem}"

    _real_open = builtins.open
    fake_json_bytes = json.dumps(
        {"version": "1.0", "seed": 42, "total_examples": 0, "examples": []}
    )

    def fake_open(file, mode="r", *args, **kwargs):
        if str(file).endswith(".json") and "r" in str(mode):
            return io.StringIO(fake_json_bytes)
        if "w" in str(mode) or "a" in str(mode):
            return io.StringIO()
        return _real_open(file, mode, *args, **kwargs)

    mock_modules = {name: MagicMock() for name in MOCK_MODULE_NAMES}
    mock_modules["langgraph.graph"].START = "START"
    mock_modules["langgraph.graph"].END = "END"

    fake_llm = MagicMock()
    fake_embeddings = MagicMock()
    mock_modules["config"].create_chat_model = MagicMock(return_value=fake_llm)
    mock_modules["config"].create_embeddings = MagicMock(return_value=fake_embeddings)

    # Keep real pydantic so Pydantic models in scripts work correctly
    del mock_modules["pydantic"]

    original_modules = {
        k: sys.modules.get(k)
        for k in MOCK_MODULE_NAMES
        if k != "pydantic"
    }
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
# Module-scope fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage2_run_on_mod():
    return load_diarization_module("stage_02_one_shot/01_fix_run_on.py")


@pytest.fixture(scope="module")
def stage2_head_mod():
    return load_diarization_module("stage_02_one_shot/02_fix_head_attached.py")


@pytest.fixture(scope="module")
def stage2_tail_mod():
    return load_diarization_module("stage_02_one_shot/03_fix_tail_attached.py")


@pytest.fixture(scope="module")
def stage4_graph_mod():
    return load_diarization_module("stage_04_langgraph/01_correction_graph.py")
