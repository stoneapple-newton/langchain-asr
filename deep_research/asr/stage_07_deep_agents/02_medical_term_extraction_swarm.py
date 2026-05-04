"""
Stage 7: Medical Term + Medication Extraction Swarm
===================================================
CONCEPT: Build a Deep Agents supervisor focused on clinically relevant entity
extraction from ASR transcript JSON.

This swarm is useful after transcription when you want structured outputs such
as symptoms, conditions, procedures, and medicine names with local context.

Run this file after installing deepagents:
  uv add deepagents
  uv run deep_research/asr/stage_07_deep_agents/02_medical_term_extraction_swarm.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_TRANSCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "sample_data" / "sample_transcript.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

MEDICATION_SUFFIXES = (
    "mab",
    "nib",
    "pril",
    "sartan",
    "statin",
    "oxetine",
    "zepam",
    "caine",
)
MEDICAL_CONTEXT_TRIGGERS = {
    "diagnosed",
    "diagnosis",
    "symptom",
    "symptoms",
    "dose",
    "dosage",
    "tablet",
    "capsule",
    "prescribed",
    "medication",
    "medicine",
    "pain",
    "allergy",
    "history",
    "treatment",
}


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _load_transcript(path: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(_resolve_path(path).read_text(encoding="utf-8"))
    segments = raw.get("segments", [])
    words = [word for segment in segments for word in segment.get("words", [])]
    return raw, segments, words


def _segment_text(segment: dict[str, Any]) -> str:
    return str(segment.get("text", "")).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'+.-]*", text)


def _context_window(tokens: list[str], index: int, radius: int = 4) -> str:
    start = max(0, index - radius)
    end = min(len(tokens), index + radius + 1)
    return " ".join(tokens[start:end])


@tool
def transcript_overview(path: str) -> str:
    """Summarize transcript metadata and segment counts for medical extraction."""
    raw, segments, words = _load_transcript(path)
    speaker_counts: Counter[str] = Counter(
        str(segment.get("speaker") or "UNKNOWN") for segment in segments
    )
    lines = [
        f"path: {_resolve_path(path)}",
        f"duration: {raw.get('duration')}",
        f"language: {raw.get('language', 'unknown')}",
        f"segments: {len(segments)}",
        f"words: {len(words)}",
        f"speakers: {len(speaker_counts)}",
        f"speaker_distribution: {json.dumps(dict(speaker_counts), indent=2)}",
    ]
    return "\n".join(lines)


@tool
def medical_term_candidates(path: str, min_occurrences: int = 1) -> str:
    """Extract likely medical terms with contextual evidence from transcript text."""
    _, segments, _ = _load_transcript(path)
    term_hits: dict[str, list[str]] = defaultdict(list)

    for segment in segments:
        tokens = _tokenize(_segment_text(segment))
        lower_tokens = [token.lower() for token in tokens]
        for index, token in enumerate(tokens):
            cleaned = token.strip(".,?!:;()[]{}\"'")
            lowered = cleaned.lower()

            looks_medical = (
                lowered.endswith(("itis", "osis", "emia", "algia", "ectomy", "pathy"))
                or lowered in MEDICAL_CONTEXT_TRIGGERS
                or any(trigger in _context_window(lower_tokens, index) for trigger in MEDICAL_CONTEXT_TRIGGERS)
            )
            if looks_medical:
                term_hits[cleaned].append(_context_window(tokens, index, radius=5))

    ranked_terms = sorted(term_hits.items(), key=lambda item: len(item[1]), reverse=True)
    lines = []
    for term, contexts in ranked_terms:
        if len(contexts) < min_occurrences:
            continue
        lines.append(f"- {term}: {len(contexts)} mentions")
        for context in contexts[:2]:
            lines.append(f"  context: {context}")

    return "\n".join(lines) or "No medical term candidates detected."


@tool
def medication_name_candidates(path: str, min_occurrences: int = 1) -> str:
    """Extract likely medication names and dosage-adjacent context windows."""
    _, segments, _ = _load_transcript(path)
    med_hits: dict[str, list[str]] = defaultdict(list)

    dosage_pattern = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|g|units?)\b", flags=re.IGNORECASE)

    for segment in segments:
        text = _segment_text(segment)
        tokens = _tokenize(text)
        lower_tokens = [token.lower() for token in tokens]
        has_dose = bool(dosage_pattern.search(text))

        for index, token in enumerate(tokens):
            normalized = token.strip(".,?!:;()[]{}\"'")
            lowered = normalized.lower()
            suffix_match = lowered.endswith(MEDICATION_SUFFIXES)
            nearby_med_context = any(
                marker in _context_window(lower_tokens, index)
                for marker in ("dose", "dosage", "take", "prescribed", "medication", "tablet")
            )
            if suffix_match or has_dose or nearby_med_context:
                if len(normalized) >= 3:
                    med_hits[normalized].append(_context_window(tokens, index, radius=5))

    ranked = sorted(med_hits.items(), key=lambda item: len(item[1]), reverse=True)
    lines = []
    for medicine, contexts in ranked:
        if len(contexts) < min_occurrences:
            continue
        lines.append(f"- {medicine}: {len(contexts)} mentions")
        for context in contexts[:2]:
            lines.append(f"  context: {context}")
    return "\n".join(lines) or "No medication candidates detected."


@tool
def export_json_report(output_path: str, payload: str) -> str:
    """Persist extraction JSON payload to disk and return the saved path."""
    resolved = _resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    parsed = json.loads(payload)
    resolved.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"saved:{resolved}"


MEDICAL_EXTRACTION_BLUEPRINTS: list[dict[str, str]] = [
    {
        "name": "clinical_intake_agent",
        "description": "Validate transcript quality before entity extraction.",
        "system_prompt": "Start with transcript_overview. Flag missing metadata, short transcripts, and extraction risks before downstream medical labeling.",
    },
    {
        "name": "medical_terms_agent",
        "description": "Extract candidate medical terms with context snippets.",
        "system_prompt": "Use medical_term_candidates to produce a normalized candidate list. Keep context snippets so humans can validate each term.",
    },
    {
        "name": "medication_names_agent",
        "description": "Extract likely medication names and dosage context.",
        "system_prompt": "Use medication_name_candidates and return a medicine list with confidence notes, dosage clues, and ambiguous terms for review.",
    },
    {
        "name": "ambiguity_resolution_agent",
        "description": "Separate likely clinical entities from common language collisions.",
        "system_prompt": "Review term and medication candidates, then mark ambiguous tokens that may be false positives based on local transcript context.",
    },
    {
        "name": "clinical_summary_agent",
        "description": "Merge specialist outputs into a final extraction report.",
        "system_prompt": "Synthesize findings into JSON sections: medical_terms, medications, unresolved_ambiguities, and follow_up_questions.",
    },
]


MAIN_SYSTEM_PROMPT = f"""
You are a medical ASR extraction supervisor built with Deep Agents.

Goal:
- extract medical terms from transcript context
- extract medicine names from transcript context
- return structured outputs suitable for manual validation

Workflow:
1. Run transcript_overview on the transcript path.
2. Delegate to medical_terms_agent and medication_names_agent first.
3. Delegate to ambiguity_resolution_agent for false-positive review.
4. Ask clinical_summary_agent to build final JSON and optionally export with
   export_json_report under /deep_research/asr/stage_07_deep_agents/outputs/.

Default sample transcript:
{SAMPLE_TRANSCRIPT_PATH}
"""


def build_shared_tools() -> list[Any]:
    """Return tools shared by the supervisor and all medical extraction subagents."""
    return [
        transcript_overview,
        medical_term_candidates,
        medication_name_candidates,
        export_json_report,
    ]


def build_subagents(shared_tools: list[Any] | None = None) -> list[dict[str, Any]]:
    """Build subagent specs for medical term and medicine extraction."""
    tools = shared_tools or build_shared_tools()
    return [
        {
            "name": blueprint["name"],
            "description": blueprint["description"],
            "system_prompt": blueprint["system_prompt"],
            "tools": tools,
        }
        for blueprint in MEDICAL_EXTRACTION_BLUEPRINTS
    ]


def build_medical_extraction_swarm(root_dir: str | Path | None = None):
    """Create a Deep Agents supervisor for medical-term and medicine extraction."""
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
    except ImportError as exc:
        raise RuntimeError(
            "deepagents is not installed. Install it with `uv add deepagents` "
            "before building the medical extraction swarm."
        ) from exc

    from langgraph.checkpoint.memory import MemorySaver

    from config import create_chat_model

    repo_root = Path(root_dir) if root_dir else REPO_ROOT
    model = create_chat_model("asr", temperature=0, max_tokens=4096)
    shared_tools = build_shared_tools()

    return create_deep_agent(
        name="asr-medical-extraction-swarm",
        model=model,
        tools=shared_tools,
        system_prompt=MAIN_SYSTEM_PROMPT,
        subagents=build_subagents(shared_tools),
        backend=FilesystemBackend(root_dir=str(repo_root), virtual_mode=True),
        checkpointer=MemorySaver(),
    )


def example_request(transcript_path: str | Path = SAMPLE_TRANSCRIPT_PATH) -> dict[str, Any]:
    """Return an example invocation payload for medical extraction."""
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Extract medical terms and medicine names from this ASR transcript. "
                    f"Return structured JSON with context evidence: {transcript_path}"
                ),
            }
        ]
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("MEDICAL TERM + MEDICATION EXTRACTION SWARM")
    print("=" * 72)
    print(f"Defined specialists: {len(MEDICAL_EXTRACTION_BLUEPRINTS)}")
    for index, blueprint in enumerate(MEDICAL_EXTRACTION_BLUEPRINTS, start=1):
        print(f"{index:>2}. {blueprint['name']} -> {blueprint['description']}")
    print("\nInstall deepagents first if you want to run the swarm:")
    print("  uv add deepagents")
    print(
        "Then build it with build_medical_extraction_swarm() and invoke with example_request()."
    )
