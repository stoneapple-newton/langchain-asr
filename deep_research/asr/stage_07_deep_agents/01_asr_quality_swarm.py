"""
Stage 7: ASR Quality Swarm With Deep Agents
===========================================
CONCEPT: Use the Deep Agents harness to coordinate many specialised ASR
workers under one supervisor.

This stage extends the earlier ASR pipeline into a "swarm" architecture:

  supervisor deep agent
    -> delegates with `task(...)`
    -> specialised ASR subagents inspect one failure mode each
    -> subagents use shared transcript-analysis tools and built-in filesystem tools
    -> supervisor merges findings into a correction plan or review packet

Run this file after installing deepagents:
  uv add deepagents
  uv run deep_research/asr/stage_07_deep_agents/01_asr_quality_swarm.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_TRANSCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "sample_data" / "sample_transcript.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FILLERS = {"uh", "um", "you know", "i mean", "like", "so", "right"}


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _load_transcript(path: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = _resolve_path(path)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    segments = raw.get("segments", [])
    words = [word for segment in segments for word in segment.get("words", [])]
    return raw, segments, words


def _segment_text(segment: dict[str, Any]) -> str:
    return str(segment.get("text", "")).strip()


def _segment_word_count(segment: dict[str, Any]) -> int:
    words = segment.get("words", [])
    if words:
        return len(words)
    return len(_segment_text(segment).split())


def _speaker_counts(segments: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for segment in segments:
        counts[str(segment.get("speaker") or "UNKNOWN")] += 1
    return dict(counts)


def _quality_metrics(segments: list[dict[str, Any]], words: list[dict[str, Any]]) -> dict[str, Any]:
    total_segments = max(len(segments), 1)
    total_words = max(len(words), 1)
    punctuation_score = sum(
        1 for segment in segments if re.search(r"[.!?]$", _segment_text(segment))
    ) / total_segments
    capitalisation_score = sum(
        1
        for segment in segments
        if _segment_text(segment) and _segment_text(segment)[0].isupper()
    ) / total_segments
    filler_hits = sum(
        1
        for word in words
        if str(word.get("word", "")).lower().strip(".,?!") in FILLERS
    )
    avg_confidence = sum(float(word.get("score", 1.0)) for word in words) / total_words
    missing_speakers = sum(1 for segment in segments if not segment.get("speaker"))
    return {
        "segment_count": len(segments),
        "word_count": len(words),
        "speaker_count": len(_speaker_counts(segments)),
        "missing_speaker_segments": missing_speakers,
        "punctuation_score": round(punctuation_score, 3),
        "capitalisation_score": round(capitalisation_score, 3),
        "filler_rate": round(filler_hits / total_words, 3),
        "avg_confidence": round(avg_confidence, 3),
    }


@tool
def transcript_overview(path: str) -> str:
    """Summarize transcript metadata, quality metrics, and speaker coverage."""
    raw, segments, words = _load_transcript(path)
    metrics = _quality_metrics(segments, words)
    speaker_counts = _speaker_counts(segments)
    duration = raw.get("duration")
    lines = [
        f"path: {_resolve_path(path)}",
        f"duration: {duration}",
        f"language: {raw.get('language', 'unknown')}",
        f"segments: {metrics['segment_count']}",
        f"words: {metrics['word_count']}",
        f"speakers: {metrics['speaker_count']}",
        f"missing_speaker_segments: {metrics['missing_speaker_segments']}",
        f"avg_confidence: {metrics['avg_confidence']}",
        f"punctuation_score: {metrics['punctuation_score']}",
        f"capitalisation_score: {metrics['capitalisation_score']}",
        f"filler_rate: {metrics['filler_rate']}",
        f"speaker_distribution: {json.dumps(speaker_counts, indent=2)}",
    ]
    return "\n".join(lines)


@tool
def low_confidence_spans(path: str, threshold: float = 0.8, limit: int = 10) -> str:
    """List the lowest-confidence words and their segment context."""
    _, segments, words = _load_transcript(path)
    low_words = [
        {
            "word": str(word.get("word", "")),
            "score": float(word.get("score", 1.0)),
            "start": word.get("start"),
            "end": word.get("end"),
            "speaker": word.get("speaker"),
        }
        for word in words
        if float(word.get("score", 1.0)) < threshold
    ]
    low_words.sort(key=lambda item: item["score"])
    lines = [f"low_confidence_words<{threshold}: {len(low_words)}"]
    for item in low_words[:limit]:
        context = ""
        for segment in segments:
            segment_words = [str(word.get("word", "")) for word in segment.get("words", [])]
            if item["word"] in segment_words:
                context = _segment_text(segment)[:120]
                break
        lines.append(
            f"- word={item['word']} score={item['score']:.3f} "
            f"time={item['start']}-{item['end']} speaker={item['speaker']} context={context}"
        )
    return "\n".join(lines)


@tool
def diarization_anomalies(path: str, fast_switch_threshold: float = 0.15) -> str:
    """Find missing speakers, short turns, and suspicious rapid speaker switches."""
    _, segments, _ = _load_transcript(path)
    findings: list[str] = []
    for index, segment in enumerate(segments):
        speaker = segment.get("speaker") or "UNKNOWN"
        duration = float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))
        word_count = _segment_word_count(segment)
        if speaker == "UNKNOWN":
            findings.append(f"[seg {index}] missing speaker label")
        if duration < 1.0 or word_count <= 2:
            findings.append(
                f"[seg {index}] short_turn speaker={speaker} duration={duration:.2f}s words={word_count}"
            )
        if index > 0:
            previous = segments[index - 1]
            previous_speaker = previous.get("speaker") or "UNKNOWN"
            gap = float(segment.get("start", 0.0)) - float(previous.get("end", 0.0))
            if speaker != previous_speaker and gap < fast_switch_threshold:
                findings.append(
                    f"[seg {index}] fast_switch {previous_speaker}->{speaker} gap={gap:.2f}s"
                )
    return "\n".join(findings[:30]) or "No diarization anomalies detected."


@tool
def overlap_candidates(path: str) -> str:
    """Detect segments whose timestamps overlap or nearly collide."""
    _, segments, _ = _load_transcript(path)
    findings: list[str] = []
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        previous_end = float(previous.get("end", 0.0))
        current_start = float(current.get("start", 0.0))
        if current_start < previous_end:
            findings.append(
                f"[seg {index - 1}->{index}] overlap={previous_end - current_start:.2f}s "
                f"{previous.get('speaker')} -> {current.get('speaker')}"
            )
        elif current_start - previous_end < 0.05:
            findings.append(
                f"[seg {index - 1}->{index}] near_collision gap={current_start - previous_end:.2f}s"
            )
    return "\n".join(findings[:30]) or "No overlap candidates detected."


@tool
def glossary_candidates(path: str, min_frequency: int = 2) -> str:
    """Suggest domain terms, acronyms, and repeated proper nouns for a glossary."""
    _, segments, words = _load_transcript(path)
    token_counts: Counter[str] = Counter()
    for word in words:
        token = str(word.get("word", "")).strip(".,?!")
        if token:
            token_counts[token] += 1

    candidates: list[str] = []
    for token, count in token_counts.most_common():
        is_acronym = token.isupper() and len(token) >= 2
        is_named = token[:1].isupper() and any(char.islower() for char in token[1:])
        if count >= min_frequency and (is_acronym or is_named):
            candidates.append(f"{token} ({count} mentions)")

    if not candidates:
        # Fall back to capitalised words from text if word-level data is sparse.
        for segment in segments:
            for token in re.findall(r"\b[A-Z][A-Za-z0-9.+-]*\b", _segment_text(segment)):
                token_counts[token] += 1
        for token, count in token_counts.most_common():
            if count >= min_frequency:
                candidates.append(f"{token} ({count} mentions)")
    return "\n".join(candidates[:25]) or "No glossary candidates detected."


@tool
def filler_analysis(path: str) -> str:
    """Report filler-heavy segments and common filler tokens."""
    _, segments, words = _load_transcript(path)
    filler_counts: Counter[str] = Counter()
    segment_hits: list[str] = []
    for index, segment in enumerate(segments):
        segment_words = [
            str(word.get("word", "")).lower().strip(".,?!") for word in segment.get("words", [])
        ]
        hits = [word for word in segment_words if word in FILLERS]
        if hits:
            filler_counts.update(hits)
            segment_hits.append(
                f"[seg {index}] speaker={segment.get('speaker')} fillers={hits} text={_segment_text(segment)}"
            )
    lines = ["filler_counts: " + json.dumps(dict(filler_counts), indent=2)]
    lines.extend(segment_hits[:20])
    if not words:
        lines.append("word-level data missing; filler analysis may be incomplete")
    return "\n".join(lines)


@tool
def numeric_entity_candidates(path: str) -> str:
    """Find dates, versions, numbers, and all-caps entities that deserve verification."""
    _, segments, _ = _load_transcript(path)
    findings: list[str] = []
    patterns = [
        r"\b\d+(?:\.\d+)?\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
        r"\b(?:v?\d+\.\d+|Node\.js|JWT|WCAG|API|SLA)\b",
    ]
    for index, segment in enumerate(segments):
        text = _segment_text(segment)
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        if matches:
            findings.append(
                f"[seg {index}] speaker={segment.get('speaker')} entities={sorted(set(matches))} text={text}"
            )
    return "\n".join(findings[:20]) or "No numeric/entity candidates detected."


@tool
def export_markdown_report(output_path: str, title: str, body: str) -> str:
    """Write a markdown report to disk and return the saved path."""
    resolved = _resolve_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return f"saved:{resolved}"


ASR_DEEP_AGENT_BLUEPRINTS: list[dict[str, str]] = [
    {
        "name": "audio_intake_agent",
        "description": "Audit transcript metadata and decide whether the input is fit for downstream ASR cleanup.",
        "system_prompt": "You are the intake gatekeeper for ASR quality review. Start with transcript_overview and flag ingest problems, missing metadata, suspicious duration values, and malformed segment structure.",
    },
    {
        "name": "segmentation_agent",
        "description": "Inspect segment boundaries for overlong chunks, clipped thoughts, and poor turn segmentation.",
        "system_prompt": "You audit segmentation quality. Look for over-compressed segments, abrupt starts or endings, and places where utterances should be split or merged.",
    },
    {
        "name": "silence_hallucination_agent",
        "description": "Look for transcript spans that may be hallucinated around silence or low-confidence regions.",
        "system_prompt": "You investigate likely hallucinations. Use low_confidence_spans and transcript_overview to identify spans that deserve re-decoding or manual review.",
    },
    {
        "name": "multi_channel_router_agent",
        "description": "Decide whether the transcript would benefit from channel-aware handling instead of generic diarization.",
        "system_prompt": "You specialise in routing call-center and meeting audio. Recommend channel-based processing when speaker separation looks unstable or alternating turns appear conflated.",
    },
    {
        "name": "speaker_count_agent",
        "description": "Estimate realistic speaker cardinality and highlight under- or over-clustered diarization.",
        "system_prompt": "You estimate how many speakers are really present based on turns, label usage, and conversational structure. Provide a tight speaker-count hypothesis.",
    },
    {
        "name": "diarization_repair_agent",
        "description": "Repair missing, drifting, or unstable speaker labels.",
        "system_prompt": "You are a diarization repair specialist. Use diarization_anomalies and transcript_overview to produce a concrete speaker-label repair plan.",
    },
    {
        "name": "overlap_resolution_agent",
        "description": "Handle overlapping or near-colliding speech turns.",
        "system_prompt": "You investigate overlapping speech. Use overlap_candidates and diarization_anomalies to decide which spans need overlap-aware treatment or escalation.",
    },
    {
        "name": "backchannel_merge_agent",
        "description": "Find low-information acknowledgements that should be merged or deprioritized.",
        "system_prompt": "You specialise in short turns and backchannels. Use diarization_anomalies to spot acknowledgement fragments like 'yeah' or 'right' that hurt readability more than they help.",
    },
    {
        "name": "confidence_triage_agent",
        "description": "Prioritize the riskiest spans using confidence signals.",
        "system_prompt": "You build a ranked review queue from low_confidence_spans. Group nearby errors, explain why they matter, and recommend where human time should go first.",
    },
    {
        "name": "phonetic_correction_agent",
        "description": "Look for acoustically plausible word substitutions and homophone-like errors.",
        "system_prompt": "You hunt phonetic confusions and near-homophone substitutions. Use low_confidence_spans, glossary_candidates, and local context to suggest repair targets.",
    },
    {
        "name": "acronym_normalizer_agent",
        "description": "Normalize acronyms, initialisms, and version-like tokens.",
        "system_prompt": "You protect acronyms and technical shortforms such as JWT, WCAG, and API. Use glossary_candidates and numeric_entity_candidates to produce normalization rules.",
    },
    {
        "name": "glossary_guard_agent",
        "description": "Build or refine a project glossary for domain-specific ASR correction.",
        "system_prompt": "You are responsible for domain vocabulary. Use glossary_candidates and transcript_overview to propose a glossary with canonical spellings and casing.",
    },
    {
        "name": "punctuation_restoration_agent",
        "description": "Assess punctuation and sentence boundary quality.",
        "system_prompt": "You focus on punctuation restoration. Start with transcript_overview and identify where sentence boundaries or capitalization are degrading readability.",
    },
    {
        "name": "disfluency_policy_agent",
        "description": "Recommend which fillers and false starts to keep, remove, or annotate.",
        "system_prompt": "You define the disfluency policy. Use filler_analysis to separate harmless realism from noise that should be cleaned in production transcripts.",
    },
    {
        "name": "code_switch_audit_agent",
        "description": "Detect mixed-language or script-switching spans that need special handling.",
        "system_prompt": "You audit for code-switching and multilingual spans. Look for language mixing, transliteration issues, and segments that should not be normalized into one language.",
    },
    {
        "name": "timestamp_alignment_agent",
        "description": "Audit whether transcript content and timing remain aligned after post-processing.",
        "system_prompt": "You inspect alignment health. Focus on timing consistency, suspiciously dense segments, and whether later cleanup could break subtitle timing.",
    },
    {
        "name": "numeric_entity_verifier_agent",
        "description": "Verify dates, numbers, versions, and critical entities.",
        "system_prompt": "You are the verifier for dates, versions, counts, and all-caps entities. Use numeric_entity_candidates to extract the highest-risk factual tokens.",
    },
    {
        "name": "context_grounding_agent",
        "description": "Find places where extra project or meeting context is needed to resolve ambiguity.",
        "system_prompt": "You identify ambiguity that cannot be solved from transcript text alone. Recommend what external context, glossary, or participant roster would disambiguate each case.",
    },
    {
        "name": "severity_review_agent",
        "description": "Rank ASR issues by downstream harm rather than cosmetic impact.",
        "system_prompt": "You perform impact-aware review. Elevate negation, numbers, commitments, deadlines, and named entities over cosmetic punctuation-only issues.",
    },
    {
        "name": "redaction_review_agent",
        "description": "Identify spans that may need PHI or PII redaction before sharing.",
        "system_prompt": "You inspect transcripts for privacy risk. Look for names, dates, identifiers, account-like numbers, and sensitive project details that may require redaction.",
    },
]


MAIN_SYSTEM_PROMPT = f"""
You are an ASR quality supervisor built with Deep Agents.

Your job is to inspect WhisperX-like transcript JSON, delegate to the right
specialists, and return a practical correction plan. Prefer targeted
delegation over broad speculation.

Workflow:
1. Start with transcript_overview on the user-provided file.
2. Use task(...) to delegate to the most relevant ASR specialists.
3. Ask subagents to save intermediate notes under
   /deep_research/asr/stage_07_deep_agents/outputs/ when useful.
4. Merge their findings into:
   - priority issues
   - recommended deterministic fixes
   - recommended LLM fixes
   - recommended human-review spans

Default sample transcript:
{SAMPLE_TRANSCRIPT_PATH}
"""


def build_shared_tools() -> list[Any]:
    """Return the shared ASR inspection tools used by the supervisor and subagents."""
    return [
        transcript_overview,
        low_confidence_spans,
        diarization_anomalies,
        overlap_candidates,
        glossary_candidates,
        filler_analysis,
        numeric_entity_candidates,
        export_markdown_report,
    ]


def build_subagents(shared_tools: list[Any] | None = None) -> list[dict[str, Any]]:
    """Build the 20 ASR Deep Agent subagent specifications."""
    tools = shared_tools or build_shared_tools()
    return [
        {
            "name": blueprint["name"],
            "description": blueprint["description"],
            "system_prompt": blueprint["system_prompt"],
            "tools": tools,
        }
        for blueprint in ASR_DEEP_AGENT_BLUEPRINTS
    ]


def build_asr_quality_swarm(root_dir: str | Path | None = None):
    """Create the Deep Agents supervisor with 20 ASR specialists."""
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
    except ImportError as exc:
        raise RuntimeError(
            "deepagents is not installed. Install it with `uv add deepagents` "
            "before building the ASR quality swarm."
        ) from exc

    from langgraph.checkpoint.memory import MemorySaver

    from config import create_chat_model

    repo_root = Path(root_dir) if root_dir else REPO_ROOT
    model = create_chat_model("asr", temperature=0, max_tokens=4096)
    shared_tools = build_shared_tools()

    return create_deep_agent(
        name="asr-quality-swarm",
        model=model,
        tools=shared_tools,
        system_prompt=MAIN_SYSTEM_PROMPT,
        subagents=build_subagents(shared_tools),
        backend=FilesystemBackend(root_dir=str(repo_root), virtual_mode=True),
        checkpointer=MemorySaver(),
    )


def example_request(transcript_path: str | Path = SAMPLE_TRANSCRIPT_PATH) -> dict[str, Any]:
    """Return a ready-to-run example invoke payload."""
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Review this transcript and produce an ASR quality plan. "
                    f"Use specialist subagents where appropriate: {transcript_path}"
                ),
            }
        ]
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("ASR QUALITY SWARM (DEEP AGENTS)")
    print("=" * 72)
    print(f"Defined specialists: {len(ASR_DEEP_AGENT_BLUEPRINTS)}")
    for index, blueprint in enumerate(ASR_DEEP_AGENT_BLUEPRINTS, start=1):
        print(f"{index:>2}. {blueprint['name']} -> {blueprint['description']}")
    print("\nInstall deepagents first if you want to run the swarm:")
    print("  uv add deepagents")
    print(
        "Then build it with build_asr_quality_swarm() and invoke it with example_request()."
    )
