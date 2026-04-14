from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transcript_utils import TranscriptDocument, TranscriptSegment, load_transcript


@dataclass
class SegmentDiff:
    segment_id: str
    category: str
    reference_text: str
    candidate_text: str
    reference_speaker: str | None
    candidate_speaker: str | None
    reference_start: float
    candidate_start: float
    detail: str


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_punctuation(value: str) -> str:
    return re.sub(r"[^\w\s]", "", value)


def _token_set(value: str) -> set[str]:
    return {token for token in _strip_punctuation(value.lower()).split() if token}


def _segment_key(segment: TranscriptSegment, fallback_index: int) -> str:
    return segment.segment_id or str(fallback_index)


def _categorize_segment_difference(reference: TranscriptSegment, candidate: TranscriptSegment) -> tuple[str, str]:
    ref_text = _normalize_text(reference.text)
    cand_text = _normalize_text(candidate.text)

    if reference.speaker != candidate.speaker:
        return "speaker_change", "Speaker label changed for aligned segment"

    if ref_text == cand_text and reference.start != candidate.start:
        return "timing_shift", "Segment text is unchanged but start time moved"

    if ref_text == cand_text:
        return "unchanged", "Segment text and speaker match"

    lowered_ref = ref_text.lower()
    lowered_cand = cand_text.lower()
    if _strip_punctuation(lowered_ref) == _strip_punctuation(lowered_cand):
        return "punctuation_or_case", "Token content matches after stripping punctuation/case"

    ref_tokens = _token_set(ref_text)
    cand_tokens = _token_set(cand_text)
    if ref_tokens and cand_tokens:
        overlap = len(ref_tokens & cand_tokens) / max(len(ref_tokens), len(cand_tokens))
        if overlap >= 0.6:
            return "wording_change", f"Most tokens overlap ({overlap:.2f}) but phrasing differs"

    return "semantic_shift", "Token overlap is limited; likely a substantive transcript difference"


def compare_transcript_documents(reference: TranscriptDocument, candidate: TranscriptDocument) -> list[SegmentDiff]:
    candidate_by_id = {_segment_key(segment, idx): segment for idx, segment in enumerate(candidate.segments)}
    reference_by_id = {_segment_key(segment, idx): segment for idx, segment in enumerate(reference.segments)}

    all_keys = sorted(set(reference_by_id) | set(candidate_by_id), key=lambda value: int(value) if value.isdigit() else value)
    differences: list[SegmentDiff] = []

    for key in all_keys:
        ref_segment = reference_by_id.get(key)
        cand_segment = candidate_by_id.get(key)

        if ref_segment and not cand_segment:
            differences.append(
                SegmentDiff(
                    segment_id=key,
                    category="deletion",
                    reference_text=ref_segment.text,
                    candidate_text="",
                    reference_speaker=ref_segment.speaker,
                    candidate_speaker=None,
                    reference_start=ref_segment.start,
                    candidate_start=-1.0,
                    detail="Reference segment missing in candidate transcript",
                )
            )
            continue

        if cand_segment and not ref_segment:
            differences.append(
                SegmentDiff(
                    segment_id=key,
                    category="insertion",
                    reference_text="",
                    candidate_text=cand_segment.text,
                    reference_speaker=None,
                    candidate_speaker=cand_segment.speaker,
                    reference_start=-1.0,
                    candidate_start=cand_segment.start,
                    detail="Candidate transcript introduced a new segment",
                )
            )
            continue

        if not ref_segment or not cand_segment:
            continue

        category, detail = _categorize_segment_difference(ref_segment, cand_segment)
        differences.append(
            SegmentDiff(
                segment_id=key,
                category=category,
                reference_text=ref_segment.text,
                candidate_text=cand_segment.text,
                reference_speaker=ref_segment.speaker,
                candidate_speaker=cand_segment.speaker,
                reference_start=ref_segment.start,
                candidate_start=cand_segment.start,
                detail=detail,
            )
        )

    return differences


def build_comparison_report(reference_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    reference = load_transcript(reference_path)
    candidate = load_transcript(candidate_path)
    differences = compare_transcript_documents(reference, candidate)
    counts = Counter(item.category for item in differences)

    return {
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "totals": {
            "reference_segments": len(reference.segments),
            "candidate_segments": len(candidate.segments),
            "by_category": dict(sorted(counts.items())),
        },
        "differences": [
            {
                "segment_id": item.segment_id,
                "category": item.category,
                "reference_text": item.reference_text,
                "candidate_text": item.candidate_text,
                "reference_speaker": item.reference_speaker,
                "candidate_speaker": item.candidate_speaker,
                "reference_start": item.reference_start,
                "candidate_start": item.candidate_start,
                "detail": item.detail,
            }
            for item in differences
        ],
    }


def format_report_markdown(report: dict[str, Any], include_unchanged: bool = False) -> str:
    counts = report["totals"]["by_category"]
    lines = [
        "# ASR Transcript Diff Report",
        "",
        f"- Reference: `{Path(report['reference_path']).name}`",
        f"- Candidate: `{Path(report['candidate_path']).name}`",
        f"- Reference segments: `{report['totals']['reference_segments']}`",
        f"- Candidate segments: `{report['totals']['candidate_segments']}`",
        "",
        "## Category Counts",
    ]

    for category, count in sorted(counts.items()):
        if not include_unchanged and category == "unchanged":
            continue
        lines.append(f"- **{category}**: {count}")

    lines.append("")
    lines.append("## Segment-Level Differences")
    for item in report["differences"]:
        if not include_unchanged and item["category"] == "unchanged":
            continue
        lines.append(f"- Segment `{item['segment_id']}` — **{item['category']}**")
        lines.append(f"  - Detail: {item['detail']}")
        lines.append(f"  - Reference: {item['reference_speaker'] or 'UNKNOWN'} | {item['reference_text']}")
        lines.append(f"  - Candidate: {item['candidate_speaker'] or 'UNKNOWN'} | {item['candidate_text']}")

    lines.append("")
    return "\n".join(lines)


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=True)
