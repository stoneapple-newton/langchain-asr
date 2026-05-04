from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

_TASK_RE = re.compile(
    r"\b(will|would|should|need to|needs to|going to|have to|must|"
    r"action item|follow up|follow-up|reach out|send|schedule|set up|"
    r"check|review|update|make sure|confirm|finalize|prepare|create|"
    r"write|complete|can you|could you|please|going to)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(decided|decision|agreed|we agree|let'?s go with|we'?ll go|"
    r"confirmed|settled|approved|chosen|picked|selected|we are going with|"
    r"the plan is|we have decided)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"\b(can anyone|does anyone|who will|who is going to|what about|"
    r"when will|are we|is there|do we have|has anyone|who owns)\b",
    re.IGNORECASE,
)

ItemType = Literal["task", "decision", "open_question", "commitment"]


@dataclass
class CandidateItem:
    segment_id: str
    speaker: str | None
    start: float
    end: float
    text: str
    candidate_types: list[str] = field(default_factory=list)
    match_count: int = 0


@dataclass
class ActionItem:
    item_id: str
    item_type: ItemType
    text: str
    raw_segment_text: str
    segment_id: str
    owner: str | None
    due_context: str | None
    confidence: float


def detect_candidates(segments: list) -> list[CandidateItem]:
    """Return segments that contain action item signals, sorted by match count."""
    candidates: list[CandidateItem] = []

    for seg in segments:
        task_count = len(_TASK_RE.findall(seg.text))
        decision_count = len(_DECISION_RE.findall(seg.text))
        question_count = len(_QUESTION_RE.findall(seg.text))
        total = task_count + decision_count + question_count

        if total == 0:
            continue

        types: list[str] = []
        if task_count > 0:
            types.append("task")
        if decision_count > 0:
            types.append("decision")
        if question_count > 0:
            types.append("open_question")

        candidates.append(
            CandidateItem(
                segment_id=seg.segment_id,
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                candidate_types=types,
                match_count=total,
            )
        )

    return sorted(candidates, key=lambda c: -c.match_count)


def format_candidates_for_llm(candidates: list[CandidateItem]) -> str:
    if not candidates:
        return "(no candidates detected)"
    lines: list[str] = []
    for c in candidates:
        speaker = c.speaker or "UNKNOWN"
        types = ", ".join(c.candidate_types)
        lines.append(
            f"[seg={c.segment_id} t={c.start:.1f}s "
            f"speaker={speaker} hints=({types})]"
        )
        lines.append(f'  "{c.text}"')
    return "\n".join(lines)


def render_action_items_report(items: list[ActionItem]) -> str:
    if not items:
        return "# Action Items Report\n\nNo action items found.\n"

    lines = ["# Action Items Report", ""]

    def section(title: str, subset: list[ActionItem]) -> None:
        if not subset:
            return
        lines.append(f"## {title}")
        lines.append("")
        for item in subset:
            owner_str = f" **[{item.owner}]**" if item.owner else ""
            due_str = f" _(due: {item.due_context})_" if item.due_context else ""
            conf_str = f" conf={item.confidence:.0%}"
            lines.append(f"- `{item.item_id}`{owner_str}{due_str}{conf_str}: {item.text}")
        lines.append("")

    section("Tasks", [i for i in items if i.item_type == "task"])
    section("Decisions", [i for i in items if i.item_type == "decision"])
    section("Commitments", [i for i in items if i.item_type == "commitment"])
    section("Open Questions", [i for i in items if i.item_type == "open_question"])

    return "\n".join(lines)
