"""
ASR Transcript Comparison Utilities
====================================
Pure-Python helpers for aligning, diffing, and categorising differences
between two ASR transcript documents.  No LLM calls — deterministic only.

Public API:
  align_segments(doc_a, doc_b, iou_threshold) -> list[AlignedPair]
  categorise_pair(pair)                        -> DiffCategory
  compute_wer(ref_words, hyp_words)            -> float
  compute_document_summary(diffs)              -> dict
  render_report_markdown(diffs, summary, ...)  -> str
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Filler word vocabulary
# ---------------------------------------------------------------------------
FILLER_TOKENS: frozenset[str] = frozenset(
    {"um", "uh", "hmm", "hm", "mhm", "uh-huh", "ah", "er", "like", "you know"}
)


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class DiffCategory(str, Enum):
    MATCH = "match"
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"          # segment in B but no match in A
    DELETION = "deletion"            # segment in A but no match in B
    SPEAKER_MISMATCH = "speaker_mismatch"
    READABILITY = "readability"      # punctuation / capitalisation only
    FILLER_WORD = "filler_word"      # only filler tokens added/removed
    AMBIGUOUS = "ambiguous"          # rule-based cannot resolve → LLM


@dataclass
class AlignedPair:
    """One row produced by align_segments()."""
    pair_id: str
    ref_start: float
    ref_end: float
    ref_text: str
    ref_speaker: str | None
    hyp_text: str | None
    hyp_speaker: str | None
    hyp_start: float | None = None
    hyp_end: float | None = None
    category: DiffCategory = DiffCategory.AMBIGUOUS
    wer: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenise(text: str) -> list[str]:
    return _normalise(text).split()


def _only_filler_diff(words_a: list[str], words_b: list[str]) -> bool:
    """Return True if the symmetric difference between the two word lists
    consists entirely of filler tokens."""
    set_a = set(words_a)
    set_b = set(words_b)
    diff = set_a.symmetric_difference(set_b)
    return bool(diff) and diff.issubset(FILLER_TOKENS)


def _only_readability_diff(ref: str, hyp: str) -> bool:
    """Return True if normalised texts are identical (only punct/caps differ)."""
    return _normalise(ref) == _normalise(hyp)


# ---------------------------------------------------------------------------
# Word Error Rate (Levenshtein on word tokens)
# ---------------------------------------------------------------------------

def _levenshtein(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) between two word lists.

    Uses standard dynamic programming.  Cost: S=1, I=1, D=1.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i][j - 1],       # insertion
                    dp[i - 1][j],       # deletion
                )

    # Back-track to count S / I / D separately
    i, j = n, m
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    return subs, ins, dels


def compute_wer(ref_words: list[str], hyp_words: list[str]) -> float:
    """Standard WER = (S + I + D) / N where N = len(ref_words)."""
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    subs, ins, dels = _levenshtein(ref_words, hyp_words)
    return round((subs + ins + dels) / len(ref_words), 4)


# ---------------------------------------------------------------------------
# Segment alignment (IoU-based greedy matching)
# ---------------------------------------------------------------------------

def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection-over-Union for two time intervals."""
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(0.0, inter_end - inter_start)
    if intersection == 0.0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0


def align_segments(
    doc_a: Any,
    doc_b: Any,
    iou_threshold: float = 0.4,
) -> list[AlignedPair]:
    """Align segments from doc_a (reference) and doc_b (hypothesis) by IoU.

    Each segment in doc_a is paired with the highest-IoU segment in doc_b
    that exceeds *iou_threshold* and has not already been claimed.  Unmatched
    doc_a segments become DELETION pairs; unmatched doc_b segments become
    INSERTION pairs.
    """
    segs_a = doc_a.segments
    segs_b = list(doc_b.segments)
    matched_b: set[int] = set()
    pairs: list[AlignedPair] = []

    for idx_a, seg_a in enumerate(segs_a):
        best_iou = 0.0
        best_b_idx = -1
        for idx_b, seg_b in enumerate(segs_b):
            if idx_b in matched_b:
                continue
            score = _iou(seg_a.start, seg_a.end, seg_b.start, seg_b.end)
            if score > best_iou:
                best_iou = score
                best_b_idx = idx_b

        if best_b_idx >= 0 and best_iou >= iou_threshold:
            matched_b.add(best_b_idx)
            seg_b = segs_b[best_b_idx]
            pairs.append(
                AlignedPair(
                    pair_id=f"pair_{idx_a:04d}",
                    ref_start=seg_a.start,
                    ref_end=seg_a.end,
                    ref_text=seg_a.text,
                    ref_speaker=seg_a.speaker,
                    hyp_text=seg_b.text,
                    hyp_speaker=seg_b.speaker,
                    hyp_start=seg_b.start,
                    hyp_end=seg_b.end,
                    metadata={"iou": round(best_iou, 4)},
                )
            )
        else:
            pairs.append(
                AlignedPair(
                    pair_id=f"pair_{idx_a:04d}",
                    ref_start=seg_a.start,
                    ref_end=seg_a.end,
                    ref_text=seg_a.text,
                    ref_speaker=seg_a.speaker,
                    hyp_text=None,
                    hyp_speaker=None,
                    category=DiffCategory.DELETION,
                    metadata={"iou": 0.0},
                )
            )

    # Remaining unmatched B segments → INSERTION
    for idx_b, seg_b in enumerate(segs_b):
        if idx_b not in matched_b:
            pairs.append(
                AlignedPair(
                    pair_id=f"ins_{idx_b:04d}",
                    ref_start=seg_b.start,
                    ref_end=seg_b.end,
                    ref_text="",
                    ref_speaker=None,
                    hyp_text=seg_b.text,
                    hyp_speaker=seg_b.speaker,
                    hyp_start=seg_b.start,
                    hyp_end=seg_b.end,
                    category=DiffCategory.INSERTION,
                    metadata={"iou": 0.0},
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# Rule-based categorisation
# ---------------------------------------------------------------------------

def categorise_pair(pair: AlignedPair) -> AlignedPair:
    """Assign a DiffCategory to an aligned pair in-place.  Returns the pair.

    Already-assigned INSERTION / DELETION are passed through unchanged.
    """
    if pair.category in (DiffCategory.INSERTION, DiffCategory.DELETION):
        return pair

    ref = pair.ref_text or ""
    hyp = pair.hyp_text or ""

    # 1. Only punct / caps differ → READABILITY
    if _only_readability_diff(ref, hyp):
        norm_ref = _normalise(ref)
        norm_hyp = _normalise(hyp)
        if norm_ref == norm_hyp:
            # Identical normalised text
            if pair.ref_speaker and pair.hyp_speaker and pair.ref_speaker != pair.hyp_speaker:
                pair.category = DiffCategory.SPEAKER_MISMATCH
            elif ref == hyp:
                pair.category = DiffCategory.MATCH
            else:
                pair.category = DiffCategory.READABILITY
            pair.wer = 0.0
            return pair

    ref_words = _tokenise(ref)
    hyp_words = _tokenise(hyp)

    # 2. Check speaker mismatch on otherwise-identical text
    if ref_words == hyp_words:
        if pair.ref_speaker and pair.hyp_speaker and pair.ref_speaker != pair.hyp_speaker:
            pair.category = DiffCategory.SPEAKER_MISMATCH
        else:
            pair.category = DiffCategory.MATCH
        pair.wer = 0.0
        return pair

    # 3. Filler-word-only difference
    if _only_filler_diff(ref_words, hyp_words):
        pair.category = DiffCategory.FILLER_WORD
        pair.wer = compute_wer(ref_words, hyp_words)
        return pair

    # 4. High-confidence small edit → SUBSTITUTION
    wer = compute_wer(ref_words, hyp_words)
    pair.wer = wer
    if wer <= 0.30:
        pair.category = DiffCategory.SUBSTITUTION
        return pair

    # 5. Fall through → AMBIGUOUS (LLM will resolve)
    pair.category = DiffCategory.AMBIGUOUS
    return pair


def categorise_all(pairs: list[AlignedPair]) -> list[AlignedPair]:
    """Apply rule-based categorisation to every pair."""
    return [categorise_pair(pair) for pair in pairs]


# ---------------------------------------------------------------------------
# Document-level summary
# ---------------------------------------------------------------------------

def compute_document_summary(
    diffs: list[AlignedPair],
    label_a: str = "reference",
    label_b: str = "hypothesis",
) -> dict[str, Any]:
    """Compute per-category counts, overall WER, and per-speaker WER."""
    counts: dict[str, int] = {cat.value: 0 for cat in DiffCategory}
    total_ref_words = 0
    total_subs = total_ins = total_dels = 0

    for diff in diffs:
        counts[diff.category.value] += 1
        if diff.category not in (DiffCategory.INSERTION, DiffCategory.DELETION, DiffCategory.MATCH):
            ref_words = _tokenise(diff.ref_text or "")
            hyp_words = _tokenise(diff.hyp_text or "")
            s, i, d = _levenshtein(ref_words, hyp_words)
            total_ref_words += len(ref_words)
            total_subs += s
            total_ins += i
            total_dels += d
        elif diff.category == DiffCategory.DELETION:
            ref_words = _tokenise(diff.ref_text or "")
            total_ref_words += len(ref_words)
            total_dels += len(ref_words)
        elif diff.category == DiffCategory.INSERTION:
            hyp_words = _tokenise(diff.hyp_text or "")
            total_ins += len(hyp_words)

    overall_wer = (
        round((total_subs + total_ins + total_dels) / total_ref_words, 4)
        if total_ref_words > 0
        else 0.0
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "total_pairs": len(diffs),
        "category_counts": counts,
        "overall_wer": overall_wer,
        "word_errors": {
            "substitutions": total_subs,
            "insertions": total_ins,
            "deletions": total_dels,
            "ref_word_count": total_ref_words,
        },
    }


# ---------------------------------------------------------------------------
# Markdown report renderer
# ---------------------------------------------------------------------------

def render_report_markdown(
    diffs: list[AlignedPair],
    summary: dict[str, Any],
    label_a: str = "reference",
    label_b: str = "hypothesis",
) -> str:
    lines = [
        "# ASR Transcript Comparison Report",
        "",
        f"- **Reference** (`{label_a}`)",
        f"- **Hypothesis** (`{label_b}`)",
        f"- **Total aligned pairs**: {summary['total_pairs']}",
        f"- **Overall WER**: {summary['overall_wer']:.1%}",
        "",
        "## Category Summary",
        "",
        "| Category | Count |",
        "|---|---|",
    ]
    for cat, count in summary["category_counts"].items():
        if count > 0:
            lines.append(f"| {cat} | {count} |")

    lines += [
        "",
        "## Diff Details",
        "",
    ]

    category_emoji = {
        "match": "✓",
        "substitution": "~",
        "insertion": "+",
        "deletion": "-",
        "speaker_mismatch": "S",
        "readability": "R",
        "filler_word": "F",
        "ambiguous": "?",
    }

    for diff in diffs:
        if diff.category == DiffCategory.MATCH:
            continue
        emoji = category_emoji.get(diff.category.value, "?")
        speaker_info = ""
        if diff.ref_speaker or diff.hyp_speaker:
            speaker_info = f" | spk: {diff.ref_speaker} → {diff.hyp_speaker}"
        wer_info = f" | WER: {diff.wer:.0%}" if diff.wer > 0 else ""
        lines.append(
            f"**[{emoji}] {diff.pair_id}** "
            f"[{diff.ref_start:.2f}–{diff.ref_end:.2f}]{speaker_info}{wer_info}"
        )
        lines.append(f"- REF: `{diff.ref_text}`")
        lines.append(f"- HYP: `{diff.hyp_text}`")
        lines.append("")

    return "\n".join(lines)
