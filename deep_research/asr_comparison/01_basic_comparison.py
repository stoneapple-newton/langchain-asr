"""
01_basic_comparison.py
======================
CONCEPT: Rule-based comparison of two ASR transcripts — no LLM, no LangGraph.

Demonstrates the core pipeline:
  1. Load both transcripts using the shared transcript_utils loader
  2. Align segments by time-window IoU
  3. Categorise each pair with deterministic rules
  4. Print a summary table and diff details

Run this file:
  uv run deep_research/asr_comparison/01_basic_comparison.py
"""

import sys
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ASR_V2_ROOT = Path(__file__).resolve().parents[1] / "asr-v2"
REPO_ROOT = Path(__file__).resolve().parents[2]

for p in (REPO_ROOT, ASR_V2_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# ── imports ───────────────────────────────────────────────────────────────────
from shared.transcript_utils import load_transcript

from shared.comparison_utils import (  # both live in asr-v2/shared
    DiffCategory,
    align_segments,
    categorise_all,
    compute_document_summary,
    render_report_markdown,
)

# ── paths ─────────────────────────────────────────────────────────────────────
SAMPLE_A = ROOT / "sample_data" / "transcript_a.json"
SAMPLE_B = ROOT / "sample_data" / "transcript_b.json"
LABEL_A = "raw_asr"
LABEL_B = "enhanced_asr"

# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  ASR Transcript Comparison — Rule-based")
    print("=" * 60)

    # 1. Load
    doc_a = load_transcript(SAMPLE_A)
    doc_b = load_transcript(SAMPLE_B)
    print(f"\nLoaded '{LABEL_A}': {len(doc_a.segments)} segments")
    print(f"Loaded '{LABEL_B}': {len(doc_b.segments)} segments")

    # 2. Align
    pairs = align_segments(doc_a, doc_b, iou_threshold=0.4)
    print(f"\nAligned pairs: {len(pairs)}")

    # 3. Categorise
    diffs = categorise_all(pairs)

    # 4. Summary
    summary = compute_document_summary(diffs, LABEL_A, LABEL_B)

    print("\n── Category breakdown ─────────────────────────────────")
    for cat, count in summary["category_counts"].items():
        if count > 0:
            print(f"  {cat:<20}  {count}")
    print(f"\n  Overall WER: {summary['overall_wer']:.1%}")
    errs = summary["word_errors"]
    print(f"  Substitutions: {errs['substitutions']}  "
          f"Insertions: {errs['insertions']}  "
          f"Deletions: {errs['deletions']}  "
          f"Ref words: {errs['ref_word_count']}")

    print("\n── Per-pair details ────────────────────────────────────")
    for diff in diffs:
        if diff.category == DiffCategory.MATCH:
            continue
        wer_str = f"  WER={diff.wer:.0%}" if diff.wer > 0 else ""
        spk_str = ""
        if diff.ref_speaker != diff.hyp_speaker:
            spk_str = f"  spk: {diff.ref_speaker} → {diff.hyp_speaker}"
        print(f"\n  [{diff.category.value.upper()}] {diff.pair_id} "
              f"[{diff.ref_start:.2f}–{diff.ref_end:.2f}]{wer_str}{spk_str}")
        print(f"    REF: {diff.ref_text!r}")
        print(f"    HYP: {diff.hyp_text!r}")

    # 5. Save markdown report
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    report_md = render_report_markdown(diffs, summary, LABEL_A, LABEL_B)
    report_path = output_dir / "comparison_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
