---
name: asr-transcript-comparison
description: "INVOKE THIS SKILL when comparing two ASR transcripts to identify and categorise differences. Covers segment alignment by IoU, rule-based diff categorisation (substitution, insertion, deletion, speaker_mismatch, readability, filler_word), LLM resolution of ambiguous cases, WER computation, and report generation."
---

<overview>
The ASR comparison agent aligns two transcripts by time-window overlap (IoU matching), then classifies every difference into one of seven categories:

| Category | Meaning |
|---|---|
| `match` | Identical text and speaker |
| `readability` | Only punctuation / capitalisation / contractions differ |
| `filler_word` | Only filler tokens (um, uh, hmm, yeah) added or removed |
| `substitution` | One or more words changed (likely transcription error) |
| `speaker_mismatch` | Same words, different speaker attribution |
| `insertion` | Segment exists in hypothesis but has no match in reference |
| `deletion` | Segment exists in reference but has no match in hypothesis |

Rule-based categorisation handles clear cases. Ambiguous diffs (high WER, unclear change type) are batched to an LLM for resolution. A final report is written as both JSON and Markdown with per-category counts and document-level WER.
</overview>

<when-to-use>

| Use this agent when | Use something else when |
|---|---|
| Comparing two ASR engine outputs on the same audio | Comparing meeting summaries (use summary_agents eval) |
| Auditing before/after ASR post-processing | General text diff (no timestamps involved) |
| Measuring WER between a reference and hypothesis | Real-time transcription monitoring |
| Identifying speaker attribution errors | |

</when-to-use>

<diff-categories>

### Decision tree (rule-based, no LLM)

```
aligned pair present?
  NO  → INSERTION (only in hyp) or DELETION (only in ref)
  YES →
    normalise both texts (lowercase, strip punct)
    identical?
      YES → same speaker? → MATCH
              diff speaker? → SPEAKER_MISMATCH
      NO  →
        only filler tokens differ? → FILLER_WORD
        only punct/caps differ?    → READABILITY
        pair WER ≤ 0.30?           → SUBSTITUTION
        else                       → AMBIGUOUS → LLM resolves
```

The LLM resolves AMBIGUOUS to: `substitution | readability | filler_word | speaker_mismatch`.

</diff-categories>

<ex-basic>
<python>
Run the rule-based-only comparison (no LLM, no LangGraph).
```python
# uv run deep_research/asr_comparison/01_basic_comparison.py

# Or call programmatically:
import sys
sys.path.insert(0, "deep_research/asr-v2")
sys.path.insert(0, "deep_research/asr_comparison")

from shared.transcript_utils import load_transcript
from shared.comparison_utils import (
    align_segments,
    categorise_all,
    compute_document_summary,
    render_report_markdown,
)

doc_a = load_transcript("path/to/transcript_a.json")
doc_b = load_transcript("path/to/transcript_b.json")

pairs = align_segments(doc_a, doc_b, iou_threshold=0.4)
diffs = categorise_all(pairs)
summary = compute_document_summary(diffs, label_a="raw", label_b="enhanced")

print(f"Overall WER: {summary['overall_wer']:.1%}")
print(summary["category_counts"])

report_md = render_report_markdown(diffs, summary)
```
</python>
</ex-basic>

<ex-langgraph>
<python>
Run the full LangGraph agent with LLM resolution of ambiguous diffs.
```python
# uv run deep_research/asr_comparison/02_langgraph_agent.py

# Or use the graph directly:
import sys
sys.path.insert(0, "deep_research/asr-v2")
sys.path.insert(0, "deep_research/asr_comparison")

from deep_research.asr_comparison.02_langgraph_agent import build_comparison_graph

app = build_comparison_graph()
result = app.invoke({
    "ref_path": "path/to/transcript_a.json",
    "hyp_path": "path/to/transcript_b.json",
    "label_a": "whisper_v2",
    "label_b": "azure_speech",
    "output_dir": "outputs/",
})

print(result["summary"])
# → {"overall_wer": 0.05, "category_counts": {...}, ...}
```

Graph topology:
```
START → load → align → rule_diff
                            ↓
                route_after_rule_diff
                ↙ (ambiguous)  ↘ (none)
          llm_resolve        report
                ↘               ↙
                report → save → END
```
</python>
</ex-langgraph>

<output-format>

### JSON report structure (`comparison_report.json`)

```json
{
  "summary": {
    "label_a": "raw_asr",
    "label_b": "enhanced_asr",
    "total_pairs": 5,
    "category_counts": {
      "match": 0,
      "substitution": 1,
      "insertion": 0,
      "deletion": 0,
      "speaker_mismatch": 1,
      "readability": 2,
      "filler_word": 1,
      "ambiguous": 0
    },
    "overall_wer": 0.0263,
    "word_errors": {
      "substitutions": 1,
      "insertions": 0,
      "deletions": 0,
      "ref_word_count": 38
    }
  },
  "diffs": [
    {
      "pair_id": "pair_0002",
      "category": "substitution",
      "ref_start": 5.5,
      "ref_end": 7.2,
      "ref_text": "great the weather looks good for the demo on friday",
      "ref_speaker": "SPEAKER_00",
      "hyp_text": "Great, the whether looks good for the demo on Friday.",
      "hyp_speaker": "SPEAKER_00",
      "wer": 0.1,
      "metadata": {"iou": 1.0}
    }
  ]
}
```

### Markdown report

The markdown report (`comparison_report.md`) contains:
- Summary table with category counts and overall WER
- Per-diff detail block for every non-MATCH pair showing REF and HYP text

</output-format>

<sample-data>

Two sample transcripts are provided in `deep_research/asr_comparison/sample_data/`:

| File | Description |
|---|---|
| `transcript_a.json` | Raw ASR output — lowercase, no punctuation, contains filler words, one speaker attribution gap |
| `transcript_b.json` | Enhanced ASR output — capitalised, punctuated; filler words removed; one word substitution (`weather`→`whether`); one speaker mismatch |

Running either script against these files exercises all diff categories.

</sample-data>

<boundaries>

- Alignment requires transcripts with valid `start`/`end` timestamps on every segment
- The IoU threshold (default `0.4`) can be lowered for engines with very different segmentation
- LLM resolution in `02_langgraph_agent.py` uses the `asr_v2` chat profile — configure via `CHAT__PROFILES__ASR_V2__*` env vars
- WER is computed at the segment level; document-level WER is the aggregate across all aligned pairs

</boundaries>
