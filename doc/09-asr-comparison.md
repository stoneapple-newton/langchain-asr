# ASR Comparison Feature

## Feature Summary

The ASR comparison feature aligns two transcript outputs, classifies segment
differences, computes WER-style metrics, and produces Markdown or JSON reports.
It has a deterministic script and a LangGraph script with optional LLM
resolution of ambiguous differences.

## Business Outcome

Developers can compare ASR systems, cleanup variants, or before-and-after
transcripts with enough detail to understand what changed and whether changes
are readability-only or substantive.

## Scope

Primary folder:

- `deep_research/asr_comparison/`

Shared dependency:

- `deep_research/asr-v2/shared/comparison_utils.py`

Important files:

- `01_basic_comparison.py`
- `02_langgraph_agent.py`
- `sample_data/*.json`
- `outputs/*.md`
- `tests/test_comparison_utils.py`

## Current Architecture

The feature intentionally reuses ASR-v2 shared modules by adding
`deep_research/asr-v2` to `sys.path`. The deterministic path loads transcripts,
aligns by time-window IoU, categorizes differences, computes document summary
metrics, and writes a report.

The LangGraph path adds explicit nodes:

- `load`
- `align`
- `rule_diff`
- optional `llm_resolve`
- `report`
- `save`

LLM resolution is used only for ambiguous diffs and falls back to substitution
if the model call fails.

## Key APIs and Data Contracts

From `comparison_utils.py`:

- `DiffCategory`
- `AlignedPair`
- `align_segments(doc_a, doc_b, iou_threshold=0.4)`
- `categorise_pair(pair)`
- `categorise_all(pairs)`
- `compute_wer(ref_words, hyp_words)`
- `compute_document_summary(diffs, label_a, label_b)`
- `render_report_markdown(diffs, summary, label_a, label_b)`

Diff categories:

- `match`
- `substitution`
- `insertion`
- `deletion`
- `speaker_mismatch`
- `readability`
- `filler_word`
- `ambiguous`

## Dependencies

- ASR-v2 transcript utilities.
- ASR-v2 comparison utilities.
- Shared config for LangGraph LLM resolution.
- LangGraph for the graph agent.
- Pydantic and LangChain output parsing.

## Nonfunctional Requirements

- Deterministic comparison must run without an LLM.
- LLM resolution should be bounded to ambiguous cases.
- Reports should include both category counts and detailed segment diffs.
- WER calculations should remain deterministic and tested.

## Acceptance Criteria

- Two transcripts can be compared by running `01_basic_comparison.py`.
- Ambiguous differences route to `llm_resolve` in `02_langgraph_agent.py`.
- JSON and Markdown reports are saved under `outputs/`.
- Tests cover normalization, filler detection, WER, IoU alignment,
  categorization, and document summary.

## Risks and Open Questions

- `02_langgraph_agent.py` creates `_llm` and `_chain` at module import time,
  which can fail during import if the configured provider is unavailable.
- The test docstring references `deep_research/asr_comparison/shared`, but the
  actual shared module lives in ASR-v2.
- IoU alignment is greedy. It is simple and explainable, but may not be optimal
  for heavily segmented transcripts.

## Development Guidance

Use `01_basic_comparison.py` and `comparison_utils.py` for deterministic
regression checks. Add LLM resolution only when categories matter more than
repeatability. If comparison becomes a first-class feature, move shared helpers
into a package-safe namespace instead of relying on `sys.path`.
