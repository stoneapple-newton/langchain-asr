# Summary Agents Feature

## Feature Summary

The summary agents track demonstrates meeting transcript summarization across
one-shot prompting, structured output, LangChain agents, LangGraph workflows,
RAG, memory, Deep Agents, and evaluation.

## Business Outcome

Developers can compare summary implementation approaches and select the
simplest layer that satisfies output quality, context handling, memory, and
evaluation needs.

## Scope

Primary folder:

- `deep_research/summary_agents/`

Important folders:

- `shared/`
- `stage_01_one_shot/`
- `stage_02_langchain_agents/`
- `stage_03_langgraph/`
- `stage_04_rag_enhanced/`
- `stage_05_memory_persistent/`
- `stage_06_deep_agents/`
- `stage_07_evaluation/`

## Current Architecture

`shared/transcript_loader.py` defines transcript dataclasses and formatting
utilities:

- `TranscriptSegment`
- `MeetingTranscript`
- `load_transcript()`
- `format_transcript_for_llm()`

`shared/evaluation.py` defines summary quality scoring and datasets:

- `SummaryQualityScore`
- `EvaluationResult`
- `SummaryEvaluator`
- `EvaluationExample`
- `EvaluationDataset`
- `create_evaluation_dataset()`

The stages then layer more framework capability on top of those helpers.

## Key APIs and Data Contracts

- `MeetingTranscript.language`
- `MeetingTranscript.duration`
- `MeetingTranscript.segments`
- `MeetingTranscript.speakers`
- `MeetingTranscript.total_words`
- `MeetingTranscript.average_confidence`
- Transcript formats: `speaker_turns`, `paragraph`, `dialogue`.
- Evaluation metrics: ROUGE if installed, LLM completeness, accuracy,
  conciseness, action items, and overall score.

## Dependencies

- Shared config and structured output helper.
- LangChain prompts and parsers.
- LangGraph for stateful summaries.
- RAG dependencies for context-enhanced summaries.
- Optional `rouge-score` package for overlap metrics.
- Optional Deep Agents package.

## Nonfunctional Requirements

- Summary outputs should be grounded in transcript content.
- Evaluation should be able to run without ROUGE installed.
- Long transcript formatting should support truncation.
- Memory examples should make thread or store boundaries explicit.

## Acceptance Criteria

- New summarization examples use `load_transcript()` and
  `format_transcript_for_llm()` where possible.
- Structured summaries use Pydantic models or another explicit schema.
- Evaluation examples include reference summaries and metadata.
- LLM-as-judge failures are captured in metadata instead of aborting the whole
  evaluation.

## Risks and Open Questions

- ROUGE is optional but not declared in `pyproject.toml`; code handles missing
  imports by skipping overlap metrics.
- Some stages print outputs directly and are less reusable than shared helpers.
- LLM-as-judge scores are subjective and should not be treated as the only
  quality signal.

## Development Guidance

Use this track to prototype summary variants. Promote only stable transcript
loading, formatting, scoring, and dataset logic to `shared/`. Add deterministic
tests for formatting and dataset behavior before relying on LLM evaluation.
