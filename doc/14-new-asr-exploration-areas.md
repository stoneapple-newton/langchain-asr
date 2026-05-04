# New ASR Exploration Areas

## Epic Summary

This document identifies three unexplored areas in the ASR workspace and
proposes concrete feature tracks for each. The existing tracks cover transcript
cleanup, diarization repair, PII redaction, translation, summarization, medical
verification, and comparison. The gaps are:

1. **Sentiment and Emotion Analysis** — per-speaker emotional arc and tone profiling.
2. **Topic Segmentation** — dividing a long transcript into coherent topic blocks.
3. **Action Item Extraction** — mining tasks, decisions, and open questions from
   meeting transcripts.

Each area follows the same staged pattern used by existing tracks: one-shot
baseline, then a LangGraph agent with structured state and deterministic helpers.

---

## 14a — Sentiment and Emotion Analysis

### Business Outcome

Customer-service and call-centre operations need per-turn sentiment labels and
per-speaker tone summaries to flag at-risk interactions, coach agents, and
measure satisfaction without reading every call.

### Scope

In scope:
- Deterministic keyword-based baseline scoring per segment.
- LangGraph agent that refines labels using an LLM and aggregates by speaker.
- Per-speaker sentiment profile: dominant label, positive/negative ratio, filler count.
- Markdown and JSON output.
- Unit tests for all deterministic helpers.

Out of scope:
- Audio-level prosody or pitch analysis.
- Real-time streaming scoring.
- Integration with CRM or ticketing systems.

### Architecture

```
deep_research/asr_sentiment_analysis/
├── shared/
│   └── sentiment_utils.py   # deterministic scoring, aggregation, rendering
├── sample_data/
│   └── call_center_sample.json
├── stage_01_basics/
│   └── 01_one_shot_sentiment.py
├── stage_02_langgraph/
│   └── 01_sentiment_agent.py
└── tests/
    └── test_sentiment_utils.py
```

### LangGraph Graph

```
START → load → score_naive → llm_refine → aggregate → save → END
```

- `load`: reads transcript via `load_transcript()`.
- `score_naive`: runs `score_segment_naive()` on all segments (deterministic).
- `llm_refine`: sends chunks of naive labels to the LLM; LLM returns
  `SentimentRefinement` (label, intensity 0-1, key phrases).
- `aggregate`: builds per-speaker `SpeakerSentimentProfile`.
- `save`: writes JSON sidecar and Markdown report.

### Key APIs

```python
# sentiment_utils.py
SegmentSentimentLabel(segment_id, speaker, start, end, text,
                      positive_count, negative_count, filler_count, naive_label)

SpeakerSentimentProfile(speaker, segment_count,
                        positive_ratio, negative_ratio,
                        dominant_label, total_fillers)

score_segment_naive(segment_id, speaker, start, end, text) -> SegmentSentimentLabel
aggregate_speaker_profiles(labels) -> dict[str, SpeakerSentimentProfile]
render_sentiment_report(labels, profiles) -> str
```

### LLM Output Contract

```python
class SentimentRefinement(BaseModel):
    segment_id: str
    label: Literal["positive", "negative", "neutral", "mixed"]
    intensity: float          # 0.0 – 1.0
    key_phrases: list[str]
```

### Dependencies

- `config.create_chat_model()` for LLM access.
- `asr-v2/shared/transcript_utils.py` for `load_transcript`, `TranscriptSegment`.
- No new third-party packages required.

### Nonfunctional Requirements

- `score_segment_naive` must be deterministic and require no LLM.
- Tests must not make real model calls.
- Output written under `deep_research/asr_sentiment_analysis/outputs/`.

### Acceptance Criteria

- Naive scorer correctly classifies a "great" segment as positive and a
  "terrible" segment as negative.
- LangGraph compiles without import-time LLM calls (use lazy factory).
- A Markdown report renders per-speaker summary and per-segment labels.

---

## 14b — Topic Segmentation

### Business Outcome

Long meetings, lectures, and interviews contain multiple distinct topics. Topic
segmentation enables automatic chapter generation, searchable indexes, and more
precise summarisation.

### Scope

In scope:
- Deterministic boundary signal detection (shift phrases, speaker changes, gaps).
- LangGraph agent that uses an LLM to name and summarise each detected topic block.
- JSON and Markdown output with titles, time ranges, and summaries.
- Unit tests for deterministic helpers.

Out of scope:
- Embedding-based semantic similarity for boundary detection.
- Real-time or streaming segmentation.
- Automatic slide-deck generation.

### Architecture

```
deep_research/asr_topic_segmentation/
├── shared/
│   └── segmentation_utils.py
├── stage_01_basics/
│   └── 01_one_shot_segmentation.py
├── stage_02_langgraph/
│   └── 01_topic_segmentation_agent.py
└── tests/
    └── test_segmentation_utils.py
```

### LangGraph Graph

```
START → load → detect_signals → group_topics → llm_label → save → END
```

- `load`: `load_transcript()`.
- `detect_signals`: `detect_boundary_signals()` — scores each segment.
- `group_topics`: `group_into_topic_segments()` — splits at high-score boundaries.
- `llm_label`: sends each topic's raw text to LLM; LLM returns `TopicLabel`
  (title, one-sentence summary).
- `save`: writes JSON and Markdown.

### Key APIs

```python
# segmentation_utils.py
TopicBoundarySignal(segment_id, index, start,
                    has_shift_phrase, has_question,
                    speaker_changed, gap_seconds, signal_score)

TopicSegment(topic_id, title, start_time, end_time,
             segment_ids, summary)

detect_boundary_signals(segments) -> list[TopicBoundarySignal]
group_into_topic_segments(segments, signals, threshold=0.5) -> list[TopicSegment]
render_segmentation_report(topic_segments) -> str
```

### LLM Output Contract

```python
class TopicLabel(BaseModel):
    topic_id: str
    title: str          # short label, ≤ 8 words
    summary: str        # one sentence
```

### Dependencies

- `config.create_chat_model()`.
- `asr-v2/shared/transcript_utils.py`.

### Nonfunctional Requirements

- `detect_boundary_signals` must return a result for every input segment.
- LLM is called only in `llm_label`; all other nodes are deterministic.
- Outputs under `deep_research/asr_topic_segmentation/outputs/`.

### Acceptance Criteria

- Boundary detector gives a score > 0 for a segment containing "next" or a gap
  longer than 2 seconds.
- LangGraph compiles without import-time LLM calls.
- Markdown report lists each topic with title, time range, and summary.

---

## 14c — Action Item Extraction

### Business Outcome

Meeting recordings are only valuable if follow-through happens. Extracting
tasks, decisions, open questions, and commitments from transcripts saves manual
review time and feeds project management tools.

### Scope

In scope:
- Regex-based candidate detection (action verbs, decision phrases, question patterns).
- LangGraph agent that classifies candidates with an LLM and enriches with
  owner and due-context fields.
- Structured JSON output and Markdown checklist.
- Unit tests for deterministic helpers.

Out of scope:
- Integration with Jira, Asana, or similar tools.
- Speaker de-identification before extraction.
- Automatic reminder scheduling.

### Architecture

```
deep_research/asr_action_items/
├── shared/
│   └── action_item_utils.py
├── stage_01_basics/
│   └── 01_one_shot_extraction.py
├── stage_02_langgraph/
│   └── 01_action_item_agent.py
└── tests/
    └── test_action_item_utils.py
```

### LangGraph Graph

```
START → load → detect_candidates → llm_classify → save → END
```

- `load`: `load_transcript()`.
- `detect_candidates`: `detect_candidates()` — regex pass over segments,
  returns `CandidateItem` list.
- `llm_classify`: sends formatted candidates to LLM; LLM returns
  `ActionItemList` with typed `ActionItem` objects.
- `save`: writes JSON and Markdown report.

### Key APIs

```python
# action_item_utils.py
CandidateItem(segment_id, speaker, start, end, text,
              candidate_types, match_count)

ActionItem(item_id, item_type, text, raw_segment_text,
           segment_id, owner, due_context, confidence)

detect_candidates(segments) -> list[CandidateItem]
format_candidates_for_llm(candidates) -> str
render_action_items_report(items) -> str
```

### LLM Output Contract

```python
class ActionItem(BaseModel):
    item_id: str
    item_type: Literal["task", "decision", "open_question", "commitment"]
    text: str
    segment_id: str
    owner: str | None
    due_context: str | None
    confidence: float
```

### Dependencies

- `config.create_chat_model()`.
- `asr-v2/shared/transcript_utils.py`.

### Nonfunctional Requirements

- `detect_candidates` must be deterministic and return candidates sorted by
  match count descending.
- LLM is called only in `llm_classify`.
- Outputs under `deep_research/asr_action_items/outputs/`.

### Acceptance Criteria

- Candidate detector flags "will send the report by Friday" as a `task` candidate.
- Candidate detector flags "we agreed to postpone the launch" as a `decision`.
- LangGraph compiles without import-time LLM calls.
- Markdown report groups items by type (tasks, decisions, open questions,
  commitments).

---

## Shared Patterns Across All Three Tracks

| Pattern | Detail |
|---|---|
| Transcript loading | `load_transcript()` from `asr-v2/shared/transcript_utils.py` |
| LLM access | `create_chat_model()` called inside node functions, not at import |
| State schema | `TypedDict` with explicit fields |
| Output location | Track-local `outputs/` folder |
| Tests | Deterministic helpers tested without model calls |
| Samples | Track-local `sample_data/` folder |

## Extension Points

- **Sentiment**: add emotion categories (joy, anger, fear, surprise) using a
  finer-grained LLM schema; add trend chart generation.
- **Topic segmentation**: add embedding-based cosine similarity as an
  alternative boundary signal; add a RAG retriever to link topics to a
  knowledge base.
- **Action items**: add confidence thresholding UI; export to CSV for import
  into project management tools; add a follow-up tracker that compares two
  consecutive meeting transcripts.
