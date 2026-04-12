"""
Stage 7, File 3: LangSmith Tracing
===================================
CONCEPT: Comprehensive observability for summary agents.

LangSmith provides:
- Automatic tracing of all LLM calls
- Token usage and cost tracking
- Latency breakdown by component
- Dataset evaluation integration
- Human feedback collection

Key concepts:
  - Automatic tracing via env vars
  - @traceable decorator
  - Run metadata and tags
  - Feedback logging

Run this file:
  uv run deep_research/summary_agents/stage_07_evaluation/03_langsmith_tracing.py
"""

import os
from pathlib import Path
import sys
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Setup LangSmith tracing
from config import get_settings

settings = get_settings()
for key, value in settings.tracing.to_langsmith_env(
    default_enabled=True,
    default_project="summary-agents-benchmark"
).items():
    os.environ.setdefault(key, value)

print("=" * 70)
print("  STAGE 7.3: LANGSMITH TRACING")
print("=" * 70)

print("""
LANGSMITH SETUP:
----------------
Environment configured for tracing:
""")

for key in ["LANGSMITH_TRACING", "LANGSMITH_PROJECT"]:
    value = os.environ.get(key, "not set")
    print(f"  {key}={value}")

print("""
With these settings:
  ✅ All LLM calls are automatically traced
  ✅ Token usage is tracked per call
  ✅ Latency is measured for each component
  ✅ Runs appear in LangSmith dashboard

VIEW TRACES:
------------
https://smith.langchain.com

Look for project: summary-agents-benchmark
""")


# ---------------------------------------------------------------------------
# Tracing Example Code
# ---------------------------------------------------------------------------

EXAMPLE_CODE = '''
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from config import create_chat_model

llm = create_chat_model()

# @traceable decorator traces any function
@traceable(name="summarize_meeting", run_type="chain")
def summarize_meeting(transcript: str) -> str:
    """Summarize a meeting transcript."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this meeting concisely."),
        ("human", "{transcript}")
    ])
    
    chain = prompt | llm
    return chain.invoke({"transcript": transcript})

# This call will be traced with full visibility:
# - Prompt template used
# - LLM call with token usage
# - Response content
# - Execution time
result = summarize_meeting("Alice: Let's discuss the API...")

# Add custom metadata
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    tags=["production", "v2.0"],
    metadata={
        "meeting_type": "planning",
        "duration_minutes": 45,
        "speakers": 4,
    }
)

result = summarize_meeting(transcript, config=config)
'''

print("-" * 70)
print("  TRACING EXAMPLE CODE")
print("-" * 70)
print(EXAMPLE_CODE)


# ---------------------------------------------------------------------------
# Tracing Best Practices
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  TRACING BEST PRACTICES")
print("-" * 70)
print("""
1. USE TAGS FOR FILTERING
   ```python
   config = RunnableConfig(tags=["stage_03", "production"])
   ```
   Filter in LangSmith UI by these tags.

2. ADD METADATA FOR CONTEXT
   ```python
   metadata={
       "transcript_length": len(transcript),
       "meeting_duration": 45,
       "num_speakers": 4,
       "model": "gpt-4",
   }
   ```

3. GROUP RELATED RUNS
   ```python
   # Use consistent run_id for multi-step operations
   run_id = str(uuid.uuid4())
   step1.invoke(..., config={"run_id": run_id})
   step2.invoke(..., config={"run_id": run_id})
   ```

4. ADD CUSTOM SPANS
   ```python
   from langsmith import traceable
   
   @traceable(name="custom_processing")
   def my_custom_step(data):
       # This appears as its own span
       return processed_data
   ```

5. LOG FEEDBACK
   ```python
   from langsmith import Client
   
   client = Client()
   client.create_feedback(
       run_id=run_id,
       key="user_rating",
       score=1,  # 1 = thumbs up, 0 = thumbs down
       comment="Good summary but missed one action item"
   )
   ```
""")


# ---------------------------------------------------------------------------
# Tracing Integration with Stages
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  TRACING INTEGRATION BY STAGE")
print("-" * 70)

stages = [
    ("Stage 1: One-Shot", "Basic LCEL chains - automatic tracing"),
    ("Stage 2: Agents", "Tool calls traced individually"),
    ("Stage 3: LangGraph", "Nodes traced as child spans"),
    ("Stage 4: RAG", "Retrieval and generation traced separately"),
    ("Stage 5: Memory", "State changes tracked across checkpoints"),
    ("Stage 6: Deep Agents", "Built-in tracing for all operations"),
]

for stage, description in stages:
    print(f"  {stage:<20} → {description}")

print("""

VISIBILITY PROVIDED:
--------------------
┌─────────────────────────────────────────────────────────────────────┐
│ Run: summarize_meeting_abc123                                       │
├─────────────────────────────────────────────────────────────────────┤
│ Tags: [stage_03, production]                                        │
│ Metadata: {meeting_type: planning, speakers: 4}                     │
├─────────────────────────────────────────────────────────────────────┤
│ 13:45:01 │ prompt_template  │ 2ms                                  │
│ 13:45:01 │ extract_points   │ 1.2s │ 1,250 tokens                  │
│ 13:45:02 │ identify_topics  │ 0.8s │ 890 tokens                    │
│ 13:45:03 │ generate_summary │ 1.5s │ 1,500 tokens                  │
├─────────────────────────────────────────────────────────────────────┤
│ Total: 3.5s │ Tokens: 3,640 │ Cost: $0.15                         │
└─────────────────────────────────────────────────────────────────────┘
""")


# ---------------------------------------------------------------------------
# Debugging with Traces
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  DEBUGGING WORKFLOW")
print("-" * 70)
print("""
WHEN A SUMMARY IS POOR:

1. Find the run in LangSmith UI
   → Filter by tags or time range

2. Inspect the prompt
   → Did the template include all context?
   → Were instructions clear?

3. Check token usage
   → Was context truncated?
   → Is prompt too long?

4. Review tool calls (for agents)
   → Did agent call right tools?
   → Were tool outputs correct?

5. Check latency
   → Which step is slow?
   → Can we parallelize?

6. Compare versions
   → Tag runs with version numbers
   → A/B test prompt changes
""")


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  COST TRACKING")
print("-" * 70)
print("""
LANGSMITH PROVIDES:
- Token usage per run
- Cost estimates per model
- Aggregation by tag/time

MONITORING SETUP:
```python
# Alert on high-cost runs
if run.total_tokens > 10000:
    send_alert(f"Expensive run: {run.id}")

# Daily cost report
daily_cost = sum(r.cost for r in runs if r.date == today)
```
""")

print("\n" + "=" * 70)
print("  NEXT: Stage 7.4 → Compare Quality Analysis")
print("=" * 70)
