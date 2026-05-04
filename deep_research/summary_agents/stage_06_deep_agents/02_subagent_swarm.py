"""
Stage 6, File 2: Summary Subagent Swarm
========================================
CONCEPT: Specialized subagents for different summary aspects.

Instead of one agent doing everything, we have specialist subagents:
- ActionItemExtractor: Finds todos and assignments
- DecisionRecorder: Captures key decisions made
- TopicAnalyzer: Identifies discussed topics
- SummaryFormatter: Formats final output

Key concepts:
  - SubAgentMiddleware delegation
  - Specialized subagents with focused prompts
  - Supervisor coordination

Run this file (requires deepagents package):
  uv add deepagents
  uv run deep_research/summary_agents/stage_06_deep_agents/02_subagent_swarm.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain.tools import tool


# ---------------------------------------------------------------------------
# Subagent Blueprints
# ---------------------------------------------------------------------------

SUBAGENT_BLUEPRINTS = [
    {
        "name": "action_item_extractor",
        "description": "Specialist in extracting action items, todos, and task assignments from meeting transcripts. Identifies owners and deadlines.",
        "system_prompt": """You are an Action Item Extraction Specialist.

Your sole focus is finding tasks, todos, and action items in meeting transcripts.

For each action item you find:
- Task description (clear and specific)
- Assignee (who is responsible)
- Deadline (if mentioned)
- Priority (if indicated)

Output format:
ACTION ITEMS:
1. [Task] - Owner: [Name] - Due: [Date/Timeframe] - Priority: [High/Medium/Low]

Be thorough but concise. Don't miss implicit action items ("I'll look into that").
""",
    },
    {
        "name": "decision_recorder",
        "description": "Specialist in identifying and documenting decisions made during meetings. Captures context and implications.",
        "system_prompt": """You are a Decision Recording Specialist.

Your sole focus is identifying decisions made during meetings.

Look for:
- Explicit decisions ("We decided to...", "Let's go with...")
- Implicit decisions (consensus reached, direction chosen)
- Technical choices (libraries, architectures, approaches)
- Process changes (workflow updates, policy changes)

For each decision:
- What was decided
- Context/why it was made
- Any dissent or alternatives considered

Output format:
DECISIONS:
1. [Decision] - Context: [Why] - Alternatives: [Other options considered]
""",
    },
    {
        "name": "topic_analyzer",
        "description": "Specialist in identifying main topics, themes, and discussion threads in meeting transcripts.",
        "system_prompt": """You are a Topic Analysis Specialist.

Your sole focus is identifying the main topics and themes discussed.

Categorize discussion into:
- Primary topics (main agenda items)
- Secondary topics (briefly mentioned)
- Cross-cutting concerns (applies to multiple topics)

For each topic:
- Topic name
- Time spent/discussion depth
- Key stakeholders
- Outcome

Output format:
TOPICS:
1. [Topic] - Depth: [Brief/Moderate/Deep] - Stakeholders: [Names] - Outcome: [Result]
""",
    },
    {
        "name": "summary_formatter",
        "description": "Specialist in formatting meeting information into professional, readable summary documents.",
        "system_prompt": """You are a Summary Formatting Specialist.

Your job is to take extracted information and format it into a professional meeting summary.

Input you'll receive:
- Action items (from action_item_extractor)
- Decisions (from decision_recorder)
- Topics (from topic_analyzer)

Your output format:
# Meeting Summary

## Overview
(1-2 sentence summary of the meeting)

## Topics Discussed
(Formatted list with context)

## Decisions Made
(Clear, formatted decisions)

## Action Items
(Formatted with owners and deadlines)

## Next Steps
(High-level summary of what's next)

Use professional language. Ensure formatting is clean and readable.
""",
    },
]


# ---------------------------------------------------------------------------
# Supervisor System Prompt
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are a Meeting Summary Supervisor coordinating a team of specialist subagents.

Your team:
1. action_item_extractor - Finds todos and tasks
2. decision_recorder - Captures key decisions
3. topic_analyzer - Identifies main topics
4. summary_formatter - Formats final output

WORKFLOW:
1. Load the transcript
2. Delegate to action_item_extractor, decision_recorder, and topic_analyzer IN PARALLEL
3. Collect their outputs
4. Delegate to summary_formatter with all collected information
5. Review and deliver final summary

Use the `task` tool to delegate to subagents. Be clear about what you need from each.
"""


# ---------------------------------------------------------------------------
# Build Swarm
# ---------------------------------------------------------------------------

def build_summary_swarm():
    """Build the supervisor with specialist subagents."""
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as e:
        raise RuntimeError("deepagents not installed. Run: uv add deepagents") from e
    
    from config import create_chat_model
    
    model = create_chat_model(profile="asr_v2", temperature=0, max_tokens=4096)
    
    return create_deep_agent(
        name="summary-supervisor",
        model=model,
        tools=[],  # Subagents have the specialized tools
        system_prompt=SUPERVISOR_PROMPT,
        subagents=SUBAGENT_BLUEPRINTS,
        backend=FilesystemBackend(root_dir=str(REPO_ROOT), virtual_mode=True),
        checkpointer=MemorySaver(),
    )


# ---------------------------------------------------------------------------
# Demo Documentation
# ---------------------------------------------------------------------------

print("=" * 70)
print("  STAGE 6.2: SUMMARY SUBAGENT SWARM")
print("=" * 70)

print(f"""
This file demonstrates the Subagent Swarm pattern for meeting summarization.

SWARM ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━

                    Supervisor Agent
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Action  │   │ Decision │   │  Topic   │
    │ Extractor│   │ Recorder │   │ Analyzer │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
                 ┌──────────┐
                 │ Summary  │
                 │ Formatter│
                 └──────────┘

SUBAGENT ROLES:
---------------
""")

for i, agent in enumerate(SUBAGENT_BLUEPRINTS, 1):
    print(f"{i}. {agent['name']}")
    print(f"   Purpose: {agent['description']}")
    print()

print("""
WORKFLOW:
---------
1. User sends transcript to Supervisor
2. Supervisor uses `task()` to delegate to 3 extractors in parallel:
   
   ```
   task("action_item_extractor", "Extract all action items from this transcript: ...")
   task("decision_recorder", "Record all decisions from this transcript: ...")
   task("topic_analyzer", "Identify main topics in this transcript: ...")
   ```

3. Supervisor collects results
4. Supervisor delegates to formatter:
   
   ```
   task("summary_formatter", "Format this into a meeting summary. Action items: ... Decisions: ... Topics: ...")
   ```

5. Supervisor reviews and delivers final output

BENEFITS:
---------
  ✅ Specialization: Each agent masters one task
  ✅ Parallelism: Extractors run simultaneously
  ✅ Modularity: Easy to add new specialists
  ✅ Quality: Focused prompts = better results
  ✅ Maintainability: Update one specialist without affecting others

IMPLEMENTATION:
---------------
""")

print("-" * 70)
print("  BUILDING THE SWARM")
print("-" * 70)
print("""
```python
from deepagents import create_deep_agent

# Define subagent blueprints (specialists)
subagents = [
    {
        "name": "action_item_extractor",
        "description": "Extracts action items from transcripts",
        "system_prompt": "You are an action item extraction specialist..."
    },
    # ... more specialists
]

# Create supervisor with subagents
supervisor = create_deep_agent(
    name="summary-supervisor",
    model=model,
    system_prompt="Coordinate the summary team using task()...",
    subagents=subagents,  # ← Delegation targets
    checkpointer=MemorySaver()
)
```
""")

print("-" * 70)
print("  EXAMPLE USAGE")
print("-" * 70)
print("""
User: "Summarize this meeting transcript"

Supervisor:
  [Plans: Delegate to 3 extractors, then formatter]
  
  → task(action_item_extractor): "Find all action items..."
  → task(decision_recorder): "Record all decisions..."
  → task(topic_analyzer): "Identify topics..."
  
  [Collects results]
  
  → task(summary_formatter): "Format: action_items=..., decisions=..., topics=..."
  
  [Receives formatted summary]
  
  → User: "## Meeting Summary\n..."
""")

print("\n" + "=" * 70)
print("  NEXT: Stage 6.3 → Quality Review with Human-in-the-Loop")
print("=" * 70)
