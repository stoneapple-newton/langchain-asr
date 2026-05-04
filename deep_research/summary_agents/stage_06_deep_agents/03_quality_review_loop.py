"""
Stage 6, File 3: Quality Review with HITL
==========================================
CONCEPT: Human-in-the-loop approval for summary quality gates.

Not all summaries should go out automatically. This workflow:
1. Generates summary
2. Presents for human review
3. Waits for approval or feedback
4. Revises if needed
5. Proceeds only after approval

Key concepts:
  - Interrupt for human approval
  - Command(resume=...) to continue
  - Quality gates in production

Run this file (requires deepagents package):
  uv add deepagents
  uv run deep_research/summary_agents/stage_06_deep_agents/03_quality_review_loop.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Quality Review Configuration
# ---------------------------------------------------------------------------

QUALITY_THRESHOLD = 7  # Minimum quality score (1-10)

QUALITY_CHECK_PROMPT = """You are a quality assurance specialist for meeting summaries.

Review this generated summary against the original transcript and assess:

1. COMPLETENESS (1-10): Are all key points covered?
2. ACCURACY (1-10): Is the information factually correct?
3. CLARITY (1-10): Is it easy to understand?
4. ACTIONABILITY (1-10): Are action items clear with owners?

Provide:
- Overall score (1-10)
- Whether it passes quality gate (score >= 7)
- Specific feedback if improvements needed
"""


# ---------------------------------------------------------------------------
# Deep Agent with HITL
# ---------------------------------------------------------------------------

def build_quality_aware_agent():
    """Build agent with quality review and human approval."""
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as e:
        raise RuntimeError("deepagents not installed") from e
    
    from config import create_chat_model
    from langchain.tools import tool
    
    @tool
    def assess_quality(summary: str, transcript: str) -> str:
        """Assess summary quality against transcript."""
        # In production, would use LLM with QUALITY_CHECK_PROMPT
        # For demo, return simulated assessment
        return f"""Quality Assessment:
- Completeness: 8/10
- Accuracy: 9/10
- Clarity: 8/10
- Actionability: 7/10
- OVERALL: 8/10
- PASSES GATE: Yes
- Feedback: Good summary, could improve action item specificity."""
    
    model = create_chat_model(profile="asr_v2", temperature=0, max_tokens=4096)
    
    return create_deep_agent(
        name="quality-gate-agent",
        model=model,
        tools=[assess_quality],
        system_prompt="""You are a meeting summary agent with quality gates.

WORKFLOW:
1. Generate meeting summary
2. Run quality assessment
3. If quality < 7: automatically revise
4. If quality >= 7: present for human approval
5. Wait for human feedback via interrupt
6. Revise based on feedback if needed
7. Finalize only after human approval

Use interrupt_on to pause for human review.
Always explain quality scores to the user.
""",
        backend=FilesystemBackend(root_dir=str(REPO_ROOT), virtual_mode=True),
        checkpointer=MemorySaver(),
        interrupt_on={"write_file": True},  # Pause before saving
    )


# ---------------------------------------------------------------------------
# Human-in-the-Loop Workflow Documentation
# ---------------------------------------------------------------------------

print("=" * 70)
print("  STAGE 6.3: QUALITY REVIEW WITH HITL")
print("=" * 70)

print("""
This file demonstrates human-in-the-loop approval for meeting summaries.

QUALITY GATE WORKFLOW:
━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────┐
│   Generate  │
│   Summary   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Quality   │
│   Check     │
└──────┬──────┘
       │
       ├───── Score < 7 ───→ Revise ───┐
       │                                │
       └───── Score >= 7 ───→ Present ──┤
       │                                │
       ▼                                │
┌─────────────┐                         │
│ Human Review│◄────────────────────────┘
│  (interrupt)│
└──────┬──────┘
       │
       ├───── Approve ─────→ Finalize
       │
       └───── Feedback ────→ Revise ───→ Human Review

HUMAN INTERACTION POINTS:
-------------------------
1. Quality Review: "Summary scored 8/10. Approve or provide feedback?"
2. Revision Review: "Revised summary ready. Approve now?"
3. Final Approval: Save to filesystem

INTERRUPT MECHANISM:
--------------------
```python
agent = create_deep_agent(
    ...
    checkpointer=MemorySaver(),  # Required for interrupts
    interrupt_on={"write_file": True}  # Pause before file operations
)

# First invocation runs until interrupt
result = agent.invoke({...}, config)
# State: waiting for human input

# Resume with human feedback
from langgraph.types import Command
result = agent.invoke(
    Command(resume="Add more detail about the API decisions"),
    config
)
```

QUALITY CRITERIA:
-----------------
┌─────────────────┬─────────┬──────────────────────────────────────┐
│ Dimension       │ Weight  │ Criteria                             │
├─────────────────┼─────────┼──────────────────────────────────────┤
│ Completeness    │ 25%     │ All key points covered               │
│ Accuracy        │ 30%     │ Facts match transcript               │
│ Clarity         │ 25%     │ Easy to understand                   │
│ Actionability   │ 20%     │ Clear action items with owners       │
└─────────────────┴─────────┴──────────────────────────────────────┘

Minimum passing score: 7/10
""")

print("-" * 70)
print("  IMPLEMENTATION PATTERN")
print("-" * 70)
print("""
```python
from langgraph.types import Command

# Configuration with thread_id for persistence
config = {"configurable": {"thread_id": "review-001"}}

# Step 1: Initial run (generates summary, runs quality check)
result = agent.invoke({
    "messages": [{"role": "user", "content": "Summarize this transcript..."}]
}, config)

# If interrupted, result shows what tool would be called
if result.get("__interrupt__"):
    print("Agent paused for human review:")
    print(result["summary"])
    print("\nQuality: 8/10 - Passes gate")
    
    # Get human input
    human_feedback = input("Approve (y) or provide feedback: ")
    
    if human_feedback.lower() == 'y':
        # Resume with approval
        final = agent.invoke(Command(resume="APPROVED"), config)
    else:
        # Resume with feedback for revision
        final = agent.invoke(Command(resume=human_feedback), config)
```
""")

print("-" * 70)
print("  BENEFITS")
print("-" * 70)
print("""
  ✅ Quality assurance before delivery
  ✅ Human oversight for important summaries
  ✅ Feedback loop improves summary quality
  ✅ Audit trail of human approvals
  ✅ Compliance for regulated industries

USE CASES:
----------
  • Executive summaries requiring accuracy
  • Legal/regulatory meeting records
  • Customer-facing meeting notes
  • High-stakes decision documentation
  • Training data for model improvement
""")

print("\n" + "=" * 70)
print("  NEXT: Stage 7 → Evaluation and Tracing")
print("=" * 70)
