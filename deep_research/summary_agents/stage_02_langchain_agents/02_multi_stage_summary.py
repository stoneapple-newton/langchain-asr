"""
Stage 2, File 2: Multi-Stage Summary Pipeline
==============================================
CONCEPT: Explicit pipeline with controlled agent workflow.

Unlike the ReAct agent which decides its own path, here we define
an explicit pipeline: Extract → Analyze → Summarize. The agent
follows this sequence, with each stage being a tool.

Key concepts:
  - Explicit workflow vs autonomous agent
  - State passing between stages
  - Pipeline control and predictability

Run this file:
  uv run deep_research/summary_agents/stage_02_langchain_agents/02_multi_stage_summary.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Shared State (Module-level for demo)
# ---------------------------------------------------------------------------
_pipeline_state: dict = {}


# ---------------------------------------------------------------------------
# Pipeline Stage Tools
# ---------------------------------------------------------------------------

@tool
def stage1_extract(transcript_text: str) -> str:
    """
    STAGE 1: Extract raw information from the transcript.
    Identify: speakers, topics mentioned, timestamps of key moments.
    Returns structured extraction results.
    """
    global _pipeline_state
    
    # Store in shared state
    _pipeline_state["transcript"] = transcript_text
    _pipeline_state["extraction"] = {
        "speakers": list(set(
            line.split(":")[0] for line in transcript_text.split("\n")
            if ":" in line
        )),
        "length_chars": len(transcript_text),
        "keywords": ["API", "JWT", "WCAG", "Node.js"]  # Simplified for demo
    }
    
    return f"""Extraction Complete:
- Speakers identified: {len(_pipeline_state['extraction']['speakers'])}
- Transcript length: {_pipeline_state['extraction']['length_chars']} characters
- Key terms detected: {', '.join(_pipeline_state['extraction']['keywords'])}"""


@tool
def stage2_analyze() -> str:
    """
    STAGE 2: Analyze extracted information.
    Determine: meeting type, importance of topics, sentiment.
    Requires stage1_extract to be called first.
    """
    global _pipeline_state
    
    if "extraction" not in _pipeline_state:
        return "ERROR: Must run stage1_extract first!"
    
    extraction = _pipeline_state["extraction"]
    
    # Simple analysis logic
    analysis = {
        "meeting_type": "Technical Planning",
        "priority_topics": ["API Design", "Authentication"],
        "sentiment": "productive",
        "decision_count": 2,
    }
    
    _pipeline_state["analysis"] = analysis
    
    return f"""Analysis Complete:
- Meeting type: {analysis['meeting_type']}
- Priority topics: {', '.join(analysis['priority_topics'])}
- Overall sentiment: {analysis['sentiment']}
- Decisions identified: {analysis['decision_count']}"""


@tool
def stage3_summarize(summary_type: str = "comprehensive") -> str:
    """
    STAGE 3: Generate final summary based on extraction and analysis.
    Options: comprehensive, executive_brief, action_items_only
    Requires stage1_extract and stage2_analyze to be called first.
    """
    global _pipeline_state
    
    if "analysis" not in _pipeline_state:
        return "ERROR: Must run stage1_extract and stage2_analyze first!"
    
    extraction = _pipeline_state["extraction"]
    analysis = _pipeline_state["analysis"]
    
    # Generate different summary types
    if summary_type == "executive_brief":
        summary = f"""# Executive Brief

**Meeting Type:** {analysis['meeting_type']}
**Participants:** {len(extraction['speakers'])} speakers
**Key Focus:** {', '.join(analysis['priority_topics'])}

Bottom line: Technical planning meeting covering API and authentication decisions.
"""
    elif summary_type == "action_items_only":
        summary = """# Action Items

1. Review JWT implementation approach
2. Assess WCAG compliance requirements
3. Continue Node.js vs Python evaluation
"""
    else:  # comprehensive
        summary = f"""# Comprehensive Meeting Summary

## Meeting Context
- **Type:** {analysis['meeting_type']}
- **Sentiment:** {analysis['sentiment']}
- **Participants:** {', '.join(extraction['speakers'])}

## Priority Topics
{chr(10).join(f"- {t}" for t in analysis['priority_topics'])}

## Technical Terms Discussed
{', '.join(extraction['keywords'])}

## Decisions Made
{analysis['decision_count']} key decisions were made during this session.

## Recommended Follow-up
- Document API design decisions
- Schedule authentication deep-dive
- Prepare WCAG compliance checklist
"""
    
    _pipeline_state["summary"] = summary
    return f"Summary generated ({summary_type}):\n{summary}"


@tool
def reset_pipeline() -> str:
    """Reset the pipeline state. Call this before processing a new transcript."""
    global _pipeline_state
    _pipeline_state = {}
    return "Pipeline state reset. Ready for new transcript."


# ---------------------------------------------------------------------------
# Setup Pipeline Agent
# ---------------------------------------------------------------------------

tools = [
    stage1_extract,
    stage2_analyze,
    stage3_summarize,
    reset_pipeline,
]

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=2048,
)

system_prompt = """You are a pipeline-based summarization agent. Follow this EXACT workflow:

1. ALWAYS call reset_pipeline first to clear any previous state
2. Call stage1_extract with the transcript
3. Call stage2_analyze (no arguments needed)
4. Call stage3_summarize with the desired summary_type

Do not skip steps or change the order. The pipeline depends on state from previous stages."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
)


# ---------------------------------------------------------------------------
# Run Pipeline
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 2.2: MULTI-STAGE PIPELINE AGENT")
print("=" * 70)
print("\nPipeline: reset → extract → analyze → summarize")
print(f"\nTranscript: {len(transcript_text)} chars\n")

result = agent_executor.invoke({
    "input": f"Process this transcript through the full pipeline:\n\n{transcript_text[:1000]}"
})

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("=" * 70)


# ---------------------------------------------------------------------------
# Alternative: Direct Pipeline (No Agent)
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  ALTERNATIVE: DIRECT PIPELINE (NO AGENT)")
print("-" * 70)
print("""
For predictable workflows, you might skip the agent entirely:

    result = (
        reset_pipeline()
        | stage1_extract
        | stage2_analyze  
        | stage3_summarize
    ).invoke(transcript)

This gives you:
- Full control over execution order
- No reasoning overhead
- Easier debugging
- Lower cost

Use agents when you need flexibility; use pipelines when you need control.
""")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Explicit pipeline = predictable execution order
✅ State management between stages (module-level for demo)
✅ Each stage can validate prerequisites
✅ Easy to insert new stages or modify existing ones

⚠️ Trade-offs:
   - More rigid than ReAct agent
   - State management complexity
   - Still uses agent overhead even with fixed workflow

When to use explicit pipelines:
   - Compliance/audit requirements need reproducibility
   - Each stage is expensive (don't want redundant calls)
   - Need to checkpoint/save intermediate results
   - Debugging production issues

Next: Stage 2.3 → Context-aware summarization
""")
