"""
Stage 2, File 3: Context-Aware Summary Agent
=============================================
CONCEPT: Enhancing summaries with meeting context and domain knowledge.

Meetings don't happen in a vacuum. Adding context like:
- Project glossary (acronyms, codenames)
- Attendee roles and responsibilities  
- Historical meeting references
improves summary quality significantly.

Key concepts:
  - Context retrieval tools
  - Knowledge augmentation
  - Domain-aware summarization

Run this file:
  uv run deep_research/summary_agents/stage_02_langchain_agents/03_summary_with_context.py
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
# Mock Knowledge Base (in production, this would be a real database/RAG)
# ---------------------------------------------------------------------------

PROJECT_CONTEXT = {
    "project_name": "Nexus Platform",
    "glossary": {
        "JWT": "JSON Web Token - authentication standard",
        "WCAG": "Web Content Accessibility Guidelines",
        "API": "Application Programming Interface",
        "SLA": "Service Level Agreement",
        "Q4": "Fourth quarter (Oct-Dec)",
        "Node": "Node.js JavaScript runtime",
    },
    "team_roles": {
        "SPEAKER_00": "Product Manager",
        "SPEAKER_01": "Lead Engineer", 
        "SPEAKER_02": "Frontend Developer",
    },
    "current_sprint": "Sprint 23",
    "upcoming_milestones": [
        "Nov 15: Beta release",
        "Dec 1: WCAG audit",
        "Dec 15: Q4 deliverables",
    ],
}


# ---------------------------------------------------------------------------
# Context Tools
# ---------------------------------------------------------------------------

@tool
def load_project_context() -> str:
    """
    Load project context including glossary, roles, and milestones.
    Use this to understand acronyms and codenames in the transcript.
    """
    ctx = PROJECT_CONTEXT
    
    glossary_str = "\n".join(f"  {k}: {v}" for k, v in ctx["glossary"].items())
    milestones_str = "\n".join(f"  - {m}" for m in ctx["upcoming_milestones"])
    
    return f"""Project: {ctx['project_name']}
Current Sprint: {ctx['current_sprint']}

Glossary:
{glossary_str}

Upcoming Milestones:
{milestones_str}"""


@tool
def get_speaker_role(speaker_id: str) -> str:
    """
    Get the role/responsibility of a speaker.
    Helps identify who has authority on different topics.
    """
    roles = PROJECT_CONTEXT["team_roles"]
    role = roles.get(speaker_id, "Unknown role")
    return f"{speaker_id}: {role}"


@tool
def expand_acronym(acronym: str) -> str:
    """
    Expand an acronym using the project glossary.
    Returns the full meaning or 'Not found in glossary'.
    """
    glossary = PROJECT_CONTEXT["glossary"]
    meaning = glossary.get(acronym.upper(), "Not found in glossary")
    return f"{acronym}: {meaning}"


@tool
def summarize_with_context(transcript_text: str, context: str) -> str:
    """
    Generate a summary that incorporates project context.
    This tool combines the transcript with contextual knowledge.
    """
    # In practice, this would call an LLM with both inputs
    # For demo, we simulate the enhanced output
    
    summary = f"""# Nexus Platform Team Meeting Summary

*Context: {PROJECT_CONTEXT['project_name']} - {PROJECT_CONTEXT['current_sprint']}*

## Overview
Technical planning session for Q4 deliverables, focusing on authentication (JWT) 
and accessibility (WCAG) compliance for the upcoming beta release (Nov 15).

## Context-Enriched Discussion Points

### API & Authentication (Led by Lead Engineer)
The team discussed JWT implementation for the Nexus Platform API. This is critical 
for the beta release milestone on Nov 15. [JWT = JSON Web Token - authentication standard]

### Accessibility Compliance  
WCAG audit scheduled for Dec 1. The Frontend Developer raised concerns about 
meeting accessibility standards. [WCAG = Web Content Accessibility Guidelines]

### Backend Technology Decision
Discussion on Node.js vs other options for the service backend. Decision needed 
to proceed with Q4 roadmap. [Node = Node.js JavaScript runtime]

## Timeline Context
- Beta release: Nov 15 (3 weeks away)
- WCAG audit: Dec 1  
- Q4 deliverables: Dec 15

## Action Items
1. Lead Engineer: Finalize JWT library selection
2. Frontend Developer: Prepare WCAG compliance checklist  
3. Product Manager: Schedule backend tech decision meeting

## Notes
- Meeting occurred during Sprint 23
- Q4 planning context: End of quarter delivery pressure
- Accessibility compliance is a hard requirement for launch
"""
    return summary


# ---------------------------------------------------------------------------
# Comparison: Without Context
# ---------------------------------------------------------------------------

@tool
def summarize_without_context(transcript_text: str) -> str:
    """
    Generate a basic summary without project context.
    Use this to compare against context-aware summarization.
    """
    return """# Meeting Summary

## Overview  
Team discussed Q4 planning, API integration, and authentication.

## Key Points
- Talked about JWT and API design
- Mentioned WCAG requirements
- Discussed Node.js backend options
- Reviewed roadmap priorities

## Action Items
- Look into JWT options
- Check WCAG compliance
- Decide on backend technology

## Decisions
- Need to prioritize authentication work
- WCAG compliance required
"""


# ---------------------------------------------------------------------------
# Setup Context-Aware Agent
# ---------------------------------------------------------------------------

tools = [
    load_project_context,
    get_speaker_role,
    expand_acronym,
    summarize_with_context,
    summarize_without_context,
]

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=2048,
)

system_prompt = """You are a context-aware meeting summarizer for the Nexus Platform project.

Your workflow:
1. First, load the project context to understand acronyms and milestones
2. Identify speakers and their roles
3. Generate a summary that incorporates this context

Context helps you:
- Expand acronyms (JWT, WCAG, etc.) for clarity
- Connect discussions to project milestones  
- Identify who has decision authority on different topics
- Highlight time-sensitive items relative to upcoming deadlines"""

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
# Run Comparison
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 2.3: CONTEXT-AWARE SUMMARY AGENT")
print("=" * 70)

print("\n" + "-" * 70)
print("  WITHOUT CONTEXT (Basic Summary)")
print("-" * 70)
print(summarize_without_context.invoke({"transcript_text": transcript_text}))

print("\n" + "-" * 70)
print("  WITH CONTEXT (Knowledge-Augmented)")
print("-" * 70)
print(summarize_with_context.invoke({
    "transcript_text": transcript_text,
    "context": load_project_context.invoke({})
}))


# ---------------------------------------------------------------------------
# Show the Agent Process
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  AGENT WORKFLOW (Context Loading + Summary)")
print("=" * 70)

result = agent_executor.invoke({
    "input": f"Create a context-aware summary of this meeting:\n\n{transcript_text[:800]}"
})

print("\n" + "=" * 70)
print("  AGENT OUTPUT")
print("=" * 70)
print(result["output"])


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Context improves summary quality significantly
✅ Acronyms get expanded for clarity
✅ Timeline context highlights urgency
✅ Role information clarifies decision authority

Types of Context to Consider:
   1. Glossary/Acronyms - Expand technical terms
   2. Project Timeline - Connect to milestones
   3. Attendee Roles - Understand who decides what
   4. Historical - Reference previous related meetings
   5. Domain Knowledge - Industry-specific concepts

Context Sources:
   - Project management tools (Jira, Linear)
   - Previous meeting summaries
   - Team directories
   - Documentation/Wikis
   - Code repositories

Next: Stage 3 → LangGraph for stateful workflows
""")
