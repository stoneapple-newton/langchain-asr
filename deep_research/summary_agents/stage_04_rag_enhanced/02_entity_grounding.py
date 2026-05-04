"""
Stage 4, File 2: Entity Grounding
==================================
CONCEPT: Ground summary claims with verified entity knowledge.

ASR transcripts can mishear names, companies, and technical terms.
Entity grounding verifies extracted entities against a knowledge base
to ensure accuracy in summaries.

Key concepts:
  - Named Entity Recognition (NER)
  - Entity verification against knowledge base
  - Grounded summary generation

Run this file:
  uv run deep_research/summary_agents/stage_04_rag_enhanced/02_entity_grounding.py
"""

from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import create_chat_model, structured_output_chain
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Mock Knowledge Base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "people": {
        "alice chen": {"role": "Senior Engineer", "team": "Platform", "email": "alice@company.com"},
        "bob smith": {"role": "Product Manager", "team": "Product", "email": "bob@company.com"},
        "carol jones": {"role": "Tech Lead", "team": "Platform", "email": "carol@company.com"},
    },
    "projects": {
        "nexus": {"full_name": "Nexus Platform", "status": "active", "lead": "carol jones"},
        "atlas": {"full_name": "Atlas Dashboard", "status": "planning", "lead": "alice chen"},
    },
    "technologies": {
        "jwt": {"full_name": "JSON Web Tokens", "category": "authentication", "version": "RFC 7519"},
        "oauth": {"full_name": "OAuth 2.0", "category": "authentication", "version": "2.0"},
        "wcag": {"full_name": "Web Content Accessibility Guidelines", "category": "standards", "version": "2.1"},
        "node.js": {"full_name": "Node.js", "category": "runtime", "version": "20 LTS"},
        "postgres": {"full_name": "PostgreSQL", "category": "database", "version": "15"},
    },
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """An entity extracted from the transcript."""
    name: str = Field(description="The entity text as mentioned")
    type: str = Field(description="Entity type: person, project, technology, other")
    confidence: str = Field(description="high, medium, or low")


class EntityExtraction(BaseModel):
    """Collection of entities extracted from a transcript."""
    entities: list[ExtractedEntity]


class VerifiedEntity(BaseModel):
    """An entity verified against the knowledge base."""
    original: str
    canonical: str | None
    type: str
    status: str  # verified, corrected, unverified
    info: dict | None


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class GroundingState(TypedDict):
    """State for entity grounding workflow."""
    
    transcript: str
    
    # Entity extraction
    extracted_entities: list[dict]
    
    # Verified entities
    verified_entities: list[dict]
    
    # Output
    grounded_summary: str
    
    # Audit
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Entity Extraction Node
# ---------------------------------------------------------------------------

def extract_entities(state: GroundingState) -> dict:
    """Extract entities from the transcript."""
    print("  [extract_entities] Finding named entities...")
    
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract named entities from this meeting transcript. "
            "Look for: people (names), projects (codenames), technologies (tools, standards), "
            "and organizations.\n\n{format_instructions}"
        ),
        ("human", "{transcript}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = structured_output_chain(llm, prompt, EntityExtraction)
    result = chain.invoke({"transcript": state["transcript"][:2000]})
    
    entities = result.get("entities", [])
    print(f"    Found {len(entities)} entities:")
    for e in entities[:5]:
        print(f"      - {e.get('name')} ({e.get('type')}, {e.get('confidence')})")
    
    return {
        "extracted_entities": entities,
        "messages": [HumanMessage(content=f"Extracted {len(entities)} entities")],
    }


# ---------------------------------------------------------------------------
# Entity Verification Node
# ---------------------------------------------------------------------------

def verify_entities(state: GroundingState) -> dict:
    """Verify extracted entities against knowledge base."""
    print("  [verify_entities] Checking against knowledge base...")
    
    verified = []
    
    for entity in state.get("extracted_entities", []):
        name = entity.get("name", "").lower()
        entity_type = entity.get("type", "other").lower()
        
        # Check knowledge base by type
        kb_category = None
        if entity_type in ["person", "people"]:
            kb_category = "people"
        elif entity_type in ["project", "projects"]:
            kb_category = "projects"
        elif entity_type in ["technology", "technologies", "tool"]:
            kb_category = "technologies"
        
        if kb_category and kb_category in KNOWLEDGE_BASE:
            kb_entries = KNOWLEDGE_BASE[kb_category]
            
            # Look for exact or partial match
            match = None
            if name in kb_entries:
                match = name
            else:
                # Try partial match
                for kb_name in kb_entries:
                    if kb_name in name or name in kb_name:
                        match = kb_name
                        break
            
            if match:
                verified.append({
                    "original": entity.get("name"),
                    "canonical": match,
                    "type": entity_type,
                    "status": "verified",
                    "info": kb_entries[match],
                })
                print(f"    ✓ {entity.get('name')} → {match}")
            else:
                verified.append({
                    "original": entity.get("name"),
                    "canonical": None,
                    "type": entity_type,
                    "status": "unverified",
                    "info": None,
                })
                print(f"    ? {entity.get('name')} (not in KB)")
        else:
            verified.append({
                "original": entity.get("name"),
                "canonical": None,
                "type": entity_type,
                "status": "unverified",
                "info": None,
            })
    
    return {
        "verified_entities": verified,
        "messages": [HumanMessage(content=f"Verified {len(verified)} entities")],
    }


# ---------------------------------------------------------------------------
# Grounded Summary Generation
# ---------------------------------------------------------------------------

def generate_grounded_summary(state: GroundingState) -> dict:
    """Generate summary with entity grounding."""
    print("  [generate_grounded_summary] Creating grounded summary...")
    
    # Build entity verification report
    verified = state.get("verified_entities", [])
    
    verified_info = []
    corrections = []
    
    for v in verified:
        if v["status"] == "verified" and v["info"]:
            info_str = ", ".join(f"{k}={v}" for k, v in v["info"].items())
            verified_info.append(f"- {v['original']} ({v['type']}): {info_str}")
        elif v["status"] == "unverified":
            corrections.append(f"- {v['original']}: NOT VERIFIED - check spelling")
    
    entity_context = "Verified Entities:\n" + "\n".join(verified_info)
    if corrections:
        entity_context += "\n\nVerification Notes:\n" + "\n".join(corrections)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a meeting summary using verified entity information.\n\n"
            "{entity_context}\n\n"
            "Use canonical names for verified entities. "
            "Flag any unverified entities with [?]."
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "entity_context": entity_context,
        "transcript": state["transcript"][:2000],
    })
    
    return {
        "grounded_summary": summary,
        "messages": [HumanMessage(content="Generated grounded summary")],
    }


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(GroundingState)

workflow.add_node("extract", extract_entities)
workflow.add_node("verify", verify_entities)
workflow.add_node("generate", generate_grounded_summary)

workflow.add_edge(START, "extract")
workflow.add_edge("extract", "verify")
workflow.add_edge("verify", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 4.2: ENTITY GROUNDING")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars\n")

result = app.invoke({
    "transcript": transcript_text,
    "extracted_entities": [],
    "verified_entities": [],
    "grounded_summary": "",
    "messages": [],
})

print("\n" + "-" * 70)
print("  VERIFIED ENTITIES")
print("-" * 70)
for v in result["verified_entities"]:
    status_icon = "✓" if v["status"] == "verified" else "?"
    info_str = ""
    if v["info"]:
        info_str = f" → {v['info']}"
    print(f"  {status_icon} {v['original']} ({v['type']}){info_str}")

print("\n" + "=" * 70)
print("  GROUNDED SUMMARY")
print("=" * 70)
print(result["grounded_summary"])


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Entity extraction identifies important named items
✅ Knowledge base verification ensures accuracy
✅ Grounded summaries use canonical names
✅ Unverified entities can be flagged for review

Entity Grounding Pipeline:
  Transcript → Extract Entities → Verify KB → Generate with Grounding

Benefits:
   - Corrects ASR transcription errors in names
   - Consistent entity references
   - Links to entity metadata (roles, versions)
   - Flags unknown entities for KB expansion

Knowledge Base Sources:
   - HR system (people, org chart)
   - Project management tools (projects, codenames)
   - Documentation (technologies, standards)
   - Previous meeting entity extractions

Next: Stage 4.3 → Persona-aware summaries
""")
