"""
ASR Stage 5, File 2: Domain-Aware Multi-Pass Correction
=========================================================
CONCEPT: Using specialised retrievers per correction type for higher precision.

A single "correct this segment" prompt is a blunt instrument. Production
quality requires specialised passes, each using a different retrieval index:

  Pass 1 — Terminology retriever : fixes acronyms, technical spellings
  Pass 2 — Participant retriever  : attributes names and roles correctly
  Pass 3 — Project retriever      : fills in project codenames and deadlines

Then a final consolidation step merges all suggestions into one clean output
and a quality report compares before/after across the full transcript.

New patterns:
  - Multiple specialised vector stores (one per knowledge domain)
  - Ensemble retriever: merge results from two stores
  - Structured correction output with change tracking per segment

Run this file:
  uv run deep_research/asr/stage_05_rag_context/02_domain_aware_correction.py
"""

import os
import json
import copy
import re
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=512,
)
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)


# ---------------------------------------------------------------------------
# 1. Build specialised knowledge stores
# ---------------------------------------------------------------------------

TERM_DOCS = [
    Document(page_content="JWT: JSON Web Token. Always written in uppercase.",
             metadata={"type": "term"}),
    Document(page_content="DevOps: capitalised as one word. Never 'devops' or 'dev ops'.",
             metadata={"type": "term"}),
    Document(page_content="Node.js: JavaScript runtime. 'Node 18' refers to version 18.",
             metadata={"type": "term"}),
    Document(page_content="WCAG: Web Content Accessibility Guidelines for accessibility standards.",
             metadata={"type": "term"}),
    Document(page_content="authentication: the process of verifying identity. Abbrev 'auth' is acceptable.",
             metadata={"type": "term"}),
    Document(page_content="staging: the pre-production environment. Always lowercase.",
             metadata={"type": "term"}),
    Document(page_content="breakpoints: CSS responsive design widths at which layout changes.",
             metadata={"type": "term"}),
    Document(page_content="scaffolding: initial boilerplate code structure created before full implementation.",
             metadata={"type": "term"}),
]

PARTICIPANT_DOCS = [
    Document(page_content="Alice Chen is the Product Manager. She facilitates meetings and owns the roadmap.",
             metadata={"type": "participant", "speaker": "SPEAKER_00"}),
    Document(page_content="Bob Nakamura is the Senior Engineer. He works on authentication and backend systems.",
             metadata={"type": "participant", "speaker": "SPEAKER_01"}),
    Document(page_content="Carol Rivera is the UX Designer. She handles dashboard design and accessibility.",
             metadata={"type": "participant", "speaker": "SPEAKER_02"}),
]

PROJECT_DOCS = [
    Document(page_content="Dashboard v2: analytics dashboard with charts. Hard deadline October 15th.",
             metadata={"type": "project"}),
    Document(page_content="Auth Service Overhaul: fixing JWT token invalidation bug from library upgrade.",
             metadata={"type": "project"}),
    Document(page_content="Analytics Integration: event tracking spec for conversion funnel. Owner: Alice.",
             metadata={"type": "project"}),
    Document(page_content="Node upgrade: moving from Node 16 to Node 18. Requires DevOps approval.",
             metadata={"type": "project"}),
]

term_store = InMemoryVectorStore.from_documents(TERM_DOCS, embeddings)
participant_store = InMemoryVectorStore.from_documents(PARTICIPANT_DOCS, embeddings)
project_store = InMemoryVectorStore.from_documents(PROJECT_DOCS, embeddings)

term_retriever = term_store.as_retriever(search_kwargs={"k": 3})
participant_retriever = participant_store.as_retriever(search_kwargs={"k": 2})
project_retriever = project_store.as_retriever(search_kwargs={"k": 2})

print("=== 1. Specialised stores built ===")
print(f"  Term store        : {len(TERM_DOCS)} docs")
print(f"  Participant store : {len(PARTICIPANT_DOCS)} docs")
print(f"  Project store     : {len(PROJECT_DOCS)} docs")
print()


# ---------------------------------------------------------------------------
# 2. Correction chain per pass
# ---------------------------------------------------------------------------

class SegmentCorrection(BaseModel):
    original: str
    corrected: str
    changes: list[str] = Field(description="List of specific changes made")
    confidence: float


correction_parser = JsonOutputParser(pydantic_object=SegmentCorrection)


def make_correction_chain(system_instruction: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction + "\n\nRespond with JSON: {format_instructions}"),
        ("human", "Context:\n{context}\n\nText: {text}"),
    ]).partial(format_instructions=correction_parser.get_format_instructions())
    return prompt | llm | correction_parser


term_chain = make_correction_chain(
    "Fix technical term spellings using the provided terminology context. "
    "Only change terms that appear in the context. Keep all other text identical."
)

participant_chain = make_correction_chain(
    "Use the participant context to correctly attribute names and roles. "
    "Only change words that clearly refer to a participant. Keep all other text identical."
)

project_chain = make_correction_chain(
    "Use the project context to correctly spell project names and key terms. "
    "Only change words clearly related to a project. Keep all other text identical."
)


def retrieve_and_format(retriever, text: str) -> str:
    docs = retriever.invoke(text)
    return "\n".join(f"• {doc.page_content}" for doc in docs)


# ---------------------------------------------------------------------------
# 3. Multi-pass correction
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    original: str
    after_terms: str
    after_participants: str
    after_projects: str
    final: str
    all_changes: list[str] = field(default_factory=list)
    confidence: float = 1.0


def multi_pass_correct(text: str, speaker: str) -> SegmentResult:
    result = SegmentResult(
        original=text, after_terms=text,
        after_participants=text, after_projects=text, final=text,
    )

    # Pass 1: terminology
    try:
        ctx = retrieve_and_format(term_retriever, text)
        r1 = term_chain.invoke({"context": ctx, "text": text})
        r1 = r1 if isinstance(r1, dict) else r1.dict()
        result.after_terms = r1.get("corrected", text)
        result.all_changes.extend(r1.get("changes", []))
        result.confidence = min(result.confidence, r1.get("confidence", 1.0))
    except Exception:
        result.after_terms = text

    # Pass 2: participants
    try:
        ctx = retrieve_and_format(participant_retriever, result.after_terms)
        r2 = participant_chain.invoke({"context": ctx, "text": result.after_terms})
        r2 = r2 if isinstance(r2, dict) else r2.dict()
        result.after_participants = r2.get("corrected", result.after_terms)
        result.all_changes.extend(r2.get("changes", []))
    except Exception:
        result.after_participants = result.after_terms

    # Pass 3: projects
    try:
        ctx = retrieve_and_format(project_retriever, result.after_participants)
        r3 = project_chain.invoke({"context": ctx, "text": result.after_participants})
        r3 = r3 if isinstance(r3, dict) else r3.dict()
        result.after_projects = r3.get("corrected", result.after_participants)
        result.all_changes.extend(r3.get("changes", []))
    except Exception:
        result.after_projects = result.after_participants

    result.final = result.after_projects
    return result


print("=== 2. Multi-pass domain-aware correction ===")
test_segments = [
    ("SPEAKER_01", "the authentication service its been really flaky in staging"),
    ("SPEAKER_01", "since we updated the jwt library the old tokens arent being invalidated properly"),
    ("SPEAKER_02", "we went through like three iterations on the color palette to meet accessibility contrast ratios"),
    ("SPEAKER_01", "we might need to upgrade our node version for the new chart library it requires node 18 or above"),
]

for speaker, text in test_segments:
    result = multi_pass_correct(text, speaker)
    print(f"\n  [{speaker}]")
    print(f"  Original  : {result.original}")
    print(f"  After terms: {result.after_terms}")
    print(f"  After parts: {result.after_participants}")
    print(f"  Final      : {result.final}")
    if result.all_changes:
        print(f"  Changes    : {result.all_changes[:3]}")
print()


# ---------------------------------------------------------------------------
# 4. Quality comparison: before vs. after
# ---------------------------------------------------------------------------

print("=== 3. Quality comparison ===")

TECH_TERMS = {"jwt", "devops", "node.js", "wcag", "authentication"}
FILLERS = {"uh", "um", "like", "you know"}


def score_transcript(segments: list[dict]) -> dict:
    all_text = " ".join(s["text"].lower() for s in segments)
    words = all_text.split()
    total = max(len(words), 1)
    segs = max(len(segments), 1)

    correct_terms = sum(1 for t in TECH_TERMS if t in all_text)
    wrong_terms = sum(1 for t in {"jwt", "devops"} if t.upper() not in " ".join(s["text"] for s in segments))
    fillers = sum(1 for w in words if w in FILLERS)
    has_punct = sum(1 for s in segments if re.search(r"[.!?]$", s["text"].strip()))

    return {
        "filler_rate": round(fillers / total, 3),
        "punctuation_rate": round(has_punct / segs, 3),
        "recognised_tech_terms": correct_terms,
    }


before_score = score_transcript(segments)

# Apply multi-pass to first 8 segments
improved = copy.deepcopy(segments)
for i, seg in enumerate(improved[:8]):
    result = multi_pass_correct(seg["text"].strip(), seg.get("speaker", ""))
    improved[i]["text"] = " " + result.final

after_score = score_transcript(improved)

print(f"  {'Metric':<28} {'Before':>8} {'After':>8}")
print("  " + "-" * 46)
for key in ["filler_rate", "punctuation_rate", "recognised_tech_terms"]:
    b = before_score[key]
    a = after_score[key]
    delta = a - b
    print(f"  {key:<28} {b:>8.3f} {a:>8.3f}  ({delta:+.3f})")
print()


# ---------------------------------------------------------------------------
# 5. Save the domain-corrected transcript
# ---------------------------------------------------------------------------

out = copy.deepcopy(raw)
out["segments"] = improved + segments[8:]  # improved first 8, rest unchanged
out_path = OUT_DIR / "transcript_domain_corrected.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved: {out_path.name}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Three specialised stores (terms, participants, projects) > one general store
# ✅ Multi-pass: each pass uses a targeted retriever and a targeted prompt
# ✅ Chain the passes: Pass 2 input = Pass 1 output — corrections compound
# ✅ SegmentResult tracks the text at every pass — easy to debug regressions
# ✅ Score before AND after every major change — proves improvement
# ✅ This pattern scales: add more passes (e.g., sentiment normalisation, acronym expansion)
