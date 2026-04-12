"""
ASR Stage 5, File 3: Grounded Correction with Query Expansion and Citations
============================================================================
CONCEPT: Three LLM upgrades to make RAG-based correction more precise
         and auditable.

Stage 5 Files 1-2 use a basic RAG pattern:
  segment text → retriever → top-k docs → correction prompt

Problems with basic RAG for ASR correction:
  1. Query mismatch: "itll need to go through devops" doesn't semantically
     match "DevOps: capitalised as one word" — retriever may miss it
  2. Irrelevant context: retrieved docs may be loosely related and add noise
  3. No citations: we can't tell if the LLM used the context or just guessed

Upgrades in this file:
  1. LLM Query Expansion — before retrieval, ask LLM to rewrite the segment
     as a precise knowledge query to improve retrieval recall
  2. LLM Relevance Grading — after retrieval, LLM scores each doc's relevance
     to the segment and filters low-relevance context
  3. Grounded Correction — correction chain must cite which document
     supports each change, producing an auditable change log

New patterns:
  - Query expansion as a pre-retrieval step in an LCEL chain
  - RunnableLambda for inline filtering
  - Citation-backed structured output
  - RAG faithfulness: LLM is constrained to only use what it retrieved

Run this file:
  uv run deep_research/asr/stage_05_rag_context/03_grounded_correction.py
"""

import os
import json
import copy
from pathlib import Path
import sys

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, create_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]

llm = create_chat_model(
    "asr",
    temperature=0,
    max_tokens=512,
)
embeddings = create_embeddings("asr")


# ---------------------------------------------------------------------------
# 1. Knowledge base (same domain knowledge as Stage 5 File 1)
# ---------------------------------------------------------------------------

KNOWLEDGE_DOCS = [
    Document(page_content="JWT = JSON Web Token. Always written in uppercase as 'JWT'.",
             metadata={"id": "term_jwt", "category": "terminology"}),
    Document(page_content="DevOps: capitalised as one word. Never 'devops' or 'dev ops'. "
             "Node 16→Node 18 upgrade requires DevOps team approval.",
             metadata={"id": "term_devops", "category": "terminology"}),
    Document(page_content="Node.js runtime. 'Node 18' refers to version 18. Requires DevOps approval to upgrade.",
             metadata={"id": "term_nodejs", "category": "terminology"}),
    Document(page_content="WCAG: Web Content Accessibility Guidelines. Contrast ratios are a WCAG requirement.",
             metadata={"id": "term_wcag", "category": "terminology"}),
    Document(page_content="authentication: process of verifying identity. Abbreviated 'auth'. "
             "Auth service has a JWT token invalidation bug from a library upgrade.",
             metadata={"id": "term_auth", "category": "terminology"}),
    Document(page_content="Alice Chen is the Product Manager. She facilitates meetings and owns the roadmap. "
             "Analytics Integration event tracking spec is her project.",
             metadata={"id": "participant_alice", "category": "participant"}),
    Document(page_content="Bob Nakamura is the Senior Engineer responsible for backend and authentication service.",
             metadata={"id": "participant_bob", "category": "participant"}),
    Document(page_content="Carol Rivera is the UX Designer. She handles dashboard design, "
             "accessibility requirements, and mobile breakpoints.",
             metadata={"id": "participant_carol", "category": "participant"}),
    Document(page_content="Dashboard v2: new analytics dashboard with custom charts. Hard deadline October 15th. "
             "Carol handles design, Bob handles engineering.",
             metadata={"id": "project_dashboard", "category": "project"}),
    Document(page_content="Auth Service Overhaul: fixing JWT token invalidation bug introduced in library upgrade. "
             "Bob Nakamura owns this project.",
             metadata={"id": "project_auth", "category": "project"}),
    Document(page_content="Staging: the pre-production environment for testing. "
             "Breakpoints: CSS responsive design widths. "
             "Scaffolding: initial code structure before full implementation.",
             metadata={"id": "term_env", "category": "terminology"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(KNOWLEDGE_DOCS)
store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = store.as_retriever(search_kwargs={"k": 5})

print(f"=== Knowledge base: {len(chunks)} chunks from {len(KNOWLEDGE_DOCS)} docs ===\n")


# ---------------------------------------------------------------------------
# 2. Pydantic schemas
# ---------------------------------------------------------------------------

class RelevanceGrade(BaseModel):
    doc_id: str = Field(description="The metadata id of the document")
    relevance: float = Field(description="0.0-1.0: how relevant this document is to the segment")
    reason: str = Field(description="One-line reason for the grade")


class DocumentGrades(BaseModel):
    grades: list[RelevanceGrade]


class GroundedCorrection(BaseModel):
    original: str
    corrected: str
    changes: list[str] = Field(description="Each change made, e.g. 'jwt → JWT (source: term_jwt)'")
    citations: list[str] = Field(description="Document IDs actually used to justify changes")
    no_change_reason: str = Field(
        description="If no changes were made, why. Empty string if changes were made."
    )


grade_parser = JsonOutputParser(pydantic_object=DocumentGrades)
correction_parser = JsonOutputParser(pydantic_object=GroundedCorrection)


# ---------------------------------------------------------------------------
# 3. Query expansion chain
# ---------------------------------------------------------------------------

expand_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query optimizer for a transcript correction system.\n"
     "Given a raw ASR segment, rewrite it as a precise search query that will retrieve "
     "relevant terminology, participant, or project information from a knowledge base.\n\n"
     "Rules:\n"
     "  - Extract key entities: names, technical terms, project references\n"
     "  - Expand abbreviations if you know them (jwt → JWT JSON Web Token)\n"
     "  - Add synonyms for terms that might be mis-spelled\n"
     "  - Return ONLY the query string, no explanation"),
    ("human", "ASR segment: {text}"),
])

expand_chain = expand_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 4. Relevance grading chain
# ---------------------------------------------------------------------------

grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance judge. Rate how useful each retrieved document is for "
     "correcting the given transcript segment. "
     "Score 0.0 = completely irrelevant, 1.0 = directly answers a correction need.\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Segment to correct: {text}\n\n"
     "Retrieved documents:\n{docs}"),
]).partial(format_instructions=grade_parser.get_format_instructions())

grade_chain = grade_prompt | llm | grade_parser


# ---------------------------------------------------------------------------
# 5. Grounded correction chain
# ---------------------------------------------------------------------------

correction_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise transcript editor. Correct the segment using ONLY the provided "
     "knowledge documents. You MUST cite which document ID justifies each change.\n\n"
     "Rules:\n"
     "  - Only make changes that are directly supported by a document\n"
     "  - Do not guess or use general knowledge\n"
     "  - If a document says 'JWT is uppercase', change 'jwt' to 'JWT' and cite it\n"
     "  - If no document supports a change, do not make it\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Relevant knowledge (already filtered for relevance):\n{context}\n\n"
     "Segment: {text}"),
]).partial(format_instructions=correction_parser.get_format_instructions())

correction_chain = correction_prompt | llm | correction_parser


# ---------------------------------------------------------------------------
# 6. Full pipeline: expand → retrieve → grade → filter → correct
# ---------------------------------------------------------------------------

RELEVANCE_THRESHOLD = 0.5   # filter out docs below this grade


def expand_and_retrieve(text: str) -> tuple[str, list[Document]]:
    """Expand query, then retrieve."""
    try:
        expanded = expand_chain.invoke({"text": text})
    except Exception:
        expanded = text
    docs = retriever.invoke(expanded)
    return expanded, docs


def grade_and_filter(text: str, docs: list[Document]) -> list[Document]:
    """Grade retrieved docs and filter to relevant ones."""
    if not docs:
        return []

    docs_text = "\n\n".join(
        f"[{doc.metadata.get('id', f'doc_{i}')}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )
    try:
        result = grade_chain.invoke({"text": text, "docs": docs_text})
        grades = result.get("grades", []) if isinstance(result, dict) else []
        grade_map = {g.get("doc_id"): g.get("relevance", 0.0) for g in grades}
    except Exception:
        return docs  # fallback: use all retrieved

    # Filter docs by relevance grade
    filtered = []
    for doc in docs:
        doc_id = doc.metadata.get("id", "")
        if grade_map.get(doc_id, 0.0) >= RELEVANCE_THRESHOLD:
            filtered.append(doc)
    return filtered


def grounded_correct(text: str) -> dict:
    """Full pipeline for one segment."""
    expanded, docs = expand_and_retrieve(text)
    filtered_docs = grade_and_filter(text, docs)

    if not filtered_docs:
        return {
            "original": text,
            "corrected": text,
            "changes": [],
            "citations": [],
            "no_change_reason": "No relevant documents found after relevance grading",
        }

    context = "\n\n".join(
        f"[{doc.metadata.get('id', '?')}] {doc.page_content}"
        for doc in filtered_docs
    )

    try:
        result = correction_chain.invoke({"text": text, "context": context})
        r = result if isinstance(result, dict) else {}
        r["_expanded_query"] = expanded
        r["_docs_before_filter"] = len(docs)
        r["_docs_after_filter"] = len(filtered_docs)
        return r
    except Exception as e:
        return {
            "original": text, "corrected": text,
            "changes": [], "citations": [],
            "no_change_reason": f"Correction chain error: {e}",
            "_expanded_query": expanded,
        }


# ---------------------------------------------------------------------------
# 7. Demo on test segments
# ---------------------------------------------------------------------------

print("=== 1. Query Expansion Demo ===\n")
test_queries = [
    "the jwt library the old tokens arent being invalidated",
    "it requires node 18 or above so itll need to go through devops",
    "the mobile responsive version the breakpoints dont shift",
]
for q in test_queries:
    expanded = expand_chain.invoke({"text": q})
    print(f"  Original : {q}")
    print(f"  Expanded : {expanded}")
    print()


print("=== 2. Grounded Correction with Citations ===\n")
demo_segments = [
    ("SPEAKER_01", "the authentication service its been really flaky in staging"),
    ("SPEAKER_01", "the jwt library the old tokens arent being invalidated properly"),
    ("SPEAKER_00", "the analytics spec the conversion funnel events owner is alice"),
    ("SPEAKER_02", "we went through three iterations on the color palette to meet wcag contrast"),
    ("SPEAKER_01", "itll need to go through devops for the node upgrade"),
]

all_results = []
for speaker, text in demo_segments:
    result = grounded_correct(text)
    all_results.append((speaker, text, result))

    changed = result.get("corrected", text) != text
    print(f"  [{speaker}]")
    print(f"  Before   : {text}")
    print(f"  After    : {result.get('corrected', text)}")
    print(f"  Query    : {result.get('_expanded_query', '')[:60]}")
    print(f"  Docs     : {result.get('_docs_before_filter', 0)} retrieved → "
          f"{result.get('_docs_after_filter', 0)} relevant")
    if result.get("changes"):
        for change in result["changes"]:
            print(f"  Change   : {change}")
    if result.get("citations"):
        print(f"  Citations: {result['citations']}")
    if not changed:
        print(f"  No change: {result.get('no_change_reason', '')}")
    print()


# ---------------------------------------------------------------------------
# 8. Apply to full transcript
# ---------------------------------------------------------------------------

print("=== 3. Full Transcript Correction ===\n")

improved = copy.deepcopy(segments)
total_changes = 0
total_citations = set()

for i, seg in enumerate(improved[:12]):  # first 12 segments for demo
    result = grounded_correct(seg["text"].strip())
    corrected = result.get("corrected", "").strip()
    if corrected and corrected != seg["text"].strip():
        improved[i]["text"] = " " + corrected
        total_changes += len(result.get("changes", []))
    total_citations.update(result.get("citations", []))

print(f"  Segments processed : 12")
print(f"  Changes applied    : {total_changes}")
print(f"  Documents cited    : {sorted(total_citations)}")
print()

out = copy.deepcopy(raw)
out["segments"] = improved + segments[12:]
out_path = OUT_DIR / "transcript_grounded_correction.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved: {out_path.name}\n")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Query expansion improves retrieval recall — raw ASR text is a poor search query
# ✅ Relevance grading removes noise before it reaches the correction prompt
# ✅ Citations make every change auditable: you know exactly which doc justified it
# ✅ "Only change what the docs support" is RAG faithfulness — prevents hallucination
# ✅ The pipeline is: expand → retrieve → grade → filter → correct (5 LLM steps for precision)
# ✅ no_change_reason tells you when no relevant context was found — not a silent skip
# ✅ This pattern scales: richer knowledge base = better corrections, no code changes
