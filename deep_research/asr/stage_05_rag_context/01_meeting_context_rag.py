"""
ASR Stage 5, File 1: Meeting Context RAG
==========================================
CONCEPT: Injecting domain knowledge into the correction pipeline via retrieval.

The LLM has never heard of your company's products, internal acronyms,
team member names, or project codenames. Without context, it will:
  • Mis-capitalise product names ("langchain" instead of "LangChain")
  • Mis-spell technical terms ("jay-wee-tee" → "JWT")
  • Fail to recognise project names ("falcon" → no idea this is a codename)

RAG solution: build a context store with:
  1. Company/project glossary — terms and their correct spellings
  2. Participant directory  — who is on the call, their roles and names
  3. Prior meeting notes    — recurring topics and decisions

At correction time, retrieve the most relevant context for each segment
and inject it into the prompt so the LLM can make informed corrections.

Run this file:
  uv run deep_research/asr/stage_05_rag_context/01_meeting_context_rag.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
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
# 1. Build the context knowledge base
# ---------------------------------------------------------------------------

CONTEXT_DOCUMENTS = [
    # --- Glossary ---
    Document(
        page_content=(
            "Technical acronyms and correct spellings:\n"
            "JWT = JSON Web Token (authentication mechanism)\n"
            "devops = DevOps (capitalised as one word)\n"
            "auth = authentication\n"
            "Q4 = fourth quarter\n"
            "Node = Node.js (JavaScript runtime)\n"
            "WCAG = Web Content Accessibility Guidelines\n"
            "SRT = SubRip Text (subtitle file format)\n"
            "API = Application Programming Interface\n"
            "PR = Pull Request\n"
            "UI = User Interface, UX = User Experience"
        ),
        metadata={"category": "glossary", "source": "company_glossary"},
    ),
    # --- Participants ---
    Document(
        page_content=(
            "Meeting participants for Q4 Planning Sync (2024-10-07):\n"
            "- Alice Chen (Product Manager): facilitates meetings, sets priorities, owns the roadmap\n"
            "- Bob Nakamura (Senior Engineer): responsible for authentication service, backend systems\n"
            "- Carol Rivera (UX Designer): responsible for dashboard designs, accessibility, mobile"
        ),
        metadata={"category": "participants", "source": "company_directory"},
    ),
    # --- Project context ---
    Document(
        page_content=(
            "Current Q4 projects:\n"
            "Dashboard v2: new analytics dashboard with custom charts. Owner: Carol (design), Bob (engineering).\n"
            "Auth Service Overhaul: fixing JWT token invalidation bug introduced in library upgrade.\n"
            "Analytics Integration: event tracking spec for conversion funnel. Owner: Alice.\n"
            "Deadline: October 15th is a hard deadline for Dashboard v2."
        ),
        metadata={"category": "project_context", "source": "project_tracker"},
    ),
    # --- Prior meeting notes ---
    Document(
        page_content=(
            "Prior meeting decisions (Q3 retrospective, 2024-09-30):\n"
            "- Team agreed to prioritise security fixes over feature work\n"
            "- Dashboard accessibility was flagged by enterprise customer Acme Corp\n"
            "- Node 16 → Node 18 upgrade requires DevOps approval process\n"
            "- All PRs require at least one code review before merging"
        ),
        metadata={"category": "meeting_history", "source": "meeting_notes_q3"},
    ),
    # --- Domain-specific terms ---
    Document(
        page_content=(
            "Engineering terminology used in this team:\n"
            "Staging = the pre-production environment for testing\n"
            "Scaffolding = initial code structure/skeleton\n"
            "Breakpoints = CSS responsive design breakpoints (mobile/tablet/desktop)\n"
            "Conversion funnel = sequence of user actions leading to purchase/signup\n"
            "Invalidated tokens = expired or revoked JWT tokens that should no longer grant access\n"
            "Color palette = the set of brand colors used in the UI\n"
            "Hover states = visual feedback when a user hovers over an element"
        ),
        metadata={"category": "domain_terms", "source": "engineering_wiki"},
    ),
]

# Split larger documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
chunks = splitter.split_documents(CONTEXT_DOCUMENTS)

context_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = context_store.as_retriever(search_kwargs={"k": 3})

print(f"=== Context store built: {len(chunks)} chunks from {len(CONTEXT_DOCUMENTS)} documents ===")
print()


# ---------------------------------------------------------------------------
# 2. Context-aware correction chain
# ---------------------------------------------------------------------------
# The RAG pattern: retrieve relevant context, inject into correction prompt.

def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('category', 'info')}] {doc.page_content}"
        for doc in docs
    )


correction_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a transcript editor with access to meeting context.\n"
     "Use the context to:\n"
     "  1. Correct technical term spellings (e.g. jwt → JWT, auth → authentication)\n"
     "  2. Use correct participant names if identifiable\n"
     "  3. Fix domain-specific terms using the provided context\n"
     "  4. Add punctuation and capitalise correctly\n"
     "Return ONLY the corrected text, no explanations.\n\n"
     "Meeting context:\n{context}"),
    ("human", "Speaker: {speaker}\nText: {text}"),
])

context_correction_chain = (
    {
        "context": (lambda x: x["text"]) | retriever | format_context,
        "speaker": lambda x: x["speaker"],
        "text": lambda x: x["text"],
    }
    | correction_prompt
    | llm
    | StrOutputParser()
)


print("=== 2. Context-aware correction ===")
test_cases = [
    ("SPEAKER_01", "the authentication service its been really flaky in staging"),
    ("SPEAKER_01", "it requires node 18 or above so itll need to go through devops"),
    ("SPEAKER_02", "the mobile responsive version the breakpoints dont shift too much"),
    ("SPEAKER_00", "the analytics spec the conversion funnel events"),
    ("SPEAKER_00", "auth service fix is top priority the jwt library the old tokens arent being invalidated"),
]

for speaker, text in test_cases:
    corrected = context_correction_chain.invoke({"speaker": speaker, "text": text})
    print(f"\n  Speaker : {speaker}")
    print(f"  Before  : {text}")
    print(f"  After   : {corrected}")
print()


# ---------------------------------------------------------------------------
# 3. Apply to full transcript with RAG
# ---------------------------------------------------------------------------

print("=== 3. Full transcript context-aware correction ===")

import copy
improved_segments = copy.deepcopy(segments)
batch_results = context_correction_chain.batch(
    [{"speaker": s.get("speaker", "UNKNOWN"), "text": s["text"].strip()}
     for s in improved_segments[:10]],
    config={"max_concurrency": 3},
)

changes = 0
for seg, corrected in zip(improved_segments[:10], batch_results):
    if corrected.strip() != seg["text"].strip():
        seg["text"] = " " + corrected.strip()
        changes += 1

print(f"  Processed 10 segments, {changes} changed by context-aware correction")
print()


# ---------------------------------------------------------------------------
# 4. Demonstrate retrieval quality
# ---------------------------------------------------------------------------

print("=== 4. What context gets retrieved per segment? ===")
test_queries = [
    "jwt authentication token invalidation",
    "node version devops approval",
    "color palette accessibility breakpoints mobile",
]

for query in test_queries:
    docs = retriever.invoke(query)
    print(f"\n  Query: '{query}'")
    for doc in docs:
        print(f"    [{doc.metadata['category']}] {doc.page_content[:80]}...")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Build a context store before processing — participant names, glossary, projects
# ✅ RAG retrieval is per-segment: each segment fetches its own relevant context
# ✅ format_context() is the bridge between Documents and the prompt's {context}
# ✅ .batch() with max_concurrency=3 processes segments in parallel safely
# ✅ Small, focused documents beat one giant document — retrieval precision matters
# ✅ This context store grows over time: add more meetings, the model gets smarter
