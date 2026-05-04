"""
Stage 4, File 2: The RAG Pipeline
===================================
CONCEPT: Grounding LLM answers in your own documents.

Without RAG, the LLM can only use its training data (frozen at a cutoff date).
RAG lets you inject up-to-date, domain-specific knowledge at query time:

  User question
      │
      ▼
  Embed question → similarity search vector store → retrieve top-k chunks
      │
      ▼
  Inject chunks as context into prompt → LLM generates grounded answer

Pipeline stages implemented here:
  1. Load     — raw text (inline strings, simulating a document loader)
  2. Split    — RecursiveCharacterTextSplitter → smaller chunks
  3. Embed    — OllamaEmbeddings → vectors
  4. Store    — InMemoryVectorStore
  5. Retrieve — vector store as a LangChain Retriever
  6. Generate — ChatPromptTemplate + LLM + StrOutputParser

Key classes introduced:
  - RecursiveCharacterTextSplitter : splits long text into overlapping chunks
  - .as_retriever()                : turns a vector store into a Retriever Runnable
  - RunnablePassthrough            : passes the question through the chain unchanged
  - context formatter              : joins retrieved Document objects into a string

Run this file:
  uv run deep_research/deep_research_agent/stage_04_rag/02_retrieval_chain.py
"""

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, create_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


llm = create_chat_model(
    temperature=0,
    max_tokens=4096,
)

embeddings = create_embeddings()


# ---------------------------------------------------------------------------
# STEP 1 — Load: raw documents (simulating a real document loader)
# ---------------------------------------------------------------------------
# In production you'd use TextLoader, PDFLoader, WebBaseLoader, etc.
# Here we inline text so the tutorial is self-contained.

print("=== STEP 1: Load documents ===")

RAW_DOCUMENTS = [
    {
        "text": """
LangGraph is a library from LangChain for building stateful, multi-actor
applications with large language models. Unlike traditional LangChain chains
which are DAGs, LangGraph supports cyclic computation graphs essential for
creating autonomous agent loops.

LangGraph introduces a StateGraph class where developers define a shared state
schema using TypedDict. Nodes are Python functions that read from and write to
this state. Edges define the flow between nodes, including conditional edges
that allow dynamic routing based on state values.

The key advantage of LangGraph is persistence: state can be checkpointed using
a MemorySaver (in-memory) or SqliteSaver (disk) checkpointer, enabling agents
to pause, resume, and even time-travel through prior states.
        """,
        "source": "langgraph_overview.txt",
    },
    {
        "text": """
Retrieval-Augmented Generation (RAG) was introduced in a 2020 paper by
Lewis et al. at Facebook AI Research. RAG combines a parametric memory
(the neural network weights) with a non-parametric memory (a retrieval corpus).

In a RAG system, when a question is asked, the system first retrieves relevant
passages from a knowledge base, then conditions the language model's generation
on both the original question and the retrieved passages. This grounds the
model's answer in external, up-to-date knowledge.

RAG systems typically consist of: a document loader, a text splitter, an
embedding model, a vector store (such as FAISS, Chroma, or Pinecone), a
retriever, and finally a generation component.
        """,
        "source": "rag_paper_summary.txt",
    },
    {
        "text": """
LangChain's LCEL (LangChain Expression Language) is a declarative way to
compose Runnable components using the pipe operator (|). Every LangChain
component — prompt templates, LLMs, output parsers, retrievers, and custom
functions wrapped in RunnableLambda — implements the Runnable interface.

The canonical LCEL chain is: prompt | llm | output_parser
This reads as: "format the prompt, pass to the LLM, parse the output".

RunnableParallel runs multiple branches concurrently on the same input.
RunnablePassthrough passes input unchanged — crucial in RAG chains where
you need to keep the original question while adding retrieved context.
        """,
        "source": "lcel_guide.txt",
    },
    {
        "text": """
Vector databases are specialised storage systems optimised for storing and
querying high-dimensional embedding vectors. Unlike traditional databases that
use B-trees for indexing, vector databases use approximate nearest-neighbour
(ANN) algorithms such as HNSW (Hierarchical Navigable Small World) or IVF
(Inverted File Index).

Popular vector databases include:
  • FAISS (Facebook AI Similarity Search) — open source, runs in memory or on disk
  • Chroma — lightweight, developer-friendly, good for local prototyping
  • Pinecone — managed cloud service, production-grade scaling
  • Weaviate — open source, supports hybrid (keyword + vector) search
  • Qdrant — open source, Rust-based, high performance

For development and learning, InMemoryVectorStore from LangChain Core requires
no external dependencies and works identically to production stores.
        """,
        "source": "vector_databases.txt",
    },
]

raw_docs = [
    Document(page_content=d["text"].strip(), metadata={"source": d["source"]})
    for d in RAW_DOCUMENTS
]
print(f"Loaded {len(raw_docs)} documents")
print()


# ---------------------------------------------------------------------------
# STEP 2 — Split: chunk large documents into smaller pieces
# ---------------------------------------------------------------------------
# Why split? LLMs have context limits, and retrieving one giant document wastes
# tokens. Smaller chunks also improve retrieval precision.
#
# RecursiveCharacterTextSplitter splits on paragraphs, then sentences, then
# words — whichever keeps chunks within the target size.
#
#   chunk_size    : target characters per chunk
#   chunk_overlap : characters shared between adjacent chunks (preserves context)

print("=== STEP 2: Split into chunks ===")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    length_function=len,
)

chunks = splitter.split_documents(raw_docs)
print(f"Split {len(raw_docs)} documents into {len(chunks)} chunks")
print(f"Sample chunk ({len(chunks[0].page_content)} chars): {chunks[0].page_content[:120]}...")
print()


# ---------------------------------------------------------------------------
# STEP 3 & 4 — Embed and Store
# ---------------------------------------------------------------------------

print("=== STEP 3 & 4: Embed and store chunks ===")

vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
print(f"Embedded and stored {len(chunks)} chunks")
print()


# ---------------------------------------------------------------------------
# STEP 5 — Retriever
# ---------------------------------------------------------------------------
# .as_retriever() wraps the vector store as a LangChain Retriever — a Runnable
# that takes a string query and returns a list of Documents.

print("=== STEP 5: Retriever ===")

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},    # return top-3 most relevant chunks
)

# Test the retriever in isolation
test_results = retriever.invoke("How does LangGraph differ from LangChain?")
print(f"Retrieved {len(test_results)} chunks for test query:")
for doc in test_results:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:80]}...")
print()


# ---------------------------------------------------------------------------
# STEP 6 — RAG Chain
# ---------------------------------------------------------------------------
# The classic RAG LCEL chain:
#
#   {"context": retriever, "question": RunnablePassthrough()}
#       ↓
#   ChatPromptTemplate
#       ↓
#   LLM
#       ↓
#   StrOutputParser
#
# RunnablePassthrough() forwards the input string as-is to the "question" key.
# The retriever gets the same input string and returns Documents → context.

def format_docs(docs: list[Document]) -> str:
    """Join retrieved documents into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in docs
    )


rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the provided context. "
     "If the context does not contain enough information, say so.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {
        "context": retriever | format_docs,   # retrieve → format
        "question": RunnablePassthrough(),     # pass question through unchanged
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("=== STEP 6: RAG Chain ===")

questions = [
    "What is the main difference between LangGraph and regular LangChain?",
    "What are some popular vector databases and when would I choose each?",
    "What is the pipe operator in LCEL and how does it work?",
    "Who invented RAG and what problem does it solve?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer[:300]}")
print()


# ---------------------------------------------------------------------------
# BONUS — Compare RAG vs no-RAG
# ---------------------------------------------------------------------------
print("=== BONUS: RAG vs. no-RAG comparison ===")

specific_question = "What ANN algorithms do vector databases use, and name four specific databases?"

plain_chain = rag_prompt | llm | StrOutputParser()

print(f"Q: {specific_question}")
print("\n[Without RAG — LLM relies only on training data]")
no_rag_answer = plain_chain.invoke({
    "question": specific_question,
    "context": "(no context provided)",
})
print(no_rag_answer[:200])

print("\n[With RAG — answer grounded in our documents]")
rag_answer = rag_chain.invoke(specific_question)
print(rag_answer[:300])
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Split large docs into chunks (400-1000 chars) before embedding
# ✅ chunk_overlap prevents context loss at chunk boundaries
# ✅ retriever = vector_store.as_retriever() — integrates into LCEL chains
# ✅ RunnablePassthrough() lets you pass the question alongside retrieved context
# ✅ Always format retrieved Documents into a readable string before the prompt
# ✅ RAG answers are grounded in your documents, not just training data
