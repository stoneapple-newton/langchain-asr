"""
Stage 4, File 1: Embeddings & Vector Stores
=============================================
CONCEPT: Converting text to vectors to enable semantic similarity search.

An embedding is a list of numbers (a vector) that captures the *meaning* of
text. Texts with similar meanings produce vectors that are close together in
vector space, regardless of the exact words used.

This is the foundation of RAG:
  • Store knowledge as embeddings in a vector store
  • At query time, embed the question and find the nearest knowledge chunks
  • Feed those chunks as context to the LLM

Key classes introduced:
  - OllamaEmbeddings     : generates embeddings using a local Ollama model
  - InMemoryVectorStore  : in-process vector store (no extra infra needed)
  - Document             : a piece of text + metadata (page_content, metadata)
  - .similarity_search() : find the k nearest documents to a query string
  - .similarity_search_with_score() : same, but also returns relevance scores

Run this file:
  uv run deep_research/deep_research_agent/stage_04_rag/01_embeddings.py
"""

import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from ollama import ResponseError

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Create an embeddings model
# ---------------------------------------------------------------------------
# Chat and embedding models are often different in Ollama. Most chat-first
# models do NOT implement the embeddings endpoint, so use a dedicated
# embedding model such as nomic-embed-text or mxbai-embed-large.

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)


def embed_query_or_raise(text: str) -> list[float]:
    """Raise a setup-focused error when the Ollama embedding model is missing."""
    try:
        return embeddings.embed_query(text)
    except ResponseError as exc:
        if "does not support embeddings" in str(exc):
            raise RuntimeError(
                f"Ollama model '{EMBEDDING_MODEL}' does not support embeddings. "
                "Set OLLAMA_EMBEDDING_MODEL to an embedding model such as "
                "'nomic-embed-text' or 'mxbai-embed-large'."
            ) from exc
        raise RuntimeError(
            f"Failed to create embeddings with Ollama model '{EMBEDDING_MODEL}'. "
            f"Pull it first with: ollama pull {EMBEDDING_MODEL}"
        ) from exc

print("=== 1. Embed a single string ===")

vector = embed_query_or_raise("What is retrieval-augmented generation?")
print(f"Embedding dimensions : {len(vector)}")
print(f"First 5 values       : {[round(v, 4) for v in vector[:5]]}")
print()


# ---------------------------------------------------------------------------
# 2. Semantic similarity between vectors
# ---------------------------------------------------------------------------
# Similar texts produce vectors with high cosine similarity.
# You don't usually compute this manually — the vector store does it for you.

import math

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b)

print("=== 2. Semantic similarity (manual) ===")

pairs = [
    ("dog", "cat"),                         # similar
    ("dog", "automobile"),                  # dissimilar
    ("machine learning", "deep learning"),  # similar
    ("pizza", "quantum physics"),           # dissimilar
]

for text_a, text_b in pairs:
    vec_a = embed_query_or_raise(text_a)
    vec_b = embed_query_or_raise(text_b)
    sim = cosine_similarity(vec_a, vec_b)
    bar = "#" * int(sim * 20)
    print(f"  '{text_a}' vs '{text_b}': {sim:.3f}  {bar}")
print()


# ---------------------------------------------------------------------------
# 3. Build an in-memory vector store from documents
# ---------------------------------------------------------------------------
# Document has two fields:
#   page_content : the text that gets embedded and searched
#   metadata     : arbitrary dict (source, date, topic, etc.)

print("=== 3. Building a vector store ===")

documents = [
    Document(
        page_content="LangChain is a framework for building LLM applications using composable components.",
        metadata={"source": "langchain_docs", "topic": "langchain"},
    ),
    Document(
        page_content="LangGraph extends LangChain with stateful, cyclic computation graphs for agent loops.",
        metadata={"source": "langgraph_docs", "topic": "langgraph"},
    ),
    Document(
        page_content="RAG (Retrieval-Augmented Generation) combines a retriever with a language model to ground responses in external knowledge.",
        metadata={"source": "rag_paper", "topic": "rag"},
    ),
    Document(
        page_content="Embeddings are dense vector representations of text that capture semantic meaning.",
        metadata={"source": "ml_textbook", "topic": "embeddings"},
    ),
    Document(
        page_content="Ollama lets you run large language models locally on your own machine, without API costs.",
        metadata={"source": "ollama_docs", "topic": "ollama"},
    ),
    Document(
        page_content="A vector store indexes embeddings and supports fast approximate nearest-neighbour search.",
        metadata={"source": "ml_textbook", "topic": "vector_store"},
    ),
    Document(
        page_content="FAISS, Chroma, and Pinecone are popular vector databases used in production RAG systems.",
        metadata={"source": "ml_blog", "topic": "vector_store"},
    ),
    Document(
        page_content="Python is a general-purpose programming language widely used in data science and AI.",
        metadata={"source": "python_docs", "topic": "python"},
    ),
]

# InMemoryVectorStore.from_documents() embeds all documents immediately
vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
print(f"Stored {len(documents)} documents")
print()


# ---------------------------------------------------------------------------
# 4. Similarity search
# ---------------------------------------------------------------------------
# .similarity_search(query, k=N) returns the N most relevant Documents.
# The query is embedded on the fly and compared to all stored vectors.

print("=== 4. Similarity search ===")

queries = [
    "How do agent loops work?",
    "What tools can I use for storing vectors?",
    "How can I run LLMs without internet?",
]

for query in queries:
    results = vector_store.similarity_search(query, k=2)
    print(f"\nQuery: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] (topic={doc.metadata['topic']}) {doc.page_content[:80]}...")
print()


# ---------------------------------------------------------------------------
# 5. Similarity search with relevance scores
# ---------------------------------------------------------------------------
# Scores are cosine similarities in [0, 1] — higher means more relevant.

print("=== 5. Similarity search with scores ===")

query = "stateful graphs for building AI agents"
results_with_scores = vector_store.similarity_search_with_score(query, k=4)

print(f"Query: '{query}'")
for doc, score in results_with_scores:
    bar = "#" * int(score * 30)
    print(f"  {score:.3f} {bar}")
    print(f"         {doc.page_content[:90]}...")
print()


# ---------------------------------------------------------------------------
# 6. Filtering by metadata
# ---------------------------------------------------------------------------
# Most vector stores support metadata filters to narrow the search space.

print("=== 6. Metadata filtering ===")

# InMemoryVectorStore expects a callable filter, not a metadata dict.
def vector_store_topic_filter(doc: Document) -> bool:
    return doc.metadata.get("topic") == "vector_store"

filtered_results = vector_store.similarity_search(
    "databases and indexing",
    k=3,
    filter=vector_store_topic_filter,
)

print("Only searching documents where topic='vector_store':")
for doc in filtered_results:
    print(f"  • {doc.page_content[:90]}...")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# OK: Embeddings turn text into vectors; similar meanings end up nearby
# OK: Document = page_content (embedded) + metadata (for filtering)
# OK: InMemoryVectorStore needs no external infrastructure; good for prototyping
# OK: similarity_search() finds relevant chunks without exact keyword matching
# OK: Metadata filters let you pre-scope search to specific document subsets
