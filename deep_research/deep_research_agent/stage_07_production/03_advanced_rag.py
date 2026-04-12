"""
Stage 7, File 3: Advanced RAG
===============================
CONCEPT: Improving RAG quality beyond basic similarity search.

Basic RAG (Stage 4) has several weaknesses:
  1. Retrieval precision: top-k by cosine similarity often returns noisy results
  2. Context window waste: full chunks may contain irrelevant sentences
  3. Keyword mismatch: "LLM" vs "large language model" may not match semantically
  4. Ephemeral storage: InMemoryVectorStore is lost on restart

This file covers the production-grade upgrades:

  • Chroma           : persistent, file-backed vector store (replaces InMemory)
  • Reranking        : score and re-sort retrieved chunks with a cross-encoder
  • Hybrid search    : BM25 keyword retrieval + vector retrieval, merged results
  • Contextual compression : trim chunks to only the relevant sentences
  • Multi-query retrieval  : generate multiple query variants for better recall

Run this file:
  uv run deep_research/deep_research_agent/stage_07_production/03_advanced_rag.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

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

CHROMA_DIR = Path(__file__).parent / "chroma_db"   # persists to disk


# ---------------------------------------------------------------------------
# Shared knowledge base
# ---------------------------------------------------------------------------

DOCUMENTS = [
    Document(page_content="FAISS is an in-process vector library from Facebook. It uses IVF and HNSW algorithms for fast approximate nearest-neighbour search. It supports both CPU and GPU and has no server overhead.", metadata={"source": "faiss", "category": "vector_db"}),
    Document(page_content="Chroma is a lightweight embedded vector database designed for developer productivity. It stores vectors, metadata, and documents together. Chroma can run in-process or as a client-server service.", metadata={"source": "chroma", "category": "vector_db"}),
    Document(page_content="Pinecone is a managed cloud vector database. It automatically handles infrastructure, supports billions of vectors, and offers hybrid (dense + sparse) search. It requires an API key and has a free tier.", metadata={"source": "pinecone", "category": "vector_db"}),
    Document(page_content="Qdrant is an open-source vector database written in Rust. It is known for high throughput and offers both in-memory and disk-based storage. Qdrant has native support for payload filtering.", metadata={"source": "qdrant", "category": "vector_db"}),
    Document(page_content="Reranking improves retrieval quality by applying a cross-encoder model to score (query, document) pairs. Unlike bi-encoders used for initial retrieval, cross-encoders see both query and document together, giving more accurate relevance scores.", metadata={"source": "ml_blog", "category": "rag_technique"}),
    Document(page_content="Hybrid search combines dense (vector) retrieval with sparse (BM25 keyword) retrieval. The results are merged using Reciprocal Rank Fusion (RRF). This improves recall for exact terms like product codes, names, and acronyms.", metadata={"source": "ml_blog", "category": "rag_technique"}),
    Document(page_content="Contextual compression filters retrieved documents to keep only the sentences relevant to the query. This reduces noise, saves tokens, and improves answer quality.", metadata={"source": "langchain_docs", "category": "rag_technique"}),
    Document(page_content="Multi-query retrieval generates multiple paraphrases of the user's question, retrieves for each, then deduplicates results. This improves recall when the original query is ambiguous or phrased in an unusual way.", metadata={"source": "langchain_docs", "category": "rag_technique"}),
    Document(page_content="LangGraph adds cycles and persistence to LangChain. It uses a StateGraph where TypedDict defines shared state. Nodes are Python functions. Edges can be conditional, enabling dynamic routing and loops.", metadata={"source": "langgraph", "category": "framework"}),
    Document(page_content="Large language models have a context window limit — the maximum number of tokens they can process at once. GPT-4 Turbo supports 128k tokens; Claude 3 supports 200k; Gemini 1.5 supports 1 million tokens.", metadata={"source": "llm_comparison", "category": "llm"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
chunks = splitter.split_documents(DOCUMENTS)


# ---------------------------------------------------------------------------
# UPGRADE 1 — Chroma: persistent vector store
# ---------------------------------------------------------------------------
# InMemoryVectorStore is lost on restart. Chroma persists to disk.
# The second time you run this script, the store is loaded from disk instantly.

print("=" * 60)
print("  UPGRADE 1: Chroma — persistent vector store")
print("=" * 60)

from langchain_chroma import Chroma

if CHROMA_DIR.exists():
    # Load existing store from disk
    chroma_store = Chroma(
        collection_name="stage7_knowledge",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    count = chroma_store._collection.count()
    print(f"  Loaded existing Chroma store ({count} vectors from {CHROMA_DIR.name}/)")
else:
    # Create and populate on first run
    chroma_store = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="stage7_knowledge",
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  Created Chroma store with {len(chunks)} chunks → {CHROMA_DIR.name}/")

retriever = chroma_store.as_retriever(search_kwargs={"k": 5})

basic_results = retriever.invoke("What vector databases run without a server?")
print(f"  Basic retrieval returned {len(basic_results)} chunks")
print()


# ---------------------------------------------------------------------------
# UPGRADE 2 — Reranking: sort by cross-encoder relevance
# ---------------------------------------------------------------------------
# Retrieve more (k=10) then rerank down to top-3.
# Cross-encoders read (query + document) together → more accurate than cosine.

print("=" * 60)
print("  UPGRADE 2: Reranking with FlashRank")
print("=" * 60)

try:
    from langchain_community.document_compressors import FlashrankRerank
    from langchain.retrievers import ContextualCompressionRetriever

    compressor = FlashrankRerank(top_n=3)   # keep top 3 after reranking
    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=chroma_store.as_retriever(search_kwargs={"k": 8}),
    )

    query = "Which vector database is best for production scale?"
    reranked = reranking_retriever.invoke(query)
    print(f"  Query: '{query}'")
    for i, doc in enumerate(reranked, 1):
        relevance = getattr(doc, "metadata", {}).get("relevance_score", "N/A")
        print(f"  [{i}] score={relevance} | {doc.page_content[:80]}...")
    print()
except ImportError:
    print("  (FlashRank not installed — run: uv add flashrank)")
    reranking_retriever = retriever
    print()


# ---------------------------------------------------------------------------
# UPGRADE 3 — Contextual compression: trim chunks to relevant sentences
# ---------------------------------------------------------------------------
# LLMChainExtractor keeps only the sentences that answer the query.
# Reduces token usage and removes noise from large chunks.

print("=" * 60)
print("  UPGRADE 3: Contextual compression")
print("=" * 60)

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

extractor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=extractor,
    base_retriever=chroma_store.as_retriever(search_kwargs={"k": 4}),
)

query = "What algorithms do vector databases use internally?"
print(f"  Query: '{query}'")

raw_docs = retriever.invoke(query)
compressed_docs = compression_retriever.invoke(query)

total_raw_chars = sum(len(d.page_content) for d in raw_docs)
total_compressed_chars = sum(len(d.page_content) for d in compressed_docs)

print(f"  Raw chunks:        {len(raw_docs)} docs, {total_raw_chars} chars")
print(f"  Compressed chunks: {len(compressed_docs)} docs, {total_compressed_chars} chars")
print(f"  Token reduction:   ~{100 - int(total_compressed_chars/max(total_raw_chars,1)*100)}%")
for doc in compressed_docs:
    print(f"  → {doc.page_content[:100]}")
print()


# ---------------------------------------------------------------------------
# UPGRADE 4 — Multi-query retrieval: expand the query for better recall
# ---------------------------------------------------------------------------
# Generates 3 paraphrases of the question → retrieves for each → deduplicates.
# Helps when the original query is phrased in an unusual way.

print("=" * 60)
print("  UPGRADE 4: Multi-query retrieval")
print("=" * 60)

from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=chroma_store.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
)

query = "embedded databases for AI"
print(f"  Query: '{query}'")
multi_results = multi_query_retriever.invoke(query)
print(f"  Retrieved {len(multi_results)} unique docs (with auto-deduplication)")
for doc in multi_results[:3]:
    print(f"  → [{doc.metadata.get('source')}] {doc.page_content[:80]}...")
print()


# ---------------------------------------------------------------------------
# UPGRADE 5 — Putting it all together: a production-quality RAG chain
# ---------------------------------------------------------------------------

print("=" * 60)
print("  UPGRADE 5: Production RAG chain (Chroma + reranking + compression)")
print("=" * 60)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in docs
    )

# Use the best available retriever from the upgrades above
best_retriever = reranking_retriever  # reranking retriever from upgrade 2

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the provided context. "
     "Cite sources using [source_name] notation.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

production_rag_chain = (
    {"context": best_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Compare FAISS and Chroma for local development",
    "How does reranking improve RAG quality?",
    "Which database would you choose for a production system with billions of vectors?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = production_rag_chain.invoke(q)
    print(f"A: {answer[:300]}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Chroma → persistent vector store; swap InMemoryVectorStore in one line
# ✅ Reranking: retrieve broad (k=8-10), rerank to top-3 → precision++
# ✅ ContextualCompressionRetriever wraps ANY retriever — composable pattern
# ✅ LLMChainExtractor trims chunks to relevant sentences → token savings
# ✅ MultiQueryRetriever generates paraphrases → recall++ for edge cases
# ✅ All retrievers implement the same Runnable interface — mix and match freely
