"""
Stage 4, File 3: Conversational RAG
=====================================
CONCEPT: RAG that remembers the conversation history.

The RAG chain from File 2 has a critical limitation: each question is answered
in isolation. If the user asks "What about its performance?" after asking about
FAISS, the retriever doesn't know what "its" refers to.

The fix: a history-aware retriever that uses the conversation history to
reformulate vague follow-up questions into standalone queries before retrieval.

Two-step architecture:
  Step 1 — Reformulate (if needed):
    history + follow-up question → LLM → standalone search query

  Step 2 — Retrieve + Generate:
    standalone query → retriever → context → LLM → answer

Key classes introduced:
  - create_history_aware_retriever : wraps a retriever with history awareness
  - create_stuff_documents_chain   : combines docs into a prompt for the LLM
  - create_retrieval_chain         : combines history-aware retriever + QA chain
  - LangGraph with MemorySaver     : manages conversation history per thread

Run this file:
  uv run deep_research/deep_research_agent/stage_04_rag/03_conversational_rag.py
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

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


# ---------------------------------------------------------------------------
# 1. Build the knowledge base (same documents as File 2)
# ---------------------------------------------------------------------------

TEXTS = [
    ("FAISS (Facebook AI Similarity Search) is an open-source library for "
     "efficient similarity search of dense vectors. It supports both CPU and "
     "GPU execution. FAISS uses IVF (Inverted File Index) and HNSW algorithms. "
     "It is the go-to choice for in-process vector search with no external dependencies.",
     "faiss_docs.txt"),

    ("Chroma is a lightweight, open-source vector database designed for "
     "developer productivity. It runs embedded in Python processes (no server "
     "needed) or as a client-server application. Chroma stores vectors, "
     "metadata, and documents together and is ideal for local RAG prototyping.",
     "chroma_docs.txt"),

    ("Pinecone is a managed, cloud-native vector database built for production "
     "scale. It handles infrastructure automatically and supports billions of "
     "vectors with low-latency queries. Pinecone offers hybrid search combining "
     "dense (semantic) and sparse (keyword) retrieval. It requires an API key.",
     "pinecone_docs.txt"),

    ("LangChain is a framework for building LLM applications. It provides "
     "abstractions for prompts, models, memory, agents, and tools. The "
     "LangChain Expression Language (LCEL) allows composing these components "
     "with the pipe operator. LangChain integrates with over 50 LLM providers.",
     "langchain_docs.txt"),

    ("LangGraph extends LangChain for stateful agent applications. It uses "
     "a StateGraph where nodes are processing functions and edges define "
     "transitions. LangGraph supports cycles (loops), conditional routing, "
     "and built-in checkpointing for conversation persistence.",
     "langgraph_docs.txt"),

    ("RAG combines retrieval and generation. The retriever finds relevant "
     "documents; the generator produces answers conditioned on those documents. "
     "Key metrics: precision (are retrieved docs relevant?), recall (are all "
     "relevant docs found?), and faithfulness (is the answer grounded in docs?).",
     "rag_concepts.txt"),
]

docs = [Document(page_content=t, metadata={"source": s}) for t, s in TEXTS]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print(f"Knowledge base ready: {len(chunks)} chunks from {len(docs)} documents")
print()


# ---------------------------------------------------------------------------
# 2. History-aware retriever
# ---------------------------------------------------------------------------
# Problem: follow-up questions like "What about its performance?" are useless
# as search queries — the retriever needs a standalone query.
#
# create_history_aware_retriever wraps our retriever with a "condense" step:
# if there's chat history, the LLM rewrites the question as a standalone query.

condense_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),   # the conversation so far
    ("human", "{input}"),
    ("human",
     "Given the above conversation, rephrase the follow-up question as a "
     "self-contained search query. Return ONLY the rephrased query."),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, condense_prompt
)


# ---------------------------------------------------------------------------
# 3. Question-answering chain
# ---------------------------------------------------------------------------

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer using ONLY the context below.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# create_stuff_documents_chain: formats retrieved docs into the {context} slot
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# create_retrieval_chain: history_aware_retriever + qa_chain wired together
# Input keys: {"input": question, "chat_history": [messages]}
# Output keys: {"answer": str, "context": [Documents]}
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


# ---------------------------------------------------------------------------
# 4. Wrap in LangGraph for automatic history management
# ---------------------------------------------------------------------------
# Without this, you'd have to manually build and pass chat_history every turn.
# LangGraph + MemorySaver handles it automatically per thread_id.

class ConvRAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def rag_node(state: ConvRAGState) -> dict:
    """Extract history and latest question, run the RAG chain."""
    messages = state["messages"]
    # Split: last message is the current question, prior messages are history
    chat_history = messages[:-1]
    question = messages[-1].content

    result = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history,
    })

    answer = result["answer"]
    sources = list({doc.metadata["source"] for doc in result.get("context", [])})
    source_note = f"\n[Sources: {', '.join(sources)}]" if sources else ""

    return {"messages": [AIMessage(content=answer + source_note)]}


checkpointer = MemorySaver()

builder = StateGraph(ConvRAGState)
builder.add_node("rag", rag_node)
builder.add_edge(START, "rag")
builder.add_edge("rag", END)
conv_rag_app = builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 5. Multi-turn conversation demo
# ---------------------------------------------------------------------------

def chat(app, thread_id: str, question: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    return result["messages"][-1].content


print("=" * 60)
print("  Multi-turn conversation (thread: user_1)")
print("=" * 60)

thread = "user_1"

questions = [
    "What is FAISS and what algorithms does it use?",
    "How does it compare to Chroma?",           # "it" refers to FAISS
    "Which one would you recommend for local development?",
    "What about for production at large scale?",
    "How does LangGraph help with building RAG systems?",
]

for q in questions:
    print(f"\n[User]  {q}")
    answer = chat(conv_rag_app, thread, q)
    print(f"[Agent] {answer[:300]}")
print()


# ---------------------------------------------------------------------------
# 6. Verify thread isolation
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Thread isolation: new conversation, no shared history")
print("=" * 60)

print(f"\n[User (new thread)]  What algorithms does it use?")
isolated_answer = chat(conv_rag_app, "user_2", "What algorithms does it use?")
print(f"[Agent] {isolated_answer[:250]}")
print()
print("↑ Different thread has no context about FAISS — 'it' is ambiguous here")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ create_history_aware_retriever reformulates vague follow-ups into good queries
# ✅ MessagesPlaceholder slots chat_history into prompts
# ✅ create_retrieval_chain wires the history-aware retriever + QA chain together
# ✅ LangGraph + MemorySaver eliminates manual history tracking
# ✅ thread_id isolates conversations — different users, different histories
