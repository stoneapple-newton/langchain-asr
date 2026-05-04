"""
Stage 7, File 4: Async Execution & FastAPI Serving
====================================================
CONCEPT: Serving your LangGraph agent as a real HTTP API with streaming.

All prior scripts are synchronous — fine for one-at-a-time local runs.
Production needs:
  • Async: handle many concurrent users without blocking threads
  • Streaming: send tokens to the browser as they arrive (not after)
  • HTTP API: so any frontend, mobile app, or service can call your agent

Two skills covered:
  A. Async LangChain/LangGraph with .ainvoke() / .astream()
  B. FastAPI server with a streaming SSE endpoint

How to run the server:
  uv run uvicorn deep_research.deep_research_agent.stage_07_production.04_async_serving:app --reload --port 8000

How to test it (in another terminal):
  curl -N http://localhost:8000/chat/stream?message=What+is+LangGraph
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"What is RAG?"}'

Or run the demo client at the bottom of this file:
  uv run deep_research/deep_research_agent/stage_07_production/04_async_serving.py
"""

import asyncio
from typing import AsyncIterator, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, get_settings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# A. Async LangChain — the basics
# ---------------------------------------------------------------------------
# Every LangChain Runnable has async equivalents:
#   .invoke()   → .ainvoke()
#   .stream()   → .astream()
#   .batch()    → .abatch()
#
# Use these whenever you're in an async context (FastAPI, asyncio, Jupyter).


async def demo_async_basics():
    """Show async invoke and streaming."""
    llm_async = create_chat_model(
        temperature=0,
        max_tokens=128,
    )

    chain = (
        ChatPromptTemplate.from_template("Explain {concept} in one sentence.")
        | llm_async
        | StrOutputParser()
    )

    print("=== A1. Async invoke (awaits full response) ===")
    result = await chain.ainvoke({"concept": "LangGraph async streaming"})
    print(result)
    print()

    print("=== A2. Async stream (tokens arrive as generated) ===")
    print("Streaming: ", end="", flush=True)
    async for chunk in chain.astream({"concept": "vector embeddings"}):
        print(chunk, end="", flush=True)
    print("\n")

    print("=== A3. Async batch (run multiple inputs concurrently) ===")
    inputs = [
        {"concept": "RAG"},
        {"concept": "LangGraph"},
        {"concept": "MemorySaver"},
    ]
    # abatch() runs all three concurrently — much faster than sequential
    results = await chain.abatch(inputs)
    for inp, res in zip(inputs, results):
        print(f"  [{inp['concept']}] {res}")
    print()


# ---------------------------------------------------------------------------
# B. Async LangGraph — the agent graph
# ---------------------------------------------------------------------------


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


llm_sync = create_chat_model(
    temperature=0.2,
    max_tokens=4096,
)
settings = get_settings()

SYSTEM = SystemMessage(content="You are a helpful assistant. Reply concisely.")


def chat_node(state: ChatState) -> dict:
    msgs = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [SYSTEM] + msgs
    return {"messages": [llm_sync.invoke(msgs)]}


checkpointer = MemorySaver()
builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile(checkpointer=checkpointer)


async def demo_async_graph():
    """Show async graph invocation and streaming."""
    config = {"configurable": {"thread_id": "async_demo"}}

    print("=== B1. Async graph invoke ===")
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is a checkpointer in LangGraph?")]},
        config=config,
    )
    print(result["messages"][-1].content)
    print()

    print("=== B2. Async graph stream (node-by-node updates) ===")
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content="How does streaming work in LangGraph?")]},
        config={"configurable": {"thread_id": "async_stream_demo"}},
        stream_mode="updates",
    ):
        node = list(chunk.keys())[0]
        msgs = chunk[node].get("messages", [])
        if msgs:
            print(f"  [{node}] {msgs[-1].content[:80]}...")
    print()


# ---------------------------------------------------------------------------
# C. FastAPI server with streaming SSE endpoint
# ---------------------------------------------------------------------------
# SSE (Server-Sent Events) is the standard way to stream text to browsers.
# The client reads a stream of "data: ..." lines until the connection closes.

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="LangChain Agent API", version="1.0")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Standard (non-streaming) chat endpoint."""
    config = {"configurable": {"thread_id": req.thread_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, thread_id=req.thread_id)


@app.get("/chat/stream")
async def chat_stream_endpoint(
    message: str = Query(..., description="User message"),
    thread_id: str = Query("default", description="Conversation thread ID"),
):
    """Streaming chat endpoint using Server-Sent Events."""
    config = {"configurable": {"thread_id": thread_id}}

    async def token_generator() -> AsyncIterator[str]:
        """Stream tokens as they arrive from the LLM."""
        # astream_events gives the finest-grained stream (individual tokens)
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = event["event"]
            # "on_chat_model_stream" fires for each token chunk
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # SSE format: "data: <text>\n\n"
                    yield f"data: {content}\n\n"

        yield "data: [DONE]\n\n"   # signal end of stream

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering
        },
    )


@app.get("/threads/{thread_id}/history")
async def get_history(thread_id: str):
    """Retrieve the full message history for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values:
        return {"thread_id": thread_id, "messages": []}

    messages = []
    for msg in state.values.get("messages", []):
        messages.append({
            "role": type(msg).__name__.replace("Message", "").lower(),
            "content": msg.content,
        })
    return {"thread_id": thread_id, "messages": messages}


@app.get("/health")
async def health():
    profile = settings.get_chat_profile()
    return {
        "status": "ok",
        "provider": profile.provider,
        "model": profile.model,
    }


# ---------------------------------------------------------------------------
# D. Demo client — run without starting the server
# ---------------------------------------------------------------------------

async def demo_client():
    """Run the async demos (no server needed)."""
    print("=" * 60)
    print("  PART A: Async LangChain basics")
    print("=" * 60)
    await demo_async_basics()

    print("=" * 60)
    print("  PART B: Async LangGraph")
    print("=" * 60)
    await demo_async_graph()

    print("=" * 60)
    print("  SERVER: Run with uvicorn to test the API")
    print("=" * 60)
    print("  uv run uvicorn deep_research.deep_research_agent.stage_07_production.04_async_serving:app --reload")
    print("  Then: curl http://localhost:8000/chat/stream?message=Hello")
    print()


# Run the async demo directly when executed as a script
if __name__ == "__main__":
    asyncio.run(demo_client())


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ .ainvoke() / .astream() / .abatch() — drop-in async replacements
# ✅ abatch() runs inputs concurrently — much faster than sequential loops
# ✅ astream_events with "on_chat_model_stream" gives token-level streaming
# ✅ FastAPI + StreamingResponse + SSE = standard browser-compatible streaming
# ✅ Run with: uvicorn <module>:app --reload (never python directly for FastAPI)
# ✅ Separate thread_ids per user = isolated conversations in one running server
