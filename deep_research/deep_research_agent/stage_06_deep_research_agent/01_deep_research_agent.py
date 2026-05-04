"""
Stage 6: Deep Research Agent
==============================
CONCEPT: Combining everything — a multi-step research agent that plans,
searches, synthesises, and writes a structured report.

This file brings together every concept from the previous stages:
  Stage 1 → LCEL chains, structured output with Pydantic
  Stage 2 → Web search tools (Tavily / DuckDuckGo)
  Stage 3 → LangGraph StateGraph, conditional edges, tool loops
  Stage 4 → Embeddings + InMemoryVectorStore for de-duplicating findings
  Stage 5 → MemorySaver for checkpointing the research session

Agent workflow (graph nodes):
  ┌─────────┐   ┌────────┐   ┌────────┐   ┌──────────┐   ┌────────┐
  │  plan   │──▶│ search │──▶│ store  │──▶│ analyse  │──▶│ report │
  └─────────┘   └────────┘   └────────┘   └──────────┘   └────────┘
                     ▲             │
                     └─── loop ────┘  (until all sub-questions searched)

  plan     : break the topic into 3-4 research sub-questions
  search   : run a web search for the current sub-question
  store    : embed search result into a mini vector store
  analyse  : decide if we have enough info or need more searches
  report   : synthesise all findings into a structured markdown report

Run this file:
  uv run deep_research/deep_research_agent/stage_06_deep_research_agent/01_deep_research_agent.py

Optional: set TAVILY_API_KEY in .env for higher quality search results.
"""

import json
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, create_embeddings, get_settings, structured_output_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

llm = create_chat_model(
    temperature=0,
    max_tokens=4096,
)

embeddings = create_embeddings()
settings = get_settings()

# Pick best available search tool
if settings.search.has_tavily_api_key:
    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(max_results=3, include_answer=True)
    print("Search: Tavily ✅")
else:
    search_tool = DuckDuckGoSearchRun()
    print("Search: DuckDuckGo (set SEARCH__TAVILY_API_KEY or TAVILY_API_KEY for better results)")


# ---------------------------------------------------------------------------
# Pydantic schemas for structured output
# ---------------------------------------------------------------------------

class ResearchPlan(BaseModel):
    """The LLM returns this after reading the topic."""
    topic: str = Field(description="The research topic, cleaned and clarified")
    sub_questions: list[str] = Field(description="3-4 specific sub-questions to research")
    context: str = Field(description="Why this topic matters — 1 sentence")


class AnalysisDecision(BaseModel):
    """The LLM returns this when deciding whether to continue or stop."""
    sufficient: bool = Field(description="True if we have enough to write a report")
    reason: str = Field(description="Why we do or don't have enough information")
    next_question_index: int = Field(
        description="Index of next sub-question to search (if not sufficient)"
    )


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    # Input
    topic: str

    # Plan phase
    research_plan: dict            # ResearchPlan as a dict
    sub_questions: list[str]       # extracted for easy access

    # Search phase
    current_question_idx: int      # which sub-question we're currently searching
    search_results: list[str]      # accumulated raw search results (one per question)
    searches_completed: int        # how many searches done

    # Analysis
    sufficient: bool               # did the analyser say we have enough?

    # Report
    final_report: str              # the finished markdown report

    # Audit trail
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Node 1: plan
# ---------------------------------------------------------------------------

plan_parser = JsonOutputParser()

plan_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research planner. Given a topic, generate a focused research plan.\n"
     "Respond with valid JSON matching this schema:\n{format_instructions}"),
    ("human", "Research topic: {topic}"),
]).partial(format_instructions=plan_parser.get_format_instructions())

plan_chain = structured_output_chain(llm, plan_prompt, ResearchPlan)


def plan_node(state: ResearchState) -> dict:
    """Break the research topic into targeted sub-questions."""
    print(f"\n{'═'*60}")
    print(f"  [PLAN] Topic: {state['topic']}")
    print(f"{'═'*60}")

    plan = plan_chain.invoke({"topic": state["topic"]})

    print(f"  Context: {plan.get('context', '')}")
    print(f"  Sub-questions:")
    for i, q in enumerate(plan.get("sub_questions", []), 1):
        print(f"    {i}. {q}")

    return {
        "research_plan": plan,
        "sub_questions": plan.get("sub_questions", []),
        "current_question_idx": 0,
        "search_results": [],
        "searches_completed": 0,
        "sufficient": False,
        "messages": [HumanMessage(content=f"Researching: {state['topic']}")],
    }


# ---------------------------------------------------------------------------
# Node 2: search
# ---------------------------------------------------------------------------

def search_node(state: ResearchState) -> dict:
    """Search the web for the current sub-question."""
    idx = state["current_question_idx"]
    questions = state["sub_questions"]

    if idx >= len(questions):
        return {}

    question = questions[idx]
    print(f"\n  [SEARCH #{idx + 1}] {question}")

    try:
        raw = search_tool.invoke({"query": question})
        result_text = str(raw)[:2000]   # cap to save tokens
    except Exception as e:
        result_text = f"Search failed: {e}"

    # Attach the question to the result for context
    annotated = f"Q: {question}\n\nResults:\n{result_text}"
    print(f"  Retrieved {len(result_text)} characters")

    updated_results = state.get("search_results", []) + [annotated]

    return {
        "search_results": updated_results,
        "searches_completed": state.get("searches_completed", 0) + 1,
        "messages": [HumanMessage(content=f"Searched: {question}")],
    }


# ---------------------------------------------------------------------------
# Node 3: store (embed search results for de-duplication)
# ---------------------------------------------------------------------------

# Module-level vector store — accumulates across the research session
_vector_store: InMemoryVectorStore | None = None


def store_node(state: ResearchState) -> dict:
    """Embed the latest search result and store it in the vector store."""
    global _vector_store

    results = state.get("search_results", [])
    if not results:
        return {}

    latest = results[-1]
    idx = state["current_question_idx"]

    doc = Document(
        page_content=latest,
        metadata={"question_idx": idx, "question": state["sub_questions"][idx]},
    )

    if _vector_store is None:
        _vector_store = InMemoryVectorStore.from_documents([doc], embeddings)
    else:
        _vector_store.add_documents([doc])

    print(f"  [STORE]  Embedded result #{idx + 1} into vector store")
    return {}


# ---------------------------------------------------------------------------
# Node 4: analyse
# ---------------------------------------------------------------------------

analysis_parser = JsonOutputParser()

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research quality assessor. Decide if the collected information "
     "is sufficient to write a comprehensive report on the topic.\n"
     "Consider: coverage of sub-questions, depth of information, any critical gaps.\n"
     "Respond with valid JSON:\n{format_instructions}"),
    ("human",
     "Topic: {topic}\n\n"
     "Sub-questions to cover:\n{sub_questions}\n\n"
     "Information collected so far:\n{collected}\n\n"
     "Searches done: {done} / {total}"),
]).partial(format_instructions=analysis_parser.get_format_instructions())

analysis_chain = structured_output_chain(llm, analysis_prompt, AnalysisDecision)


def analyse_node(state: ResearchState) -> dict:
    """Decide: do we have enough information, or should we search more?"""
    questions = state["sub_questions"]
    done = state.get("searches_completed", 0)
    total = len(questions)

    # Summarise collected information (truncate to save tokens)
    collected_summary = "\n\n".join(
        f"[Result {i+1}] {r[:400]}" for i, r in enumerate(state.get("search_results", []))
    )

    print(f"\n  [ANALYSE] {done}/{total} sub-questions searched")

    try:
        decision = analysis_chain.invoke({
            "topic": state["topic"],
            "sub_questions": "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions)),
            "collected": collected_summary,
            "done": done,
            "total": total,
        })

        sufficient = decision.get("sufficient", False)
        reason = decision.get("reason", "")
        next_idx = decision.get("next_question_index", done)

        print(f"  Sufficient: {sufficient}")
        print(f"  Reason:     {reason[:100]}")

    except Exception:
        # If parsing fails, fall back to simple heuristic
        sufficient = done >= total
        next_idx = done

    return {
        "sufficient": sufficient,
        "current_question_idx": next_idx if not sufficient else state["current_question_idx"],
    }


# ---------------------------------------------------------------------------
# Node 5: report
# ---------------------------------------------------------------------------

report_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research writer. Write a clear, well-structured research report "
     "in markdown format. Include:\n"
     "  # Title\n"
     "  ## Overview (2-3 sentences)\n"
     "  ## Key Findings (one section per sub-question, with bullet points)\n"
     "  ## Summary\n"
     "Base the report ONLY on the provided research findings."),
    ("human",
     "Topic: {topic}\n\n"
     "Research findings:\n{findings}"),
])

report_chain = report_prompt | llm | StrOutputParser()


def report_node(state: ResearchState) -> dict:
    """Write the final markdown research report."""
    print(f"\n  [REPORT] Writing final report...")

    findings = "\n\n".join(
        f"=== Finding {i+1} ===\n{r}"
        for i, r in enumerate(state.get("search_results", []))
    )

    report = report_chain.invoke({
        "topic": state["topic"],
        "findings": findings[:4000],   # cap to avoid context overflow
    })

    return {
        "final_report": report,
        "messages": [HumanMessage(content="Research complete. Report written.")],
    }


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

MAX_SEARCHES = 6  # absolute safety cap


def route_after_analyse(state: ResearchState) -> str:
    """Route to 'search' (next sub-question) or 'report' (done)."""
    done = state.get("searches_completed", 0)
    total = len(state.get("sub_questions", []))
    sufficient = state.get("sufficient", False)

    if sufficient or done >= total or done >= MAX_SEARCHES:
        return "report"
    return "search"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(ResearchState)

builder.add_node("plan",    plan_node)
builder.add_node("search",  search_node)
builder.add_node("store",   store_node)
builder.add_node("analyse", analyse_node)
builder.add_node("report",  report_node)

builder.add_edge(START,     "plan")
builder.add_edge("plan",    "search")
builder.add_edge("search",  "store")
builder.add_edge("store",   "analyse")
builder.add_edge("report",  END)

# After analyse: either search the next sub-question or write the report
builder.add_conditional_edges(
    "analyse",
    route_after_analyse,
    {"search": "search", "report": "report"},
)

checkpointer = MemorySaver()
research_agent = builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

def run_research(topic: str, thread_id: str = "research_session") -> str:
    """Run a full research cycle on the given topic."""
    global _vector_store
    _vector_store = None   # reset vector store for new research session

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'█'*60}")
    print(f"  DEEP RESEARCH AGENT")
    print(f"  Topic: {topic}")
    print(f"{'█'*60}")

    final_state = research_agent.invoke(
        {"topic": topic},
        config=config,
    )

    return final_state.get("final_report", "No report generated.")


# ---------------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------------

RESEARCH_TOPIC = "How do large language models handle long-context reasoning?"

report = run_research(RESEARCH_TOPIC)

print("\n" + "═" * 60)
print("  FINAL REPORT")
print("═" * 60)
print(report)

# ---------------------------------------------------------------------------
# BONUS: Query the accumulated knowledge with RAG
# ---------------------------------------------------------------------------
if _vector_store is not None:
    print("\n" + "═" * 60)
    print("  BONUS: Query gathered knowledge via vector search")
    print("═" * 60)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based only on the provided research context.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()

    follow_up = "What are the main limitations mentioned in the research?"
    retriever = _vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(follow_up)
    context = "\n\n".join(d.page_content[:500] for d in docs)

    answer = rag_chain.invoke({"question": follow_up, "context": context})
    print(f"\nQ: {follow_up}")
    print(f"A: {answer[:400]}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Multi-node graphs can represent complex, multi-step workflows
# ✅ Conditional edges with a loop (analyse → search → analyse) drive iteration
# ✅ Pydantic models enforce structured output at every decision point
# ✅ A module-level vector store accumulates knowledge across the research loop
# ✅ MemorySaver checkpoints every step — you can inspect any intermediate state
# ✅ This pattern (plan → search loop → synthesise) is the foundation of
#    production research agents like Perplexity, GPT Researcher, and STORM
