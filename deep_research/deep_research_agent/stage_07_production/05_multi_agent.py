"""
Stage 7, File 5: Multi-Agent Systems
======================================
CONCEPT: Networks of specialised agents collaborating on complex tasks.

Single agents work well for focused tasks. Complex tasks benefit from
specialisation — just like software teams have dedicated roles.

Two fundamental multi-agent patterns:

  SUPERVISOR pattern (top-down orchestration):
    ┌────────────┐
    │ Supervisor │  ← decides which sub-agent to call next
    └────────────┘
         │  ▲
    ┌────┘  └───────────────────┐
    ▼                           ▼
  [researcher]  [writer]  [critic]   ← specialised sub-agents

  HANDOFF pattern (peer-to-peer routing):
    [agent A] ──handoff──▶ [agent B] ──handoff──▶ [agent C]
                                                       │
                                                      END

Key LangGraph concepts introduced:
  - Subgraph             : a compiled graph used as a node in a parent graph
  - Command(goto=)       : dynamic routing to any node (including across graphs)
  - Supervisor node      : an LLM that selects the next worker and routes to it
  - Structured tool call : workers exposed as tools to the supervisor LLM

Run this file:
  uv run deep_research/deep_research_agent/stage_07_production/05_multi_agent.py
"""

import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=512,
)


# ===========================================================================
# PATTERN 1 — Supervisor with worker agents
# ===========================================================================
# The supervisor LLM sees the task and the list of workers.
# It decides which worker to call, calls it as a "tool", reads the result,
# and decides next steps — until it determines the task is complete.

print("=" * 60)
print("  PATTERN 1: Supervisor + Worker Agents")
print("=" * 60)


# --- Worker agents as tools -----------------------------------------------
# Each worker is exposed as a @tool so the supervisor can call them.
# The tool docstring is critical — the supervisor reads it to know what each worker does.

@tool
def researcher(query: str) -> str:
    """Research a topic and return a factual summary.
    Use this for fact-finding, background research, and gathering information.
    Input: a specific research question or topic."""
    prompt = ChatPromptTemplate.from_template(
        "You are a research specialist. Provide a factual, concise summary about: {query}"
    )
    return (prompt | llm | StrOutputParser()).invoke({"query": query})


@tool
def writer(content_brief: str) -> str:
    """Write polished content based on a brief or research findings.
    Use this when you have information and need it structured into readable text.
    Input: a description of what to write, including key points to include."""
    prompt = ChatPromptTemplate.from_template(
        "You are a professional writer. Write clear, engaging content based on this brief:\n{content_brief}"
    )
    return (prompt | llm | StrOutputParser()).invoke({"content_brief": content_brief})


@tool
def critic(text_to_review: str) -> str:
    """Review and critique a piece of writing for accuracy, clarity, and completeness.
    Use this to quality-check work before finalising.
    Input: the text to review."""
    prompt = ChatPromptTemplate.from_template(
        "You are an editorial critic. Review this text and provide brief, constructive feedback:\n{text_to_review}"
    )
    return (prompt | llm | StrOutputParser()).invoke({"text_to_review": text_to_review})


workers = [researcher, writer, critic]
supervisor_llm = llm.bind_tools(workers)


# --- Supervisor state and nodes -------------------------------------------

class SupervisorState(TypedDict):
    task: str
    messages: Annotated[list, add_messages]
    iterations: int    # guard against infinite loops


MAX_ITERATIONS = 6

SUPERVISOR_SYSTEM = (
    "You are a task supervisor coordinating a team of specialists.\n"
    "Workers available:\n"
    "  - researcher : for gathering facts and information\n"
    "  - writer     : for drafting polished text\n"
    "  - critic     : for reviewing and improving work\n\n"
    "Workflow: research first, then write, then optionally critique and revise.\n"
    "When the task is fully complete, respond with your final answer directly (no tool call)."
)


def supervisor_node(state: SupervisorState) -> dict:
    """The supervisor decides what to do next."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + messages

    response = supervisor_llm.invoke(messages)

    if response.tool_calls:
        calls = [f"{tc['name']}({list(tc['args'].values())[0][:40]}...)" for tc in response.tool_calls]
        print(f"  [supervisor] → calling: {calls}")
    else:
        print(f"  [supervisor] → final answer ready")

    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
    }


def should_continue(state: SupervisorState) -> str:
    last = state["messages"][-1]
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


worker_node = ToolNode(workers)

s1 = StateGraph(SupervisorState)
s1.add_node("supervisor", supervisor_node)
s1.add_node("tools", worker_node)
s1.add_edge(START, "supervisor")
s1.add_conditional_edges("supervisor", should_continue)
s1.add_edge("tools", "supervisor")   # results go back to supervisor

supervisor_app = s1.compile(checkpointer=MemorySaver())


def run_supervisor(task: str) -> str:
    config = {"configurable": {"thread_id": f"sup_{hash(task)}"}}
    result = supervisor_app.invoke(
        {"task": task, "messages": [HumanMessage(content=task)], "iterations": 0},
        config=config,
    )
    # Walk back to find the last non-tool-call message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", []):
            return msg.content
    return result["messages"][-1].content


task = "Research what LangGraph is, then write a 3-bullet summary suitable for a README, and critique it."
print(f"Task: {task}\n")
output = run_supervisor(task)
print(f"\nFinal output:\n{output}")
print()


# ===========================================================================
# PATTERN 2 — Handoff pattern (agent-to-agent routing)
# ===========================================================================
# Agents transfer control to each other using Command(goto=target).
# Each agent decides when it's done with its part and who should take over.
# No central supervisor — agents are peers.

print("=" * 60)
print("  PATTERN 2: Peer-to-peer handoff")
print("=" * 60)


class PipelineState(TypedDict):
    topic: str
    research: str
    draft: str
    final: str
    messages: Annotated[list, add_messages]


def research_agent(state: PipelineState) -> Command:
    """Specialised researcher — hands off to writer when done."""
    print(f"  [research_agent] Researching: {state['topic']}")

    prompt = ChatPromptTemplate.from_template(
        "Research this topic and list 4 key facts: {topic}"
    )
    research = (prompt | llm | StrOutputParser()).invoke({"topic": state["topic"]})

    print(f"  [research_agent] → handing off to writer_agent")

    # Command(goto=) routes to any node by name
    return Command(
        goto="writer_agent",
        update={
            "research": research,
            "messages": [AIMessage(content=f"Research complete: {research[:80]}...")],
        },
    )


def writer_agent(state: PipelineState) -> Command:
    """Specialised writer — hands off to editor when done."""
    print(f"  [writer_agent]   Writing draft from research")

    prompt = ChatPromptTemplate.from_template(
        "Using these research findings, write a short paragraph (3-4 sentences):\n{research}"
    )
    draft = (prompt | llm | StrOutputParser()).invoke({"research": state["research"]})

    print(f"  [writer_agent]   → handing off to editor_agent")

    return Command(
        goto="editor_agent",
        update={
            "draft": draft,
            "messages": [AIMessage(content=f"Draft complete: {draft[:80]}...")],
        },
    )


def editor_agent(state: PipelineState) -> Command:
    """Specialised editor — terminal node, routes to END."""
    print(f"  [editor_agent]   Polishing draft")

    prompt = ChatPromptTemplate.from_template(
        "Polish this paragraph for clarity and flow. Return only the improved version:\n{draft}"
    )
    final = (prompt | llm | StrOutputParser()).invoke({"draft": state["draft"]})

    print(f"  [editor_agent]   → done, routing to END")

    return Command(
        goto=END,
        update={
            "final": final,
            "messages": [AIMessage(content="Editing complete.")],
        },
    )


s2 = StateGraph(PipelineState)
s2.add_node("research_agent", research_agent)
s2.add_node("writer_agent",   writer_agent)
s2.add_node("editor_agent",   editor_agent)
s2.add_edge(START, "research_agent")
# No add_edge() between agents — Command(goto=) handles routing dynamically

pipeline_app = s2.compile()

pipeline_result = pipeline_app.invoke({
    "topic": "how Retrieval-Augmented Generation works",
    "research": "", "draft": "", "final": "",
    "messages": [],
})

print(f"\nFinal polished output:")
print(pipeline_result["final"])
print()


# ===========================================================================
# PATTERN 3 — Parallel subgraph execution
# ===========================================================================
# Run multiple specialised subgraphs in parallel, then merge their outputs.
# This is the most powerful pattern for speed: independent tasks run concurrently.

print("=" * 60)
print("  PATTERN 3: Parallel agent execution")
print("=" * 60)

from langgraph.types import Send


class ParallelState(TypedDict):
    topic: str
    perspectives: list[str]   # collected from parallel branches
    synthesis: str


def create_perspective(angle: str, topic: str) -> str:
    """Generate one perspective on the topic."""
    prompt = ChatPromptTemplate.from_template(
        "In 2 sentences, describe {topic} from the perspective of a {angle}."
    )
    return (prompt | llm | StrOutputParser()).invoke({"topic": topic, "angle": angle})


ANGLES = ["software engineer", "product manager", "end user"]


def fan_out(state: ParallelState) -> list[Send]:
    """Dispatch one task per perspective in parallel."""
    return [
        Send("generate_perspective", {"angle": angle, "topic": state["topic"]})
        for angle in ANGLES
    ]


class PerspectiveInput(TypedDict):
    angle: str
    topic: str


def generate_perspective_node(state: PerspectiveInput) -> dict:
    """One parallel branch — generates a single perspective."""
    print(f"  [parallel] Generating {state['angle']} perspective...")
    perspective = create_perspective(state["angle"], state["topic"])
    return {"perspectives": [perspective]}   # add_messages-style accumulation


def synthesise(state: ParallelState) -> dict:
    """Merge all parallel perspectives into one synthesis."""
    combined = "\n".join(f"• {p}" for p in state["perspectives"])
    prompt = ChatPromptTemplate.from_template(
        "Synthesise these perspectives into a single cohesive paragraph:\n{combined}"
    )
    synthesis = (prompt | llm | StrOutputParser()).invoke({"combined": combined})
    return {"synthesis": synthesis}


# ParallelState needs perspectives to accumulate (not overwrite)
class ParallelStateWithAccum(TypedDict):
    topic: str
    perspectives: Annotated[list, lambda a, b: a + b]   # custom accumulator
    synthesis: str


s3 = StateGraph(ParallelStateWithAccum)
s3.add_node("generate_perspective", generate_perspective_node)
s3.add_node("synthesise", synthesise)
s3.add_conditional_edges(START, fan_out, ["generate_perspective"])
s3.add_edge("generate_perspective", "synthesise")
s3.add_edge("synthesise", END)

parallel_app = s3.compile()

parallel_result = parallel_app.invoke({
    "topic": "LangGraph for building AI agents",
    "perspectives": [],
    "synthesis": "",
})

print(f"Synthesis of {len(parallel_result['perspectives'])} parallel perspectives:")
print(parallel_result["synthesis"])
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Supervisor pattern: one LLM orchestrates workers exposed as tools
# ✅ Handoff pattern: Command(goto=node) gives agents direct routing control
# ✅ Parallel pattern: Send() fans out tasks; a custom reducer accumulates results
# ✅ Workers as tools = the supervisor sees their descriptions → better routing
# ✅ Always add MAX_ITERATIONS guard to prevent runaway supervisor loops
# ✅ Choose by task structure:
#      → Sequential dependencies  → handoff chain
#      → Dynamic task selection   → supervisor
#      → Independent subtasks     → parallel Send()
