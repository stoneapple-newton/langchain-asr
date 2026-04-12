"""
Stage 7, File 2: Human-in-the-Loop (HITL)
==========================================
CONCEPT: Pausing an agent mid-run to get human approval before continuing.

Fully autonomous agents are powerful but dangerous for irreversible actions:
  • Sending an email
  • Deleting files
  • Posting to social media
  • Executing a database write

Human-in-the-loop (HITL) gives you a "pause button" at any point in the graph.
The agent pauses, a human reviews the proposed action, and the graph resumes
(or is cancelled) based on the human's decision.

LangGraph implements HITL with two primitives:
  - interrupt(value)   : pause the graph and surface a value to the caller
  - Command(resume=x)  : resume the graph, passing x back to the interrupted node
  - interrupt_before=  : interrupt before a specified node (no code change needed)

Three HITL patterns covered:
  1. interrupt_before   — pause before a dangerous node (zero code in the node)
  2. interrupt()        — pause inside a node and act on the human's response
  3. Approval workflow  — full approve / reject / edit cycle

Run this file:
  uv run deep_research/deep_research_agent/stage_07_production/02_human_in_the_loop.py
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command


llm = create_chat_model(
    temperature=0,
    max_tokens=256,
)


# ---------------------------------------------------------------------------
# PATTERN 1 — interrupt_before: pause before a node, no code change
# ---------------------------------------------------------------------------
# The simplest pattern. You don't touch the dangerous node at all —
# just tell compile() to pause before it executes.

print("=" * 60)
print("  PATTERN 1: interrupt_before (zero code in the node)")
print("=" * 60)


class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]
    action: str   # the action the agent wants to take


def plan_action(state: SimpleState) -> dict:
    """Agent decides what to do."""
    response = llm.invoke([
        SystemMessage(content="You are an assistant. Respond with a short action plan."),
        *state["messages"],
    ])
    # Extract a proposed action from the response (simplified)
    action = f"Send email: '{response.content[:60]}...'"
    print(f"  [plan]    Agent proposes: {action[:80]}")
    return {"messages": [response], "action": action}


def execute_action(state: SimpleState) -> dict:
    """The dangerous node — would actually send the email."""
    print(f"  [execute] Performing: {state['action']}")
    return {"messages": [AIMessage(content=f"Done: {state['action']}")]}


checkpointer_1 = MemorySaver()
b1 = StateGraph(SimpleState)
b1.add_node("plan", plan_action)
b1.add_node("execute", execute_action)
b1.add_edge(START, "plan")
b1.add_edge("plan", "execute")
b1.add_edge("execute", END)

# ← interrupt_before=["execute"] pauses the graph after "plan" and before "execute"
app1 = b1.compile(checkpointer=checkpointer_1, interrupt_before=["execute"])

config = {"configurable": {"thread_id": "hitl_1"}}

# First invoke — runs "plan", then pauses
state = app1.invoke(
    {"messages": [HumanMessage(content="Draft and send a summary email to the team")],
     "action": ""},
    config=config,
)
print(f"  Graph paused. Proposed action: {state['action'][:80]}")

# Human review happens here (in production: web UI, Slack approval bot, etc.)
human_decision = input("  Approve? (y/n) [default: y]: ").strip().lower() or "y"

if human_decision == "y":
    # Resume with no change — just call invoke() again with the same config
    final = app1.invoke(None, config=config)   # None = resume from checkpoint
    print(f"  Resumed. Result: {final['messages'][-1].content[:80]}")
else:
    print("  Action cancelled by human.")
print()


# ---------------------------------------------------------------------------
# PATTERN 2 — interrupt() inside a node: surface data and act on the response
# ---------------------------------------------------------------------------
# interrupt(value) pauses AND returns value to the caller.
# When resumed with Command(resume=x), x is the return value of interrupt().

print("=" * 60)
print("  PATTERN 2: interrupt() — surface proposed action to the human")
print("=" * 60)


class WriteState(TypedDict):
    topic: str
    draft: str
    final: str


def write_draft(state: WriteState) -> dict:
    """Write a draft, then ask the human to approve/edit it."""
    prompt_text = f"Write a 2-sentence summary about: {state['topic']}"
    draft = llm.invoke(prompt_text).content
    print(f"  [write]  Draft: {draft[:100]}...")

    # interrupt() pauses here and surfaces the draft to the caller
    # When resumed, human_feedback is whatever the caller passed to Command(resume=)
    human_feedback = interrupt({
        "draft": draft,
        "instruction": "Edit the draft or press Enter to accept it as-is",
    })

    # Act on human feedback
    if human_feedback and human_feedback.strip():
        # Human provided edits — revise
        revised = llm.invoke(
            f"Revise this text based on feedback.\n"
            f"Original: {draft}\nFeedback: {human_feedback}"
        ).content
        print(f"  [write]  Revised: {revised[:100]}...")
        return {"draft": draft, "final": revised}
    else:
        # Human accepted the draft
        return {"draft": draft, "final": draft}


checkpointer_2 = MemorySaver()
b2 = StateGraph(WriteState)
b2.add_node("write", write_draft)
b2.add_edge(START, "write")
b2.add_edge("write", END)
app2 = b2.compile(checkpointer=checkpointer_2)

config2 = {"configurable": {"thread_id": "hitl_2"}}

# First invoke — runs until interrupt()
result = app2.invoke({"topic": "LangGraph human-in-the-loop", "draft": "", "final": ""}, config=config2)

# The graph is paused; result is the interrupt value
if "__interrupt__" in str(result):
    # In real usage: result contains the interrupt payload
    print(f"  Graph paused at interrupt()")

# Simulate human decision
user_edit = input("  Enter edit (or press Enter to accept): ").strip()

# Resume with the human's input
final_result = app2.invoke(Command(resume=user_edit), config=config2)
print(f"  Final text: {final_result.get('final', '')[:150]}")
print()


# ---------------------------------------------------------------------------
# PATTERN 3 — Full approval workflow: approve / reject / edit
# ---------------------------------------------------------------------------
# A more realistic pattern: the agent proposes an action, the human can
# approve, reject with a reason, or provide corrections.

print("=" * 60)
print("  PATTERN 3: Approve / Reject / Edit workflow")
print("=" * 60)


class ApprovalState(TypedDict):
    request: str          # what the user asked for
    plan: str             # what the agent plans to do
    decision: str         # "approved" | "rejected" | "edited"
    edit: str             # human's edited version (if any)
    result: str           # final output


def agent_plan(state: ApprovalState) -> dict:
    """Agent generates an action plan."""
    plan = llm.invoke(
        f"Write a step-by-step plan (3 steps) for: {state['request']}"
    ).content
    print(f"  [agent]  Plan generated ({len(plan)} chars)")
    return {"plan": plan}


def human_review(state: ApprovalState) -> dict:
    """Pause and ask the human to approve, reject, or edit."""
    print(f"\n  Proposed plan:\n  {state['plan'][:300]}\n")

    # Surface the plan via interrupt()
    response = interrupt({
        "plan": state["plan"],
        "options": "Type 'approve', 'reject', or paste your edited version",
    })

    # Parse the response
    r = (response or "approve").strip().lower()
    if r == "approve":
        return {"decision": "approved", "edit": ""}
    elif r == "reject":
        return {"decision": "rejected", "edit": ""}
    else:
        return {"decision": "edited", "edit": response}


def execute_plan(state: ApprovalState) -> dict:
    """Execute the approved/edited plan."""
    plan_to_use = state["edit"] if state["decision"] == "edited" else state["plan"]
    result = f"[Executed] Plan: {plan_to_use[:100]}..."
    print(f"  [execute] {result}")
    return {"result": result}


def reject_handler(state: ApprovalState) -> dict:
    """Handle a rejected plan."""
    print(f"  [reject]  Plan was rejected. Notifying requester.")
    return {"result": "Action was rejected by the reviewer."}


def route_after_review(state: ApprovalState) -> str:
    d = state.get("decision", "approved")
    if d == "rejected":
        return "reject"
    return "execute"   # approved or edited both go to execute


checkpointer_3 = MemorySaver()
b3 = StateGraph(ApprovalState)
b3.add_node("plan",   agent_plan)
b3.add_node("review", human_review)
b3.add_node("execute", execute_plan)
b3.add_node("reject", reject_handler)
b3.add_edge(START, "plan")
b3.add_edge("plan", "review")
b3.add_edge("execute", END)
b3.add_edge("reject", END)
b3.add_conditional_edges("review", route_after_review, {"execute": "execute", "reject": "reject"})

app3 = b3.compile(checkpointer=checkpointer_3)

config3 = {"configurable": {"thread_id": "hitl_3"}}

# Run until interrupt
app3.invoke({
    "request": "Migrate the database to the new schema",
    "plan": "", "decision": "", "edit": "", "result": "",
}, config=config3)

# Human review
human_input = input("  Your decision (approve/reject/edit): ").strip() or "approve"
final = app3.invoke(Command(resume=human_input), config=config3)
print(f"  Final result: {final.get('result', '')}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ interrupt_before=["node"] pauses before a node — no code changes inside it
# ✅ interrupt(value) pauses inside a node and surfaces data to the caller
# ✅ Command(resume=x) resumes the graph; x becomes the return value of interrupt()
# ✅ invoke(None, config) is shorthand for resuming with no input
# ✅ HITL is essential for any agent that takes irreversible real-world actions
# ✅ The same graph handles both automatic and human-supervised runs — same code
