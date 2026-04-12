"""
Stage 3, File 3: ReAct Agent with LangGraph
============================================
CONCEPT: A self-directed tool-calling loop — the canonical LLM agent pattern.

ReAct (Reason + Act) is the most common agent architecture:
  1. Agent (LLM) looks at the conversation and decides what to do
  2. If the LLM calls a tool  → execute it, feed results back, loop
  3. If the LLM replies normally → we're done, return the answer

LangGraph makes this trivially easy with two prebuilt helpers:
  - ToolNode        : executes whatever tool_calls the LLM returned
  - tools_condition : router that returns "tools" or END based on tool_calls

Key concepts introduced:
  - ToolNode        : automatic tool execution node
  - tools_condition : prebuilt conditional router for tool-calling loops
  - The agent loop  : agent → tools → agent → tools → ... → END

Run this file:
  uv run deep_research/deep_research_agent/stage_03_langgraph/03_react_agent.py
"""

import os
import math
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition



# ---------------------------------------------------------------------------
# 1. Define tools (same pattern as Stage 2)
# ---------------------------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """Returns today's date and current time. Use for any time-sensitive questions."""
    return datetime.now().strftime("%A, %B %d, %Y at %H:%M")


@tool
def calculate(expression: str) -> str:
    """Evaluates a Python math expression safely.
    Examples: '2 ** 10', 'math.sqrt(144)', '100 / 3'.
    Only use for mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Converts between common units.
    Supported conversions:
      Length : km ↔ miles, m ↔ feet
      Weight : kg ↔ pounds
      Temp   : celsius ↔ fahrenheit
    """
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("m", "feet"):   lambda v: v * 3.28084,
        ("feet", "m"):   lambda v: v / 3.28084,
        ("kg", "pounds"): lambda v: v * 2.20462,
        ("pounds", "kg"): lambda v: v / 2.20462,
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"Unsupported conversion: {from_unit} → {to_unit}"
    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


tools = [get_current_datetime, calculate, unit_converter]


# ---------------------------------------------------------------------------
# 2. LLM with tools bound
# ---------------------------------------------------------------------------

llm = create_chat_model(
    temperature=0,
    max_tokens=512,
)

llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 3. State — just a message list
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# 4. The two nodes
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools. "
    "Use tools whenever you need current data or calculations. "
    "Always show your reasoning before calling a tool."
))


def agent_node(state: AgentState) -> dict:
    """The reasoning step: ask the LLM what to do next."""
    # Always prepend the system prompt on the first call
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SYSTEM_PROMPT] + messages

    response = llm_with_tools.invoke(messages)

    # Show what the agent decided
    if response.tool_calls:
        calls = [f"{tc['name']}({tc['args']})" for tc in response.tool_calls]
        print(f"  [agent]  → calling tools: {calls}")
    else:
        print(f"  [agent]  → final answer: {response.content[:80]}...")

    return {"messages": [response]}


# ToolNode automatically:
#   • reads tool_calls from the last AIMessage
#   • executes each tool function with the provided args
#   • wraps results in ToolMessage objects and returns them
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# 5. Build the ReAct graph
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

# Entry point
builder.add_edge(START, "agent")

# Conditional edge after agent:
#   tools_condition returns "tools" if last message has tool_calls, else END
builder.add_conditional_edges("agent", tools_condition)

# After tools run, always go back to agent (the loop)
builder.add_edge("tools", "agent")

agent_app = builder.compile()


# ---------------------------------------------------------------------------
# 6. Run the agent on several queries
# ---------------------------------------------------------------------------

def run_agent(question: str):
    print(f"\n{'='*60}")
    print(f"  Question: {question}")
    print(f"{'='*60}")

    result = agent_app.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # The last message is the final answer
    final = result["messages"][-1]
    print(f"\n  Answer: {final.content}")
    print(f"  Total messages in state: {len(result['messages'])}")


run_agent("What is today's date, and what is 2 to the power of 16?")
run_agent("I ran 10 km. How many miles is that? Also, what's the square root of 1764?")
run_agent("Convert 37 degrees Celsius to Fahrenheit.")


# ---------------------------------------------------------------------------
# BONUS: Streaming agent steps
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  BONUS: Streaming the agent's steps")
print("=" * 60)

for step in agent_app.stream(
    {"messages": [HumanMessage(content="What is 15 miles in km, and what is today's date?")]},
    stream_mode="updates",   # yields {node_name: state_update} per step
):
    node_name = list(step.keys())[0]
    update = step[node_name]
    msgs = update.get("messages", [])
    if msgs:
        last = msgs[-1]
        role = type(last).__name__
        content_preview = getattr(last, "content", "")[:60]
        print(f"  [{node_name}] {role}: {content_preview}...")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ ToolNode handles all tool execution — you never call tools manually
# ✅ tools_condition is the standard router: tool_calls? → "tools" : END
# ✅ The loop is just: agent → tools → agent (until no tool_calls)
# ✅ bind_tools() + ToolNode must use the SAME tools list
# ✅ stream_mode="updates" gives you one dict per node — great for visibility
