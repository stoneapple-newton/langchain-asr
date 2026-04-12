"""
Stage 7, File 1: LangSmith Observability
==========================================
CONCEPT: Tracing, debugging, and evaluating LLM applications in production.

When an agent gives a wrong answer, how do you debug it?
  • Which prompt was used?
  • Which retrieval results were passed as context?
  • Which tool was called, with what arguments, and what did it return?
  • How many tokens were consumed and how long did each step take?

LangSmith answers all of these questions by automatically recording every
run in your LangChain/LangGraph application.

What you get for free with two env vars:
  - Full traces: every node, LLM call, tool, retriever step
  - Latency breakdown per step
  - Token usage and cost tracking
  - The exact prompt sent to the LLM (rendered, not templated)
  - Side-by-side comparison of different prompt versions

Additional features covered here:
  - @traceable           : trace any Python function (not just LangChain code)
  - run_on_dataset()     : automated eval against a golden dataset
  - LangSmithClient      : log feedback (thumbs up/down) from users

Setup (already in your .env):
  TRACING__ENABLED=true
  TRACING__API_KEY=<your key>
  TRACING__PROJECT=test-langchain   ← optional project name

Legacy aliases such as LANGSMITH_API_KEY and LANGCHAIN_API_KEY are still accepted.

Run this file:
  uv run deep_research/deep_research_agent/stage_07_production/01_langsmith_observability.py
"""

import os


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, get_settings

settings = get_settings()
for key, value in settings.tracing.to_langsmith_env(
    default_enabled=True,
    default_project="test-langchain-stage7",
).items():
    os.environ.setdefault(key, value)

llm = create_chat_model(
    temperature=0,
    max_tokens=256,
)


# ---------------------------------------------------------------------------
# 1. Automatic tracing — zero code changes needed
# ---------------------------------------------------------------------------
# With LANGSMITH_TRACING=true, every .invoke() is traced automatically.
# View traces at: https://smith.langchain.com

print("=== 1. Automatic tracing (check LangSmith after running) ===")

chain = (
    ChatPromptTemplate.from_template("Explain {concept} in one sentence.")
    | llm
    | StrOutputParser()
)

# This call is automatically recorded in LangSmith — no extra code
result = chain.invoke({"concept": "retrieval-augmented generation"})
print(result)
print()


# ---------------------------------------------------------------------------
# 2. @traceable — trace non-LangChain code
# ---------------------------------------------------------------------------
# Any Python function can be traced with @traceable.
# Useful for custom preprocessing, postprocessing, or business logic.

from langsmith import traceable

@traceable(name="research_pipeline", tags=["demo", "stage7"])
def research_and_answer(topic: str, depth: str = "brief") -> dict:
    """A multi-step pipeline — each inner step appears as a child span."""

    # Step 1: Generate questions
    questions = generate_questions(topic)

    # Step 2: Answer each question
    answers = [answer_question(q) for q in questions[:2]]

    return {
        "topic": topic,
        "questions": questions,
        "answers": answers,
    }


@traceable(name="generate_questions")
def generate_questions(topic: str) -> list[str]:
    prompt = ChatPromptTemplate.from_template(
        "List 3 key questions to research about: {topic}. "
        "Return only the questions, one per line."
    )
    raw = (prompt | llm | StrOutputParser()).invoke({"topic": topic})
    return [q.strip() for q in raw.strip().split("\n") if q.strip()][:3]


@traceable(name="answer_question")
def answer_question(question: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "Answer briefly (1 sentence): {question}"
    )
    return (prompt | llm | StrOutputParser()).invoke({"question": question})


print("=== 2. @traceable — tracing a multi-step pipeline ===")
output = research_and_answer("LangGraph checkpointers")
print(f"Topic: {output['topic']}")
print("Questions:")
for q in output["questions"]:
    print(f"  • {q}")
print("Answers:")
for a in output["answers"]:
    print(f"  → {a}")
print()


# ---------------------------------------------------------------------------
# 3. Attaching metadata to runs
# ---------------------------------------------------------------------------
# Metadata and tags let you filter and compare runs in the LangSmith UI.
# Example: compare runs from different users or prompt versions.

from langchain_core.runnables import RunnableConfig

print("=== 3. Run metadata and tags ===")

config = RunnableConfig(
    tags=["v2-prompt", "production"],
    metadata={
        "user_id": "user_123",
        "prompt_version": "2.1",
        "environment": "staging",
    },
)

result = chain.invoke({"concept": "vector embeddings"}, config=config)
print(f"Result: {result}")
print("→ Check LangSmith: this run is tagged 'v2-prompt' and 'production'")
print()


# ---------------------------------------------------------------------------
# 4. Logging user feedback
# ---------------------------------------------------------------------------
# In a real app, users click thumbs-up/down. You send that signal to
# LangSmith so you can track which runs users rated poorly.

from langsmith import Client

if settings.tracing.api_key:
    print("=== 4. Logging user feedback ===")

    client = Client()

    # Run a chain and capture the run ID
    import uuid
    run_id = str(uuid.uuid4())

    with_run_id = chain.with_config({"run_id": run_id})
    answer = with_run_id.invoke({"concept": "LangGraph conditional edges"})

    # Simulate a user rating this answer positively
    try:
        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=1,        # 1 = thumbs up, 0 = thumbs down
            comment="Clear and concise explanation",
        )
        print(f"Feedback logged for run {run_id[:8]}...")
    except Exception as e:
        print(f"(Feedback logging skipped: {e})")
    print()
else:
    print("=== 4. Logging user feedback ===")
    print("  Set TRACING__API_KEY or LANGSMITH_API_KEY in .env to enable LangSmith features")
    print()


# ---------------------------------------------------------------------------
# 5. Dataset-based evaluation
# ---------------------------------------------------------------------------
# The professional way to test: build a dataset of (input, expected_output)
# pairs, run your chain against it, score each output.
#
# This is too long to run inline, but here's the pattern:

print("=== 5. Evaluation pattern (code walkthrough — not executed) ===")

EVAL_PATTERN = '''
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

client = Client()

# 1. Create a dataset in LangSmith
dataset = client.create_dataset("rag-qa-v1")
client.create_examples(
    inputs=[{"question": "What is RAG?"}, {"question": "What is LangGraph?"}],
    outputs=[{"answer": "RAG combines retrieval with generation..."},
             {"answer": "LangGraph is a stateful agent graph library..."}],
    dataset_id=dataset.id,
)

# 2. Define an evaluator (LLM-as-judge or exact match)
qa_evaluator = LangChainStringEvaluator("qa", config={"llm": llm})

# 3. Run your chain against the dataset
results = evaluate(
    lambda x: chain.invoke(x),    # your chain
    data="rag-qa-v1",             # dataset name
    evaluators=[qa_evaluator],
    experiment_prefix="v2-prompt",
)
# → Results appear in LangSmith UI with pass/fail rates
'''
print(EVAL_PATTERN)


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Two env vars (LANGSMITH_TRACING, LANGSMITH_API_KEY) = automatic tracing
# ✅ @traceable wraps any function — not just LangChain components
# ✅ Tags + metadata let you filter, compare, and A/B test runs in the UI
# ✅ create_feedback() captures user signals for quality monitoring
# ✅ evaluate() + datasets = regression tests for your prompts
# ✅ Always enable LangSmith in production — it's the debugger for LLM apps
