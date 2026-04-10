"""
Stage 1, File 3: LangChain Expression Language (LCEL) Chains
==============================================================
CONCEPT: Composing processing steps into pipelines.

LCEL is LangChain's declarative "pipe" syntax (inspired by Unix pipes and
functional programming). Every component that implements the Runnable
interface can be chained with |. This includes:

  prompts | llm | output_parsers | other_runnables

Key classes introduced:
  - StrOutputParser       : extracts .content from AIMessage → plain string
  - JsonOutputParser      : parses JSON from model output into a dict
  - RunnableLambda        : wraps any Python function as a Runnable
  - RunnableParallel      : runs multiple chains concurrently, merges results
  - RunnablePassthrough   : passes input through unchanged (useful in RAG)

Run this file:
  python 03_chains.py
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# ---------------------------------------------------------------------------
# 1. The canonical chain: prompt | llm | parser
# ---------------------------------------------------------------------------
print("=== 1. prompt | llm | StrOutputParser ===")

chain = (
    ChatPromptTemplate.from_template("What is {concept}? One paragraph.")
    | llm
    | StrOutputParser()   # strips the AIMessage wrapper → plain str
)

result = chain.invoke({"concept": "retrieval-augmented generation"})
print(type(result))   # <class 'str'>  (not AIMessage!)
print(result[:200])
print()

# ---------------------------------------------------------------------------
# 2. Structured output with JsonOutputParser + Pydantic
# ---------------------------------------------------------------------------
print("=== 2. Structured / JSON output ===")

class ResearchPlan(BaseModel):
    """Schema for a research plan returned by the LLM."""
    topic: str = Field(description="The research topic")
    sub_questions: list[str] = Field(description="3-5 sub-questions to investigate")
    estimated_sources: int = Field(description="Estimated number of sources needed")

parser = JsonOutputParser(pydantic_object=ResearchPlan)

structured_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a research planner. Respond ONLY with valid JSON matching this schema:\n{format_instructions}"),
        ("human", "Create a research plan for: {topic}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

plan = structured_chain.invoke({"topic": "impact of social media on teen mental health"})
print(f"Topic:          {plan['topic']}")
print(f"Sub-questions:  {plan['sub_questions']}")
print(f"Est. sources:   {plan['estimated_sources']}")
print()

# ---------------------------------------------------------------------------
# 3. RunnableLambda: inserting plain Python functions into a chain
# ---------------------------------------------------------------------------
print("=== 3. RunnableLambda (custom Python step) ===")

def add_word_count(text: str) -> dict:
    """Annotate the text with its word count."""
    return {"text": text, "word_count": len(text.split())}

augmented_chain = (
    ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
    | llm
    | StrOutputParser()
    | RunnableLambda(add_word_count)   # wrap our function
)

output = augmented_chain.invoke({"topic": "vector embeddings"})
print(f"Word count: {output['word_count']}")
print(f"Preview:    {output['text'][:150]}...")
print()

# ---------------------------------------------------------------------------
# 4. RunnableParallel: fan-out → run multiple chains on the same input
# ---------------------------------------------------------------------------
print("=== 4. RunnableParallel (concurrent chains) ===")

summary_chain = (
    ChatPromptTemplate.from_template("Summarize in 1 sentence: {topic}")
    | llm | StrOutputParser()
)
pros_chain = (
    ChatPromptTemplate.from_template("List 3 pros of {topic}")
    | llm | StrOutputParser()
)
cons_chain = (
    ChatPromptTemplate.from_template("List 3 cons of {topic}")
    | llm | StrOutputParser()
)

# All three run concurrently, results merged into a dict
parallel = RunnableParallel(
    summary=summary_chain,
    pros=pros_chain,
    cons=cons_chain,
)

results = parallel.invoke({"topic": "AI-generated research papers"})
print("Summary:", results["summary"])
print("\nPros:", results["pros"])
print("\nCons:", results["cons"])
print()

# ---------------------------------------------------------------------------
# 5. RunnablePassthrough: preserve original input alongside chain output
# ---------------------------------------------------------------------------
print("=== 5. RunnablePassthrough (pass-through pattern) ===")

# This pattern is the backbone of RAG: keep the original question while
# adding retrieved context alongside it.
passthrough_demo = RunnableParallel(
    question=RunnablePassthrough(),    # original input passes straight through
    answer=ChatPromptTemplate.from_template("Answer briefly: {question}")
            | llm
            | StrOutputParser(),
)

out = passthrough_demo.invoke("What year was Python created?")
print(f"Original question: {out['question']}")
print(f"Answer:            {out['answer']}")
print()

# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Every | step must be a Runnable (prompt, llm, parser, lambda, etc.)
# ✅ StrOutputParser → plain string; JsonOutputParser → dict / Pydantic model
# ✅ RunnableLambda wraps any function into the chain
# ✅ RunnableParallel runs branches concurrently → merged dict
# ✅ RunnablePassthrough pipes input unchanged — crucial for RAG
