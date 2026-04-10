"""
Stage 2, File 2: Web Search Tool (Tavily)
==========================================
CONCEPT: Connecting agents to real-time web information.

Tavily is a search API purpose-built for LLM agents. Unlike Google/Bing,
it returns clean, pre-extracted content (not raw HTML), which means less
token waste and better answers.

You need a free Tavily API key: https://app.tavily.com
Add it to your .env file as: TAVILY_API_KEY=tvly-...

Key classes introduced:
  - TavilySearchResults   : LangChain tool that wraps Tavily's search API
  - DuckDuckGoSearchRun   : free alternative (no API key needed)
  - WikipediaQueryRun     : for encyclopedic background information

Run this file:
  python 02_web_search.py
"""

import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


# ---------------------------------------------------------------------------
# Helper: pretty-print search results
# ---------------------------------------------------------------------------
def print_results(label: str, results):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if isinstance(results, list):
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r.get('title', 'No title')}")
            print(f"    URL: {r.get('url', '')}")
            content = r.get('content', '')
            print(f"    {content[:200]}..." if len(content) > 200 else f"    {content}")
    else:
        print(results[:600] if len(str(results)) > 600 else results)


# ---------------------------------------------------------------------------
# 1. Tavily Search (recommended — best quality for agents)
# ---------------------------------------------------------------------------
# max_results: how many URLs to scrape and summarize
# include_raw_content: get full page text (more tokens but richer)
# include_images: whether to include image URLs

if os.getenv("TAVILY_API_KEY"):
    tavily = TavilySearchResults(
        max_results=3,
        include_answer=True,       # Tavily's own 1-sentence summary
        include_raw_content=False, # set True for deeper research
        include_images=False,
    )

    results = tavily.invoke("What is LangGraph and how does it differ from LangChain?")
    print_results("Tavily Search Results", results)

    # Tavily also supports search_depth="advanced" for deeper research
    tavily_deep = TavilySearchResults(
        max_results=5,
        search_depth="advanced",  # scrapes more content per page
    )
    print("\nTavily is configured and ready ✅")
else:
    print("⚠️  TAVILY_API_KEY not found. Skipping Tavily demo. Add it to .env to enable.")
    tavily = None


# ---------------------------------------------------------------------------
# 2. DuckDuckGo Search (free, no API key needed — good for dev/testing)
# ---------------------------------------------------------------------------
ddg = DuckDuckGoSearchRun()

# DuckDuckGo returns a single string with concatenated snippets
ddg_result = ddg.invoke("LangGraph state machine agents 2024")
print_results("DuckDuckGo Result", ddg_result)


# ---------------------------------------------------------------------------
# 3. Wikipedia (great for factual background, stable knowledge)
# ---------------------------------------------------------------------------
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2,       # number of Wikipedia articles to retrieve
    doc_content_chars_max=1500,  # max chars per article
)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

wiki_result = wiki_tool.invoke("Retrieval-Augmented Generation")
print_results("Wikipedia Result", wiki_result)


# ---------------------------------------------------------------------------
# 4. Using search tools in a simple LLM chain
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("\n=== Search-Augmented Chain ===")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Pick whichever search tool is available
search_tool = tavily if tavily else ddg

def search_and_answer(question: str) -> str:
    """Manually run search → inject results → ask LLM."""
    # Step 1: Search
    raw = search_tool.invoke(question)
    search_context = str(raw)[:2000]  # trim to save tokens

    # Step 2: Feed results to LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Answer based on the search results provided. "
                   "Cite specific information from the results."),
        ("human", "Question: {question}\n\nSearch Results:\n{context}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": search_context})

answer = search_and_answer("What are the latest features in LangGraph?")
print(answer)


# ---------------------------------------------------------------------------
# 5. Tool comparison summary
# ---------------------------------------------------------------------------
print("\n=== Tool Comparison ===")
comparison = {
    "TavilySearchResults": "Best quality, structured output, designed for LLM agents. Requires API key.",
    "DuckDuckGoSearchRun": "Free, no key needed, returns raw text. Good for development.",
    "WikipediaQueryRun": "Encyclopedic, stable facts, no key needed. Poor for current events.",
}
for tool_name, desc in comparison.items():
    print(f"  {tool_name:28s} → {desc}")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ TavilySearchResults is the recommended search tool for production agents
# ✅ DuckDuckGoSearchRun is free — great for prototyping without API keys
# ✅ WikipediaQueryRun is great for factual background research
# ✅ Tools return strings or dicts that get injected into the LLM's context
