"""
Stage 1, File 3: Chunked Summary for Long Transcripts
=======================================================
CONCEPT: Handling transcripts that exceed context window via chunking.

Long meetings (1+ hours) produce transcripts that exceed LLM context limits.
The map-reduce pattern solves this:
  1. MAP: Summarize each chunk independently
  2. REDUCE: Combine chunk summaries into final summary

Key concepts:
  - Text splitting strategies
  - Map-reduce pattern for long documents
  - Token counting and budget management

Run this file:
  uv run deep_research/summary_agents/stage_01_one_shot/03_chunked_summary.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=4096,
)

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 1.3: CHUNKED SUMMARY (MAP-REDUCE)")
print("=" * 70)
print(f"\nTranscript length: {len(transcript_text)} characters")
print(f"Word count estimate: ~{len(transcript_text.split())} words")


# ---------------------------------------------------------------------------
# Simple Chunking Strategy
# ---------------------------------------------------------------------------

def chunk_by_segments(transcript, max_chunks: int = 3) -> list[str]:
    """
    Split transcript into roughly equal chunks by segments.
    In production, you'd use token-based chunking.
    """
    segments = transcript.segments
    chunk_size = max(1, len(segments) // max_chunks)
    
    chunks = []
    for i in range(0, len(segments), chunk_size):
        chunk_segments = segments[i:i + chunk_size]
        # Reconstruct text for this chunk
        chunk_text = "\n".join(
            f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text
            for seg in chunk_segments
        )
        chunks.append(chunk_text)
    
    return chunks


# For demo, artificially create "chunks" by splitting the meeting into parts
# In a real scenario, this would be genuinely long content
chunks = chunk_by_segments(transcript, max_chunks=3)

print(f"\nSplit into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks, 1):
    print(f"  Chunk {i}: {len(chunk)} chars, ~{len(chunk.split())} words")


# ---------------------------------------------------------------------------
# MAP: Summarize Each Chunk
# ---------------------------------------------------------------------------

map_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize this portion of a meeting transcript. "
        "Focus on key points, decisions, and action items. "
        "Be concise but complete."
    ),
    (
        "human",
        "Meeting segment:\n{chunk}\n\nSummary:"
    ),
])

map_chain = map_prompt | llm | StrOutputParser()

print("\n" + "-" * 70)
print("  STEP 1: MAP (Summarize each chunk)")
print("-" * 70)

chunk_summaries = []
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Processing Chunk {i} ---")
    summary = map_chain.invoke({"chunk": chunk[:500] + "..." if len(chunk) > 500 else chunk})
    chunk_summaries.append(summary)
    print(f"Summary: {summary[:200]}...")


# ---------------------------------------------------------------------------
# REDUCE: Combine Summaries
# ---------------------------------------------------------------------------

reduce_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are combining summaries from different segments of the same meeting. "
        "Create a cohesive final summary with these sections:\n"
        "- Overview\n"
        "- Key Discussion Points\n"
        "- Decisions Made\n"
        "- Action Items\n\n"
        "Remove duplicates and organize logically."
    ),
    (
        "human",
        "Individual segment summaries:\n{summaries}\n\n"
        "Combined final summary:"
    ),
])

reduce_chain = reduce_prompt | llm | StrOutputParser()

print("\n" + "-" * 70)
print("  STEP 2: REDUCE (Combine summaries)")
print("-" * 70)

# Combine chunk summaries
combined_summaries = "\n\n---\n\n".join(
    f"Segment {i}:\n{summary}"
    for i, summary in enumerate(chunk_summaries, 1)
)

final_summary = reduce_chain.invoke({"summaries": combined_summaries})
print(f"\n{final_summary}")


# ---------------------------------------------------------------------------
# Optimized: Stepped Reduce (for many chunks)
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  OPTIMIZATION: STEPPED REDUCE")
print("-" * 70)
print("""
For many chunks, we can reduce in multiple stages to stay within context limits:
  Stage 1: chunks 1-4 → partial_summary_1
  Stage 2: chunks 5-8 → partial_summary_2
  Stage 3: partial_summary_1 + partial_summary_2 → final

This hierarchical approach keeps each LLM call manageable.
""")


# ---------------------------------------------------------------------------
# Alternative: Refine Pattern
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  ALTERNATIVE: REFINE PATTERN")
print("-" * 70)

refine_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Refine the existing summary with new information from the next segment."
    ),
    (
        "human",
        "Current summary:\n{current_summary}\n\n"
        "New segment to incorporate:\n{chunk}\n\n"
        "Refined summary:"
    ),
])

refine_chain = refine_prompt | llm | StrOutputParser()

# Start with first chunk
running_summary = chunk_summaries[0]
print(f"Starting with chunk 1 summary...")

# Iteratively refine with each subsequent chunk
for i, chunk_summary in enumerate(chunk_summaries[1:], 2):
    # In real usage, we'd pass the actual chunk content
    # For demo, we use the pre-generated summaries
    running_summary = refine_chain.invoke({
        "current_summary": running_summary,
        "chunk": f"[Summary of chunk {i}]: {chunk_summary[:300]}..."
    })
    print(f"Refined with chunk {i}...")

print(f"\nFinal refined summary:\n{running_summary[:500]}...")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Map-Reduce pattern handles arbitrarily long transcripts
✅ Chunking by speaker turns preserves conversational flow
✅ Stepped reduce prevents context overflow in combine step
✅ Refine pattern can capture dependencies across chunks

⚠️ Limitations:
   - Information can be lost at chunk boundaries
   - Cross-chunk context (e.g., references to earlier discussion) may be missed
   - Multiple LLM calls = higher cost and latency
   - Harder to debug when final summary is off

Trade-offs:
   | Pattern    | Parallel | Context Preservation | Cost |
   |------------|----------|---------------------|------|
   | Map-Reduce | Yes      | Medium              | High |
   | Refine     | No       | High                | High |
   | Single     | N/A      | Full                | Low  |

Next: Stage 2 → Agent-based summarization with tools
""")
