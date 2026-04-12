"""
ASR Stage 2, File 1: Punctuation & Cleanup with LLMs
======================================================
CONCEPT: Using an LLM to restore punctuation and remove noise from raw ASR text.

WhisperX outputs lowercase text with no punctuation. This is technically
correct (that's what was spoken) but unreadable. An LLM can:
  1. Capitalise the first word of each sentence
  2. Add commas, periods, question marks from prosody context
  3. Remove or replace filler words (uh, um, like)
  4. Collapse duplicated/stuttered words
  5. Fix contractions (arent → aren't, ill → I'll)

Strategy: process segment-by-segment (cheap, fast) vs. full-context batches
(higher quality but more tokens). We implement both and compare.

Key LangChain patterns:
  - ChatPromptTemplate with few-shot examples for consistent formatting
  - RunnableLambda for pre/post-processing steps in the chain
  - .batch() to process multiple segments concurrently
  - Pydantic output schema to enforce structure

Run this file:
  uv run deep_research/asr/stage_02_llm_enhancement/01_punctuation_cleanup.py
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=512,
)


# ---------------------------------------------------------------------------
# 1. Rule-based pre-pass (fast, no LLM)
# ---------------------------------------------------------------------------
# Handle the easy cases before touching the LLM — saves tokens and latency.

FILLER_PATTERN = re.compile(
    r"\b(uh|um|uh um|um uh|you know|i mean|like,?)\b\s*",
    re.IGNORECASE,
)
DUPLICATE_PATTERN = re.compile(r"\b(\w+) \1\b", re.IGNORECASE)
CONTRACTION_MAP = {
    r"\barent\b": "aren't", r"\bcant\b": "can't", r"\bwont\b": "won't",
    r"\bdont\b": "don't",   r"\bdidnt\b": "didn't", r"\bisnt\b": "isn't",
    r"\bwasnt\b": "wasn't", r"\bwouldnt\b": "wouldn't", r"\bcouldnt\b": "couldn't",
    r"\bshouldnt\b": "shouldn't", r"\bim\b": "I'm", r"\bills\b": "I'll",
    r"\bill\b": "I'll",     r"\bive\b": "I've", r"\bthats\b": "that's",
    r"\bits\b": "it's",     r"\blets\b": "let's", r"\bwere\b": "we're",
    r"\btheyre\b": "they're", r"\bweve\b": "we've", r"\bitll\b": "it'll",
}


def rule_based_cleanup(text: str) -> str:
    """Fast rule-based cleanup that runs before the LLM."""
    t = text.strip()
    # Remove fillers
    t = FILLER_PATTERN.sub("", t)
    # Collapse duplicate words (stutters)
    t = DUPLICATE_PATTERN.sub(r"\1", t)
    # Fix contractions
    for pattern, replacement in CONTRACTION_MAP.items():
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    # Collapse multiple spaces
    t = re.sub(r" {2,}", " ", t).strip()
    return t


print("=== 1. Rule-based cleanup (no LLM) ===")
sample = segments[4]["text"]  # the segment with stutters/contractions
cleaned = rule_based_cleanup(sample)
print(f"Original : {sample.strip()}")
print(f"Cleaned  : {cleaned}")
print()


# ---------------------------------------------------------------------------
# 2. LLM punctuation chain (segment-by-segment)
# ---------------------------------------------------------------------------

class CleanedText(BaseModel):
    text: str = Field(description="The cleaned, punctuated text")
    changes: list[str] = Field(description="Brief list of changes made")


punct_parser = JsonOutputParser(pydantic_object=CleanedText)

punct_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an ASR post-processing expert. Clean transcription text by:\n"
     "  1. Adding correct punctuation (commas, periods, question marks)\n"
     "  2. Capitalising the first word and proper nouns\n"
     "  3. Keeping the meaning and all content words exactly as spoken\n"
     "  4. Do NOT add or remove content words — only fix punctuation and capitalisation\n\n"
     "Reply with JSON matching: {format_instructions}"),
    ("human",
     "Speaker: {speaker}\n"
     "Preceding context: {context}\n"
     "Text to clean: {text}"),
]).partial(format_instructions=punct_parser.get_format_instructions())

punct_chain = punct_prompt | llm | punct_parser


def build_context(segments: list[dict], idx: int, window: int = 1) -> str:
    """Get the preceding segment text as context for the LLM."""
    start = max(0, idx - window)
    ctx_parts = [segments[i]["text"].strip() for i in range(start, idx)]
    return " / ".join(ctx_parts) if ctx_parts else "(start of transcript)"


print("=== 2. LLM punctuation — segment by segment ===")
for i in range(0, min(5, len(segments))):
    seg = segments[i]
    context = build_context(segments, i)
    result = punct_chain.invoke({
        "speaker": seg.get("speaker", "UNKNOWN"),
        "context": context,
        "text": seg["text"].strip(),
    })
    print(f"\n  [{seg.get('speaker')}]")
    print(f"  Before : {seg['text'].strip()}")
    print(f"  After  : {result.get('text', '')}")
    if result.get("changes"):
        print(f"  Changes: {result['changes']}")
print()


# ---------------------------------------------------------------------------
# 3. Batch processing — all segments concurrently
# ---------------------------------------------------------------------------

def prepare_batch_input(segments: list[dict]) -> list[dict]:
    inputs = []
    for i, seg in enumerate(segments):
        inputs.append({
            "speaker": seg.get("speaker", "UNKNOWN"),
            "context": build_context(segments, i),
            "text": rule_based_cleanup(seg["text"]),  # pre-clean first
        })
    return inputs


print("=== 3. Batch processing (concurrent) ===")
batch_inputs = prepare_batch_input(segments[:8])
batch_results = punct_chain.batch(batch_inputs, config={"max_concurrency": 4})

print(f"Processed {len(batch_results)} segments")
for i, (seg, result) in enumerate(zip(segments[:8], batch_results)):
    original = seg["text"].strip()
    improved = result.get("text", original)
    if original.lower() != improved.lower():
        print(f"\n  Seg {i:02d} [{seg.get('speaker')}]")
        print(f"    Before : {original[:80]}")
        print(f"    After  : {improved[:80]}")
print()


# ---------------------------------------------------------------------------
# 4. Full-context window approach (higher quality)
# ---------------------------------------------------------------------------
# Instead of one segment at a time, send a block of segments together.
# The LLM sees sentence boundaries more naturally → better punctuation.

full_context_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an ASR transcript editor. You will receive a block of raw transcription "
     "lines, each prefixed with [SPEAKER_XX]. Your task:\n"
     "  1. Add correct punctuation (commas, periods, question marks, exclamation marks)\n"
     "  2. Capitalise sentence starts and proper nouns\n"
     "  3. Remove filler words (uh, um, like) that add no meaning\n"
     "  4. Fix contractions (arent → aren't, cant → can't, ill → I'll, etc.)\n"
     "  5. Preserve ALL speaker labels exactly as [SPEAKER_XX]\n"
     "  6. Return the SAME number of lines as input — one line per input line\n"
     "  7. Do NOT merge or split lines\n\n"
     "Return ONLY the corrected lines, nothing else."),
    ("human", "{block}"),
])

full_context_chain = full_context_prompt | llm | StrOutputParser()


def process_block(segments: list[dict], start: int, block_size: int = 6) -> list[str]:
    """Process a block of segments together for better context."""
    end = min(start + block_size, len(segments))
    block_segs = segments[start:end]
    block_text = "\n".join(
        f"[{s.get('speaker', 'UNKNOWN')}] {rule_based_cleanup(s['text'])}"
        for s in block_segs
    )
    result = full_context_chain.invoke({"block": block_text})
    lines = [ln.strip() for ln in result.strip().split("\n") if ln.strip()]
    # Extract just the text part (after [SPEAKER_XX] prefix)
    cleaned_lines = []
    for ln in lines:
        match = re.match(r"^\[SPEAKER_\d+\]\s*(.*)", ln)
        cleaned_lines.append(match.group(1) if match else ln)
    return cleaned_lines


print("=== 4. Full-context block processing ===")
improved_texts = process_block(segments, 0, block_size=6)
print("Block 0–5 results:")
for i, (seg, text) in enumerate(zip(segments[:6], improved_texts)):
    print(f"\n  Seg {i:02d} [{seg.get('speaker')}]")
    print(f"    Before : {seg['text'].strip()}")
    print(f"    After  : {text}")
print()


# ---------------------------------------------------------------------------
# 5. Apply to full transcript and save
# ---------------------------------------------------------------------------

print("=== 5. Apply to full transcript ===")
import copy

improved = copy.deepcopy(raw)
block_size = 6

all_improved = []
for start in range(0, len(segments), block_size):
    block_texts = process_block(segments, start, block_size)
    all_improved.extend(block_texts)
    print(f"  Processed segments {start}–{min(start+block_size-1, len(segments)-1)}")

# Write back into the improved copy
for seg, new_text in zip(improved["segments"], all_improved):
    seg["text"] = " " + new_text  # WhisperX convention: leading space

out_path = TRANSCRIPT_PATH.parent / "transcript_punctuated.json"
with open(out_path, "w") as f:
    json.dump(improved, f, indent=2)
print(f"\n  Saved: {out_path.name}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Always run a rule-based pre-pass first — it's free and handles easy cases
# ✅ Segment-by-segment is simple but misses sentence boundary context
# ✅ Block processing gives better punctuation by seeing surrounding turns
# ✅ Use .batch() with max_concurrency to process segments in parallel
# ✅ Enforce "same number of lines out as in" so you can zip results back to segments
# ✅ Save intermediate JSON at each stage — makes debugging much easier
