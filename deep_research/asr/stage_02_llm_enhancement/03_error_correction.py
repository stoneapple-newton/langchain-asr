"""
ASR Stage 2, File 3: Transcription Error Correction
=====================================================
CONCEPT: Using LLMs to fix words the ASR model got wrong.

Low-confidence words (score < 0.75) are where the ASR model was uncertain.
They are the most likely transcription errors. Common failure modes:
  • Homophones  — "their/there/they're", "to/too/two", "auth/off"
  • Technical terms — "JWT", "devops", "scaffolding", "breakpoints"
  • Proper nouns — product names, people's names, company names
  • Domain jargon — field-specific vocabulary the model hasn't seen much

Correction strategy:
  1. Flag words below a confidence threshold
  2. Provide the surrounding context to the LLM
  3. Ask the LLM to suggest the most likely correct word
  4. Apply corrections and track changes for review

We also implement a domain glossary approach: inject known correct terms
so the LLM doesn't have to guess technical vocabulary.

Run this file:
  uv run deep_research/asr/stage_02_llm_enhancement/03_error_correction.py
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
    num_predict=256,
)

LOW_CONF_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# 1. Locate low-confidence words with context
# ---------------------------------------------------------------------------

@dataclass
class ErrorCandidate:
    segment_idx: int
    word_idx: int
    word: str
    score: float
    speaker: str
    left_context: str    # 3 words before
    right_context: str   # 3 words after
    segment_text: str


def find_error_candidates(
    segments: list[dict],
    threshold: float = LOW_CONF_THRESHOLD,
) -> list[ErrorCandidate]:
    candidates = []
    for seg_idx, seg in enumerate(segments):
        words = seg.get("words", [])
        for w_idx, word in enumerate(words):
            if word.get("score", 1.0) < threshold:
                left = " ".join(w["word"] for w in words[max(0, w_idx-3):w_idx])
                right = " ".join(w["word"] for w in words[w_idx+1:w_idx+4])
                candidates.append(ErrorCandidate(
                    segment_idx=seg_idx,
                    word_idx=w_idx,
                    word=word["word"],
                    score=word["score"],
                    speaker=seg.get("speaker", "UNKNOWN"),
                    left_context=left,
                    right_context=right,
                    segment_text=seg["text"].strip(),
                ))
    return candidates


candidates = find_error_candidates(segments)
print(f"=== 1. Low-confidence words found: {len(candidates)} ===")
for c in sorted(candidates, key=lambda x: x.score)[:10]:
    print(f"  [{c.score:.2f}] '{c.left_context} >>>{c.word}<<< {c.right_context}' "
          f"  (seg {c.segment_idx}, {c.speaker})")
print()


# ---------------------------------------------------------------------------
# 2. Domain glossary — inject known correct terms
# ---------------------------------------------------------------------------
# If you know the domain vocabulary, list it here. The LLM will prefer
# these terms when they fit the context, dramatically reducing hallucinations.

DOMAIN_GLOSSARY = [
    "JWT", "JSON Web Token", "authentication", "diarization",
    "Q4", "roadmap", "dashboard", "analytics", "devops", "DevOps",
    "Node.js", "Node 18", "breakpoints", "scaffolding", "staging",
    "color palette", "accessibility", "WCAG", "hex code",
    "implementation", "invalidated", "conversion funnel",
]


# ---------------------------------------------------------------------------
# 3. Word correction chain
# ---------------------------------------------------------------------------

class WordCorrection(BaseModel):
    original: str = Field(description="The original (possibly wrong) word")
    corrected: str = Field(description="The corrected word, or the same word if it seems correct")
    confidence: float = Field(description="Confidence that the correction is right (0–1)")
    reason: str = Field(description="Why you chose this correction")


class BatchCorrections(BaseModel):
    corrections: list[WordCorrection]


correction_parser = JsonOutputParser(pydantic_object=BatchCorrections)

correction_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an ASR correction specialist. For each low-confidence word, "
     "determine the most likely correct word based on context.\n\n"
     "Domain vocabulary to prefer: {glossary}\n\n"
     "Rules:\n"
     "  • If the word looks correct in context, return it unchanged\n"
     "  • For technical terms, check against the domain vocabulary\n"
     "  • Only suggest a correction if you are confident\n"
     "  • Never change named entities unless obviously wrong\n\n"
     "Respond with JSON: {format_instructions}"),
    ("human",
     "Correct these low-confidence words:\n{candidates}"),
]).partial(
    format_instructions=correction_parser.get_format_instructions(),
    glossary=", ".join(DOMAIN_GLOSSARY),
)

correction_chain = correction_prompt | llm | correction_parser


def format_candidates_for_prompt(candidates: list[ErrorCandidate]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"{i+1}. Word: '{c.word}' (score={c.score:.2f})\n"
            f"   Context: '...{c.left_context} [WORD] {c.right_context}...'\n"
            f"   Full segment: '{c.segment_text[:100]}'"
        )
    return "\n\n".join(lines)


print("=== 2. LLM correction chain ===")

# Process in batches of 5 to avoid overloading the context window
BATCH_SIZE = 5
all_corrections: list[WordCorrection] = []

for batch_start in range(0, len(candidates), BATCH_SIZE):
    batch = candidates[batch_start:batch_start + BATCH_SIZE]
    prompt_text = format_candidates_for_prompt(batch)
    result = correction_chain.invoke({"candidates": prompt_text})
    corrections = result.get("corrections", [])
    all_corrections.extend(corrections)
    print(f"  Batch {batch_start//BATCH_SIZE + 1}: corrected {len(corrections)} words")

print()
print("  Corrections made:")
for c in all_corrections:
    orig = c.get("original", "") if isinstance(c, dict) else c.original
    corr = c.get("corrected", "") if isinstance(c, dict) else c.corrected
    conf = c.get("confidence", 0) if isinstance(c, dict) else c.confidence
    reason = c.get("reason", "") if isinstance(c, dict) else c.reason
    if orig.lower() != corr.lower():
        print(f"    '{orig}' → '{corr}'  (conf={conf:.0%})  {reason[:60]}")
print()


# ---------------------------------------------------------------------------
# 4. Apply corrections to transcript
# ---------------------------------------------------------------------------

def apply_corrections(
    segments: list[dict],
    candidates: list[ErrorCandidate],
    corrections: list,
    min_confidence: float = 0.7,
) -> list[dict]:
    """
    Apply accepted corrections back into the transcript segments.
    Only apply if correction confidence >= min_confidence.
    """
    import copy
    corrected_segs = copy.deepcopy(segments)

    applied = 0
    for candidate, correction in zip(candidates, corrections):
        orig = correction.get("original", "") if isinstance(correction, dict) else correction.original
        corr = correction.get("corrected", "") if isinstance(correction, dict) else correction.corrected
        conf = correction.get("confidence", 0) if isinstance(correction, dict) else correction.confidence

        if orig.lower() == corr.lower():
            continue  # no change needed
        if conf < min_confidence:
            continue  # not confident enough

        # Apply the correction at the exact word position
        seg = corrected_segs[candidate.segment_idx]
        if candidate.word_idx < len(seg.get("words", [])):
            seg["words"][candidate.word_idx]["word"] = corr
            # Also update the segment text
            words = [w["word"] for w in seg["words"]]
            seg["text"] = " " + " ".join(words)
            applied += 1

    print(f"  Applied {applied} corrections (out of {len(corrections)} candidates)")
    return corrected_segs


corrected_segments = apply_corrections(segments, candidates, all_corrections)
print("=== 3. Before/After comparison ===")
for i, (orig, corr) in enumerate(zip(segments, corrected_segments)):
    if orig["text"] != corr["text"]:
        print(f"\n  Segment {i:02d} [{orig.get('speaker')}]")
        print(f"    Before: {orig['text'].strip()}")
        print(f"    After : {corr['text'].strip()}")
print()


# ---------------------------------------------------------------------------
# 5. Segment-level coherence check
# ---------------------------------------------------------------------------
# After word-level corrections, verify the segment reads naturally.
# A final LLM pass catches any remaining issues the word-level pass missed.

coherence_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Review this transcription segment for any remaining errors. "
     "Return ONLY the corrected text — no explanation, no formatting.\n"
     "If the text is correct as-is, return it unchanged.\n"
     "Domain terms: {glossary}"),
    ("human",
     "Speaker: {speaker}\n"
     "Context: {context}\n"
     "Text: {text}"),
]).partial(glossary=", ".join(DOMAIN_GLOSSARY[:10]))

from langchain_core.output_parsers import StrOutputParser
coherence_chain = coherence_prompt | llm | StrOutputParser()

print("=== 4. Coherence check on corrected segments ===")
for i, seg in enumerate(corrected_segments[:6]):
    context = segments[i-1]["text"].strip() if i > 0 else ""
    final_text = coherence_chain.invoke({
        "speaker": seg.get("speaker", "UNKNOWN"),
        "context": context,
        "text": seg["text"].strip(),
    })
    if final_text.strip() != seg["text"].strip():
        print(f"  Seg {i:02d}: '{seg['text'].strip()[:60]}' → '{final_text.strip()[:60]}'")
    else:
        print(f"  Seg {i:02d}: OK (no change)")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ score < 0.75 is your error-detection signal — always start here
# ✅ Always provide left + right context — the LLM cannot correct without it
# ✅ Domain glossaries dramatically reduce hallucination on technical terms
# ✅ Process in batches of 5 — balance context quality vs. token cost
# ✅ Set a minimum correction confidence threshold — don't apply uncertain fixes
# ✅ Coherence check is the final safety net — catches what word-level missed
