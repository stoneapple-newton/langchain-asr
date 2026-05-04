"""
ASR Stage 3, File 3: Building & Persisting Speaker Profiles
=============================================================
CONCEPT: Accumulating speaker knowledge across sessions for better diarization.

A single meeting gives limited evidence about a speaker. But across multiple
meetings, you can build a rich profile:
  • Vocabulary and speaking style
  • Topics they typically discuss
  • Average turn duration and pacing
  • Role and organisational context

These profiles are stored in a vector store so that future transcripts can
query "who sounds most like this speaker?" using semantic similarity.

This is the bridge between Stage 3 (diarization) and Stage 5 (RAG) —
profiles become the knowledge base for context-aware correction.

Key patterns:
  - Speaker profile as a Document with metadata
  - Shared embeddings config for profile vectors
  - InMemoryVectorStore (or Chroma) for retrieval
  - Profile merging across multiple transcripts

Run this file:
  uv run deep_research/asr/stage_03_diarization/03_speaker_profiles.py
"""

import os
import json
import re
from pathlib import Path
import sys
from dataclasses import dataclass, field
from collections import Counter

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, create_embeddings, structured_output_chain
from pydantic import BaseModel, Field


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]

name_map_path = TRANSCRIPT_PATH.parent / "speaker_names.json"
name_map: dict[str, str] = {}
if name_map_path.exists():
    with open(name_map_path) as f:
        name_map = json.load(f)

llm = create_chat_model(
    "asr",
    temperature=0,
    max_tokens=4096,
)
embeddings = create_embeddings("asr")

STOPWORDS = {"the", "a", "an", "is", "are", "was", "we", "i", "to", "of",
             "and", "it", "in", "that", "for", "this", "you", "so", "be",
             "on", "have", "do", "with", "by", "at", "yeah", "uh", "um"}


# ---------------------------------------------------------------------------
# 1. Extract per-speaker content from this transcript
# ---------------------------------------------------------------------------

def extract_speaker_content(segments: list[dict]) -> dict[str, str]:
    """Collect all text per speaker."""
    content: dict[str, str] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        content[spk] = content.get(spk, "") + " " + seg["text"].strip()
    return {k: v.strip() for k, v in content.items()}


def top_words(text: str, n: int = 10) -> list[str]:
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(filtered).most_common(n)]


speaker_content = extract_speaker_content(segments)
print("=== 1. Per-speaker vocabulary snapshot ===")
for spk, text in speaker_content.items():
    name = name_map.get(spk, spk)
    vocab = top_words(text)
    print(f"  {name:<20} ({spk}): {vocab}")
print()


# ---------------------------------------------------------------------------
# 2. LLM-generated speaker profile narrative
# ---------------------------------------------------------------------------

class SpeakerProfileNarrative(BaseModel):
    speaker_id: str
    display_name: str
    role: str
    speaking_style: str = Field(description="How this person communicates (e.g. 'direct and technical', 'facilitative')")
    key_topics: list[str] = Field(description="Main topics this speaker discussed")
    notable_phrases: list[str] = Field(description="Phrases or patterns characteristic of this speaker")
    profile_text: str = Field(description="2-3 sentence narrative profile for embedding")


profile_parser = JsonOutputParser()

profile_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Analyse this speaker's transcript contribution and create a speaker profile.\n"
     "The profile_text field is especially important — it will be used as an embedding "
     "for future speaker identification, so make it descriptive and distinctive.\n\n"
     "Respond with JSON: {format_instructions}"),
    ("human",
     "Speaker ID    : {speaker_id}\n"
     "Known name    : {known_name}\n"
     "Meeting title : {meeting_title}\n\n"
     "All text by this speaker:\n{text}"),
]).partial(format_instructions=profile_parser.get_format_instructions())

profile_chain = structured_output_chain(llm, profile_prompt, SpeakerProfileNarrative)

print("=== 2. Generating speaker profiles ===")
profiles: list[dict] = []
meta = raw.get("meeting_metadata", {})

for spk, text in speaker_content.items():
    known_name = name_map.get(spk, "Unknown")
    print(f"  Profiling {spk} ({known_name})...")

    result = profile_chain.invoke({
        "speaker_id": spk,
        "known_name": known_name,
        "meeting_title": meta.get("title", "Meeting"),
        "text": text[:1200],  # cap to save tokens
    })

    profile = result if isinstance(result, dict) else result.dict()
    profiles.append(profile)

    print(f"    Role         : {profile.get('role', 'Unknown')}")
    print(f"    Style        : {profile.get('speaking_style', '')}")
    print(f"    Key topics   : {profile.get('key_topics', [])[:4]}")
    print(f"    Profile text : {profile.get('profile_text', '')[:100]}...")
    print()


# ---------------------------------------------------------------------------
# 3. Store profiles in a vector store for future retrieval
# ---------------------------------------------------------------------------
# Each speaker profile becomes a Document.
# When a future meeting has an unknown speaker, we embed their text and
# do a similarity search to find the most similar known profile.

def profiles_to_documents(profiles: list[dict], meeting_id: str) -> list[Document]:
    """Convert speaker profiles into LangChain Documents for embedding."""
    docs = []
    for p in profiles:
        # The embedding text combines profile narrative + key topics + style
        embed_text = (
            f"{p.get('profile_text', '')} "
            f"Topics: {', '.join(p.get('key_topics', []))}. "
            f"Style: {p.get('speaking_style', '')}."
        )
        docs.append(Document(
            page_content=embed_text,
            metadata={
                "speaker_id": p.get("speaker_id", ""),
                "display_name": p.get("display_name", ""),
                "role": p.get("role", ""),
                "meeting_id": meeting_id,
                "source": "speaker_profile",
            },
        ))
    return docs


meeting_id = meta.get("date", "unknown_meeting")
profile_docs = profiles_to_documents(profiles, meeting_id)

profile_store = InMemoryVectorStore.from_documents(profile_docs, embeddings)
print(f"=== 3. Vector store built with {len(profile_docs)} speaker profiles ===")
print()


# ---------------------------------------------------------------------------
# 4. Speaker identification by similarity search
# ---------------------------------------------------------------------------
# Given a snippet of text from an unknown speaker, find the best match
# in the profile store.

def identify_speaker(text_snippet: str, store: InMemoryVectorStore, k: int = 2) -> list[dict]:
    """Find the most similar known speaker profile for a text snippet."""
    results = store.similarity_search_with_score(text_snippet, k=k)
    return [
        {
            "speaker_id": doc.metadata["speaker_id"],
            "display_name": doc.metadata["display_name"],
            "role": doc.metadata["role"],
            "score": round(score, 3),
        }
        for doc, score in results
    ]


print("=== 4. Speaker identification by profile similarity ===")

test_snippets = [
    "We need to prioritize the security fix and make sure it ships this week.",
    "The color palette needs three iterations to meet accessibility contrast ratios.",
    "The authentication service requires a Node upgrade approved by devops.",
    "Let's wrap up — any other blockers before we close?",
]

for snippet in test_snippets:
    matches = identify_speaker(snippet, profile_store)
    print(f"  Snippet: '{snippet[:60]}...'")
    for m in matches:
        print(f"    → {m['display_name']:<20} ({m['speaker_id']})  score={m['score']:.3f}")
    print()


# ---------------------------------------------------------------------------
# 5. Save profiles to JSON for persistence
# ---------------------------------------------------------------------------

profiles_path = TRANSCRIPT_PATH.parent / "speaker_profiles.json"
with open(profiles_path, "w") as f:
    json.dump({
        "meeting_id": meeting_id,
        "profiles": profiles,
        "name_map": name_map,
    }, f, indent=2)

print(f"  Profiles saved: {profiles_path.name}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Speaker profiles embed style + topics + role → richer than name alone
# ✅ profile_text is purpose-built for embedding — make it descriptive
# ✅ InMemoryVectorStore enables similarity-based speaker ID from text alone
# ✅ This pattern extends to multiple meetings: add more docs to the store
# ✅ Speaker similarity search is a fallback when diarization labels are missing
# ✅ Persist profiles to JSON so they accumulate across sessions → better recall
