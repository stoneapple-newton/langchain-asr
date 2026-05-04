"""
Stage 1: One-Shot Chinese to English Translation
================================================
CONCEPT: Use a single prompt to translate each transcript segment while
preserving segment alignment for downstream scoring.

Run this file:
  uv run deep_research/asr-v2/translation/stage_01_basics/01_one_shot_translation.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, structured_output_chain

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_shared import (
    TranslationDraftModel,
    build_langsmith_config,
    finalize_translation_result,
    format_example_notes,
    format_segment_preserving_prompt,
    load_dataset,
    load_source_document,
    save_translation_outputs,
    translation_rules,
)


VARIANT_NAME = "one_shot_translation"

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You translate WhisperX-like transcript segments from Chinese into English.\n"
            "{rules}\n\n"
            "Return JSON only.\n{format_instructions}",
        ),
        (
            "human",
            "Example ID: {example_id}\n"
            "Terminology notes:\n{notes}\n\n"
            "Segments:\n{segment_block}",
        ),
    ]
).partial(
    rules=translation_rules(),
    format_instructions=parser.get_format_instructions(),
)


def _run_one_shot_translation(example: dict) -> dict:
    doc = load_source_document(example["source_document"], source_path=f"{example['id']}.json")
    llm = create_chat_model("asr_v2", temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, prompt, TranslationDraftModel)
    return chain.invoke(
        {
            "example_id": example["id"],
            "notes": format_example_notes(example),
            "segment_block": format_segment_preserving_prompt(doc),
        },
        config=build_langsmith_config(
            VARIANT_NAME,
            example,
            extra_tags=["stage_01_basics"],
        ),
    )


def translate_example(example: dict) -> dict:
    raw = _run_one_shot_translation(example)
    return finalize_translation_result(
        example,
        raw.get("translated_segments", []),
        variant=VARIANT_NAME,
        notes=raw.get("notes", []),
        trace_metadata={"variant": VARIANT_NAME, "example_id": example["id"]},
    )


if __name__ == "__main__":
    example = load_dataset()[0]
    result = translate_example(example)
    paths = save_translation_outputs(example, result, ROOT / "outputs")

    print("=" * 72)
    print("ONE-SHOT TRANSLATION")
    print("=" * 72)
    for segment in result["translated_segments"]:
        print(f"{segment['speaker']}: {segment['text']}")
    print(f"\nSaved JSON: {paths['json_path']}")
    print(f"Saved Markdown: {paths['markdown_path']}")
