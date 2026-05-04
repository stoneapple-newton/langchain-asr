"""
Stage 1: One-Shot PII Redaction
===============================
CONCEPT: A single prompt extracts PII spans, then shared utilities apply
canonical placeholders so results are comparable across implementations.

Run this file:
  uv run deep_research/pii_redaction/stage_01_basics/01_one_shot_prompt_redactor.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, structured_output_chain
from deep_research.pii_redaction.shared.pii_utils import (
    PiiPredictionModel,
    dataset_path,
    finalize_prediction,
    format_supported_labels,
    load_dataset,
)


VARIANT_NAME = "one_shot_prompt"

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You remove personally identifiable information from text.\n"
            "Return JSON only.\n"
            "Rules:\n"
            "- Extract only direct PII spans that appear exactly in the input.\n"
            "- Do not paraphrase, normalize, or invent missing text.\n"
            "- Only use these labels:\n{supported_labels}\n\n"
            "{format_instructions}",
        ),
        ("human", "Text:\n{text}"),
    ]
).partial(
    supported_labels=format_supported_labels(),
    format_instructions=parser.get_format_instructions(),
)


def redact_text(text: str) -> dict:
    llm = create_chat_model(temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, prompt, PiiPredictionModel)
    raw = chain.invoke({"text": text})
    return finalize_prediction(
        text,
        raw.get("entities", []),
        variant=VARIANT_NAME,
        notes=raw.get("notes", []),
    )


if __name__ == "__main__":
    example = load_dataset(dataset_path())[0]
    result = redact_text(example["text"])
    print("=" * 60)
    print("  ONE-SHOT PII REDACTION")
    print("=" * 60)
    print(f"Input:    {example['text']}")
    print(f"Output:   {result['redacted_text']}")
    print(f"Entities: {result['entities']}")
