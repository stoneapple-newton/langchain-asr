from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


SUPPORTED_LABELS = {
    "PERSON": "[PERSON]",
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
    "ADDRESS": "[ADDRESS]",
    "SSN": "[SSN]",
    "DOB": "[DOB]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "IP_ADDRESS": "[IP_ADDRESS]",
    "EMPLOYEE_ID": "[EMPLOYEE_ID]",
    "MRN": "[MRN]",
    "ACCOUNT_NUMBER": "[ACCOUNT_NUMBER]",
    "ROUTING_NUMBER": "[ROUTING_NUMBER]",
    "PASSPORT_NUMBER": "[PASSPORT_NUMBER]",
}

LABEL_ALIASES = {
    "NAME": "PERSON",
    "FULL_NAME": "PERSON",
    "PERSON_NAME": "PERSON",
    "DATE_OF_BIRTH": "DOB",
    "BIRTHDATE": "DOB",
    "DATE": "DOB",
    "CARD_NUMBER": "CREDIT_CARD",
    "CREDITCARD": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "IP": "IP_ADDRESS",
    "EMP_ID": "EMPLOYEE_ID",
    "EMPLOYEEID": "EMPLOYEE_ID",
    "MEDICAL_RECORD_NUMBER": "MRN",
    "BANK_ACCOUNT": "ACCOUNT_NUMBER",
    "ACCOUNT": "ACCOUNT_NUMBER",
    "ROUTING": "ROUTING_NUMBER",
    "PASSPORT": "PASSPORT_NUMBER",
}


class PiiEntityModel(BaseModel):
    label: str = Field(description="Supported label for the PII entity.")
    value: str = Field(description="Exact substring copied from the source text.")
    justification: str = Field(
        default="",
        description="Short reason this value should be redacted.",
    )


class PiiPredictionModel(BaseModel):
    entities: list[PiiEntityModel] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ResolvedEntity:
    label: str
    value: str
    start: int
    end: int
    justification: str = ""


def dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dataset" / "pii_redaction_dataset.json"


def load_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    resolved = Path(path) if path else dataset_path()
    return json.loads(resolved.read_text(encoding="utf-8"))


def normalize_label(label: str | None) -> str | None:
    if not label:
        return None
    cleaned = str(label).strip().upper().replace(" ", "_")
    cleaned = LABEL_ALIASES.get(cleaned, cleaned)
    if cleaned in SUPPORTED_LABELS:
        return cleaned
    return None


def format_supported_labels() -> str:
    return "\n".join(
        f"- {label}: replace with {placeholder}"
        for label, placeholder in SUPPORTED_LABELS.items()
    )


def _coerce_entities(raw_entities: list[Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for item in raw_entities:
        if isinstance(item, PiiEntityModel):
            entities.append(item.model_dump())
        elif isinstance(item, dict):
            entities.append(item)
    return entities


def _find_unused_span(
    text: str,
    value: str,
    used_spans: list[tuple[int, int]],
) -> tuple[int, int] | None:
    start = text.find(value)
    while start != -1:
        end = start + len(value)
        if not any(
            not (end <= used_start or start >= used_end)
            for used_start, used_end in used_spans
        ):
            return start, end
        start = text.find(value, start + 1)

    lowered_text = text.lower()
    lowered_value = value.lower()
    start = lowered_text.find(lowered_value)
    while start != -1:
        end = start + len(value)
        if text[start:end].lower() == lowered_value and not any(
            not (end <= used_start or start >= used_end)
            for used_start, used_end in used_spans
        ):
            return start, end
        start = lowered_text.find(lowered_value, start + 1)
    return None


def normalize_entities(text: str, raw_entities: list[Any]) -> list[ResolvedEntity]:
    cleaned = _coerce_entities(raw_entities)
    cleaned.sort(key=lambda item: len(str(item.get("value", ""))), reverse=True)

    used_spans: list[tuple[int, int]] = []
    seen_keys: set[tuple[str, str]] = set()
    resolved: list[ResolvedEntity] = []

    for item in cleaned:
        label = normalize_label(item.get("label"))
        value = str(item.get("value", "")).strip()
        justification = str(item.get("justification", "")).strip()
        if not label or not value:
            continue

        key = (label, value.lower())
        if key in seen_keys:
            continue

        span = _find_unused_span(text, value, used_spans)
        if span is None:
            continue

        start, end = span
        used_spans.append((start, end))
        seen_keys.add(key)
        resolved.append(
            ResolvedEntity(
                label=label,
                value=text[start:end],
                start=start,
                end=end,
                justification=justification,
            )
        )

    resolved.sort(key=lambda entity: entity.start)
    return resolved


def apply_redaction(text: str, entities: list[ResolvedEntity]) -> str:
    redacted = text
    for entity in sorted(entities, key=lambda item: item.start, reverse=True):
        placeholder = SUPPORTED_LABELS[entity.label]
        redacted = redacted[:entity.start] + placeholder + redacted[entity.end :]
    return redacted


def finalize_prediction(
    text: str,
    raw_entities: list[Any],
    *,
    variant: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    entities = normalize_entities(text, raw_entities)
    redacted_text = apply_redaction(text, entities)
    return {
        "variant": variant,
        "entities": [
            {
                "label": entity.label,
                "value": entity.value,
                "start": entity.start,
                "end": entity.end,
                "justification": entity.justification,
            }
            for entity in entities
        ],
        "redacted_text": redacted_text,
        "notes": notes or [],
    }


REGEX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("EMPLOYEE_ID", re.compile(r"\bEMP-\d{3,}\b", re.IGNORECASE)),
    ("PASSPORT_NUMBER", re.compile(r"\b[A-Z]\d{7}\b")),
]

ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Terrace|Lane|Ln|Drive|Dr)"
    r"(?:,\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})*"
    r"(?:,\s*[A-Z]{2}\s+\d{5})?\b"
)


def regex_candidate_entities(text: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []

    for label, pattern in REGEX_PATTERNS:
        for match in pattern.finditer(text):
            entities.append(
                {
                    "label": label,
                    "value": match.group(0),
                    "justification": "Matched deterministic regex candidate.",
                }
            )

    dob_match = re.search(r"\bDOB\s+(\d{4}-\d{2}-\d{2})\b", text, flags=re.IGNORECASE)
    if dob_match:
        entities.append(
            {
                "label": "DOB",
                "value": dob_match.group(1),
                "justification": "Appears after DOB marker.",
            }
        )

    mrn_match = re.search(r"\bMRN\s+([A-Z]?\d{5,8})\b", text, flags=re.IGNORECASE)
    if mrn_match:
        entities.append(
            {
                "label": "MRN",
                "value": mrn_match.group(1),
                "justification": "Appears after MRN marker.",
            }
        )

    account_match = re.search(r"\baccount\s+(\d{8,12})\b", text, flags=re.IGNORECASE)
    if account_match:
        entities.append(
            {
                "label": "ACCOUNT_NUMBER",
                "value": account_match.group(1),
                "justification": "Appears after account marker.",
            }
        )

    routing_match = re.search(r"\brouting\s+(\d{9})\b", text, flags=re.IGNORECASE)
    if routing_match:
        entities.append(
            {
                "label": "ROUTING_NUMBER",
                "value": routing_match.group(1),
                "justification": "Appears after routing marker.",
            }
        )

    for match in ADDRESS_PATTERN.finditer(text):
        entities.append(
            {
                "label": "ADDRESS",
                "value": match.group(0),
                "justification": "Matched address-like pattern.",
            }
        )

    return entities


def evaluate_example(example: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = {
        (normalize_label(entity["label"]), str(entity["value"]).lower())
        for entity in example.get("entities", [])
    }
    predicted = {
        (normalize_label(entity["label"]), str(entity["value"]).lower())
        for entity in prediction.get("entities", [])
    }

    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "id": example["id"],
        "variant": prediction.get("variant", "unknown"),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "exact_redaction": prediction.get("redacted_text", "") == example.get("expected_redacted_text", ""),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "cases": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_redaction_rate": 0.0,
        }

    tp = sum(result["true_positives"] for result in results)
    fp = sum(result["false_positives"] for result in results)
    fn = sum(result["false_negatives"] for result in results)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact_rate = sum(1 for result in results if result["exact_redaction"]) / len(results)
    return {
        "cases": len(results),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "exact_redaction_rate": round(exact_rate, 3),
    }
