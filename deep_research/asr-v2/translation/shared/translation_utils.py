from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
ASR_V2_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_transcript_utils_module():
    module_path = ASR_V2_ROOT / "shared" / "transcript_utils.py"
    spec = importlib.util.spec_from_file_location("_asr_v2_transcript_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_transcript_utils = _load_transcript_utils_module()
TranscriptDocument = _transcript_utils.TranscriptDocument
TranscriptSegment = _transcript_utils.TranscriptSegment
WordToken = _transcript_utils.WordToken


class TranslationDraftModel(BaseModel):
    translated_segments: list[str] = Field(
        default_factory=list,
        description="English translations for each input segment in the same order.",
    )
    notes: list[str] = Field(default_factory=list)


def dataset_path() -> Path:
    return ROOT / "dataset" / "translation_dataset.json"


def load_dataset_bundle(path: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(path) if path else dataset_path()
    bundle = json.loads(resolved.read_text(encoding="utf-8"))
    if "dataset_id" not in bundle or "examples" not in bundle:
        raise ValueError(f"Invalid dataset bundle: {resolved}")
    return bundle


def load_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    bundle = load_dataset_bundle(path)
    examples: list[dict[str, Any]] = []
    for raw_example in bundle["examples"]:
        example = dict(raw_example)
        example.setdefault("dataset_id", bundle["dataset_id"])
        example.setdefault("source_language", bundle.get("source_language", "zh"))
        example.setdefault("target_language", bundle.get("target_language", "en"))
        examples.append(example)
    return examples


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_word(raw_word: dict[str, Any]) -> WordToken:
    return WordToken(
        text=str(raw_word.get("word") or raw_word.get("text") or "").strip(),
        start=raw_word.get("start"),
        end=raw_word.get("end"),
        speaker=raw_word.get("speaker"),
        score=raw_word.get("score") or raw_word.get("probability"),
        metadata={
            key: value
            for key, value in raw_word.items()
            if key not in {"word", "text", "start", "end", "speaker", "score", "probability"}
        },
    )


def load_source_document(
    payload: dict[str, Any],
    *,
    source_path: str = "in_memory_translation_source.json",
) -> TranscriptDocument:
    segments: list[TranscriptSegment] = []
    for idx, raw_segment in enumerate(payload.get("segments", [])):
        start = _safe_float(raw_segment.get("start"))
        end = _safe_float(raw_segment.get("end"), default=start)
        segments.append(
            TranscriptSegment(
                segment_id=str(raw_segment.get("id", idx)),
                start=start,
                end=end,
                text=str(raw_segment.get("text") or "").strip(),
                speaker=raw_segment.get("speaker"),
                words=[_normalize_word(word) for word in raw_segment.get("words", [])],
                metadata={
                    key: value
                    for key, value in raw_segment.items()
                    if key not in {"id", "start", "end", "text", "speaker", "words"}
                },
            )
        )

    return TranscriptDocument(
        source_path=source_path,
        language=payload.get("language"),
        raw_data=dict(payload),
        segments=segments,
        metadata={key: value for key, value in payload.items() if key not in {"segments", "language"}},
    )


def translation_rules() -> str:
    return (
        "Translation rules:\n"
        "- Translate Mandarin Chinese into natural English.\n"
        "- Preserve the number of segments exactly.\n"
        "- Do not merge, split, reorder, or omit segments.\n"
        "- Do not change speaker labels or timestamps.\n"
        "- Translate each segment independently but keep local meeting context in mind.\n"
        "- Keep product names, acronyms, and technical terms accurate.\n"
        "- Return English text only for each segment."
    )


def format_segment_preserving_prompt(doc: TranscriptDocument) -> str:
    lines = []
    for index, segment in enumerate(doc.segments):
        speaker = segment.speaker or "UNKNOWN"
        lines.append(
            f"[{index}] [{segment.start:07.2f}-{segment.end:07.2f}] {speaker}: {segment.text}"
        )
    return "\n".join(lines)


def format_example_notes(example: dict[str, Any]) -> str:
    notes = example.get("notes")
    if not notes:
        return "No additional terminology notes."
    if isinstance(notes, list):
        return "\n".join(f"- {note}" for note in notes)
    return str(notes)


def build_langsmith_config(
    variant: str,
    example: dict[str, Any],
    *,
    extra_metadata: dict[str, Any] | None = None,
    extra_tags: list[str] | None = None,
    thread_id: str | None = None,
):
    from langchain_core.runnables import RunnableConfig

    metadata = {
        "variant": variant,
        "dataset_id": example.get("dataset_id", "unknown_dataset"),
        "example_id": example.get("id", "unknown_example"),
        "source_language": example.get("source_language", "zh"),
        "target_language": example.get("target_language", "en"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    tags = [
        f"variant={variant}",
        f"dataset_id={metadata['dataset_id']}",
        f"source_language={metadata['source_language']}",
        f"target_language={metadata['target_language']}",
    ]
    if extra_tags:
        tags.extend(extra_tags)

    return RunnableConfig(
        tags=tags,
        metadata=metadata,
        configurable={"thread_id": thread_id or f"{variant}-{metadata['example_id']}"},
    )


def validate_segment_preservation(
    source_document: dict[str, Any],
    translated_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    source_segments = source_document.get("segments", [])
    errors: list[str] = []

    segment_count_preserved = len(source_segments) == len(translated_segments)
    if not segment_count_preserved:
        errors.append(
            f"segment count changed: expected {len(source_segments)}, got {len(translated_segments)}"
        )

    speakers_preserved = True
    timestamps_preserved = True

    for index, source_segment in enumerate(source_segments):
        if index >= len(translated_segments):
            speakers_preserved = False
            timestamps_preserved = False
            break

        translated = translated_segments[index]
        if (source_segment.get("speaker") or "") != (translated.get("speaker") or ""):
            speakers_preserved = False
            errors.append(
                f"speaker changed at segment {index}: expected {source_segment.get('speaker')!r}, "
                f"got {translated.get('speaker')!r}"
            )

        same_start = abs(_safe_float(source_segment.get("start")) - _safe_float(translated.get("start"))) < 1e-6
        same_end = abs(_safe_float(source_segment.get("end")) - _safe_float(translated.get("end"))) < 1e-6
        if not (same_start and same_end):
            timestamps_preserved = False
            errors.append(
                f"timestamps changed at segment {index}: "
                f"expected ({source_segment.get('start')}, {source_segment.get('end')}), "
                f"got ({translated.get('start')}, {translated.get('end')})"
            )

    return {
        "segment_count_preserved": segment_count_preserved,
        "speakers_preserved": speakers_preserved,
        "timestamps_preserved": timestamps_preserved,
        "structure_preserved": segment_count_preserved and speakers_preserved and timestamps_preserved,
        "errors": errors,
    }


def finalize_translation_result(
    example: dict[str, Any],
    translated_texts: list[str],
    *,
    variant: str,
    notes: list[str] | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_document = example["source_document"]
    source_segments = source_document.get("segments", [])
    if len(translated_texts) != len(source_segments):
        raise ValueError(
            f"Expected {len(source_segments)} translated segments, got {len(translated_texts)}"
        )

    translated_segments = []
    for source_segment, translated_text in zip(source_segments, translated_texts):
        translated_segments.append(
            {
                "id": source_segment.get("id"),
                "start": source_segment.get("start"),
                "end": source_segment.get("end"),
                "speaker": source_segment.get("speaker"),
                "text": str(translated_text).strip(),
                "source_text": source_segment.get("text", ""),
            }
        )

    validation = validate_segment_preservation(source_document, translated_segments)
    if not validation["structure_preserved"]:
        raise ValueError("; ".join(validation["errors"]) or "Translation broke transcript structure.")

    result = {
        "variant": variant,
        "translated_segments": translated_segments,
        "notes": list(notes or []),
    }
    if trace_metadata:
        result["trace_metadata"] = dict(trace_metadata)
    return result


def normalize_english_text(text: str) -> str:
    normalized = str(text).lower()
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"[^a-z0-9'\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _token_counter(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        normalized = normalize_english_text(text)
        if not normalized:
            continue
        counter.update(normalized.split())
    return counter


def evaluate_translation(example: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    reference_segments = list(example.get("reference_segments", []))
    predicted_segments = list(prediction.get("translated_segments", []))
    validation = validate_segment_preservation(example["source_document"], predicted_segments)

    predicted_texts = [segment.get("text", "") for segment in predicted_segments[: len(reference_segments)]]
    if len(predicted_texts) < len(reference_segments):
        predicted_texts.extend([""] * (len(reference_segments) - len(predicted_texts)))

    normalized_reference = [normalize_english_text(text) for text in reference_segments]
    normalized_predicted = [normalize_english_text(text) for text in predicted_texts]

    exact_matches = sum(
        1 for expected, actual in zip(normalized_reference, normalized_predicted) if expected == actual
    )
    segment_total = max(len(reference_segments), 1)

    reference_tokens = _token_counter(reference_segments)
    predicted_tokens = _token_counter(predicted_texts)
    overlap = sum((reference_tokens & predicted_tokens).values())
    predicted_total = sum(predicted_tokens.values())
    reference_total = sum(reference_tokens.values())

    precision = overlap / max(predicted_total, 1)
    recall = overlap / max(reference_total, 1)
    token_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "id": example["id"],
        "variant": prediction.get("variant", "unknown"),
        "segment_exact_match_rate": round(exact_matches / segment_total, 3),
        "full_transcript_exact_match_rate": round(
            1.0 if exact_matches == len(reference_segments) and validation["structure_preserved"] else 0.0,
            3,
        ),
        "token_precision": round(precision, 3),
        "token_recall": round(recall, 3),
        "token_f1": round(token_f1, 3),
        "segment_count_preserved": validation["segment_count_preserved"],
        "speakers_preserved": validation["speakers_preserved"],
        "timestamps_preserved": validation["timestamps_preserved"],
        "structure_preserved": validation["structure_preserved"],
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "segment_exact_match_rate": 0.0,
            "full_transcript_exact_match_rate": 0.0,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
            "structure_preserved_rate": 0.0,
            "segment_count_preserved_rate": 0.0,
            "speakers_preserved_rate": 0.0,
            "timestamps_preserved_rate": 0.0,
        }

    total = len(rows)
    return {
        "cases": total,
        "segment_exact_match_rate": round(sum(row["segment_exact_match_rate"] for row in rows) / total, 3),
        "full_transcript_exact_match_rate": round(
            sum(row["full_transcript_exact_match_rate"] for row in rows) / total,
            3,
        ),
        "token_precision": round(sum(row["token_precision"] for row in rows) / total, 3),
        "token_recall": round(sum(row["token_recall"] for row in rows) / total, 3),
        "token_f1": round(sum(row["token_f1"] for row in rows) / total, 3),
        "structure_preserved_rate": round(
            sum(1 for row in rows if row["structure_preserved"]) / total,
            3,
        ),
        "segment_count_preserved_rate": round(
            sum(1 for row in rows if row["segment_count_preserved"]) / total,
            3,
        ),
        "speakers_preserved_rate": round(
            sum(1 for row in rows if row["speakers_preserved"]) / total,
            3,
        ),
        "timestamps_preserved_rate": round(
            sum(1 for row in rows if row["timestamps_preserved"]) / total,
            3,
        ),
    }


def to_translated_json(example: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "dataset_id": example.get("dataset_id"),
        "source_language": example.get("source_language"),
        "target_language": example.get("target_language"),
        "variant": prediction["variant"],
        "notes": prediction.get("notes", []),
        "segments": prediction["translated_segments"],
    }


def render_translation_markdown(example: dict[str, Any], prediction: dict[str, Any]) -> str:
    lines = [
        "# Segment-Preserving Translation",
        "",
        f"- Example: `{example['id']}`",
        f"- Variant: `{prediction['variant']}`",
        f"- Language Pair: `{example['source_language']} -> {example['target_language']}`",
        f"- Segments: `{len(prediction['translated_segments'])}`",
        "",
    ]
    for segment in prediction["translated_segments"]:
        lines.append(
            f"- [{segment['start']:07.2f}-{segment['end']:07.2f}] "
            f"**{segment.get('speaker') or 'UNKNOWN'}**: {segment['text']}"
        )
    lines.append("")
    return "\n".join(lines)


def save_translation_outputs(
    example: dict[str, Any],
    prediction: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    run_dir = output_dir / example["id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / f"{example['id']}.{prediction['variant']}.json"
    markdown_path = run_dir / f"{example['id']}.{prediction['variant']}.md"
    json_path.write_text(
        json.dumps(to_translated_json(example, prediction), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_translation_markdown(example, prediction), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def copy_source_baseline(example: dict[str, Any]) -> dict[str, Any]:
    source_texts = [segment.get("text", "") for segment in example["source_document"].get("segments", [])]
    return finalize_translation_result(
        example,
        source_texts,
        variant="copy_source_baseline",
        notes=["Deterministic baseline that copies the Chinese source text unchanged."],
    )
