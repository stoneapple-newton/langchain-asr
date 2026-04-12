from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TRANSLATION_ROOT = ROOT / "deep_research" / "asr-v2" / "translation"
if str(TRANSLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSLATION_ROOT))

from translation_shared import (
    copy_source_baseline,
    dataset_path,
    evaluate_translation,
    finalize_translation_result,
    format_segment_preserving_prompt,
    load_dataset,
    load_dataset_bundle,
    load_source_document,
    normalize_english_text,
    save_translation_outputs,
    summarize_results,
    validate_segment_preservation,
)


def test_dataset_loads_and_examples_are_aligned():
    bundle = load_dataset_bundle(dataset_path())
    examples = load_dataset(dataset_path())

    assert bundle["dataset_id"] == "asr_translation_zh_en_v1"
    assert len(examples) >= 4
    assert all(example["source_language"] == "zh" for example in examples)
    assert all(example["target_language"] == "en" for example in examples)
    assert all(
        len(example["source_document"]["segments"]) == len(example["reference_segments"])
        for example in examples
    )


def test_load_source_document_and_prompt_rendering():
    example = load_dataset()[0]
    doc = load_source_document(example["source_document"], source_path="sample.json")

    assert doc.language == "zh"
    assert len(doc.segments) == len(example["reference_segments"])
    assert doc.segments[0].segment_id == "0"

    prompt_block = format_segment_preserving_prompt(doc)
    assert "[0]" in prompt_block
    assert "SPEAKER_00" in prompt_block


def test_normalize_english_text_collapses_case_and_punctuation():
    text = " Hello, WORLD!  We're   shipping. "
    assert normalize_english_text(text) == "hello world we're shipping"


def test_validate_segment_preservation_detects_speaker_change():
    example = load_dataset()[0]
    broken_segments = [
        {
            "id": segment["id"],
            "start": segment["start"],
            "end": segment["end"],
            "speaker": "SPEAKER_99" if index == 0 else segment["speaker"],
            "text": "placeholder",
        }
        for index, segment in enumerate(example["source_document"]["segments"])
    ]

    validation = validate_segment_preservation(example["source_document"], broken_segments)

    assert validation["segment_count_preserved"] is True
    assert validation["speakers_preserved"] is False
    assert validation["structure_preserved"] is False


def test_exact_match_metrics_and_summary():
    example = load_dataset()[0]
    prediction = finalize_translation_result(
        example,
        example["reference_segments"],
        variant="test_variant",
        notes=["perfect stub"],
    )

    metrics = evaluate_translation(example, prediction)
    summary = summarize_results([metrics, metrics])

    assert metrics["segment_exact_match_rate"] == 1.0
    assert metrics["full_transcript_exact_match_rate"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["structure_preserved"] is True
    assert summary["cases"] == 2
    assert summary["token_f1"] == 1.0


def test_copy_source_baseline_preserves_structure_but_scores_poorly():
    example = load_dataset()[0]
    prediction = copy_source_baseline(example)
    metrics = evaluate_translation(example, prediction)

    assert metrics["structure_preserved"] is True
    assert metrics["segment_exact_match_rate"] == 0.0
    assert metrics["token_f1"] == 0.0


def test_save_translation_outputs_writes_json_and_markdown(tmp_path):
    example = load_dataset()[0]
    prediction = finalize_translation_result(
        example,
        example["reference_segments"],
        variant="test_variant",
    )

    paths = save_translation_outputs(example, prediction, tmp_path)
    json_path = Path(paths["json_path"])
    markdown_path = Path(paths["markdown_path"])

    assert json_path.exists()
    assert markdown_path.exists()
    assert '"variant": "test_variant"' in json_path.read_text(encoding="utf-8")
    assert "# Segment-Preserving Translation" in markdown_path.read_text(encoding="utf-8")
