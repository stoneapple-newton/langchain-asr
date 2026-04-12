from __future__ import annotations

import json
from pathlib import Path

from deep_research.asr_medical_verification.shared.detection import identify_error_candidates
from deep_research.asr_medical_verification.shared.medical_glossary import (
    build_llm_hint,
    link_medical_terms,
    load_medical_glossary,
)
from deep_research.asr_medical_verification.shared.transcript_utils import (
    apply_corrections,
    load_transcript,
)
from deep_research.asr_medical_verification.workflow import run_asr_medical_verification


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRANSCRIPT = ROOT / "deep_research" / "asr_medical_verification" / "data" / "sample_transcript.json"
SAMPLE_GLOSSARY = ROOT / "deep_research" / "asr_medical_verification" / "data" / "medical_glossary.json"


def test_glossary_links_phonetic_and_multilingual_variants():
    glossary = load_medical_glossary(SAMPLE_GLOSSARY)

    linked = link_medical_terms("The ECG suggests a trial fibrillation episode.", glossary)
    assert any(item["canonical_en"] == "atrial fibrillation" for item in linked)

    chinese_linked = link_medical_terms("患者今天出现了高血压。", glossary)
    assert any(item["canonical_en"] == "hypertension" for item in chinese_linked)

    hint = build_llm_hint("cardiology", glossary, max_terms=3)
    assert "atrial fibrillation" in hint


def test_detection_finds_error_and_medical_candidates():
    doc = load_transcript(SAMPLE_TRANSCRIPT)
    glossary = load_medical_glossary(SAMPLE_GLOSSARY)

    result = identify_error_candidates(doc, glossary, confidence_threshold=0.75)

    assert result["all_errors"]
    assert result["medical_candidates"]
    assert any(item.error_type == "low_confidence" for item in result["all_errors"])
    assert any(item.specialty == "cardiology" for item in result["medical_candidates"])


def test_apply_corrections_only_updates_accepted_high_confidence_records():
    doc = load_transcript(SAMPLE_TRANSCRIPT)
    corrected_doc, applied = apply_corrections(
        doc,
        [
            {
                "segment_id": "3",
                "corrected_text": "We should consider starting him on lisinopril for the hypertension.",
                "confidence": 0.91,
                "method": "llm_text",
                "correction_type": "medical_term",
                "medical_terms": ["lisinopril", "hypertension"],
                "accepted": True,
            },
            {
                "segment_id": "4",
                "corrected_text": "ignored change",
                "confidence": 0.4,
                "method": "llm_text",
                "correction_type": "error",
                "medical_terms": [],
                "accepted": True,
            },
        ],
    )

    assert len(applied) == 1
    assert corrected_doc.segments[3].text.endswith("the hypertension.")
    assert corrected_doc.segments[4].text != "ignored change"


def test_workflow_overwrites_transcript_and_writes_sidecar(monkeypatch):
    tmp_dir = ROOT / "tests" / "_tmp_asr_medical_verification"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = tmp_dir / "sample_transcript.json"
    transcript_path.write_text(SAMPLE_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    def fake_retranscribe(
        wav_path,
        *,
        method,
        context_text,
        original_text,
        issue_reason,
        language="en",
        specialty_hint="",
        llm=None,
    ):
        if "trial fibrillation" in original_text:
            return "The ECG showed signs of atrial fibrillation, which is concerning.", 0.95, "llm_text"
        if "lissanopril" in original_text:
            return "We should consider starting him on lisinopril for the hypertension.", 0.94, "llm_text"
        if "atorvastaten" in original_text:
            return "He's already on atorvastatin and metformin for the diabetes.", 0.93, "llm_text"
        return original_text, 0.4, "llm_text"

    monkeypatch.setattr(
        "deep_research.asr_medical_verification.workflow.retranscribe_segment",
        fake_retranscribe,
    )

    result = run_asr_medical_verification(
        transcript_path,
        glossary_path=SAMPLE_GLOSSARY,
        retranscription_method="auto",
        confidence_threshold=0.72,
    )

    saved = json.loads(transcript_path.read_text(encoding="utf-8"))
    texts = [segment["text"] for segment in saved["segments"]]
    assert any("atrial fibrillation" in text for text in texts)
    assert any("lisinopril" in text for text in texts)

    sidecar_path = transcript_path.with_suffix(".corrections.json")
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["applied_count"] >= 2
    assert str(transcript_path) in result["output_paths"]
