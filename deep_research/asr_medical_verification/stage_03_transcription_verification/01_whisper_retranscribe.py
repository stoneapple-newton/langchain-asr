"""
Stage 3a: Whisper Re-transcription
==================================
Re-transcribe extracted audio segments using Faster-Whisper.
This provides an alternative transcription for comparison.

Run this file:
  uv run deep_research/asr_medical_verification/stage_03_transcription_verification/01_whisper_retranscribe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.asr_medical_verification.shared.audio_utils import (
    FasterWhisperTranscriber,
    FASTER_WHISPER_AVAILABLE,
)


def retranscribe_segments(
    extraction_report_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
) -> dict[str, Any]:
    """
    Re-transcribe extracted audio segments using Faster-Whisper.
    
    Args:
        extraction_report_path: Path to extraction report from stage 2
        model_size: Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
        language: Language code (if None, auto-detect)
    
    Returns:
        Updated report with re-transcriptions
    """
    if not FASTER_WHISPER_AVAILABLE:
        print("Warning: faster-whisper not installed.")
        print("Install with: pip install faster-whisper")
        return {"error": "faster-whisper not installed"}
    
    extraction_report_path = Path(extraction_report_path)
    report = json.loads(extraction_report_path.read_text())
    
    audio_extraction = report.get("audio_extraction", {})
    if not audio_extraction.get("success"):
        print("Audio extraction was not successful, cannot re-transcribe.")
        return report
    
    segments = audio_extraction.get("segments", [])
    if not segments:
        print("No segments to transcribe.")
        return report
    
    print("=" * 60)
    print("Faster-Whisper Re-transcription")
    print("=" * 60)
    print(f"Model: {model_size}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Segments to process: {len(segments)}")
    print()
    
    # Initialize transcriber
    print("Loading Faster-Whisper model...")
    transcriber = FasterWhisperTranscriber(
        model_size=model_size,
        device="cpu",
        compute_type="int8",
    )
    print("Model loaded.")
    print()
    
    # Re-transcribe each segment
    retranscriptions = []
    
    for i, segment_info in enumerate(segments):
        if "segment_path" not in segment_info:
            print(f"[{i+1}/{len(segments)}] Skipping (no audio file)")
            continue
        
        segment_path = Path(segment_info["segment_path"])
        
        if not segment_path.exists():
            print(f"[{i+1}/{len(segments)}] Audio file not found: {segment_path}")
            continue
        
        print(f"[{i+1}/{len(segments)}] Transcribing: {segment_path.name}")
        print(f"    Original word: '{segment_info['word']}'")
        
        try:
            # Transcribe
            result = transcriber.transcribe(
                segment_path,
                language=language,
                word_timestamps=True,
            )
            
            retranscriptions.append({
                "candidate_index": segment_info.get("candidate_index"),
                "original_word": segment_info["word"],
                "error_type": segment_info["error_type"],
                "segment_path": str(segment_path),
                "whisper_transcription": result["text"],
                "detected_language": result["language"],
                "language_probability": result["language_probability"],
                "segments": result["segments"],
            })
            
            print(f"    Re-transcribed: '{result['text']}'")
            print(f"    Language: {result['language']} ({result['language_probability']:.2%})")
            print()
            
        except Exception as e:
            print(f"    Error: {e}")
            retranscriptions.append({
                "candidate_index": segment_info.get("candidate_index"),
                "original_word": segment_info["word"],
                "error_type": segment_info["error_type"],
                "segment_path": str(segment_path),
                "error": str(e),
            })
    
    # Also process medical term segments if available
    medical_retranscriptions = []
    medical_extraction = report.get("medical_term_extraction", {})
    
    if medical_extraction.get("success"):
        medical_segments = medical_extraction.get("segments", [])
        print(f"\nProcessing {len(medical_segments)} medical term segments...")
        
        for i, segment_info in enumerate(medical_segments):
            segment_path = Path(segment_info["segment_path"])
            
            if not segment_path.exists():
                continue
            
            print(f"[Medical {i+1}/{len(medical_segments)}] {segment_info['term_id']}")
            
            try:
                result = transcriber.transcribe(segment_path, language=language)
                
                medical_retranscriptions.append({
                    "term_index": segment_info["term_index"],
                    "term_id": segment_info["term_id"],
                    "matched_text": segment_info["matched_text"],
                    "english": segment_info["english"],
                    "whisper_transcription": result["text"],
                    "detected_language": result["language"],
                })
                
                print(f"    Expected: '{segment_info['english']}'")
                print(f"    Heard: '{result['text']}'")
                print()
                
            except Exception as e:
                print(f"    Error: {e}")
    
    # Update report
    report["whisper_retranscription"] = {
        "model_size": model_size,
        "language": language,
        "total_processed": len(retranscriptions),
        "total_successful": len([r for r in retranscriptions if "error" not in r]),
        "total_failed": len([r for r in retranscriptions if "error" in r]),
        "retranscriptions": retranscriptions,
        "medical_retranscriptions": medical_retranscriptions,
    }
    
    print("=" * 60)
    print("Re-transcription Summary")
    print("=" * 60)
    print(f"Total processed: {report['whisper_retranscription']['total_processed']}")
    print(f"Successful: {report['whisper_retranscription']['total_successful']}")
    print(f"Failed: {report['whisper_retranscription']['total_failed']}")
    
    return report


def compare_transcriptions(
    original: str,
    whisper_result: str,
) -> dict[str, Any]:
    """
    Compare original transcription with Whisper re-transcription.
    
    Returns comparison metrics and suggested correction.
    """
    # Normalize for comparison
    orig_norm = original.lower().strip()
    whisper_norm = whisper_result.lower().strip()
    
    # Simple string similarity
    if orig_norm == whisper_norm:
        similarity = 1.0
        is_match = True
    else:
        # Calculate word overlap
        orig_words = set(orig_norm.split())
        whisper_words = set(whisper_norm.split())
        
        if orig_words and whisper_words:
            intersection = orig_words & whisper_words
            union = orig_words | whisper_words
            similarity = len(intersection) / len(union)
        else:
            similarity = 0.0
        
        is_match = similarity > 0.8
    
    return {
        "original": original,
        "whisper_transcription": whisper_result,
        "similarity_score": round(similarity, 3),
        "is_match": is_match,
        "suggested_correction": whisper_result if not is_match else original,
        "confidence": "high" if similarity > 0.9 else ("medium" if similarity > 0.6 else "low"),
    }


def generate_correction_suggestions(report: dict[str, Any]) -> dict[str, Any]:
    """
    Generate correction suggestions based on Whisper re-transcriptions.
    """
    whisper_data = report.get("whisper_retranscription", {})
    retranscriptions = whisper_data.get("retranscriptions", [])
    
    if not retranscriptions:
        return report
    
    corrections = []
    
    for item in retranscriptions:
        if "error" in item:
            continue
        
        original_word = item["original_word"]
        whisper_text = item["whisper_transcription"]
        
        comparison = compare_transcriptions(original_word, whisper_text)
        
        corrections.append({
            "candidate_index": item["candidate_index"],
            "original_word": original_word,
            "error_type": item["error_type"],
            **comparison,
        })
    
    report["correction_suggestions"] = {
        "total_suggestions": len(corrections),
        "high_confidence": len([c for c in corrections if c["confidence"] == "high"]),
        "medium_confidence": len([c for c in corrections if c["confidence"] == "medium"]),
        "low_confidence": len([c for c in corrections if c["confidence"] == "low"]),
        "corrections": corrections,
    }
    
    return report


def main():
    """Main entry point."""
    print("=" * 60)
    print("ASR Medical Verification - Stage 3a: Whisper Re-transcription")
    print("=" * 60)
    
    # Load extraction report from stage 2
    stage2_dir = Path(__file__).parent.parent / "stage_02_audio_extraction"
    extraction_report_path = stage2_dir / "extraction_report.json"
    
    if not extraction_report_path.exists():
        print(f"Extraction report not found: {extraction_report_path}")
        print("Please run stage 02_audio_extraction first.")
        return
    
    # Run re-transcription
    report = retranscribe_segments(
        extraction_report_path=extraction_report_path,
        model_size="base",
        language=None,  # Auto-detect
    )
    
    # Generate correction suggestions
    report = generate_correction_suggestions(report)
    
    # Save updated report
    output_path = Path(__file__).parent / "whisper_verification_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nVerification report saved to: {output_path}")
    
    # Print high-confidence corrections
    corrections = report.get("correction_suggestions", {}).get("corrections", [])
    high_conf = [c for c in corrections if c["confidence"] == "high" and not c["is_match"]]
    
    if high_conf:
        print("\nHigh-confidence corrections suggested:")
        for c in high_conf[:5]:
            print(f"  '{c['original_word']}' -> '{c['suggested_correction']}'")


if __name__ == "__main__":
    main()
