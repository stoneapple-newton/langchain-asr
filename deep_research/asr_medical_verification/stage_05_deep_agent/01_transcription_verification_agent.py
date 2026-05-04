"""
Stage 5: Deep Agent for Transcription Verification
==================================================
A comprehensive Deep Agent that orchestrates the entire workflow:
1. Identifies errors in transcription
2. Extracts corresponding audio segments
3. Verifies using Faster-Whisper or multimodal LLM
4. Updates transcription with corrections
5. Detects and verifies medical terms using multilingual glossary

Run this file:
  uv run deep_research/asr_medical_verification/stage_05_deep_agent/01_transcription_verification_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain.tools import tool
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model

# Import our shared modules
from deep_research.asr_medical_verification.shared.transcript_utils import (
    load_transcript,
    analyze_transcript,
    find_low_confidence_regions,
    ErrorCandidate,
    document_to_dict,
    save_document,
    TranscriptDocument,
)
from deep_research.asr_medical_verification.shared.medical_glossary import (
    load_medical_glossary,
    MultilingualMedicalGlossary,
    MedicalTerm,
    detect_medical_terms,
)
from deep_research.asr_medical_verification.shared.audio_utils import (
    AudioSegmentExtractor,
    FasterWhisperTranscriber,
    FASTER_WHISPER_AVAILABLE,
)

# Try to import deepagents
try:
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver
    DEEP_AGENTS_AVAILABLE = True
except ImportError:
    DEEP_AGENTS_AVAILABLE = False
    print("Warning: deepagents not installed. Falling back to basic agent.")


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class TranscriptionError(BaseModel):
    """A detected transcription error."""
    segment_idx: int = Field(description="Index of the segment containing the error")
    word_idx: int = Field(description="Index of the word in the segment")
    original_word: str = Field(description="The original (potentially incorrect) word")
    confidence: float = Field(description="ASR confidence score")
    context: str = Field(description="Surrounding text for context")
    error_type: str = Field(description="Type of error: low_confidence, homophone, medical_term, unknown")


class CorrectionSuggestion(BaseModel):
    """A suggested correction for a transcription error."""
    error: TranscriptionError = Field(description="The original error")
    suggested_word: str = Field(description="The suggested correction")
    confidence: str = Field(description="Confidence level: high, medium, low")
    reasoning: str = Field(description="Explanation for the correction")
    verification_method: str = Field(description="How this was verified: whisper, llm, glossary")


class MedicalTermMatch(BaseModel):
    """A detected medical term."""
    matched_text: str = Field(description="Text as it appeared in transcript")
    term_id: str = Field(description="ID of the term in glossary")
    english: str = Field(description="Standard English form")
    category: str = Field(description="Medical category")
    translations: dict[str, str] = Field(description="Translations in other languages")
    needs_verification: bool = Field(description="Whether this needs audio verification")


class VerificationResult(BaseModel):
    """Result of audio verification."""
    original_text: str = Field(description="Original transcription")
    verified_text: str = Field(description="Text after verification")
    is_correct: bool = Field(description="Whether original was correct")
    confidence: str = Field(description="Verification confidence")
    method: str = Field(description="Verification method used")


class TranscriptionVerificationReport(BaseModel):
    """Complete verification report."""
    transcript_path: str = Field(description="Path to input transcript")
    audio_path: str | None = Field(description="Path to source audio")
    errors_detected: list[TranscriptionError] = Field(default_factory=list)
    medical_terms_found: list[MedicalTermMatch] = Field(default_factory=list)
    corrections: list[CorrectionSuggestion] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Tools for the Agent
# ============================================================================

llm = create_chat_model(temperature=0, max_tokens=4096)


@tool
def identify_transcription_errors(
    transcript_path: str,
    confidence_threshold: float = 0.75,
) -> str:
    """
    Identify potential errors in a transcription.
    
    Args:
        transcript_path: Path to the transcript JSON file
        confidence_threshold: Minimum confidence score to flag (default 0.75)
    
    Returns:
        JSON string with detected errors
    """
    doc = load_transcript(transcript_path)
    
    # Find low confidence regions
    candidates = find_low_confidence_regions(doc, threshold=confidence_threshold)
    
    errors = []
    for c in candidates:
        errors.append({
            "segment_idx": c.segment_idx,
            "word_idx": c.word_idx,
            "original_word": c.word,
            "confidence": c.score,
            "context": f"{c.left_context} [{c.word}] {c.right_context}",
            "start_time": c.start_time,
            "end_time": c.end_time,
            "error_type": c.error_type,
        })
    
    # Also detect homophones and unknown terms
    # (simplified for this example)
    
    result = {
        "transcript_path": transcript_path,
        "total_segments": len(doc.segments),
        "errors_found": len(errors),
        "errors": errors,
    }
    
    return json.dumps(result, indent=2)


@tool
def extract_audio_for_verification(
    transcript_path: str,
    segment_indices: list[int],
    output_dir: str,
    padding: float = 1.0,
) -> str:
    """
    Extract audio segments for verification.
    
    Args:
        transcript_path: Path to transcript JSON
        segment_indices: List of segment indices to extract
        output_dir: Directory to save extracted audio
        padding: Extra audio padding in seconds
    
    Returns:
        JSON with paths to extracted audio files
    """
    doc = load_transcript(transcript_path)
    
    if not doc.audio_path:
        return json.dumps({"error": "No audio path found in transcript"})
    
    audio_path = Path(doc.audio_path)
    if not audio_path.exists():
        return json.dumps({"error": f"Audio file not found: {audio_path}"})
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = AudioSegmentExtractor(audio_path)
    extracted = []
    
    for idx in segment_indices:
        if idx >= len(doc.segments):
            continue
        
        segment = doc.segments[idx]
        
        try:
            audio_seg = extractor.extract_segment(
                start_time=segment.start,
                end_time=segment.end,
                output_format="wav",
                sample_rate=16000,
                padding=padding,
            )
            
            output_path = output_dir / f"segment_{idx:04d}.wav"
            audio_seg.save(output_path)
            
            extracted.append({
                "segment_idx": idx,
                "audio_path": str(output_path),
                "original_text": segment.text,
                "start_time": segment.start,
                "end_time": segment.end,
            })
        except Exception as e:
            extracted.append({
                "segment_idx": idx,
                "error": str(e),
            })
    
    return json.dumps({
        "total_extracted": len([e for e in extracted if "error" not in e]),
        "total_failed": len([e for e in extracted if "error" in e]),
        "segments": extracted,
    }, indent=2)


@tool
def verify_with_whisper(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
) -> str:
    """
    Verify audio transcription using Faster-Whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (None for auto-detect)
    
    Returns:
        JSON with transcription result
    """
    if not FASTER_WHISPER_AVAILABLE:
        return json.dumps({"error": "faster-whisper not installed"})
    
    try:
        transcriber = FasterWhisperTranscriber(model_size=model_size)
        result = transcriber.transcribe(audio_path, language=language)
        
        return json.dumps({
            "success": True,
            "transcription": result["text"],
            "language": result["language"],
            "confidence": result["language_probability"],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def detect_medical_terms_in_transcript(
    transcript_path: str,
    target_languages: list[str] | None = None,
) -> str:
    """
    Detect medical terms in transcription using multilingual glossary.
    
    Args:
        transcript_path: Path to transcript JSON
        target_languages: List of language codes to include (default: en, es, fr)
    
    Returns:
        JSON with detected medical terms and translations
    """
    if target_languages is None:
        target_languages = ["en", "es", "fr"]
    
    doc = load_transcript(transcript_path)
    glossary = load_medical_glossary()
    
    # Detect terms in full text
    full_text = " ".join(s.text for s in doc.segments)
    matches = detect_medical_terms(full_text, glossary)
    
    results = []
    for matched_text, term in matches:
        # Find segment location
        segment_idx = None
        for i, seg in enumerate(doc.segments):
            if matched_text.lower() in seg.text.lower():
                segment_idx = i
                break
        
        translations = {
            lang: term.get_term(lang)
            for lang in target_languages
            if lang in term.translations or lang == "en"
        }
        
        results.append({
            "matched_text": matched_text,
            "term_id": term.term_id,
            "english": term.english,
            "category": term.category,
            "translations": translations,
            "synonyms": term.synonyms,
            "abbreviations": term.abbreviations,
            "segment_idx": segment_idx,
        })
    
    return json.dumps({
        "total_terms": len(results),
        "terms": results,
    }, indent=2)


@tool
def suggest_transcription_corrections(
    transcript_path: str,
    verification_results: str,  # JSON string
) -> str:
    """
    Suggest corrections based on verification results.
    
    Args:
        transcript_path: Path to transcript JSON
        verification_results: JSON string with verification results
    
    Returns:
        JSON with suggested corrections
    """
    doc = load_transcript(transcript_path)
    verifications = json.loads(verification_results)
    
    corrections = []
    
    for verification in verifications.get("results", []):
        original = verification.get("original_text", "")
        verified = verification.get("verified_text", "")
        
        if original != verified:
            corrections.append({
                "segment_idx": verification.get("segment_idx"),
                "original": original,
                "suggested": verified,
                "confidence": verification.get("confidence", "low"),
                "method": verification.get("method"),
            })
    
    return json.dumps({
        "total_corrections": len(corrections),
        "corrections": corrections,
    }, indent=2)


@tool
def apply_corrections_to_transcript(
    transcript_path: str,
    corrections: str,  # JSON string
    output_path: str,
) -> str:
    """
    Apply corrections to a transcript and save the result.
    
    Args:
        transcript_path: Path to original transcript
        corrections: JSON string with corrections to apply
        output_path: Where to save corrected transcript
    
    Returns:
        JSON with result summary
    """
    doc = load_transcript(transcript_path)
    corrections_list = json.loads(corrections).get("corrections", [])
    
    applied = 0
    skipped = 0
    
    for correction in corrections_list:
        idx = correction.get("segment_idx")
        if idx is None or idx >= len(doc.segments):
            skipped += 1
            continue
        
        # Only apply high/medium confidence corrections
        if correction.get("confidence") == "low":
            skipped += 1
            continue
        
        # Apply the correction
        doc.segments[idx].text = correction.get("suggested", doc.segments[idx].text)
        applied += 1
    
    # Save corrected transcript
    save_document(doc, output_path)
    
    return json.dumps({
        "success": True,
        "total_corrections": len(corrections_list),
        "applied": applied,
        "skipped": skipped,
        "output_path": output_path,
    }, indent=2)


@tool
def get_multilingual_medical_context(
    term_id: str,
    languages: list[str] | None = None,
) -> str:
    """
    Get multilingual context for a medical term.
    
    Args:
        term_id: ID of the medical term in glossary
        languages: List of language codes (default: en, es, fr, de, zh)
    
    Returns:
        JSON with multilingual term information
    """
    if languages is None:
        languages = ["en", "es", "fr", "de", "zh"]
    
    glossary = load_medical_glossary()
    term = glossary.get_term(term_id)
    
    if term is None:
        return json.dumps({"error": f"Term {term_id} not found"})
    
    context = {
        "term_id": term.term_id,
        "english": term.english,
        "category": term.category,
        "synonyms": term.synonyms,
        "abbreviations": term.abbreviations,
        "translations": {},
    }
    
    for lang in languages:
        translation = term.get_term(lang)
        definition = term.definitions.get(lang)
        if translation != term.english or lang == "en":
            context["translations"][lang] = {
                "term": translation,
                "definition": definition,
            }
    
    return json.dumps(context, indent=2, ensure_ascii=False)


# ============================================================================
# Agent State and Graph
# ============================================================================

class VerificationState(TypedDict):
    """State for the verification workflow."""
    transcript_path: str
    audio_path: str | None
    output_dir: str
    
    # Intermediate results
    errors: list[dict]
    medical_terms: list[dict]
    extracted_audio: list[dict]
    verification_results: list[dict]
    corrections: list[dict]
    
    # Final output
    corrected_transcript_path: str | None
    report: dict | None
    
    # Progress tracking
    stage: str
    error_count: int


def create_verification_tools():
    """Create all tools for the verification agent."""
    return [
        identify_transcription_errors,
        extract_audio_for_verification,
        verify_with_whisper,
        detect_medical_terms_in_transcript,
        suggest_transcription_corrections,
        apply_corrections_to_transcript,
        get_multilingual_medical_context,
    ]


def create_verification_agent():
    """Create the main transcription verification agent."""
    
    system_prompt = """You are an expert Transcription Verification Agent specialized in medical terminology.

Your mission is to:
1. Analyze transcriptions for errors (low confidence words, homophones, unknown terms)
2. Detect medical terms using a multilingual glossary
3. Extract relevant audio segments for verification
4. Verify using Faster-Whisper or other methods
5. Suggest and apply corrections
6. Generate a comprehensive report

Workflow:
1. First, identify_transcription_errors to find potential issues
2. Then, detect_medical_terms_in_transcript for medical terminology
3. For critical errors or medical terms, extract_audio_for_verification
4. Use verify_with_whisper to get alternative transcriptions
5. Use suggest_transcription_corrections to propose fixes
6. Finally, apply_corrections_to_transcript to save the corrected version

Always be thorough and document your reasoning at each step."""

    tools = create_verification_tools()
    
    if DEEP_AGENTS_AVAILABLE:
        # Create deep agent with filesystem backend
        agent = create_deep_agent(
            model="claude-sonnet-4-5-20250929",
            tools=tools,
            system_prompt=system_prompt,
            backend=FilesystemBackend(root_dir=".", virtual_mode=True),
            checkpointer=MemorySaver(),
        )
        return agent
    else:
        # Fallback: return tools for manual use
        return tools, system_prompt


# ============================================================================
# Main Entry Point
# ============================================================================

def run_verification(
    transcript_path: str,
    audio_path: str | None = None,
    output_dir: str = "./verification_output",
) -> dict[str, Any]:
    """
    Run the complete transcription verification pipeline.
    
    This is a simplified version that runs the steps sequentially.
    For full agent capabilities, use create_verification_agent().
    """
    print("=" * 60)
    print("Transcription Verification Agent")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Identify errors
    print("\n[Step 1] Identifying transcription errors...")
    errors_result = identify_transcription_errors.invoke({
        "transcript_path": transcript_path,
        "confidence_threshold": 0.75,
    })
    errors_data = json.loads(errors_result)
    print(f"  Found {errors_data['errors_found']} potential errors")
    
    # Step 2: Detect medical terms
    print("\n[Step 2] Detecting medical terms...")
    medical_result = detect_medical_terms_in_transcript.invoke({
        "transcript_path": transcript_path,
        "target_languages": ["en", "es", "fr"],
    })
    medical_data = json.loads(medical_result)
    print(f"  Found {medical_data['total_terms']} medical terms")
    
    # Step 3: Extract audio for high-priority segments
    print("\n[Step 3] Extracting audio segments...")
    
    # Prioritize: medical terms + low confidence errors
    priority_indices = []
    
    # Add segments with medical terms
    for term in medical_data.get("terms", []):
        idx = term.get("segment_idx")
        if idx is not None and idx not in priority_indices:
            priority_indices.append(idx)
    
    # Add segments with errors (first 5)
    for error in errors_data.get("errors", [])[:5]:
        idx = error.get("segment_idx")
        if idx is not None and idx not in priority_indices:
            priority_indices.append(idx)
    
    if audio_path and priority_indices:
        audio_result = extract_audio_for_verification.invoke({
            "transcript_path": transcript_path,
            "segment_indices": priority_indices[:10],  # Limit to 10
            "output_dir": str(output_dir / "audio_segments"),
            "padding": 1.0,
        })
        audio_data = json.loads(audio_result)
        print(f"  Extracted {audio_data['total_extracted']} audio segments")
    else:
        audio_data = {"segments": []}
        print("  Skipped (no audio path provided)")
    
    # Step 4: Verify with Whisper (if available)
    print("\n[Step 4] Verifying with Faster-Whisper...")
    
    verifications = []
    if FASTER_WHISPER_AVAILABLE and audio_data.get("segments"):
        for seg in audio_data["segments"]:
            if "error" in seg:
                continue
            
            verify_result = verify_with_whisper.invoke({
                "audio_path": seg["audio_path"],
                "model_size": "base",
                "language": None,
            })
            verify_data = json.loads(verify_result)
            
            if "error" not in verify_data:
                verifications.append({
                    "segment_idx": seg["segment_idx"],
                    "original_text": seg["original_text"],
                    "verified_text": verify_data["transcription"],
                    "is_correct": seg["original_text"].strip().lower() == verify_data["transcription"].strip().lower(),
                    "confidence": "high" if verify_data.get("confidence", 0) > 0.9 else "medium",
                    "method": "faster-whisper",
                })
        
        print(f"  Verified {len(verifications)} segments")
    else:
        print("  Skipped (Faster-Whisper not available)")
    
    # Step 5: Suggest corrections
    print("\n[Step 5] Suggesting corrections...")
    
    # Build verification results
    verification_input = {"results": verifications}
    
    corrections_result = suggest_transcription_corrections.invoke({
        "transcript_path": transcript_path,
        "verification_results": json.dumps(verification_input),
    })
    corrections_data = json.loads(corrections_result)
    print(f"  Generated {corrections_data['total_corrections']} correction suggestions")
    
    # Step 6: Apply corrections
    print("\n[Step 6] Applying corrections...")
    
    output_transcript = output_dir / "corrected_transcript.json"
    apply_result = apply_corrections_to_transcript.invoke({
        "transcript_path": transcript_path,
        "corrections": corrections_result,
        "output_path": str(output_transcript),
    })
    apply_data = json.loads(apply_result)
    print(f"  Applied {apply_data['applied']} corrections")
    
    # Generate final report
    report = {
        "input": {
            "transcript_path": transcript_path,
            "audio_path": audio_path,
        },
        "output": {
            "corrected_transcript": str(output_transcript),
            "output_dir": str(output_dir),
        },
        "summary": {
            "errors_found": errors_data['errors_found'],
            "medical_terms_found": medical_data['total_terms'],
            "segments_verified": len(verifications),
            "corrections_applied": apply_data['applied'],
        },
        "details": {
            "errors": errors_data,
            "medical_terms": medical_data,
            "verifications": verifications,
            "corrections": corrections_data,
        },
    }
    
    # Save report
    report_path = output_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    
    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)
    print(f"Corrected transcript: {output_transcript}")
    print(f"Full report: {report_path}")
    
    return report


def main():
    """Main entry point."""
    print("=" * 60)
    print("ASR Medical Verification - Stage 5: Deep Agent")
    print("=" * 60)
    
    # Check if we have a sample transcript from stage 1
    stage1_dir = Path(__file__).parent.parent / "stage_01_error_detection"
    sample_transcript = stage1_dir / "sample_transcript.json"
    
    if sample_transcript.exists():
        print(f"\nRunning verification on: {sample_transcript}")
        
        # Run the verification pipeline
        report = run_verification(
            transcript_path=str(sample_transcript),
            audio_path=None,  # No audio for sample
            output_dir="./verification_output",
        )
        
        print("\nFinal Report Summary:")
        print(f"  Errors found: {report['summary']['errors_found']}")
        print(f"  Medical terms: {report['summary']['medical_terms_found']}")
        print(f"  Corrections applied: {report['summary']['corrections_applied']}")
    else:
        print("\nNo sample transcript found.")
        print("Please run stage 01_error_detection first.")
        print("\nUsage:")
        print("  run_verification(transcript_path, audio_path, output_dir)")


if __name__ == "__main__":
    main()
