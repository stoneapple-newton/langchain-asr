"""
Stage 3b: Multimodal LLM Verification
=====================================
Use a multimodal LLM to verify transcriptions from audio.
This can be used alongside or instead of Whisper re-transcription.

Note: This requires a multimodal LLM that supports audio input.
Examples: OpenAI GPT-4o-audio, Anthropic Claude with audio, etc.

Run this file:
  uv run deep_research/asr_medical_verification/stage_03_transcription_verification/02_multimodal_llm_verify.py
"""

from __future__ import annotations

import json
import base64
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from deep_research.asr_medical_verification.shared.medical_glossary import (
    load_medical_glossary,
    MultilingualMedicalGlossary,
)


def create_multimodal_llm_client() -> Callable[[list[dict]], str] | None:
    """
    Create a client function for multimodal LLM.
    
    Returns a function that takes messages and returns text response.
    Returns None if no suitable multimodal LLM is available.
    """
    # Try to create a chat model
    try:
        llm = create_chat_model(temperature=0)
        
        def client(messages: list[dict]) -> str:
            """Call LLM with messages."""
            try:
                response = llm.invoke(messages)
                return response.content
            except Exception as e:
                return f"Error: {e}"
        
        return client
    except Exception as e:
        print(f"Could not initialize multimodal LLM: {e}")
        return None


def verify_with_llm(
    audio_segment_path: str | Path,
    original_text: str,
    llm_client: Callable[[list[dict]], str],
    context: str | None = None,
) -> dict[str, Any]:
    """
    Verify a transcription using multimodal LLM.
    
    Args:
        audio_segment_path: Path to audio file
        original_text: Original transcription to verify
        llm_client: Function to call LLM
        context: Optional context about the audio
    
    Returns:
        Verification result with assessment
    """
    audio_segment_path = Path(audio_segment_path)
    
    # Read audio file
    audio_data = audio_segment_path.read_bytes()
    audio_b64 = base64.b64encode(audio_data).decode()
    
    # Build messages
    system_msg = (
        "You are an expert at verifying audio transcriptions. "
        "Listen to the audio carefully and compare it to the provided text. "
        "Determine if the transcription is accurate. "
        "If there are errors, provide the correct transcription. "
        "Respond in JSON format with: is_correct (boolean), corrected_text (string), explanation (string)"
    )
    
    user_content = [
        {
            "type": "text",
            "text": f"Original transcription: '{original_text}'\n\nIs this transcription correct? If not, what is the correct text?",
        },
        {
            "type": "audio",
            "source": {
                "type": "base64",
                "media_type": "audio/wav",
                "data": audio_b64,
            },
        },
    ]
    
    if context:
        user_content[0]["text"] += f"\n\nContext: {context}"
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    
    # Call LLM
    response = llm_client(messages)
    
    # Try to parse JSON response
    try:
        result = json.loads(response)
        return {
            "is_correct": result.get("is_correct", False),
            "corrected_text": result.get("corrected_text", original_text),
            "explanation": result.get("explanation", ""),
            "original_text": original_text,
            "source": "multimodal_llm",
        }
    except json.JSONDecodeError:
        # Parse as plain text
        is_correct = "correct" in response.lower() and "incorrect" not in response.lower()
        return {
            "is_correct": is_correct,
            "corrected_text": original_text if is_correct else response,
            "explanation": response,
            "original_text": original_text,
            "source": "multimodal_llm",
            "raw_response": response,
        }


def verify_medical_term_with_llm(
    audio_segment_path: str | Path,
    expected_term: str,
    glossary: MultilingualMedicalGlossary,
    llm_client: Callable[[list[dict]], str],
    language: str = "en",
) -> dict[str, Any]:
    """
    Verify a medical term specifically, using glossary context.
    
    Args:
        audio_segment_path: Path to audio file
        expected_term: The medical term expected
        glossary: Medical glossary for context
        llm_client: Function to call LLM
        language: Language code
    
    Returns:
        Verification result
    """
    audio_segment_path = Path(audio_segment_path)
    
    # Get term info from glossary
    term = None
    for t in glossary.terms.values():
        if t.english.lower() == expected_term.lower() or \
           any(syn.lower() == expected_term.lower() for syn in t.synonyms):
            term = t
            break
    
    # Build context from glossary
    context_parts = ["Medical terminology verification."]
    if term:
        context_parts.append(f"Expected term: {term.english}")
        context_parts.append(f"Category: {term.category}")
        if term.translations:
            lang_term = term.get_term(language)
            if lang_term != term.english:
                context_parts.append(f"In target language: {lang_term}")
        if term.synonyms:
            context_parts.append(f"Also known as: {', '.join(term.synonyms[:3])}")
    
    context = " ".join(context_parts)
    
    # Read audio
    audio_data = audio_segment_path.read_bytes()
    audio_b64 = base64.b64encode(audio_data).decode()
    
    # Build prompt
    system_msg = (
        "You are a medical transcription specialist. "
        "Listen to the audio and identify the medical term being spoken. "
        "Compare it to the expected term and provide the correct term if different. "
        "Respond in JSON format: {\"identified_term\": string, \"confidence\": \"high|medium|low\", \"explanation\": string}"
    )
    
    user_content = [
        {
            "type": "text",
            "text": f"Expected medical term: '{expected_term}'\n\n{context}\n\nWhat medical term do you hear in the audio?",
        },
        {
            "type": "audio",
            "source": {
                "type": "base64",
                "media_type": "audio/wav",
                "data": audio_b64,
            },
        },
    ]
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    
    # Call LLM
    response = llm_client(messages)
    
    # Parse response
    try:
        result = json.loads(response)
        identified = result.get("identified_term", expected_term)
        return {
            "expected_term": expected_term,
            "identified_term": identified,
            "is_match": identified.lower() == expected_term.lower(),
            "confidence": result.get("confidence", "low"),
            "explanation": result.get("explanation", ""),
            "source": "multimodal_llm_medical",
        }
    except json.JSONDecodeError:
        is_match = expected_term.lower() in response.lower()
        return {
            "expected_term": expected_term,
            "identified_term": response,
            "is_match": is_match,
            "confidence": "unknown",
            "explanation": response,
            "source": "multimodal_llm_medical",
            "raw_response": response,
        }


def verify_all_segments_with_llm(
    extraction_report_path: str | Path,
    llm_client: Callable[[list[dict]], str] | None = None,
) -> dict[str, Any]:
    """
    Verify all extracted segments using multimodal LLM.
    
    Args:
        extraction_report_path: Path to extraction report
        llm_client: Optional LLM client (if None, tries to create one)
    
    Returns:
        Updated report with LLM verifications
    """
    if llm_client is None:
        llm_client = create_multimodal_llm_client()
    
    if llm_client is None:
        print("No multimodal LLM available for verification.")
        return {"error": "No multimodal LLM available"}
    
    extraction_report_path = Path(extraction_report_path)
    report = json.loads(extraction_report_path.read_text())
    
    print("=" * 60)
    print("Multimodal LLM Verification")
    print("=" * 60)
    
    # Verify error segments
    audio_extraction = report.get("audio_extraction", {})
    segments = audio_extraction.get("segments", [])
    
    llm_verifications = []
    
    for i, segment_info in enumerate(segments):
        if "segment_path" not in segment_info:
            continue
        
        segment_path = Path(segment_info["segment_path"])
        if not segment_path.exists():
            continue
        
        original_word = segment_info["word"]
        
        print(f"[{i+1}/{len(segments)}] Verifying: '{original_word}'")
        
        try:
            result = verify_with_llm(
                segment_path,
                original_word,
                llm_client,
            )
            
            llm_verifications.append({
                "candidate_index": segment_info.get("candidate_index"),
                "original_word": original_word,
                "error_type": segment_info["error_type"],
                **result,
            })
            
            if not result["is_correct"]:
                print(f"    Correction: '{result.get('corrected_text', original_word)}'")
            else:
                print(f"    Verified correct")
                
        except Exception as e:
            print(f"    Error: {e}")
    
    # Verify medical term segments
    medical_verifications = []
    medical_extraction = report.get("medical_term_extraction", {})
    glossary = load_medical_glossary()
    
    if medical_extraction.get("success"):
        medical_segments = medical_extraction.get("segments", [])
        print(f"\nVerifying {len(medical_segments)} medical terms...")
        
        for i, segment_info in enumerate(medical_segments):
            segment_path = Path(segment_info["segment_path"])
            if not segment_path.exists():
                continue
            
            expected_term = segment_info["english"]
            
            print(f"[Medical {i+1}/{len(medical_segments)}] {expected_term}")
            
            try:
                result = verify_medical_term_with_llm(
                    segment_path,
                    expected_term,
                    glossary,
                    llm_client,
                )
                
                medical_verifications.append({
                    "term_index": segment_info["term_index"],
                    "term_id": segment_info["term_id"],
                    **result,
                })
                
                print(f"    Identified: '{result['identified_term']}' (match: {result['is_match']})")
                
            except Exception as e:
                print(f"    Error: {e}")
    
    # Update report
    report["llm_verification"] = {
        "total_verified": len(llm_verifications),
        "correct_count": len([v for v in llm_verifications if v.get("is_correct")]),
        "incorrect_count": len([v for v in llm_verifications if not v.get("is_correct")]),
        "verifications": llm_verifications,
        "medical_verifications": medical_verifications,
    }
    
    print("\n" + "=" * 60)
    print("LLM Verification Summary")
    print("=" * 60)
    print(f"Total verified: {report['llm_verification']['total_verified']}")
    print(f"Correct: {report['llm_verification']['correct_count']}")
    print(f"Incorrect: {report['llm_verification']['incorrect_count']}")
    
    return report


def main():
    """Main entry point."""
    print("=" * 60)
    print("ASR Medical Verification - Stage 3b: Multimodal LLM Verification")
    print("=" * 60)
    
    # Check if we have a multimodal LLM available
    llm_client = create_multimodal_llm_client()
    
    if llm_client is None:
        print("\nNo multimodal LLM available.")
        print("This stage requires a multimodal LLM that supports audio input.")
        print("\nSupported models:")
        print("  - OpenAI GPT-4o-audio")
        print("  - Anthropic Claude (with audio support)")
        print("\nTo use this feature:")
        print("  1. Configure your LLM provider in .env")
        print("  2. Ensure your API key is set")
        print("  3. Run this script again")
        return
    
    print("\nMultimodal LLM client initialized successfully.")
    
    # Load extraction report
    stage2_dir = Path(__file__).parent.parent / "stage_02_audio_extraction"
    extraction_report_path = stage2_dir / "extraction_report.json"
    
    if not extraction_report_path.exists():
        print(f"\nExtraction report not found: {extraction_report_path}")
        print("Please run stage 02_audio_extraction first.")
        return
    
    # Run verification
    report = verify_all_segments_with_llm(
        extraction_report_path=extraction_report_path,
        llm_client=llm_client,
    )
    
    # Save report
    output_path = Path(__file__).parent / "llm_verification_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nLLM verification report saved to: {output_path}")


if __name__ == "__main__":
    main()
