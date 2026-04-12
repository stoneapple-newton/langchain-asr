"""
Stage 4: Medical Term Detection
===============================
Identify potential medical terms based on context and verify them.
This includes:
1. Detecting medical terms in transcription
2. Cross-referencing with multilingual glossary
3. Verifying and extracting audio for medical terms
4. Applying corrections based on verification

Run this file:
  uv run deep_research/asr_medical_verification/stage_04_medical_terms/01_medical_term_detection.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.asr_medical_verification.shared.transcript_utils import (
    load_transcript,
    TranscriptDocument,
    TranscriptSegment,
)
from deep_research.asr_medical_verification.shared.medical_glossary import (
    load_medical_glossary,
    MultilingualMedicalGlossary,
    MedicalTerm,
    detect_medical_terms,
)


def detect_potential_medical_terms_in_context(
    doc: TranscriptDocument,
    glossary: MultilingualMedicalGlossary | None = None,
) -> list[dict[str, Any]]:
    """
    Detect potential medical terms using context clues.
    
    This looks for:
    - Known medical terms from glossary
    - Words with medical-like patterns
    - Terms that appear in medical contexts
    """
    if glossary is None:
        glossary = load_medical_glossary()
    
    potential_terms = []
    
    # Medical context indicators
    context_indicators = [
        "patient", "doctor", "physician", "nurse", "hospital", "clinic",
        "diagnosis", "treatment", "symptom", "prescription", "medication",
        "surgery", "procedure", "exam", "test", "result", "medical",
        "health", "disease", "condition", "pain", "blood", "heart",
        "lung", "brain", "kidney", "liver", "aspirin", "insulin",
    ]
    
    # Medical-like word patterns
    medical_patterns = [
        r"\w+itis$",  # Inflammation (arthritis, bronchitis)
        r"\w+osis$",  # Condition (fibrosis, cirrhosis)
        r"\w+emia$",  # Blood condition (anemia, leukemia)
        r"\w+oma$",   # Tumor (carcinoma, lymphoma)
        r"\w+ectomy$", # Surgical removal (appendectomy)
        r"\w+scopy$", # Visual exam (endoscopy)
        r"\w+gram$",  # Recording (electrocardiogram)
    ]
    
    for seg_idx, segment in enumerate(doc.segments):
        text = segment.text.lower()
        
        # Check for context indicators
        has_medical_context = any(indicator in text for indicator in context_indicators)
        
        if not has_medical_context:
            continue
        
        # Look for words matching medical patterns
        words = text.split()
        for w_idx, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word)
            
            # Check against patterns
            for pattern in medical_patterns:
                if re.search(pattern, clean_word, re.IGNORECASE):
                    # Get context
                    left = " ".join(words[max(0, w_idx-3):w_idx])
                    right = " ".join(words[w_idx+1:w_idx+4])
                    
                    potential_terms.append({
                        "segment_idx": seg_idx,
                        "word_idx": w_idx,
                        "word": clean_word,
                        "pattern_matched": pattern,
                        "left_context": left,
                        "right_context": right,
                        "segment_text": segment.text,
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "detection_method": "pattern",
                    })
                    break
    
    return potential_terms


def cross_reference_with_glossary(
    doc: TranscriptDocument,
    glossary: MultilingualMedicalGlossary | None = None,
    language: str = "en",
) -> list[dict[str, Any]]:
    """
    Cross-reference transcript with multilingual medical glossary.
    
    Returns matches found in the transcript with their glossary entries.
    """
    if glossary is None:
        glossary = load_medical_glossary()
    
    # Combine all text
    full_text = " ".join(s.text for s in doc.segments)
    
    # Detect terms
    matches = detect_medical_terms(full_text, glossary)
    
    # Enrich with segment locations
    enriched_matches = []
    
    for matched_text, term in matches:
        # Find which segment(s) contain this term
        for seg_idx, segment in enumerate(doc.segments):
            if matched_text.lower() in segment.text.lower():
                enriched_matches.append({
                    "matched_text": matched_text,
                    "term_id": term.term_id,
                    "english": term.english,
                    "category": term.category,
                    "translations": term.translations,
                    "synonyms": term.synonyms,
                    "segment_idx": seg_idx,
                    "segment_text": segment.text,
                    "start_time": segment.start,
                    "end_time": segment.end,
                    "speaker": segment.speaker,
                })
    
    return enriched_matches


def find_similar_medical_terms(
    word: str,
    glossary: MultilingualMedicalGlossary | None = None,
    max_results: int = 5,
) -> list[tuple[MedicalTerm, float]]:
    """
    Find medical terms similar to a given word.
    
    Uses simple string similarity to find potential corrections.
    """
    if glossary is None:
        glossary = load_medical_glossary()
    
    word_lower = word.lower()
    similarities = []
    
    for term in glossary.terms.values():
        for form in term.get_all_forms():
            form_lower = form.lower()
            
            # Calculate simple similarity
            if word_lower == form_lower:
                similarity = 1.0
            elif word_lower in form_lower or form_lower in word_lower:
                similarity = 0.8
            else:
                # Count common characters
                common = set(word_lower) & set(form_lower)
                similarity = len(common) / max(len(set(word_lower)), len(set(form_lower)))
            
            similarities.append((term, similarity))
    
    # Sort by similarity and return top results
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Remove duplicates (same term)
    seen_terms = set()
    unique = []
    for term, sim in similarities:
        if term.term_id not in seen_terms and sim > 0.5:
            seen_terms.add(term.term_id)
            unique.append((term, sim))
            if len(unique) >= max_results:
                break
    
    return unique


def suggest_medical_corrections(
    doc: TranscriptDocument,
    glossary: MultilingualMedicalGlossary | None = None,
) -> list[dict[str, Any]]:
    """
    Suggest corrections for potential medical term errors.
    
    This combines:
    1. Low confidence words that might be medical terms
    2. Unknown terms that match medical patterns
    3. Cross-referencing with glossary for similar terms
    """
    if glossary is None:
        glossary = load_medical_glossary()
    
    corrections = []
    
    # Get potential medical terms from context
    potential = detect_potential_medical_terms_in_context(doc, glossary)
    
    # Get detected glossary terms
    detected = cross_reference_with_glossary(doc, glossary)
    
    # Find low-confidence words in medical contexts
    from deep_research.asr_medical_verification.shared.transcript_utils import (
        find_low_confidence_regions,
    )
    
    low_conf = find_low_confidence_regions(doc, threshold=0.75)
    
    # Check if low-confidence words might be medical terms
    for candidate in low_conf:
        # Find similar medical terms
        similar = find_similar_medical_terms(candidate.word, glossary)
        
        if similar:
            corrections.append({
                "type": "potential_medical_term",
                "original_word": candidate.word,
                "score": candidate.score,
                "context": f"{candidate.left_context} [{candidate.word}] {candidate.right_context}",
                "segment_idx": candidate.segment_idx,
                "start_time": candidate.start_time,
                "end_time": candidate.end_time,
                "suggested_corrections": [
                    {
                        "term": term.english,
                        "similarity": round(sim, 3),
                        "category": term.category,
                    }
                    for term, sim in similar[:3]
                ],
            })
    
    # Add detected medical terms that might need verification
    for match in detected:
        corrections.append({
            "type": "detected_medical_term",
            "matched_text": match["matched_text"],
            "term_id": match["term_id"],
            "english": match["english"],
            "category": match["category"],
            "segment_idx": match["segment_idx"],
            "start_time": match["start_time"],
            "end_time": match["end_time"],
            "needs_verification": match["matched_text"].lower() != match["english"].lower(),
        })
    
    return corrections


def generate_multilingual_medical_context(
    term_id: str,
    glossary: MultilingualMedicalGlossary | None = None,
    target_languages: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate context about a medical term in multiple languages.
    
    This helps LLMs understand the term across different languages.
    """
    if glossary is None:
        glossary = load_medical_glossary()
    
    term = glossary.get_term(term_id)
    if term is None:
        return {"error": f"Term {term_id} not found"}
    
    if target_languages is None:
        target_languages = ["en", "es", "fr", "de", "zh"]
    
    context = {
        "term_id": term.term_id,
        "primary_term": term.english,
        "category": term.category,
        "translations": {},
        "synonyms": term.synonyms,
        "abbreviations": term.abbreviations,
    }
    
    for lang in target_languages:
        translation = term.get_term(lang)
        if translation != term.english or lang == "en":
            context["translations"][lang] = {
                "term": translation,
                "definition": term.definitions.get(lang),
            }
    
    # Build a prompt-friendly context string
    context_lines = [
        f"Medical Term: {term.english}",
        f"Category: {term.category}",
    ]
    
    if term.synonyms:
        context_lines.append(f"Also known as: {', '.join(term.synonyms)}")
    
    if term.abbreviations:
        context_lines.append(f"Abbreviations: {', '.join(term.abbreviations)}")
    
    if term.translations:
        context_lines.append("Translations:")
        for lang, trans in term.translations.items():
            context_lines.append(f"  {lang}: {trans}")
    
    context["context_string"] = "\n".join(context_lines)
    
    return context


def process_medical_terms(
    transcript_path: str | Path,
    audio_path: str | Path | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """
    Main function to process and detect medical terms in a transcript.
    
    Returns comprehensive report with all detected terms and suggestions.
    """
    print("=" * 60)
    print("Medical Term Detection")
    print("=" * 60)
    
    # Load transcript
    doc = load_transcript(transcript_path, audio_path)
    print(f"Loaded transcript: {len(doc.segments)} segments")
    
    # Load glossary
    glossary = load_medical_glossary()
    print(f"Loaded glossary: {len(glossary.terms)} terms")
    
    # 1. Detect potential medical terms
    print("\n1. Detecting potential medical terms...")
    potential = detect_potential_medical_terms_in_context(doc, glossary)
    print(f"   Found {len(potential)} potential terms by pattern matching")
    
    # 2. Cross-reference with glossary
    print("\n2. Cross-referencing with medical glossary...")
    detected = cross_reference_with_glossary(doc, glossary, language)
    print(f"   Found {len(detected)} matching glossary terms")
    
    # 3. Suggest corrections
    print("\n3. Suggesting medical term corrections...")
    corrections = suggest_medical_corrections(doc, glossary)
    print(f"   Generated {len(corrections)} correction suggestions")
    
    # 4. Generate multilingual context for detected terms
    print("\n4. Generating multilingual context...")
    multilingual_contexts = {}
    for match in detected:
        term_id = match["term_id"]
        if term_id not in multilingual_contexts:
            multilingual_contexts[term_id] = generate_multilingual_medical_context(
                term_id, glossary
            )
    print(f"   Generated contexts for {len(multilingual_contexts)} unique terms")
    
    # Compile report
    report = {
        "transcript_path": str(transcript_path),
        "audio_path": doc.audio_path,
        "language": language,
        "segment_count": len(doc.segments),
        "detection_results": {
            "potential_terms": potential,
            "detected_glossary_terms": detected,
            "correction_suggestions": corrections,
        },
        "multilingual_contexts": multilingual_contexts,
        "summary": {
            "total_potential_terms": len(potential),
            "total_detected_terms": len(detected),
            "total_correction_suggestions": len(corrections),
            "categories_found": list(set(m["category"] for m in detected)),
        },
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Potential medical terms: {len(potential)}")
    print(f"Detected glossary terms: {len(detected)}")
    print(f"Correction suggestions: {len(corrections)}")
    print(f"Categories: {', '.join(report['summary']['categories_found'])}")
    
    return report


def main():
    """Main entry point."""
    print("=" * 60)
    print("ASR Medical Verification - Stage 4: Medical Term Detection")
    print("=" * 60)
    
    # Try to load a transcript
    # First check if we have a sample from stage 1
    stage1_dir = Path(__file__).parent.parent / "stage_01_error_detection"
    sample_transcript = stage1_dir / "sample_transcript.json"
    
    if sample_transcript.exists():
        report = process_medical_terms(
            transcript_path=sample_transcript,
            language="en",
        )
        
        # Save report
        output_path = Path(__file__).parent / "medical_terms_report.json"
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nMedical terms report saved to: {output_path}")
    else:
        print("\nNo sample transcript found.")
        print("Please run stage 01_error_detection first.")


if __name__ == "__main__":
    main()
