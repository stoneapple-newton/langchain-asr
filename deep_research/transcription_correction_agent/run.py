"""
Transcription Correction Agent — Entry Point
=============================================
Usage:
  uv run deep_research/transcription_correction_agent/run.py

  # With a real audio file:
  uv run deep_research/transcription_correction_agent/run.py \\
      --transcript path/to/transcript.json \\
      --audio     path/to/audio.mp3 \\
      --method    auto

Arguments:
  --transcript   Path to WhisperX-style JSON transcript (default: sample)
  --audio        Path to source audio file (optional; enables audio retranscription)
  --method       Retranscription method: faster_whisper | multimodal_llm | auto
                 (default: auto — tries faster-whisper first, falls back to LLM)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Sample data (used when no --transcript is supplied)
# ---------------------------------------------------------------------------
_SAMPLE_TRANSCRIPT = _HERE / "data" / "sample_transcript.json"
_SAMPLE_AUDIO = ""  # no audio — will use text-only fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Transcription Correction Agent"
    )
    parser.add_argument(
        "--transcript",
        default=str(_SAMPLE_TRANSCRIPT),
        help="Path to the input transcript JSON",
    )
    parser.add_argument(
        "--audio",
        default=_SAMPLE_AUDIO,
        help="Path to source audio file (enables audio retranscription)",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["faster_whisper", "multimodal_llm", "auto"],
        help="Retranscription method (default: auto)",
    )
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        # Create sample transcript if it doesn't exist
        _create_sample_transcript(transcript_path)

    from agent import correction_graph  # noqa: PLC0415

    print("=" * 64)
    print("  TRANSCRIPTION CORRECTION AGENT")
    print("=" * 64)
    print(f"  Transcript : {transcript_path}")
    print(f"  Audio      : {args.audio or '(none — text-only correction)'}")
    print(f"  Method     : {args.method}")
    print()

    initial_state = {
        "transcript_path": str(transcript_path),
        "audio_path": args.audio,
        "retranscription_method": args.method,
        "segments": [],
        "medical_glossary": {},
        "error_candidates": [],
        "medical_candidates": [],
        "corrections": [],
        "corrected_segments": [],
        "stage_log": [],
        "output_paths": [],
    }

    final = correction_graph.invoke(initial_state)

    print()
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)
    corrections = final.get("corrections") or []
    changed = [r for r in corrections if r["corrected_text"] != r["original_text"]]
    medical = [r for r in corrections if r["correction_type"] == "medical_term"]
    terms: list[str] = []
    for r in medical:
        terms.extend(r.get("medical_terms", []))

    print(f"  Segments processed : {len(final.get('segments', []))}")
    print(f"  Error corrections  : {len([r for r in changed if r['correction_type'] == 'error'])}")
    print(f"  Medical corrections: {len([r for r in changed if r['correction_type'] == 'medical_term'])}")
    print(f"  Medical terms found: {list(dict.fromkeys(terms))}")
    print()
    print("  Stage log:")
    for entry in final.get("stage_log", []):
        print(f"    • {entry}")
    print()
    print("  Output files:")
    for p in final.get("output_paths", []):
        print(f"    • {p}")


# ---------------------------------------------------------------------------
# Sample transcript generator (for demo when no real data is provided)
# ---------------------------------------------------------------------------

def _create_sample_transcript(path: Path) -> None:
    """Write a synthetic medical transcript for demonstration."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "duration": 92.5,
        "language": "en",
        "segments": [
            {
                "id": "0", "start": 0.0, "end": 4.2,
                "speaker": "DOCTOR",
                "text": "Good morning, let's review the patient's cardiac situation.",
                "words": []
            },
            {
                "id": "1", "start": 4.5, "end": 9.8,
                "speaker": "DOCTOR",
                "text": "The ECG showed signs of a trial fibrilation, which is concerning.",
                "words": []
            },
            {
                "id": "2", "start": 10.1, "end": 15.6,
                "speaker": "NURSE",
                "text": "Yes, and his blood pressure has been elevated consistently, around 160 over 95.",
                "words": []
            },
            {
                "id": "3", "start": 16.0, "end": 22.3,
                "speaker": "DOCTOR",
                "text": "We should consider starting him on lissanopril for the hyper tension.",
                "words": []
            },
            {
                "id": "4", "start": 22.8, "end": 28.1,
                "speaker": "NURSE",
                "text": "He's already on atorvastaten and metphormin for the diabetes.",
                "words": []
            },
            {
                "id": "5", "start": 28.5, "end": 34.7,
                "speaker": "DOCTOR",
                "text": "Right. The echo cardio gram from last week showed reduced ejection fraction.",
                "words": []
            },
            {
                "id": "6", "start": 35.0, "end": 41.2,
                "speaker": "DOCTOR",
                "text": "I'm also worried about the the the signs of early encephalo pathy.",
                "words": []
            },
            {
                "id": "7", "start": 41.5, "end": 47.8,
                "speaker": "NURSE",
                "text": "Should we order a bronco scopy for the persistent cough as well?",
                "words": []
            },
            {
                "id": "8", "start": 48.2, "end": 55.0,
                "speaker": "DOCTOR",
                "text": "Yes, and let's check the creatinine levels to rule out nephro pathy.",
                "words": []
            },
            {
                "id": "9", "start": 55.3, "end": 61.6,
                "speaker": "NURSE",
                "text": "The oncology team flagged possible meta stasis in the mediastinum.",
                "words": []
            },
            {
                "id": "10", "start": 62.0, "end": 68.4,
                "speaker": "DOCTOR",
                "text": "That's serious. We'll need a biopsy. Also start azithromycin for the pnuemonia.",
                "words": []
            },
            {
                "id": "11", "start": 68.8, "end": 75.1,
                "speaker": "NURSE",
                "text": "I'll coordinate with pharmacy for the new medications.",
                "words": []
            },
            {
                "id": "12", "start": 75.5, "end": 82.0,
                "speaker": "DOCTOR",
                "text": "Great. We also need to monitor for tacky cardia given his current status.",
                "words": []
            },
            {
                "id": "13", "start": 82.3, "end": 92.5,
                "speaker": "NURSE",
                "text": "Understood. I'll update the chart and schedule the necessary referrals.",
                "words": []
            },
        ]
    }
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"  [run] Created sample transcript at {path}")


if __name__ == "__main__":
    main()
