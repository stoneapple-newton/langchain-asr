"""
Stage 5: run the end-to-end ASR medical verification workflow.

Run:
  uv run deep_research/asr_medical_verification/stage_05_deep_agent/01_asr_medical_verification_agent.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_research.asr_medical_verification.workflow import run_asr_medical_verification


HERE = Path(__file__).resolve().parent
SAMPLE_TRANSCRIPT = HERE.parent / "data" / "sample_transcript.json"
SAMPLE_GLOSSARY = HERE.parent / "data" / "medical_glossary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASR medical verification workflow.")
    parser.add_argument("--transcript", default=str(SAMPLE_TRANSCRIPT))
    parser.add_argument("--audio", default="")
    parser.add_argument("--glossary", default=str(SAMPLE_GLOSSARY))
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "faster_whisper", "multimodal_llm"],
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.72)
    args = parser.parse_args()

    result = run_asr_medical_verification(
        args.transcript,
        audio_path=args.audio or None,
        glossary_path=args.glossary,
        retranscription_method=args.method,
        confidence_threshold=args.confidence_threshold,
    )
    summary = {
        "error_candidates": len(result.get("error_candidates", [])),
        "medical_candidates": len(result.get("medical_candidates", [])),
        "corrections": len(result.get("corrections", [])),
        "outputs": result.get("output_paths", []),
        "stage_log": result.get("stage_log", []),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
