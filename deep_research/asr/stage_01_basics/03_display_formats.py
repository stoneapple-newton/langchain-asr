"""
ASR Stage 1, File 3: Display & Export Formats
===============================================
CONCEPT: Rendering a WhisperX transcript into different human-readable formats.

The raw WhisperX JSON is not meant to be read by humans. Before the AI
improvement pipeline runs (and after it runs), you need to render the
transcript into formats that are easy to review, compare, and share.

Formats implemented:
  1. Speaker-attributed text  — clean "Name: sentence" format for reading
  2. SRT subtitles            — standard subtitle file for video players
  3. VTT (WebVTT)             — browser-native subtitle format
  4. Markdown report          — structured summary with metadata
  5. Side-by-side diff        — compare two versions of the same transcript

These renderers are reused in every later stage to show before/after results.

Run this file:
  uv run deep_research/asr/stage_01_basics/03_display_formats.py
"""

import json
import re
from pathlib import Path
from datetime import timedelta

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)

segments = raw["segments"]


# ---------------------------------------------------------------------------
# Helper: seconds → timestamp strings
# ---------------------------------------------------------------------------

def to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_vtt_time(seconds: float) -> str:
    """Convert seconds to VTT timestamp: HH:MM:SS.mmm"""
    return to_srt_time(seconds).replace(",", ".")


def fmt_mmss(seconds: float) -> str:
    """Short MM:SS format for display."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# 1. Speaker-attributed transcript (most human-readable)
# ---------------------------------------------------------------------------

def render_speaker_transcript(
    segments: list[dict],
    speaker_names: dict[str, str] | None = None,
    merge_consecutive: bool = True,
) -> str:
    """
    Render as attributed dialogue, e.g.:
      SPEAKER_00 [00:00]: Alright everyone, let's get started.

    Args:
        speaker_names : optional mapping {"SPEAKER_00": "Alice"}
        merge_consecutive : merge adjacent segments from same speaker
    """
    names = speaker_names or {}
    lines = []
    merged = []

    # Merge consecutive same-speaker segments
    if merge_consecutive:
        for seg in segments:
            spk = seg.get("speaker") or "UNKNOWN"
            if merged and merged[-1]["speaker"] == spk:
                merged[-1]["text"] += " " + seg["text"].strip()
                merged[-1]["end"] = seg["end"]
            else:
                merged.append({
                    "speaker": spk,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                })
    else:
        merged = [
            {"speaker": seg.get("speaker") or "UNKNOWN",
             "start": seg["start"], "end": seg["end"],
             "text": seg["text"].strip()}
            for seg in segments
        ]

    for m in merged:
        label = names.get(m["speaker"], m["speaker"])
        lines.append(f"{label} [{fmt_mmss(m['start'])}]\n  {m['text']}\n")

    return "\n".join(lines)


print("=" * 60)
print("  FORMAT 1: Speaker-attributed transcript")
print("=" * 60)
print(render_speaker_transcript(segments[:8]))


# ---------------------------------------------------------------------------
# 2. SRT subtitle format
# ---------------------------------------------------------------------------

def render_srt(segments: list[dict], speaker_names: dict[str, str] | None = None) -> str:
    """
    Standard .srt format used by VLC, Premiere, DaVinci Resolve, etc.

    1
    00:00:00,210 --> 00:00:04,850
    SPEAKER_00: alright everyone lets get started
    """
    names = speaker_names or {}
    blocks = []
    for i, seg in enumerate(segments, 1):
        spk = seg.get("speaker") or "UNKNOWN"
        label = names.get(spk, spk)
        text = seg["text"].strip()
        block = (
            f"{i}\n"
            f"{to_srt_time(seg['start'])} --> {to_srt_time(seg['end'])}\n"
            f"{label}: {text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


print("=" * 60)
print("  FORMAT 2: SRT subtitle file (first 4 segments)")
print("=" * 60)
srt_output = render_srt(segments[:4])
print(srt_output)
print()

# Save to file
srt_path = TRANSCRIPT_PATH.parent / "sample_transcript.srt"
srt_path.write_text(render_srt(segments), encoding="utf-8")
print(f"  Saved: {srt_path.name}")
print()


# ---------------------------------------------------------------------------
# 3. WebVTT format
# ---------------------------------------------------------------------------

def render_vtt(segments: list[dict], speaker_names: dict[str, str] | None = None) -> str:
    """
    WebVTT (.vtt) — used by HTML5 <video>, YouTube, and most web players.
    """
    names = speaker_names or {}
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        spk = seg.get("speaker") or "UNKNOWN"
        label = names.get(spk, spk)
        text = seg["text"].strip()
        lines.append(f"{i}")
        lines.append(f"{to_vtt_time(seg['start'])} --> {to_vtt_time(seg['end'])}")
        lines.append(f"<v {label}>{text}</v>")
        lines.append("")
    return "\n".join(lines)


vtt_path = TRANSCRIPT_PATH.parent / "sample_transcript.vtt"
vtt_path.write_text(render_vtt(segments), encoding="utf-8")
print(f"  Saved: {vtt_path.name}")
print()


# ---------------------------------------------------------------------------
# 4. Markdown meeting report
# ---------------------------------------------------------------------------

def render_markdown_report(
    raw: dict,
    speaker_names: dict[str, str] | None = None,
) -> str:
    """
    Structured markdown doc with metadata, speaker stats, and full transcript.
    """
    names = speaker_names or {}
    segs = raw["segments"]
    duration = raw["duration"]
    speakers = sorted({s.get("speaker") for s in segs if s.get("speaker")})

    # Speaking time
    speaking_time: dict[str, float] = {}
    for s in segs:
        spk = s.get("speaker") or "UNKNOWN"
        speaking_time[spk] = speaking_time.get(spk, 0) + (s["end"] - s["start"])

    md = []
    meta = raw.get("meeting_metadata", {})
    md.append(f"# {meta.get('title', 'Meeting Transcript')}")
    md.append(f"**Date:** {meta.get('date', 'Unknown')}  ")
    md.append(f"**Duration:** {int(duration // 60)}m {int(duration % 60)}s  ")
    md.append(f"**Language:** {raw['language'].upper()}")
    md.append("")

    md.append("## Participants")
    for spk in speakers:
        label = names.get(spk, spk)
        t = speaking_time.get(spk, 0)
        pct = t / duration * 100
        md.append(f"- **{label}** — {t:.0f}s ({pct:.0f}% of meeting)")
    md.append("")

    md.append("## Transcript")
    prev_spk = None
    for seg in segs:
        spk = seg.get("speaker") or "UNKNOWN"
        label = names.get(spk, spk)
        if spk != prev_spk:
            md.append(f"\n**{label}** `[{fmt_mmss(seg['start'])}]`")
            prev_spk = spk
        md.append(seg["text"].strip())

    return "\n".join(md)


md_path = TRANSCRIPT_PATH.parent / "sample_transcript.md"
md_content = render_markdown_report(raw)
md_path.write_text(md_content, encoding="utf-8")

print("=" * 60)
print("  FORMAT 3: Markdown meeting report (excerpt)")
print("=" * 60)
print("\n".join(md_content.split("\n")[:30]))
print(f"\n  [... full report saved to {md_path.name}]")
print()


# ---------------------------------------------------------------------------
# 5. Side-by-side diff (before vs. after comparison)
# ---------------------------------------------------------------------------

def render_diff(
    original: list[dict],
    improved: list[dict],
    speaker_names: dict[str, str] | None = None,
    max_segments: int = 10,
) -> str:
    """
    Show original vs improved side by side for the first N segments.
    Used to validate that the improvement pipeline is working correctly.
    """
    names = speaker_names or {}
    lines = [f"{'ORIGINAL':<55}  {'IMPROVED':<55}", "─" * 112]
    for orig, impr in zip(original[:max_segments], improved[:max_segments]):
        spk = names.get(orig.get("speaker", ""), orig.get("speaker", "?"))
        o_text = f"[{spk}] {orig['text'].strip()}"
        i_text = f"[{spk}] {impr['text'].strip()}"
        # Wrap at 53 chars
        o_lines = [o_text[i:i+53] for i in range(0, max(len(o_text), 1), 53)]
        i_lines = [i_text[i:i+53] for i in range(0, max(len(i_text), 1), 53)]
        max_l = max(len(o_lines), len(i_lines))
        for j in range(max_l):
            o = o_lines[j] if j < len(o_lines) else ""
            i = i_lines[j] if j < len(i_lines) else ""
            changed = "◀" if o != i else " "
            lines.append(f"  {o:<53}  {changed}  {i:<53}")
        lines.append("")
    return "\n".join(lines)


# Simulate an "improved" version for demo purposes
import copy
improved_segments = copy.deepcopy(segments)
for seg in improved_segments:
    t = seg["text"].strip()
    # Capitalize first letter
    t = t[0].upper() + t[1:] if t else t
    # Remove leading fillers
    t = re.sub(r"^(uh|um|uh um|um uh)\s+", "", t, flags=re.IGNORECASE)
    # Add period at end if missing
    if t and t[-1] not in ".!?,":
        t += "."
    seg["text"] = " " + t

print("=" * 60)
print("  FORMAT 4: Before/after diff")
print("=" * 60)
print(render_diff(segments, improved_segments, max_segments=6))


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Render early and often — always inspect output visually, not just by score
# ✅ SRT/VTT let you validate diarization directly in a video player
# ✅ Markdown report gives stakeholders a shareable document
# ✅ render_diff() is essential — use it to verify every pipeline stage
# ✅ speaker_names dict is the bridge between SPEAKER_00 and real names
# ✅ merge_consecutive=True collapses same-speaker micro-segments into readable turns
