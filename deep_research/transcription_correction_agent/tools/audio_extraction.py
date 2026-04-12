"""
Audio Extraction Tool
=====================
Extracts a time-bounded audio segment from a source audio file.

Uses ffmpeg via subprocess — no extra Python packages required.
Falls back to pydub if available.

Produces a WAV file (16 kHz, mono) suitable for faster-whisper or
base64-encoding for multimodal LLMs.
"""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from pathlib import Path


def extract_segment(
    audio_path: str | Path,
    start: float,
    end: float,
    padding_seconds: float = 0.3,
    output_dir: str | Path | None = None,
) -> str:
    """
    Extract [start, end] from *audio_path* and return the path to a
    temporary WAV file (16 kHz, mono).

    Args:
        audio_path: Source audio file (any format ffmpeg understands).
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        padding_seconds: Extra context to include before/after the segment.
        output_dir: Where to write the temp file. Uses system temp if None.

    Returns:
        Absolute path to the extracted WAV file.

    Raises:
        FileNotFoundError: If ffmpeg is not installed.
        RuntimeError: If extraction fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    t_start = max(0.0, start - padding_seconds)
    duration = (end + padding_seconds) - t_start

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    seg_name = f"seg_{start:.3f}_{end:.3f}.wav".replace(".", "_")
    out_path = output_dir / seg_name

    if _ffmpeg_available():
        _extract_with_ffmpeg(audio_path, t_start, duration, out_path)
    else:
        _extract_with_pydub(audio_path, t_start, duration, out_path)

    return str(out_path)


# ---------------------------------------------------------------------------
# ffmpeg backend (preferred)
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_with_ffmpeg(
    src: Path,
    start: float,
    duration: float,
    dest: Path,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",                     # overwrite
        "-ss", f"{start:.3f}",    # seek before input (fast)
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-ac", "1",               # mono
        "-ar", "16000",           # 16 kHz for ASR
        "-f", "wav",
        str(dest),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# pydub fallback (requires: pip install pydub)
# ---------------------------------------------------------------------------

def _extract_with_pydub(
    src: Path,
    start: float,
    duration: float,
    dest: Path,
) -> None:
    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Neither ffmpeg nor pydub is available. "
            "Install ffmpeg (https://ffmpeg.org) or run: pip install pydub"
        ) from exc

    audio = AudioSegment.from_file(str(src))
    audio = audio.set_channels(1).set_frame_rate(16000)
    segment = audio[int(start * 1000): int((start + duration) * 1000)]
    segment.export(str(dest), format="wav")


# ---------------------------------------------------------------------------
# Convenience: read extracted WAV as base64 (for multimodal LLM)
# ---------------------------------------------------------------------------

def wav_to_base64(wav_path: str | Path) -> str:
    """Return base64-encoded WAV content as a string."""
    import base64
    return base64.b64encode(Path(wav_path).read_bytes()).decode("utf-8")
