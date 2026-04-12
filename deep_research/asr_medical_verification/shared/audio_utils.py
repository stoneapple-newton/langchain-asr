"""
Audio extraction helpers for ASR verification workflows.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioSegment:
    audio_data: bytes
    format: str
    start_time: float
    end_time: float
    sample_rate: int = 16000
    channels: int = 1

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.write_bytes(self.audio_data)
        return output_path

    def to_temp_file(self, suffix: str = ".wav") -> Path:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(self.audio_data)
            return Path(handle.name)


class AudioSegmentExtractor:
    """Extract time-bounded clips from an audio file."""

    def __init__(self, audio_path: str | Path):
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        self._duration: float | None = None

    def get_duration(self) -> float:
        if self._duration is not None:
            return self._duration
        if shutil.which("ffprobe") is None:
            raise RuntimeError("ffprobe not found. Install ffmpeg to inspect audio duration.")
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(self.audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self._duration = float(result.stdout.strip())
        return self._duration

    def extract_segment(
        self,
        start_time: float,
        end_time: float,
        *,
        output_format: str = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
        padding: float = 0.3,
    ) -> AudioSegment:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found. Install ffmpeg to extract audio segments.")

        actual_start = max(0.0, start_time - padding)
        actual_end = max(actual_start, end_time + padding)
        duration = actual_end - actual_start
        if duration <= 0:
            duration = max(end_time - start_time, 0.1)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{actual_start:.3f}",
            "-i",
            str(self.audio_path),
            "-t",
            f"{duration:.3f}",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
        ]
        if output_format == "wav":
            cmd.extend(["-f", "wav", "-acodec", "pcm_s16le"])
        elif output_format == "mp3":
            cmd.extend(["-f", "mp3", "-q:a", "2"])
        elif output_format == "flac":
            cmd.extend(["-f", "flac"])
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        cmd.append("-")

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return AudioSegment(
            audio_data=result.stdout,
            format=output_format,
            start_time=actual_start,
            end_time=actual_end,
            sample_rate=sample_rate,
            channels=channels,
        )

    def extract_for_transcription(
        self,
        start_time: float,
        end_time: float,
        *,
        context_window: float = 1.0,
    ) -> AudioSegment:
        return self.extract_segment(
            start_time,
            end_time,
            output_format="wav",
            sample_rate=16000,
            channels=1,
            padding=context_window / 2,
        )


def get_audio_duration(audio_path: str | Path) -> float:
    return AudioSegmentExtractor(audio_path).get_duration()


def extract_audio_segment(
    audio_path: str | Path,
    start_time: float,
    end_time: float,
    *,
    output_path: str | Path | None = None,
    padding: float = 0.3,
) -> Path:
    segment = AudioSegmentExtractor(audio_path).extract_for_transcription(
        start_time,
        end_time,
        context_window=padding * 2,
    )
    if output_path is None:
        return segment.to_temp_file(".wav")
    return segment.save(output_path)
