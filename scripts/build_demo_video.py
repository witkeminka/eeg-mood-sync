#!/usr/bin/env python3
"""Build a ~45s portfolio demo MP4 from screenshots + WAV."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
WAV = ROOT / "outputs" / "sub01_track.wav"
OUT = ROOT / "docs" / "demo.mp4"
DEMO_WAV = ROOT / "docs" / "demo.wav"
DURATION = 45  # seconds


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    if not WAV.exists():
        raise SystemExit(f"Missing {WAV}. Run openneuro with --wav first.")

    slides = [
        (SHOTS / "band_landscape.png", 16, "EEG band landscape"),
        (SHOTS / "band_landscape_mood.png", 16, "Music parameters"),
        (SHOTS / "confusion_matrix.png", 13, "ML task classifier"),
    ]
    for path, _, _ in slides:
        if not path.exists():
            raise SystemExit(f"Missing screenshot: {path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        list_file = tmp_path / "slides.txt"
        lines: list[str] = []
        for path, dur, _ in slides:
            lines.append(f"file '{path}'")
            lines.append(f"duration {dur}")
        lines.append(f"file '{slides[-1][0]}'")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        silent_video = tmp_path / "slides.mp4"
        trimmed_wav = tmp_path / "audio.wav"
        final = OUT

        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-vf",
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                str(silent_video),
            ]
        )

        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(WAV),
                "-t",
                str(DURATION),
                "-ac",
                "1",
                str(trimmed_wav),
            ]
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(WAV),
                "-t",
                str(DURATION),
                "-ac",
                "1",
                str(DEMO_WAV),
            ]
        )

        final.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(silent_video),
                "-i",
                str(trimmed_wav),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                str(final),
            ]
        )

    print(f"Demo video: {final} ({DURATION}s)")
    print(f"Demo audio: {DEMO_WAV} ({DURATION}s)")


if __name__ == "__main__":
    main()
