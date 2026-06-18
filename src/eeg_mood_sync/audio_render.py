from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile


def _midi_note_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def midi_to_wav(
    midi_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = 44100,
    amplitude: float = 0.2,
) -> Path:
    """
    Render MIDI to WAV with a simple sine synthesizer (no FluidSynth required).
  Suitable for quick portfolio demos.
    """
    midi_path = Path(midi_path)
    wav_path = Path(wav_path)
    mid = mido.MidiFile(str(midi_path))

    tempo = 500_000  # default 120 BPM
    ticks_per_beat = mid.ticks_per_beat or 480
    events: list[tuple[float, str, int, int]] = []  # time_s, type, note, velocity

    for track in mid.tracks:
        t_ticks = 0
        for msg in track:
            t_ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                t_s = mido.tick2second(t_ticks, ticks_per_beat, tempo)
                events.append((t_s, "on", msg.note, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                t_s = mido.tick2second(t_ticks, ticks_per_beat, tempo)
                events.append((t_s, "off", msg.note, 0))

    if not events:
        raise ValueError(f"No notes found in {midi_path}")

    end_time = max(e[0] for e in events) + 2.0
    n_samples = int(end_time * sample_rate) + 1
    audio = np.zeros(n_samples, dtype=np.float32)

    active: dict[int, tuple[float, int]] = {}
    events.sort(key=lambda e: e[0])

    for time_s, kind, note, velocity in events:
        if kind == "on":
            active[note] = (time_s, velocity)
        else:
            if note not in active:
                continue
            start_s, vel = active.pop(note)
            start_i = int(start_s * sample_rate)
            end_i = min(int(time_s * sample_rate), n_samples)
            if end_i <= start_i:
                continue
            t = np.arange(end_i - start_i) / sample_rate
            freq = _midi_note_to_hz(note)
            wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
            env = np.linspace(0.0, 1.0, min(400, len(wave)))
            if len(wave) > len(env):
                env = np.concatenate([env, np.ones(len(wave) - len(env))])
                env[-min(800, len(wave)) :] = np.linspace(1.0, 0.0, min(800, len(wave)))
            wave *= env * amplitude * (vel / 127.0)
            audio[start_i:end_i] += wave

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(wav_path), sample_rate, (audio * 32767).astype(np.int16))
    return wav_path
