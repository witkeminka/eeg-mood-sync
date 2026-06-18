from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile


def _midi_note_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _synth_voice(
    t: np.ndarray,
    freq: float,
    *,
    voice: str,
) -> np.ndarray:
    """Lightweight multi-partial synth — different timbres per layer."""
    phase = 2 * np.pi * freq * t
    if voice == "bass":
        wave = np.sin(phase) + 0.45 * np.sin(phase * 0.5)
    elif voice == "pad":
        wave = (
            0.55 * np.sin(phase)
            + 0.30 * np.sin(phase * 2.0)
            + 0.12 * np.sin(phase * 3.0)
        )
    else:  # melody
        wave = (
            0.70 * np.sin(phase)
            + 0.22 * np.sin(phase * 2.0)
            + 0.08 * np.sin(phase * 3.0)
        )
        # Gentle vibrato on lead.
        vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * t)
        wave *= vibrato
    return wave.astype(np.float32)


def _adsr(length: int, *, attack: int, release: int, sample_rate: int) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    atk = min(attack, length // 3)
    rel = min(release, length // 2)
    if atk > 0:
        env[:atk] = np.linspace(0.0, 1.0, atk, dtype=np.float32)
    if rel > 0 and length > rel:
        env[-rel:] = np.linspace(1.0, 0.0, rel, dtype=np.float32)
    return env


def _voice_for_channel(channel: int, note: int) -> str:
    if channel == 2 or note < 45:
        return "bass"
    if channel == 1 or note < 62:
        return "pad"
    return "melody"


def _add_space(audio: np.ndarray, sample_rate: int, mix: float = 0.28) -> np.ndarray:
    """Simple stereo-ish ambience via staggered delays."""
    if len(audio) < sample_rate:
        return audio
    delays = [int(0.028 * sample_rate), int(0.061 * sample_rate), int(0.117 * sample_rate)]
    gains = [0.38, 0.24, 0.14]
    wet = np.zeros_like(audio)
    for delay, gain in zip(delays, gains):
        wet[delay:] += audio[:-delay] * gain
    wet *= mix
    return (audio + wet).astype(np.float32)


def midi_to_wav(
    midi_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = 44100,
    amplitude: float = 0.22,
) -> Path:
    """
    Render MIDI to WAV with a lightweight multi-voice synthesizer.
    Channels 0/1/2 map to melody / pad / bass timbres.
    """
    midi_path = Path(midi_path)
    wav_path = Path(wav_path)
    mid = mido.MidiFile(str(midi_path))

    tempo = 500_000
    ticks_per_beat = mid.ticks_per_beat or 480
    events: list[tuple[float, str, int, int, int]] = []  # time_s, kind, channel, note, velocity

    for track in mid.tracks:
        t_ticks = 0
        for msg in track:
            t_ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                t_s = mido.tick2second(t_ticks, ticks_per_beat, tempo)
                events.append((t_s, "on", msg.channel, msg.note, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                t_s = mido.tick2second(t_ticks, ticks_per_beat, tempo)
                events.append((t_s, "off", msg.channel, msg.note, 0))

    if not events:
        raise ValueError(f"No notes found in {midi_path}")

    end_time = max(e[0] for e in events) + 2.5
    n_samples = int(end_time * sample_rate) + 1
    audio = np.zeros(n_samples, dtype=np.float32)

    active: dict[tuple[int, int], tuple[float, int]] = {}
    events.sort(key=lambda e: e[0])

    for time_s, kind, channel, note, velocity in events:
        key = (channel, note)
        if kind == "on":
            active[key] = (time_s, velocity)
        else:
            if key not in active:
                continue
            start_s, vel = active.pop(key)
            start_i = int(start_s * sample_rate)
            end_i = min(int(time_s * sample_rate), n_samples)
            if end_i <= start_i:
                continue
            t = np.arange(end_i - start_i, dtype=np.float32) / sample_rate
            freq = _midi_note_to_hz(note)
            voice = _voice_for_channel(channel, note)
            wave = _synth_voice(t, freq, voice=voice)
            atk = int(0.012 * sample_rate) if voice == "melody" else int(0.04 * sample_rate)
            rel = int(0.18 * sample_rate) if voice == "melody" else int(0.35 * sample_rate)
            env = _adsr(len(wave), attack=atk, release=rel, sample_rate=sample_rate)
            wave *= env * amplitude * (vel / 127.0)
            audio[start_i:end_i] += wave

    audio = _add_space(audio, sample_rate)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.95

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(wav_path), sample_rate, (audio * 32767).astype(np.int16))
    return wav_path
