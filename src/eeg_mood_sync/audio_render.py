from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile


def _midi_note_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _synth_voice(t: np.ndarray, freq: float, *, voice: str) -> np.ndarray:
    phase = 2 * np.pi * freq * t
    if voice == "kick":
        pitch_env = np.exp(-18.0 * t)
        freq_kick = 90.0 + 110.0 * pitch_env
        return np.sin(2 * np.pi * freq_kick * t).astype(np.float32)
    if voice == "snare":
        tone = 0.35 * np.sin(2 * np.pi * 180.0 * t) * np.exp(-25.0 * t)
        noise = rng_noise(len(t)) * np.exp(-20.0 * t)
        return (tone + noise).astype(np.float32)
    if voice == "hat":
        noise = rng_noise(len(t))
        return (noise * np.exp(-40.0 * t)).astype(np.float32)
    if voice == "bass":
        return (np.sin(phase) + 0.5 * np.sin(phase * 2.0) + 0.2 * np.sin(phase * 0.5)).astype(np.float32)
    if voice == "arp":
        pluck = np.exp(-7.0 * t)
        return (pluck * (0.65 * np.sin(phase) + 0.25 * np.sin(phase * 2.0))).astype(np.float32)
    if voice == "pad":
        return (
            0.5 * np.sin(phase)
            + 0.32 * np.sin(phase * 2.0)
            + 0.15 * np.sin(phase * 3.0)
            + 0.06 * np.sin(phase * 4.0)
        ).astype(np.float32)
    # melody lead
    vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 5.5 * t)
    return (
        vibrato
        * (0.68 * np.sin(phase) + 0.26 * np.sin(phase * 2.0) + 0.1 * np.sin(phase * 3.0))
    ).astype(np.float32)


def rng_noise(n: int) -> np.ndarray:
    return (np.random.default_rng(0).random(n).astype(np.float32) * 2.0 - 1.0)


def _adsr(length: int, *, attack: int, release: int) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    atk = min(attack, length // 4)
    rel = min(release, length // 2)
    if atk > 0:
        env[:atk] = np.linspace(0.0, 1.0, atk, dtype=np.float32)
    if rel > 0 and length > rel:
        env[-rel:] = np.linspace(1.0, 0.0, rel, dtype=np.float32)
    return env


def _voice_for_channel(channel: int, note: int) -> str:
    if channel == 9:
        if note in (36, 35):
            return "kick"
        if note in (38, 40):
            return "snare"
        return "hat"
    if channel == 2 or note < 45:
        return "bass"
    if channel == 3:
        return "arp"
    if channel == 1 or note < 62:
        return "pad"
    return "melody"


def _mix_and_master(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(audio) < sample_rate:
        return audio
    # Light bus compression
    threshold = 0.35
    ratio = 3.0
    abs_a = np.abs(audio)
    over = np.maximum(abs_a - threshold, 0.0)
    compressed = np.sign(audio) * (abs_a - over + over / ratio)

    # Short ambience
    delays = [int(0.025 * sample_rate), int(0.055 * sample_rate), int(0.11 * sample_rate)]
    wet = np.zeros_like(compressed)
    for i, delay in enumerate(delays):
        wet[delay:] += compressed[:-delay] * (0.22 - i * 0.05)
    out = compressed * 0.82 + wet * 0.35

    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak * 0.97
    return out.astype(np.float32)


def midi_to_wav(
    midi_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = 44100,
    amplitude: float = 0.28,
) -> Path:
    """Render MIDI to WAV with drums, bass, arp, pad, and lead timbres."""
    midi_path = Path(midi_path)
    wav_path = Path(wav_path)
    mid = mido.MidiFile(str(midi_path))

    tempo = 500_000
    ticks_per_beat = mid.ticks_per_beat or 480
    events: list[tuple[float, str, int, int, int]] = []

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

    end_time = max(e[0] for e in events) + 2.0
    n_samples = int(end_time * sample_rate) + 1
    audio = np.zeros(n_samples, dtype=np.float32)
    kick_envelope = np.ones(n_samples, dtype=np.float32)

    active: dict[tuple[int, int], tuple[float, int]] = {}
    events.sort(key=lambda e: e[0])

    for time_s, kind, channel, note, velocity in events:
        key = (channel, note)
        if kind == "on":
            active[key] = (time_s, velocity)
            if channel == 9 and note in (36, 35):
                ki = int(time_s * sample_rate)
                duck_len = int(0.08 * sample_rate)
                end = min(ki + duck_len, n_samples)
                kick_envelope[ki:end] *= np.linspace(0.55, 1.0, end - ki, dtype=np.float32)
        else:
            if key not in active:
                continue
            start_s, vel = active.pop(key)
            start_i = int(start_s * sample_rate)
            end_i = min(int(time_s * sample_rate), n_samples)
            if end_i <= start_i:
                continue
            t = np.arange(end_i - start_i, dtype=np.float32) / sample_rate
            voice = _voice_for_channel(channel, note)
            freq = _midi_note_to_hz(note) if voice not in ("kick", "snare", "hat") else 0.0
            wave = _synth_voice(t, freq, voice=voice)

            if voice == "kick":
                atk, rel = int(0.002 * sample_rate), int(0.12 * sample_rate)
                amp = amplitude * 1.4
            elif voice == "snare":
                atk, rel = int(0.001 * sample_rate), int(0.08 * sample_rate)
                amp = amplitude * 0.9
            elif voice == "hat":
                atk, rel = int(0.001 * sample_rate), int(0.04 * sample_rate)
                amp = amplitude * 0.45
            elif voice == "arp":
                atk, rel = int(0.003 * sample_rate), int(0.06 * sample_rate)
                amp = amplitude * 0.75
            elif voice == "bass":
                atk, rel = int(0.008 * sample_rate), int(0.10 * sample_rate)
                amp = amplitude * 1.0
            elif voice == "pad":
                atk, rel = int(0.04 * sample_rate), int(0.25 * sample_rate)
                amp = amplitude * 0.7
            else:
                atk, rel = int(0.006 * sample_rate), int(0.12 * sample_rate)
                amp = amplitude * 0.95

            env = _adsr(len(wave), attack=atk, release=rel)
            wave *= env * amp * (vel / 127.0)
            audio[start_i:end_i] += wave

    audio *= kick_envelope
    audio = _mix_and_master(audio, sample_rate)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(wav_path), sample_rate, (audio * 32767).astype(np.int16))
    return wav_path
