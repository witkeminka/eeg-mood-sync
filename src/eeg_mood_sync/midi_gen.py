from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mido
import numpy as np

from .mapping import MoodParams


@dataclass(frozen=True)
class MidiConfig:
    key_root: int = 48  # C3
    scale: tuple[int, ...] = (0, 2, 3, 5, 7, 10)  # minor pentatonic-ish
    channel: int = 0
    velocity_min: int = 40
    velocity_max: int = 90


def _ticks_per_beat() -> int:
    return 480


def _choose_note(rng: np.random.Generator, *, root: int, register: int, scale: tuple[int, ...]) -> int:
    degree = int(rng.choice(scale))
    octave = int(rng.integers(0, 3)) + register
    note = root + degree + 12 * octave
    return int(max(0, min(127, note)))


def generate_ambient_midi(
    params_stream: Iterable[MoodParams],
    *,
    seconds: float,
    seed: int | None = 0,
    config: MidiConfig = MidiConfig(),
) -> mido.MidiFile:
    rng = np.random.default_rng(seed)
    mid = mido.MidiFile(ticks_per_beat=_ticks_per_beat())
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Start with a default tempo; we'll update as we go
    tempo = mido.bpm2tempo(90)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # We'll generate in ~0.5s steps, updating params each step.
    step_s = 0.5
    steps = int(max(1, round(seconds / step_s)))
    params_list = list(params_stream)
    if not params_list:
        params_list = [MoodParams(bpm=90, register=0, density=0.6, velocity=65, gate=0.7)]

    for i in range(steps):
        p = params_list[min(i, len(params_list) - 1)]

        new_tempo = mido.bpm2tempo(int(p.bpm))
        if new_tempo != tempo:
            tempo = new_tempo
            track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        # Convert step duration to ticks
        beats = step_s * (p.bpm / 60.0)
        step_ticks = int(round(beats * mid.ticks_per_beat))
        step_ticks = max(1, step_ticks)

        if rng.random() < p.density:
            note = _choose_note(rng, root=config.key_root, register=p.register, scale=config.scale)
            vel_lo = max(1, p.velocity - 12)
            vel_hi = min(127, p.velocity + 12)
            vel = int(rng.integers(vel_lo, vel_hi + 1))

            gate = p.gate
            on_ticks = int(max(1, round(step_ticks * gate)))
            off_ticks = int(max(1, step_ticks - on_ticks))

            track.append(mido.Message("note_on", note=note, velocity=vel, channel=config.channel, time=0))
            track.append(mido.Message("note_off", note=note, velocity=0, channel=config.channel, time=on_ticks))
            if off_ticks:
                track.append(mido.Message("note_off", note=note, velocity=0, channel=config.channel, time=off_ticks))
        else:
            # Just wait
            track.append(mido.Message("note_off", note=0, velocity=0, channel=config.channel, time=step_ticks))

    return mid
