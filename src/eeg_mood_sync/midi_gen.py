from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mido
import numpy as np

from .mapping import MoodParams


@dataclass(frozen=True)
class MidiConfig:
    key_root: int = 48  # C3
    channel_melody: int = 0
    channel_pad: int = 1
    channel_bass: int = 2


# Scale degrees (semitones from root within one octave, then wrapped)
SCALES: dict[str, tuple[int, ...]] = {
    "relaxed": (0, 2, 4, 7, 9),  # major pentatonic
    "neutral": (0, 2, 3, 5, 7, 9, 10),  # dorian
    "focused": (0, 3, 5, 7, 10),  # minor pentatonic
    "aroused": (0, 2, 3, 5, 7, 8, 11),  # harmonic minor-ish
}

CHORD_DEGREES: dict[str, tuple[tuple[int, ...], ...]] = {
    "relaxed": ((0, 4, 7), (9, 12, 16), (5, 9, 12)),  # I – vi – IV
    "neutral": ((0, 3, 7), (5, 8, 12), (3, 7, 10)),
    "focused": ((0, 3, 7), (8, 12, 15), (5, 8, 12)),
    "aroused": ((0, 4, 7), (3, 7, 10), (8, 11, 15)),
}


def _ticks_per_beat() -> int:
    return 480


def _scale_for_params(p: MoodParams) -> tuple[int, ...]:
    if p.bpm < 78 and p.gate > 0.65:
        return SCALES["relaxed"]
    if p.density > 0.72:
        return SCALES["aroused"]
    if p.density > 0.5:
        return SCALES["focused"]
    if p.bpm < 95:
        return SCALES["relaxed"]
    return SCALES["neutral"]


def _chords_for_params(p: MoodParams) -> tuple[tuple[int, ...], ...]:
    if p.bpm < 78 and p.gate > 0.65:
        return CHORD_DEGREES["relaxed"]
    if p.density > 0.72:
        return CHORD_DEGREES["aroused"]
    if p.density > 0.5:
        return CHORD_DEGREES["focused"]
    return CHORD_DEGREES["neutral"]


def _degree_to_midi(
    degree_idx: int,
    *,
    root: int,
    scale: tuple[int, ...],
    octave: int,
) -> int:
    n = len(scale)
    wrapped = degree_idx % n
    octaves = degree_idx // n
    semitone = scale[wrapped] + 12 * (octave + octaves)
    return int(max(0, min(127, root + semitone)))


def _step_degree(
    rng: np.random.Generator,
    current: int,
    *,
    scale_len: int,
    arousal: float,
    motif_step: int | None,
) -> int:
    if motif_step is not None:
        return int(np.clip(current + motif_step, 0, scale_len * 3 - 1))
    # Prefer small melodic steps; arousal adds occasional leaps.
    weights = np.array([0.35, 0.30, 0.20, 0.10, 0.05], dtype=np.float64)
    if arousal > 0.55:
        weights = np.array([0.20, 0.25, 0.20, 0.20, 0.15])
    step = int(rng.choice([-2, -1, 0, 1, 2], p=weights / weights.sum()))
    return int(np.clip(current + step, 0, scale_len * 3 - 1))


@dataclass
class _MelodyState:
    degree: int
    motif: tuple[int, ...]
    motif_pos: int
    last_note: int | None = None


def _append_note(
    events: list[tuple[int, str, int, int, int]],
    *,
    tick: int,
    channel: int,
    note: int,
    velocity: int,
    duration_ticks: int,
) -> None:
    events.append((tick, "on", channel, note, velocity))
    events.append((tick + duration_ticks, "off", channel, note, 0))


def _events_to_track(events: list[tuple[int, str, int, int, int]]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    events.sort(key=lambda e: e[0])
    last_tick = 0
    for tick, kind, channel, note, velocity in events:
        delta = max(0, tick - last_tick)
        last_tick = tick
        if kind == "on":
            track.append(
                mido.Message("note_on", note=note, velocity=velocity, channel=channel, time=delta)
            )
        else:
            track.append(
                mido.Message("note_off", note=note, velocity=0, channel=channel, time=delta)
            )
    return track


def generate_ambient_midi(
    params_stream: Iterable[MoodParams],
    *,
    seconds: float,
    seed: int | None = 0,
    config: MidiConfig = MidiConfig(),
) -> mido.MidiFile:
    """
    Multi-layer generative ambient piece driven by MoodParams.

    Layers:
    - Bass root (slow, grounding)
    - Pad chords (warm harmony, EEG-driven progression)
    - Melody (motif + stepwise improvisation)
    """
    rng = np.random.default_rng(seed)
    params_list = list(params_stream)
    if not params_list:
        params_list = [MoodParams(bpm=90, register=0, density=0.6, velocity=65, gate=0.7)]

    step_s = 0.35
    steps = int(max(8, round(seconds / step_s)))

    melody_events: list[tuple[int, str, int, int, int]] = []
    pad_events: list[tuple[int, str, int, int, int]] = []
    bass_events: list[tuple[int, str, int, int, int]] = []
    tempo_events: list[tuple[int, int]] = [(0, mido.bpm2tempo(int(params_list[0].bpm)))]

    melody = _MelodyState(
        degree=2,
        motif=tuple(int(x) for x in rng.choice([-1, 0, 1, 2], size=5)),
        motif_pos=0,
    )

    cumulative_tick = 0
    chord_idx = 0
    current_chord_notes: list[int] = []

    for i in range(steps):
        p = params_list[min(i, len(params_list) - 1)]
        arousal = float(np.clip(p.density * 0.7 + (1.0 - p.gate) * 0.3, 0.0, 1.0))
        relax = float(np.clip(p.gate * 0.6 + (1.0 - p.density) * 0.4, 0.0, 1.0))

        scale = _scale_for_params(p)
        chords = _chords_for_params(p)
        beats = step_s * (p.bpm / 60.0)
        step_ticks = max(1, int(round(beats * _ticks_per_beat())))

        if i == 0 or params_list[min(i - 1, len(params_list) - 1)].bpm != p.bpm:
            tempo_events.append((cumulative_tick, mido.bpm2tempo(int(p.bpm))))

        # --- Bass (every 2 steps, root of current chord) ---
        if i % 2 == 0:
            chord = chords[chord_idx % len(chords)]
            root_note = _degree_to_midi(
                chord[0] // 4,
                root=config.key_root - 12,
                scale=scale,
                octave=-1 + p.register // 2,
            )
            bass_vel = int(np.clip(45 + 25 * relax, 35, 85))
            bass_dur = int(step_ticks * (2.2 + p.gate))
            _append_note(
                bass_events,
                tick=cumulative_tick,
                channel=config.channel_bass,
                note=root_note,
                velocity=bass_vel,
                duration_ticks=bass_dur,
            )

        # --- Pad chord (every 4 steps) ---
        if i % 4 == 0:
            chord = chords[chord_idx % len(chords)]
            chord_idx += 1
            pad_vel = int(np.clip(28 + 35 * relax + 10 * p.velocity / 127.0, 22, 72))
            pad_dur = int(step_ticks * (3.5 + 2.0 * p.gate))
            current_chord_notes = []
            for interval in chord:
                deg = max(0, interval // 3)
                note = _degree_to_midi(
                    deg,
                    root=config.key_root,
                    scale=scale,
                    octave=p.register,
                )
                current_chord_notes.append(note)
                _append_note(
                    pad_events,
                    tick=cumulative_tick,
                    channel=config.channel_pad,
                    note=note,
                    velocity=pad_vel,
                    duration_ticks=pad_dur,
                )

        # --- Melody: motif + stepwise motion ---
        subdiv = 2 if p.density > 0.65 else 1
        sub_tick = cumulative_tick
        sub_step_ticks = max(1, step_ticks // subdiv)

        for sub in range(subdiv):
            play_prob = 0.55 + 0.35 * p.density - 0.15 * (sub % 2)
            if rng.random() > play_prob:
                sub_tick += sub_step_ticks
                continue

            use_motif = rng.random() < 0.42
            motif_step = melody.motif[melody.motif_pos % len(melody.motif)] if use_motif else None
            if use_motif:
                melody.motif_pos += 1

            melody.degree = _step_degree(
                rng,
                melody.degree,
                scale_len=len(scale),
                arousal=arousal,
                motif_step=motif_step,
            )
            note = _degree_to_midi(
                melody.degree,
                root=config.key_root + 12,
                scale=scale,
                octave=p.register + 1,
            )

            # Avoid immediate repeat; nudge by a scale step if needed.
            if melody.last_note is not None and note == melody.last_note:
                melody.degree = int(np.clip(melody.degree + rng.choice([-1, 1]), 0, len(scale) * 3 - 1))
                note = _degree_to_midi(
                    melody.degree,
                    root=config.key_root + 12,
                    scale=scale,
                    octave=p.register + 1,
                )
            melody.last_note = note

            mel_vel = int(np.clip(p.velocity - 8 + rng.integers(-6, 8), 40, 105))
            mel_dur = int(sub_step_ticks * (0.55 + 0.45 * p.gate))
            _append_note(
                melody_events,
                tick=sub_tick,
                channel=config.channel_melody,
                note=note,
                velocity=mel_vel,
                duration_ticks=mel_dur,
            )

            # Soft arpeggio echo on high arousal (beta/gamma profile)
            if arousal > 0.58 and current_chord_notes and rng.random() < 0.35:
                arp_note = int(rng.choice(current_chord_notes)) + 12
                _append_note(
                    melody_events,
                    tick=sub_tick + sub_step_ticks // 3,
                    channel=config.channel_melody,
                    note=min(127, arp_note),
                    velocity=int(mel_vel * 0.65),
                    duration_ticks=max(1, mel_dur // 2),
                )

            sub_tick += sub_step_ticks

        cumulative_tick += step_ticks

    mid = mido.MidiFile(ticks_per_beat=_ticks_per_beat())
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    last = 0
    for tick, tempo in sorted(tempo_events, key=lambda x: x[0]):
        delta = max(0, tick - last)
        meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=delta))
        last = tick

    for ev in (bass_events, pad_events, melody_events):
        if ev:
            mid.tracks.append(_events_to_track(ev))

    if len(mid.tracks) == 1:
        # Fallback: at least one note so WAV render works.
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=60, velocity=60, channel=0, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))

    return mid
