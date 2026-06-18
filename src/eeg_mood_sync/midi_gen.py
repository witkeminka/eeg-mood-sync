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
    channel_arp: int = 3
    channel_drums: int = 9


SCALES: dict[str, tuple[int, ...]] = {
    "relaxed": (0, 2, 4, 7, 9),
    "neutral": (0, 2, 3, 5, 7, 9, 10),
    "focused": (0, 3, 5, 7, 10),
    "aroused": (0, 2, 3, 5, 7, 8, 11),
}

CHORD_DEGREES: dict[str, tuple[tuple[int, ...], ...]] = {
    "relaxed": ((0, 4, 7), (9, 12, 16), (5, 9, 12), (7, 11, 14)),
    "neutral": ((0, 3, 7), (5, 8, 12), (3, 7, 10), (8, 12, 15)),
    "focused": ((0, 3, 7), (8, 12, 15), (5, 8, 12), (3, 7, 10)),
    "aroused": ((0, 4, 7), (3, 7, 10), (8, 11, 15), (5, 9, 12)),
}

# General MIDI drum map (channel 10)
KICK, SNARE, CLOSED_HAT, OPEN_HAT = 36, 38, 42, 46


def _ticks_per_beat() -> int:
    return 480


def _scale_for_params(p: MoodParams) -> tuple[int, ...]:
    if p.bpm < 88 and p.gate > 0.55:
        return SCALES["relaxed"]
    if p.density > 0.78:
        return SCALES["aroused"]
    if p.density > 0.58:
        return SCALES["focused"]
    if p.bpm < 105:
        return SCALES["relaxed"]
    return SCALES["neutral"]


def _chords_for_params(p: MoodParams) -> tuple[tuple[int, ...], ...]:
    if p.bpm < 88 and p.gate > 0.55:
        return CHORD_DEGREES["relaxed"]
    if p.density > 0.78:
        return CHORD_DEGREES["aroused"]
    if p.density > 0.58:
        return CHORD_DEGREES["focused"]
    return CHORD_DEGREES["neutral"]


def _degree_to_midi(degree_idx: int, *, root: int, scale: tuple[int, ...], octave: int) -> int:
    n = len(scale)
    wrapped = degree_idx % n
    octaves = degree_idx // n
    semitone = scale[wrapped] + 12 * (octave + octaves)
    return int(max(0, min(127, root + semitone)))


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
    events.append((tick + max(1, duration_ticks), "off", channel, note, 0))


def _append_drum(
    events: list[tuple[int, str, int, int, int]],
    *,
    tick: int,
    note: int,
    velocity: int,
    duration_ticks: int = 80,
) -> None:
    _append_note(events, tick=tick, channel=9, note=note, velocity=velocity, duration_ticks=duration_ticks)


def _events_to_track(events: list[tuple[int, str, int, int, int]]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    events.sort(key=lambda e: e[0])
    last_tick = 0
    for tick, kind, channel, note, velocity in events:
        delta = max(0, tick - last_tick)
        last_tick = tick
        if kind == "on":
            track.append(mido.Message("note_on", note=note, velocity=velocity, channel=channel, time=delta))
        else:
            track.append(mido.Message("note_off", note=note, velocity=0, channel=channel, time=delta))
    return track


def _sixteenth_ticks(bpm: int) -> int:
    return max(1, int(round(_ticks_per_beat() / 4)))


def _params_at(params_list: list[MoodParams], step_idx: int, steps_per_param: int) -> MoodParams:
    param_idx = min(len(params_list) - 1, step_idx // max(1, steps_per_param))
    return params_list[param_idx]


def generate_ambient_midi(
    params_stream: Iterable[MoodParams],
    *,
    seconds: float,
    seed: int | None = 0,
    config: MidiConfig = MidiConfig(),
) -> mido.MidiFile:
    """
    Beat-synced generative track driven by MoodParams.

    Layers: drums, bass, arp, pad chords, lead melody.
    """
    rng = np.random.default_rng(seed)
    params_list = list(params_stream)
    if not params_list:
        params_list = [MoodParams(bpm=110, register=0, density=0.75, velocity=72, gate=0.55)]

    avg_bpm = int(np.mean([p.bpm for p in params_list]))
    sixteenth = _sixteenth_ticks(avg_bpm)
    sixteenth_s = (sixteenth / _ticks_per_beat()) * (60.0 / avg_bpm)
    total_steps = int(max(32, round(seconds / sixteenth_s)))
    steps_per_param = max(8, total_steps // max(len(params_list), 1))

    melody_events: list[tuple[int, str, int, int, int]] = []
    pad_events: list[tuple[int, str, int, int, int]] = []
    bass_events: list[tuple[int, str, int, int, int]] = []
    arp_events: list[tuple[int, str, int, int, int]] = []
    drum_events: list[tuple[int, str, int, int, int]] = []
    tempo_events: list[tuple[int, int]] = [(0, mido.bpm2tempo(avg_bpm))]

    melody = _MelodyState(
        degree=3,
        motif=tuple(int(x) for x in rng.choice([-2, -1, 0, 1, 2, 3], size=8)),
        motif_pos=0,
    )

    chord_idx = 0
    arp_degree = 0
    last_bpm = avg_bpm

    for step in range(total_steps):
        tick = step * sixteenth
        beat_in_bar = step % 16
        bar = step // 16

        p = _params_at(params_list, step, steps_per_param)
        arousal = float(np.clip(p.density * 0.75 + (1.0 - p.gate) * 0.25, 0.0, 1.0))
        relax = float(np.clip(p.gate * 0.5 + (1.0 - p.density) * 0.5, 0.0, 1.0))
        scale = _scale_for_params(p)
        chords = _chords_for_params(p)
        sixteenth = _sixteenth_ticks(p.bpm)

        if p.bpm != last_bpm:
            tempo_events.append((tick, mido.bpm2tempo(int(p.bpm))))
            last_bpm = p.bpm

        chord = chords[(bar // 2) % len(chords)]
        chord_notes = [
            _degree_to_midi(max(0, iv // 3), root=config.key_root, scale=scale, octave=p.register)
            for iv in chord
        ]
        bass_root = _degree_to_midi(
            0, root=config.key_root - 12, scale=scale, octave=-1 + p.register // 2
        )

        # --- Drums: driving 16th grid ---
        hat_vel = int(np.clip(55 + 40 * arousal, 45, 105))
        if beat_in_bar % 2 == 0:
            _append_drum(drum_events, tick=tick, note=CLOSED_HAT, velocity=hat_vel, duration_ticks=60)
        if arousal > 0.5 and beat_in_bar % 4 == 2:
            _append_drum(drum_events, tick=tick, note=OPEN_HAT, velocity=int(hat_vel * 0.8), duration_ticks=100)

        if beat_in_bar % 4 == 0:
            kick_vel = int(np.clip(90 + 20 * arousal, 80, 115))
            _append_drum(drum_events, tick=tick, note=KICK, velocity=kick_vel, duration_ticks=120)

        if beat_in_bar in (4, 12):
            snare_vel = int(np.clip(75 + 30 * arousal, 65, 115))
            _append_drum(drum_events, tick=tick, note=SNARE, velocity=snare_vel, duration_ticks=150)
        elif arousal > 0.65 and beat_in_bar in (7, 15):
            ghost_vel = int(np.clip(75 + 30 * arousal, 65, 115) * 0.55)
            _append_drum(drum_events, tick=tick, note=SNARE, velocity=ghost_vel, duration_ticks=80)

        # --- Bass: 8th-note groove ---
        if beat_in_bar % 2 == 0:
            bass_note = bass_root
            if beat_in_bar in (4, 12) and arousal > 0.45:
                bass_note = _degree_to_midi(2, root=config.key_root - 12, scale=scale, octave=-1 + p.register // 2)
            elif beat_in_bar in (8,) and relax > 0.5:
                bass_note = _degree_to_midi(4, root=config.key_root - 12, scale=scale, octave=-1 + p.register // 2)
            bass_vel = int(np.clip(70 + 35 * arousal, 55, 115))
            bass_dur = int(sixteenth * (1.6 + 0.4 * p.gate))
            _append_note(
                bass_events,
                tick=tick,
                channel=config.channel_bass,
                note=bass_note,
                velocity=bass_vel,
                duration_ticks=bass_dur,
            )

        # --- Arp: constant 16ths (dense texture) ---
        arp_degree = (arp_degree + (1 if arousal > 0.5 else 0)) % (len(scale) * 2)
        arp_note = _degree_to_midi(
            arp_degree, root=config.key_root + 12, scale=scale, octave=p.register + 1
        )
        arp_vel = int(np.clip(40 + 45 * arousal + 15 * relax, 35, 100))
        if beat_in_bar % (1 if arousal > 0.7 else 2) == 0:
            _append_note(
                arp_events,
                tick=tick,
                channel=config.channel_arp,
                note=arp_note,
                velocity=arp_vel,
                duration_ticks=int(sixteenth * 0.7),
            )

        # --- Pad: chord stab every bar ---
        if beat_in_bar == 0:
            chord_idx += 1
            pad_vel = int(np.clip(45 + 40 * relax, 38, 95))
            pad_dur = int(sixteenth * (10 + 4 * p.gate))
            for note in chord_notes:
                _append_note(
                    pad_events,
                    tick=tick,
                    channel=config.channel_pad,
                    note=note,
                    velocity=pad_vel,
                    duration_ticks=pad_dur,
                )
            # Extra upper layer every 2 bars for lift
            if bar % 2 == 1 and arousal > 0.4:
                for note in chord_notes:
                    _append_note(
                        pad_events,
                        tick=tick,
                        channel=config.channel_pad,
                        note=min(127, note + 12),
                        velocity=int(pad_vel * 0.65),
                        duration_ticks=int(pad_dur * 0.8),
                    )

        # --- Lead melody: 8ths + fills on downbeats ---
        melody_step = beat_in_bar % 8
        play_lead = (
            melody_step in (0, 2, 4, 6)
            or (arousal > 0.6 and melody_step in (1, 5))
            or (beat_in_bar == 0 and bar % 4 == 3)
        )
        if play_lead and rng.random() < (0.55 + 0.4 * p.density):
            use_motif = rng.random() < 0.5
            if use_motif:
                melody.degree = int(
                    np.clip(
                        melody.degree + melody.motif[melody.motif_pos % len(melody.motif)],
                        0,
                        len(scale) * 3 - 1,
                    )
                )
                melody.motif_pos += 1
            else:
                step_dir = int(rng.choice([-1, 0, 1, 2], p=[0.3, 0.15, 0.4, 0.15]))
                melody.degree = int(np.clip(melody.degree + step_dir, 0, len(scale) * 3 - 1))

            note = _degree_to_midi(
                melody.degree, root=config.key_root + 12, scale=scale, octave=p.register + 2
            )
            if melody.last_note == note:
                melody.degree = int(np.clip(melody.degree + 1, 0, len(scale) * 3 - 1))
                note = _degree_to_midi(
                    melody.degree, root=config.key_root + 12, scale=scale, octave=p.register + 2
                )
            melody.last_note = note

            mel_vel = int(np.clip(p.velocity + rng.integers(-5, 12), 55, 118))
            mel_dur = int(sixteenth * (1.2 + 1.5 * p.gate))
            _append_note(
                melody_events,
                tick=tick,
                channel=config.channel_melody,
                note=note,
                velocity=mel_vel,
                duration_ticks=mel_dur,
            )

    mid = mido.MidiFile(ticks_per_beat=_ticks_per_beat())
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    last = 0
    for t, tempo in sorted(tempo_events, key=lambda x: x[0]):
        meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=max(0, t - last)))
        last = t

    for ev in (drum_events, bass_events, arp_events, pad_events, melody_events):
        if ev:
            mid.tracks.append(_events_to_track(ev))

    if len(mid.tracks) == 1:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=60, velocity=80, channel=0, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))

    return mid
