from __future__ import annotations

from dataclasses import dataclass

from .eeg import BandPowers


@dataclass(frozen=True)
class MoodParams:
    bpm: int
    register: int  # MIDI octave-ish shift
    density: float  # 0..1 note probability
    velocity: int  # MIDI note velocity 1..127
    gate: float  # note length fraction 0..1 (higher = longer / more pad-like)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def bands_to_params(bands: BandPowers) -> MoodParams:
    """
    Map full band vector to music parameters.

    Relaxed profile (delta+theta+alpha) -> slow, low, soft, long notes.
    Aroused profile (beta+gamma)         -> fast, high, dense, short notes.
    """
    n = bands.normalized()
    relax = n["delta"] + n["theta"] + n["alpha"]
    arousal = n["beta"] + n["gamma"]

    # 0 = aroused, 1 = relaxed
    t = relax / max(relax + arousal, 1e-12)

    bpm = int(round(128 - 38 * t + 22 * n["gamma"]))
    register = int(round(2 - 3 * t + 2 * n["gamma"] - n["delta"]))
    density = clamp(0.58 + 0.38 * arousal + 0.12 * n["gamma"], 0.52, 0.96)
    velocity = int(round(55 + 45 * relax + 25 * n["alpha"]))
    gate = clamp(0.38 + 0.32 * t + 0.10 * n["theta"], 0.32, 0.78)

    return MoodParams(
        bpm=int(clamp(bpm, 55, 150)),
        register=int(clamp(register, -2, 4)),
        density=density,
        velocity=int(clamp(velocity, 25, 110)),
        gate=gate,
    )


def ratio_to_params(alpha_beta_ratio: float) -> MoodParams:
    """Legacy α/β-only mapping."""
    r = clamp(alpha_beta_ratio, 0.2, 5.0)
    t = (r - 0.2) / (5.0 - 0.2)
    return MoodParams(
        bpm=int(round(120 - 60 * t)),
        register=int(round(2 - 3 * t)),
        density=clamp(0.85 - 0.55 * t, 0.25, 0.9),
        velocity=int(round(50 + 35 * t)),
        gate=clamp(0.5 + 0.35 * t, 0.4, 0.9),
    )
