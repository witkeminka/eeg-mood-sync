from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import welch

# Standard EEG band limits (Hz)
BAND_LIMITS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

BAND_ORDER: tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma")


@dataclass(frozen=True)
class BandPowers:
    delta: float
    theta: float
    alpha: float
    beta: float
    gamma: float

    @property
    def alpha_beta_ratio(self) -> float:
        return float(self.alpha / max(self.beta, 1e-12))

    @property
    def total(self) -> float:
        return float(self.delta + self.theta + self.alpha + self.beta + self.gamma)

    def as_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }

    def normalized(self) -> dict[str, float]:
        tot = max(self.total, 1e-12)
        return {k: v / tot for k, v in self.as_dict().items()}


def bandpower_welch(
    x: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    *,
    nperseg: int | None = None,
) -> float:
    """Average power in [fmin, fmax] using Welch PSD."""
    if x.ndim != 1:
        raise ValueError("Expected 1D signal")
    if sfreq <= 0:
        raise ValueError("sfreq must be > 0")
    if fmin < 0 or fmax <= fmin:
        raise ValueError("Invalid band")

    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def band_powers(x: np.ndarray, sfreq: float) -> BandPowers:
    return BandPowers(
        **{name: bandpower_welch(x, sfreq, lo, hi) for name, (lo, hi) in BAND_LIMITS.items()}
    )


def alpha_beta_powers(x: np.ndarray, sfreq: float) -> BandPowers:
    """Backward-compatible alias."""
    return band_powers(x, sfreq)


def simulate_eeg(
    *,
    sfreq: float,
    seconds: float,
    alpha_amp: float = 1.0,
    beta_amp: float = 0.6,
    theta_amp: float = 0.4,
    delta_amp: float = 0.2,
    gamma_amp: float = 0.15,
    noise_amp: float = 0.25,
    seed: int | None = 0,
) -> np.ndarray:
    """Synthetic 1-channel EEG with all band components + noise."""
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sfreq))
    t = np.arange(n) / sfreq

    x = noise_amp * rng.standard_normal(n)
    components = (
        (delta_amp, rng.uniform(1.5, 3.0)),
        (theta_amp, rng.uniform(5.0, 7.0)),
        (alpha_amp, rng.uniform(9.0, 11.0)),
        (beta_amp, rng.uniform(18.0, 24.0)),
        (gamma_amp, rng.uniform(32.0, 40.0)),
    )
    for amp, freq in components:
        x += amp * np.sin(2 * np.pi * freq * t)
    return x.astype(np.float32)


def windowed_features(
    x: np.ndarray,
    sfreq: float,
    *,
    window_s: float = 2.0,
    hop_s: float = 0.5,
) -> Iterable[BandPowers]:
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must be > 0")
    win = int(round(window_s * sfreq))
    hop = int(round(hop_s * sfreq))
    if win <= 2 or hop <= 0:
        raise ValueError("window/hop too small")

    for start in range(0, max(len(x) - win + 1, 0), hop):
        seg = x[start : start + win]
        yield band_powers(seg, sfreq)
