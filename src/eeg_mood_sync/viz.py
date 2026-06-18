from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .eeg import BAND_ORDER, BandPowers
from .mapping import MoodParams


BAND_COLORS: dict[str, str] = {
    "delta": "#1a237e",
    "theta": "#5c6bc0",
    "alpha": "#66bb6a",
    "beta": "#ffa726",
    "gamma": "#ef5350",
}


def features_to_arrays(features: Sequence[BandPowers]) -> dict[str, np.ndarray]:
    return {band: np.array([f.as_dict()[band] for f in features], dtype=float) for band in BAND_ORDER}


def plot_band_landscape(
    features: Sequence[BandPowers],
    *,
    title: str = "EEG band landscape (normalized)",
    segment_boundaries: Sequence[int] | None = None,
    segment_labels: Sequence[str] | None = None,
):
    """
    Stacked area chart: how the relative band mix changes over time/windows.
    This is the main visual that "morphs" as mental state changes.
    """
    if not features:
        raise ValueError("No features to plot")

    n_win = len(features)
    x = np.arange(n_win)
    norm = np.vstack([[f.normalized()[b] for b in BAND_ORDER] for f in features]).T

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(
        x,
        norm,
        labels=[b.capitalize() for b in BAND_ORDER],
        colors=[BAND_COLORS[b] for b in BAND_ORDER],
        alpha=0.88,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("time window")
    ax.set_ylabel("power share")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=5)
    ax.grid(True, alpha=0.2)

    if segment_boundaries and segment_labels:
        for idx, label in zip(segment_boundaries, segment_labels):
            if 0 < idx < n_win:
                ax.axvline(idx, color="white", linewidth=1.2, linestyle="--", alpha=0.8)
                ax.text(idx + 0.2, 0.98, label, fontsize=7, color="white", va="top")

    fig.tight_layout()
    return fig


def plot_mood_params(
    params: Sequence[MoodParams],
    *,
    title: str = "Music parameters over time",
):
    if not params:
        raise ValueError("No params to plot")

    x = np.arange(len(params))
    bpm = np.array([p.bpm for p in params], dtype=float)
    reg = np.array([p.register for p in params], dtype=float)
    dens = np.array([p.density for p in params], dtype=float) * 100
    vel = np.array([p.velocity for p in params], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
    axes[0, 0].plot(x, bpm, color="#ef5350", linewidth=2)
    axes[0, 0].set_ylabel("BPM")
    axes[0, 1].plot(x, reg, color="#42a5f5", linewidth=2)
    axes[0, 1].set_ylabel("rejestr")
    axes[1, 0].plot(x, dens, color="#ffa726", linewidth=2)
    axes[1, 0].set_ylabel("density %")
    axes[1, 1].plot(x, vel, color="#66bb6a", linewidth=2)
    axes[1, 1].set_ylabel("velocity")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("time window")
    axes[1, 1].set_xlabel("time window")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def segment_window_boundaries(
    segment_lengths: Iterable[int],
) -> tuple[list[int], list[str]]:
    """Cumulative window counts per segment for landscape annotations."""
    bounds: list[int] = []
    labels: list[str] = []
    total = 0
    for i, length in enumerate(segment_lengths):
        if i > 0:
            bounds.append(total)
            labels.append(f"seg{i + 1}")
        total += length
    return bounds, labels
