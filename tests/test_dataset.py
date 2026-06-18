from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_mood_sync.dataset_ds004148 import (
    EegSegment,
    segments_to_features,
    vhdr_path,
)
from eeg_mood_sync.eeg import simulate_eeg
from eeg_mood_sync.mapping import bands_to_params


def test_vhdr_path():
    path = vhdr_path(Path("data/ds004148"), 1, 1, "eyesclosed")
    assert path.name == "sub-01_ses-session1_task-eyesclosed_eeg.vhdr"


def test_segments_to_features_pipeline():
    signal = simulate_eeg(sfreq=256, seconds=4, seed=0)
    segments = [
        EegSegment("eyesclosed", "relaxation", signal, 256.0, "Oz"),
        EegSegment("mathematic", "skupienie", signal * 0.8, 256.0, "Oz"),
    ]
    rows, params = segments_to_features(segments, window_s=2.0, hop_s=1.0)
    assert len(rows) > 0
    assert len(params) == len(rows)
    assert "gamma" in rows[0]
    assert "relax_index" in rows[0]
    assert params[0].velocity > 0
    assert 0 < params[0].gate <= 1


def test_bands_to_params_range():
    from eeg_mood_sync.eeg import BandPowers

    relaxed = BandPowers(delta=1, theta=2, alpha=5, beta=0.5, gamma=0.2)
    aroused = BandPowers(delta=0.2, theta=0.3, alpha=0.5, beta=4, gamma=2)

    p_relax = bands_to_params(relaxed)
    p_arousal = bands_to_params(aroused)

    assert p_relax.bpm < p_arousal.bpm
    assert p_relax.density < p_arousal.density
    assert p_relax.velocity >= 25
    assert np.isfinite(p_relax.gate)
