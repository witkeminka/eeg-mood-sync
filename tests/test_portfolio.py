from __future__ import annotations

import numpy as np
import pytest

from eeg_mood_sync.audio_render import midi_to_wav
from eeg_mood_sync.mapping import bands_to_params
from eeg_mood_sync.midi_gen import generate_ambient_midi
from eeg_mood_sync.eeg import BandPowers


def test_midi_to_wav(tmp_path):
    params = [bands_to_params(BandPowers(1, 1, 2, 0.5, 0.2)) for _ in range(4)]
    mid_path = tmp_path / "t.mid"
    wav_path = tmp_path / "t.wav"
    generate_ambient_midi(params, seconds=2.0, seed=0).save(str(mid_path))
    out = midi_to_wav(mid_path, wav_path)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_train_metrics_structure(tmp_path):
    from eeg_mood_sync.ml_train import train_models
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    tasks = ["eyesclosed", "eyesopen", "mathematic"]
    for task in tasks:
        for _ in range(30):
            bands = rng.random(5) * 0.01
            feat = BandPowers(*bands)
            rows.append(
                {
                    "subject": "sub-01",
                    "task": task,
                    "channel": "Oz",
                    **feat.as_dict(),
                    "relax_index": 0.5,
                    "arousal_index": 0.5,
                    "bpm": bands_to_params(feat).bpm,
                }
            )
    df = pd.DataFrame(rows)
    metrics = train_models(df, model_dir=tmp_path)
    assert metrics.n_samples == 90
    assert 0.0 <= metrics.classifier_accuracy <= 1.0
    assert (tmp_path / "task_classifier.joblib").exists()
