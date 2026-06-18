from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from eeg_mood_sync.dataset_ds004148 import load_task_segments, segments_to_features
from eeg_mood_sync.eeg import simulate_eeg, windowed_features
from eeg_mood_sync.mapping import bands_to_params
from eeg_mood_sync.midi_gen import generate_ambient_midi
from eeg_mood_sync.viz import plot_band_landscape, plot_mood_params, segment_window_boundaries


st.set_page_config(page_title="EEG Mood-Sync", layout="wide")
st.title("EEG Mood-Sync")
st.caption("δ θ α β γ → band landscape → MIDI parameters")

mode = st.sidebar.radio("Source", ["Real EEG (ds004148)", "Simulator"])

with st.sidebar:
    window_s = st.slider("Window (s)", 0.5, 5.0, 2.0, 0.5)
    hop_s = st.slider("Hop (s)", 0.25, 2.0, 0.5, 0.25)
    max_seconds = st.slider("Seconds per task", 10, 120, 45, 5)
    out_path = st.text_input("MIDI path", value="outputs/streamlit_demo.mid")
    plot_path = st.text_input("Plot PNG", value="outputs/band_landscape.png")

if mode == "Real EEG (ds004148)":
    data_dir = st.sidebar.text_input("Data directory", value="data/ds004148")
    subject = st.sidebar.number_input("Subject", 1, 60, 1)
    session = st.sidebar.number_input("Session (1–3)", 1, 3, 1)
    tasks = st.sidebar.multiselect(
        "Tasks",
        ["eyesclosed", "eyesopen", "mathematic", "memory", "music"],
        default=["eyesclosed", "eyesopen", "mathematic"],
    )
else:
    sfreq = st.sidebar.selectbox("Sample rate (Hz)", [128, 256, 512], index=1)
    seconds = st.sidebar.slider("Duration (s)", 10, 120, 45, 5)
    alpha_amp = st.sidebar.slider("Alpha", 0.0, 2.0, 1.0, 0.05)
    beta_amp = st.sidebar.slider("Beta", 0.0, 2.0, 0.6, 0.05)
    theta_amp = st.sidebar.slider("Theta", 0.0, 2.0, 0.4, 0.05)
    seed = st.sidebar.number_input("Seed", 0, 10_000, 0)

if st.button("Generate MIDI + plots", type="primary"):
    seg_lengths: list[int] = []
    all_feats = []
    all_params = []
    seg_labels: list[str] = []

    if mode == "Real EEG (ds004148)":
        if not tasks:
            st.error("Select at least one task.")
            st.stop()
        segments = load_task_segments(
            Path(data_dir),
            subject=int(subject),
            session=int(session),
            tasks=tuple(tasks),
            max_seconds=float(max_seconds),
        )
        _rows, params = segments_to_features(
            segments, window_s=float(window_s), hop_s=float(hop_s)
        )
        for seg in segments:
            feats = list(
                windowed_features(seg.signal, seg.sfreq, window_s=float(window_s), hop_s=float(hop_s))
            )
            seg_lengths.append(len(feats))
            seg_labels.append(seg.task)
            all_feats.extend(feats)
        all_params = params
        total_seconds = sum(len(s.signal) / s.sfreq for s in segments)
        title = f"sub-{int(subject):02d} ses-session{int(session)}"
    else:
        x = simulate_eeg(
            sfreq=float(sfreq),
            seconds=float(seconds),
            alpha_amp=float(alpha_amp),
            beta_amp=float(beta_amp),
            theta_amp=float(theta_amp),
            seed=int(seed),
        )
        all_feats = list(windowed_features(x, float(sfreq), window_s=float(window_s), hop_s=float(hop_s)))
        all_params = [bands_to_params(f) for f in all_feats]
        seg_lengths = [len(all_feats)]
        seg_labels = ["simulation"]
        total_seconds = float(seconds)
        title = "simulated signal"

    mid = generate_ambient_midi(all_params, seconds=total_seconds, seed=0)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out))

    bounds, _ = segment_window_boundaries(seg_lengths)

    fig1 = plot_band_landscape(
        all_feats,
        title=f"Band landscape — {title}",
        segment_boundaries=bounds,
        segment_labels=seg_labels,
    )
    fig2 = plot_mood_params(all_params, title="Music parameters over time")

    png = Path(plot_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(png, dpi=150)

    st.success(f"MIDI: {out} | plot: {png}")
    if str(out).endswith(".mid"):
        wav_out = out.with_suffix(".wav")
        try:
            from eeg_mood_sync.audio_render import midi_to_wav

            midi_to_wav(out, wav_out)
            st.audio(str(wav_out))
            st.caption(f"WAV: {wav_out}")
        except Exception as exc:
            st.warning(f"Could not render WAV: {exc}")

    model_dir = Path("models")
    if (model_dir / "task_classifier.joblib").exists():
        from eeg_mood_sync.ml_train import predict_bpm, predict_task

        last = all_feats[-1]
        bands = last.as_dict()
        st.info(
            f"ML: task≈{predict_task(bands)} | BPM≈{predict_bpm(bands):.0f} (models in models/)"
        )

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
        st.markdown(
            """
**How to read the band landscape:**
- More **green (α)** and **purple (θ)** → relaxed profile, slower music
- More **orange (β)** and **red (γ)** → aroused profile, faster music
- Vertical lines = task boundaries in the recording
            """
        )
    with col2:
        st.pyplot(fig2)

    plt.close(fig1)
    plt.close(fig2)

    st.subheader("Latest window — band shares")
    if all_feats:
        st.bar_chart(all_feats[-1].normalized())

    st.subheader("MIDI parameters (latest window)")
    st.json(all_params[-1].__dict__ if all_params else {})
