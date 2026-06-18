from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audio_render import midi_to_wav
from .dataset_ds004148 import (
    DATASET_ID,
    DEFAULT_TASKS,
    FEATURE_CSV_FIELDS,
    download_tasks,
    load_task_segments,
    segments_to_features,
    total_duration_seconds,
)
from .eeg import BAND_ORDER, simulate_eeg, windowed_features
from .mapping import bands_to_params
from .midi_gen import generate_ambient_midi
from .ml_train import (
    DEFAULT_MODEL_DIR,
    collect_windows,
    predict_bpm,
    predict_task,
    save_confusion_matrix_plot,
    train_models,
)
from .viz import plot_band_landscape, plot_mood_params, segment_window_boundaries


def _maybe_wav(mid_path: Path, wav_arg: str) -> None:
    if not wav_arg:
        return
    wav_path = Path(wav_arg)
    midi_to_wav(mid_path, wav_path)
    print(f"Saved WAV to {wav_path}")


def _maybe_ml_report(rows: list[dict], model_dir: Path) -> None:
    if not (model_dir / "task_classifier.joblib").exists():
        return
    print("\nML predictions (last window per task):")
    for task in sorted({r["task"] for r in rows}):
        task_rows = [r for r in rows if r["task"] == task]
        last = task_rows[-1]
        bands = {b: last[b] for b in BAND_ORDER}
        pred_task = predict_task(bands, model_dir)
        pred_bpm = predict_bpm(bands, model_dir)
        print(
            f"  {task}: true task={task}, pred={pred_task}, "
            f"heuristic bpm={last['bpm']:.0f}, ml bpm={pred_bpm:.0f}"
        )


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    x = simulate_eeg(
        sfreq=args.sfreq,
        seconds=args.seconds,
        alpha_amp=args.alpha_amp,
        beta_amp=args.beta_amp,
        noise_amp=args.noise_amp,
        seed=args.seed,
    )

    feats = list(windowed_features(x, args.sfreq, window_s=args.window_s, hop_s=args.hop_s))
    params = [bands_to_params(f) for f in feats]

    mid = generate_ambient_midi(params, seconds=args.seconds, seed=args.seed)
    mid.save(str(out))
    print(f"Saved MIDI to {out}")
    _maybe_wav(out, args.wav)

    if args.dump_stats:
        print("Band means:")
        for band in BAND_ORDER:
            vals = np.array([f.as_dict()[band] for f in feats], dtype=float)
            print(f"  {band}: mean={vals.mean():.6f}")

    return 0


def cmd_openneuro(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    tasks = tuple(args.tasks)
    if args.download:
        print(f"Downloading {DATASET_ID} for sub-{args.subject:02d}, ses-{args.session}, tasks={tasks}")
        download_tasks(
            data_dir,
            subject=args.subject,
            session=args.session,
            tasks=tasks,
        )

    segments = load_task_segments(
        data_dir,
        subject=args.subject,
        session=args.session,
        tasks=tasks,
        max_seconds=args.max_seconds_per_task,
        verbose=args.verbose,
    )

    rows, params = segments_to_features(
        segments,
        window_s=args.window_s,
        hop_s=args.hop_s,
    )
    duration = total_duration_seconds(segments)

    mid = generate_ambient_midi(params, seconds=duration, seed=args.seed)
    mid.save(str(out))
    print(f"Saved MIDI to {out}")
    _maybe_wav(out, args.wav)
    print(f"Segments: {', '.join(f'{s.task} ({s.channel})' for s in segments)}")
    print(f"Total audio length mapped: {duration:.1f}s")

    if args.features_csv:
        features_path = Path(args.features_csv)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        with features_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FEATURE_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved features to {features_path}")

    if args.dump_stats:
        for task in tasks:
            task_rows = [r for r in rows if r["task"] == task]
            if not task_rows:
                continue
            print(f"\n{task}:")
            for band in BAND_ORDER:
                vals = np.array([r[band] for r in task_rows], dtype=float)
                print(f"  {band}: mean={vals.mean():.6f}")
            print(f"  relax_index: mean={np.mean([r['relax_index'] for r in task_rows]):.3f}")
            print(f"  arousal_index: mean={np.mean([r['arousal_index'] for r in task_rows]):.3f}")
            print(f"  bpm: mean={np.mean([r['bpm'] for r in task_rows]):.1f}")

    if args.ml_report:
        _maybe_ml_report(rows, model_dir)

    if args.plot_png:
        import matplotlib.pyplot as plt

        seg_lengths = []
        all_feats = []
        for seg in segments:
            feats = list(
                windowed_features(seg.signal, seg.sfreq, window_s=args.window_s, hop_s=args.hop_s)
            )
            seg_lengths.append(len(feats))
            all_feats.extend(feats)
        bounds, _ = segment_window_boundaries(seg_lengths)
        fig1 = plot_band_landscape(
            all_feats,
            title=f"sub-{args.subject:02d} session {args.session}",
            segment_boundaries=bounds,
            segment_labels=[s.task for s in segments],
        )
        fig2 = plot_mood_params(params)
        plot_path = Path(args.plot_png)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(plot_path, dpi=150)
        mood_path = plot_path.with_name(plot_path.stem + "_mood.png")
        fig2.savefig(mood_path, dpi=150)
        plt.close(fig1)
        plt.close(fig2)
        print(f"Saved plots to {plot_path} and {mood_path}")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    subjects = args.subjects

    df = collect_windows(
        data_dir,
        subjects=subjects,
        session=args.session,
        tasks=tuple(args.tasks),
        max_seconds_per_task=args.max_seconds_per_task,
        window_s=args.window_s,
        hop_s=args.hop_s,
    )
    metrics = train_models(df, model_dir=model_dir)

    cm_png = Path(args.confusion_png)
    save_confusion_matrix_plot(model_dir, cm_png)

    print(f"Training samples: {metrics.n_samples}")
    print(f"Subjects: {', '.join(metrics.subjects)}")
    print(f"Task classifier accuracy: {metrics.classifier_accuracy:.3f}")
    print(f"Task classifier CV accuracy: {metrics.classifier_cv_mean:.3f} ± {metrics.classifier_cv_std:.3f}")
    print(f"BPM regressor R²: {metrics.regressor_r2:.3f}, MAE: {metrics.regressor_mae:.1f}")
    print(f"BPM regressor CV R²: {metrics.regressor_cv_mean:.3f}")
    print(f"Models saved to {model_dir}/")
    print(f"Confusion matrix: {cm_png}")
    print(json.dumps(metrics.__dict__, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eeg-mood-sync",
        description="EEG band-power pipeline with ML models and generative MIDI/WAV.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Generate demo MIDI from simulated EEG")
    demo.add_argument("--out", default="outputs/demo.mid")
    demo.add_argument("--wav", default="", help="Optional WAV output path")
    demo.add_argument("--seconds", type=float, default=30.0)
    demo.add_argument("--sfreq", type=float, default=256.0)
    demo.add_argument("--alpha-amp", type=float, default=1.0)
    demo.add_argument("--beta-amp", type=float, default=0.6)
    demo.add_argument("--noise-amp", type=float, default=0.25)
    demo.add_argument("--window-s", type=float, default=2.0)
    demo.add_argument("--hop-s", type=float, default=0.5)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--dump-stats", action="store_true")
    demo.set_defaults(func=cmd_demo)

    openneuro = sub.add_parser("openneuro", help=f"Generate MIDI/WAV from OpenNeuro {DATASET_ID}")
    openneuro.add_argument("--data-dir", default="data/ds004148")
    openneuro.add_argument("--out", default="outputs/ds004148_sub01.mid")
    openneuro.add_argument("--wav", default="outputs/ds004148_sub01.wav")
    openneuro.add_argument("--features-csv", default="outputs/ds004148_features.csv")
    openneuro.add_argument("--plot-png", default="docs/screenshots/band_landscape.png")
    openneuro.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    openneuro.add_argument("--ml-report", action="store_true", help="Print ML predictions if models exist")
    openneuro.add_argument("--subject", type=int, default=1)
    openneuro.add_argument("--session", type=int, default=1)
    openneuro.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=["eyesclosed", "eyesopen", "mathematic", "memory", "music"],
    )
    openneuro.add_argument("--max-seconds-per-task", type=float, default=60.0)
    openneuro.add_argument("--window-s", type=float, default=2.0)
    openneuro.add_argument("--hop-s", type=float, default=0.5)
    openneuro.add_argument("--seed", type=int, default=0)
    openneuro.add_argument("--download", action="store_true")
    openneuro.add_argument("--dump-stats", action="store_true")
    openneuro.add_argument("--verbose", action="store_true")
    openneuro.set_defaults(func=cmd_openneuro)

    train = sub.add_parser("train", help="Train task classifier + BPM regressor on local ds004148")
    train.add_argument("--data-dir", default=str(Path.home() / "Desktop/AIML/data/ds004148"))
    train.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    train.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    train.add_argument("--session", type=int, default=1)
    train.add_argument(
        "--tasks",
        nargs="+",
        default=["eyesclosed", "eyesopen", "mathematic", "memory", "music"],
    )
    train.add_argument("--max-seconds-per-task", type=float, default=60.0)
    train.add_argument("--window-s", type=float, default=2.0)
    train.add_argument("--hop-s", type=float, default=0.5)
    train.add_argument(
        "--confusion-png",
        default="docs/screenshots/confusion_matrix.png",
    )
    train.set_defaults(func=cmd_train)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
