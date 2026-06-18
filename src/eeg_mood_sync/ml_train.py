from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset_ds004148 import DEFAULT_TASKS, load_task_segments
from .eeg import BAND_ORDER, windowed_features
from .mapping import bands_to_params

FEATURE_COLS = list(BAND_ORDER)
DEFAULT_MODEL_DIR = Path("models")


@dataclass(frozen=True)
class TrainMetrics:
    classifier_accuracy: float
    classifier_cv_mean: float
    classifier_cv_std: float
    regressor_r2: float
    regressor_mae: float
    regressor_cv_mean: float
    n_samples: int
    subjects: list[str]
    tasks: list[str]


def collect_windows(
    data_dir: Path,
    *,
    subjects: Sequence[int | str],
    session: int = 1,
    tasks: Sequence[str] = DEFAULT_TASKS,
    max_seconds_per_task: float = 60.0,
    window_s: float = 2.0,
    hop_s: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for subject in subjects:
        sub_id = f"sub-{int(subject):02d}" if str(subject).isdigit() else str(subject)
        try:
            segments = load_task_segments(
                data_dir,
                subject=subject,
                session=session,
                tasks=tuple(tasks),
                max_seconds=max_seconds_per_task,
            )
        except FileNotFoundError:
            continue

        for seg in segments:
            for feat in windowed_features(
                seg.signal, seg.sfreq, window_s=window_s, hop_s=hop_s
            ):
                mood = bands_to_params(feat)
                rows.append(
                    {
                        "subject": sub_id,
                        "task": seg.task,
                        "channel": seg.channel,
                        **feat.as_dict(),
                        "relax_index": sum(feat.normalized()[b] for b in ("delta", "theta", "alpha")),
                        "arousal_index": sum(feat.normalized()[b] for b in ("beta", "gamma")),
                        "bpm": mood.bpm,
                    }
                )
    if not rows:
        raise FileNotFoundError(
            f"No EEG windows found under {data_dir}. "
            "Download ds004148 first (sub-01 … sub-05)."
        )
    return pd.DataFrame(rows)


def train_models(
    df: pd.DataFrame,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainMetrics:
    model_dir.mkdir(parents=True, exist_ok=True)

    x = df[FEATURE_COLS].to_numpy(dtype=float)
    y_task = df["task"].to_numpy()
    y_bpm = df["bpm"].to_numpy(dtype=float)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ]
    )
    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )

    x_train, x_test, y_train, y_test, bpm_train, bpm_test = train_test_split(
        x, y_task, y_bpm, test_size=test_size, random_state=random_state, stratify=y_task
    )

    clf.fit(x_train, y_train)
    reg.fit(x_train, bpm_train)

    y_pred = clf.predict(x_test)
    bpm_pred = reg.predict(x_test)

    acc = float(accuracy_score(y_test, y_pred))
    r2 = float(r2_score(bpm_test, bpm_pred))
    mae = float(mean_absolute_error(bpm_test, bpm_pred))

    cv_scores = cross_val_score(clf, x, y_task, cv=5, scoring="accuracy")
    cv_reg = cross_val_score(reg, x, y_bpm, cv=5, scoring="r2")

    joblib.dump(clf, model_dir / "task_classifier.joblib")
    joblib.dump(reg, model_dir / "bpm_regressor.joblib")
    joblib.dump(FEATURE_COLS, model_dir / "feature_cols.joblib")

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(df["task"].unique()))

    metrics = TrainMetrics(
        classifier_accuracy=acc,
        classifier_cv_mean=float(cv_scores.mean()),
        classifier_cv_std=float(cv_scores.std()),
        regressor_r2=r2,
        regressor_mae=mae,
        regressor_cv_mean=float(cv_reg.mean()),
        n_samples=len(df),
        subjects=sorted(df["subject"].unique().tolist()),
        tasks=sorted(df["task"].unique().tolist()),
    )

    payload = {
        **asdict(metrics),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": sorted(df["task"].unique().tolist()),
    }
    (model_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return metrics


def save_confusion_matrix_plot(model_dir: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    payload = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    cm = np.array(payload["confusion_matrix"])
    labels = payload["confusion_labels"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Task classifier — confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_classifier(model_dir: Path = DEFAULT_MODEL_DIR):
    return joblib.load(model_dir / "task_classifier.joblib")


def load_regressor(model_dir: Path = DEFAULT_MODEL_DIR):
    return joblib.load(model_dir / "bpm_regressor.joblib")


def predict_task(bands: dict[str, float], model_dir: Path = DEFAULT_MODEL_DIR) -> str:
    clf = load_classifier(model_dir)
    x = np.array([[bands[b] for b in FEATURE_COLS]], dtype=float)
    return str(clf.predict(x)[0])


def predict_bpm(bands: dict[str, float], model_dir: Path = DEFAULT_MODEL_DIR) -> float:
    reg = load_regressor(model_dir)
    x = np.array([[bands[b] for b in FEATURE_COLS]], dtype=float)
    return float(reg.predict(x)[0])
