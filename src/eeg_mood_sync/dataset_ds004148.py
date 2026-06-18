from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mne
import numpy as np

from .eeg import BAND_ORDER, windowed_features
from .mapping import bands_to_params

FEATURE_CSV_FIELDS = [
    "task",
    "label",
    "channel",
    *BAND_ORDER,
    "alpha_beta_ratio",
    "relax_index",
    "arousal_index",
    "bpm",
    "register",
    "density",
    "velocity",
    "gate",
]

DATASET_ID = "ds004148"
S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"
DEFAULT_TASKS: tuple[str, ...] = ("eyesclosed", "eyesopen", "mathematic")

TASK_LABELS: dict[str, str] = {
    "eyesclosed": "Eyes closed (relaxation)",
    "eyesopen": "Eyes open (neutral)",
    "mathematic": "Mental subtraction (focus)",
    "memory": "Episodic memory recall",
    "music": "Music imagery",
}

PREFERRED_CHANNELS: tuple[str, ...] = ("Oz", "O1", "O2", "Pz", "POz", "P3", "P4")


@dataclass(frozen=True)
class EegSegment:
    task: str
    label: str
    signal: np.ndarray
    sfreq: float
    channel: str


def _subject_id(subject: int | str) -> str:
    if isinstance(subject, int):
        return f"sub-{subject:02d}"
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text
    return f"sub-{int(text):02d}"


def _session_id(session: int | str) -> str:
    """ds004148 uses ses-session1, ses-session2, ses-session3 (not ses-01)."""
    if isinstance(session, int):
        return f"ses-session{session}"
    text = str(session).strip()
    if text.startswith("ses-session"):
        return text
    if text.startswith("session") and text[7:].isdigit():
        return f"ses-{text}"
    if text.isdigit():
        return f"ses-session{int(text)}"
    if text.startswith("ses-"):
        suffix = text[4:]
        if suffix.isdigit():
            return f"ses-session{int(suffix)}"
        return text
    return text


def vhdr_path(data_dir: Path, subject: int | str, session: int | str, task: str) -> Path:
    sub = _subject_id(subject)
    ses = _session_id(session)
    return data_dir / sub / ses / "eeg" / f"{sub}_{ses}_task-{task}_eeg.vhdr"


def _task_filenames(subject: int | str, session: int | str, task: str) -> tuple[str, ...]:
    sub = _subject_id(subject)
    ses = _session_id(session)
    stem = f"{sub}_{ses}_task-{task}_eeg"
    return tuple(f"{stem}.{ext}" for ext in ("vhdr", "eeg", "vmrk"))


def _download_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        dest.write_bytes(response.read())


def download_tasks_s3(
    data_dir: Path,
    *,
    subject: int | str = 1,
    session: int | str = 1,
    tasks: Sequence[str] = DEFAULT_TASKS,
) -> None:
    """Direct S3 download (no OpenNeuro GraphQL API)."""
    sub = _subject_id(subject)
    ses = _session_id(session)
    for task in tasks:
        for filename in _task_filenames(subject, session, task):
            rel = Path(sub) / ses / "eeg" / filename
            url = f"{S3_BASE}/{rel.as_posix()}"
            dest = data_dir / rel
            _download_file(url, dest)


def download_tasks(
    data_dir: Path,
    *,
    subject: int | str = 1,
    session: int | str = 1,
    tasks: Sequence[str] = DEFAULT_TASKS,
    prefer_s3: bool = True,
) -> None:
    """Download selected BIDS EEG files from OpenNeuro ds004148."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if prefer_s3:
        download_tasks_s3(data_dir, subject=subject, session=session, tasks=tasks)
        return

    from openneuro import download

    sub = _subject_id(subject)
    ses = _session_id(session)
    includes = [f"{sub}/{ses}/eeg/*task-{task}*" for task in tasks]
    download(dataset=DATASET_ID, target_dir=str(data_dir), include=includes)


def _pick_channel(raw: mne.io.BaseRaw) -> str:
    for name in PREFERRED_CHANNELS:
        if name in raw.ch_names:
            return name
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        raise RuntimeError("No EEG channels found in recording")
    return raw.ch_names[int(picks[0])]


def load_task_segment(
    data_dir: Path,
    *,
    subject: int | str,
    session: int | str,
    task: str,
    max_seconds: float | None = 120.0,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    target_sfreq: float = 256.0,
    verbose: bool | str = False,
) -> EegSegment:
    path = vhdr_path(data_dir, subject, session, task)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Run download first, e.g.:\n"
            f"  python -m eeg_mood_sync.cli openneuro --download"
        )

    raw = mne.io.read_raw_brainvision(path, preload=True, verbose=verbose)
    channel = _pick_channel(raw)
    raw.pick([channel])
    raw.filter(l_freq, h_freq, verbose=verbose)
    if target_sfreq and abs(raw.info["sfreq"] - target_sfreq) > 1e-3:
        raw.resample(target_sfreq, verbose=verbose)

    if max_seconds is not None:
        raw.crop(tmax=min(max_seconds, raw.times[-1]))

    signal = raw.get_data()[0].astype(np.float32)
    return EegSegment(
        task=task,
        label=TASK_LABELS.get(task, task),
        signal=signal,
        sfreq=float(raw.info["sfreq"]),
        channel=channel,
    )


def load_task_segments(
    data_dir: Path,
    *,
    subject: int | str = 1,
    session: int | str = 1,
    tasks: Sequence[str] = DEFAULT_TASKS,
    max_seconds: float | None = 120.0,
    verbose: bool | str = False,
) -> list[EegSegment]:
    segments: list[EegSegment] = []
    for task in tasks:
        segments.append(
            load_task_segment(
                data_dir,
                subject=subject,
                session=session,
                task=task,
                max_seconds=max_seconds,
                verbose=verbose,
            )
        )
    return segments


def segments_to_features(
    segments: Iterable[EegSegment],
    *,
    window_s: float = 2.0,
    hop_s: float = 0.5,
):
    rows = []
    params = []
    for seg in segments:
        feats = list(windowed_features(seg.signal, seg.sfreq, window_s=window_s, hop_s=hop_s))
        seg_params = [bands_to_params(f) for f in feats]
        params.extend(seg_params)
        for f, p in zip(feats, seg_params):
            n = f.normalized()
            relax = n["delta"] + n["theta"] + n["alpha"]
            arousal = n["beta"] + n["gamma"]
            rows.append(
                {
                    "task": seg.task,
                    "label": seg.label,
                    "channel": seg.channel,
                    **f.as_dict(),
                    "alpha_beta_ratio": f.alpha_beta_ratio,
                    "relax_index": relax,
                    "arousal_index": arousal,
                    "bpm": p.bpm,
                    "register": p.register,
                    "density": p.density,
                    "velocity": p.velocity,
                    "gate": p.gate,
                }
            )
    return rows, params


def total_duration_seconds(segments: Iterable[EegSegment]) -> float:
    return float(sum(len(seg.signal) / seg.sfreq for seg in segments))
