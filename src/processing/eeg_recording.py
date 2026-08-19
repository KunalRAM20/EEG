"""EEG upload parsing, metadata extraction, quality checks and windowing."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import welch


SUPPORTED_EEG_EXTENSIONS = {".edf", ".bdf", ".csv", ".txt", ".npy"}


def _parse_channel_names(channel_names: str | Iterable[str] | None) -> list[str]:
    if channel_names is None:
        return []
    if isinstance(channel_names, str):
        items = re.split(r"[\n,;]+", channel_names)
    else:
        items = list(channel_names)
    return [item.strip() for item in items if str(item).strip()]


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _coerce_matrix(data: np.ndarray, channel_names: list[str]) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("EEG input must be one- or two-dimensional.")
    if array.shape[1] == 1:
        return array
    if array.shape[0] == 1:
        return array.T
    if channel_names:
        if len(channel_names) == array.shape[1]:
            return array
        if len(channel_names) == array.shape[0]:
            return array.T
        if len(channel_names) < array.shape[1]:
            return array[:, :len(channel_names)]
        return array[:, :array.shape[1]]
    raise ValueError("Multi-channel numeric EEG requires explicit channel names so no arbitrary channel is selected.")


def _read_textual_matrix(path: Path) -> tuple[np.ndarray, list[str], float | None]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as fh:
        first_line = fh.readline().strip()
    tokens = [t for t in re.split(r"[\s,;\t]+", first_line) if t]
    header_like = any(not _is_number(token) for token in tokens)
    
    try:
        if header_like:
            frame = pd.read_csv(path, sep=r"[\s,;\t]+", engine="python")
        else:
            frame = pd.read_csv(path, header=None, sep=r"[\s,;\t]+", engine="python")
    except Exception:
        if header_like:
            frame = pd.read_csv(path)
        else:
            frame = pd.read_csv(path, header=None)

    auto_sr = None
    ch_names = []
    if header_like and hasattr(frame, "columns"):
        # Auto-detect sampling rate from sampling_rate_hz column
        sr_cols = [col for col in frame.columns if str(col).strip().lower() in {"sampling_rate", "sampling_rate_hz", "fs", "sr"}]
        if sr_cols:
            val = pd.to_numeric(frame[sr_cols[0]], errors="coerce").dropna()
            if not val.empty and val.iloc[0] > 0:
                auto_sr = float(val.iloc[0])

        # Auto-detect sampling rate from time_sec deltas if not already found
        if auto_sr is None:
            time_cols = [col for col in frame.columns if str(col).strip().lower() in {"time", "time_sec", "time_s", "t"}]
            if time_cols:
                t_vals = pd.to_numeric(frame[time_cols[0]], errors="coerce").dropna()
                if len(t_vals) >= 2:
                    dt = float(t_vals.iloc[1] - t_vals.iloc[0])
                    if dt > 0:
                        auto_sr = round(1.0 / dt, 2)

        drop_cols = [col for col in frame.columns if str(col).strip().lower() in {
            "sample", "index", "time", "time_sec", "time_s", "timestamp", "sampling_rate", "sampling_rate_hz", "fs"
        }]
        if drop_cols and len(drop_cols) < len(frame.columns):
            frame = frame.drop(columns=drop_cols)
        ch_names = [str(col).strip() for col in frame.columns]

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if numeric.empty or numeric.isna().all(axis=None):
        raise ValueError("No numeric EEG samples were found in the uploaded file.")
    return numeric.to_numpy(dtype=np.float64), ch_names, auto_sr


def _detect_start_time(value) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value.timestamp(), tz=timezone.utc).isoformat()
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class EEGRecording:
    signal: np.ndarray
    sampling_rate: float
    channel_names: list[str]
    units: list[str | None]
    metadata: dict[str, object]

    @property
    def channel_count(self) -> int:
        return int(self.signal.shape[1])

    @property
    def sample_count(self) -> int:
        return int(self.signal.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(self.sample_count / self.sampling_rate)

    @property
    def compatible_single_channel(self) -> bool:
        return self.signal.ndim == 2 and self.signal.shape[1] == 1


def load_recording(path: str | os.PathLike[str], *,
                   sampling_rate: float | None = None,
                   channel_names: str | Iterable[str] | None = None) -> EEGRecording:
    """Load a user-uploaded EEG file and preserve its metadata."""
    path = Path(path)
    suffix = path.suffix.lower()
    parsed_names = _parse_channel_names(channel_names)

    if suffix not in SUPPORTED_EEG_EXTENSIONS:
        raise ValueError(f"Unsupported EEG format: {suffix or path.name}")

    if suffix in {".edf", ".bdf"}:
        try:
            import mne
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError("EDF/BDF support requires the mne package.") from exc
        reader = mne.io.read_raw_edf if suffix == ".edf" else mne.io.read_raw_bdf
        raw = reader(str(path), preload=True, verbose="ERROR")
        signal = raw.get_data().T.astype(np.float64)
        sampling_rate = float(raw.info.get("sfreq") or sampling_rate or 173.61)
        if sampling_rate <= 0:
            sampling_rate = 173.61
        annotations = []
        for onset, duration, description in zip(raw.annotations.onset,
                                                raw.annotations.duration,
                                                raw.annotations.description):
            annotations.append({
                "onset": float(onset),
                "duration": float(duration),
                "description": str(description),
            })
        metadata = {
            "recording_start_time": _detect_start_time(raw.info.get("meas_date")),
            "annotations": annotations,
        }
        channel_names = list(raw.ch_names)
        units = [raw._orig_units.get(name) if hasattr(raw, "_orig_units") else None
                 for name in channel_names]
    elif suffix == ".npy":
        signal = np.load(path, allow_pickle=False)
        if sampling_rate is None or sampling_rate <= 0:
            raise ValueError("Raw NPY recordings require a sampling rate.")
        metadata = {"recording_start_time": None, "annotations": []}
        units = []
    else:
        signal, detected_names, auto_sr = _read_textual_matrix(path)
        if sampling_rate is None or sampling_rate <= 0:
            sampling_rate = auto_sr
        if sampling_rate is None or sampling_rate <= 0:
            raise ValueError("Raw CSV/TXT recordings require a sampling rate.")
        metadata = {"recording_start_time": None, "annotations": []}
        units = []
        if not parsed_names and detected_names:
            parsed_names = detected_names

    matrix = _coerce_matrix(signal, parsed_names)
    if parsed_names:
        channel_names = parsed_names if len(parsed_names) == matrix.shape[1] else parsed_names[:matrix.shape[1]]
    elif suffix in {".edf", ".bdf"}:
        channel_names = channel_names
    else:
        channel_names = [f"EEG {i + 1}" for i in range(matrix.shape[1])]

    if not np.isfinite(matrix).all():
        raise ValueError("The uploaded EEG contains non-finite values.")

    metadata.update({
        "format": suffix.lstrip("."),
        "channel_count": int(matrix.shape[1]),
        "channel_names": list(channel_names),
        "units": list(units) if units else [None] * int(matrix.shape[1]),
        "sample_count": int(matrix.shape[0]),
        "duration_seconds": float(matrix.shape[0] / float(sampling_rate)),
    })
    return EEGRecording(
        signal=matrix,
        sampling_rate=float(sampling_rate),
        channel_names=list(channel_names),
        units=list(units) if units else [None] * int(matrix.shape[1]),
        metadata=metadata,
    )


def recording_hash(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess_window_quality(signal: np.ndarray, sampling_rate: float) -> dict[str, object]:
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    issues: list[str] = []
    warnings: list[str] = []
    if not np.isfinite(array).all():
        issues.append("non-finite values")
    if array.size < max(8, int(sampling_rate // 2)):
        issues.append("insufficient duration")
    amplitude = float(np.ptp(array)) if array.size else 0.0
    std = float(np.std(array)) if array.size else 0.0
    if amplitude <= 1e-12 or std <= 1e-12:
        issues.append("flat or constant signal")
    if array.size and np.max(np.abs(array)) > 5000:
        issues.append("implausible amplitude")
    if array.size > 4 and np.max(np.abs(np.diff(array))) > max(50 * std, 1e4):
        warnings.append("abrupt discontinuity")
    if array.size > 8:
        freqs, psd = welch(array, fs=sampling_rate, nperseg=min(256, array.size))
        mains_score = None
        for freq in (50.0, 60.0):
            mask = (freqs >= freq - 1.0) & (freqs <= freq + 1.0)
            if np.any(mask):
                mains_power = float(np.trapezoid(psd[mask], freqs[mask]))
                nearby_mask = (freqs >= freq - 5.0) & (freqs <= freq + 5.0)
                nearby = float(np.trapezoid(psd[nearby_mask], freqs[nearby_mask]))
                mains_score = mains_power / max(nearby, 1e-12)
                break
        if mains_score is not None and mains_score > 0.4:
            warnings.append("excessive mains interference")
    status = "suitable"
    if issues:
        status = "unsuitable"
    elif warnings:
        status = "suitable_with_warnings"
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "amplitude": amplitude,
        "std": std,
    }


def window_recording(signal: np.ndarray, sampling_rate: float,
                     window_seconds: float, overlap_seconds: float = 0.0) -> list[dict[str, object]]:
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    window_size = max(1, int(round(window_seconds * sampling_rate)))
    overlap = max(0, int(round(overlap_seconds * sampling_rate)))
    step = max(1, window_size - overlap)
    windows = []
    for start in range(0, max(1, array.size - window_size + 1), step):
        end = min(array.size, start + window_size)
        if end - start < window_size:
            break
        windows.append({
            "start_index": int(start),
            "end_index": int(end),
            "start_seconds": float(start / sampling_rate),
            "end_seconds": float(end / sampling_rate),
            "signal": array[start:end],
            "quality": assess_window_quality(array[start:end], sampling_rate),
        })
    return windows


def aggregate_candidate_events(windows: list[dict[str, object]],
                               *, threshold: float) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for window in windows:
        probability = float(window.get("seizure_probability") or 0.0)
        quality = window.get("quality", {})
        positive = probability >= threshold and quality.get("status") != "unsuitable"
        if not positive:
            if current is not None:
                events.append(current)
                current = None
            continue
        if current is None:
            current = {
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
                "max_probability": probability,
                "mean_probability": probability,
                "supporting_windows": 1,
                "channels": list(window.get("channels_used", [])),
                "quality_warnings": list(quality.get("warnings", [])),
            }
            continue
        current["end_seconds"] = window["end_seconds"]
        current["max_probability"] = max(float(current["max_probability"]), probability)
        current["mean_probability"] = (
            float(current["mean_probability"]) * float(current["supporting_windows"]) + probability
        ) / (float(current["supporting_windows"]) + 1.0)
        current["supporting_windows"] = int(current["supporting_windows"]) + 1
        current["quality_warnings"] = sorted(set(
            list(current["quality_warnings"]) + list(quality.get("warnings", []))))
    if current is not None:
        events.append(current)
    for event in events:
        event["duration_seconds"] = float(event["end_seconds"]) - float(event["start_seconds"])
    return events