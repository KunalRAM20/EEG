"""
EEG signal preprocessing: band-pass filtering, normalization and segmentation.

These operations mirror the "signal filtering, normalization, segmentation"
stage described in the PDR and produce clean, fixed-length windows that feed
both the feature extractor (classical models) and the deep-learning models.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, resample

import config


def bandpass_filter(signal: np.ndarray,
                    low: float = config.BANDPASS_LOW,
                    high: float = config.BANDPASS_HIGH,
                    fs: float = config.SAMPLING_RATE,
                    order: int = config.FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (0.5-40 Hz by default)."""
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("EEG signal must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("EEG signal contains non-finite values.")
    if fs <= 0 or order < 1 or not (0 <= low < high):
        raise ValueError("Invalid band-pass frequency, sampling rate, or order.")
    nyq = 0.5 * fs
    if low >= nyq:
        raise ValueError("Band-pass low frequency must be below Nyquist.")
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    # filtfilt needs the signal longer than the padding length
    padlen = 3 * max(len(a), len(b))
    if signal.shape[0] <= padlen:
        return signal - np.mean(signal)
    return filtfilt(b, a, signal)


def zscore_normalize(signal: np.ndarray) -> np.ndarray:
    """Standardize a signal to zero mean / unit variance."""
    std = np.std(signal)
    if std < 1e-12:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def preprocess_segment(signal: np.ndarray) -> np.ndarray:
    """Full single-segment preprocessing: band-pass then z-score normalize."""
    return zscore_normalize(bandpass_filter(np.asarray(signal, dtype=np.float64)))


def segment_signal(signal: np.ndarray, window: int, overlap: float = 0.0):
    """
    Split a long signal into fixed-length windows.

    Kept for completeness / streaming use — the Bonn segments are already
    fixed length, so the training pipeline treats each segment as one window.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the range [0, 1)")
    signal = np.asarray(signal)
    step = max(1, int(window * (1 - overlap)))
    windows = [
        signal[start : start + window]
        for start in range(0, len(signal) - window + 1, step)
    ]
    return np.array(windows) if windows else np.empty((0, window))


def to_dl_window(signal: np.ndarray, size: int = config.DL_WINDOW) -> np.ndarray:
    """
    Preprocess and resample a segment to the fixed length used by the neural
    networks (keeps the CNN/LSTM inputs small and CPU-friendly).
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    pre = preprocess_segment(signal)
    if len(pre) != size:
        pre = resample(pre, size)
    return zscore_normalize(pre)
