"""
Feature extraction from preprocessed EEG segments.

Produces a NAMED feature vector per segment covering the time domain, frequency
domain, wavelet sub-bands and signal-complexity measures commonly used in
seizure detection. The names are carried through training so the XAI module can
explain predictions in terms of interpretable quantities (e.g. "gamma_rel_power",
"line_length", "dwt_d3_energy").

IMPORTANT — amplitude matters. Ictal (seizure) EEG is distinguished above all by
large amplitude / energy. The signal is therefore band-pass filtered but NOT
per-segment z-scored before feature extraction, so amplitude-bearing features
(std, rms, energy, line_length, peak-to-peak, band absolute powers, wavelet
energies) carry real discriminative signal. Cross-sample standardization is done
later by the StandardScaler inside each model pipeline — that is the correct place
to normalize, because it preserves between-class amplitude differences.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

import config
from src.preprocessing.signal_processing import bandpass_filter

# PyWavelets loads a compiled extension.  Import it only when DWT features are
# actually requested so metadata-only callers (notably web-app startup) do not
# pay that native-library startup cost or appear to hang here.
pywt = None


def _require_pywt():
    """Load PyWavelets on first use and report native-loader failures clearly."""
    global pywt
    if pywt is None:
        try:
            import pywt as module
        except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "PyWavelets is required for the fixed 44-feature EEG schema. "
                "Repair it with: python -m pip install --force-reinstall "
                "--no-cache-dir PyWavelets"
            ) from exc
        pywt = module
    return pywt

# Bump when the feature definition changes so cached matrices are invalidated.
FEATURE_VERSION = 4

# Wavelet configuration (discrete wavelet transform sub-band energies).
_DWT_WAVELET = "db4"
_DWT_LEVEL = 5
_DWT_BANDS = [f"dwt_{'a' if i == 0 else 'd'}{_DWT_LEVEL if i == 0 else _DWT_LEVEL - i + 1}"
              for i in range(_DWT_LEVEL + 1)]  # a5, d5, d4, d3, d2, d1

# --------------------------------------------------------------------------- #
# Feature name schedule (order is fixed and shared with XAI / reporting layers)
# --------------------------------------------------------------------------- #
STAT_FEATURES = ["mean", "std", "variance", "skewness", "kurtosis",
                 "rms", "energy", "line_length", "zero_cross_rate",
                 "peak_to_peak", "mean_abs", "n_peaks",
                 "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
                 "spectral_entropy", "spectral_centroid", "spectral_edge_90",
                 "band_ratio_theta_alpha", "band_ratio_lowfast",
                 "higuchi_fd", "katz_fd"]
BAND_FEATURES = [f"{b}_abs_power" for b in config.FREQ_BANDS] + \
                [f"{b}_rel_power" for b in config.FREQ_BANDS]
# The schema must never depend on which packages happened to import when a
# process started. An optional-PyWavelets fallback once produced a 32-feature
# model in one environment and a 44-feature vector in another.
DWT_FEATURES = ([f"{b}_energy" for b in _DWT_BANDS] +
                [f"{b}_relenergy" for b in _DWT_BANDS])
FEATURE_NAMES = STAT_FEATURES + BAND_FEATURES + DWT_FEATURES


# --------------------------------------------------------------------------- #
# Individual feature groups
# --------------------------------------------------------------------------- #
def _hjorth(sig: np.ndarray):
    """Hjorth activity, mobility and complexity."""
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var0 = np.var(sig) + 1e-12
    var1 = np.var(d1) + 1e-12
    var2 = np.var(d2) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return activity, mobility, complexity


def _spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (np.sum(psd) + 1e-12)
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p))
    return float(ent / np.log2(len(p))) if len(p) > 1 else 0.0


def _spectral_shape(freqs: np.ndarray, psd: np.ndarray):
    """Spectral centroid (Hz) and 90% spectral edge frequency (Hz)."""
    total = np.sum(psd) + 1e-12
    centroid = float(np.sum(freqs * psd) / total)
    cumulative = np.cumsum(psd) / total
    edge_idx = int(np.searchsorted(cumulative, 0.90))
    edge_idx = min(edge_idx, len(freqs) - 1)
    return centroid, float(freqs[edge_idx])


def _band_powers(freqs: np.ndarray, psd: np.ndarray):
    # ``np.trapz`` was removed in NumPy 2.4. ``trapezoid`` is its direct
    # replacement (available since NumPy 2.0).
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy 1.x compatibility (requirements allow 1.26).
        integrate = np.trapz
    total = integrate(psd, freqs) + 1e-12
    abs_powers, rel_powers = [], []
    for (lo, hi) in config.FREQ_BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        bp = integrate(psd[mask], freqs[mask]) if np.any(mask) else 0.0
        abs_powers.append(float(bp))
        rel_powers.append(float(bp / total))
    return abs_powers, rel_powers


def _higuchi_fd(sig: np.ndarray, kmax: int = 10) -> float:
    """Higuchi fractal dimension — a strong nonlinear seizure-EEG descriptor."""
    x = np.asarray(sig, dtype=np.float64)
    n = len(x)
    if n < kmax * 2:
        return 0.0
    lengths, ks = [], []
    for k in range(1, kmax + 1):
        lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            norm = (n - 1) / ((len(idx) - 1) * k)
            lk.append(np.sum(np.abs(np.diff(x[idx]))) * norm / k)
        if lk:
            lengths.append(np.log(np.mean(lk) + 1e-12))
            ks.append(np.log(1.0 / k))
    if len(lengths) < 2:
        return 0.0
    return float(np.polyfit(ks, lengths, 1)[0])


def _katz_fd(sig: np.ndarray) -> float:
    """Katz fractal dimension (cheap curve-complexity measure)."""
    x = np.asarray(sig, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    length = float(np.sum(np.abs(np.diff(x))))
    dist = float(np.max(np.abs(x - x[0])))
    n = len(x) - 1
    if length <= 0 or dist <= 0:
        return 0.0
    log_n = np.log10(n)
    return float(log_n / (log_n + np.log10(dist / length)))


def _n_peaks(sig: np.ndarray) -> float:
    """Count local maxima above one std above the mean (spike proxy)."""
    if len(sig) < 3:
        return 0.0
    thr = np.mean(sig) + np.std(sig)
    interior = sig[1:-1]
    is_peak = (interior > sig[:-2]) & (interior > sig[2:]) & (interior > thr)
    return float(np.count_nonzero(is_peak))


def _dwt_energies(sig: np.ndarray):
    """Discrete-wavelet sub-band energies (absolute + relative)."""
    wavelets = _require_pywt()
    max_lvl = wavelets.dwt_max_level(
        len(sig), wavelets.Wavelet(_DWT_WAVELET).dec_len)
    level = min(_DWT_LEVEL, max(1, max_lvl))
    coeffs = wavelets.wavedec(sig, _DWT_WAVELET, level=level)
    actual_bands = [f"dwt_a{level}"] + [
        f"dwt_d{detail}" for detail in range(level, 0, -1)
    ]
    energy_by_band = {
        band: float(np.sum(np.square(coeff)))
        for band, coeff in zip(actual_bands, coeffs)
    }
    # Preserve the fixed a5,d5,d4,...,d1 schema. Short UCI windows cannot
    # support level 5, so a5/d5 are explicitly zero rather than shifting a4/d4
    # values into incorrectly labelled slots.
    energies = [energy_by_band.get(band, 0.0) for band in _DWT_BANDS]
    total = float(np.sum(energies)) + 1e-12
    rel = [e / total for e in energies]
    return energies, rel


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_features(signal: np.ndarray, preprocess: bool = True) -> np.ndarray:
    """
    Return the feature vector (order == FEATURE_NAMES) for one segment.

    ``preprocess`` band-pass filters the signal (0.5-40 Hz) but deliberately does
    NOT z-score it, so amplitude-bearing features remain discriminative.
    """
    raw = np.asarray(signal, dtype=np.float64).squeeze()
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError("EEG signal must be a one-dimensional array with at least 2 values.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("EEG signal contains non-finite values.")
    sig = bandpass_filter(raw) if preprocess else raw

    mean = float(np.mean(sig))
    std = float(np.std(sig))
    variance = float(np.var(sig))
    sk = float(skew(sig)) if std > 1e-9 else 0.0
    kt = float(kurtosis(sig)) if std > 1e-9 else 0.0
    rms = float(np.sqrt(np.mean(sig**2)))
    energy = float(np.sum(sig**2))
    line_length = float(np.sum(np.abs(np.diff(sig))))
    zcr = float(np.mean(np.abs(np.diff(np.sign(sig))) > 0))
    peak_to_peak = float(np.max(sig) - np.min(sig))
    mean_abs = float(np.mean(np.abs(sig)))
    n_peaks = _n_peaks(sig)
    higuchi = _higuchi_fd(sig)
    katz = _katz_fd(sig)
    activity, mobility, complexity = _hjorth(sig)

    freqs, psd = welch(sig, fs=config.SAMPLING_RATE, nperseg=min(256, len(sig)))
    spec_ent = _spectral_entropy(psd)
    centroid, edge90 = _spectral_shape(freqs, psd)
    abs_powers, rel_powers = _band_powers(freqs, psd)

    # Interpretable band ratios (theta/alpha slowing; slow-vs-fast balance).
    delta, theta, alpha, beta, gamma = abs_powers
    band_ratio_theta_alpha = float(theta / (alpha + 1e-12))
    band_ratio_lowfast = float((delta + theta) / (beta + gamma + 1e-12))

    dwt_abs, dwt_rel = _dwt_energies(sig)

    stat = [mean, std, variance, sk, kt, rms, energy, line_length, zcr,
            peak_to_peak, mean_abs, n_peaks,
            float(activity), float(mobility), float(complexity),
            spec_ent, centroid, edge90,
            band_ratio_theta_alpha, band_ratio_lowfast,
            higuchi, katz]
    vector = np.array(stat + abs_powers + rel_powers + dwt_abs + dwt_rel,
                      dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Feature extraction produced non-finite values.")
    return vector


def extract_feature_matrix(segments, preprocess: bool = True,
                           chunk_size: int = 0, progress: bool = False) -> np.ndarray:
    """
    Extraction over many segments -> (n, n_features) matrix.

    ``chunk_size`` > 0 processes the segments in batches (CPU-friendly, prints
    progress when ``progress`` is set); the result is identical either way.
    """
    n = len(segments)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    if chunk_size and chunk_size > 0:
        rows = []
        for start in range(0, n, chunk_size):
            batch = segments[start:start + chunk_size]
            rows.append(np.vstack([extract_features(s, preprocess) for s in batch]))
            if progress:
                done = min(start + chunk_size, n)
                print(f"    [features] {done}/{n} segments "
                      f"({100 * done // n}%)", flush=True)
        return np.vstack(rows)
    return np.vstack([extract_features(s, preprocess=preprocess) for s in segments])


def features_to_dict(vector: np.ndarray) -> dict:
    """Map a feature vector back to {name: value} for reporting / XAI."""
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if len(vector) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {len(vector)}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Feature vector contains non-finite values.")
    return {name: float(v) for name, v in zip(FEATURE_NAMES, vector)}
