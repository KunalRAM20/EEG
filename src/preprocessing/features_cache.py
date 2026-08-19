"""
Chunked, parallel, disk-cached feature/window extraction.

Extracting 44 features from ~11,500 EEG segments is the heaviest CPU step in the
whole pipeline. This module:

  * runs extraction in CHUNKS across all CPU cores (joblib) with progress output,
    so a GPU-less laptop stays responsive, and
  * CACHES the resulting matrices to ``models_store/cache/`` keyed by data source,
    row count and FEATURE_VERSION / DL_WINDOW — so every rerun after the first is
    instant and training iterations are fast.

Delete the cache (or bump FEATURE_VERSION) to force re-extraction.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
from joblib import Parallel, delayed

import config
from src.preprocessing.feature_extraction import (FEATURE_NAMES, FEATURE_VERSION,
                                                  extract_features)
from src.preprocessing.signal_processing import to_dl_window

CACHE_DIR = os.path.join(config.MODELS_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_CHUNK = 512


def _segments_fingerprint(segments) -> str:
    """Short stable identity for cache invalidation when signal data changes."""
    array = np.ascontiguousarray(np.asarray(segments, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()[:16]


def _feat_path(source: str, n: int, fingerprint: str) -> str:
    return os.path.join(
        CACHE_DIR,
        f"Xfeat_{source}_{n}_{fingerprint}_v{FEATURE_VERSION}_"
        f"{len(FEATURE_NAMES)}.npy")


def _win_path(source: str, n: int, fingerprint: str) -> str:
    return os.path.join(
        CACHE_DIR,
        f"Xwin_{source}_{n}_{fingerprint}_w{config.DL_WINDOW}.npy")


def _extract_chunked(segments, fn, chunk_size: int, n_jobs: int,
                     progress: bool, label: str) -> np.ndarray:
    n = len(segments)
    if n == 0:
        raise ValueError("Cannot extract a matrix from an empty segment collection.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    out = []
    for start in range(0, n, chunk_size):
        batch = segments[start:start + chunk_size]
        rows = Parallel(n_jobs=n_jobs)(delayed(fn)(s) for s in batch)
        out.append(np.vstack(rows))
        if progress:
            done = min(start + chunk_size, n)
            print(f"    [{label}] {done}/{n} ({100 * done // n}%)", flush=True)
    return np.vstack(out)


def _load_valid_cache(path: str, expected: tuple[int, int]):
    try:
        cached = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if cached.shape != expected or not np.all(np.isfinite(cached)):
        return None
    return cached


def _atomic_save(path: str, matrix: np.ndarray):
    """Replace a cache only after a complete NumPy file has been written."""
    fd, temp_path = tempfile.mkstemp(prefix="cache_", suffix=".npy",
                                     dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.save(temp_path, matrix)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_feature_matrix(segments, source: str, *, chunk_size: int = DEFAULT_CHUNK,
                       n_jobs: int = -1, progress: bool = True,
                       use_cache: bool = True) -> np.ndarray:
    """Return the (n, n_features) matrix, using / populating the disk cache."""
    path = _feat_path(source, len(segments), _segments_fingerprint(segments))
    if use_cache and os.path.exists(path):
        expected = (len(segments), len(FEATURE_NAMES))
        cached = _load_valid_cache(path, expected)
        if cached is not None:
            if progress:
                print(f"[cache] features hit -> {os.path.basename(path)}")
            return cached
        if progress:
            print("[cache] invalid feature cache; rebuilding")
    if progress:
        print(f"[extract] {len(segments)} segments x {len(FEATURE_NAMES)} features "
              f"in chunks of {chunk_size} ...")
    X = _extract_chunked(segments, extract_features, chunk_size, n_jobs,
                         progress, "features")
    if use_cache:
        _atomic_save(path, X)
        if progress:
            print(f"[cache] saved -> {os.path.basename(path)}")
    return X


def get_window_matrix(segments, source: str, *, chunk_size: int = DEFAULT_CHUNK,
                      n_jobs: int = -1, progress: bool = True,
                      use_cache: bool = True) -> np.ndarray:
    """Return the (n, DL_WINDOW) normalized-window matrix for deep models."""
    path = _win_path(source, len(segments), _segments_fingerprint(segments))
    if use_cache and os.path.exists(path):
        expected = (len(segments), config.DL_WINDOW)
        cached = _load_valid_cache(path, expected)
        if cached is not None:
            if progress:
                print(f"[cache] windows hit -> {os.path.basename(path)}")
            return cached
        if progress:
            print("[cache] invalid window cache; rebuilding")
    X = _extract_chunked(segments, to_dl_window, chunk_size, n_jobs,
                         progress, "windows")
    if use_cache:
        _atomic_save(path, X)
    return X
