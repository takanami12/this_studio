"""
Hand-crafted biological features for PhaBERT-CNN bio-feature branch.

Computes per-contig statistical descriptors of the DNA sequence:
    - k-mer frequencies (k=3 and k=4)        -> 64 + 256 = 320
    - GC content statistics                  -> 5
    - GC skew Fourier descriptors            -> 8
    - Dinucleotide odds ratio (rho-prime)    -> 16

Design notes:
    - Features are deterministic and depend only on the raw DNA string,
      so they can safely be precomputed once per dataset instantiation
      (no DataLoader-worker recomputation).
    - Normalisation (z-score) is fitted on the training split only and
      applied to all splits to avoid information leakage.
    - All values are float32 numpy arrays for cheap pickling.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


_BASE_TO_IDX = {b: i for i, b in enumerate(b"ACGT")}
_KMER_SIZES = (3, 4)
_KMER_DIMS = {k: 4 ** k for k in _KMER_SIZES}
_GC_WINDOW = 100
_GC_SKEW_FFT_COMPONENTS = 8
BIO_FEATURE_DIM = sum(_KMER_DIMS.values()) + 5 + _GC_SKEW_FFT_COMPONENTS + 16  # 349


def _seq_to_idx(seq: str) -> np.ndarray:
    """Map DNA string to integer base indices (A=0,C=1,G=2,T=3, other=-1)."""
    arr = np.frombuffer(seq.upper().encode("ascii", errors="replace"), dtype=np.uint8)
    out = np.full(arr.shape, -1, dtype=np.int8)
    out[arr == ord("A")] = 0
    out[arr == ord("C")] = 1
    out[arr == ord("G")] = 2
    out[arr == ord("T")] = 3
    return out


def _kmer_frequency(idx: np.ndarray, k: int) -> np.ndarray:
    """Normalised k-mer frequency vector of length 4^k. Skips k-mers with N."""
    dim = 4 ** k
    if idx.size < k:
        return np.zeros(dim, dtype=np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(idx, k)
    valid = (windows >= 0).all(axis=1)
    if not valid.any():
        return np.zeros(dim, dtype=np.float32)

    valid_windows = windows[valid].astype(np.int64)
    powers = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    codes = valid_windows @ powers
    counts = np.bincount(codes, minlength=dim).astype(np.float32)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def _gc_features(idx: np.ndarray) -> np.ndarray:
    """GC content global + 100bp-window stats."""
    if idx.size == 0:
        return np.zeros(5, dtype=np.float32)

    is_g = idx == 2
    is_c = idx == 1
    valid = idx >= 0

    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros(5, dtype=np.float32)

    gc_global = float((is_g.sum() + is_c.sum()) / n_valid)

    if idx.size < _GC_WINDOW:
        win_mean = gc_global
        win_std = 0.0
        win_min = gc_global
        win_max = gc_global
    else:
        gc_per_pos = (is_g | is_c).astype(np.float32)
        # cumulative sum trick for window mean
        cumulative = np.concatenate(([0.0], np.cumsum(gc_per_pos)))
        window_sums = cumulative[_GC_WINDOW:] - cumulative[:-_GC_WINDOW]
        gc_windows = window_sums / _GC_WINDOW
        win_mean = float(gc_windows.mean())
        win_std = float(gc_windows.std())
        win_min = float(gc_windows.min())
        win_max = float(gc_windows.max())

    return np.array([gc_global, win_mean, win_std, win_min, win_max], dtype=np.float32)


def _gc_skew_fft(idx: np.ndarray) -> np.ndarray:
    """First N magnitudes of the FFT of the GC-skew signal (G-C)/(G+C)."""
    n = _GC_SKEW_FFT_COMPONENTS
    if idx.size == 0:
        return np.zeros(n, dtype=np.float32)

    is_g = (idx == 2).astype(np.float32)
    is_c = (idx == 1).astype(np.float32)
    denom = is_g + is_c
    skew = np.zeros_like(denom)
    nonzero = denom > 0
    skew[nonzero] = (is_g[nonzero] - is_c[nonzero]) / denom[nonzero]

    spectrum = np.abs(np.fft.rfft(skew))
    out = np.zeros(n, dtype=np.float32)
    take = min(n, spectrum.size)
    if take > 0:
        # Drop the DC component (mean) so the descriptor focuses on shape.
        if spectrum.size > 1:
            out[:take] = spectrum[1:take + 1] if spectrum.size >= take + 1 else \
                np.pad(spectrum[1:], (0, max(0, take - (spectrum.size - 1))))[:take]
        else:
            out[0] = spectrum[0]
    # Normalise by sequence length to make magnitudes comparable across contigs.
    if idx.size > 0:
        out /= idx.size
    return out.astype(np.float32)


def _dinucleotide_odds_ratio(idx: np.ndarray) -> np.ndarray:
    """
    rho*(XY) = f(XY) / (f(X) * f(Y)).

    Karlin-style genomic signature (only forward strand here for speed).
    A value of 1 means the dinucleotide occurs as expected by chance;
    < 1 = under-represented (e.g. CpG depletion in many phages).
    """
    if idx.size < 2:
        return np.zeros(16, dtype=np.float32)

    valid_mono = idx >= 0
    n_valid = int(valid_mono.sum())
    if n_valid == 0:
        return np.zeros(16, dtype=np.float32)

    mono_counts = np.bincount(idx[valid_mono].astype(np.int64), minlength=4).astype(np.float32)
    mono_freq = mono_counts / n_valid

    pairs = np.lib.stride_tricks.sliding_window_view(idx, 2)
    valid_pair = (pairs >= 0).all(axis=1)
    if not valid_pair.any():
        return np.zeros(16, dtype=np.float32)

    pair_codes = pairs[valid_pair, 0].astype(np.int64) * 4 + pairs[valid_pair, 1].astype(np.int64)
    pair_counts = np.bincount(pair_codes, minlength=16).astype(np.float32)
    pair_freq = pair_counts / pair_counts.sum()

    out = np.zeros(16, dtype=np.float32)
    for x in range(4):
        for y in range(4):
            denom = mono_freq[x] * mono_freq[y]
            if denom > 0:
                out[x * 4 + y] = pair_freq[x * 4 + y] / denom
    return out


def compute_features(seq: str) -> np.ndarray:
    """All bio-features concatenated into a single (BIO_FEATURE_DIM,) vector."""
    idx = _seq_to_idx(seq)
    parts: List[np.ndarray] = [_kmer_frequency(idx, k) for k in _KMER_SIZES]
    parts.append(_gc_features(idx))
    parts.append(_gc_skew_fft(idx))
    parts.append(_dinucleotide_odds_ratio(idx))
    out = np.concatenate(parts).astype(np.float32)
    assert out.shape[0] == BIO_FEATURE_DIM, (out.shape, BIO_FEATURE_DIM)
    return out


def compute_features_batch(sequences: Sequence[str]) -> np.ndarray:
    """Stack of features for many sequences -> (N, BIO_FEATURE_DIM) float32 array."""
    out = np.empty((len(sequences), BIO_FEATURE_DIM), dtype=np.float32)
    for i, seq in enumerate(sequences):
        out[i] = compute_features(seq)
    return out


def fit_normalizer(features: np.ndarray) -> dict:
    """Fit z-score normaliser on training features. Avoids div-by-zero on constant cols."""
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return {"mean": mean, "std": std}


def apply_normalizer(features: np.ndarray, normalizer: dict) -> np.ndarray:
    return ((features - normalizer["mean"]) / normalizer["std"]).astype(np.float32)
