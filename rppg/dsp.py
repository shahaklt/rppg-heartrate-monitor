"""Shared DSP helpers used by rPPG extraction, peak detection, and self-tests.

Centralised here so the bandpass definition is identical everywhere (the CHROM
extractor, heartpy pre-filter, and signal-quality checks must agree on the band).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def butter_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 3):
    """Return second-order-sections for a Butterworth bandpass.

    Clamps the high cutoff just below Nyquist so low-fps signals don't raise.
    """
    nyq = 0.5 * fs
    high = min(high_hz, nyq * 0.99)
    low = max(low_hz, 1e-4)
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def bandpass_filter(signal: np.ndarray, fs: float,
                    low_hz: float = 0.7, high_hz: float = 3.0,
                    order: int = 3) -> np.ndarray:
    """Zero-phase bandpass. Returns float64 array, NaNs interpolated first."""
    x = np.asarray(signal, dtype=np.float64).copy()
    if x.size == 0:
        return x
    x = _interp_nan(x)
    # sosfiltfilt needs length > padlen; fall back to raw signal if too short.
    sos = butter_bandpass_sos(low_hz, high_hz, fs, order)
    padlen = 3 * (sos.shape[0] * 2)
    if x.size <= padlen:
        return x - np.mean(x)
    return sosfiltfilt(sos, x)


def _interp_nan(x: np.ndarray) -> np.ndarray:
    mask = np.isnan(x)
    if not mask.any():
        return x
    if mask.all():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return x


def dominant_frequency(signal: np.ndarray, fs: float,
                       band=(0.7, 3.0)) -> float:
    """Peak frequency (Hz) of the power spectrum within `band`."""
    x = _interp_nan(np.asarray(signal, dtype=np.float64))
    x = x - np.mean(x)
    nperseg = min(len(x), 256) if len(x) >= 32 else len(x)
    freqs, psd = welch(x, fs=fs, nperseg=max(nperseg, 8))
    lo, hi = band
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any():
        return float("nan")
    band_freqs, band_psd = freqs[in_band], psd[in_band]
    return float(band_freqs[int(np.argmax(band_psd))])


def synthetic_bvp(duration_s: float, fs: float, hr_bpm: float = 72.0,
                  hrv_ms: float = 30.0, noise: float = 0.05,
                  seed: int = 0) -> np.ndarray:
    """Generate a physiologically plausible BVP-like waveform for testing.

    Beats are placed with Gaussian-jittered RR intervals (giving controllable
    HRV), each beat rendered as a short pulse, then low-noise added.
    """
    rng = np.random.default_rng(seed)
    mean_rr = 60.0 / hr_bpm
    t_end = duration_s
    beat_times = []
    t = 0.0
    while t < t_end:
        beat_times.append(t)
        rr = mean_rr + rng.normal(0.0, hrv_ms / 1000.0)
        rr = float(np.clip(rr, 0.3, 2.0))
        t += rr
    n = int(round(duration_s * fs))
    tax = np.arange(n) / fs
    sig = np.zeros(n)
    for bt in beat_times:
        # systolic peak + small dicrotic notch
        sig += np.exp(-0.5 * ((tax - bt) / 0.05) ** 2)
        sig += 0.3 * np.exp(-0.5 * ((tax - bt - 0.15) / 0.04) ** 2)
    sig += noise * rng.standard_normal(n)
    return sig.astype(np.float64)
