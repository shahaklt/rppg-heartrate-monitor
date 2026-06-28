"""Signal-quality gate. Decides whether an extracted BVP is trustworthy enough
to feed HRV analysis. Cheap to compute, runs before every classification.

Returns {'quality','snr_db','peak_hz','nan_fraction','reason'}.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import welch

from config_loader import load_config
from rppg.dsp import _interp_nan

_SQ = load_config()["signal_quality"]


def _snr_db(signal: np.ndarray, fs: float, hr_band=(0.7, 3.0),
            harmonics: int = 2, halfwidth: float = 0.2) -> tuple:
    """Canonical rPPG pulse SNR (de Haan & Jeanne 2013, eq. 5).

    A real pulse waveform is not a pure sinusoid: its power sits at the heart-rate
    fundamental AND its harmonics, and HRV smears each line over a small band. So
    "signal" = power within +/-`halfwidth` Hz of the fundamental and its first
    `harmonics-1` overtones; "noise" = the remaining power across the analysis
    band. Counting the harmonics as signal is what lets a clean PPG score high
    while motion/garbage (broadband, no harmonic comb) scores low. (The earlier
    peak-only definition wrongly scored clean contact PPG at -5 dB because the
    2nd harmonic and HRV spread were charged as noise.)
    """
    x = _interp_nan(np.asarray(signal, dtype=np.float64))
    if x.size < 16:
        return float("-inf"), float("nan")
    x = x - np.mean(x)
    nyq = 0.5 * fs
    # analysis band must reach the harmonics but stay below Nyquist
    hi_band = min(harmonics * hr_band[1] + 0.5, 0.95 * nyq)
    nperseg = int(np.clip(len(x), 8, 512))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    anal = (freqs >= hr_band[0]) & (freqs <= hi_band)
    fund = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    if not anal.any() or not fund.any() or psd[fund].sum() == 0:
        return float("-inf"), float("nan")
    peak_hz = float(freqs[fund][int(np.argmax(psd[fund]))])
    sig_mask = np.zeros_like(freqs, dtype=bool)
    for k in range(1, harmonics + 1):
        center = k * peak_hz
        if center > hi_band:
            break
        sig_mask |= np.abs(freqs - center) <= halfwidth
    sig_mask &= anal
    noise_mask = anal & ~sig_mask
    sig_power = psd[sig_mask].sum()
    noise_power = psd[noise_mask].sum()
    if sig_power <= 0:
        return float("-inf"), peak_hz
    if noise_power <= 1e-12:
        return float("inf"), peak_hz
    snr = 10.0 * np.log10(sig_power / noise_power)
    return float(snr), peak_hz


def _stationarity_hz(signal: np.ndarray, fs: float, band=(0.7, 3.0)) -> float:
    """Max spread of per-segment peak frequency across 4 splits."""
    x = _interp_nan(np.asarray(signal, dtype=np.float64))
    segs = np.array_split(x, 4)
    peaks = []
    for s in segs:
        if len(s) < 16:
            continue
        _, p = _snr_db(s, fs, band)
        if not np.isnan(p):
            peaks.append(p)
    if len(peaks) < 2:
        return 0.0
    return float(max(peaks) - min(peaks))


def assess(signal: np.ndarray, fs: float,
           nan_mask: np.ndarray = None, band=(0.7, 3.0)) -> Dict:
    """Full quality verdict for one BVP window."""
    sig = np.asarray(signal, dtype=np.float64)
    if nan_mask is None:
        nan_mask = np.isnan(sig)
    nan_frac = float(np.mean(nan_mask)) if sig.size else 1.0

    snr_db, peak_hz = _snr_db(sig, fs, band)
    drift = _stationarity_hz(sig, fs, band)

    reasons = []
    if snr_db < _SQ["snr_min_db"]:
        reasons.append(f"low SNR {snr_db:.1f}dB")
    if drift > _SQ["stationarity_max_hz"]:
        reasons.append(f"non-stationary {drift:.2f}Hz drift")
    if nan_frac > _SQ["nan_max_fraction"]:
        reasons.append(f"coverage {nan_frac:.0%} NaN")

    quality = "HIGH" if not reasons else "LOW"
    return {
        "quality": quality,
        "snr_db": snr_db,
        "peak_hz": peak_hz,
        "nan_fraction": nan_frac,
        "drift_hz": drift,
        "reason": "ok" if quality == "HIGH" else "; ".join(reasons),
    }


def _self_test():
    from rppg.dsp import synthetic_bvp
    fs = 35.0
    clean = synthetic_bvp(30, fs, hr_bpm=72, noise=0.02, seed=2)
    rng = np.random.default_rng(3)
    corrupt = clean + 10.0 * rng.standard_normal(clean.size)

    q_clean = assess(clean, fs)
    q_corrupt = assess(corrupt, fs)
    ok1 = q_clean["quality"] == "HIGH"
    ok2 = q_corrupt["quality"] == "LOW" and q_corrupt["snr_db"] < 5
    print(f"[signal_quality self-test] clean={q_clean['quality']} "
          f"snr={q_clean['snr_db']:.1f} | corrupt={q_corrupt['quality']} "
          f"snr={q_corrupt['snr_db']:.1f} -> "
          f"{'PASS' if ok1 and ok2 else 'FAIL'}")
    return ok1 and ok2


if __name__ == "__main__":
    _self_test()
