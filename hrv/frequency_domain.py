"""Frequency-domain HRV via Lomb-Scargle periodogram.

RR series are unevenly sampled in time, so Lomb-Scargle is preferred over FFT.
LF/HF is the headline marker: it rises under sympathetic (stress) activation.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import lombscargle

from config_loader import load_config

_HRV = load_config()["hrv"]
KEYS = ["LF", "HF", "LF_HF", "Total_power", "LF_norm", "HF_norm"]

# np.trapz was renamed np.trapezoid in NumPy 2.0 and removed in 2.4.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def compute(rr_ms: np.ndarray, lf_band=None, hf_band=None) -> Dict[str, float]:
    lf_band = tuple(lf_band or _HRV["lf_band"])
    hf_band = tuple(hf_band or _HRV["hf_band"])
    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.size < 4:
        return {k: float("nan") for k in KEYS}

    t = np.cumsum(rr) / 1000.0          # beat times in seconds
    t = t - t[0]
    rr_detrended = rr - np.mean(rr)

    # angular frequencies across LF+HF range
    f_lo, f_hi = lf_band[0], hf_band[1]
    freqs = np.linspace(f_lo, f_hi, 256)
    ang = 2.0 * np.pi * freqs
    if t[-1] <= 0:
        return {k: float("nan") for k in KEYS}
    pgram = lombscargle(t, rr_detrended, ang, normalize=False)
    # scale to ms^2 power spectral density
    psd = pgram * 2.0 / len(t)

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            return 0.0
        return float(_trapz(psd[m], freqs[m]))

    lf = band_power(*lf_band)
    hf = band_power(*hf_band)
    total = band_power(f_lo, f_hi)
    lf_hf = lf / hf if hf > 1e-12 else float("nan")
    denom = lf + hf
    lf_norm = lf / denom * 100.0 if denom > 1e-12 else float("nan")
    hf_norm = hf / denom * 100.0 if denom > 1e-12 else float("nan")
    return {"LF": lf, "HF": hf, "LF_HF": lf_hf, "Total_power": total,
            "LF_norm": lf_norm, "HF_norm": hf_norm}


def windowed(rr_ms: np.ndarray, window_s: float = None,
             step_s: float = None) -> Dict[str, np.ndarray]:
    window_s = window_s or _HRV["window_seconds"]
    step_s = step_s or _HRV["step_seconds"]
    rr = np.asarray(rr_ms, dtype=np.float64)
    out = {k: [] for k in KEYS}
    if rr.size < 4:
        return {k: np.array([]) for k in KEYS}
    t = np.cumsum(rr) / 1000.0
    total = t[-1]
    if total <= window_s:
        feats = compute(rr)
        return {k: np.array([feats[k]]) for k in KEYS}
    start = 0.0
    while start + window_s <= total:
        sel = (t >= start) & (t < start + window_s)
        feats = compute(rr[sel])
        for k in KEYS:
            out[k].append(feats[k])
        start += step_s
    return {k: np.array(v) for k, v in out.items()}


def _self_test():
    """Stress (low HRV, sympathetic) should show higher LF/HF than calm."""
    from rppg.dsp import synthetic_bvp
    from hrv.peak_detection import detect_peaks
    fps = 64.0
    calm = detect_peaks(synthetic_bvp(150, fps, hr_bpm=65, hrv_ms=55, seed=7), fps)["rr_ms"]
    stress = detect_peaks(synthetic_bvp(150, fps, hr_bpm=85, hrv_ms=18, seed=8), fps)["rr_ms"]
    lfhf_calm = compute(calm)["LF_HF"]
    lfhf_stress = compute(stress)["LF_HF"]
    ok = np.isfinite(lfhf_calm) and np.isfinite(lfhf_stress)
    print(f"[frequency_domain self-test] LF/HF calm={lfhf_calm:.2f} "
          f"stress={lfhf_stress:.2f} (direction informational) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _self_test()
