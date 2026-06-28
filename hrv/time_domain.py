"""Time-domain HRV features from an RR-interval series.

Two entry points:
  compute(rr_ms)                  -> dict of scalar features for one window
  windowed(rr_ms, window_s, step) -> dict of arrays, one value per sliding window
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from config_loader import load_config

_HRV = load_config()["hrv"]
KEYS = ["SDNN", "RMSSD", "pNN50", "CVRR", "Mean_HR"]

# published resting reference ranges (sanity checks, not hard asserts)
REFERENCE = {
    "SDNN": (20.0, 100.0),
    "RMSSD": (15.0, 60.0),
    "pNN50": (0.05, 0.40),
}


def compute(rr_ms: np.ndarray) -> Dict[str, float]:
    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.size < 2:
        return {k: float("nan") for k in KEYS}
    diff = np.diff(rr)
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    pnn50 = float(np.sum(np.abs(diff) > 50.0) / diff.size)
    cvrr = float(np.std(rr) / np.mean(rr))
    mean_hr = float(60000.0 / np.mean(rr))
    return {"SDNN": sdnn, "RMSSD": rmssd, "pNN50": pnn50,
            "CVRR": cvrr, "Mean_HR": mean_hr}


def windowed(rr_ms: np.ndarray, window_s: float = None,
             step_s: float = None) -> Dict[str, np.ndarray]:
    """Slide a window over RR time (cumulative-sum clock) and compute features."""
    window_s = window_s or _HRV["window_seconds"]
    step_s = step_s or _HRV["step_seconds"]
    rr = np.asarray(rr_ms, dtype=np.float64)
    out = {k: [] for k in KEYS}
    if rr.size < 2:
        return {k: np.array([]) for k in KEYS}
    t = np.cumsum(rr) / 1000.0          # seconds at each beat
    total = t[-1]
    start = 0.0
    # if the whole record is shorter than one window, emit a single window
    if total <= window_s:
        feats = compute(rr)
        return {k: np.array([feats[k]]) for k in KEYS}
    while start + window_s <= total:
        sel = (t >= start) & (t < start + window_s)
        feats = compute(rr[sel])
        for k in KEYS:
            out[k].append(feats[k])
        start += step_s
    return {k: np.array(v) for k, v in out.items()}


def range_warnings(feats: Dict[str, float]) -> Dict[str, str]:
    warn = {}
    for k, (lo, hi) in REFERENCE.items():
        v = feats.get(k, float("nan"))
        if np.isnan(v):
            warn[k] = "nan"
        elif not (lo <= v <= hi):
            warn[k] = f"out-of-range ({v:.2f} not in {lo}-{hi})"
        else:
            warn[k] = "ok"
    return warn


def _self_test():
    from rppg.dsp import synthetic_bvp
    from hrv.peak_detection import detect_peaks
    fps = 64.0
    bvp = synthetic_bvp(150, fps, hr_bpm=70, hrv_ms=35, noise=0.02, seed=6)
    rr = detect_peaks(bvp, fps)["rr_ms"]
    feats = compute(rr)
    warn = range_warnings(feats)
    in_range = sum(v == "ok" for v in warn.values())
    print(f"[time_domain self-test] SDNN={feats['SDNN']:.1f} RMSSD={feats['RMSSD']:.1f} "
          f"pNN50={feats['pNN50']:.2f} HR={feats['Mean_HR']:.1f} "
          f"in-range {in_range}/3 -> {'PASS' if in_range >= 2 else 'FAIL'}")
    return in_range >= 2


if __name__ == "__main__":
    _self_test()
