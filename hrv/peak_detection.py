"""R-peak (pulse-peak) detection on a BVP waveform -> RR-interval series.

Primary path uses heartpy.process(). If heartpy is unavailable or fails on a
window, a scipy.find_peaks fallback keeps the pipeline alive. RR intervals are
artifact-rejected before being returned.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import find_peaks

from config_loader import load_config
from rppg.dsp import bandpass_filter, dominant_frequency

_HRV = load_config()["hrv"]


def _reject_artifacts(rr_ms: np.ndarray, rr_min: float, rr_max: float,
                      jump_frac: float):
    """Drop physiologically impossible RR and large beat-to-beat jumps."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    keep = (rr >= rr_min) & (rr <= rr_max)
    rr = rr[keep]
    if rr.size < 2:
        return rr, int((~keep).sum())
    removed = int((~keep).sum())
    # iterative jump rejection vs running median
    cleaned = [rr[0]]
    for v in rr[1:]:
        ref = cleaned[-1]
        if abs(v - ref) / ref > jump_frac:
            removed += 1
            continue
        cleaned.append(v)
    return np.array(cleaned), removed


def _heartpy_peaks(bvp: np.ndarray, fs: float):
    import heartpy as hp
    wd, _ = hp.process(bvp.astype(np.float64), sample_rate=float(fs))
    peaks = np.asarray(wd["peaklist"], dtype=int)
    # heartpy marks rejected peaks; keep only accepted if available
    if "binary_peaklist" in wd:
        binary = np.asarray(wd["binary_peaklist"], dtype=bool)
        if binary.size == peaks.size:
            peaks = peaks[binary]
    return peaks


def _scipy_peaks(bvp: np.ndarray, fs: float):
    # min distance = shortest plausible beat (200 BPM -> 0.3 s)
    distance = max(int(fs * 0.3), 1)
    peaks, _ = find_peaks(bvp, distance=distance)
    return peaks


def _implied_hr(peaks: np.ndarray, fs: float) -> float:
    """HR (BPM) implied by the median inter-peak interval."""
    if peaks.size < 2:
        return float("nan")
    rr = np.diff(peaks) / fs
    return float(60.0 / np.median(rr)) if rr.size and np.median(rr) > 0 else float("nan")


def detect_peaks(bvp: np.ndarray, fps: float, prefilter: bool = True) -> Dict:
    """BVP (T,) -> {'peaks','rr_ms','artifacts_removed','source'}.

    heartpy and scipy.find_peaks are both run, then the one whose median-RR heart
    rate best matches the *spectral* dominant frequency is chosen. heartpy silently
    halves the beat count on noisy wrist PPG (detecting ~40 of ~95 real beats);
    the spectrum is the ground truth on rate, so cross-checking against it rejects
    that failure instead of blindly trusting heartpy whenever it returns >=3 peaks.
    """
    sig = np.asarray(bvp, dtype=np.float64)
    if prefilter:
        sig = bandpass_filter(sig, fps)

    spectral_hz = dominant_frequency(sig, fps)
    spectral_hr = spectral_hz * 60.0 if np.isfinite(spectral_hz) else float("nan")

    candidates = []
    try:
        hp = _heartpy_peaks(sig, fps)
        if hp.size >= 3:
            candidates.append(("heartpy", hp))
    except Exception:
        pass
    sp = _scipy_peaks(sig, fps)
    if sp.size >= 3:
        candidates.append(("scipy", sp))

    if not candidates:
        peaks, source = sp, "scipy"
    elif np.isfinite(spectral_hr):
        def _rate_err(item):
            hr = _implied_hr(item[1], fps)
            return abs(hr - spectral_hr) / spectral_hr if np.isfinite(hr) else np.inf
        source, peaks = min(candidates, key=_rate_err)
    else:
        source, peaks = candidates[0]   # heartpy preferred when spectrum unclear

    rr_ms = np.diff(peaks) / fps * 1000.0
    rr_clean, removed = _reject_artifacts(
        rr_ms, _HRV["rr_min_ms"], _HRV["rr_max_ms"], _HRV["rr_jump_frac"])

    return {
        "peaks": peaks,
        "rr_ms": rr_clean,
        "artifacts_removed": removed,
        "source": source,
    }


def hr_from_rr(rr_ms: np.ndarray) -> float:
    rr = np.asarray(rr_ms, dtype=np.float64)
    return float(60000.0 / np.mean(rr)) if rr.size else float("nan")


def _self_test():
    from rppg.dsp import synthetic_bvp
    fps = 64.0
    # exact 72 BPM (1.2 Hz) clean signal
    bvp = synthetic_bvp(30, fps, hr_bpm=72.0, hrv_ms=0.0, noise=0.01, seed=5)
    res = detect_peaks(bvp, fps)
    hr = hr_from_rr(res["rr_ms"])
    ok = abs(hr - 72.0) < 2.0
    print(f"[peak_detection self-test] HR={hr:.1f} BPM expected 72+/-2 "
          f"src={res['source']} removed={res['artifacts_removed']} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _self_test()
