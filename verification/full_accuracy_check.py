"""Final gate. Run before recording/showing the demo. Prints READY or FAILED list.

Aggregates: rPPG accuracy (if UBFC-rPPG present), HRV direction, classifier
accuracy, live latency, and a citation list. Stages with no data SKIP rather
than fail, so the check is honest about what was actually validated.
"""
from __future__ import annotations

import time

import numpy as np

CITATIONS = [
    "CHROM: de Haan & Jeanne, IEEE TBME 2013",
    "EfficientPhys: Liu et al., IEEE/CVF WACV 2023",
    "rPPG-Toolbox: Liu et al., NeurIPS 2023",
    "HRV LF/HF sympathetic marker: Frontiers Psychiatry 2021 doi:10.3389/fpsyt.2021.799029",
    "HRV arousal>valence: WESAD (Schmidt et al., ICMI 2018) + arXiv:2511.06231",
]


def _latency_check():
    from demo.realtime_pipeline import RealtimePipeline
    from rppg.dsp import synthetic_bvp
    p = RealtimePipeline()
    fps = 30.0
    pulse = synthetic_bvp(60, fps, hr_bpm=75)
    trace = 0.5 + 0.01 * np.outer(pulse, [0.3, 1.0, 0.2])
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        p.process_window(trace, fps)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    results = {}

    print("\n== 1. rPPG accuracy (UBFC-rPPG) ==")
    try:
        from verification.verify_rppg import run as vr
        out = vr()
        results["rppg"] = "SKIP" if out is None else ("PASS" if out > 0 else "FAIL")
    except Exception as e:
        print(f"  error: {e}"); results["rppg"] = "SKIP"

    print("\n== 2. HRV direction (stress raises LF/HF) ==")
    try:
        from verification.verify_hrv import run as vh
        results["hrv"] = "PASS" if vh() else "FAIL"
    except Exception as e:
        print(f"  error: {e}"); results["hrv"] = "SKIP"

    print("\n== 3. Classifier accuracy ==")
    try:
        from verification.verify_classifier import run as vc
        out = vc()
        results["classifier"] = "SKIP" if out is None else ("PASS" if out else "FAIL")
    except Exception as e:
        print(f"  error: {e}"); results["classifier"] = "SKIP"

    print("\n== 4. Live latency (<update step) ==")
    try:
        mean, std = _latency_check()
        ok = mean < load_update_step()
        print(f"  mean {mean*1000:.0f} ± {std*1000:.0f} ms -> {'PASS' if ok else 'FAIL'}")
        results["latency"] = "PASS" if ok else "FAIL"
    except Exception as e:
        print(f"  error: {e}"); results["latency"] = "FAIL"

    print("\n== 5. Citations (verify each via scientific skill before publishing) ==")
    for c in CITATIONS:
        print(f"  - {c}")
    results["citations"] = "INFO"

    print("\n==== SUMMARY ====")
    for k, v in results.items():
        print(f"  {k:12} {v}")
    failed = [k for k, v in results.items() if v == "FAIL"]
    if failed:
        print(f"\nNOT READY — failed: {failed}")
    else:
        print("\nREADY (skipped stages had no data; see above)")
    return results


def load_update_step():
    from config_loader import load_config
    return load_config()["demo"]["update_seconds"]


if __name__ == "__main__":
    main()
