"""Verify rPPG extraction against UBFC-rPPG ground-truth pulse.

For each available subject: extract BVP from the webcam video with CHROM (and
EfficientPhys if weights are present), align to the ground-truth PPG, and report
Pearson r plus heart-rate error. Passes a subject if any method reaches r > 0.5.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample
from scipy.stats import pearsonr

from data.load_ubfc_rppg import list_subjects, load_ground_truth, read_video
from hrv.peak_detection import detect_peaks, hr_from_rr
from rppg.chrom_extractor import CHROMExtractor
from rppg.dsp import bandpass_filter
from rppg.face_roi import FaceROIExtractor


def _align(a, b):
    """Resample b to len(a), bandpass both, return aligned pair."""
    if len(b) != len(a):
        b = resample(b, len(a))
    return a, b


def verify_subject(video_path, gt_path, max_seconds=30.0):
    frames, fps = read_video(video_path, max_seconds=max_seconds)
    if len(frames) < fps * 5:
        return None
    roi = FaceROIExtractor().process_video_frames(frames)
    names = [k for k in roi if k != "failed_frames"]
    trace = np.nanmean(np.stack([roi[n] for n in names]), axis=0)

    ppg, hr_gt, gt_fs = load_ground_truth(gt_path)
    # match ground truth window to the analysed video span
    gt_span = ppg[: int(max_seconds * gt_fs)] if max_seconds else ppg

    results = {}
    # CHROM
    chrom_bvp = CHROMExtractor().extract(trace, fps)
    a, b = _align(bandpass_filter(chrom_bvp, fps), bandpass_filter(gt_span, gt_fs))
    r_chrom = float(pearsonr(a, b)[0]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    hr_pred = hr_from_rr(detect_peaks(chrom_bvp, fps)["rr_ms"])
    hr_true = hr_from_rr(detect_peaks(gt_span, gt_fs)["rr_ms"])
    results["chrom"] = {"r": r_chrom, "hr_pred": hr_pred, "hr_true": hr_true,
                        "hr_err": abs(hr_pred - hr_true)}

    # EfficientPhys (only meaningful with trained weights)
    try:
        from rppg.efficientphys_extractor import EfficientPhysExtractor
        ext = EfficientPhysExtractor()
        if getattr(ext, "has_weights", False):
            ep_bvp = ext.extract(frames, fps)
            a, b = _align(bandpass_filter(ep_bvp, fps), bandpass_filter(gt_span, gt_fs))
            r_ep = float(pearsonr(a, b)[0]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
            results["efficientphys"] = {"r": r_ep}
    except Exception as e:
        results["efficientphys_error"] = str(e)
    return results


def run(n_subjects: int = 3, max_seconds: float = 30.0):
    subs = list_subjects()
    if not subs:
        print("verify_rppg: no UBFC-rPPG videos available yet — SKIP")
        print("  (Google Drive quota blocked the large .avi files; use Kaggle mirror)")
        return None
    print(f"{'subject':12} {'CHROM r':>8} {'HR pred':>8} {'HR true':>8} {'HR err':>7} {'verdict'}")
    passed = 0
    for sid, vid, gt in subs[:n_subjects]:
        res = verify_subject(vid, gt, max_seconds)
        if res is None:
            print(f"{sid:12} {'too short':>8}")
            continue
        c = res["chrom"]
        best_r = max([c["r"]] + ([res["efficientphys"]["r"]] if "efficientphys" in res else []))
        ok = best_r > 0.5
        passed += ok
        print(f"{sid:12} {c['r']:8.3f} {c['hr_pred']:8.1f} {c['hr_true']:8.1f} "
              f"{c['hr_err']:7.1f} {'PASS' if ok else 'FAIL'}")
    print(f"\n{passed}/{min(n_subjects, len(subs))} subjects passed (r>0.5)")
    return passed


if __name__ == "__main__":
    run()
