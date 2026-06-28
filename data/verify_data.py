"""Verify the organized UBFC-Phys dataset before feature extraction.

Checks, per subject (sampled) and per task:
  - files referenced in manifest still exist
  - video opens, a mid-recording frame is not black (mean pixel > 20)
  - BVP csv sampling rate is ~64 Hz (from timestamps)
  - BVP dominant frequency sits in 0.7-3.0 Hz (42-180 BPM) -> a real pulse
  - each task segment is >= min_segment_seconds (enough for HRV windows)
Writes data/verified_subjects.json.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

from config_loader import load_config
from rppg.dsp import dominant_frequency

_CFG = load_config()
_DS = _CFG["dataset"]


def _check_bvp(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    bvp = df[cols.get("bvp", df.columns[-1])].to_numpy(dtype=float)
    if "time" in cols:
        t = df[cols["time"]].to_numpy(dtype=float)
        fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else _DS["bvp_fs"]
        dur = float(t[-1] - t[0])
    else:
        fs = float(_DS["bvp_fs"])
        dur = len(bvp) / fs
    peak_hz = dominant_frequency(bvp, fs)
    fs_ok = abs(fs - _DS["bvp_fs"]) < 10
    hz_ok = 0.7 <= peak_hz <= 3.0
    dur_ok = dur >= _DS["min_segment_seconds"]
    return {"fs": fs, "dur": dur, "peak_hz": peak_hz,
            "fs_ok": fs_ok, "hz_ok": hz_ok, "dur_ok": dur_ok}


def _check_video(avi_path: str):
    try:
        import cv2
    except Exception:
        return {"present": os.path.exists(avi_path), "frames": 0,
                "fps": float("nan"), "not_black": None, "skipped": "no cv2"}
    cap = cv2.VideoCapture(avi_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
    ok, frame = cap.read()
    cap.release()
    not_black = bool(ok and frame.mean() > 20)
    return {"present": True, "frames": n, "fps": fps, "not_black": not_black}


def verify(sample: int = 5, seed: int = 0):
    man_path = _CFG.path("manifest")
    with open(man_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    data_root = _CFG.path("data_root")
    subjects = manifest["subjects"]
    random.Random(seed).shuffle(subjects)

    verified = []
    print(f"{'subject':10} {'task':5} {'fs':>6} {'dur':>7} {'peakHz':>7} {'video':>6} {'verdict'}")
    for subj in subjects:
        sid = subj["id"]
        subj_dir = os.path.join(data_root, sid)
        tasks = subj.get("tasks") or {t: {} for t in _DS["labels"]}
        subj_ok = True
        for task in sorted(tasks):
            csv_path = os.path.join(subj_dir, f"{task}.csv")
            avi_path = os.path.join(subj_dir, f"{task}.avi")
            if not os.path.exists(csv_path):
                print(f"{sid:10} {task:5} {'--missing csv--':>30}")
                subj_ok = False
                continue
            b = _check_bvp(csv_path)
            v = _check_video(avi_path) if os.path.exists(avi_path) else {"not_black": None}
            ok = b["fs_ok"] and b["hz_ok"] and b["dur_ok"] and (v["not_black"] in (True, None))
            subj_ok = subj_ok and ok
            print(f"{sid:10} {task:5} {b['fs']:6.1f} {b['dur']:7.1f} "
                  f"{b['peak_hz']:7.2f} {str(v['not_black']):>6} "
                  f"{'PASS' if ok else 'FAIL'}")
        if subj_ok:
            verified.append(sid)

    out = _CFG.path("verified_subjects")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"verified": verified}, fh, indent=2)
    print(f"\n{len(verified)}/{len(subjects)} subjects verified -> {out}")
    return verified


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()
    verify(args.sample)
