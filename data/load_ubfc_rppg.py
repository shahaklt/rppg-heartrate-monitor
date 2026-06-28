"""Loader for UBFC-rPPG (rPPG validation set: webcam video + pulse-ox ground truth).

Folder layouts handled:
  DATASET_2/subject{n}/vid.avi + ground_truth.txt   (realistic subset)
  DATASET_1/{n}-gt/...                               (simple subset)
ground_truth.txt holds 3 rows: [PPG trace], [HR per sample], [timestamps],
sampled per video frame (~30 Hz).
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np

from config_loader import load_config, project_path

_CFG = load_config()
_U = _CFG["ubfc_rppg"]


def list_subjects(root: str = None):
    """Return [(subject_id, video_path, gt_path), ...] for available subjects."""
    root = root or project_path(_U["root"])
    out = []
    for vid in glob(os.path.join(root, "**", "vid.avi"), recursive=True):
        d = os.path.dirname(vid)
        gt = os.path.join(d, "ground_truth.txt")
        if not os.path.exists(gt):
            cand = glob(os.path.join(d, "*.txt"))
            gt = cand[0] if cand else None
        if gt:
            out.append((os.path.basename(d), vid, gt))
    return sorted(out)


def load_ground_truth(gt_path: str):
    """Return (ppg_trace, hr_series, fs). ground_truth.txt = 3 whitespace rows."""
    rows = []
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            vals = [float(x) for x in line.replace(",", " ").split()]
            if vals:
                rows.append(np.array(vals))
    if not rows:
        raise ValueError(f"empty ground truth {gt_path}")
    ppg = rows[0]
    hr = rows[1] if len(rows) > 1 else np.array([])
    times = rows[2] if len(rows) > 2 else None
    if times is not None and len(times) > 1:
        fs = 1.0 / np.median(np.diff(times))
    else:
        fs = float(_U["gt_fs"])
    return ppg, hr, float(fs)


def read_video(video_path: str, max_seconds: float = None):
    """Return (frames RGB (T,H,W,3), fps). max_seconds caps length for speed."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or _U["video_fps"]
    max_frames = int(max_seconds * fps) if max_seconds else None
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames), float(fps)


if __name__ == "__main__":
    subs = list_subjects()
    print(f"UBFC-rPPG subjects found: {len(subs)}")
    for sid, vid, gt in subs[:5]:
        print(f"  {sid}: {os.path.basename(vid)} + {os.path.basename(gt)}")
    if not subs:
        print("None yet. Video download likely blocked by Google Drive quota.")
        print("Fallback: Kaggle mirror — see data/fetch_ubfc_rppg_kaggle.md")
