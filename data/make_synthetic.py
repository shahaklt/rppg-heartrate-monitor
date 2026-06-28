"""Generate a large UBFC-Phys-shaped synthetic cohort.

UBFC-Phys requires manual registration, so for development / maximum-data training
we fabricate subjects with *emotion-dependent* physiology that mirrors the real
findings:

  T1 neutral  : lower HR, high HRV (RMSSD), low  LF/HF  (parasympathetic)
  T2 stress   : higher HR, low  HRV,        high LF/HF  (sympathetic dominance)
  T3 amusement: elevated HR, mid HRV,       mid  LF/HF  (high arousal, overlaps T2)

The T2/T3 overlap is intentional — it reproduces the documented arousal-vs-valence
confusion (HRV separates calm-vs-aroused well, stress-vs-amused poorly).

Each subject gets bvp.csv (always) and, for the first --videos subjects, a small
pulsing vid.avi so the rPPG (CHROM/EfficientPhys) path is also exercisable.
Everything is written under data/ubfc_phys so the whole project is one deletable
folder.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from config_loader import load_config, project_path
from rppg.dsp import synthetic_bvp

_CFG = load_config()
_BVP_FS = _CFG["dataset"]["bvp_fs"]

# per-emotion physiology priors: (HR mean, HR sd, HRV ms mean, HRV sd)
EMOTION_PRIORS = {
    "T1": dict(hr=68, hr_sd=5, hrv=48, hrv_sd=8),    # calm / neutral
    "T2": dict(hr=90, hr_sd=7, hrv=20, hrv_sd=5),    # stress
    "T3": dict(hr=82, hr_sd=6, hrv=30, hrv_sd=6),    # amusement
}


def _subject_signature(rng):
    """Stable per-subject offsets so LOSO-CV is non-trivial."""
    return dict(hr_bias=rng.normal(0, 4), hrv_bias=rng.normal(0, 5))


def _gen_segment(task, seg_seconds, sig, rng):
    p = EMOTION_PRIORS[task]
    hr = max(45.0, rng.normal(p["hr"], p["hr_sd"]) + sig["hr_bias"])
    hrv = max(5.0, rng.normal(p["hrv"], p["hrv_sd"]) + sig["hrv_bias"])
    noise = rng.uniform(0.03, 0.08)
    bvp = synthetic_bvp(seg_seconds, _BVP_FS, hr_bpm=hr, hrv_ms=hrv,
                        noise=noise, seed=int(rng.integers(1e9)))
    return bvp, hr, hrv


def _write_video(path, bvp, fps, size=48):
    """Tiny pulsing 'face' video whose green channel tracks the BVP."""
    try:
        import cv2
    except Exception:
        return False
    T = len(bvp)
    # resample BVP to video fps
    vlen = int(T / _BVP_FS * fps)
    idx = np.linspace(0, T - 1, vlen).astype(int)
    pulse = bvp[idx]
    pulse = (pulse - pulse.min()) / (np.ptp(pulse) + 1e-9)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(path, fourcc, fps, (size, size))
    rng = np.random.default_rng(0)
    for p in pulse:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        # skin-ish base + green modulation by pulse
        frame[..., 2] = 150 + rng.integers(-3, 3, (size, size))   # R (BGR)
        frame[..., 1] = int(110 + 25 * p)                          # G tracks pulse
        frame[..., 0] = 100
        vw.write(frame)
    vw.release()
    return True


def generate(n_subjects: int, seg_seconds: int, n_videos: int,
             out_root: str = None, seed: int = 0):
    out_root = out_root or _CFG.path("data_root")
    os.makedirs(out_root, exist_ok=True)
    rng = np.random.default_rng(seed)
    video_fps = _CFG["dataset"]["video_fps"]
    manifest = {"source": "synthetic", "subjects": []}

    for n in range(1, n_subjects + 1):
        sid = f"subj_{n}"
        subj_dir = os.path.join(out_root, sid)
        os.makedirs(subj_dir, exist_ok=True)
        sig = _subject_signature(rng)

        all_bvp, labels, t_cursor, fps = [], [], 0.0, _BVP_FS
        for task in ("T1", "T2", "T3"):
            bvp, hr, hrv = _gen_segment(task, seg_seconds, sig, rng)
            start = t_cursor
            end = t_cursor + seg_seconds
            labels.append((task, start, end))
            all_bvp.append(bvp)
            t_cursor = end

        full = np.concatenate(all_bvp)
        times = np.arange(full.size) / fps
        # bvp.csv
        import pandas as pd
        pd.DataFrame({"time": times, "BVP": full}).to_csv(
            os.path.join(subj_dir, "bvp.csv"), index=False)
        # label.txt
        with open(os.path.join(subj_dir, "label.txt"), "w", encoding="utf-8") as fh:
            for task, s, e in labels:
                fh.write(f"{task} {s:.2f} {e:.2f}\n")

        has_video = False
        if n <= n_videos:
            has_video = _write_video(
                os.path.join(subj_dir, "vid.avi"), full, video_fps)

        manifest["subjects"].append({
            "id": sid, "bvp_len": int(full.size), "fps": fps,
            "segments": [{"task": t, "start": s, "end": e} for t, s, e in labels],
            "has_video": has_video,
        })
        if n % 20 == 0:
            print(f"  generated {n}/{n_subjects} subjects")

    man_path = _CFG.path("manifest")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    total_windows_est = n_subjects * 3 * (
        (seg_seconds - _CFG["hrv"]["window_seconds"]) // _CFG["hrv"]["step_seconds"] + 1)
    print(f"Done. {n_subjects} subjects, ~{total_windows_est} HRV windows expected.")
    print(f"Manifest: {man_path}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=120,
                    help="number of synthetic subjects")
    ap.add_argument("--seg-seconds", type=int, default=300,
                    help="duration of each T1/T2/T3 segment")
    ap.add_argument("--videos", type=int, default=4,
                    help="how many subjects also get a pulsing vid.avi")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.subjects, args.seg_seconds, args.videos, seed=args.seed)
