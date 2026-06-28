"""Build the HRV emotion dataset from WESAD.

WESAD ships one pickle per subject: WESAD/S{n}/S{n}.pkl with
  data['signal']['wrist']['BVP']  -> Empatica E4 BVP @ 64 Hz, shape (M,1)
  data['label']                   -> protocol label @ 700 Hz, shape (N,)
Label codes: 1 baseline, 2 stress, 3 amusement, 4 meditation, others transient.

For each subject and each kept condition we slice the wrist BVP for that
condition's time span, then slide 60 s HRV windows over the *signal*, detecting
pulse peaks per window (FeaturePipeline.extract_from_bvp). The Empatica E4 BVP is
a contact reference, so we do NOT apply the rPPG signal-quality gate here (that
gate is for noisy webcam rPPG in the live demo); windows are kept whenever enough
beats survive RR artifact rejection. Output is data/hrv_dataset.npz
(X, y, subjects) — the same artifact emotion/train.py consumes.

Run:
    python -m data.load_wesad            # extracts zip if needed, builds dataset
"""
from __future__ import annotations

import os
import pickle
import zipfile
from glob import glob

import numpy as np

from config_loader import load_config, project_path
from hrv.feature_pipeline import FEATURE_ORDER, FeaturePipeline

_CFG = load_config()
_W = _CFG["wesad"]
_BVP_FS = _W["bvp_fs"]
_LABEL_FS = _W["label_fs"]
_LABEL_MAP = {int(k): int(v) for k, v in _W["label_map"].items()}


def _ensure_extracted():
    root = project_path(_W["root"])
    if glob(os.path.join(root, "**", "S*.pkl"), recursive=True):
        return root
    zip_path = project_path(_W["raw_zip"])
    if not os.path.exists(zip_path):
        raise SystemExit(f"WESAD zip not found at {zip_path}. Download still running?")
    os.makedirs(root, exist_ok=True)
    print(f"Extracting {zip_path} -> {root} (large, one-time)...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return root


def _load_pkl(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh, encoding="latin1")


def _contiguous_runs(mask: np.ndarray):
    """Yield (start, end) index pairs for each contiguous True run in mask."""
    if not mask.any():
        return
    idx = np.where(mask)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    for s, e in zip(starts, ends):
        yield int(s), int(e)


def _subject_segments(data: dict):
    """Yield (class_label, bvp_segment) for each kept condition run."""
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float64).ravel()
    label = np.asarray(data["label"]).ravel().astype(int)
    for code, cls in _LABEL_MAP.items():
        for s, e in _contiguous_runs(label == code):
            t0, t1 = s / _LABEL_FS, e / _LABEL_FS
            i0, i1 = int(t0 * _BVP_FS), int(t1 * _BVP_FS)
            seg = bvp[i0:i1]
            if (i1 - i0) / _BVP_FS >= _CFG["dataset"]["min_segment_seconds"]:
                yield cls, seg


def build(max_per_class: int = None, save: bool = True):
    root = _ensure_extracted()
    pkls = sorted(glob(os.path.join(root, "**", "S*.pkl"), recursive=True))
    if not pkls:
        raise SystemExit(f"No S*.pkl under {root}")
    print(f"Found {len(pkls)} WESAD subjects")

    pipe = FeaturePipeline()
    X_parts, y_parts, subj_parts = [], [], []
    for pkl in pkls:
        sid = os.path.splitext(os.path.basename(pkl))[0]   # 'S2'
        data = _load_pkl(pkl)
        n_win = 0
        for cls, seg in _subject_segments(data):
            feats = pipe.extract_from_bvp(seg, _BVP_FS)
            if feats.shape[0] == 0:
                continue
            X_parts.append(feats)
            y_parts.append(np.full(feats.shape[0], cls))
            subj_parts.append(np.full(feats.shape[0], sid, dtype=object))
            n_win += feats.shape[0]
        print(f"  {sid}: {n_win} windows")

    if not X_parts:
        raise SystemExit("No usable windows extracted from WESAD — check data.")
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    subjects = np.concatenate(subj_parts)

    if max_per_class:
        X, y, subjects = _balance(X, y, subjects, max_per_class)

    names = _CFG["dataset"]["task_names"]
    print("\nclass balance:")
    for c in sorted(np.unique(y)):
        print(f"  {c} ({names[int(c)]}): {int((y == c).sum())}")
    print(f"total: {X.shape[0]} windows x {X.shape[1]} features, "
          f"{len(np.unique(subjects))} subjects")

    if save:
        out = _CFG.path("hrv_dataset")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, X=X, y=y, subjects=subjects,
                 feature_order=np.array(FEATURE_ORDER, dtype=object))
        print(f"saved -> {out}")
    return X, y, subjects


def _balance(X, y, subjects, cap):
    keep = []
    rng = np.random.default_rng(0)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], subjects[keep]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap windows per class to reduce imbalance")
    args = ap.parse_args()
    build(max_per_class=args.max_per_class)
