"""HRV feature pipeline: RR series / video -> 11-D feature matrix -> dataset.

Feature order (matches configs/config.yaml hrv.feature_order):
  [SDNN, RMSSD, pNN50, CVRR, Mean_HR, LF, HF, LF_HF, Total_power, LF_norm, HF_norm]
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from config_loader import load_config, project_path
from hrv import time_domain, frequency_domain
from hrv.peak_detection import detect_peaks
from rppg.dsp import bandpass_filter
from rppg.signal_quality import assess

_CFG = load_config()
_HRV = _CFG["hrv"]
FEATURE_ORDER = list(_HRV["feature_order"])
_TD_KEYS = set(time_domain.KEYS)
_FD_KEYS = set(frequency_domain.KEYS)


class FeaturePipeline:
    def __init__(self, window_s: float = None, step_s: float = None):
        self.window_s = window_s or _HRV["window_seconds"]
        self.step_s = step_s or _HRV["step_seconds"]

    def extract(self, rr_ms: np.ndarray, fps_bvp: float = None) -> np.ndarray:
        """RR series -> (n_windows, 11) feature matrix."""
        td = time_domain.windowed(rr_ms, self.window_s, self.step_s)
        fd = frequency_domain.windowed(rr_ms, self.window_s, self.step_s)
        n = min(len(next(iter(td.values()))), len(next(iter(fd.values()))))
        if n == 0:
            return np.empty((0, len(FEATURE_ORDER)))
        rows = []
        for i in range(n):
            row = []
            for feat in FEATURE_ORDER:
                if feat in _TD_KEYS:
                    row.append(td[feat][i])
                else:
                    row.append(fd[feat][i])
            rows.append(row)
        mat = np.array(rows, dtype=np.float64)
        return _sanitize(mat)

    def extract_from_bvp(self, bvp: np.ndarray, fs: float,
                         min_beats: int = None,
                         min_snr_db: float = None) -> np.ndarray:
        """Slide a window over the BVP *signal* (not a precomputed RR series),
        peak-detect each window independently -> (n_windows, 11) matrix.

        Detecting peaks per 60 s window (rather than once over a 20-min record)
        keeps RR estimation local, so heart-rate drift across a long segment
        can't smear a single global peak threshold over every window. It is also
        exactly what the live demo does per window, giving training/inference
        parity. `min_snr_db` is optional per-window quality control; WESAD
        training leaves it off (contact reference BVP, maximise data) and relies
        on `min_beats` + RR artifact rejection to drop collapsed windows.
        """
        sig = bandpass_filter(np.asarray(bvp, dtype=np.float64), fs)
        w = int(round(self.window_s * fs))
        s = int(round(self.step_s * fs))
        if w <= 0 or s <= 0 or sig.size < w:
            return np.empty((0, len(FEATURE_ORDER)))
        if min_beats is None:
            min_beats = max(4, int(self.window_s * 0.4))   # 60 s -> >=24 beats
        rows = []
        for start in range(0, sig.size - w + 1, s):
            seg = sig[start:start + w]
            if min_snr_db is not None:
                from rppg.signal_quality import _snr_db
                if _snr_db(seg, fs)[0] < min_snr_db:
                    continue
            rr = detect_peaks(seg, fs, prefilter=False)["rr_ms"]
            if rr.size < min_beats:
                continue
            td = time_domain.compute(rr)
            fd = frequency_domain.compute(rr)
            row = [td[f] if f in _TD_KEYS else fd[f] for f in FEATURE_ORDER]
            rows.append(row)
        if not rows:
            return np.empty((0, len(FEATURE_ORDER)))
        return _sanitize(np.array(rows, dtype=np.float64))

    # ---- per-subject extraction ------------------------------------------
    def _bvp_for_task(self, subj_dir: str, task: str, extractor: str):
        """Return (bvp, fps) for one task file (T1/T2/T3).

        extractor:
          'ground_truth' -> read T{k}.csv (Empatica E4 reference BVP)
          'chrom'/'efficientphys' -> run rPPG on the T{k}.avi frames
        """
        if extractor == "ground_truth":
            return _load_bvp_csv(os.path.join(subj_dir, f"{task}.csv"))
        frames, vfps = _read_video(os.path.join(subj_dir, f"{task}.avi"))
        from rppg.face_roi import FaceROIExtractor
        roi = FaceROIExtractor().process_video_frames(frames)
        names = [k for k in roi if k != "failed_frames"]
        trace = np.nanmean(np.stack([roi[n] for n in names]), axis=0)
        if extractor == "efficientphys":
            from rppg.efficientphys_extractor import EfficientPhysExtractor
            bvp = EfficientPhysExtractor().extract(frames, vfps)
        else:
            from rppg.chrom_extractor import CHROMExtractor
            bvp = CHROMExtractor().extract(trace, vfps)
        return bvp, vfps

    def extract_from_video(self, subj_dir: str, label_path: str = None,
                           extractor: str = "ground_truth") -> List[Dict]:
        """Run the pipeline for one subject dir -> list of per-task records.

        (Per-task UBFC-Phys path; the active WESAD pipeline uses extract_from_bvp.)
        """
        labels = _CFG["dataset"].get("labels", {"T1": 0, "T2": 1, "T3": 2})
        subject_id = os.path.basename(subj_dir.rstrip("/\\"))
        records = []
        for task, label in labels.items():
            need = f"{task}.csv" if extractor == "ground_truth" else f"{task}.avi"
            if not os.path.exists(os.path.join(subj_dir, need)):
                continue
            bvp, fps = self._bvp_for_task(subj_dir, task, extractor)
            q = assess(bvp, fps)
            if q["quality"] == "LOW":
                records.append({"features": np.empty((0, len(FEATURE_ORDER))),
                                "label": label, "quality": "LOW", "task": task,
                                "subject_id": subject_id, "reason": q["reason"]})
                continue
            rr = detect_peaks(bvp, fps)["rr_ms"]
            feats = self.extract(rr, fps)
            records.append({"features": feats, "label": label, "task": task,
                            "quality": "HIGH", "subject_id": subject_id,
                            "reason": "ok"})
        return records

    def build_dataset(self, subject_ids: List[str], data_root: str = None,
                      extractor: str = "ground_truth",
                      save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        data_root = data_root or _CFG.path("data_root")
        save_path = save_path or _CFG.path("hrv_dataset")
        X_parts, y_parts, subj_parts = [], [], []
        for sid in subject_ids:
            subj_dir = os.path.join(data_root, sid)
            if not os.path.isdir(subj_dir):
                print(f"  skip {sid}: missing dir")
                continue
            for rec in self.extract_from_video(subj_dir, extractor=extractor):
                if rec["quality"] == "LOW" or rec["features"].shape[0] == 0:
                    continue
                X_parts.append(rec["features"])
                y_parts.append(np.full(rec["features"].shape[0], rec["label"]))
                subj_parts.append(np.full(rec["features"].shape[0], sid, dtype=object))
        if not X_parts:
            raise RuntimeError("No usable windows extracted — check data/extractor")
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        subjects = np.concatenate(subj_parts)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, X=X, y=y, subjects=subjects,
                 feature_order=np.array(FEATURE_ORDER, dtype=object))
        _print_balance(y)
        print(f"  saved {X.shape[0]} windows x {X.shape[1]} feats -> {save_path}")
        return X, y


# ---- helpers -------------------------------------------------------------
def _sanitize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mat[~np.isfinite(mat)] = 0.0
    return mat


def _load_bvp_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    bvp = df[cols.get("bvp", df.columns[-1])].to_numpy(dtype=np.float64)
    if "time" in cols:
        t = df[cols["time"]].to_numpy(dtype=np.float64)
        fps = 1.0 / np.median(np.diff(t)) if len(t) > 1 else _CFG["dataset"]["bvp_fs"]
    else:
        fps = _CFG["dataset"]["bvp_fs"]
    return bvp, float(fps)


def _read_video(path: str, max_frames: int = None) -> Tuple[np.ndarray, float]:
    """Read a whole task video -> (frames RGB (T,H,W,3), fps)."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or _CFG["dataset"]["video_fps"]
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames), float(fps)


def _print_balance(y: np.ndarray):
    names = load_config()["dataset"]["task_names"]
    print("  class balance:")
    for cls in sorted(np.unique(y)):
        print(f"    {int(cls)} ({names.get(int(cls), '?')}): {int((y == cls).sum())}")


def _self_test():
    from rppg.dsp import synthetic_bvp
    fps = 64.0
    bvp = synthetic_bvp(300, fps, hr_bpm=72, hrv_ms=35, seed=9)
    rr = detect_peaks(bvp, fps)["rr_ms"]
    feats = FeaturePipeline().extract(rr, fps)
    ok = (feats.ndim == 2 and feats.shape[1] == 11
          and np.isfinite(feats).all() and feats.shape[0] >= 1)
    print(f"[feature_pipeline self-test] matrix={feats.shape} finite={np.isfinite(feats).all()} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _self_test()
