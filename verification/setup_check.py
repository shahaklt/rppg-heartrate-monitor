"""Phase 0 environment check. Prints a PASS/FAIL table and logs to logs/setup_check.txt.

Webcam is treated as a WARN (not FAIL) when absent — the training pipeline runs
headless; only the live demo needs a camera.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from config_loader import project_path

CHECKS = []


def record(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    flag = "PASS" if ok else ("WARN" if ok is None else "FAIL")
    print(f"[{flag}] {name}: {detail}")


def check_imports():
    mods = ["numpy", "scipy", "heartpy", "sklearn", "xgboost", "matplotlib",
            "pandas", "yaml", "h5py", "antropy", "neurokit2", "cv2",
            "mediapipe", "gradio", "torch"]
    missing = []
    for m in mods:
        try:
            __import__(m)
        except Exception as e:
            missing.append(f"{m}({type(e).__name__})")
    record("imports", not missing, "all ok" if not missing else f"missing: {missing}")


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            record("cuda", True, f"{name}, {vram:.1f} GB")
        else:
            record("cuda", None, "no CUDA — CPU fallback (fine for this project)")
    except Exception as e:
        record("cuda", None, f"torch error: {e}")


def check_webcam():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            record("webcam", None, "no camera (only needed for live demo)")
            return
        shapes = []
        for _ in range(5):
            ok, fr = cap.read()
            if ok:
                shapes.append(fr.shape)
        cap.release()
        ok = len(shapes) == 5 and all(len(s) == 3 and s[2] == 3 for s in shapes)
        record("webcam", ok if ok else None,
               f"captured {len(shapes)} frames, shape {shapes[0] if shapes else 'n/a'}")
    except Exception as e:
        record("webcam", None, f"camera error: {e}")


def check_mediapipe():
    try:
        from rppg.face_roi import FaceROIExtractor
        img = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
        ext = FaceROIExtractor()
        ext.process_frame(img)   # downloads model once, runs a real detection pass
        ext.close()
        record("mediapipe", True, "FaceLandmarker forward pass ran")
    except Exception as e:
        record("mediapipe", False, f"error: {e}")


def main():
    print("=== Phase 0 setup check ===")
    check_imports()
    check_cuda()
    check_mediapipe()
    check_webcam()

    os.makedirs(project_path("logs"), exist_ok=True)
    log = project_path("logs", "setup_check.txt")
    with open(log, "w", encoding="utf-8") as fh:
        for name, ok, detail in CHECKS:
            flag = "PASS" if ok else ("WARN" if ok is None else "FAIL")
            fh.write(f"[{flag}] {name}: {detail}\n")

    hard_fail = [n for n, ok, _ in CHECKS if ok is False]
    print(f"\nlog -> {log}")
    if hard_fail:
        print(f"FAILED: {hard_fail} — fix before Phase 1")
        sys.exit(1)
    print("setup OK (warnings are non-blocking)")


if __name__ == "__main__":
    main()
