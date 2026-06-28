"""Content-ready figures (300 DPI) saved to results/.

  pipeline_diagram.png    stage flow
  hrv_comparison.png      LF/HF per emotion class (shows stress elevates LF/HF)
  confusion_matrix.png    annotated, percent labels
  bvp_vs_groundtruth.png  extracted vs reference pulse (UBFC-rPPG if present)
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config_loader import load_config

_CFG = load_config()
_RES = _CFG.path("results")
_NAMES = _CFG["dataset"]["task_names"]


def pipeline_diagram():
    stages = ["Webcam", "Face ROI\n(MediaPipe)", "BVP\n(CHROM /\nEfficientPhys)",
              "R-peaks\n(heartpy)", "HRV features\n(11-D)", "Emotion\n(XGBoost)"]
    fig, ax = plt.subplots(figsize=(12, 2.4))
    x = 0
    for i, s in enumerate(stages):
        ax.add_patch(plt.Rectangle((x, 0), 1.6, 1, fc="#eaf2f8", ec="#2e86c1", lw=1.5))
        ax.text(x + 0.8, 0.5, s, ha="center", va="center", fontsize=9)
        if i < len(stages) - 1:
            ax.annotate("", (x + 1.9, 0.5), (x + 1.6, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#555"))
        x += 1.9
    ax.set_xlim(-0.1, x); ax.set_ylim(-0.1, 1.1); ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(_RES, "pipeline_diagram.png"), dpi=300)
    plt.close(fig)


def hrv_comparison():
    ds = _CFG.path("hrv_dataset")
    if not os.path.exists(ds):
        return
    d = np.load(ds, allow_pickle=True)
    X, y = d["X"], d["y"]
    order = list(d["feature_order"])
    lfhf = order.index("LF_HF")
    classes = sorted(np.unique(y))
    means = [np.nanmean(X[y == c, lfhf]) for c in classes]
    sems = [np.nanstd(X[y == c, lfhf]) / np.sqrt((y == c).sum()) for c in classes]
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#27ae60", "#c0392b", "#e67e22"]
    ax.bar([_NAMES[int(c)] for c in classes], means, yerr=sems,
           color=[colors[int(c) % 3] for c in classes], capsize=4)
    ax.set_ylabel("LF/HF ratio")
    ax.set_title("Sympathetic balance (LF/HF) by state\nhigher = more stress arousal")
    fig.tight_layout(); fig.savefig(os.path.join(_RES, "hrv_comparison.png"), dpi=300)
    plt.close(fig)


def confusion_matrix():
    path = os.path.join(_RES, "classifier_results.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        cm = np.array(json.load(fh)["confusion_matrix"], dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    labels = [_NAMES[i] for i in range(cm.shape[0])]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{pct[i, j]:.0f}%\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=9)
    fig.colorbar(im, label="% of true class")
    ax.set_title("Confusion matrix (LOSO-CV)")
    fig.tight_layout(); fig.savefig(os.path.join(_RES, "confusion_matrix.png"), dpi=300)
    plt.close(fig)


def bvp_vs_groundtruth():
    """UBFC-rPPG overlay if available; otherwise an illustrative synthetic overlay."""
    from rppg.chrom_extractor import CHROMExtractor
    from rppg.dsp import bandpass_filter
    from scipy.signal import resample
    from scipy.stats import pearsonr

    title = "Extracted vs reference BVP"
    try:
        from data.load_ubfc_rppg import list_subjects, load_ground_truth, read_video
        from rppg.face_roi import FaceROIExtractor
        subs = list_subjects()
        if subs:
            sid, vid, gt = subs[0]
            frames, fps = read_video(vid, max_seconds=12)
            roi = FaceROIExtractor().process_video_frames(frames)
            tr = np.nanmean(np.stack([roi[k] for k in roi if k != "failed_frames"]), axis=0)
            ext = bandpass_filter(CHROMExtractor().extract(tr, fps), fps)
            ppg, _, gtfs = load_ground_truth(gt)
            ref = bandpass_filter(resample(ppg[: int(12 * gtfs)], len(ext)), fps)
            title = f"{sid}: extracted vs ground truth"
        else:
            raise FileNotFoundError
    except Exception:
        from rppg.dsp import synthetic_bvp
        fps = 30.0
        truth = synthetic_bvp(12, fps, hr_bpm=72, noise=0.0)
        ext = bandpass_filter(truth + 0.4 * np.random.default_rng(0).standard_normal(len(truth)), fps)
        ref = bandpass_filter(truth, fps)
        title += " (illustrative synthetic)"

    seg = slice(0, int(10 * 30))
    a, b = ext[seg], ref[seg]
    r = pearsonr(a / (np.std(a) + 1e-9), b / (np.std(b) + 1e-9))[0]
    t = np.arange(len(a)) / 30.0
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, a / (np.std(a) + 1e-9), color="#2e86c1", label="extracted")
    ax.plot(t, b / (np.std(b) + 1e-9), color="#e67e22", label="reference", alpha=0.8)
    ax.set_title(f"{title}  (r={r:.2f})")
    ax.set_xlabel("s"); ax.set_yticks([]); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(_RES, "bvp_vs_groundtruth.png"), dpi=300)
    plt.close(fig)


def main():
    os.makedirs(_RES, exist_ok=True)
    pipeline_diagram(); print("  pipeline_diagram.png")
    hrv_comparison(); print("  hrv_comparison.png")
    confusion_matrix(); print("  confusion_matrix.png")
    bvp_vs_groundtruth(); print("  bvp_vs_groundtruth.png")
    print(f"figures -> {_RES}")


if __name__ == "__main__":
    main()
