# rPPG Emotion Fingerprinting

Contactless emotion inference from a webcam, running locally. A face video is
turned into a pulse signal (rPPG), the pulse into heart-rate-variability (HRV)
features, and those features into an emotion estimate — all on-device.

```
Webcam ─▶ Face ROI ─▶ BVP ─▶ R-peaks ─▶ HRV features ─▶ Emotion
        MediaPipe   CHROM /   heartpy     11-D vector    XGBoost
                    EfficientPhys
```

## What's here vs. what's novel
Every stage reuses established methods (CHROM, EfficientPhys, heartpy, standard
HRV). The contribution is the **live, local, end-to-end** pipeline with **signal-
quality gating** (no predictions on bad signal) and an explicit **arousal-vs-
valence** evaluation.

## Data (both free, no IEEE)
- **WESAD** (emotion training): 15 subjects, Empatica E4 wrist **BVP @64 Hz**,
  conditions baseline / stress / amusement. Trains the HRV→emotion classifier.
- **UBFC-rPPG** (rPPG validation): webcam video + pulse-oximeter ground truth.
  Validates the camera→pulse stage (Pearson r, HR error). See
  `data/fetch_ubfc_rppg_kaggle.md` (Google Drive quota often blocks the videos;
  Kaggle mirror is the reliable route).

HRV features (RR intervals) are sensor-agnostic, so a classifier trained on wrist
BVP transfers to webcam rPPG at inference — with a stated domain-shift caveat.

> Note on labels: UBFC-Phys (neutral/stress/amusement) is IEEE-gated, so this
> project uses WESAD instead; classes are **baseline / stress / amusement**.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m verification.setup_check
```

## Run
```powershell
.\run.ps1 data       # download+extract WESAD, build data/hrv_dataset.npz
.\run.ps1 train      # XGBoost LOSO-CV -> checkpoints/ + results/
.\run.ps1 evaluate   # importance, SHAP, arousal-vs-valence
.\run.ps1 verify     # full accuracy gate (READY / FAILED)
.\run.ps1 visuals    # 300-DPI figures in results/
.\run.ps1 web        # browser webcam demo with live face dots -> http://localhost:5000
.\run.ps1 demo       # live webcam Gradio app on localhost:7860
```
The **web** demo (`demo/web_app.py`) is the simplest one to show: open the page,
click *start camera*, and it overlays the face-mesh dots (cheek/forehead ROIs
highlighted), shows a live heart rate within ~10 s, and reports calm/aroused once
the 60 s window fills. The browser only captures frames; all rPPG/HRV/emotion runs
in the local Python server.
Single module runs also work, e.g. `python -m emotion.train`.

## Measured results (15-subject LOSO, this build)
| Metric | Value |
|---|---|
| 3-class accuracy (baseline/stress/amusement) | **57.8%** (chance 33%) |
| Binary arousal (calm vs aroused) | **68.7%** |
| Stress recall | **73%** |
| Per-class F1 | baseline 0.69 · stress 0.64 · amusement 0.20 |
| HRV validity: LF/HF rises under stress | **12/15 subjects (80%)** |
| RMSSD baseline→stress | 69→59 ms (drops, as expected) |
| Live latency / window | ~80 ms (≪ 2 s update step) |

Arousal beats 3-class by +10.9 pts — HRV reads arousal, not valence (amusement,
mild-arousal, is the weak class). rPPG-vs-groundtruth is the one unrun stage
(UBFC-rPPG videos blocked by a Drive quota); its harness is wired and ready.

## Results files
Generated into `results/` after `train`/`evaluate`/`visuals`:
- `classifier_results.json` — LOSO accuracy, per-class F1, confusion matrix
- `confusion_matrix.png`, `feature_importance.png`, `shap_beeswarm.png`
- `hrv_comparison.png` — LF/HF per state (stress elevates it)
- `interpretation.md` — plain-language findings
- `bvp_vs_groundtruth.png` — extracted vs reference pulse (UBFC-rPPG)

## Limitations (not optional)
- HRV reads **arousal** (calm vs activated) well; **valence** (stress vs amusement)
  is weak — they share sympathetic activation. The evaluation quantifies this.
- 60-second window → ~60 s warmup and ~60 s lag; not strictly instantaneous.
- Needs steady lighting and a mostly still head; low-quality windows are dropped.
- Not medical-grade. Trained on 15 WESAD subjects; expect domain shift to webcam.

## What you can / can't claim
CAN: local contactless emotion pipeline; combined rPPG+HRV; quality gating;
explicit arousal/valence separation.
CAN'T: reliable calm-vs-amused; beating published methods; real-time in the strict
sense; clinical accuracy.

## Layout
```
data/        download + dataset builders (WESAD, UBFC-rPPG)
rppg/        face ROI, CHROM, EfficientPhys, signal quality, DSP
hrv/         peak detection, time/frequency HRV, feature pipeline
emotion/     dataset, train (XGBoost), MLP baseline, evaluate
demo/        webcam capture, realtime pipeline, Gradio app, visuals
verification/setup + per-stage checks + full accuracy gate
configs/     config.yaml (single source of truth)
```

Delete the whole `rppg-emotion/` folder to remove everything (venv, data, models).
