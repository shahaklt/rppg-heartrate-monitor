# Webcam heart rate to emotion

A small project that reads your pulse straight from a webcam, with no watch and
nothing touching you, and takes a guess at whether you're calm or worked up. It
all runs locally.

The idea: every heartbeat moves blood through your face and changes your skin
colour by a tiny amount, way too small to see but enough for a camera to pick up.
Once you have the pulse, the spacing between beats (heart-rate variability) shifts
with how activated you are. Stress tightens it, calm loosens it. Put those two
together and you can pull a rough emotion read out of a plain video feed.

```
Webcam -> face ROI -> pulse (BVP) -> heartbeats -> HRV features -> emotion
          MediaPipe   CHROM          peak detect   11 numbers      XGBoost
```

None of the individual pieces are new: CHROM, heartpy, the usual HRV features, a
gradient-boosted classifier. The fun part was wiring them into one thing that runs
live, start to finish, and stays honest about bad input. When the signal is too
noisy it says so instead of inventing a number.

## Data

Trained on WESAD, which is 15 people wearing an Empatica E4 wrist sensor (BVP at
64 Hz) through a rest baseline, a stress task, and a funny-video block. HRV
features only care about the timing between beats, not where the pulse came from,
so a model trained on wrist data still works on a face pulse at run time, with
some expected drop from the change of source.

There's also a loader for UBFC-rPPG (webcam video plus a pulse-oximeter reference)
to check the camera-to-pulse stage on its own. Those video files sit behind a
flaky Google Drive quota, so that check isn't run yet. `data/fetch_ubfc_rppg_kaggle.md`
has the Kaggle mirror if you want to grab them.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m verification.setup_check
```

## Running it

```powershell
.\run.ps1 data       # download + build the HRV dataset from WESAD
.\run.ps1 train      # XGBoost, leave-one-subject-out
.\run.ps1 evaluate   # feature importance, SHAP, arousal vs valence
.\run.ps1 verify     # sanity checks across the whole pipeline
.\run.ps1 visuals    # figures into results/
.\run.ps1 web        # the browser demo at http://localhost:5000
.\run.ps1 demo       # Gradio version at http://localhost:7860
```

The web demo is the one to try. Open the page, hit start camera, and you'll see
the face-mesh dots tracking you, with the cheeks and forehead picked out because
that's where it reads the pulse from. A live heart rate shows up after about 10
seconds. The calm or aroused readout needs roughly a minute to fill its first
window. The browser only sends frames over, and all the actual processing happens
in the local Python server. Any stage also runs on its own, for example
`python -m emotion.train`.

## How well does it work

Leave-one-subject-out across the 15 people:

| | |
|---|---|
| 3-class (baseline / stress / amusement) | 57.8% (chance is 33%) |
| calm vs aroused (binary) | 68.7% |
| stress recall | 73% |
| per-class F1 | baseline 0.69, stress 0.64, amusement 0.20 |
| LF/HF rises under stress | 12 of 15 people |
| RMSSD, baseline to stress | 69 to 59 ms |

The gap between the binary and 3-class numbers is the whole story. Heart rhythm
tells you how activated someone is, not what they're feeling. Stress and amusement
both spike arousal, and amusement is mild enough to look a lot like baseline, so
the model mixes them up. That's why amusement is the weak class. It's a known limit
of HRV, not something broken here.

## Worth keeping in mind

- It reads arousal (calm vs activated) reasonably well. It can't reliably tell
  stress apart from amusement.
- The 60-second window means about a minute of warmup and lag, so it's live but
  not instant.
- It needs steady lighting and a fairly still head. Noisy windows get dropped
  rather than guessed at.
- Trained on 15 people's wrist recordings and run on your face, so treat it as a
  fun estimate, not a medical reading.

## Layout

```
data/         dataset download + builders
rppg/         face ROI, CHROM, EfficientPhys, signal quality, DSP
hrv/          peak detection, time/frequency HRV, feature pipeline
emotion/      dataset, training (XGBoost + MLP baseline), evaluation
demo/         webcam capture, realtime pipeline, web + Gradio apps
verification/ per-stage sanity checks
configs/      config.yaml
```

Everything lives in one folder. Delete it and it's all gone, venv and data
included.
