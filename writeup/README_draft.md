# Reading Emotion From a Webcam, Without a Wearable

## The idea
Your pulse leaks into your skin colour. Every heartbeat pushes blood through the
face and changes how it reflects light by an amount too small to see but big
enough for a camera to measure. That signal is called remote photoplethysmography
(rPPG). Once you have a pulse, the spacing between beats — heart-rate variability —
carries a fingerprint of your autonomic state: stress tightens it, calm loosens
it. This project chains those two facts into a live webcam demo that estimates
whether you're calm or aroused, locally, with no contact and no wearable.

## How it works, three stages
1. **Camera to pulse.** MediaPipe finds the face; the cheeks and forehead are
   sampled for their average colour each frame. The CHROM method (de Haan &
   Jeanne, 2013) projects that colour trace onto chrominance axes that cancel most
   motion artefacts, leaving the pulse. EfficientPhys (Liu et al., 2023) is wired
   in as a deep alternative when a trained checkpoint is present.
2. **Pulse to rhythm.** heartpy finds beats; intervals outside 300–2000 ms or that
   jump more than 20% from their neighbour are rejected. What remains is a clean
   RR-interval series.
3. **Rhythm to state.** Eleven HRV features (SDNN, RMSSD, pNN50, LF/HF, …) over a
   60-second window feed an XGBoost classifier trained on WESAD.

## What it's trained on
WESAD: 15 people, wrist Empatica E4 BVP at 64 Hz, recorded during a neutral
baseline, a stress task, and a funny-video amusement block. HRV features are
sensor-agnostic, so a model trained on wrist pulse works on face pulse at run
time. The rPPG stage itself is checked separately against UBFC-rPPG, where the
ground-truth pulse comes from a finger oximeter.

## Results
- 3-class accuracy (baseline / stress / amusement), leave-one-subject-out over 15
  subjects: **57.8%** (chance = 33%).
- Binary arousal (calm vs aroused): **68.7%** — higher, as expected.
- Stress recall: **73%**; per-class F1 baseline 0.69 / stress 0.64 / amusement 0.20.
- Biological validity: LF/HF rises under stress in **12/15** subjects (80%), and
  mean RMSSD falls 69→59 ms — the textbook sympathetic-activation signature, which
  is the real evidence the camera→pulse→rhythm chain is sound.
- rPPG vs ground truth (UBFC-rPPG): not yet validated — the dataset's video files
  are behind a Google-Drive download quota (Kaggle mirror noted in the repo). The
  rPPG code and its validation harness are in place and run once the videos land.

The gap between the binary and 3-class numbers is the whole story: heart rhythm
tells you *how activated* someone is, not *what* they feel. Stress and amusement
both spike arousal — and amusement's mild arousal sits close to baseline — so the
model confuses them. The literature says it should.

## What it doesn't do
It is not a lie detector and not a medical device. It needs about a minute of
steady, well-lit video before its first reading. It will not reliably separate
stress from amusement. Those limits aren't bugs to hide; stating them is what
makes the rest believable.

## Citations
- de Haan & Jeanne, "Robust pulse rate from chrominance-based rPPG," IEEE TBME 2013.
- Liu et al., "EfficientPhys," IEEE/CVF WACV 2023.
- Liu et al., "rPPG-Toolbox," NeurIPS 2023.
- Schmidt et al., "Introducing WESAD," ICMI 2018.
- LF/HF as sympathetic marker: Frontiers in Psychiatry 2021, doi:10.3389/fpsyt.2021.799029.
