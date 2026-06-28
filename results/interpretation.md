# HRV Emotion Classifier — Interpretation

## Most informative HRV features
Ranked by XGBoost importance, the top drivers are: **Mean_HR, CVRR, SDNN**.
Full ranking in `feature_importance.png`.

## Arousal vs valence
- 3-class accuracy (baseline / stress / amusement): **57.8%**
- Binary arousal accuracy (calm vs aroused): **68.7%**

The binary arousal task is easier than the 3-class
task by +10.9%. This matches the literature: HRV separates *arousal* (calm vs
activated) well, but struggles with *valence* — stress and amusement share elevated
sympathetic activation, so telling them apart from heart-rhythm alone is hard.

## What this means for the demo
The live system reports a confident calm-vs-aroused signal; the stress-vs-amusement
distinction is the weak axis and is reported with lower confidence.
SHAP beeswarm saved to `shap_beeswarm.png`.
