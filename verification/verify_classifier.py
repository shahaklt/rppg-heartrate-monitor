"""Gate on classifier results. Reads results/classifier_results.json."""
from __future__ import annotations

import json
import os

import numpy as np

from config_loader import load_config

_CFG = load_config()


def run(min_acc: float = 0.55, min_stress_recall: float = 0.50):
    path = os.path.join(_CFG.path("results"), "classifier_results.json")
    if not os.path.exists(path):
        print("verify_classifier: no results yet — run emotion/train.py first")
        return None
    with open(path) as fh:
        r = json.load(fh)

    acc = r["overall_accuracy"]
    cm = np.array(r["confusion_matrix"])
    names = _CFG["dataset"]["task_names"]
    stress_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0

    print(f"overall accuracy: {acc:.3f}  (need > {min_acc})")
    print(f"stress recall:    {stress_recall:.3f}  (need > {min_stress_recall})")
    print("per-class F1:", r.get("per_class_f1"))
    print("confusion matrix:")
    print("        " + " ".join(f"{names[c]:>10}" for c in range(cm.shape[0])))
    for i in range(cm.shape[0]):
        print(f"{names[i]:>8} " + " ".join(f"{cm[i, j]:10d}" for j in range(cm.shape[1])))

    ok = acc > min_acc and stress_recall > min_stress_recall
    print(f"\nverify_classifier -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    run()
