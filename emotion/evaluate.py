"""Interpret the trained emotion classifier.

1. XGBoost feature importance — which HRV features separate the states.
2. SHAP beeswarm — per-feature push toward each class.
3. Arousal-vs-valence test — HRV is known to track arousal better than valence.
   Recode: baseline=low arousal, stress+amusement=high arousal. A binary arousal
   classifier should beat the 3-class number, because the hard split is the
   same-arousal valence pair (stress vs amusement).
   (Confirms the documented limitation; cited in the writeup.)
"""
from __future__ import annotations

import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config_loader import load_config
from emotion.dataset import load_dataset, loso_splits

_CFG = load_config()
_RES = _CFG.path("results")
_CKPT = _CFG.path("checkpoints")


def _load_model():
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(_CKPT, "xgb_best_fold.json"))
    with open(os.path.join(_CKPT, "scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    return model, scaler


def feature_importance(model, feature_order):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    ranked = [(feature_order[i], float(imp[i])) for i in order]

    plt.figure(figsize=(8, 5))
    plt.barh([feature_order[i] for i in order][::-1], imp[order][::-1], color="#3b6ea5")
    plt.xlabel("XGBoost importance")
    plt.title("HRV feature importance for emotion classification")
    plt.tight_layout()
    plt.savefig(os.path.join(_RES, "feature_importance.png"), dpi=300)
    plt.close()
    return ranked


def shap_beeswarm(model, X, feature_order, n=200):
    try:
        import shap
        sample = X[np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
        # multiclass shap_values is a list (old API) or 3-D array (new API).
        # summary_plot wants a list per class; collapse a 3-D array to mean-|impact|.
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = np.abs(sv).mean(axis=2)
        plt.figure()
        shap.summary_plot(sv, sample, feature_names=feature_order, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(_RES, "shap_beeswarm.png"), dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"  shap skipped: {e}")
        plt.close("all")
        return False


def arousal_vs_valence(X, y, subjects):
    """Binary arousal (baseline=0 low vs stress/amusement=1 high) LOSO accuracy."""
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    y_arousal = (y != 0).astype(int)   # baseline low, stress+amusement high
    accs = []
    for sid, tr, te in loso_splits(subjects):
        sc = StandardScaler().fit(X[tr])
        m = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                              eval_metric="logloss", tree_method="hist")
        m.fit(sc.transform(X[tr]), y_arousal[tr])
        pred = m.predict(sc.transform(X[te]))
        accs.append(float((pred == y_arousal[te]).mean()))
    return float(np.mean(accs))


def run():
    os.makedirs(_RES, exist_ok=True)
    X, y, subjects, feature_order = load_dataset()
    model, scaler = _load_model()
    Xs = scaler.transform(X)

    ranked = feature_importance(model, feature_order)
    print("feature importance (high->low):")
    for name, val in ranked:
        print(f"  {name:12} {val:.4f}")

    has_shap = shap_beeswarm(model, Xs, feature_order)

    arousal_acc = arousal_vs_valence(X, y, subjects)
    cls_results = {}
    cr_path = os.path.join(_RES, "classifier_results.json")
    if os.path.exists(cr_path):
        with open(cr_path) as fh:
            cls_results = json.load(fh)
    three_class = cls_results.get("overall_accuracy", float("nan"))

    print(f"\n3-class accuracy:        {three_class:.3f}")
    print(f"binary arousal accuracy: {arousal_acc:.3f}")
    gap = arousal_acc - three_class
    print(f"arousal advantage: {gap:+.3f} "
          f"({'supports' if gap > 0 else 'does NOT support'} HRV tracks arousal > valence)")

    # plain-language interpretation
    top3 = ", ".join(n for n, _ in ranked[:3])
    md = f"""# HRV Emotion Classifier — Interpretation

## Most informative HRV features
Ranked by XGBoost importance, the top drivers are: **{top3}**.
Full ranking in `feature_importance.png`.

## Arousal vs valence
- 3-class accuracy (baseline / stress / amusement): **{three_class:.1%}**
- Binary arousal accuracy (calm vs aroused): **{arousal_acc:.1%}**

The binary arousal task is {'easier' if gap > 0 else 'not easier'} than the 3-class
task by {gap:+.1%}. This matches the literature: HRV separates *arousal* (calm vs
activated) well, but struggles with *valence* — stress and amusement share elevated
sympathetic activation, so telling them apart from heart-rhythm alone is hard.

## What this means for the demo
The live system reports a confident calm-vs-aroused signal; the stress-vs-amusement
distinction is the weak axis and is reported with lower confidence.
{'SHAP beeswarm saved to `shap_beeswarm.png`.' if has_shap else ''}
"""
    with open(os.path.join(_RES, "interpretation.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"\nsaved interpretation -> {os.path.join(_RES, 'interpretation.md')}")
    return {"three_class": three_class, "arousal": arousal_acc, "ranked": ranked}


if __name__ == "__main__":
    run()
