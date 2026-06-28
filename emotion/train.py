"""Train an XGBoost emotion classifier on HRV features with LOSO-CV.

Scaler is fit on training folds only (no leakage). Saves the best fold model and
its scaler, plus full results json. Honest accuracy expectation on real UBFC-Phys:
~55-70% 3-class (HRV separates arousal well, valence poorly).
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

from config_loader import load_config
from emotion.dataset import class_balance, load_dataset, loso_splits

_CFG = load_config()
_XGB = _CFG["classifier"]["xgboost"]


def _confusion(y_true, y_pred, n=3):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


def _per_class_f1(cm):
    f1 = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return f1


def train(dataset_path: str = None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    import xgboost as xgb

    X, y, subjects, feature_order = load_dataset(dataset_path)
    print(f"Loaded X={X.shape}, balance={class_balance(y)}")
    n_classes = len(np.unique(y))

    fold_results = []
    all_true, all_pred = [], []
    best_acc, best_model, best_scaler = -1.0, None, None

    for sid, tr, te in loso_splits(subjects):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

        model = xgb.XGBClassifier(
            n_estimators=_XGB["n_estimators"], max_depth=_XGB["max_depth"],
            learning_rate=_XGB["learning_rate"], subsample=_XGB["subsample"],
            colsample_bytree=_XGB["colsample_bytree"],
            eval_metric=_XGB["eval_metric"], tree_method="hist",
            num_class=n_classes, objective="multi:softprob")
        # balanced weights (computed on the training fold only — no leakage) so the
        # 64%-baseline majority can't drown stress/amusement recall.
        sw = compute_sample_weight("balanced", y[tr])
        model.fit(Xtr, y[tr], sample_weight=sw)
        pred = model.predict(Xte)
        acc = float((pred == y[te]).mean())
        fold_results.append({"subject": sid, "n_test": int(len(te)), "acc": acc})
        all_true.extend(y[te].tolist())
        all_pred.extend(pred.tolist())
        if acc > best_acc:
            best_acc, best_model, best_scaler = acc, model, scaler

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall = float((all_true == all_pred).mean())
    cm = _confusion(all_true, all_pred, n_classes)
    f1 = _per_class_f1(cm)

    # save best fold artifacts
    ckpt_dir = _CFG.path("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model.save_model(os.path.join(ckpt_dir, "xgb_best_fold.json"))
    with open(os.path.join(ckpt_dir, "scaler.pkl"), "wb") as fh:
        pickle.dump(best_scaler, fh)

    names = _CFG["dataset"]["task_names"]
    print("\nper-subject accuracy:")
    for r in fold_results:
        print(f"  {r['subject']:10} n={r['n_test']:4d} acc={r['acc']:.3f}")
    print(f"\noverall LOSO accuracy: {overall:.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{names[c]:>11}" for c in range(n_classes)))
    for i in range(n_classes):
        print(f"{names[i]:>8} " + " ".join(f"{cm[i, j]:11d}" for j in range(n_classes)))
    print("per-class F1:", {names[i]: round(f1[i], 3) for i in range(n_classes)})

    # honesty flags
    if overall < 0.55:
        print("WARNING: overall accuracy < 55% — pipeline may be broken")
    stress_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
    if stress_recall < 0.60:
        print(f"WARNING: stress (T2) recall {stress_recall:.2f} < 0.60")

    results = {
        "overall_accuracy": overall,
        "per_subject": fold_results,
        "confusion_matrix": cm.tolist(),
        "per_class_f1": {names[i]: f1[i] for i in range(n_classes)},
        "stress_recall": float(stress_recall),
        "feature_order": feature_order,
        "n_windows": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
    }
    res_dir = _CFG.path("results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "classifier_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nsaved -> {os.path.join(res_dir, 'classifier_results.json')}")
    return results


if __name__ == "__main__":
    train()
