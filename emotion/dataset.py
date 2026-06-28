"""Load the HRV feature dataset and yield leave-one-subject-out (LOSO) folds.

LOSO is the honest protocol for physiological classification: a window from a
subject in training must never appear in test, otherwise accuracy is inflated by
subject identity rather than emotion.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from config_loader import load_config

_CFG = load_config()


def load_dataset(path: str = None):
    path = path or _CFG.path("hrv_dataset")
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"].astype(int)
    subjects = d["subjects"].astype(str)
    feature_order = list(d["feature_order"]) if "feature_order" in d else None
    return X, y, subjects, feature_order


def loso_splits(subjects: np.ndarray) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_subject, train_idx, test_idx)."""
    for sid in sorted(np.unique(subjects)):
        test_idx = np.where(subjects == sid)[0]
        train_idx = np.where(subjects != sid)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield sid, train_idx, test_idx


def class_balance(y: np.ndarray) -> dict:
    names = _CFG["dataset"]["task_names"]
    return {f"{int(c)}:{names.get(int(c), '?')}": int((y == c).sum())
            for c in sorted(np.unique(y))}


if __name__ == "__main__":
    X, y, subjects, order = load_dataset()
    print(f"X={X.shape} y={y.shape} subjects={len(np.unique(subjects))}")
    print("features:", order)
    print("balance:", class_balance(y))
    print("LOSO folds:", len(list(loso_splits(subjects))))
