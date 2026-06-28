"""Small PyTorch MLP emotion classifier — second approach to compare vs XGBoost.

Deliberately tiny (11 -> 64 -> 32 -> 3). Same LOSO-CV protocol and per-fold
scaling. If XGBoost wins, the demo uses XGBoost (simpler, interpretable); MLP is
kept as a baseline to justify that choice.
"""
from __future__ import annotations

import json
import os

import numpy as np

from config_loader import load_config
from emotion.dataset import class_balance, load_dataset, loso_splits

_CFG = load_config()
_MLP = _CFG["classifier"]["mlp"]


def _build_mlp(n_in: int, n_out: int, dropout: float):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(32, n_out),
    )


def _train_fold(Xtr, ytr, Xte, yte, n_out, device):
    import torch
    import torch.nn as nn
    model = _build_mlp(Xtr.shape[1], n_out, _MLP["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=_MLP["lr"])
    # balanced class weights from the training fold only (matches XGBoost path)
    classes, counts = np.unique(ytr, return_counts=True)
    weight = torch.ones(n_out, device=device)
    for c, n in zip(classes, counts):
        weight[int(c)] = len(ytr) / (len(classes) * n)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    best_loss, patience, best_state = float("inf"), 0, None
    for epoch in range(_MLP["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(Xtr_t)
        loss = loss_fn(out, ytr_t)
        loss.backward()
        opt.step()
        if loss.item() < best_loss - 1e-4:
            best_loss, patience = loss.item(), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= _MLP["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


def train(dataset_path: str = None):
    import torch
    from sklearn.preprocessing import StandardScaler

    X, y, subjects, _ = load_dataset(dataset_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MLP on {device}, X={X.shape}, balance={class_balance(y)}")
    n_out = len(np.unique(y))

    accs = []
    for sid, tr, te in loso_splits(subjects):
        scaler = StandardScaler().fit(X[tr])
        acc = _train_fold(scaler.transform(X[tr]), y[tr],
                          scaler.transform(X[te]), y[te], n_out, device)
        accs.append(acc)
        print(f"  {sid:10} acc={acc:.3f}")
    overall = float(np.mean(accs))
    print(f"\nMLP overall LOSO accuracy: {overall:.3f}")

    res_dir = _CFG.path("results")
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "mlp_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"overall_accuracy": overall, "per_fold": accs}, fh, indent=2)
    return overall


if __name__ == "__main__":
    train()
