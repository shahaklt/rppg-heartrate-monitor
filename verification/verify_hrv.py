"""Critical biological-validity check: stress must raise LF/HF vs baseline.

LF/HF rises with sympathetic activation. If the pipeline is sound, most subjects
show LF/HF(stress) > LF/HF(baseline). We require this in >= 70% of subjects.
Operates on the built data/hrv_dataset.npz (no raw re-read needed).
"""
from __future__ import annotations

import numpy as np

from config_loader import load_config
from emotion.dataset import load_dataset

_CFG = load_config()


def run(threshold: float = 0.70):
    X, y, subjects, order = load_dataset()
    lfhf = order.index("LF_HF")
    rmssd = order.index("RMSSD")

    rises, drops, total = 0, 0, 0
    print(f"{'subject':10} {'LFHF base':>10} {'LFHF stress':>12} {'dir':>5}")
    for sid in sorted(np.unique(subjects)):
        m = subjects == sid
        base = X[m & (y == 0), lfhf]
        stress = X[m & (y == 1), lfhf]
        if len(base) == 0 or len(stress) == 0:
            continue
        total += 1
        b, s = np.nanmean(base), np.nanmean(stress)
        up = s > b
        rises += up
        drops += not up
        print(f"{sid:10} {b:10.2f} {s:12.2f} {'up' if up else 'down':>5}")

    frac = rises / total if total else 0.0
    # arousal sanity: RMSSD (vagal) should drop under stress
    base_rmssd = np.nanmean(X[y == 0, rmssd])
    stress_rmssd = np.nanmean(X[y == 1, rmssd])
    ok = frac >= threshold
    print(f"\nLF/HF rises under stress in {rises}/{total} subjects ({frac:.0%}) "
          f"-> {'PASS' if ok else 'FAIL'} (need >={threshold:.0%})")
    print(f"RMSSD baseline={base_rmssd:.1f}ms stress={stress_rmssd:.1f}ms "
          f"({'drops as expected' if stress_rmssd < base_rmssd else 'unexpected rise'})")
    if not ok:
        print("WARNING: HRV direction wrong — inspect peak detection / band settings")
    return ok


if __name__ == "__main__":
    run()
