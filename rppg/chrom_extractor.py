"""CHROM rPPG extractor — de Haan & Jeanne, IEEE TBME 2013.

Chrominance-based method. Projects normalised RGB onto two chrominance axes that
are largely invariant to motion-induced specular changes, then combines them with
a self-tuned alpha. Classical CPU baseline.
"""
from __future__ import annotations

import numpy as np

from rppg.dsp import bandpass_filter, dominant_frequency


class CHROMExtractor:
    def __init__(self, low_hz: float = 0.7, high_hz: float = 3.0, order: int = 3):
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def extract(self, rgb_trace: np.ndarray, fps: float) -> np.ndarray:
        """rgb_trace: (T, 3) mean R,G,B per frame -> bvp: (T,).

        Steps (paper):
          1. temporal normalisation  C_n = C / mean(C)
          2. Xs = 3R - 2G ; Ys = 1.5R + G - 1.5B
          3. bandpass Xs, Ys
          4. alpha = std(Xf)/std(Yf)
          5. bvp = Xf - alpha*Yf
        """
        rgb = np.asarray(rgb_trace, dtype=np.float64)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"rgb_trace must be (T,3), got {rgb.shape}")
        T = rgb.shape[0]
        if T < 4:
            return np.zeros(T)

        # 1. normalise each channel by its temporal mean (handle NaN frames)
        means = np.nanmean(rgb, axis=0)
        means[means == 0] = 1.0
        norm = rgb / means

        R, G, B = norm[:, 0], norm[:, 1], norm[:, 2]

        # 2. chrominance signals
        Xs = 3.0 * R - 2.0 * G
        Ys = 1.5 * R + G - 1.5 * B

        # 3. zero-phase bandpass
        Xf = bandpass_filter(Xs, fps, self.low_hz, self.high_hz, self.order)
        Yf = bandpass_filter(Ys, fps, self.low_hz, self.high_hz, self.order)

        # 4. alpha tuning
        sy = np.std(Yf)
        alpha = np.std(Xf) / sy if sy > 1e-8 else 0.0

        # 5. combine
        bvp = Xf - alpha * Yf
        # standardise output scale
        std = np.std(bvp)
        if std > 1e-8:
            bvp = (bvp - np.mean(bvp)) / std
        return bvp


def _self_test():
    """Synthetic check: build an RGB trace whose green channel pulses at 1.2 Hz
    and confirm the recovered BVP's dominant frequency matches."""
    from rppg.dsp import synthetic_bvp

    fps = 35.0
    dur = 20.0
    pulse = synthetic_bvp(dur, fps, hr_bpm=72.0, noise=0.0)
    T = pulse.size
    rng = np.random.default_rng(1)
    base = np.array([0.6, 0.5, 0.4])
    # green carries strongest PPG modulation
    mod = np.outer(pulse, np.array([0.3, 1.0, 0.2]))
    rgb = base + 0.02 * mod + 0.001 * rng.standard_normal((T, 3))

    bvp = CHROMExtractor().extract(rgb, fps)
    peak_hz = dominant_frequency(bvp, fps)
    expected = 72.0 / 60.0
    ok = abs(peak_hz - expected) < 0.2
    print(f"[CHROM self-test] dominant={peak_hz:.3f} Hz expected~{expected:.3f} Hz "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _self_test()
