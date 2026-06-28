"""Real-time pipeline: ROI RGB trace -> BVP -> HRV -> emotion.

process_window picks the higher-quality BVP (CHROM always; EfficientPhys when
trained weights + frames are supplied), gates on signal quality, and returns a
prediction dict — or None when the signal is too poor to trust.
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from config_loader import load_config
from hrv.feature_pipeline import FEATURE_ORDER, FeaturePipeline
from hrv.peak_detection import detect_peaks, hr_from_rr
from rppg.chrom_extractor import CHROMExtractor
from rppg.signal_quality import assess

_CFG = load_config()
_NAMES = {int(k): v for k, v in _CFG["demo"]["emotion_names"].items()}


class RealtimePipeline:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        ckpt = _CFG.path("checkpoints")
        model_path = model_path or os.path.join(ckpt, "xgb_best_fold.json")
        scaler_path = scaler_path or os.path.join(ckpt, "scaler.pkl")
        self.ready = os.path.exists(model_path) and os.path.exists(scaler_path)
        self.model = None
        self.scaler = None
        if self.ready:
            import xgboost as xgb
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            with open(scaler_path, "rb") as fh:
                self.scaler = pickle.load(fh)
        self.chrom = CHROMExtractor()
        self.pipe = FeaturePipeline()
        self._ep = None
        self._lat = self._latency_selftest()

    def _ep_extractor(self):
        if self._ep is None:
            from rppg.efficientphys_extractor import EfficientPhysExtractor
            self._ep = EfficientPhysExtractor()
        return self._ep

    def _feature_vector(self, bvp, fps):
        rr = detect_peaks(bvp, fps)["rr_ms"]
        if rr.size < 4:
            return None, None
        feats = self.pipe.extract(rr, fps)
        if feats.shape[0] == 0:
            return None, None
        return feats[-1], rr

    def process_window(self, rgb_trace, fps, frames=None):
        # candidate BVPs
        candidates = []
        bvp_c = self.chrom.extract(np.asarray(rgb_trace), fps)
        candidates.append(("chrom", bvp_c, assess(bvp_c, fps)))
        if frames is not None:
            ext = self._ep_extractor()
            if getattr(ext, "has_weights", False):
                bvp_e = ext.extract(frames, fps)
                candidates.append(("efficientphys", bvp_e, assess(bvp_e, fps)))

        # prefer HIGH quality, break ties by SNR
        high = [c for c in candidates if c[2]["quality"] == "HIGH"]
        pool = high or candidates
        name, bvp, q = max(pool, key=lambda c: c[2]["snr_db"])
        if q["quality"] == "LOW":
            return None   # don't fake predictions on bad signal

        vec, rr = self._feature_vector(bvp, fps)
        if vec is None:
            return None

        result = {
            "hr_bpm": float(hr_from_rr(rr)),
            "rmssd": float(vec[FEATURE_ORDER.index("RMSSD")]),
            "lf_hf": float(vec[FEATURE_ORDER.index("LF_HF")]),
            "quality": q["quality"],
            "snr_db": float(q["snr_db"]),
            "extractor_used": name,
        }
        if self.ready:
            x = self.scaler.transform(vec.reshape(1, -1))
            proba = self.model.predict_proba(x)[0]
            cls = int(np.argmax(proba))
            result.update({
                "emotion": _NAMES.get(cls, str(cls)),
                "class": cls,
                "confidence": float(proba[cls]),
                "arousal": float(1.0 - proba[0]),   # P(not baseline/calm)
                "proba": {_NAMES.get(i, str(i)): float(p) for i, p in enumerate(proba)},
            })
        else:
            result.update({"emotion": "model-not-trained", "confidence": 0.0,
                           "arousal": float("nan")})
        return result

    def _latency_selftest(self):
        rng = np.random.default_rng(0)
        T = int(_CFG["demo"]["window_seconds"] * _CFG["demo"]["target_fps"])
        trace = 0.5 + 0.01 * rng.standard_normal((T, 3))
        t0 = time.perf_counter()
        self.process_window(trace, _CFG["demo"]["target_fps"])
        dt = time.perf_counter() - t0
        if dt >= _CFG["demo"]["update_seconds"]:
            print(f"WARNING: process_window {dt*1000:.0f}ms exceeds "
                  f"{_CFG['demo']['update_seconds']}s update step")
        return dt


if __name__ == "__main__":
    p = RealtimePipeline()
    print(f"pipeline ready={p.ready} latency={p._lat*1000:.1f}ms")
    rng = np.random.default_rng(1)
    from rppg.dsp import synthetic_bvp
    fps = 30.0
    pulse = synthetic_bvp(60, fps, hr_bpm=75)
    trace = 0.5 + 0.01 * np.outer(pulse, [0.3, 1.0, 0.2]) + 0.001 * rng.standard_normal((len(pulse), 3))
    print(p.process_window(trace, fps))
