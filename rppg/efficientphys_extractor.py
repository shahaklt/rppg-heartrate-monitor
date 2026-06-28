"""EfficientPhys rPPG extractor (Liu et al., WACV 2023).

Two execution paths:
  1. If a trained checkpoint is available (rPPG-Toolbox model zoo, EfficientPhys
     trained on PURE), load it and run real deep inference.
  2. If no checkpoint / no torch, degrade gracefully to CHROM on the spatial-mean
     RGB of the cropped frames so downstream HRV still gets a waveform. The
     `extractor_used` field on the result records which path ran.

The network below mirrors the EfficientPhys design: it consumes normalised
*difference* frames, applies Temporal-Shift Modules (TSM) for cheap temporal
modelling, a self-attention mask to focus on skin, and a small conv head that
regresses the BVP first-derivative, which we cumulatively sum back to BVP.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from config_loader import load_config
from rppg.chrom_extractor import CHROMExtractor
from rppg.dsp import bandpass_filter

_CFG = load_config()["rppg"]["efficientphys"]


def _build_torch_model(frame_depth: int = 128, img_size: int = 72):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TSM(nn.Module):
        """Temporal shift: shift a fraction of channels +/-1 along time."""

        def __init__(self, n_segment: int, fold_div: int = 3):
            super().__init__()
            self.n_segment = n_segment
            self.fold_div = fold_div

        def forward(self, x):
            nt, c, h, w = x.size()
            n_batch = nt // self.n_segment
            x = x.view(n_batch, self.n_segment, c, h, w)
            fold = c // self.fold_div
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]            # shift left
            out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]       # keep
            return out.view(nt, c, h, w)

    class EfficientPhys(nn.Module):
        def __init__(self, frame_depth=frame_depth, img_size=img_size):
            super().__init__()
            self.frame_depth = frame_depth
            self.tsm1 = TSM(frame_depth)
            self.tsm2 = TSM(frame_depth)
            self.tsm3 = TSM(frame_depth)
            self.tsm4 = TSM(frame_depth)
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(3)
            self.attn = nn.Conv2d(32, 1, 1)
            self.pool = nn.AvgPool2d(2)
            self.drop = nn.Dropout(0.25)
            feat = (img_size // 4) ** 2 * 64
            self.fc1 = nn.Linear(feat, 128)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, x):
            # x: (N*T, 3, H, W) difference frames (BN over the batch normalises)
            x = self.bn1(x)
            x = self.tsm1(x)
            d = torch.tanh(self.conv1(x))
            x = self.tsm2(d)
            x = torch.tanh(self.conv2(x))
            mask = torch.sigmoid(self.attn(x))
            x = x * mask
            x = self.drop(self.pool(x))
            x = self.tsm3(x)
            x = torch.tanh(self.conv3(x))
            x = self.tsm4(x)
            x = torch.tanh(self.conv4(x))
            x = self.drop(self.pool(x))
            x = torch.flatten(x, 1)
            x = torch.tanh(self.fc1(x))
            return self.fc2(x).squeeze(-1)   # (N*T,) BVP derivative

    return EfficientPhys()


class EfficientPhysExtractor:
    def __init__(self, checkpoint: Optional[str] = None,
                 window: int = None, stride: int = None,
                 img_size: int = None, fp16: bool = None):
        self.checkpoint = checkpoint or _CFG["checkpoint"]
        self.window = window or _CFG["window"]
        self.stride = stride or _CFG["stride"]
        self.img_size = img_size or _CFG["input_size"]
        self.fp16 = _CFG["fp16"] if fp16 is None else fp16
        self.device = "cpu"
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            import torch
        except Exception:
            self.model = None
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _build_torch_model(self.window, self.img_size)
        if os.path.exists(self.checkpoint):
            state = torch.load(self.checkpoint, map_location=self.device)
            state = state.get("state_dict", state)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            self.has_weights = True
        else:
            self.has_weights = False
        model.to(self.device).eval()
        if self.fp16 and self.device == "cuda":
            model.half()
        self.model = model

    def _frames_to_diff(self, frames: np.ndarray):
        import cv2
        import torch
        T = len(frames)
        small = np.empty((T, self.img_size, self.img_size, 3), dtype=np.float32)
        for i in range(T):
            small[i] = cv2.resize(frames[i], (self.img_size, self.img_size))
        small = small / 255.0 * 2.0 - 1.0          # normalise to [-1,1]
        diff = np.zeros_like(small)
        diff[1:] = np.diff(small, axis=0)           # difference frames
        x = torch.from_numpy(diff).permute(0, 3, 1, 2).contiguous()
        return x

    def extract(self, frames: np.ndarray, fps: float) -> np.ndarray:
        """frames: (T,H,W,3) uint8 RGB -> bvp (T,)."""
        frames = np.asarray(frames)
        T = len(frames)
        if self.model is None or not getattr(self, "has_weights", False):
            return self._chrom_fallback(frames, fps)

        import torch
        x_all = self._frames_to_diff(frames)
        bvp = np.zeros(T, dtype=np.float64)
        counts = np.zeros(T, dtype=np.float64)
        dtype = torch.float16 if (self.fp16 and self.device == "cuda") else torch.float32
        with torch.no_grad():
            for start in range(0, max(T - self.window + 1, 1), self.stride):
                end = min(start + self.window, T)
                chunk = x_all[start:end].to(self.device, dtype=dtype)
                if chunk.shape[0] < self.window:
                    pad = self.window - chunk.shape[0]
                    chunk = torch.cat([chunk, chunk[-1:].repeat(pad, 1, 1, 1)], 0)
                deriv = self.model(chunk).float().cpu().numpy()[: end - start]
                bvp[start:end] += np.cumsum(deriv)
                counts[start:end] += 1
        counts[counts == 0] = 1
        bvp /= counts
        bvp = bandpass_filter(bvp, fps)
        std = np.std(bvp)
        return (bvp - np.mean(bvp)) / std if std > 1e-8 else bvp

    def _chrom_fallback(self, frames: np.ndarray, fps: float) -> np.ndarray:
        """No weights available: spatial-mean RGB -> CHROM."""
        rgb = frames.reshape(len(frames), -1, 3).mean(axis=1)
        return CHROMExtractor().extract(rgb, fps)

    def benchmark_window(self) -> float:
        """Mean ms per `window`-frame forward pass (dummy data)."""
        if self.model is None:
            return float("nan")
        import torch
        dtype = torch.float16 if (self.fp16 and self.device == "cuda") else torch.float32
        x = torch.randn(self.window, 3, self.img_size, self.img_size,
                        device=self.device, dtype=dtype)
        with torch.no_grad():
            for _ in range(3):           # warmup
                self.model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                self.model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 10 * 1000.0


def _self_test():
    """Offline: confirm extractor returns a finite BVP of the right length on
    synthetic frames (no pretrained weights expected here)."""
    rng = np.random.default_rng(4)
    T = 160
    frames = (rng.random((T, 36, 36, 3)) * 255).astype(np.uint8)
    ext = EfficientPhysExtractor(window=64, stride=32, img_size=36)
    bvp = ext.extract(frames, fps=30.0)
    ok = bvp.shape == (T,) and np.isfinite(bvp).all()
    speed = ext.benchmark_window()
    print(f"[efficientphys self-test] out_shape={bvp.shape} weights={getattr(ext,'has_weights',False)} "
          f"device={ext.device} ms/window={speed:.1f} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _self_test()
