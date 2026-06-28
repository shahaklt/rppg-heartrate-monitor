"""Webcam capture + live ROI extraction for the demo.

Maintains a rolling RGB-trace buffer (mean cheek/forehead RGB per frame) so the
realtime pipeline can pull the last N seconds cheaply without re-running face
detection over history.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from config_loader import load_config
from rppg.face_roi import FaceROIExtractor

_CFG = load_config()["demo"]


class WebcamCapture:
    def __init__(self, camera_idx: int = None, target_fps: int = None,
                 width: int = None, height: int = None):
        import cv2
        self.cv2 = cv2
        self.camera_idx = camera_idx if camera_idx is not None else _CFG["camera_idx"]
        self.target_fps = target_fps or _CFG["target_fps"]
        self.cap = cv2.VideoCapture(self.camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width or _CFG["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height or _CFG["height"])
        self.roi = FaceROIExtractor()
        self.actual_fps = self._measure_fps()
        maxlen = int(self.actual_fps * _CFG["window_seconds"])
        self.trace_buf = deque(maxlen=maxlen)
        self.last_frame = None
        self.last_face_ok = False

    def _measure_fps(self, n=30):
        t0 = time.perf_counter()
        got = 0
        for _ in range(n):
            ok, fr = self.cap.read()
            if ok:
                got += 1
                self.last_frame = fr
        dt = time.perf_counter() - t0
        return got / dt if dt > 0 else self.target_fps

    def _frame_rgb(self):
        ok, frame_bgr = self.cap.read()
        if not ok:
            return None
        self.last_frame = frame_bgr
        return self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)

    def lighting_status(self, frame_rgb) -> str:
        m = float(frame_rgb.mean())
        if m < 60:
            return "TOO DARK — move to better lighting"
        if m > 220:
            return "TOO BRIGHT — reduce exposure"
        return "ok"

    def step(self):
        """Grab one frame, push its ROI RGB onto the buffer. Returns status dict."""
        rgb = self._frame_rgb()
        if rgb is None:
            return {"ok": False, "reason": "camera read failed"}
        roi = self.roi.process_frame(rgb)
        names = [k for k in roi]
        vals = np.stack([roi[n] for n in names])
        face_ok = not np.isnan(vals).all()
        self.last_face_ok = face_ok
        mean_rgb = np.nanmean(vals, axis=0) if face_ok else np.full(3, np.nan)
        self.trace_buf.append(mean_rgb)
        return {"ok": True, "face": face_ok, "rgb_frame": rgb,
                "lighting": self.lighting_status(rgb)}

    def get_trace(self) -> np.ndarray:
        """Current rolling RGB trace, shape (T,3)."""
        return np.array(self.trace_buf) if self.trace_buf else np.empty((0, 3))

    def capture_buffer(self, duration_seconds: float = None):
        duration_seconds = duration_seconds or _CFG["window_seconds"]
        n = int(duration_seconds * self.actual_fps)
        frames, n_face = [], 0
        for _ in range(n):
            rgb = self._frame_rgb()
            if rgb is None:
                break
            frames.append(rgb)
            roi = self.roi.process_frame(rgb)
            if not np.isnan(np.stack([roi[k] for k in roi])).all():
                n_face += 1
        frames = np.asarray(frames)
        roi_traces = self.roi.process_video_frames(frames) if len(frames) else {}
        return {"frames": frames, "rgb_traces": roi_traces,
                "actual_fps": self.actual_fps, "n_face_detected": n_face}

    def release(self):
        self.cap.release()
        self.roi.close()


if __name__ == "__main__":
    try:
        cam = WebcamCapture()
        print(f"camera {cam.camera_idx} fps~{cam.actual_fps:.1f}")
        s = cam.step()
        print("step:", {k: v for k, v in s.items() if k != "rgb_frame"})
        cam.release()
    except Exception as e:
        print(f"webcam unavailable: {e}")
