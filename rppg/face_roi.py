"""Face ROI extraction with MediaPipe FaceLandmarker (Tasks API).

Detects 468 face-mesh landmarks, builds left-cheek / right-cheek / forehead
polygon masks, and returns the mean RGB per ROI per frame. Frames with no face
become NaN rows and are reported via `failed_frames`.

Uses the modern MediaPipe Tasks API (`mp.tasks.vision.FaceLandmarker`) — recent
mediapipe wheels ship only `tasks`, not the legacy `mp.solutions.face_mesh`. The
landmark model (`face_landmarker.task`) is downloaded once into checkpoints/.
MediaPipe is imported lazily so the numeric pipeline / self-tests load without it.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Dict, List

import numpy as np

from config_loader import load_config, project_path

_CFG = load_config()
ROI_LANDMARKS: Dict[str, List[int]] = {
    k: list(v) for k, v in _CFG["rppg"]["roi_landmarks"].items()
}

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")


def _model_path() -> str:
    """Path to the FaceLandmarker model, downloading it once if missing."""
    p = project_path("checkpoints", "face_landmarker.task")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        print("downloading face_landmarker.task (~3.6 MB, one-time)...")
        urllib.request.urlretrieve(_MODEL_URL, p)
    return p


class FaceROIExtractor:
    def __init__(self, static_image_mode: bool = False,
                 roi_landmarks: Dict[str, List[int]] = None):
        self.roi_landmarks = roi_landmarks or ROI_LANDMARKS
        self._mp = None
        self._mesh = None
        self._static = static_image_mode

    def _ensure_mesh(self):
        if self._mesh is None:
            import mediapipe as mp  # lazy
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._mp = mp
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_model_path()),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(opts)
        return self._mesh

    def _detect_xy(self, frame_rgb: np.ndarray):
        """Run the landmarker -> (N,2) normalized landmark coords, or None."""
        landmarker = self._ensure_mesh()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb, dtype=np.uint8))
        res = landmarker.detect(image)
        if not res.face_landmarks:
            return None
        return np.array([[p.x, p.y] for p in res.face_landmarks[0]], dtype=np.float64)

    @staticmethod
    def _polygon_mean_rgb(frame_rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Mean RGB inside the convex polygon given by pts (N,2) pixel coords."""
        import cv2
        h, w = frame_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1)
        sel = mask.astype(bool)
        if sel.sum() == 0:
            return np.full(3, np.nan)
        region = frame_rgb[sel]
        return region.mean(axis=0).astype(np.float64)

    def process_frame(self, frame_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """One RGB frame -> {roi_name: (3,) mean RGB or NaN}."""
        h, w = frame_rgb.shape[:2]
        out = {name: np.full(3, np.nan) for name in self.roi_landmarks}
        xy = self._detect_xy(frame_rgb)
        if xy is None:
            return out
        coords = xy * np.array([w, h])
        for name, idxs in self.roi_landmarks.items():
            out[name] = self._polygon_mean_rgb(frame_rgb, coords[idxs])
        return out

    def process_frame_detailed(self, frame_rgb: np.ndarray) -> Dict:
        """One RGB frame -> {'roi': {name:(3,)}, 'landmarks': (N,2) normalized 0-1,
        'roi_points': {name:[[x,y],...] normalized}}. Single detection pass.

        Used by the live web demo: 'landmarks' drives the on-screen face dots and
        'roi_points' highlights the cheek/forehead regions the pulse is read from.
        landmarks is None when no face is found.
        """
        h, w = frame_rgb.shape[:2]
        out = {"roi": {name: np.full(3, np.nan) for name in self.roi_landmarks},
               "landmarks": None, "roi_points": {}}
        xy = self._detect_xy(frame_rgb)
        if xy is None:
            return out
        coords = xy * np.array([w, h])
        for name, idxs in self.roi_landmarks.items():
            out["roi"][name] = self._polygon_mean_rgb(frame_rgb, coords[idxs])
            out["roi_points"][name] = xy[idxs].tolist()
        out["landmarks"] = xy.tolist()
        return out

    def process_video_frames(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """frames: (T,H,W,3) RGB uint8 -> {roi: (T,3)} plus 'failed_frames'.

        Missing detections are NaN rows; their indices are returned under the
        key 'failed_frames'.
        """
        T = len(frames)
        traces = {name: np.full((T, 3), np.nan) for name in self.roi_landmarks}
        failed = []
        for t in range(T):
            res = self.process_frame(frames[t])
            any_face = False
            for name in self.roi_landmarks:
                traces[name][t] = res[name]
                any_face = any_face or not np.isnan(res[name]).any()
            if not any_face:
                failed.append(t)
        traces["failed_frames"] = np.array(failed, dtype=int)
        return traces

    def close(self):
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None


def green_channel_periodicity(trace: np.ndarray, fps: float) -> float:
    """Dominant frequency (Hz) of the green channel of an ROI trace (T,3).

    Verification helper: a real face ROI's G channel should peak near ~1 Hz.
    """
    from rppg.dsp import dominant_frequency
    g = np.asarray(trace, dtype=np.float64)[:, 1]
    return dominant_frequency(g, fps)


def _self_test():
    """Without a camera we validate the periodicity helper on a synthetic ROI
    whose green channel pulses at 1.0 Hz."""
    fps = 30.0
    t = np.arange(int(fps * 15)) / fps
    g = 0.5 + 0.01 * np.sin(2 * np.pi * 1.0 * t)
    trace = np.stack([0.6 + 0 * t, g, 0.4 + 0 * t], axis=1)
    hz = green_channel_periodicity(trace, fps)
    ok = abs(hz - 1.0) < 0.2
    print(f"[face_roi self-test] green periodicity={hz:.3f} Hz expected~1.0 "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    _self_test()
