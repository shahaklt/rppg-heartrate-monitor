"""Live Gradio demo: webcam -> rPPG -> HRV -> emotion.

Single page. The webcam streams frames; each frame's ROI mean RGB is pushed onto a
rolling buffer. Every `update_seconds` the realtime pipeline runs on the buffered
trace and the panels refresh. Predictions are suppressed until a full window is
collected and whenever signal quality is LOW (no fake outputs).
"""
from __future__ import annotations

import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config_loader import load_config
from demo.realtime_pipeline import RealtimePipeline
from rppg.chrom_extractor import CHROMExtractor
from rppg.dsp import bandpass_filter
from rppg.face_roi import FaceROIExtractor

_CFG = load_config()["demo"]
_WIN = _CFG["window_seconds"]
_UPDATE = _CFG["update_seconds"]

# Plain-language explanation shown in the "How this works" panel.
HOW_IT_WORKS = """
**1. Camera → skin colour.** Face landmarks mark the cheeks and forehead; the average
colour of that skin is tracked frame by frame.

**2. Colour → heartbeat.** Each beat changes blood volume and tints the skin slightly.
The CHROM method pulls that faint pulse out of the colour wobble — no contact needed.

**3. Heartbeat → emotion.** The spacing between beats (heart-rate variability) shifts
with arousal: stress tightens it, calm loosens it. A model trained on real wearable
recordings reads those shifts and reports calm vs aroused.

Heads up: it needs ~60 seconds to fill its first window, steady lighting, and a still
head. It reads arousal well; telling stress from amusement is genuinely hard.
"""


def _new_state():
    return {
        "trace": deque(maxlen=int(_CFG["target_fps"] * _WIN)),
        "history": deque(maxlen=120),
        "last_pred": 0.0,
        "roi": FaceROIExtractor(),
        "pipe": RealtimePipeline(),
        "chrom": CHROMExtractor(),
        "fps": float(_CFG["target_fps"]),
    }


def _fig_to_array(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def _bvp_plot(trace, fps):
    fig, ax = plt.subplots(figsize=(4, 2))
    if len(trace) > fps * 2:
        bvp = bandpass_filter(CHROMExtractor().extract(np.array(trace), fps), fps)
        seg = bvp[-int(fps * 10):]
        ax.plot(np.arange(len(seg)) / fps, seg, color="#c0392b", lw=1)
        ax.set_title("BVP (last 10 s)")
    else:
        ax.text(0.5, 0.5, "collecting...", ha="center", va="center")
    ax.set_xlabel("s"); ax.set_yticks([])
    fig.tight_layout()
    return _fig_to_array(fig)


def _history_plot(history):
    fig, ax = plt.subplots(figsize=(4, 2))
    if history:
        vals = list(history)
        ax.plot(vals, color="#2e86c1", lw=1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("arousal over time")
    else:
        ax.text(0.5, 0.5, "no data yet", ha="center", va="center")
    ax.set_xlabel("update #"); ax.set_ylabel("arousal")
    fig.tight_layout()
    return _fig_to_array(fig)


def stream(frame, state):
    if state is None:
        state = _new_state()
    if frame is None:
        return state, "No camera frame.", "", None, None

    fps = state["fps"]
    roi = state["roi"].process_frame(frame)
    vals = np.stack([roi[k] for k in roi])
    face_ok = not np.isnan(vals).all()
    state["trace"].append(np.nanmean(vals, axis=0) if face_ok else np.full(3, np.nan))

    n = len(state["trace"])
    filled = n >= int(fps * _WIN) * 0.9
    bvp_img = _bvp_plot(state["trace"], fps)
    hist_img = _history_plot(state["history"])

    lighting = frame.mean()
    if not face_ok:
        return state, "### No face detected", "Center your face in view.", bvp_img, hist_img
    if lighting < 60:
        return state, "### Too dark", "Move to better lighting.", bvp_img, hist_img
    if not filled:
        pct = int(100 * n / (fps * _WIN))
        return state, f"### Collecting baseline… {pct}%", \
            "Hold still ~60 s for the first reading.", bvp_img, hist_img

    now = time.time()
    if now - state["last_pred"] < _UPDATE:
        # keep last shown panels; cheap path
        return state, state.get("_last_label", "…"), state.get("_last_hrv", ""), bvp_img, hist_img
    state["last_pred"] = now

    res = state["pipe"].process_window(np.array(state["trace"]), fps)
    if res is None:
        return state, "### Low signal quality", \
            "Reduce head movement / improve lighting.", bvp_img, hist_img

    state["history"].append(res.get("arousal", 0.0))
    emotion = res.get("emotion", "?").upper()
    conf = res.get("confidence", 0.0)
    label = f"### {emotion}\nconfidence {conf:.0%} · arousal {res.get('arousal', float('nan')):.0%}"
    hrv = (f"**HR** {res['hr_bpm']:.0f} bpm  \n"
           f"**RMSSD** {res['rmssd']:.0f} ms  \n"
           f"**LF/HF** {res['lf_hf']:.2f}  \n"
           f"**quality** {res['quality']} ({res['snr_db']:.1f} dB)  \n"
           f"**extractor** {res['extractor_used']}")
    state["_last_label"], state["_last_hrv"] = label, hrv
    return state, label, hrv, bvp_img, hist_img


def build_ui():
    import gradio as gr
    with gr.Blocks(title="rPPG Emotion") as demo:
        gr.Markdown("# Contactless Emotion from Webcam\nrPPG → HRV → emotion, fully local.")
        st = gr.State()
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, label="webcam", type="numpy")
            with gr.Column():
                label = gr.Markdown("### Collecting baseline…")
                msg = gr.Markdown("")
        with gr.Row():
            bvp = gr.Image(label="BVP waveform", type="numpy")
            hrv = gr.Markdown("")
            hist = gr.Image(label="history", type="numpy")
        with gr.Accordion("How this works", open=False):
            gr.Markdown(HOW_IT_WORKS)
        cam.stream(stream, inputs=[cam, st], outputs=[st, label, hrv, bvp, hist],
                   time_limit=None, stream_every=1.0 / _CFG["target_fps"])
    return demo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(server_port=args.port, share=args.share)
