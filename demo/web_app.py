"""Tiny web demo for the rPPG -> HRV -> emotion pipeline.

The browser grabs the webcam and POSTs frames here; this server runs the *real*
pipeline (MediaPipe face mesh -> CHROM rPPG -> HRV -> XGBoost) and sends back the
face landmarks (drawn as dots in the browser) plus the current reading. Nothing
is faked: emotion only appears once a full 60 s window is collected, and a live
heart rate shows up after ~8 s so there's something to watch while it fills.

Run:
    .\.venv\Scripts\python.exe -m demo.web_app
then open http://localhost:5000
"""
from __future__ import annotations

import base64
import math
import time
from collections import deque

import numpy as np
from flask import Flask, Response, jsonify, request

from config_loader import load_config
from demo.realtime_pipeline import RealtimePipeline
from rppg.chrom_extractor import CHROMExtractor
from rppg.dsp import dominant_frequency
from rppg.face_roi import FaceROIExtractor

_CFG = load_config()["demo"]
WIN = float(_CFG["window_seconds"])      # 60 s window the model was trained on
UPDATE = float(_CFG["update_seconds"])   # rerun emotion at most this often

app = Flask(__name__)

# one shared face mesh + pipeline + rolling buffer (single-user local demo)
_roi = FaceROIExtractor()
_pipe = RealtimePipeline()
_chrom = CHROMExtractor()
_rgb = deque(maxlen=4000)     # mean ROI RGB per received frame
_ts = deque(maxlen=4000)      # timestamp per frame -> real fps, robust to jitter
_state = {"last_run": 0.0, "result": None}


def _decode(data_url: str) -> np.ndarray:
    import cv2
    b64 = data_url.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _clean(obj):
    """Make a result JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _live_hr(fps: float):
    """Quick heart-rate estimate from the last ~10 s (just for the warmup display)."""
    arr = np.array(_rgb)[-int(fps * 10):]
    if len(arr) < fps * 4 or np.isnan(arr).all():
        return None
    try:
        bvp = _chrom.extract(arr, fps)
        hz = dominant_frequency(bvp, fps)
        return float(hz * 60.0) if np.isfinite(hz) else None
    except Exception:
        return None


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/process", methods=["POST"])
def process():
    rgb = _decode(request.json["image"])
    det = _roi.process_frame_detailed(rgb)
    face_ok = det["landmarks"] is not None

    if face_ok:
        vals = np.stack([det["roi"][k] for k in det["roi"]])
        mean_rgb = np.nanmean(vals, axis=0) if not np.isnan(vals).all() else np.full(3, np.nan)
    else:
        mean_rgb = np.full(3, np.nan)
    now = time.time()
    _rgb.append(mean_rgb)
    _ts.append(now)

    span = _ts[-1] - _ts[0] if len(_ts) > 1 else 0.0
    fps = len(_ts) / span if span > 1 else float(_CFG["target_fps"])
    resp = {
        "face": face_ok,
        "dots": det["landmarks"] or [],
        "roi_points": det["roi_points"],
        "fill": min(1.0, span / WIN),
        "fps": round(fps, 1),
        "live_hr": None,
        "result": _clean(_state["result"]),
        "status": "",
    }

    if not face_ok:
        resp["status"] = "no face — center your face in view"
        return jsonify(resp)

    resp["live_hr"] = _live_hr(fps)

    if span < WIN * 0.9:
        resp["status"] = f"collecting baseline… {int(100 * span / WIN)}%"
        return jsonify(resp)

    if now - _state["last_run"] >= UPDATE:
        _state["last_run"] = now
        _state["result"] = _pipe.process_window(np.array(_rgb), fps)
    resp["result"] = _clean(_state["result"])
    resp["status"] = "ok" if _state["result"] else "low signal — hold still / more light"
    return jsonify(resp)


@app.route("/reset", methods=["POST"])
def reset():
    _rgb.clear(); _ts.clear()
    _state["last_run"] = 0.0; _state["result"] = None
    return jsonify({"ok": True})


# ---- single-page front end (deliberately plain) --------------------------
INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>rppg emotion demo</title>
<style>
  body { background:#141414; color:#dcdcdc; font-family:Consolas,monospace; margin:24px; }
  h2 { font-weight:normal; margin:0 0 4px; }
  .sub { color:#888; font-size:13px; margin-bottom:16px; }
  #wrap { position:relative; width:640px; }
  video, #overlay { position:absolute; top:0; left:0; width:640px; height:480px;
                    transform:scaleX(-1); border-radius:4px; }       /* mirror for selfie view */
  #overlay { transform:scaleX(-1); }
  #panel { margin-top:496px; width:640px; }
  .row { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid #262626; }
  .k { color:#888; }
  #emotion { font-size:30px; margin:10px 0 6px; }
  #status { color:#e0a050; min-height:18px; margin-bottom:8px; }
  #bar { height:14px; background:#262626; border-radius:7px; overflow:hidden; }
  #fill { height:100%; width:0%; background:#e0772e; transition:width .3s; }
  button { background:#2e2e2e; color:#dcdcdc; border:1px solid #444; padding:8px 14px;
           font-family:inherit; cursor:pointer; border-radius:3px; }
  button:hover { background:#3a3a3a; }
</style>
</head>
<body>
  <h2>contactless pulse + emotion from webcam</h2>
  <div class="sub">face mesh &rarr; skin-colour pulse (rPPG) &rarr; heart-rate variability &rarr; emotion. all local. needs ~60s + steady light.</div>

  <button id="start">start camera</button>
  <div id="wrap">
    <video id="vid" autoplay playsinline muted></video>
    <canvas id="overlay" width="640" height="480"></canvas>
  </div>

  <div id="panel">
    <div id="emotion">—</div>
    <div id="status">click start</div>
    <div id="bar"><div id="fill"></div></div>
    <div style="font-size:12px;color:#888;margin:2px 0 10px;">arousal</div>
    <div class="row"><span class="k">heart rate</span><span id="hr">—</span></div>
    <div class="row"><span class="k">RMSSD (HRV)</span><span id="rmssd">—</span></div>
    <div class="row"><span class="k">LF/HF</span><span id="lfhf">—</span></div>
    <div class="row"><span class="k">confidence</span><span id="conf">—</span></div>
    <div class="row"><span class="k">signal</span><span id="sig">—</span></div>
    <div class="row"><span class="k">fps / collected</span><span id="meta">—</span></div>
  </div>

<script>
const vid = document.getElementById('vid');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const grab = document.createElement('canvas');      // offscreen, sends frames to server
grab.width = 480; grab.height = 360;
const gctx = grab.getContext('2d');
let busy = false, running = false;

document.getElementById('start').onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}});
  vid.srcObject = stream;
  await fetch('/reset', {method:'POST'});
  running = true;
  document.getElementById('status').textContent = 'starting…';
  setInterval(tick, 80);   // ~12 fps; server measures the real rate
};

function fmt(x, d=0){ return (x===null||x===undefined) ? '—' : Number(x).toFixed(d); }

async function tick(){
  if(!running || busy || vid.readyState < 2) return;
  busy = true;
  try {
    gctx.drawImage(vid, 0, 0, grab.width, grab.height);   // raw (un-mirrored) frame
    const img = grab.toDataURL('image/jpeg', 0.7);
    const r = await fetch('/process', {method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify({image: img})});
    const d = await r.json();
    draw(d);
    update(d);
  } catch(e) { /* drop a frame, keep going */ }
  busy = false;
}

function draw(d){
  octx.clearRect(0,0,overlay.width,overlay.height);
  const W = overlay.width, H = overlay.height;
  // all landmarks: small faint green dots
  octx.fillStyle = 'rgba(90,210,130,0.55)';
  for(const p of d.dots){
    octx.beginPath(); octx.arc(p[0]*W, p[1]*H, 1.3, 0, 7); octx.fill();
  }
  // ROI regions (cheeks/forehead) the pulse is actually read from: brighter, bigger
  octx.fillStyle = '#39d6e6';
  for(const name in d.roi_points){
    for(const p of d.roi_points[name]){
      octx.beginPath(); octx.arc(p[0]*W, p[1]*H, 2.6, 0, 7); octx.fill();
    }
  }
}

function update(d){
  document.getElementById('status').textContent = d.status || '';
  document.getElementById('fill').style.width = Math.round((d.fill||0)*100) + '%';
  document.getElementById('meta').textContent = fmt(d.fps,1) + ' fps / ' + Math.round((d.fill||0)*100) + '%';
  const res = d.result;
  // heart rate: live estimate during warmup, model HR once available
  const hr = res ? res.hr_bpm : d.live_hr;
  document.getElementById('hr').textContent = hr ? fmt(hr,0)+' bpm' : '—';
  if(res){
    document.getElementById('emotion').textContent = (res.emotion||'—').toUpperCase();
    document.getElementById('rmssd').textContent = fmt(res.rmssd,0)+' ms';
    document.getElementById('lfhf').textContent  = fmt(res.lf_hf,2);
    document.getElementById('conf').textContent  = res.confidence!=null ? Math.round(res.confidence*100)+'%' : '—';
    document.getElementById('sig').textContent   = (res.quality||'—')+' ('+fmt(res.snr_db,1)+' dB)';
    document.getElementById('fill').style.width  = Math.round((res.arousal||0)*100) + '%';
  } else {
    document.getElementById('emotion').textContent = d.face ? '…' : 'no face';
  }
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("rPPG emotion web demo -> http://localhost:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
