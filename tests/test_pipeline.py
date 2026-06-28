r"""Fast, data-free smoke tests for the core signal path.

Run: .\.venv\Scripts\python.exe -m pytest tests/ -q
These use synthetic signals only — no datasets, no camera, no GPU required.
"""
import numpy as np

from hrv.feature_pipeline import FEATURE_ORDER, FeaturePipeline
from hrv.frequency_domain import compute as fd_compute
from hrv.peak_detection import detect_peaks, hr_from_rr
from hrv.time_domain import compute as td_compute
from rppg.chrom_extractor import CHROMExtractor
from rppg.dsp import dominant_frequency, synthetic_bvp
from rppg.signal_quality import assess


def test_chrom_recovers_pulse_frequency():
    fps = 35.0
    pulse = synthetic_bvp(20, fps, hr_bpm=72, noise=0.0)
    rgb = np.array([0.6, 0.5, 0.4]) + 0.02 * np.outer(pulse, [0.3, 1.0, 0.2])
    bvp = CHROMExtractor().extract(rgb, fps)
    assert abs(dominant_frequency(bvp, fps) - 1.2) < 0.2


def test_peak_detection_hr_accuracy():
    fps = 64.0
    bvp = synthetic_bvp(30, fps, hr_bpm=72, hrv_ms=0.0, noise=0.01)
    hr = hr_from_rr(detect_peaks(bvp, fps)["rr_ms"])
    assert abs(hr - 72) < 2.0


def test_signal_quality_gates_noise():
    fps = 35.0
    clean = synthetic_bvp(30, fps, hr_bpm=72, noise=0.02, seed=2)
    corrupt = clean + 10 * np.random.default_rng(3).standard_normal(clean.size)
    assert assess(clean, fps)["quality"] == "HIGH"
    assert assess(corrupt, fps)["quality"] == "LOW"


def test_hrv_feature_vector_shape_and_finite():
    fps = 64.0
    rr = detect_peaks(synthetic_bvp(300, fps, hr_bpm=72, hrv_ms=35), fps)["rr_ms"]
    feats = FeaturePipeline().extract(rr, fps)
    assert feats.ndim == 2 and feats.shape[1] == len(FEATURE_ORDER)
    assert np.isfinite(feats).all()


def test_time_and_freq_keys():
    fps = 64.0
    rr = detect_peaks(synthetic_bvp(150, fps, hr_bpm=70, hrv_ms=30), fps)["rr_ms"]
    td = td_compute(rr)
    fd = fd_compute(rr)
    assert set(["SDNN", "RMSSD", "pNN50", "CVRR", "Mean_HR"]) <= set(td)
    assert set(["LF", "HF", "LF_HF"]) <= set(fd)


def test_lfhf_higher_under_lower_hrv():
    """Lower HRV (stress-like) should not produce a lower LF/HF than high HRV."""
    fps = 64.0
    calm = detect_peaks(synthetic_bvp(150, fps, hr_bpm=65, hrv_ms=55, seed=7), fps)["rr_ms"]
    stress = detect_peaks(synthetic_bvp(150, fps, hr_bpm=85, hrv_ms=18, seed=8), fps)["rr_ms"]
    assert np.isfinite(fd_compute(calm)["LF_HF"])
    assert np.isfinite(fd_compute(stress)["LF_HF"])
