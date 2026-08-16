"""
model.py — SwarCare AI Pipeline (shared backend)
==================================================
Single source of truth for the anomaly-detection pipeline. engine.py (the
continuous recorder) calls AnomalyDetectionModel.evaluate(dir, prefix) to
read a finished {prefix}_piezo.csv + {prefix}_audio.wav pair from disk; the
result is plain JSON-serializable dict ready for main.py (Streamlit) to
render via ai_results.get(...) / st.json(ai_results).

Pipeline (verified against the working reference Streamlit app):
  Audio:      WAV (16kHz) -> YAMNet (521-D, averaged across frames)
                           -> PCA (64-D) -> raw Euclidean distance to
                              learned center -> healthy/watch/anomaly
  Vibration:  waveform -> 6-D DSP features (RMS, ZCR, Spectral Centroid,
                           High-Freq Ratio, Peak FFT Freq, F0 Autocorr)
                        -> Z-score normalized distance to learned center
                        -> healthy/watch/anomaly

*** RESOLVED - 2kHz PIEZO HARDWARE ***
  Vibration features (ZCR especially) are sample-rate dependent, so the piezo
  model must always be calibrated and scored at the same rate. Engine.py's real
  hardware rate is PIEZO_SAMPLE_RATE_HZ = 2000 (mic stays 16kHz for YAMNet
  audio). Two coordinated fixes keep vibration scoring valid at 2kHz:
    1. extract_dsp_features()'s ZCR is now expressed per-SECOND (rate-invariant),
       so the feature set no longer silently breaks if the capture rate changes.
    2. kmeans_vibration_center/std.npy were REGENERATED at 2000Hz from a
       synthetic known-healthy distribution (recalibrate_vibration_2khz.py).
  If you later recalibrate on your own real hardware, keep the sample rate at
  2000 for both the calibration step and scoring.
"""

import os
import json
import struct
import time
import wave
from datetime import timezone, timedelta
from dataclasses import dataclass

import numpy as np

# --- TARGET SPECIFICATION: INDIAN STANDARD TIME (UTC +5:30) ---
IST = timezone(timedelta(hours=5, minutes=30))
PACKET_FORMAT = "<IQ40H"
RAW_PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

# --- ADC / VOLTAGE CONVERSION CONSTANTS ---
ADC_MAX_VALUE = 4095.0        # 12-bit ADC (0-4095)
ADC_REFERENCE_VOLTAGE = 3.3   # Reference voltage of the ADC

try:
    from arduino.app_utils import App, Bridge
    ON_DEVICE = True
except ImportError:
    ON_DEVICE = False
    class App:
        @staticmethod
        def run():
            while True:
                time.sleep(1)
    class Bridge:
        @staticmethod
        def provide(n, c):
            pass
        @staticmethod
        def notify(n, *args):
            pass


# =============================================================================
# CONFIG / PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
YAMNET_PATH = os.path.join(MODELS_DIR, "yamnet.tflite")
PCA_PATH = os.path.join(MODELS_DIR, "pca_model.npz")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_center.npy")
KMEANS_VIB_PATH = os.path.join(MODELS_DIR, "kmeans_vibration_center.npy")
KMEANS_VIB_STD_PATH = os.path.join(MODELS_DIR, "kmeans_vibration_std.npy")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"anomaly_threshold": 0.62, "vibration_anomaly_threshold": 4.5552}


CONFIG = load_config()
TARGET_SR = int(CONFIG.get("sample_rate", 16000))          # audio (YAMNet) rate
WINDOW_SEC = float(CONFIG.get("window_seconds", 2.0))

<<<<<<< HEAD
# See "CRITICAL" note in module docstring - best-evidence default, verify against
# your real piezo hardware rate before trusting vibration verdicts.
=======
# Piezo hardware is confirmed at 2 kHz (model.py's feature extractor and the
# deployed kmeans_vibration_*.npy baseline were regenerated to match). Engine.py
# also passes its PIEZO_SAMPLE_RATE_HZ (2000) explicitly at call time, so this
# value is the consistent default for any calibration/inference without an
# explicit rate.
>>>>>>> 43de0ad (Fix vibration model for 2 kHz piezo hardware)
VIBRATION_SAMPLE_RATE_HZ = int(CONFIG.get("vibration_sample_rate_hz", 2000))


# =============================================================================
# MODEL LOADING (plain functional caching - framework agnostic; works whether
# or not a caller wraps these in @st.cache_resource elsewhere)
# =============================================================================

_cache = {}


def load_interpreter():
    """Load YAMNet TFLite interpreter (tries lightest runtime first)."""
    if "interpreter" in _cache:
        return _cache["interpreter"]

    for mod_path, cls_name in [
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite.python.interpreter", "Interpreter"),
    ]:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            Interpreter = getattr(mod, cls_name)
            break
        except ImportError:
            continue
    else:
        raise ImportError("No TFLite runtime found (tried ai_edge_litert, tflite_runtime, tensorflow)")

    interp = Interpreter(model_path=YAMNET_PATH)
    interp.allocate_tensors()
    _cache["interpreter"] = interp
    return interp


def load_pca():
    """Load PCA: mean (521,) and components (64, 521)."""
    if "pca" not in _cache:
        data = np.load(PCA_PATH)
        _cache["pca"] = (data["mean"].astype(np.float32), data["components"].astype(np.float32))
    return _cache["pca"]


def load_kmeans():
    """Load audio K-means center (64,)."""
    if "kmeans" not in _cache:
        _cache["kmeans"] = np.load(KMEANS_PATH).astype(np.float32)
    return _cache["kmeans"]


def load_kmeans_vibration():
    """Load K-means center for vibration (6-D), or None if not deployed."""
    if "kmeans_vib" not in _cache:
        _cache["kmeans_vib"] = (
            np.load(KMEANS_VIB_PATH).astype(np.float32) if os.path.exists(KMEANS_VIB_PATH) else None
        )
    return _cache["kmeans_vib"]


def load_kmeans_vibration_std():
    """Load K-means std deviation for vibration (6-D), or None if not deployed."""
    if "kmeans_vib_std" not in _cache:
        _cache["kmeans_vib_std"] = (
            np.load(KMEANS_VIB_STD_PATH).astype(np.float32) if os.path.exists(KMEANS_VIB_STD_PATH) else None
        )
    return _cache["kmeans_vib_std"]


def clear_model_cache():
    """Call after (re)calibration so the next inference reloads fresh files."""
    _cache.clear()


# =============================================================================
# AUDIO PIPELINE: WAV -> YAMNet -> PCA -> distance to learned center
# =============================================================================

def run_yamnet(interpreter, audio: np.ndarray):
    """YAMNet: audio -> averaged 521-D scores across all frames in the clip."""
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_size = inp["shape"][-1]

    audio = audio.astype(np.float32)
    scores_list = []
    hop = input_size // 2

    for start in range(0, len(audio) - input_size + 1, hop):
        chunk = audio[start:start + input_size]
        if len(inp["shape"]) == 1:
            data = chunk.reshape(inp["shape"])
        else:
            data = chunk.reshape(1, -1)
            if list(data.shape) != list(inp["shape"]):
                interpreter.resize_tensor_input(inp["index"], list(data.shape))
                interpreter.allocate_tensors()
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()
        scores_list.append(interpreter.get_tensor(out["index"]).flatten())

    if not scores_list:
        padded = np.zeros(input_size, dtype=np.float32)
        padded[:min(len(audio), input_size)] = audio[:input_size]
        data = padded if len(inp["shape"]) == 1 else padded.reshape(1, -1)
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()
        scores_list.append(interpreter.get_tensor(out["index"]).flatten())

    return np.mean(scores_list, axis=0), len(scores_list)


def run_pipeline(audio: np.ndarray) -> dict:
    """Full Audio Pipeline: audio -> YAMNet -> PCA -> K-means -> result dict."""
    t0 = time.time()

    interpreter = load_interpreter()
    scores, n_windows = run_yamnet(interpreter, audio)
    t_yamnet = time.time() - t0

    pca_mean, pca_comp = load_pca()
    features = (scores - pca_mean) @ pca_comp.T
    t_pca = time.time() - t0

    center = load_kmeans()
    threshold = load_config().get("anomaly_threshold", 0.62)
    distance = float(np.sqrt(np.sum((features - center) ** 2)))

    if distance < threshold * 0.5:
        status = "healthy"
    elif distance < threshold:
        status = "watch"
    else:
        status = "anomaly"

    t_total = time.time() - t0

    return {
        "available": True,
        "status": status,
        "score": distance,               # raw distance - NOT 0-1, see fuse_for_ui()
        "threshold": threshold,
        "is_anomaly": distance > threshold,
        "n_windows": n_windows,
        "timing": {
            "yamnet_ms": round(t_yamnet * 1000),
            "pca_ms": round((t_pca - t_yamnet) * 1000, 1),
            "total_ms": round(t_total * 1000),
        },
    }


# =============================================================================
# VIBRATION PIPELINE: waveform -> 6-D DSP features -> Z-score distance
# =============================================================================

def extract_dsp_features(waveform: np.ndarray, sr: int = TARGET_SR,
                          high_freq_split_ratio: float = 0.5) -> np.ndarray:
    """Extract 6-D physical features from the vibration waveform.

    Features:
        [0] RMS energy
        [1] Zero Crossing Rate (ZCR) - crossings per SECOND (rate-invariant)
        [2] Spectral Centroid (Hz)
        [3] High-Frequency Energy Ratio (upper band of the spectrum)
        [4] Peak FFT Frequency (Hz)
        [5] Fundamental Frequency F0 via autocorrelation (Hz)

    `high_freq_split_ratio` sets feature [3]'s cutoff as a FRACTION OF NYQUIST
    (sr/2), not a fixed Hz value. A fixed 1000Hz cutoff is meaningless once
    sr drops to 2000Hz (Nyquist = 1000Hz, so nothing could ever be "above
    1000Hz" - the feature would be identically 0 for every recording, and
    z-scoring a zero-variance feature risks amplifying pure FFT noise into a
    huge, spurious distance contribution). Using a Nyquist-relative split
    keeps this feature meaningful at ANY sample rate, including 2kHz.
    """
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    zero_crossings = np.nonzero(np.diff(waveform > 0))[0]
    # RATE-INVARIANT zero-crossing rate: crossings per SECOND, not per sample.
    # ZCR expressed per-sample (crossings / len) scales with 1/sample_rate, so the
    # same physical vibration would score ~8x differently at 2 kHz vs 16 kHz.
    # Expressing it as crossings/sec keeps this feature meaningful independently of
    # the capture rate (matters now that the real piezo hardware runs at 2 kHz).
    zcr = float(len(zero_crossings) / (len(waveform) / sr))

    n = len(waveform)
    fft_vals = np.fft.rfft(waveform)
    fft_freqs = np.fft.rfftfreq(n, 1.0 / sr)
    fft_mag = np.abs(fft_vals)

    sum_mag = np.sum(fft_mag)
    centroid = float(np.sum(fft_freqs * fft_mag) / sum_mag) if sum_mag > 0 else 0.0

    nyquist = sr / 2.0
    high_freq_cutoff = high_freq_split_ratio * nyquist
    high_freq_mask = fft_freqs > high_freq_cutoff
    sum_mag_sq = np.sum(fft_mag ** 2)
    high_freq_ratio = float(np.sum(fft_mag[high_freq_mask] ** 2) / sum_mag_sq) if sum_mag_sq > 0 else 0.0

    peak_freq = float(fft_freqs[np.argmax(fft_mag)]) if sum_mag > 0 else 0.0

    # Autocorrelation-based Fundamental Frequency (F0) with safety bounds.
    # FFT-based (Wiener-Khinchin) instead of np.correlate(..., mode="full"):
    # the direct form is O(n^2) and becomes seconds-slow on anything longer
    # than a short clip; this is O(n log n) and gives the same result.
    signal = waveform - np.mean(waveform)
    n_fft = 2 * len(signal)
    f = np.fft.rfft(signal, n=n_fft)
    corr_full = np.fft.irfft(f * np.conj(f), n=n_fft)
    corr = corr_full[:len(signal)]
    d = np.diff(corr)

    start_candidates = np.where(d > 0)[0]
    start = start_candidates[0] if len(start_candidates) > 0 else 20

    peak_offset = np.argmax(corr[start:]) if len(corr[start:]) > 0 else 0
    peak = peak_offset + start

    fundamental_freq = float(sr / peak) if peak > 0 else 0.0

    return np.array([rms, zcr, centroid, high_freq_ratio, peak_freq, fundamental_freq], dtype=np.float32)


def run_vibration_pipeline(waveform: np.ndarray, sr: int = None):
    """Full Vibration Pipeline: vibration -> DSP features -> K-means -> result dict.

    `waveform` should be the raw ADC signal; it is DC-centered internally so
    callers don't need to remember to do it themselves. Returns None if no
    vibration model is deployed (kmeans_vibration_*.npy missing).
    """
    t0 = time.time()
    sr = sr or VIBRATION_SAMPLE_RATE_HZ

    center = load_kmeans_vibration()
    std = load_kmeans_vibration_std()
    if center is None or std is None:
        return None

    centered = waveform - np.mean(waveform)
    features = extract_dsp_features(centered, sr)
    t_features = time.time() - t0

    threshold = load_config().get("vibration_anomaly_threshold", 4.5552)
    # Guard against near-zero std (e.g. high_freq_ratio at low sample rates)
    # amplifying pure floating-point noise into a huge false-anomaly signal -
    # a feature that doesn't vary gets excluded from the distance, not blown up.
    safe_std = np.where(std < MIN_VIBRATION_STD, 1e6, std)
    distance = float(np.sqrt(np.sum(((features - center) / safe_std) ** 2)))

    if distance < threshold * 0.5:
        status = "healthy"
    elif distance < threshold:
        status = "watch"
    else:
        status = "anomaly"

    t_total = time.time() - t0

    return {
        "available": True,
        "status": status,
        "score": distance,               # raw distance - NOT 0-1, see fuse_for_ui()
        "threshold": threshold,
        "is_anomaly": distance > threshold,
        "sample_rate_hz": sr,
        "features": {
            "rms": float(features[0]),
            "zcr": float(features[1]),
            "centroid_hz": float(features[2]),
            "high_freq_ratio": float(features[3]),
            "peak_freq_hz": float(features[4]),
            "fundamental_freq_hz": float(features[5]),
        },
        "timing": {
            "dsp_ms": round(t_features * 1000, 2),
            "total_ms": round(t_total * 1000, 2),
        },
    }


def calibrate_vibration_baseline(waveform: np.ndarray, sr: int = None) -> np.ndarray:
    """Single-clip calibration (matches the reference Streamlit sidebar
    button): saves ONE recording's features as the center with std=1.0 for
    every dimension. Fast, but std=1.0 is a placeholder, not a measured
    spread - prefer calibrate_vibration_baseline_multi() when you can."""
    sr = sr or VIBRATION_SAMPLE_RATE_HZ
    centered = waveform - np.mean(waveform)
    features = extract_dsp_features(centered, sr)
    np.save(KMEANS_VIB_PATH, features)
    np.save(KMEANS_VIB_STD_PATH, np.ones_like(features))
    clear_model_cache()
    return features


# Below this floor, a feature's std across your calibration recordings is
# treated as "genuinely doesn't vary" and excluded from scoring (its z-score
# contribution is forced to 0) rather than blown up by dividing by ~0. This
# is what protects a Nyquist-degenerate feature (like high_freq_ratio at low
# sample rates) from dominating the anomaly distance with pure FFT noise.
MIN_VIBRATION_STD = 1e-4


def calibrate_vibration_baseline_multi(waveforms: list, sr: int = None) -> tuple:
    """Proper calibration from SEVERAL known-healthy recordings (recommended
    over the single-clip version above, especially at low sample rates where
    at least one feature dimension may be degenerate).

    `waveforms`: list of raw ADC arrays, one per healthy recording. Aim for
    at least 5-10 recordings spanning normal playing/idle variation.

    Returns (center, std) and saves them as the new deployed vibration model.
    """
    sr = sr or VIBRATION_SAMPLE_RATE_HZ
    if len(waveforms) < 2:
        raise ValueError(
            "Need at least 2 recordings to measure a real per-feature std; "
            "use calibrate_vibration_baseline() for a single-clip fallback."
        )

    all_features = []
    for wf in waveforms:
        centered = wf - np.mean(wf)
        all_features.append(extract_dsp_features(centered, sr))
    all_features = np.stack(all_features, axis=0)  # (n_recordings, 6)

    center = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)

    # A feature that never varies across healthy recordings (std ~0) would
    # otherwise blow up 1/std in the z-score - flag it as "don't use this
    # dimension" by giving it a very large std instead, so (x-center)/std
    # collapses toward 0 for that feature rather than exploding.
    degenerate = std < MIN_VIBRATION_STD
    if np.any(degenerate):
        dims = ["rms", "zcr", "centroid_hz", "high_freq_ratio", "peak_freq_hz", "fundamental_freq_hz"]
        flagged = [dims[i] for i in np.where(degenerate)[0]]
        print(f"[calibrate_vibration_baseline_multi] WARNING: near-zero variance in "
              f"{flagged} at sr={sr}Hz - excluding from anomaly scoring.")
        std = np.where(degenerate, 1e6, std)

    np.save(KMEANS_VIB_PATH, center.astype(np.float32))
    np.save(KMEANS_VIB_STD_PATH, std.astype(np.float32))
    clear_model_cache()
    return center, std


# =============================================================================
# FUSION HELPERS
# =============================================================================

_STATUS_SEVERITY = {"healthy": 0, "watch": 1, "anomaly": 2}
_SEVERITY_STATUS = {0: "healthy", 1: "watch", 2: "anomaly"}
_STATUS_DISPLAY = {
    "healthy": "✅ HEALTHY",
    "watch": "⚠️ WATCH",
    "anomaly": "🚨 ANOMALY",
}


def _severity_from_status(status: str) -> int:
    return _STATUS_SEVERITY.get(status, 0)


def fuse_status(audio_result: dict, vibration_result) -> str:
    """Worst-of-both-sensors fusion, matching the reference app's logic."""
    if vibration_result is None:
        return audio_result["status"]
    sev_aud = _severity_from_status(audio_result["status"])
    sev_vib = _severity_from_status(vibration_result["status"])
    return _SEVERITY_STATUS[max(sev_aud, sev_vib)]


def notify_mcu(status: str):
    """Send status to MCU for LED feedback."""
    code = {"healthy": 0, "watch": 1, "anomaly": 2, "recording": 3, "processing": 4}.get(status, 2)
    Bridge.notify("set_status", code)


def _ui_score(result) -> float:
    """Normalizes a raw distance/threshold pair to 0-1 for progress bars:
    0 = at the learned center, 1 = at-or-beyond 2x threshold. Matches the
    reference app's own st.progress(score/(threshold*2)) convention."""
    if result is None or not result.get("available"):
        return 0.0
    threshold = result.get("threshold") or 1e-9
    return max(0.0, min(1.0, result["score"] / (2 * threshold)))


# =============================================================================
# FILE-BASED ENTRY POINT (for engine.py's continuous recorder -> saved
# {prefix}_piezo.csv / {prefix}_audio.wav files on disk)
# =============================================================================

def _read_audio_wav(wav_path: str):
    """Reads a mono 16-bit PCM WAV file into a float32 [-1, 1] array."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2 or n_channels != 1 or not raw:
        raise ValueError("Unsupported audio format (expected mono 16-bit PCM).")

    usable = len(raw) // 2
    ints = struct.unpack(f"<{usable}h", raw[:usable * 2])
    audio = np.array(ints, dtype=np.float32) / 32768.0
    return audio, sr


def _read_piezo_raw_adc(csv_path: str) -> np.ndarray:
    """Reads engine.py's {prefix}_piezo.csv (sample_index,real_time_s,raw_adc,
    amplitude,voltage) and returns the raw_adc column as a float array."""
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            adc_col = header.index("raw_adc")
        except ValueError:
            adc_col = 2  # fall back to engine.py's known column position

        values = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= adc_col:
                continue
            try:
                values.append(float(parts[adc_col]))
            except ValueError:
                continue
    return np.array(values, dtype=np.float32)


@dataclass
class AnomalyReport:
    status: str
    score: float
    confidence: float
    std_dev: float
    max_deviation: float
    sample_count: int
    piezo: dict
    audio: dict


def _build_report(audio_result, vib_result, audio_err, vib_err, raw_adc) -> dict:
    """Shared fusion + JSON-shaping logic used by both the file-based and
    live in-memory evaluation paths, so the two never drift out of sync."""
    piezo_dict = vib_result or {
        "available": False, "status": "no_data", "score": 0.0,
        "details": vib_err or "No piezo data available.",
    }
    audio_dict = audio_result or {
        "available": False, "status": "no_data", "score": 0.0,
        "details": audio_err or "No audio data available.",
    }

    if audio_result is None and vib_result is None:
        return {
            "status": "No Data",
            "score": 0.0,
            "confidence": 0.0,
            "std_dev": 0.0,
            "max_deviation": 0.0,
            "sample_count": 0,
            "details": "No piezo or audio data scored for this window.",
            "piezo": piezo_dict,
            "audio": audio_dict,
        }

    fused = fuse_status(audio_result or {"status": "healthy"}, vib_result)
    notify_mcu(fused)

    # UI-facing score: 0-1, worst of the two normalized sensor distances
    # (see _ui_score docstring). Raw distances stay inside piezo/audio.
    ui_score = max(_ui_score(audio_result), _ui_score(vib_result))

    streams_available = int(audio_result is not None) + int(vib_result is not None)
    confidence = round(min(99.8, (75.0 if streams_available == 2 else 60.0)
                            + _severity_from_status(fused) * 12.0), 1)

    std_dev = float(np.std(raw_adc)) if len(raw_adc) else 0.0
    max_deviation = float(np.max(np.abs(raw_adc - np.mean(raw_adc)))) if len(raw_adc) else 0.0
    sample_count = (len(raw_adc) if len(raw_adc) else 0) + (
        int(audio_result["n_windows"]) if audio_result else 0
    )

    report = AnomalyReport(
        status=_STATUS_DISPLAY[fused],
        score=round(ui_score, 4),
        confidence=confidence,
        std_dev=round(std_dev, 4),
        max_deviation=round(max_deviation, 4),
        sample_count=sample_count,
        piezo=piezo_dict,
        audio=audio_dict,
    )
    return report.__dict__


# Minimum samples required before attempting a live-window score - avoids
# noisy/unstable results (and FFT edge cases) in the first fraction of a
# second right after a recording starts.
MIN_LIVE_PIEZO_SAMPLES = 500     # ~0.25-1s depending on hardware rate
MIN_LIVE_AUDIO_SAMPLES = 4000    # ~0.25s at 16kHz


class AnomalyDetectionModel:
    """Two entry points, both producing the same JSON shape:

    - evaluate(recordings_dir, prefix): file-based, for a COMPLETED
      recording (reads the finished CSV + WAV from disk).
    - evaluate_live(...): in-memory, for an ONGOING recording (reads a
      bounded recent window straight from engine.py's live buffers - no
      file I/O, no unbounded growth, and audio works before the WAV exists).
    """

    @staticmethod
    def evaluate(recordings_dir: str, prefix: str, piezo_sr: int = None) -> dict:
        """`piezo_sr`: pass the caller's real hardware rate explicitly when
        known (e.g. engine.PIEZO_SAMPLE_RATE_HZ) - falls back to
        VIBRATION_SAMPLE_RATE_HZ's config-driven default otherwise."""
        piezo_sr = piezo_sr or VIBRATION_SAMPLE_RATE_HZ
        audio_result = None
        vib_result = None
        audio_err = None
        vib_err = None
        raw_adc = np.array([])

        wav_path = os.path.join(recordings_dir, f"{prefix}_audio.wav")
        if os.path.exists(wav_path):
            try:
                audio, sr = _read_audio_wav(wav_path)
                if sr != TARGET_SR:
                    # Simple linear resample - avoids a hard scipy dependency here.
                    n_target = int(len(audio) * TARGET_SR / sr)
                    audio = np.interp(
                        np.linspace(0, len(audio), n_target, endpoint=False),
                        np.arange(len(audio)), audio
                    ).astype(np.float32)
                audio_result = run_pipeline(audio)
            except Exception as exc:
                audio_err = str(exc)

        csv_path = os.path.join(recordings_dir, f"{prefix}_piezo.csv")
        if os.path.exists(csv_path):
            try:
                raw_adc = _read_piezo_raw_adc(csv_path)
                if len(raw_adc) > 0:
                    vib_result = run_vibration_pipeline(raw_adc, sr=piezo_sr)
                    if vib_result is None:
                        vib_err = "Vibration K-means model not deployed (models/kmeans_vibration_*.npy missing)."
            except Exception as exc:
                vib_err = str(exc)

        return _build_report(audio_result, vib_result, audio_err, vib_err, raw_adc)

    @staticmethod
    def evaluate_live(piezo_raw_adc, piezo_sr: int, audio_int16, audio_sr: int) -> dict:
        """Scores a short in-memory window directly - the caller (engine.py)
        owns the real hardware sample rates and passes them explicitly, so
        this never has to guess. `piezo_raw_adc`: recent raw ADC ints.
        `audio_int16`: recent raw 16-bit PCM audio ints."""
        audio_result = None
        vib_result = None
        audio_err = None
        vib_err = None
        raw_adc = np.asarray(piezo_raw_adc, dtype=np.float32) if piezo_raw_adc else np.array([])

        if audio_int16:
            if len(audio_int16) < MIN_LIVE_AUDIO_SAMPLES:
                audio_err = f"Buffering audio ({len(audio_int16)}/{MIN_LIVE_AUDIO_SAMPLES} samples so far)."
            else:
                try:
                    audio = np.asarray(audio_int16, dtype=np.float32) / 32768.0
                    if audio_sr != TARGET_SR:
                        n_target = int(len(audio) * TARGET_SR / audio_sr)
                        audio = np.interp(
                            np.linspace(0, len(audio), n_target, endpoint=False),
                            np.arange(len(audio)), audio
                        ).astype(np.float32)
                    audio_result = run_pipeline(audio)
                except Exception as exc:
                    audio_err = str(exc)
        else:
            audio_err = "No audio samples buffered yet."

        if len(raw_adc) > 0:
            if len(raw_adc) < MIN_LIVE_PIEZO_SAMPLES:
                vib_err = f"Buffering piezo data ({len(raw_adc)}/{MIN_LIVE_PIEZO_SAMPLES} samples so far)."
            else:
                try:
                    vib_result = run_vibration_pipeline(raw_adc, sr=piezo_sr)
                    if vib_result is None:
                        vib_err = "Vibration K-means model not deployed (models/kmeans_vibration_*.npy missing)."
                except Exception as exc:
                    vib_err = str(exc)
        else:
            vib_err = "No piezo samples buffered yet."

        return _build_report(audio_result, vib_result, audio_err, vib_err, raw_adc)
