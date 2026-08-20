"""
model.py — SwarCare AI Pipeline (shared backend)
==================================================
Single source of truth for the Veena diagnostic pipeline.

Pipeline architecture (Parallel Hybrid):
  Physics Tuning:  WAV (16kHz) -> autocorrelation F0 -> per-string cents
                   deviation -> IN_TUNE / FLAT / SHARP / NO_PITCH
  ML Quality:      WAV (16kHz) -> YAMNet (521-D, averaged across frames)
                   + 6 pitch/physics features (527-D total)
                   -> StandardScaler -> RandomForest -> fault class

  Both branches run SIMULTANEOUSLY in two threads and results are merged.
"""

import os
import sys
import json
import struct
import time
import threading
import wave
from datetime import timezone, timedelta
from typing import Optional

import numpy as np

# Ensure the directory containing model.py is always on the Python path so that
# sibling modules (physics_pitch_engine.py) are importable regardless of the
# current working directory (e.g. when streamlit is launched from the repo root).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

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
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")

# --- Veena diagnostic model paths (lives in models/veena/) ---
VEENA_MODELS_DIR = os.path.join(MODELS_DIR, "veena")
VEENA_CLASSIFIER_PATH = os.path.join(VEENA_MODELS_DIR, "quality_classifier.joblib")
VEENA_SCALER_PATH = os.path.join(VEENA_MODELS_DIR, "scaler.joblib")
VEENA_PHYSICS_CONFIG_PATH = os.path.join(VEENA_MODELS_DIR, "physics_config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


CONFIG = load_config()
TARGET_SR = int(CONFIG.get("sample_rate", 16000))          # audio (YAMNet) rate
WINDOW_SEC = float(CONFIG.get("window_seconds", 2.0))


# =============================================================================
# MODEL LOADING (plain functional caching - framework agnostic; works whether
# or not a caller wraps these in @st.cache_resource elsewhere)
# =============================================================================

_cache = {}


# ---------------------------------------------------------------------------
# TFLite runtime discovery — ordered from lightest to heaviest.
# We try every known package name + module path variant so the code works
# across ai-edge-litert, tflite-runtime, and full TensorFlow installs.
# ---------------------------------------------------------------------------
_TFLITE_INTERPRETER_CLS = None
_TFLITE_IMPORT_ERROR: str = ""

def _discover_tflite_interpreter():
    """Probe every known TFLite import path and cache the first working class."""
    global _TFLITE_INTERPRETER_CLS, _TFLITE_IMPORT_ERROR
    if _TFLITE_INTERPRETER_CLS is not None:
        return _TFLITE_INTERPRETER_CLS

    candidates = [
        # ai-edge-litert >= 1.0  (Google's newest edge runtime)
        ("ai_edge_litert.interpreter",                   "Interpreter"),
        # ai-edge-litert < 1.0 used a different sub-package path
        ("ai_edge_litert",                               "Interpreter"),
        # tflite-runtime (Raspberry Pi / Linux ARM wheels)
        ("tflite_runtime.interpreter",                   "Interpreter"),
        # Full TensorFlow (desktop/server fallback)
        ("tensorflow.lite.python.interpreter",           "Interpreter"),
        # Older TF2 path
        ("tensorflow.lite.interpreter",                  "Interpreter"),
    ]

    errors = []
    for mod_path, cls_name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                _TFLITE_INTERPRETER_CLS = cls
                return cls
            errors.append(f"{mod_path}.{cls_name}: attribute not found")
        except ImportError as exc:
            errors.append(f"{mod_path}: {exc}")
        except Exception as exc:
            errors.append(f"{mod_path}: unexpected error — {exc}")

    _TFLITE_IMPORT_ERROR = " | ".join(errors)
    return None


# Run discovery once at import time (cached in the module-level global).
_discover_tflite_interpreter()


def load_interpreter():
    """Load YAMNet TFLite interpreter.

    Raises ImportError with a clear diagnostic message listing every runtime
    that was tried so the user knows exactly what to install.
    """
    if "interpreter" in _cache:
        return _cache["interpreter"]

    cls = _discover_tflite_interpreter()
    if cls is None:
        raise ImportError(
            "No TFLite runtime found. Install one of:\n"
            "  pip install ai-edge-litert          # recommended (Google Edge AI)\n"
            "  pip install tflite-runtime           # Raspberry Pi / Linux ARM\n"
            "  pip install tensorflow               # full TensorFlow (large)\n"
            f"Attempted imports:\n  {_TFLITE_IMPORT_ERROR}"
        )

    interp = cls(model_path=YAMNET_PATH)
    interp.allocate_tensors()
    _cache["interpreter"] = interp
    return interp


def clear_model_cache():
    """Call after (re)calibration so the next inference reloads fresh files."""
    _cache.clear()


# =============================================================================
# AUDIO PIPELINE: WAV -> YAMNet -> 521-D averaged embeddings
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


# =============================================================================
# FILE-BASED HELPER
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


# Minimum samples required before attempting a live-window score - avoids
# noisy/unstable results (and FFT edge cases) in the first fraction of a
# second right after a recording starts.
MIN_LIVE_AUDIO_SAMPLES = 4000    # ~0.25s at 16kHz


# =============================================================================
# VEENA DIAGNOSTIC MODEL — Parallel Hybrid Architecture
# =============================================================================
# CRITICAL DESIGN RULE: Physics Pitch Engine and ML Quality Classifier run
# SIMULTANEOUSLY in two threads and results are merged. Do NOT run the physics
# engine as a sequential gate before the ML model — structural defects distort
# pitch >±15 cents, causing a sequential gate to block 69.6% of fault audio
# from ever reaching the ML classifier.
# =============================================================================

_veena_cache: dict = {}


def _load_veena_classifier():
    """Lazy-load the scikit-learn RandomForest quality classifier."""
    if "clf" not in _veena_cache:
        if not os.path.exists(VEENA_CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"Veena quality classifier not found at {VEENA_CLASSIFIER_PATH}. "
                "Run veena_pipeline/export_trained_models.py first."
            )
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for the Veena classifier. pip install joblib")
        _veena_cache["clf"] = joblib.load(VEENA_CLASSIFIER_PATH)
    return _veena_cache["clf"]


def _load_veena_scaler():
    """Lazy-load the feature StandardScaler."""
    if "scaler" not in _veena_cache:
        if not os.path.exists(VEENA_SCALER_PATH):
            raise FileNotFoundError(
                f"Veena scaler not found at {VEENA_SCALER_PATH}."
            )
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required. pip install joblib")
        _veena_cache["scaler"] = joblib.load(VEENA_SCALER_PATH)
    return _veena_cache["scaler"]


def _load_veena_physics_config() -> dict:
    """Load per-string target frequencies and cents threshold from JSON."""
    if "phys_cfg" not in _veena_cache:
        if os.path.exists(VEENA_PHYSICS_CONFIG_PATH):
            with open(VEENA_PHYSICS_CONFIG_PATH) as f:
                _veena_cache["phys_cfg"] = json.load(f)
        else:
            # Safe default: Sa ~130.81 Hz (C3 tonic), ±15 cents tolerance
            _veena_cache["phys_cfg"] = {
                "tonic_hz": 130.81,
                "cents_threshold": 15.0,
                "strings": {
                    "S1": {"target_hz": 261.62, "ratio": 2.0, "name": "S1 — Sarani (Tara Sa)"},
                    "S2": {"target_hz": 196.22, "ratio": 1.5, "name": "S2 — Panchama (Pa)"},
                    "S3": {"target_hz": 130.81, "ratio": 1.0, "name": "S3 — Mandra Sa (tonic)"},
                    "S4": {"target_hz": 98.11,  "ratio": 0.75, "name": "S4 — Anumandra (lower Pa)"},
                    "T1": {"target_hz": 523.25, "ratio": 4.0, "name": "T1 — Chikari 1 (Sa, 2 oct)"},
                    "T2": {"target_hz": 784.87, "ratio": 6.0, "name": "T2 — Chikari 2 (Pa, 2 oct)"},
                    "T3": {"target_hz": 1046.50, "ratio": 8.0, "name": "T3 — Chikari 3 (Sa, 3 oct)"},
                },
            }
    return _veena_cache["phys_cfg"]


def clear_veena_cache():
    """Call to reload Veena model files (e.g. after retraining)."""
    _veena_cache.clear()


# ---------------------------------------------------------------------------
# YAMNet feature extraction (reuses the existing load_interpreter / run_yamnet)
# ---------------------------------------------------------------------------

def _extract_yamnet_embedding(audio: np.ndarray) -> np.ndarray:
    """Returns a 521-dim YAMNet averaged embedding for the given audio clip."""
    interpreter = load_interpreter()  # Reuse the existing cached interpreter
    scores, _ = run_yamnet(interpreter, audio)
    return scores  # shape (521,)


def _extract_veena_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Build the 527-dim feature vector used by the Veena RF quality classifier.

    Features (matches train_and_evaluate.py & export_trained_models.py):
      [0:521]   YAMNet 521-class averaged embeddings
      [521]     F0 fundamental frequency via autocorrelation (Hz)
      [522]     min_cents deviation across target string frequencies
      [523]     s1_err deviation from S1 Sarani (146.83 Hz)
      [524]     detuning_trigger boolean flag
      [525]     Energy Decay Rate (dRMS/dt across onset and tail)
      [526]     High-Frequency Spectral Flatness (>2000 Hz sub-band)
    """
    yamnet_521 = _extract_yamnet_embedding(audio)  # (521,)

    # 1. Fundamental frequency F0 via autocorrelation
    signal = audio.astype(np.float32) - float(np.mean(audio))
    n_sig = len(signal)
    if n_sig > 0:
        n_fft = 2 * n_sig
        fft_c = np.fft.rfft(signal, n=n_fft)
        r = np.fft.irfft(fft_c * np.conj(fft_c), n=n_fft)[:n_sig]
        min_lag, max_lag = int(sr / 400.0), int(sr / 45.0)
        if len(r) > max_lag and max_lag > min_lag:
            peak_lag = min_lag + int(np.argmax(r[min_lag:max_lag]))
            f0_val = float(sr / float(peak_lag)) if peak_lag > 0 else 0.0
        else:
            f0_val = 0.0
    else:
        f0_val = 0.0

    # 2-4. Cents deviation, S1 error, Detuning trigger
    target_f0s = np.array([146.83, 110.00, 73.42, 55.00], dtype=np.float32)
    s1_target = 146.83
    if f0_val > 30.0:
        cents_err = np.abs(1200.0 * np.log2(f0_val / target_f0s))
        min_cents = float(np.min(cents_err))
        s1_err = float(f0_val - s1_target)
        detuning_trigger = float((-25.0 <= s1_err <= -8.0) or (150.0 <= min_cents <= 400.0))
    else:
        min_cents = 1200.0
        s1_err = -100.0
        detuning_trigger = 0.0

    # 5. Energy Decay Rate (dRMS/dt): differentiates healthy decay from structural defects
    onset_samples = int(0.3 * sr)
    tail_samples = int(1.2 * sr)
    rms_onset = float(np.sqrt(np.mean(audio[:onset_samples] ** 2))) if len(audio) >= onset_samples else float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    rms_tail = float(np.sqrt(np.mean(audio[tail_samples:] ** 2))) if len(audio) > tail_samples else 0.0
    decay_rate = float((rms_onset - rms_tail) / (rms_onset + 1e-6))

    # 6. High-Frequency Spectral Flatness (>2000 Hz)
    flatness_hf = 0.0
    try:
        import librosa
        stft_hf = np.abs(librosa.stft(audio.astype(np.float32), n_fft=2048, hop_length=512))[256:, :]
        if stft_hf.size > 0:
            flatness_hf = float(np.mean(librosa.feature.spectral_flatness(S=stft_hf)))
    except Exception:
        n_fft = 2048
        fft_mag = np.abs(np.fft.rfft(audio.astype(np.float32), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        hf_mask = freqs > 2000.0
        hf_mag = fft_mag[hf_mask]
        if len(hf_mag) > 0 and np.sum(hf_mag) > 0:
            geom_mean = np.exp(np.mean(np.log(hf_mag + 1e-10)))
            arith_mean = np.mean(hf_mag)
            flatness_hf = float(geom_mean / (arith_mean + 1e-10))

    pitch_features = np.array([f0_val, min_cents, s1_err, detuning_trigger, decay_rate, flatness_hf], dtype=np.float32)
    return np.concatenate([yamnet_521, pitch_features]).astype(np.float32)  # Exactly 527 dims


# ---------------------------------------------------------------------------
# Fault label mapping (must mirror train_and_evaluate.py's CLASS_MAP)
# ---------------------------------------------------------------------------
_VEENA_FAULT_LABELS: dict = {
    0:  "Healthy",
    1:  "Healthy",   # class 1 maps to Healthy in some export configs
    2:  "Fret Wear",
    3:  "String Corrosion",
    4:  "Bridge Tilt",
    5:  "Kudam Crack",
    6:  "Loose Peg",
    7:  "String Buzz",
    8:  "Sympathetic Resonance Dampening",
    9:  "Finish Degradation",
    10: "Detached Bridge",
    11: "Nut Groove Wear",
}


# ---------------------------------------------------------------------------
# Acoustic Sound Type & Non-Veena / Speech Validation
# ---------------------------------------------------------------------------

def classify_audio_sound_type(audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Evaluates whether audio is silence, human speech, ambient noise, or genuine
    Saraswati Veena string resonance.

    Analyzes the active pluck excitation window using:
      1. Energy & Silence Gate.
      2. Autocorrelation modal periodicity with bridge foldback lags.
      3. Harmonic Energy Ratio (HER) across integer string partials.
      4. Spectral peak prominence & flatness.
      5. YAMNet AudioSet event classification (when available).
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    # 1. Silence Gate
    if rms < 0.004 or peak_amp < 0.008:
        return {
            "is_veena": False,
            "sound_type": "Silence",
            "is_silent": True,
            "confidence": 0.99,
            "reason": "Signal below audible excitation threshold.",
        }

    # 2. Extract strongest 1.0-second active pluck/vibration window (handles decaying tails and delayed plucks)
    win_len = int(1.0 * sr)
    if len(audio) > win_len:
        hop = int(0.10 * sr)
        max_win_rms = 0.0
        best_st = 0
        for st in range(0, len(audio) - win_len + 1, hop):
            w_rms = float(np.sqrt(np.mean(audio[st:st + win_len] ** 2)))
            if w_rms > max_win_rms:
                max_win_rms = w_rms
                best_st = st
        sig = audio[best_st:best_st + win_len]
        if max_win_rms < 0.005:
            return {
                "is_veena": False,
                "sound_type": "Silence",
                "is_silent": True,
                "confidence": 0.99,
                "reason": "Decayed below audible excitation threshold.",
            }
    else:
        sig = audio

    sig = sig.astype(np.float32) - float(np.mean(sig))
    n = len(sig)
    if n < 512:
        return {
            "is_veena": False,
            "sound_type": "Too Short",
            "is_silent": False,
            "confidence": 0.5,
            "reason": "Audio segment too short.",
        }

    # 3. Autocorrelation Periodicity & F0 Candidate
    n_fft = 2 * n
    f = np.fft.rfft(sig, n=n_fft)
    corr = np.fft.irfft(f * np.conj(f), n=n_fft)[:n]
    norm_corr = corr / corr[0] if corr[0] > 0 else corr

    min_lag = max(1, int(sr / 1200.0))  # Max ~1200 Hz
    max_lag = min(len(norm_corr) - 1, int(sr / 45.0))    # Min ~45 Hz
    if max_lag > min_lag:
        peak_offset = int(np.argmax(norm_corr[min_lag:max_lag]))
        best_lag = min_lag + peak_offset
        autocorr_peak = float(norm_corr[best_lag])
        f0_est = float(sr / best_lag) if best_lag > 0 else 0.0
    else:
        autocorr_peak = 0.0
        f0_est = 0.0

    # Kudirai bridge foldback lags for lower Mandra/Anumandra strings
    lag_foldback_peaks = [autocorr_peak]
    for mult in [2.0, 3.0, 4.0]:
        t_lag = int(round(best_lag * mult))
        if t_lag < len(norm_corr) - 2:
            win = max(2, int(0.10 * t_lag))
            l_min = max(1, t_lag - win)
            l_max = min(len(norm_corr) - 1, t_lag + win)
            if l_max > l_min:
                lag_foldback_peaks.append(float(np.max(norm_corr[l_min:l_max + 1])))
    best_periodicity = max(lag_foldback_peaks) if lag_foldback_peaks else autocorr_peak

    # 4. Harmonic Energy Ratio (HER) across integer harmonics (with subharmonic foldback)
    fft_mag = np.abs(np.fft.rfft(sig))
    fft_freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total_energy = float(np.sum(fft_mag ** 2)) + 1e-10

    hers = []
    for div in [1.0, 2.0, 3.0, 4.0]:
        base_f = f0_est / div
        if base_f < 35.0:
            continue
        harmonic_energy = 0.0
        for k in range(1, 14):
            k_f0 = k * base_f
            if k_f0 > (sr / 2.0) - 50:
                break
            band = max(4.0, 0.045 * k_f0)
            mask = (fft_freqs >= k_f0 - band) & (fft_freqs <= k_f0 + band)
            harmonic_energy += float(np.sum(fft_mag[mask] ** 2))
        hers.append(float(harmonic_energy / total_energy))

    her = max(hers) if hers else 0.0

    # 5. Spectral Flatness
    geom_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
    arith_mean = np.mean(fft_mag) + 1e-10
    flatness = float(geom_mean / arith_mean)

    # 6. Spectral Peak Prominence (Ratio of top discrete peaks to average spectrum)
    top_10_mean = float(np.mean(np.sort(fft_mag)[-10:]))
    peak_prominence = float(top_10_mean / arith_mean)

    # 7. YAMNet Speech class probability (when interpreter is available)
    speech_score = 0.0
    music_score = 0.0
    try:
        interpreter = load_interpreter()
        yamnet_521, _ = run_yamnet(interpreter, sig)
        speech_score = float(np.max(yamnet_521[0:68]))
        music_score = float(np.max(yamnet_521[132:160]))
    except Exception:
        pass

    if speech_score > 0.25 and speech_score > (music_score * 1.5):
        return {
            "is_veena": False,
            "sound_type": "Human Speech / Voice",
            "is_silent": False,
            "f0_est": f0_est,
            "confidence": speech_score,
            "her": her,
            "reason": f"YAMNet acoustic classifier detected human speech/voice (confidence={speech_score*100:.1f}%).",
        }

    is_string_resonance = (
        (best_periodicity >= 0.65 and her >= 0.40)
        or (her >= 0.55)
        or (best_periodicity >= 0.55 and peak_prominence >= 35.0 and flatness < 0.62)
    )

    if is_string_resonance:
        return {
            "is_veena": True,
            "sound_type": "Veena String Resonance",
            "is_silent": False,
            "f0_est": f0_est,
            "periodicity": best_periodicity,
            "her": her,
            "confidence": 0.95,
            "reason": "String modal harmonic comb verified.",
        }

    if flatness > 0.45 or best_periodicity < 0.35:
        return {
            "is_veena": False,
            "sound_type": "Ambient Noise",
            "is_silent": False,
            "f0_est": f0_est,
            "confidence": 0.85,
            "reason": f"Diffuse non-harmonic noise distribution (flatness={flatness:.2f}, HER={her*100:.1f}%).",
        }

    return {
        "is_veena": False,
        "sound_type": "Human Speech / Voice",
        "is_silent": False,
        "f0_est": f0_est,
        "confidence": 0.85,
        "reason": f"Non-string acoustic event (periodicity={best_periodicity:.2f}, HER={her*100:.1f}%).",
    }


def _run_ml_quality(audio: np.ndarray, sr: int = 16000) -> dict:
    """ML branch: extract features → scale → classify structural quality."""
    t0 = time.time()
    try:
        features = _extract_veena_features(audio, sr)          # (527,)
        scaler = _load_veena_scaler()
        clf    = _load_veena_classifier()
        feat_scaled = scaler.transform(features.reshape(1, -1))
        pred_class  = int(clf.predict(feat_scaled)[0])
        proba       = clf.predict_proba(feat_scaled)[0]
        confidence  = round(float(np.max(proba)) * 100, 1)
        label       = _VEENA_FAULT_LABELS.get(pred_class, f"Unknown ({pred_class})")
        is_healthy  = pred_class in (0, 1)
        return {
            "available": True,
            "fault_class": pred_class,
            "label": label,
            "is_healthy": is_healthy,
            "confidence": confidence,
            "timing_ms": round((time.time() - t0) * 1000),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _run_physics_tuning(audio: np.ndarray, sr: int, tonic_hz: float, cents_threshold: float,
                        string_label: Optional[str] = None) -> dict:
    """Physics branch: run PhysicsPitchEngine and return per-string tuning dict."""
    t0 = time.time()
    try:
        import importlib
        ppe_mod = importlib.import_module("physics_pitch_engine")
        PhysicsPitchEngine = ppe_mod.PhysicsPitchEngine
        engine = PhysicsPitchEngine(
            tonic_hz=tonic_hz,
            cents_threshold=cents_threshold,
            sr=sr,
        )
        result = engine.run(audio, sr=sr)
        return {
            "available":   True,
            "status":      result.status,
            "f0_hz":       round(result.f0_hz, 3),
            "cents_dev":   round(result.cents_dev, 2),
            "hz_dev":      round(result.hz_dev, 3),
            "string_name": result.string_name,
            "string_num":  result.string_num,
            "target_hz":   round(result.target_hz, 3),
            "confidence":  round(result.confidence, 3),
            "message":     result.message,
            "method":      result.method,
            "timing_ms":   round((time.time() - t0) * 1000),
        }
    except ImportError as exc:
        return {
            "available": False,
            "error": (
                f"physics_pitch_engine.py could not be imported: {exc}. "
                f"Expected at: {os.path.join(_SCRIPT_DIR, 'physics_pitch_engine.py')}"
            ),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


class VeenaDiagnosticModel:
    """Parallel hybrid Veena diagnostic model with sound-type validation."""

    @staticmethod
    def evaluate(
        recordings_dir: str,
        prefix: str,
        tonic_hz: float = 130.81,
        cents_threshold: float = 15.0,
        string_label: Optional[str] = None,
        audio_sr: int = 16000,
    ) -> dict:
        """File-based evaluation: reads {prefix}_audio.wav and runs validation & diagnostics."""
        wav_path = os.path.join(recordings_dir, f"{prefix}_audio.wav")
        if not os.path.exists(wav_path):
            return {
                "available": False,
                "error": f"WAV file not found: {wav_path}",
                "status": "No Data",
                "is_healthy": False,
                "tuning": {"available": False, "error": "No audio file"},
                "quality": {"available": False, "error": "No audio file"},
            }
        try:
            audio, sr = _read_audio_wav(wav_path)
            if sr != audio_sr:
                n_target = int(len(audio) * audio_sr / sr)
                audio = np.interp(
                    np.linspace(0, len(audio), n_target, endpoint=False),
                    np.arange(len(audio)), audio
                ).astype(np.float32)
        except Exception as exc:
            return {
                "available": False,
                "error": str(exc),
                "status": "Error",
                "is_healthy": False,
                "tuning": {"available": False, "error": str(exc)},
                "quality": {"available": False, "error": str(exc)},
            }
        return VeenaDiagnosticModel.evaluate_audio(
            audio, audio_sr, tonic_hz, cents_threshold, string_label
        )

    @staticmethod
    def evaluate_live(
        audio_int16: list,
        audio_sr: int,
        tonic_hz: float = 130.81,
        cents_threshold: float = 15.0,
        string_label: Optional[str] = None,
    ) -> dict:
        """In-memory evaluation: uses the live audio window from engine.py buffers."""
        if not audio_int16 or len(audio_int16) < MIN_LIVE_AUDIO_SAMPLES:
            msg = (
                f"Buffering audio ({len(audio_int16) if audio_int16 else 0}/"
                f"{MIN_LIVE_AUDIO_SAMPLES} samples)."
            )
            return {
                "available": False,
                "error": msg,
                "status": "Buffering",
                "is_healthy": False,
                "tuning": {"available": False, "error": msg},
                "quality": {"available": False, "error": msg},
            }
        audio = np.asarray(audio_int16, dtype=np.float32) / 32768.0
        return VeenaDiagnosticModel.evaluate_audio(
            audio, audio_sr, tonic_hz, cents_threshold, string_label
        )

    @staticmethod
    def evaluate_audio(
        audio: np.ndarray,
        sr: int,
        tonic_hz: float,
        cents_threshold: float,
        string_label: Optional[str],
    ) -> dict:
        """Core: validates sound type, launches physics + ML in parallel, merges results."""
        # 1. First Gate: Sound Type & Instrument Verification
        sound_info = classify_audio_sound_type(audio, sr)

        if sound_info.get("is_silent", False):
            return {
                "available": True,
                "status": "Silence",
                "is_healthy": False,
                "is_veena": False,
                "sound_type": "Silence",
                "message": "No audio signal detected (silence/idle). Please pluck a Saraswati Veena string.",
                "tuning": {
                    "available": True,
                    "status": "SILENCE",
                    "f0_hz": 0.0,
                    "cents_dev": 0.0,
                    "hz_dev": 0.0,
                    "string_name": "—",
                    "target_hz": 0.0,
                    "confidence": 0.0,
                    "message": "Audio stream is silent or below threshold.",
                    "method": "none",
                },
                "quality": {
                    "available": False,
                    "is_healthy": False,
                    "fault_class": -2,
                    "label": "Silence / Idle",
                    "confidence": 100.0,
                },
                "tonic_hz": round(tonic_hz, 3),
                "string_label": string_label or "auto",
                "sound_info": sound_info,
            }

        if not sound_info.get("is_veena", True):
            sound_type = sound_info.get("sound_type", "Non-Veena Sound")
            reason = sound_info.get("reason", "Acoustic signature does not match Saraswati Veena.")
            return {
                "available": True,
                "status": "Non-Veena Sound Detected",
                "is_healthy": False,
                "is_veena": False,
                "sound_type": sound_type,
                "message": f"{sound_type} detected. {reason}",
                "tuning": {
                    "available": True,
                    "status": "NON_VEENA",
                    "f0_hz": round(sound_info.get("f0_est", 0.0), 2),
                    "cents_dev": 0.0,
                    "hz_dev": 0.0,
                    "string_name": "Non-Instrument Sound",
                    "target_hz": 0.0,
                    "confidence": round(sound_info.get("confidence", 0.0), 2),
                    "message": f"{sound_type} detected. Please pluck a Saraswati Veena string.",
                    "method": "sound_classifier",
                },
                "quality": {
                    "available": False,
                    "is_healthy": False,
                    "fault_class": -1,
                    "label": sound_type,
                    "confidence": round(sound_info.get("confidence", 0.85) * 100, 1),
                },
                "tonic_hz": round(tonic_hz, 3),
                "string_label": string_label or "auto",
                "sound_info": sound_info,
            }

        # 2. Validated Veena Audio: Execute parallel physics tuning and ML structural classification
        physics_out: dict = {}
        ml_out: dict = {}

        def _physics_thread():
            nonlocal physics_out
            physics_out = _run_physics_tuning(audio, sr, tonic_hz, cents_threshold, string_label)

        def _ml_thread():
            nonlocal ml_out
            ml_out = _run_ml_quality(audio, sr)

        t_phys = threading.Thread(target=_physics_thread, daemon=True)
        t_ml   = threading.Thread(target=_ml_thread,     daemon=True)
        t_phys.start()
        t_ml.start()
        t_phys.join(timeout=10.0)
        t_ml.join(timeout=10.0)

        # If physics engine rejected the sound as non-veena or silence
        phys_status = physics_out.get("status", "NO_PITCH")
        if phys_status == "NON_VEENA":
            return {
                "available": True,
                "status": "Non-Veena Sound Detected",
                "is_healthy": False,
                "is_veena": False,
                "sound_type": "Non-Veena Sound",
                "message": physics_out.get("message", "Non-Veena sound detected."),
                "tuning": physics_out,
                "quality": {
                    "available": False,
                    "is_healthy": False,
                    "fault_class": -1,
                    "label": "Non-Veena Sound",
                    "confidence": 95.0,
                },
                "tonic_hz": round(tonic_hz, 3),
                "string_label": string_label or "auto",
                "sound_info": sound_info,
            }

        # Handle quality and health fusion
        tuning_ok = (phys_status in ["IN_TUNE", "FLAT", "SHARP"])
        if ml_out.get("available", False):
            quality_ok = ml_out.get("is_healthy", True)
            final_quality = ml_out
        else:
            # Fallback when ML classifier is buffering or TFLite is unavailable
            quality_ok = tuning_ok
            final_quality = {
                "available": False,
                "is_healthy": quality_ok,
                "fault_class": 0 if quality_ok else 1,
                "label": "Structurally Sound & Resonant" if quality_ok else "Tuning Deviation",
                "confidence": round(physics_out.get("confidence", 0.85) * 100, 1),
            }

        is_healthy = quality_ok and tuning_ok
        fused_status = "Healthy" if is_healthy else "Fault Detected"

        return {
            "available": True,
            "status":    fused_status,
            "is_healthy": is_healthy,
            "is_veena":  True,
            "sound_type": "Veena String Resonance",
            "tuning":    physics_out,
            "quality":   final_quality,
            "tonic_hz":  round(tonic_hz, 3),
            "string_label": string_label or "auto",
            "sound_info": sound_info,
        }