# SwarCare — AI/ML Layer Documentation

> **Project:** SwarCare v1.2.0  
> **Target Hardware:** Arduino UNO R4 WiFi (STM32U585 MCU) + Qualcomm QRB2210 MPU  
> **Generated:** 2026-08-21

---

## Table of Contents

1. [AI Architecture Overview](#1-ai-architecture-overview)
2. [Tech Stack & Models](#2-tech-stack--models)
3. [Core AI Components & Files](#3-core-ai-components--files)
4. [Prompt Engineering & Context Management](#4-prompt-engineering--context-management)
5. [AI Data Flow](#5-ai-data-flow)
6. [Key Constraints & Error Handling](#6-key-constraints--error-handling)

---

## 1. AI Architecture Overview

SwarCare is an **acoustic health monitoring system** for the Saraswati Veena (a classical South Indian plucked string instrument). It uses a dual-sensor setup — a **piezoelectric vibration sensor** (wired to the instrument body) and a **USB microphone** (capturing airborne sound) — paired with an Arduino UNO R4 WiFi acting as the edge data acquisition unit. The Qualcomm QRB2210 MPU runs the Python AI stack.

The system solves **three distinct AI problems simultaneously**, each handled by a dedicated pipeline:

| Pipeline | Problem | Signal Source |
|---|---|---|
| **Anomaly Detection** | Is the instrument producing an anomalous vibration or sound pattern? | Piezo ADC + Microphone WAV |
| **Veena Diagnostic (Tuning)** | Is a specific string in tune, and by how many cents? | Microphone WAV |
| **Veena Diagnostic (Structural Quality)** | Does the instrument have a structural or acoustic fault (e.g., cracked body, corroded string)? | Microphone WAV |

### High-Level Pipeline Diagram

```
┌─────────────────────────────────┐
│   Arduino UNO R4 (sketch.ino)   │
│   Piezo ADC @ 2000 Hz, 12-bit   │
│   40-sample batches via Bridge  │
└──────────────┬──────────────────┘
               │ Binary MsgPack (92 bytes/batch)
               ▼
┌─────────────────────────────────────────────────────────────┐
│                  SwarCareEngine (engine.py)                  │
│                                                             │
│  ┌──────────────────────┐   ┌────────────────────────────┐  │
│  │  Piezo Queue         │   │  ALSA arecord subprocess   │  │
│  │  → piezo_raw_window  │   │  → audio_raw_window        │  │
│  │  (deque, 3 s ring)   │   │  (deque, 3 s ring @ 16kHz) │  │
│  └──────────┬───────────┘   └──────────────┬─────────────┘  │
│             │                              │                 │
│             ▼                              ▼                 │
│  _piezo.csv (disk)               _audio.wav (disk)          │
└─────────────────────────────────────────────────────────────┘
               │                              │
               ▼                              ▼
┌───────────────────────────────────────────────────────────────┐
│                       model.py (AI Core)                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Pipeline A: Anomaly Detection                           │  │
│  │  WAV → YAMNet (521-D) → PCA (64-D) → K-means distance  │  │
│  │  CSV → 6-D DSP Features → Z-score K-means distance     │  │
│  │  Fusion: worst-of-two-sensors                          │  │
│  │  Output: healthy / watch / anomaly + score (0-1)       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Pipeline B: Veena Tuning (Physics Pitch Engine)         │  │
│  │  WAV → Sound Validation → Cepstral Lifter → HPS →      │  │
│  │  pYIN + 22-Sruti Soft Prior → Cents Decision            │  │
│  │  Output: IN_TUNE / FLAT / SHARP + cents deviation       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Pipeline C: Structural Quality (ML Classifier)          │  │
│  │  WAV → 527-D Feature Vec (YAMNet 521 + 6 DSP) →        │  │
│  │  StandardScaler → RandomForest → Fault Class            │  │
│  │  Output: Healthy / Fret Wear / Bridge Tilt / ... (12)  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Pipelines B & C run in PARALLEL threads — results fused     │
└───────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  FastAPI (main.py) — JSON REST + WS     │
│  /api/veena_analysis                    │
│  /api/anomaly_analysis                  │
│  /ws/telemetry                          │
└─────────────────────────────────────────┘
```

---

## 2. Tech Stack & Models

### Python Libraries (AI-Relevant)

| Library | Version Spec | Role |
|---|---|---|
| `numpy` | >= 1.24.0 | Core numerical computation for all DSP and linear algebra operations |
| `scipy` | >= 1.10.0 | Savitzky-Golay smoothing inside HPS; falls back to NumPy moving-average if absent |
| `librosa` | >= 0.10.0 | pYIN pitch tracking (Stage 3 of the Physics Pitch Engine); STFT for HF spectral flatness |
| `scikit-learn` | >= 1.2.0 | RandomForest quality classifier (`quality_classifier.joblib`); `StandardScaler` |
| `joblib` | >= 1.3.0 | Model serialization/deserialization for the scikit-learn artifacts |
| `tflite-runtime` | >= 2.14.0 | TFLite interpreter to run YAMNet on-device (Raspberry Pi / ARM Linux) |

**Alternative TFLite runtimes** (tried in order at import time):  
`ai_edge_litert` → `tflite_runtime` → `tensorflow.lite` (full TF, desktop fallback)

### Models

| Model File | Type | Dimensions | Purpose |
|---|---|---|---|
| `models/yamnet.tflite` | TFLite Neural Network (MobileNet-based) | Input: 15,600 samples (0.975 s @ 16kHz) → Output: 521-D AudioSet class scores | Audio feature extraction backbone for both the Anomaly pipeline and the Veena quality classifier |
| `models/pca_model.npz` | Pre-fitted PCA | 521-D → 64-D; stores `mean` (521,) + `components` (64, 521) | Dimensionality reduction of YAMNet embeddings for the Anomaly K-means step |
| `models/kmeans_center.npy` | K-means centroid | 64-D float32 array | Learned centroid of healthy audio embeddings in PCA space |
| `models/kmeans_std.npy` | K-means std | 64-D float32 array | Per-dimension std for audio (currently unused in main distance calc — raw Euclidean is used) |
| `models/kmeans_vibration_center.npy` | K-means centroid | 6-D float32 | Healthy centroid for piezo vibration DSP features |
| `models/kmeans_vibration_std.npy` | K-means std | 6-D float32 | Per-dimension std for Z-score normalization of vibration features |
| `models/veena/quality_classifier.joblib` | scikit-learn RandomForest | Input: 527-D → Output: 12-class fault label | Structural quality classifier for Veena audio |
| `models/veena/scaler.joblib` | scikit-learn StandardScaler | 527-D → 527-D | Feature normalization before the RF classifier |
| `models/veena/multimodal_classifier.joblib` | scikit-learn RandomForest (multimodal) | Larger variant | Extended multimodal classifier (available on disk, not yet wired into inference path) |
| `models/veena/vibration_classifier.joblib` | scikit-learn RandomForest | Vibration-specific | Vibration-only classifier (available on disk, not yet wired) |
| `models/config.json` | JSON config | — | Thresholds, sample rates, model file paths; runtime-configurable without code changes |

### Hardware / Signal Chain

| Component | Spec | Note |
|---|---|---|
| Piezo Sensor | 12-bit ADC, A0, 2000 Hz sample rate | Sampled on the STM32U585 MCU via Zephyr RTOS |
| Microphone | USB audio, 16kHz, 16-bit PCM mono | Captured by `arecord` subprocess on the MPU |
| MCU | Arduino UNO R4 WiFi (STM32U585) | Runs `sketch.ino`; sends 40-sample MsgPack batches every 20 ms |
| MPU | Qualcomm QRB2210 | Runs the Python AI stack |

---

## 3. Core AI Components & Files

### 3.1 `python/model.py` — The AI Core (1,417 lines)

**Single source of truth for all inference logic.** `engine.py` and `main.py` never implement model logic directly — they call `AnomalyDetectionModel` or `VeenaDiagnosticModel` from this file.

---

#### `load_interpreter()` / `load_pca()` / `load_kmeans()` / `load_kmeans_vibration()`
*Lines 182–237*

**What it does:** Lazy-loads the four model artifacts into a module-level `_cache` dict. Framework-agnostic — no `@st.cache_resource` required, but Streamlit can wrap these externally.

**TFLite Discovery** (`_discover_tflite_interpreter`, lines 141–175): Probes five known package paths at import time and caches the first working `Interpreter` class. Falls back gracefully with a clear diagnostic error listing every runtime that was tried.

**Inputs:** File system paths from `MODELS_DIR`.  
**Outputs:** Cached in-memory model objects (`Interpreter`, `(mean, components)` tuple, `np.ndarray` centroids).

---

#### `run_yamnet(interpreter, audio)` — Returns `(scores, n_windows)`
*Lines 249–280*

**What it does:** Runs YAMNet inference over an audio clip using a 50%-overlap sliding window strategy.

**Inputs:**
- `interpreter`: TFLite `Interpreter` instance (cached).
- `audio`: `np.ndarray`, float32, normalised to `[-1, 1]`, expected at 16 kHz.

**Outputs:**
- `scores`: `(521,)` float32 array — the **mean** of YAMNet's 521-class AudioSet softmax scores across all frames.
- `n_windows`: `int` — number of inference frames processed.

**Logic:**
- Window size derived from TFLite input tensor shape (15,600 samples = 0.975 s).
- Hop = `window_size // 2` (50% overlap).
- If the clip is shorter than one window, it is zero-padded and processed as a single frame.
- Tensor is dynamically resized if the shape does not match (handles variable-length inputs).

---

#### `run_pipeline(audio)` — Returns `dict`
*Lines 283–320*

**What it does:** Full audio anomaly detection pipeline — **YAMNet → PCA → K-means distance → thresholded verdict**.

**Inputs:**
- `audio`: `np.ndarray`, float32, 16 kHz, normalised.

**Outputs (dict):**
```json
{
  "available": true,
  "status": "healthy | watch | anomaly",
  "score": 0.312,
  "threshold": 0.6,
  "is_anomaly": false,
  "n_windows": 4,
  "timing": { "yamnet_ms": 120, "pca_ms": 1.2, "total_ms": 121 }
}
```

**Algorithmic logic:**
1. **YAMNet embeddings:** `scores` = averaged 521-D class scores across all windows.
2. **PCA projection:** `features = (scores - pca_mean) @ pca_components.T` → 64-D vector.
3. **Euclidean distance:** `d = ||features - kmeans_center||_2` (raw, not normalised).
4. **Thresholding:** `d < 0.5*threshold` → `"healthy"`, `d < threshold` → `"watch"`, otherwise `"anomaly"`.

> **Note:** The `score` in the output is a raw Euclidean distance, NOT a 0–1 value. The UI-normalised score (0–1) is computed by `_ui_score()` → `score / (2 * threshold)`.

---

#### `extract_dsp_features(waveform, sr)` — Returns `np.ndarray (6,)`
*Lines 327–387*

**What it does:** Extracts a 6-dimensional physical feature vector from a raw piezo vibration waveform.

**Inputs:**
- `waveform`: raw ADC float array (DC-bias removed by caller).
- `sr`: sample rate in Hz (default: `VIBRATION_SAMPLE_RATE_HZ`).

**Outputs:** `float32` array `[rms, zcr, centroid_hz, high_freq_ratio, peak_freq_hz, fundamental_freq_hz]`

| Index | Feature | Description |
|---|---|---|
| 0 | `rms` | Root Mean Square energy |
| 1 | `zcr` | Zero Crossing Rate (fraction of sign changes per sample) |
| 2 | `centroid_hz` | Spectral Centroid (energy-weighted mean frequency) |
| 3 | `high_freq_ratio` | Energy fraction above 50% of Nyquist (Nyquist-relative, not fixed-Hz) |
| 4 | `peak_freq_hz` | Peak FFT magnitude frequency |
| 5 | `fundamental_freq_hz` | F0 via FFT-based autocorrelation (Wiener-Khinchin, O(n log n)) |

> **Critical design note:** The `high_freq_ratio` cutoff is Nyquist-relative (`0.5 * sr/2`), not a fixed 1 kHz. At 2 kHz sample rate, Nyquist = 1000 Hz, so a fixed 1 kHz cutoff would create a zero-variance feature causing division-by-zero in Z-scoring.

---

#### `run_vibration_pipeline(waveform, sr)` — Returns `dict | None`
*Lines 390–444*

**What it does:** Full vibration anomaly pipeline — **DSP features → Z-score normalised K-means distance → verdict**.

**Inputs:**
- `waveform`: raw ADC float array (DC-centering applied internally).
- `sr`: piezo sample rate in Hz.

**Outputs (dict, or `None` if model not deployed):**
```json
{
  "available": true,
  "status": "healthy | watch | anomaly",
  "score": 1.82,
  "threshold": 4.5552,
  "features": { "rms": 0.02, "zcr": 0.11, "..." : "..." },
  "timing": { "dsp_ms": 3.5, "total_ms": 3.8 }
}
```

**Key guard:** Features with `std < 1e-4` (zero variance across calibration data) are excluded from the Z-score by replacing their `std` with `1e6`, forcing that dimension's contribution toward zero.

---

#### `calibrate_vibration_baseline(waveform)` / `calibrate_vibration_baseline_multi(waveforms)`
*Lines 447–510*

**What it does:** Updates the deployed vibration K-means model from known-healthy recordings and persists new `.npy` files.

- **Single-clip version:** Sets `center = features`, `std = 1.0` (placeholder). Fast but imprecise.
- **Multi-clip version (recommended):** Computes real `mean` and `std` across >= 2 recordings. Flags near-zero variance dimensions and excludes them from scoring.

---

#### `fuse_status(audio_result, vibration_result)` — Returns `str`
*Lines 530–540*

**What it does:** Implements **worst-of-both-sensors fusion** — the fused status is the more severe of the two sensor verdicts.

**Severity ordering:** `healthy (0) < watch (1) < anomaly (2)`.  
**Graceful degradation:** If either sensor result is `None`, the other sensor's verdict is used alone.

---

#### `AnomalyDetectionModel`
*Lines 677–770*

**Static class with two entry points:**

| Method | When Used | Data Source |
|---|---|---|
| `evaluate(recordings_dir, prefix, piezo_sr)` | After recording stops | Reads `{prefix}_piezo.csv` + `{prefix}_audio.wav` from disk |
| `evaluate_live(piezo_raw_adc, piezo_sr, audio_int16, audio_sr)` | During active recording | Reads from in-memory ring buffers — no file I/O |

Both return the **same JSON-serialisable dict shape** via `_build_report()` (lines 615–667):
```json
{
  "status": "HEALTHY | WATCH | ANOMALY",
  "score": 0.156,
  "confidence": 87.0,
  "std_dev": 0.041,
  "max_deviation": 0.113,
  "sample_count": 8400,
  "piezo": {},
  "audio": {}
}
```

**Minimum buffering thresholds** (live mode):
- Piezo: 500 samples (~0.25 s) before inference is attempted.
- Audio: 4,000 samples (~0.25 s at 16 kHz) before inference is attempted.

---

#### `classify_audio_sound_type(audio, sr)` — Returns `dict`
*Lines 952–1128*

**What it does:** Pre-inference validation gate — determines whether incoming audio is Saraswati Veena string resonance, human speech, ambient noise, or silence **before** committing compute to the quality classifier or physics engine.

**5-stage decision process:**

| Stage | Technique | Output |
|---|---|---|
| 1 | RMS/peak amplitude gate | Silence rejection (`rms < 0.004`) |
| 2 | Strongest 1-second window extraction | Finds the most energetic pluck onset |
| 3 | Autocorrelation periodicity + Kudirai bridge foldback lags | Measures periodic harmonic structure |
| 4 | Harmonic Energy Ratio (HER) across integer partials | Fraction of total energy in the harmonic comb |
| 5 | YAMNet AudioSet class 0–68 (speech) vs. 132–160 (music) | Secondary neural classification |

**Decision thresholds:**
- `is_string_resonance = True` if: `(periodicity >= 0.65 AND HER >= 0.40) OR (HER >= 0.55) OR (periodicity >= 0.55 AND peak_prominence >= 35 AND flatness < 0.62)`
- Speech detected if: `speech_score > 0.25 AND speech_score > music_score * 1.5`

---

#### `_extract_veena_features(audio, sr)` — Returns `np.ndarray (527,)`
*Lines 857–926*

**What it does:** Builds the 527-dimensional feature vector used by the Veena RF quality classifier.

**Feature vector breakdown:**

| Indices | Features | Source |
|---|---|---|
| `[0:521]` | YAMNet 521-class averaged embeddings | `_extract_yamnet_embedding()` |
| `[521]` | F0 fundamental frequency (Hz) | FFT-based autocorrelation, lag range: `sr/400` to `sr/45` |
| `[522]` | `min_cents` — minimum deviation from 4 target string F0s | `1200 * log2(f0 / target_f0s)` |
| `[523]` | `s1_err` — signed Hz deviation from S1 Sarani (146.83 Hz) | `f0 - 146.83` |
| `[524]` | `detuning_trigger` — boolean flag for detuning patterns | `(-25 <= s1_err <= -8) OR (150 <= min_cents <= 400)` |
| `[525]` | Energy Decay Rate (`dRMS/dt`) | `(rms_onset - rms_tail) / rms_onset`; onset = 0–0.3 s, tail = 1.2 s+ |
| `[526]` | High-Frequency Spectral Flatness (>2000 Hz sub-band) | `librosa.feature.spectral_flatness` on STFT bins above 2 kHz |

---

#### `_run_ml_quality(audio, sr)` — Returns `dict`
*Lines 1131–1153*

**What it does:** ML branch of the parallel Veena diagnostic — feature extraction → scaling → RandomForest classification.

**Outputs:**
```json
{
  "available": true,
  "fault_class": 3,
  "label": "String Corrosion",
  "is_healthy": false,
  "confidence": 87.3,
  "timing_ms": 45
}
```

**Fault class map (12 classes):**

| Class | Label |
|---|---|
| 0, 1 | Healthy |
| 2 | Fret Wear |
| 3 | String Corrosion |
| 4 | Bridge Tilt |
| 5 | Kudam Crack |
| 6 | Loose Peg |
| 7 | String Buzz |
| 8 | Sympathetic Resonance Dampening |
| 9 | Finish Degradation |
| 10 | Detached Bridge |
| 11 | Nut Groove Wear |

---

#### `VeenaDiagnosticModel`
*Lines 1196–1417*

**Parallel hybrid diagnostic model.** Runs physics tuning and ML structural quality **simultaneously in two daemon threads** and merges results.

> **Critical design rule (documented in code, lines 776–780):** Physics and ML run in parallel. Do NOT use the physics engine as a sequential gate before the ML classifier — structural defects distort pitch by >+/-15 cents, causing a sequential gate to block **69.6% of fault audio** from ever reaching the ML classifier.

**Entry points:**

| Method | Data Source |
|---|---|
| `evaluate(recordings_dir, prefix, ...)` | File-based (completed WAV) |
| `evaluate_live(audio_int16, audio_sr, ...)` | In-memory (live ring buffer) |
| `evaluate_audio(audio, sr, ...)` | Core implementation — called by both above |

**`evaluate_audio` flow:**
1. **Sound type gate** (`classify_audio_sound_type`) — rejects silence and non-Veena audio before spawning threads.
2. **Parallel execution:** `threading.Thread` for physics tuning + `threading.Thread` for ML quality; both joined with `timeout=10.0 s`.
3. **Fusion:** `is_healthy = quality_ok AND tuning_ok`. If ML unavailable, falls back to tuning result alone.

---

### 3.2 `python/physics_pitch_engine.py` — Deterministic DSP Pitch Gate (765 lines)

**Purpose:** A purely algorithmic (no learned model) deterministic DSP pipeline for Saraswati Veena string tuning detection, purpose-built to handle three known acoustic "loopholes" of the Kudirai bridge mechanism.

**Reference publications embedded in code:**
- Asokan et al. (2016) — Vibro-acoustic signatures of Saraswati Veena
- Chauhan et al. (2021) — Kudirai bridge overtone dynamics & harmonic revival
- de Cheveigné & Kawahara (2002) — YIN algorithm for F0 estimation

---

#### `PhysicsPitchEngine.run(audio, sr)` — Returns `PitchResult`
*Lines 213–264*

**Full 4-stage pipeline:**

**Stage 0 — Silence Gate** (line 230):  
`RMS < 0.008` → immediately return `PitchResult(status="SILENCE")`.

**Stage 1 — Cepstral Lifter** (`_cepstral_lifter`, lines 399–436):  
High-pass cepstral filter that decouples string excitation from Kudam body resonance (fixed structural formants at 280–300 Hz).  
Signal flow: `x[n] → FFT → log|X(f)| → IFFT → cepstrum → zero low-quefrency (<3.5 ms) → FFT → exp(.) → IFFT → lifted x'[n]`.

**Stage 2 — Harmonic Product Spectrum** (`_hps_f0_estimate`, lines 440–483):  
Collapses suppressed fundamental by multiplying R=4 downsampled copies of the magnitude spectrum: `Y(f) = product_{r=1}^{4} |X(r*f)|`. Corrects octave-doubling errors caused by the Kudirai bridge suppressing the string fundamental. Uses 8192-point FFT + Blackman window + Savitzky-Golay pre-smoothing.

**Stage 3 — Sruti-Aware pYIN** (`_sruti_aware_pyin`, lines 487–592):  
pYIN pitch tracking (`librosa.pyin`, range: A1–C7 / 55–2093 Hz) augmented with a **22-Sruti Gaussian soft prior** on the HMM.
- **Attack frames (first 200 ms):** Full weight `1.0`; pYIN freely locks onto F0.
- **Sustain frames:** Gaussian penalty `sigma = 70 cents` around each Sruti — frames far from any Carnatic Sruti are down-weighted. Prevents HMM from jumping to a revived upper harmonic mid-decay.
- If pYIN drifts >250 cents from the HPS hint, a 60/40 blend is applied.
- Falls back to autocorrelation pitch if `librosa` is not installed.

**Stage 3.5 — String Harmonicity Validation** (`_validate_string_harmonicity`, lines 310–395):  
Validates that the detected pitch corresponds to a genuine plucked-string harmonic comb (not voice or noise) by checking autocorrelation periodicity + Harmonic Energy Ratio (HER) + spectral flatness.

**Stage 4 — Cents Rule Engine** (`_cents_decision`, lines 596–675):  
`cents = 1200 * log2(f_detected / f_target)`. Finds the nearest of 7 Veena strings (S4/S3/S2/S1/T1/T2/T3), tests harmonic foldback factors [1.5x, 2.0x, 3.0x] for Kudirai bridge artifacts, then classifies:
- `|cents| <= threshold` → `IN_TUNE`
- `cents < 0` → `FLAT` + "tighten peg clockwise" message
- `cents > 0` → `SHARP` + "loosen peg counter-clockwise" message

**Output (`PitchResult` dataclass):**
```python
PitchResult(
  status="IN_TUNE",
  f0_hz=130.82,
  cents_dev=+0.8,
  hz_dev=+0.01,
  string_num=3,
  string_name="S3 — Mandra Sa (tonic)",
  target_hz=130.81,
  confidence=0.923,
  message="In tune (+0.8 cents) — S3 — Mandra Sa (tonic), target 130.81 Hz",
  method="pyin"
)
```

**Supported Tonic Options:** 22 chromatic pitches from A1 (55 Hz) to F#3 (185 Hz), runtime-configurable via `TONIC_OPTIONS` dict. Switching tonic at runtime via `update_tonic()` recomputes all 7 string target frequencies and the extended 22-Sruti prior array.

---

### 3.3 `python/engine.py` — Real-Time Data Acquisition & AI Dispatch (695 lines)

The engine's AI-facing responsibilities:

- **`piezo_raw_window`** (deque, maxlen = `3.0 s * 2000 Hz = 6000`): Rolling ring buffer of raw 12-bit ADC ints for live vibration inference.
- **`audio_raw_window`** (deque, maxlen = `3.0 s * 16000 Hz = 48000`): Rolling ring buffer of raw int16 PCM audio for live audio inference.
- **`analyze_recording_ai(prefix)`** (lines 636–654): Routes to `evaluate_live(...)` during recording (uses ring buffers, no file I/O), or `evaluate(...)` after stop (reads disk files).
- **`analyze_veena_ai(prefix, tonic_hz, cents_threshold, string_label)`** (lines 656–695): Same live/file routing for the Veena pipeline.

---

### 3.4 `python/main.py` — FastAPI Web Backend (461 lines)

AI-relevant API endpoints:

| Endpoint | Method | What It Returns |
|---|---|---|
| `GET /api/veena_analysis` | GET | VeenaDiagnosticModel result: tuning + quality + sound type |
| `GET /api/anomaly_analysis` | GET | AnomalyDetectionModel result: audio + vibration anomaly |
| `WS /ws/telemetry` | WebSocket | 8 Hz push of status + audio waveform + piezo terminal lines |

**Veena analysis query parameters:**
- `tonic_hz` (float, default 130.81) — runtime-configurable tonic.
- `cents_threshold` (float, default 15.0) — tuning tolerance in cents.
- `string_label` (str, optional) — `"S1"–"S4"`, `"T1"–"T3"`, or `null` for auto-detect.

---

### 3.5 `sketch/sketch.ino` — Arduino Hardware Sampler

The firmware is the entry point for all AI sensor data:

- **Sample rate:** 2000 Hz (Zephyr timer, 500 µs period).
- **Batch size:** 40 samples per packet = 20 ms cadence.
- **Packet format:** 92-byte MsgPack binary blob — `{batch_start_idx: u32, batch_start_us: u64, samples[40]: u16}`.
- **ADC resolution:** 12-bit (0–4095), reference 3.3 V.
- **Transport:** `Bridge.notify("piezo_batch", msgpack_bin_blob)` over the Arduino RouterBridge.

The Python `PACKET_FORMAT = "<IQ40H"` struct definition in `model.py` mirrors this exactly for zero-copy unpacking.

---

### 3.6 `python/models/config.json` — Runtime Configuration

```json
{
  "sample_rate": 16000,
  "window_seconds": 2.0,
  "anomaly_threshold": 0.6,
  "vibration_anomaly_threshold": 4.5552,
  "vibration_sample_rate_hz": 2000,
  "pca_components": 64,
  "yamnet_input_size": 15600,
  "yamnet_output_size": 521
}
```

All AI thresholds and model parameters are loaded from this file at runtime. Changing a threshold does not require code changes or model retraining.

---

## 4. Prompt Engineering & Context Management

> **Applicable only to the Veena Diagnostic module.** SwarCare uses no LLMs, no RAG, and no vector databases. The "prompts" in this system are deterministic physics formulas and learned statistical boundaries.

### Analogy: "Sruti Priors as Soft Prompts"

The closest analogy to prompt engineering in this codebase is the **22-Sruti Gaussian soft prior** in the Physics Pitch Engine. Rather than constraining the pYIN HMM to a hard frequency grid, it applies a Gaussian weight that:

- Down-weights frames that land far from any Carnatic Sruti ratio (analogous to penalising off-topic LLM output).
- Preserves full weight for the 200 ms attack window (analogous to allowing "free-form" exploration at the start).
- Uses `sigma = 70 cents` — generous enough to accommodate Gamaka ornaments (analogous to allowing paraphrase in LLM outputs).

**The 22 Sruti ratios (relative to Sa = tonic):**
```python
(1.0, 256/243, 16/15, 10/9, 9/8, 32/27, 6/5, 5/4, 81/64,
 4/3, 27/20, 45/32, 729/512, 3/2,
 128/81, 8/5, 5/3, 27/16, 16/9, 9/5, 15/8, 243/128)
```
These are extended across 4 octave multipliers (0.5x, 1x, 2x, 4x) for a full prior spanning the Veena's playable range.

### Feature Engineering as Context Management

The 527-D feature vector for the RF classifier acts as a structured "context window":
- **Dimensions 0–520:** Global acoustic scene context (YAMNet AudioSet embeddings).
- **Dimensions 521–524:** Pitch-domain context (F0, cents error, detuning triggers).
- **Dimensions 525–526:** Temporal envelope context (decay rate) + spectral texture context (HF flatness).

This multi-domain feature fusion ensures the classifier has both acoustic scene information (from a pre-trained neural backbone) and instrument-physics-grounded features simultaneously.

---

## 5. AI Data Flow

### End-to-End Example: Live Veena Diagnostic During Recording

**Step 1 — Hardware Sampling (Arduino, every 20 ms):**
```
Piezo sensor → ADC (12-bit, 2000 Hz) → Zephyr timer fires every 500 µs
→ Accumulate 40 samples → Pack into 92-byte TelemetryPacket
→ Bridge.notify("piezo_batch", binary_blob)
```

**Step 2 — Engine Ingestion (engine.py, `_process_piezo_stream` thread):**
```
bridge callback → piezo_queue.put_nowait(bytes)
→ _process_piezo_stream dequeues
→ struct.unpack("<IQ40H", payload) → (batch_idx, batch_us, [40 ADC values])
→ For each sample: compute amplitude + voltage + timestamp
→ Append to piezo_raw_window (ring deque, 6000-sample cap = 3 s)
→ Write CSV line to {prefix}_piezo.csv
```

**Step 3 — Audio Capture (engine.py, `_audio_capture_worker` thread, simultaneously):**
```
arecord subprocess → stdout pipe
→ Read 640-byte chunks (20 ms at 16 kHz int16)
→ Write to {prefix}_audio.tmp
→ Unpack int16 → append to audio_raw_window (48000-sample cap = 3 s)
→ Decimate x32 → normalise → append to audio_live_buffer (display only)
```

**Step 4 — AI Analysis Request (main.py, HTTP GET /api/veena_analysis):**
```
FastAPI handler calls backend.analyze_veena_ai(prefix, tonic_hz=130.81, ...)
→ Engine state == "RECORDING" → route to in-memory path
→ audio_raw = engine.get_recent_audio_raw_window()  # last 3 s, 48k samples
→ VeenaDiagnosticModel.evaluate_live(audio_raw, 16000, tonic_hz=130.81, ...)
```

**Step 5 — Sound Validation Gate (model.py, `classify_audio_sound_type`):**
```
audio_int16 → float32 (/ 32768.0)
→ RMS check: if < 0.004 → return {is_veena: false, sound_type: "Silence"}
→ Find strongest 1-second window (hop=0.1 s sliding search)
→ Autocorrelation (FFT-based, O(n log n)) → periodicity score
→ Harmonic Energy Ratio (HER) across k=1..13 harmonics
→ Spectral flatness + peak prominence
→ YAMNet speech class check (classes 0-68)
→ Decision: is_veena=True → proceed; else → return early
```

**Step 6 — Parallel Thread Dispatch (model.py, `VeenaDiagnosticModel.evaluate_audio`):**
```
Thread 1: _run_physics_tuning(audio, sr=16000, tonic_hz=130.81, ...)
Thread 2: _run_ml_quality(audio, sr=16000)
Both threads start simultaneously → join(timeout=10.0 s)
```

**Step 7 — Physics Thread (`PhysicsPitchEngine.run`):**
```
audio → Stage 0: RMS silence gate
→ Stage 1: Cepstral lifter (strip Kudam 280-300 Hz formants)
→ Stage 2: HPS F0 estimate (8192-pt FFT, R=4, Blackman, SavGol)
→ Stage 3: pYIN (librosa, fmin=55, fmax=2093 Hz, frame=4096, hop=512)
           + 22-Sruti Gaussian prior (sigma=70 cents, full weight on attack)
           → weighted median F0
→ Stage 3.5: Harmonic comb validation (reject if HER<0.40 AND periodicity<0.65)
→ Stage 4: Cents decision — find nearest of 7 strings → IN_TUNE/FLAT/SHARP
→ PitchResult(f0_hz=130.82, cents_dev=+0.8, status="IN_TUNE", ...)
```

**Step 8 — ML Thread (`_run_ml_quality`):**
```
audio → _extract_veena_features(audio, sr=16000)
  → _extract_yamnet_embedding() → 521-D scores
  → F0 via autocorrelation (lag range: sr/400 to sr/45)
  → cents deviation against 4 target strings, s1_err, detuning_trigger
  → Energy Decay Rate (onset vs. tail RMS ratio)
  → HF Spectral Flatness (>2000 Hz via librosa STFT bins 256+)
  → Concatenate → 527-D feature vector
→ scaler.transform(features.reshape(1, -1))
→ clf.predict([scaled]) → fault_class = 0
→ clf.predict_proba([scaled]) → confidence = 97.1%
→ label = "Healthy"
```

**Step 9 — Fusion & Response:**
```
physics: {status: "IN_TUNE", confidence: 0.923}
ml:      {is_healthy: true, confidence: 97.1%, label: "Healthy"}

is_healthy = quality_ok(True) AND tuning_ok(True) → True
fused_status = "Healthy"

Return dict:
{
  "status": "Healthy",
  "is_healthy": true,
  "is_veena": true,
  "sound_type": "Veena String Resonance",
  "tuning": { "status": "IN_TUNE", "f0_hz": 130.82, "cents_dev": 0.8 },
  "quality": { "label": "Healthy", "confidence": 97.1 },
  "tonic_hz": 130.81
}
```

**Step 10 — HTTP Response:**
```
FastAPI serialises dict → JSON → 200 OK
Frontend polls every ~2 s → renders tuning meter + fault diagnosis panel
```

---

## 6. Key Constraints & Error Handling

### 6.1 TFLite Runtime Discovery Failure

If no TFLite runtime is installed, `load_interpreter()` raises a clear `ImportError` listing all 5 attempted import paths:
```
No TFLite runtime found. Install one of:
  pip install ai-edge-litert
  pip install tflite-runtime
  pip install tensorflow
Attempted imports: ai_edge_litert.interpreter: ModuleNotFoundError(...) | ...
```
**Downstream effect:** `run_pipeline()` raises → `audio_err` string stored → `audio` sub-dict shows `available: false`. The vibration pipeline runs independently.

### 6.2 Vibration Model Not Deployed

If `kmeans_vibration_center.npy` or `kmeans_vibration_std.npy` are missing, `run_vibration_pipeline()` returns `None`. The report gracefully shows:
```json
{
  "available": false,
  "status": "no_data",
  "details": "Vibration K-means model not deployed (models/kmeans_vibration_*.npy missing)."
}
```
The anomaly fusion uses audio alone.

### 6.3 Minimum Buffer Thresholds (Live Mode)

In live mode, results are withheld until enough data accumulates:
- **Piezo:** `< 500 samples` → `vib_err = "Buffering piezo data (N/500 samples so far)."`
- **Audio:** `< 4000 samples` → `audio_err = "Buffering audio (N/4000 samples so far)."`

This prevents spurious anomaly alerts in the first fraction of a second after recording starts.

### 6.4 Sample Rate Mismatch (Critical Known Issue)

**Documented at lines 19–35 of `model.py`:**

The vibration K-means model was calibrated at a different sample rate than the real hardware. The `high_freq_ratio` feature is sample-rate-dependent; calibrating at 16 kHz then scoring at 2 kHz produces wildly inflated distances (~21 vs. threshold 4.5552), appearing as constant `anomaly` even on healthy instruments.

**Current mitigation:** The `high_freq_split_ratio` is Nyquist-relative (not fixed-Hz), and features with `std < MIN_VIBRATION_STD = 1e-4` are excluded from Z-scoring.

**Resolution path:** Run `calibrate_vibration_baseline_multi()` with actual 2 kHz hardware recordings to produce a correctly calibrated model.

### 6.5 Audio Resampling

If a WAV file's sample rate != 16 kHz (the YAMNet target), a simple linear interpolation resampler is applied:
```python
audio = np.interp(
    np.linspace(0, len(audio), n_target, endpoint=False),
    np.arange(len(audio)), audio
).astype(np.float32)
```
For live audio, the ALSA `arecord` command is always configured for exactly 16 kHz.

### 6.6 Physics Engine Failures

`_run_physics_tuning` wraps `PhysicsPitchEngine.run()` in a `try/except`:
- `ImportError` (physics module missing): returns `{available: false, error: "physics_pitch_engine.py could not be imported: ..."}`.
- Any other exception: returns `{available: false, error: str(exc)}`.

In both cases, the ML quality thread continues independently.

### 6.7 No-Pitch / Low-Confidence Rejection

If pYIN returns no voiced frames or confidence < 0.15, `PhysicsPitchEngine.run()` returns `PitchResult(status="NO_PITCH")`. The HPS estimate (`f0_hps`) is used as a fallback with `confidence=0.4`, method `"hps_autocorr"`.

### 6.8 Sound Type Rejection (Non-Veena Audio)

`classify_audio_sound_type()` acts as a two-stage guard:
1. If audio is silence → report `"Silence / Idle"` with a user message to pluck a string.
2. If audio is speech/noise (YAMNet speech_score > 0.25) → report `"Human Speech / Voice Detected"` and skip both ML and physics inference.

This prevents wasting compute on irrelevant audio and avoids generating misleading fault diagnoses.

### 6.9 Parallel Thread Timeout

Both physics and ML threads are joined with `timeout=10.0 s`. If either hangs:
- The timed-out thread leaves its output dict empty `{}`.
- Fusion checks for `{available: false}` and falls back gracefully.

### 6.10 Confidence Heuristics

The fused confidence score in `_build_report()` is a heuristic, not a calibrated probability:
```python
confidence = min(99.8, (75.0 if both_sensors else 60.0) + severity * 12.0)
```
- Both sensors active: baseline 75%.
- Single sensor: baseline 60%.
- Each severity level (watch=1, anomaly=2) adds 12 percentage points.
- Capped at 99.8% (never claims perfect certainty).

---

*End of SwarCare AI/ML Documentation*
