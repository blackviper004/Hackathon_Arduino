# SwarCare — Saraswati Veena Vibroacoustic Health Monitor

> Detect vibroacoustic anomalies in Saraswati Veena using an edge AI pipeline on the **Arduino UNO Q**.

**Competition:** Arduino Physical AI Challenge India 2026

---

## What It Does

SwarCare listens to a Saraswati Veena being played and classifies the sound as **Healthy**, **Watch** (early warning), or **Anomaly** (structural/tonal issue). The system runs entirely on the Arduino UNO Q — no cloud required.

**Pipeline:** Audio (16kHz mono) → YAMNet (521-D scores) → PCA (64-D) → K-means (distance to healthy center) → 3-tier classification

**Visual feedback:** The UNO Q's 13×8 LED matrix displays:
| Status | LED Matrix |
|--------|------------|
| Healthy | Veena icon (steady) |
| Watch | Exclamation mark (pulsing 1Hz) |
| Anomaly | X / Alert triangle (flashing 3Hz) |
| Recording | Mic icon (pulsing 2Hz) |
| Processing | Spinner animation (4Hz) |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Arduino UNO Q                       │
│                                                       │
│  ┌────────────────────────┐  Bridge  ┌─────────────┐ │
│  │  MPU (Cortex-A53)      │◄───────►│ MCU (STM32) │ │
│  │  Linux / Python        │          │ Zephyr RTOS │ │
│  │                        │          │             │ │
│  │  Streamlit WebUI       │ notify   │ LED Matrix  │ │
│  │  YAMNet → PCA → KM    │────────►│ 13×8 blue   │ │
│  │                        │          │             │ │
│  └────────────────────────┘          │ INMP441 Mic │ │
│                                       │ (I2S→D13,   │ │
│  USB Mic ──► MPU records              │  D10, D12)  │ │
│          ──► Pipeline                 │             │ │
│          ──► LED feedback             │ Piezo Disc  │ │
│                                       │ (A0 + 1MΩ) │ │
│  [*] = Future: MCU captures sensors   └─────────────┘ │
│        and streams via Bridge to MPU                   │
└──────────────────────────────────────────────────────┘
```

**Dual-processor roles:**
- **MPU** (Qualcomm QRB2210, 4× Cortex-A53 @ 2GHz, 2GB RAM): Runs Streamlit web UI + inference pipeline
- **MCU** (STM32U585, Cortex-M33 @ 160MHz): Drives LED matrix, I2S mic capture, piezo vibration ADC
- **Bridge**: `Bridge.notify("set_status", code)` sends classification result from MPU → MCU

---

## Hardware Wiring

### Components

| Component | Qty | Purpose | Approx Cost |
|-----------|-----|---------|-------------|
| Arduino UNO Q | 1 | Main board (MPU + MCU) | ₹4,880 |
| INMP441 I2S MEMS Microphone | 1 | Airborne sound capture (digital, 16kHz) | ₹170 |
| 35mm Piezo Disc (with wires) | 1 | Contact vibration pickup (analog) | ₹25 |
| 1MΩ Resistor | 1 | Discharge resistor for piezo | ₹2 |
| Breadboard + Jumper Wires | 1 set | Prototyping connections | ₹100 |
| USB-C Cable | 1 | Power + programming | included |

### Wiring Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ARDUINO UNO Q (MCU SIDE)                         │
│                                                                          │
│   ┌────────────┐         ┌──────────────────────────────────────────┐   │
│   │  INMP441   │         │          MCU Pin Header                   │   │
│   │  I2S Mic   │         │                                           │   │
│   │            │         │   3.3V ●─────────── VDD (INMP441)        │   │
│   │   VDD ─────┼─────────┼── 3.3V                                   │   │
│   │   GND ─────┼─────────┼── GND ●─────┬──── GND (INMP441)        │   │
│   │   SCK ─────┼─────────┼── D13       │                            │   │
│   │   WS  ─────┼─────────┼── D10       │                            │   │
│   │   SD  ─────┼─────────┼── D12       │                            │   │
│   │   L/R ─────┼─────────┼── GND       │     (L/R=GND → Left ch)   │   │
│   └────────────┘         │              │                            │   │
│                          │              │                            │   │
│   ┌────────────┐         │              │                            │   │
│   │  35mm      │         │              │                            │   │
│   │  PIEZO     │         │              │                            │   │
│   │  DISC      │         │              │                            │   │
│   │            │         │              │                            │   │
│   │  Red (+)───┼────┬────┼── A0        │                            │   │
│   │            │    │    │              │                            │   │
│   │            │  [1MΩ]  │              │  ← Discharge resistor      │   │
│   │            │    │    │              │    (across piezo leads)     │   │
│   │  Blk (-)──┼────┴────┼── GND ●─────┘                            │   │
│   └────────────┘         │                                           │   │
│                          └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### INMP441 Pin Connections

| INMP441 Pin | UNO Q MCU Pin | Function |
|-------------|---------------|----------|
| VDD | 3.3V | Power supply (1.8V–3.3V) |
| GND | GND | Ground |
| SCK | D13 | I2S Serial Clock (bit clock) |
| WS | D10 | I2S Word Select (left/right frame) |
| SD | D12 | I2S Serial Data (audio bits out) |
| L/R | GND | Channel select: GND=Left, VDD=Right |

**Note:** The INMP441 outputs 24-bit digital audio over I2S at up to 48kHz. We configure it for 16kHz mono (left channel) to match YAMNet's input requirement.

### Piezo Sensor Connections

| Piezo Wire | UNO Q MCU Pin | Function |
|------------|---------------|----------|
| Red (+) | A0 | Analog signal (vibration voltage) |
| Black (−) | GND | Ground reference |
| **1MΩ resistor** | **A0 ↔ GND** | Discharge path (in parallel with piezo) |

**Why the 1MΩ resistor?**
A piezo disc is essentially a capacitor (~10nF). When the veena body vibrates, it generates voltage spikes. Without the resistor:
- Charge gets trapped — ADC reads a stuck voltage
- Signal won't return to 0V between vibrations

The 1MΩ resistor provides a slow bleed path:
- RC time constant = 1MΩ × 10nF = **10ms** (clean decay, captures the waveform envelope)
- High enough resistance to not dampen the signal
- Low enough to discharge before the next vibration cycle

### Sensor Placement on the Veena

The sensor pod is designed for **repeatable vibroacoustic monitoring**, not laboratory-grade absolute acoustic measurement. The objective is to measure the same instrument the same way every time. Any residual mechanical coupling between the piezo and microphone becomes part of the instrument's baseline and is learned by the model as "normal."

![Sensor Pod — Internal Details & Placement](unoq-app/assets/docs_assets/sensor-pod-diagram.png)

| Parameter | Value | Why |
|-----------|-------|-----|
| Distance from kudurai | 2–3 cm (toward tail) | Transfer measurement of body response, not driving-point measurement |
| Lateral offset | 2–3 cm from centerline | Avoids pressure nulls of antisymmetric plate modes |
| Orientation | Port hole faces veena surface | Maximize captured acoustic radiation from top plate |
| Attachment | Removable adhesive (Blu-Tack) | Non-destructive, consistent pressure, repositionable |
| Position marking | Pencil dot on veena body | Ensures exact same placement every session |

**Why behind the kudurai (not on it):**
Mounting directly on the bridge introduces highly localized mechanical excitation and may reduce repeatability. The bridge's effective mass would also change. Mounting on the soundboard 2–3 cm behind the bridge provides a more representative structural response of how the body actually reacts to string excitation.

**Design principle:** Repeatability > absolute isolation. A consistent 5% mechanical coupling is invisible to the model. An inconsistent 5–40% coupling (from a loose or shifting mount) makes the model unstable.

#### Future Work

> Future work will experimentally characterize microphone–structure coupling using transfer function measurements and optimize the pod's internal mechanical isolation for improved separation of airborne and structural signals.

---

## Data Collection Protocol

### Overview

You need to collect **real veena audio** to replace the Freesound data and retrain the pipeline with your own recordings. This section explains exactly what to record, how, and what the expert should do.

### Recording Format Requirements

#### INMP441 — Airborne Audio

| Parameter | Value | Why |
|-----------|-------|-----|
| Sample Rate | 16,000 Hz (16kHz) | YAMNet's native input format |
| Channels | Mono (1 channel) | YAMNet expects mono |
| Bit Depth | 16-bit PCM or 32-bit float | Standard WAV format |
| File Format | `.wav` | Uncompressed, no lossy encoding |
| Segment Length | 2 seconds each | Pipeline's window size |
| Minimum Total | 5+ minutes per condition | Gives 150+ segments after silence filter |

#### Piezo Disc — Contact Vibration

| Parameter | Value | Why |
|-----------|-------|-----|
| Sample Rate | 16,000 Hz (16kHz) | Match audio rate for synchronized capture |
| Channels | 1 (single piezo) | One sensor |
| Bit Depth | 12-bit ADC (0–4095) | STM32U585 ADC resolution |
| File Format | `.csv` | Natural MCU output, easy to inspect/debug |
| Segment Length | 2 seconds each | Synchronized with audio segments |
| Voltage Range | 0–3.3V (mapped to 0–4095) | MCU ADC reference voltage |

**CSV format:**
```csv
timestamp_ms,adc_value
0.000,2048
0.0625,2051
0.125,2063
...
```
(16,000 rows per second × 2 seconds = 32,000 rows per segment)

**Both sensors should record simultaneously.** This ensures every audio segment has a matching vibration segment from the exact same moment of playing.

**For initial data collection (before MCU driver is ready):** Use your laptop/phone to record audio. The piezo data collection requires the MCU ADC sketch — we'll build that separately. Start with audio-only collection now; add piezo once the hardware is wired.

**Naming convention for paired files:**
```
healthy_scales_01_audio.wav     ← from INMP441
healthy_scales_01_vibration.csv ← from Piezo (same session, same timestamp)
```

### What to Record

You need **5 types of recordings**:

#### 1. HEALTHY (Most Important — Record the Most)

| Session | What the Expert Plays | Duration | Why |
|---------|----------------------|----------|-----|
| H1: Open strings | All 4 main strings plucked open, one at a time | 5 min | Baseline tone of each string |
| H2: Scale passages | Ascending/descending ragas (Sa Re Ga Ma...) | 10 min | Covers all fret positions |
| H3: Gamakas | Oscillations, slides between notes | 5 min | Tests sustained vibration patterns |
| H4: Fast passages | Rapid note sequences, alankaaras | 5 min | Tests transient response |
| H5: Full raga | Complete raga performance (any raga) | 10 min | Real-world playing variety |

**Target: 35+ minutes of healthy audio → 1000+ segments**

#### 2. DETUNED (Anomaly Type 1)

| Session | How to Create | Duration |
|---------|--------------|----------|
| D1: Slightly detuned | Loosen ONE string by ~¼ turn (noticeable but playable) | 5 min |
| D2: Heavily detuned | Loosen the same string by ~½ turn (clearly wrong) | 3 min |
| D3: Multiple strings | Detune 2 strings slightly | 3 min |

**Ask the expert to play the SAME passages as H1-H3 while detuned.** This gives comparable data.

#### 3. BUZZING/RATTLE (Anomaly Type 2)

| Session | How to Create | Duration |
|---------|--------------|----------|
| B1: Fret buzz | Press strings lightly (incomplete contact with fret) | 5 min |
| B2: Loose part | Place a small piece of paper/tape touching a string near the bridge | 3 min |
| B3: Sympathetic rattle | If there are taalam strings, let them buzz freely | 3 min |

**Buzzing is the most common real defect — record plenty of this.**

#### 4. MUFFLED/DEAD (Anomaly Type 3)

| Session | How to Create | Duration |
|---------|--------------|----------|
| M1: Damped strings | Lightly touch strings with palm while playing (mutes overtones) | 3 min |
| M2: Covered sound hole | Partially block the resonator opening with cloth | 3 min |
| M3: Dead string | Use an old/worn string if available (less sustain) | 3 min |

#### 5. BRIDGE ISSUES (Anomaly Type 4 — bonus for competition)

| Session | How to Create | Duration |
|---------|--------------|----------|
| BR1: Bridge shift | Slightly shift the bridge position (if safe to do) | 2 min |
| BR2: Bridge damping | Place Blu-Tack on the bridge to dampen vibration transfer | 3 min |

### Recording Session Procedure

**Equipment needed:**
- Your laptop with Audacity (free) or phone voice recorder app
- OR the Arduino UNO Q with INMP441 connected (if ready)
- The veena + an expert player

**Step-by-step:**

1. **Environment:** Record in a **quiet room**. Close doors/windows. No AC fan noise if possible.

2. **Mic placement:** 
   - If using laptop mic / phone: Place 30–50 cm from the veena body, at bridge height
   - If using INMP441 on UNO Q: Already positioned per wiring diagram above

3. **File naming convention:**
   ```
   healthy_openstrings_01.wav
   healthy_scales_01.wav
   healthy_gamakas_01.wav
   detuned_slight_01.wav
   buzzing_fretbuzz_01.wav
   muffled_damped_01.wav
   bridge_shifted_01.wav
   ```

4. **Recording flow:**
   ```
   START RECORDING → Wait 2 sec silence → Expert plays → Wait 2 sec → STOP
   ```
   The silence at start/end gets filtered out by Step 02 (RMS filter).

5. **Between conditions:**
   - Fix the veena back to healthy state between anomaly sessions
   - Re-tune properly before each healthy session
   - Let the expert confirm "this sounds right" before recording healthy data

### Folder Organization for Retraining

After recording, organize files like this:

```
pipeline-validation/
└── raw_audio/
    ├── healthy/
    │   ├── healthy_openstrings_01.wav
    │   ├── healthy_scales_01.wav
    │   ├── healthy_gamakas_01.wav
    │   ├── healthy_fast_01.wav
    │   └── healthy_fullraga_01.wav
    ├── detuned/
    │   ├── detuned_slight_01.wav
    │   └── detuned_heavy_01.wav
    ├── buzzing/
    │   ├── buzzing_fretbuzz_01.wav
    │   └── buzzing_rattle_01.wav
    ├── muffled/
    │   ├── muffled_damped_01.wav
    │   └── muffled_covered_01.wav
    └── bridge/
        ├── bridge_shifted_01.wav
        └── bridge_damped_01.wav
```

### Retraining After Collection

Once you have real data, the pipeline scripts need minor edits:

1. **02_segment_audio.py** — Point `INPUT_DIR` to `raw_audio/healthy/` (only segment healthy for training)
2. **03_prepare_training_data.py** — Instead of generating fake anomalies, copy real anomaly recordings:
   - Segment `raw_audio/detuned/` → `training_data/testing/anomaly/`
   - Segment `raw_audio/buzzing/` → `training_data/testing/anomaly/`
   - etc.
3. **07_yamnet_transfer.py** — Runs the same (no changes needed)
4. **export_models.py** — Runs the same (no changes needed)

**Expected improvement:** Real anomaly data should push accuracy from 83% → 95%+ because the model will be validated against actual defect sounds instead of synthetic corruptions.

---

## Project Structure

```
SwarCare/
├── README.md                           ← You are here
├── export_models.py                    ← Package trained models → unoq-app/python/models/
│
├── unoq-app/                           ← Complete Arduino App Lab project
│   ├── app.yaml                        ← App manifest (bricks: web_ui_streamlit)
│   ├── python/                         ← MPU-side (runs on Linux)
│   │   ├── main.py                     ← Streamlit app (3 tabs: Upload, Record, Batch)
│   │   ├── requirements.txt            ← ai-edge-litert, numpy, scipy
│   │   └── models/                     ← Deployed model artifacts
│   │       ├── yamnet.tflite           ← Google YAMNet TFLite (~3.7 MB)
│   │       ├── pca_model.npz           ← PCA: mean(521) + components(64×521)
│   │       ├── kmeans_center.npy       ← Healthy cluster center (64 floats)
│   │       ├── kmeans_std.npy          ← Std dev per dimension (64 floats)
│   │       └── config.json             ← Threshold & pipeline settings
│   ├── sketch/                         ← MCU-side (runs on Zephyr RTOS)
│   │   ├── sketch.ino                  ← LED matrix + sensor stubs
│   │   └── sketch.yaml                 ← Sketch libraries config
│   └── web-ui/                         ← PC testing version (no Arduino needed)
│       └── web_ui.py                   ← streamlit run web_ui.py
│
└── pipeline-validation/                ← Training & data pipeline (run on PC)
    ├── 01_download_audio.py            ← Download veena audio from Freesound
    ├── 02_segment_audio.py             ← Segment into 2s training clips (all healthy)
    ├── 03_prepare_training_data.py     ← Split train/test + generate anomalies
    ├── 07_yamnet_transfer.py           ← Extract embeddings, validate, fit PCA, save
    ├── raw_audio/                      ← Downloaded Freesound WAV files
    ├── segments/                       ← 2s WAV segments (output of 02)
    ├── training_data/                  ← WAV files organized for training (output of 03)
    ├── yamnet_embeddings/              ← PCA model + CSV embeddings (output of 07)
    │   └── edge_impulse_upload/        ← CSVs uploadable to Edge Impulse
    ├── yamnet.tflite                   ← YAMNet model (auto-downloaded by 07)
    └── requirements.txt                ← librosa, soundfile, tensorflow, numpy
```

---

## Quick Start

### Test on PC (no Arduino needed)

`web-ui/web_ui.py` is a PC-friendly version that uses the same models but doesn't need Arduino hardware:

```bash
cd unoq-app/web-ui
pip install streamlit numpy scipy ai-edge-litert
streamlit run web_ui.py
```

Open `http://localhost:8501` → Upload a `.wav` file → See classification result.

### Deploy to Arduino UNO Q

**Prerequisites:** Arduino UNO Q board, USB-C cable, USB microphone

#### Step 1 — Train the pipeline (if starting from scratch)

```bash
cd pipeline-validation
pip install -r requirements.txt

python 01_download_audio.py          # Get veena audio from Freesound
python 02_segment_audio.py           # Segment into 2s clips (~179 healthy files)
python 03_prepare_training_data.py   # Split train/test + generate anomalies
python 07_yamnet_transfer.py         # Extract embeddings, validate, fit PCA, save
```

#### Step 2 — Export models

```bash
cd ..
python export_models.py
```

This copies `yamnet.tflite`, computes PCA + K-means center from training embeddings, auto-calculates the anomaly threshold, and writes all model files to `unoq-app/python/models/`.

#### Step 3 — Deploy via Arduino App Lab

1. Open [Arduino App Lab](https://app.arduino.cc/) and connect your UNO Q
2. Create a new App named **SwarCare**
3. Add the **WebUI-Streamlit** brick
4. Copy project files into the App:
   - `python/main.py` → Python editor
   - `python/requirements.txt` → inside `python/` folder
   - `python/models/` → upload via file manager
   - `sketch/sketch.ino` → Sketch editor
5. Click **Run**
   - `uv` auto-installs requirements
   - Sketch compiles and flashes to MCU
   - Streamlit UI starts on port 7000
6. Open `http://<board-ip>:7000`

---

## How the Pipeline Works

### Step 02: Audio Segmentation (`02_segment_audio.py`)

Takes raw Freesound WAV files (variable length, any sample rate) and produces fixed-size training clips.

**What it does:**

```
Raw WAV file (e.g., 260 seconds, 44.1kHz stereo)
        │
        ▼
┌────────────────────────────────────────────┐
│  1. RESAMPLE & NORMALIZE                    │
│     librosa.load(sr=16000, mono=True)       │
│     → Converts any format to 16kHz mono     │
│       (YAMNet's required input format)      │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  2. SLICE INTO NON-OVERLAPPING WINDOWS      │
│     Window = 2 seconds = 32,000 samples     │
│     A 260s file → 130 possible windows      │
│     Leftover tail < 2s is discarded         │
│                                             │
│     NOTE: YAMNet only uses the first        │
│     15,600 samples (0.975s) from each       │
│     segment. The remaining ~1s in each      │
│     2s file is unused during inference.     │
│     A 1s window would be more efficient     │
│     but 2s works fine with enough data.     │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  3. SILENCE FILTER (RMS energy check)       │
│     RMS = √(mean(samples²))                │
│     If RMS < 0.01 → SKIP (dead air)        │
│     If RMS ≥ 0.01 → SAVE                   │
│                                             │
│     This removes:                           │
│     - Silence at start/end of recording     │
│     - Gaps/pauses in performance            │
│     - Near-silent ambient noise             │
└────────────────────────────────────────────┘
        │
        ▼
   segments/veena_murthy_001.wav (2s, 16kHz, mono)
   segments/veena_murthy_002.wav
   ...
```

**Our results (from 3 Freesound recordings):**

| Recording | Raw Duration | Possible 2s windows | After silence filter |
|-----------|-------------|---------------------|---------------------|
| veena_pickup (216060) | ~260s | 130 | 128 |
| veena_murthy (125655) | ~100s | 50 | 49 |
| veena_classical (380750) | ~70s | 35 | 2 (mostly silent) |
| **Total** | | | **179 segments** |

The `veena_classical` recording yielded only 2 segments because most of it was below the RMS threshold — likely a recording with long pauses between phrases.

---

### Step 03: Prepare Training Data (`03_prepare_training_data.py`)

Takes the 179 healthy segments from Step 02 and organizes them into training/testing folders + generates fake anomalies. **This script does NOT classify anything** — it's a file organizer and audio corruptor.

**Key assumption:** ALL 179 segments from Step 02 are **healthy** because the Freesound source recordings are of a properly functioning veena. The "healthy" label is assigned by us, not detected.

**What it does:**

```
179 healthy segments (from segments/)
        │
        ▼
┌──────────────────────────────────────────────┐
│  1. RANDOM 80/20 SPLIT                        │
│     Shuffle with fixed seed (reproducible)    │
│     143 segments → training/healthy/          │
│      36 segments → testing/healthy/           │
│                                               │
│     These are just file COPIES.               │
│     No analysis, no classification.           │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  2. GENERATE FAKE ANOMALIES                   │
│     Takes the 36 test segments and corrupts   │
│     each with a MATHEMATICAL operation:       │
│                                               │
│     12 → Pitch-shifted ±2 semitones           │
│          (simulates detuned strings)          │
│                                               │
│     12 → Random noise added at 10-30%         │
│          (simulates buzzing/rattle)           │
│                                               │
│     12 → Low-pass filter <1500Hz              │
│          (simulates muffled/dead strings)     │
│                                               │
│     Output → testing/anomaly/ (36 files)      │
│                                               │
│     These are NOT real broken-veena sounds.    │
│     They are healthy audio deliberately       │
│     corrupted with signal processing.         │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  3. GENERATE SYNTHETIC VEENA TONES            │
│     20 healthy = sine waves with 7 harmonics  │
│       (simulates plucked string overtones)    │
│     20 anomaly = corrupted synthetic tones    │
│                                               │
│     Output → training_data/synthetic/          │
│     Used as bonus validation data in Step 07  │
└──────────────────────────────────────────────┘
```

**Output structure:**

```
training_data/                          ← WAV files (audio segments)
├── training/
│   └── healthy/        ← 143 WAVs (model LEARNS from these)
├── testing/
│   ├── healthy/        ← 36 WAVs (should classify as healthy)
│   └── anomaly/        ← 36 WAVs (should classify as anomaly)
└── synthetic/
    ├── healthy/        ← 20 artificial veena tones
    └── anomaly/        ← 20 corrupted artificial tones
```

**Why fake anomalies?** We don't have recordings of a broken veena. So we simulate what vibroacoustic defects might sound like. The actual anomaly detection (Step 07 + on-device) works by measuring how *different* new audio is from the healthy training set — not by matching these specific corruption patterns.

---

### Step 07: YAMNet Transfer Learning (`07_yamnet_transfer.py`)

Takes WAV files from `edge_impulse_upload/` and produces trained models + CSV embeddings for all categories.

**What it does (6 phases):**

```
Phase 1: Load YAMNet TFLite (downloads if needed, ~3.9MB)

Phase 2: Extract embeddings from ALL 5 folders
┌──────────────────────────────────────────────────────┐
│  For each WAV file:                                   │
│    Load audio → resample to 16kHz                     │
│    Process in 0.975s windows (50% overlap)            │
│    Each window → YAMNet → 521 class probabilities     │
│    Average all windows → single 521-D embedding       │
│                                                       │
│  Result: 255 embeddings (143+36+36+20+20) in RAM      │
└──────────────────────────────────────────────────────┘

Phase 3: Train anomaly detector (ONLY uses 143 training/healthy)
┌──────────────────────────────────────────────────────┐
│  mean = average of 143 embeddings (the "center")      │
│  std  = standard deviation per dimension              │
│  threshold = mean(distances) + 2×std(distances)       │
│                                                       │
│  This is the ENTIRE model — one center point.         │
│  "How far is new audio from healthy center?"          │
└──────────────────────────────────────────────────────┘

Phase 4: Validate on all test data (prints accuracy, saves nothing)

Phase 5: Test full raw recordings (informational only)

Phase 6: Fit PCA + save ALL embeddings as CSVs
┌──────────────────────────────────────────────────────┐
│  PCA on 143 training embeddings: find 64 most         │
│  important directions in 521-D space                  │
│                                                       │
│  Save pca_model.npz (compression recipe)              │
│  Transform ALL 255 embeddings: 521-D → 64-D          │
│  Save as CSVs in yamnet_embeddings/edge_impulse_upload│
└──────────────────────────────────────────────────────┘
```

**Output:**

```
yamnet_embeddings/
├── pca_model.npz                           ← PCA model (mean + 64 components)
└── edge_impulse_upload/                    ← All embeddings as CSVs (64 features each)
    ├── training/healthy/   → 143 CSVs      ← Used by export_models.py
    ├── testing/healthy/    → 36 CSVs       ← For Edge Impulse verification
    ├── testing/anomaly/    → 36 CSVs       ← For Edge Impulse verification
    ├── synthetic/healthy/  → 20 CSVs       ← For Edge Impulse verification
    └── synthetic/anomaly/  → 20 CSVs       ← For Edge Impulse verification
```

**Why only training/healthy matters for deployment:** The model is one-class — it only knows "healthy." Anomaly = anything far from healthy center. The test/synthetic CSVs are for your own verification and Edge Impulse upload.

#### Using CSVs with Edge Impulse (optional)

The CSV files in `yamnet_embeddings/edge_impulse_upload/` can be uploaded to [Edge Impulse Studio](https://studio.edgeimpulse.com/) for additional analysis:

1. **Create a new Edge Impulse project** → select "Anomaly Detection"
2. **Upload training data:**
   - Go to Data Acquisition → Upload
   - Select all 143 CSVs from `edge_impulse_upload/training/healthy/`
   - Label: "healthy", Category: "Training"
3. **Upload testing data:**
   - Select 36 CSVs from `testing/healthy/` → Label: "healthy", Category: "Testing"
   - Select 36 CSVs from `testing/anomaly/` → Label: "anomaly", Category: "Testing"
   - Select 20 CSVs from `synthetic/healthy/` → Label: "healthy", Category: "Testing"
   - Select 20 CSVs from `synthetic/anomaly/` → Label: "anomaly", Category: "Testing"
4. **Benefits of Edge Impulse verification:**
   - Visual feature explorer — see healthy vs anomaly clusters in 2D/3D
   - Built-in anomaly detection (K-means / GMM) to compare against our pipeline
   - Model testing dashboard with confusion matrix
   - Export alternative TFLite models for side-by-side comparison
   - Data versioning and experiment tracking

---

### Export Models (`export_models.py`)

Packages the trained models from `pipeline-validation/` into `unoq-app/python/models/` ready for Arduino deployment. **This is the bridge between training (PC) and inference (UNO Q).**

**What it does:**

```
pipeline-validation/                         unoq-app/python/models/
│                                            │
├── yamnet.tflite ──── COPY ──────────────► ├── yamnet.tflite (3.9 MB)
│                                            │
├── yamnet_embeddings/                       │
│   ├── pca_model.npz ── COPY ───────────► ├── pca_model.npz (263 KB)
│   │                                        │
│   └── edge_impulse_upload/                 │
│       └── training/healthy/                │
│           └── 143 CSVs ── COMPUTE ──────► ├── kmeans_center.npy (64 floats)
│               │                            ├── kmeans_std.npy (64 floats)
│               │                            │
│               └── distances ── AUTO ────► └── config.json (threshold)
```

**Phase by phase:**

1. **Copy `yamnet.tflite`** — the pre-trained audio model (unchanged, just moved)

2. **Copy `pca_model.npz`** — the compression recipe trained in Step 07

3. **Compute K-means center** — reads ALL 143 training CSVs, averages them:
   ```
   143 vectors (each 64-D) → mean = center (one 64-D vector)
                            → std  = spread per dimension (one 64-D vector)
   ```
   This center IS the "healthy fingerprint." On-device inference measures distance from this point.

4. **Auto-compute threshold** — calculates how far training samples are from center:
   ```
   For each of 143 training samples:
       distance = √Σ((feature - center)²)
   
   threshold = mean(all distances) + 2 × std(all distances)
   ```
   Anything beyond this distance is "anomaly." The 2× multiplier means ~95% of healthy samples should stay within bounds.

5. **Write `config.json`** — stores threshold + all settings in one file

**Output (total ~4 MB):**

| File | What it is | Used for |
|------|-----------|----------|
| `yamnet.tflite` | Pre-trained audio AI (Google) | Extract 521-D embedding from audio |
| `pca_model.npz` | Compression recipe (mean + 64 directions) | Reduce 521-D → 64-D |
| `kmeans_center.npy` | Average of all healthy embeddings | "What healthy sounds like" |
| `kmeans_std.npy` | Spread per dimension | Normalize distance calculation |
| `config.json` | Threshold + settings | Decision boundary |

---

### Inference Pipeline (what runs on-device)

```
Audio input (WAV or live mic, any sample rate)
    │
    ▼
Resample to 16kHz mono float32
    │
    ▼
┌─────────────────────────────────────────┐
│  YAMNet TFLite                          │
│  Input: 15600 samples (0.975s)          │
│  Sliding window, 50% overlap            │
│  Average scores across all windows      │
│  Output: 521-D vector (class scores)    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  PCA Projection (521 → 64)             │
│  1. Subtract pca_mean (521 values)      │
│  2. Multiply by pca_components (64×521) │
│  Output: 64-D feature vector            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  K-means Distance                       │
│  distance = √Σ((feature-center)/std)²  │
│                                         │
│  distance < threshold × 0.5 → HEALTHY   │
│  distance < threshold       → WATCH     │
│  distance ≥ threshold       → ANOMALY   │
└─────────────────────────────────────────┘
```

**Inference time on UNO Q:** ~58ms total

---

## MCU LED Matrix Communication

```python
# Python (MPU side) — sends status after classification
Bridge.notify("set_status", code)   # code: 0-4
```

```cpp
// Arduino (MCU side) — receives and drives LED matrix
Bridge.provide("set_status", setStatus);
```

| Code | Status | LED Animation |
|------|--------|---------------|
| 0 | Healthy | Veena icon (steady) |
| 1 | Watch | Exclamation mark pulsing at 1Hz |
| 2 | Anomaly | X mark / alert triangle flashing at 3Hz |
| 3 | Recording | Mic icon pulsing at 2Hz |
| 4 | Processing | Spinner dots rotating at 4Hz |

**MCU sensor connections:**
- INMP441 I2S Microphone (SCK→D13, WS→D10, SD→D12, L/R→GND) — captures 16kHz airborne audio
- 35mm Piezo Disc (Signal→A0, GND→GND, 1MΩ across leads) — captures contact vibration from veena body

---

## Model Files

| File | Size | What it stores |
|------|------|----------------|
| `yamnet.tflite` | ~3.7 MB | Pre-trained YAMNet (Google). 521 audio classes. |
| `pca_model.npz` | ~269 KB | `mean` (521,) + `components` (64×521). Compression recipe. |
| `kmeans_center.npy` | ~512 B | Healthy center in 64-D (one array of 64 floats) |
| `kmeans_std.npy` | ~512 B | Std dev per dimension (one array of 64 floats) |
| `config.json` | ~400 B | Threshold, sample rate, window size |

**Total deployment size:** ~4 MB

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Board | Arduino UNO Q (QRB2210 MPU + STM32U585 MCU) |
| MPU Runtime | Python 3.13, Streamlit, ai-edge-litert v2.1.5 |
| MCU Runtime | Arduino/Zephyr RTOS |
| ML Model | YAMNet (TFLite) — pre-trained audio event classifier |
| Dimensionality Reduction | PCA (sklearn-trained, numpy-only inference) |
| Anomaly Detection | K-means (single-cluster, standardized Euclidean distance) |
| MPU↔MCU Communication | Arduino Bridge (RPC via arduino-router) |
| Audio Input | USB microphone (via App Lab Microphone API) |
| Visual Output | 13×8 blue LED matrix (Arduino_LED_Matrix library) |
| Web Framework | Streamlit (served by WebUI-Streamlit brick on port 7000) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No TFLite runtime found` | Install `ai-edge-litert` (ARM64) or `tflite-runtime` (PC) |
| Mic unavailable | Falls back to test tone. Ensure USB mic is connected. |
| LED matrix garbled | Verify sketch uses `uint32_t[4]` with 13-column packing |
| Models not loading | Check `models/` folder is alongside `main.py` |
| WebUI not accessible | Streamlit serves on port 7000. Check `http://<board-ip>:7000` |
| Bridge not working | Ensure arduino-router service is running: `systemctl status arduino-router` |

---

## License

This project was built for the Arduino Physical AI Challenge India 2026.
