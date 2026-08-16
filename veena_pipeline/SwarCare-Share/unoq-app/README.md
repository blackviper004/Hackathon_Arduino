# SwarCare — Saraswati Veena Vibroacoustic Health Monitor

**SwarCare** detects vibroacoustic anomalies in a Saraswati Veena using transfer learning with Google's YAMNet model. It provides a web-based interface where users can upload WAV files, record live audio from a USB microphone, or batch-test multiple files — and receive real-time classification results with LED matrix feedback on the Arduino UNO Q.

![SwarCare](assets/docs_assets/swarcare-overview.png)

## Description

The App uses a pre-trained audio AI model (YAMNet) combined with PCA dimensionality reduction and K-means anomaly detection to monitor the vibroacoustic health of a Saraswati Veena. The system processes audio through a three-stage pipeline and classifies it as **Healthy**, **Watch** (early warning), or **Anomaly** (structural/tonal issue).

Unlike simple threshold detectors, this App provides:
* **Transfer Learning:** Leverages Google's YAMNet, pre-trained on 2M+ YouTube clips, as a feature extractor — no custom model training needed on the board.
* **Three Detection Tiers:** Healthy / Watch / Anomaly classification with configurable thresholds.
* **Three Input Modes:** File upload, live USB microphone recording, and batch testing of multiple files.
* **Visual Feedback:** The UNO Q's 13×8 blue LED matrix displays status icons (veena, exclamation mark, X, mic, spinner).
* **Edge Inference:** The entire pipeline runs on the UNO Q's MPU — no cloud required.

## Bricks Used

The example uses the following Bricks:

- `web_ui_streamlit`: Brick to create a Streamlit-based web interface for audio upload, recording, and result display.

## Hardware and Software Requirements

### Hardware

- Arduino UNO Q (x1)
- USB-C® cable (for power and programming) (x1)
- USB microphone (x1) — for live recording mode

### Software

- Arduino App Lab

**Note:** File upload and batch test modes work without a USB microphone. You can also run this example using your Arduino UNO Q as a Single Board Computer (SBC) using a [USB-C hub](https://store.arduino.cc/products/usb-c-to-hdmi-multiport-adapter-with-ethernet-and-usb-hub) with a mouse, keyboard, and monitor attached.

## How to Use the Example

1. **Run the App**

   Launch the App from Arduino App Lab. The Streamlit UI will start automatically.

2. **Access the Web Interface**

   Open the App in your browser at `<board-name>.local:7000` or `http://<board-ip-address>:7000`.

3. **Choose your input method:**

   - **📁 File Upload:** Upload a `.wav` audio file using the file uploader. Click **Analyze** to classify.
   - **🎙️ Live Record:** Click **Start Recording** to capture audio from the USB microphone. Recording stops automatically after 2 seconds.
   - **📂 Batch Test:** Upload multiple `.wav` files. The App processes each file and displays results in a summary table.

4. **View classification results:**

   Each analysis returns:
   - **Status:** Healthy ✅ / Watch ⚠️ / Anomaly 🔴
   - **Distance:** Numerical distance from the healthy center (lower = more healthy)
   - **Confidence:** Percentage based on distance relative to threshold

5. **Observe LED matrix feedback:**

   The UNO Q's LED matrix updates in real-time:
   | Status | LED Display |
   |--------|------------|
   | Healthy | Veena icon (steady) |
   | Watch | Exclamation mark (pulsing 1Hz) |
   | Anomaly | X mark (flashing 3Hz) |
   | Recording | Mic icon (pulsing 2Hz) |
   | Processing | Spinner (rotating 4Hz) |

## How it Works

Once the application is running, the device performs the following operations:

- **Processing audio through a transfer learning pipeline.**

  The App uses a three-stage ML pipeline entirely on-device:

  ```
  Audio (16kHz mono) → YAMNet → 521-D → PCA → 64-D → K-means Distance → Classification
  ```

  ```python
  # Load models
  yamnet = load_yamnet("models/yamnet.tflite")
  pca = np.load("models/pca_model.npz")
  center = np.load("models/kmeans_center.npy")
  std = np.load("models/kmeans_std.npy")
  ```

  YAMNet extracts a 521-dimensional embedding from 0.975 seconds of audio. PCA compresses it to 64 dimensions. K-means measures the Euclidean distance from the healthy cluster center.

- **Classifying based on distance from healthy center.**

  The anomaly detector measures how different new audio sounds compared to known healthy veena recordings:

  ```python
  # PCA projection
  features = (embedding - pca_mean) @ pca_components.T

  # Standardized Euclidean distance
  distance = np.sqrt(np.sum(((features - center) / std) ** 2))

  # Three-tier classification
  if distance < threshold * 0.5:
      status = "healthy"
  elif distance < threshold:
      status = "watch"
  else:
      status = "anomaly"
  ```

- **Communicating results to the MCU for LED feedback.**

  The Python backend sends classification results to the STM32 microcontroller via Bridge:

  ```python
  from arduino.api import Bridge

  Bridge.notify("set_status", status_code)  # 0=healthy, 1=watch, 2=anomaly
  ```

  The MCU receives the status and drives the 13×8 LED matrix with the corresponding animation pattern.

- **Providing a web interface via Streamlit.**

  The `web_ui_streamlit` Brick hosts the Streamlit application:

  ```python
  from arduino.api import WebUI

  WebUI()  # Initialize on-device web UI
  ```

  The Streamlit app runs on port 7000 and provides three tabs for different input modes.

The high-level data flow looks like this:

```
Audio Input → Resample 16kHz → YAMNet Embedding → PCA Compression → K-means Distance → Classification → LED Matrix + Web UI
```

## Understanding the Code

Here is a brief explanation of the App components:

### 🔧 Backend (`main.py`)

The Python component handles the entire ML inference pipeline and web interface.

- **Model loading**: Loads YAMNet TFLite, PCA model, K-means center/std, and config at startup using `@st.cache_resource` for efficient caching.

- **`run_pipeline()` function**: The core inference function that processes audio through all three stages (YAMNet → PCA → K-means) and returns a result dictionary with status, distance, and confidence.

- **`extract_embedding()` function**: Runs YAMNet TFLite inference on 0.975-second windows with 50% overlap, averaging scores across all windows to produce a single 521-D embedding.

- **`notify_mcu()` function**: Sends classification status codes (0-4) to the MCU via `Bridge.notify("set_status", code)` for LED matrix feedback.

- **`record_from_mic()` function**: Captures audio from the USB microphone using the App Lab Microphone API, returning a NumPy array at 16kHz.

- **Three Streamlit tabs**: File Upload (single WAV analysis), Live Record (microphone capture), and Batch Test (multi-file processing with summary table).

- **Graceful fallback**: When running on a PC (without Arduino hardware), the App mocks Bridge and Microphone imports so the ML pipeline can still be tested.

### 🔧 Arduino Component (`sketch.ino`)

The firmware drives the LED matrix and listens for status updates from the Python backend.

- **Bridge handler**: Receives classification results via `Bridge.provide("set_status", setStatus)` and updates the LED display mode.

- **LED matrix patterns**: Five display modes with distinct animations:
  - Healthy: steady veena icon
  - Watch: pulsing exclamation mark at 1Hz
  - Anomaly: flashing X mark at 3Hz
  - Recording: pulsing mic icon at 2Hz
  - Processing: rotating spinner at 4Hz

- **LED matrix packing**: The 13×8 matrix (104 pixels) is packed into `uint32_t[4]` arrays (128 bits). Pixel mapping: `flat = row * 13 + col → word[flat / 32], bit (31 - flat % 32)`.

  ```cpp
  void setStatus(int code) {
      currentMode = code;
  }

  void loop() {
      switch (currentMode) {
          case 0: showHealthy(); break;
          case 1: showWatch(); break;
          case 2: showAnomaly(); break;
          case 3: showRecording(); break;
          case 4: showProcessing(); break;
      }
  }
  ```

- **Sensor stubs (future)**: Commented scaffolding for INMP441 I2S microphone (SCK→D13, WS→D10, SD→D12) and vibration sensor (SDA→D20, SCL→D21) for direct MCU-side audio capture.

## Model Files

The App includes pre-trained model artifacts in `python/models/`:

| File | Size | Description |
|------|------|-------------|
| `yamnet.tflite` | ~3.9 MB | Google YAMNet — pre-trained audio feature extractor (521 classes) |
| `pca_model.npz` | ~263 KB | PCA compression recipe: mean (521) + components (64×521) |
| `kmeans_center.npy` | ~512 B | Healthy cluster center in 64-D space |
| `kmeans_std.npy` | ~512 B | Standard deviation per dimension for distance normalization |
| `config.json` | ~400 B | Anomaly threshold, sample rate, and pipeline settings |

**Total deployment size:** ~4 MB

## Training Pipeline

The models are trained offline on a PC using the `pipeline-validation/` scripts:

```bash
python 01_download_audio.py          # Download veena audio from Freesound
python 02_segment_audio.py           # Segment into 2s clips
python 03_prepare_training_data.py   # Split train/test + generate anomalies
python 07_yamnet_transfer.py         # Extract embeddings, validate, fit PCA
python export_models.py              # Package models → unoq-app/python/models/
```

The training pipeline uses 3 Freesound veena recordings (179 segments) to learn "what healthy veena sounds like." Anomaly detection is one-class: the model only knows healthy, and flags anything sufficiently different.

CSV embeddings are saved to `yamnet_embeddings/edge_impulse_upload/` for optional verification in [Edge Impulse Studio](https://studio.edgeimpulse.com/).
