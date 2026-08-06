"""
SwarCare — Saraswati Veena Health Monitor (WebUI Mode)
========================================================
Arduino UNO Q App Lab Application with WebUI-Streamlit brick.

When the WebUI-Streamlit brick is added, this file IS the Streamlit app.
It provides:
  - File upload + playback + anomaly detection
  - Live mic recording + analysis
  - LED feedback via Bridge

Pipeline: Audio → YAMNet (521-D) → PCA (64-D) → K-means → Anomaly Score

ON-DEVICE FILE STRUCTURE:
  python/
    main.py              ← this file (Streamlit app)
    requirements.txt     ← ai-edge-litert, numpy, scipy
    models/
      yamnet.tflite
      pca_model.npz
      kmeans_center.npy
      config.json
  sketch/
    sketch.ino           ← LED feedback via Bridge
"""

import os
import io
import time
import json
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import resample

# ─── Arduino imports (graceful fallback for PC testing) ───
try:
    from arduino.app_peripherals.microphone import Microphone
    from arduino.app_utils import Bridge, Logger
    from arduino.app_bricks.web_ui import WebUI
    ON_DEVICE = True
except ImportError:
    ON_DEVICE = False

    class Microphone:
        CHANNELS_MONO = 1
        @staticmethod
        def record_pcm(duration, sample_rate, channels, format):
            n = int(duration * sample_rate)
            t = np.linspace(0, duration, n, dtype=np.float32)
            tone = (np.sin(2 * np.pi * 220 * t) * 16000).astype(np.int16)
            noise = np.random.randint(-500, 500, n, dtype=np.int16)
            return tone + noise

    class Bridge:
        @staticmethod
        def notify(name, *args): pass

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
YAMNET_PATH = os.path.join(MODELS_DIR, "yamnet.tflite")
PCA_PATH = os.path.join(MODELS_DIR, "pca_model.npz")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_center.npy")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")

TARGET_SR = 16000
WINDOW_SEC = 2.0

# ─── Initialize WebUI brick (required for App Lab lifecycle) ───
if ON_DEVICE:
    web_ui = WebUI()


# ─── Model Loading (cached across Streamlit reruns) ───

@st.cache_resource
def load_interpreter():
    """Load YAMNet TFLite interpreter."""
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
        raise ImportError("No TFLite runtime found")
    interp = Interpreter(model_path=YAMNET_PATH)
    interp.allocate_tensors()
    return interp


@st.cache_resource
def load_pca():
    """Load PCA: mean (521,) and components (64, 521)."""
    data = np.load(PCA_PATH)
    return data["mean"].astype(np.float32), data["components"].astype(np.float32)


@st.cache_resource
def load_kmeans():
    """Load K-means center (64,)."""
    return np.load(KMEANS_PATH).astype(np.float32)


@st.cache_resource
def load_config():
    """Load threshold config."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"anomaly_threshold": 0.62}


# ─── Inference Pipeline ───

def run_yamnet(interpreter, audio):
    """YAMNet: audio → averaged 521-D scores."""
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
        if len(inp["shape"]) == 1:
            data = padded
        else:
            data = padded.reshape(1, -1)
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()
        scores_list.append(interpreter.get_tensor(out["index"]).flatten())

    return np.mean(scores_list, axis=0), len(scores_list)


def run_pipeline(audio):
    """Full: audio → YAMNet → PCA → K-means → result dict."""
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
        "status": status,
        "score": distance,
        "threshold": threshold,
        "is_anomaly": distance > threshold,
        "n_windows": n_windows,
        "timing": {
            "yamnet_ms": round(t_yamnet * 1000),
            "pca_ms": round((t_pca - t_yamnet) * 1000, 1),
            "total_ms": round(t_total * 1000),
        },
    }


def notify_mcu(status):
    """Send status to MCU for LED feedback."""
    code = {"healthy": 0, "watch": 1, "anomaly": 2}.get(status, 2)
    Bridge.notify("set_status", code)


def load_audio_file(uploaded_file):
    """Load WAV → 16kHz mono float32."""
    audio_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    sr, audio = wavfile.read(io.BytesIO(audio_bytes))

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)

    if sr != TARGET_SR:
        num_samples = int(len(audio) * TARGET_SR / sr)
        audio = resample(audio, num_samples).astype(np.float32)
        sr = TARGET_SR

    return audio, sr


def record_from_mic():
    """Record from USB mic (or stub)."""
    try:
        pcm = Microphone.record_pcm(
            duration=WINDOW_SEC,
            sample_rate=TARGET_SR,
            channels=Microphone.CHANNELS_MONO,
            format=np.int16,
        )
        return pcm.astype(np.float32) / 32768.0
    except Exception as e:
        st.warning(f"Mic unavailable: {e}. Using test tone.")
        n = int(WINDOW_SEC * TARGET_SR)
        t = np.linspace(0, WINDOW_SEC, n, dtype=np.float32)
        return np.sin(2 * np.pi * 220 * t) * 0.5


# ─── Streamlit UI ───

st.set_page_config(page_title="SwarCare", page_icon="🎵", layout="wide")

st.title("🎵 SwarCare — Veena Anomaly Detector")
st.caption("Detect acoustic anomalies in Saraswati Veena recordings using YAMNet + PCA + K-means")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pipeline")
    st.code("WAV → YAMNet(521D) → PCA(64D) → K-means → Score", language=None)
    config = load_config()
    st.metric("Anomaly Threshold", f"{config.get('anomaly_threshold', 0.62):.2f}")
    st.divider()
    st.markdown("**Models:**")
    for name, path in [("YAMNet TFLite", YAMNET_PATH), ("PCA (npz)", PCA_PATH), ("K-means", KMEANS_PATH)]:
        ok = "✅" if os.path.exists(path) else "❌"
        sz = f"{os.path.getsize(path)/1024:.0f}KB" if os.path.exists(path) else "missing"
        st.text(f"{ok} {name} ({sz})")
    st.divider()
    st.text(f"Device: {'UNO Q' if ON_DEVICE else 'PC (testing)'}")

# Tabs
tab_file, tab_live, tab_batch = st.tabs(["📁 File Upload", "🎤 Live Record", "📂 Batch Test"])

# ─── TAB 1: File Upload ───
with tab_file:
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"], key="single")

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.audio(uploaded, format="audio/wav")
            audio, sr = load_audio_file(uploaded)
            st.caption(f"{len(audio)/sr:.2f}s | {sr}Hz | {len(audio)} samples")

        with col2:
            if st.button("🔍 Analyze", type="primary", key="analyze_file"):
                with st.spinner("Running pipeline..."):
                    result = run_pipeline(audio)
                    notify_mcu(result["status"])

                if result["status"] == "healthy":
                    st.success(f"✅ HEALTHY — Score: {result['score']:.4f}")
                elif result["status"] == "watch":
                    st.warning(f"⚠️ WATCH — Score: {result['score']:.4f}")
                else:
                    st.error(f"🚨 ANOMALY — Score: {result['score']:.4f}")

                st.progress(min(result["score"] / (result["threshold"] * 2), 1.0))
                st.caption(f"Threshold: {result['threshold']:.2f} | Score: {result['score']:.4f}")

                c1, c2, c3 = st.columns(3)
                c1.metric("YAMNet", f"{result['timing']['yamnet_ms']}ms")
                c2.metric("PCA+KM", f"{result['timing']['pca_ms']}ms")
                c3.metric("Total", f"{result['timing']['total_ms']}ms")

                with st.expander("Details"):
                    st.json(result)

# ─── TAB 2: Live Record ───
with tab_live:
    st.markdown("Record **2 seconds** from the USB microphone and analyze in real-time.")

    if st.button("🎤 Record & Analyze", type="primary", key="live_record"):
        with st.spinner(f"Recording {WINDOW_SEC}s..."):
            audio = record_from_mic()

        st.audio(audio, sample_rate=TARGET_SR)
        st.caption(f"Recorded: {len(audio)/TARGET_SR:.2f}s | {TARGET_SR}Hz")

        with st.spinner("Analyzing..."):
            result = run_pipeline(audio)
            notify_mcu(result["status"])

        if result["status"] == "healthy":
            st.success(f"✅ HEALTHY — Score: {result['score']:.4f}")
        elif result["status"] == "watch":
            st.warning(f"⚠️ WATCH — Score: {result['score']:.4f}")
        else:
            st.error(f"🚨 ANOMALY — Score: {result['score']:.4f}")

        st.progress(min(result["score"] / (result["threshold"] * 2), 1.0))

        c1, c2, c3 = st.columns(3)
        c1.metric("YAMNet", f"{result['timing']['yamnet_ms']}ms")
        c2.metric("PCA+KM", f"{result['timing']['pca_ms']}ms")
        c3.metric("Total", f"{result['timing']['total_ms']}ms")

# ─── TAB 3: Batch Test ───
with tab_batch:
    batch_files = st.file_uploader(
        "Upload multiple WAV files",
        type=["wav"],
        accept_multiple_files=True,
        key="batch",
    )

    if batch_files and st.button("🚀 Run Batch", type="primary", key="run_batch"):
        results = []
        bar = st.progress(0)
        for i, f in enumerate(batch_files):
            audio, sr = load_audio_file(f)
            r = run_pipeline(audio)
            results.append({
                "File": f.name,
                "Status": r["status"].upper(),
                "Score": f"{r['score']:.4f}",
                "Anomaly": "Yes" if r["is_anomaly"] else "No",
                "Time (ms)": r["timing"]["total_ms"],
            })
            bar.progress((i + 1) / len(batch_files))

        st.dataframe(results, use_container_width=True)

        statuses = [r["Status"] for r in results]
        col1, col2, col3 = st.columns(3)
        col1.metric("Healthy", statuses.count("HEALTHY"))
        col2.metric("Watch", statuses.count("WATCH"))
        col3.metric("Anomaly", statuses.count("ANOMALY"))
