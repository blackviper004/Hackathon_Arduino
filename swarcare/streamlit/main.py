"""
SwarCare — Streamlit Web UI (runs on UNO Q via WebUI-Streamlit brick)
======================================================================
Upload WAV files, play them, and test the YAMNet → PCA → K-means pipeline.

This file lives in: streamlit/main.py
The WebUI-Streamlit brick serves it on the UNO Q's network.
Access via browser at: http://<board-ip>:8501
"""

import os
import io
import time
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import resample

# ─── Paths (on-device layout) ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are in the python/ sibling folder
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "python", "models")
YAMNET_PATH = os.path.join(MODELS_DIR, "yamnet.tflite")
PCA_PATH = os.path.join(MODELS_DIR, "pca_model.npz")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_center.npy")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")

TARGET_SR = 16000


# ─── Model Loading (cached) ───

@st.cache_resource
def load_interpreter():
    """Load YAMNet TFLite interpreter."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=YAMNET_PATH)
    interp.allocate_tensors()
    return interp


@st.cache_resource
def load_pca():
    """Load PCA parameters."""
    data = np.load(PCA_PATH)
    return data["mean"].astype(np.float32), data["components"].astype(np.float32)


@st.cache_resource
def load_kmeans():
    """Load K-means center."""
    return np.load(KMEANS_PATH).astype(np.float32)


@st.cache_resource
def load_config():
    """Load threshold config."""
    import json
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"anomaly_threshold": 0.62}


# ─── Inference ───

def run_yamnet(interpreter, audio):
    """YAMNet inference → averaged 521-D scores."""
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
    """Full pipeline: audio → YAMNet → PCA → K-means → result."""
    t0 = time.time()

    interpreter = load_interpreter()
    scores, n_windows = run_yamnet(interpreter, audio)
    t_yamnet = time.time() - t0

    pca_mean, pca_components = load_pca()
    features = (scores - pca_mean) @ pca_components.T
    t_pca = time.time() - t0

    center = load_kmeans()
    config = load_config()
    threshold = config.get("anomaly_threshold", 0.62)
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
        "features_stats": {
            "min": float(features.min()),
            "max": float(features.max()),
            "mean": float(features.mean()),
        },
    }


def load_audio_file(uploaded_file):
    """Load and preprocess audio to 16kHz mono float32."""
    audio_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset for audio player
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


# ─── UI ───

def main():
    st.set_page_config(
        page_title="SwarCare",
        page_icon="🎵",
        layout="wide",
    )

    st.title("🎵 SwarCare — Veena Anomaly Detector")
    st.caption("Upload a WAV recording to check if the Saraswati Veena sounds healthy or has acoustic issues.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pipeline")
        st.markdown("""
        ```
        WAV (16kHz)
         → YAMNet → 521-D
         → PCA → 64-D
         → K-means → Score
        ```
        """)
        config = load_config()
        st.metric("Threshold", f"{config.get('anomaly_threshold', 0.62):.2f}")
        st.divider()
        st.markdown("**Models:**")
        for name, path in [("YAMNet", YAMNET_PATH), ("PCA", PCA_PATH), ("K-means", KMEANS_PATH)]:
            ok = "✅" if os.path.exists(path) else "❌"
            sz = f"{os.path.getsize(path)/1024:.0f}KB" if os.path.exists(path) else "missing"
            st.text(f"{ok} {name} ({sz})")

    # --- Single file analysis ---
    st.subheader("🎤 Analyze Audio")
    uploaded = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.audio(uploaded, format="audio/wav")
            audio, sr = load_audio_file(uploaded)
            st.caption(f"📊 {len(audio)/sr:.2f}s | {sr}Hz | {len(audio)} samples")

        with col2:
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("Running YAMNet → PCA → K-means..."):
                    result = run_pipeline(audio)

                # Result display
                if result["status"] == "healthy":
                    st.success(f"✅ HEALTHY — Score: {result['score']:.4f}")
                elif result["status"] == "watch":
                    st.warning(f"⚠️ WATCH — Score: {result['score']:.4f}")
                else:
                    st.error(f"🚨 ANOMALY — Score: {result['score']:.4f}")

                # Score bar
                progress_val = min(result["score"] / (result["threshold"] * 2), 1.0)
                st.progress(progress_val)
                st.caption(f"Threshold: {result['threshold']:.2f} | Your score: {result['score']:.4f}")

                # Timing metrics
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("YAMNet", f"{result['timing']['yamnet_ms']}ms")
                c2.metric("PCA+K-means", f"{result['timing']['pca_ms']}ms")
                c3.metric("Total", f"{result['timing']['total_ms']}ms")

                with st.expander("Details"):
                    st.json(result)

    # --- Batch testing ---
    st.divider()
    st.subheader("📂 Batch Test")
    batch_files = st.file_uploader(
        "Upload multiple WAV files",
        type=["wav"],
        accept_multiple_files=True,
        key="batch",
    )

    if batch_files and st.button("🚀 Run Batch", type="primary"):
        results = []
        bar = st.progress(0)
        for i, f in enumerate(batch_files):
            audio, sr = load_audio_file(f)
            r = run_pipeline(audio)
            results.append({
                "File": f.name,
                "Status": r["status"].upper(),
                "Score": f"{r['score']:.4f}",
                "Time": f"{r['timing']['total_ms']}ms",
            })
            bar.progress((i + 1) / len(batch_files))

        st.dataframe(results, use_container_width=True)
        statuses = [r["Status"] for r in results]
        st.markdown(
            f"**{statuses.count('HEALTHY')}** healthy, "
            f"**{statuses.count('WATCH')}** watch, "
            f"**{statuses.count('ANOMALY')}** anomaly"
        )


if __name__ == "__main__":
    main()
