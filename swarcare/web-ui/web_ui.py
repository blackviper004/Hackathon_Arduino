"""
SwarCare — Web UI for Audio Anomaly Detection
===============================================
Upload WAV files, play them, and test the YAMNet → PCA → K-means pipeline.
Similar to Edge Impulse's model testing interface.

Run: streamlit run web_ui.py
"""

import os
import io
import time
import numpy as np
import streamlit as st
from scipy.io import wavfile
from scipy.signal import resample

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "unoq-app", "models")
YAMNET_PATH = os.path.join(MODELS_DIR, "yamnet.tflite")
PCA_PATH = os.path.join(MODELS_DIR, "pca_model.npz")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_center.npy")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")

TARGET_SR = 16000


# ─── Model Loading (cached) ───

@st.cache_resource
def load_yamnet():
    """Load YAMNet TFLite model."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path=YAMNET_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_pca():
    """Load PCA model."""
    data = np.load(PCA_PATH)
    return data["mean"].astype(np.float32), data["components"].astype(np.float32)


@st.cache_resource
def load_kmeans():
    """Load K-means center."""
    return np.load(KMEANS_PATH).astype(np.float32)


@st.cache_resource
def load_config():
    """Load config."""
    import json
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"anomaly_threshold": 0.62}


# ─── Inference Functions ───

def yamnet_infer(interpreter, audio):
    """Run YAMNet on audio, return averaged 521-D scores."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]["shape"][-1]  # 15600

    audio = audio.astype(np.float32)
    all_scores = []
    hop = input_size // 2

    for start in range(0, len(audio) - input_size + 1, hop):
        chunk = audio[start:start + input_size]
        input_shape = input_details[0]["shape"]
        if len(input_shape) == 1:
            data = chunk.reshape(input_shape)
        else:
            data = chunk.reshape(1, -1)
            if list(data.shape) != list(input_shape):
                interpreter.resize_tensor_input(input_details[0]["index"], list(data.shape))
                interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]["index"], data)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]["index"]).flatten()
        all_scores.append(scores)

    if not all_scores:
        padded = np.zeros(input_size, dtype=np.float32)
        padded[:min(len(audio), input_size)] = audio[:input_size]
        input_shape = input_details[0]["shape"]
        if len(input_shape) == 1:
            data = padded.reshape(input_shape)
        else:
            data = padded.reshape(1, -1)
        interpreter.set_tensor(input_details[0]["index"], data)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]["index"]).flatten()
        all_scores.append(scores)

    return np.mean(all_scores, axis=0), all_scores


def pca_transform(scores, pca_mean, pca_components):
    """Apply PCA: 521 → 64."""
    return (scores - pca_mean) @ pca_components.T


def kmeans_predict(features, center, threshold):
    """Compute anomaly score (Euclidean distance to center)."""
    distance = float(np.sqrt(np.sum((features - center) ** 2)))
    if distance < threshold * 0.5:
        status = "healthy"
    elif distance < threshold:
        status = "watch"
    else:
        status = "anomaly"
    return distance, status


def load_audio(uploaded_file):
    """Load audio file and resample to 16kHz mono."""
    audio_bytes = uploaded_file.read()
    sr, audio = wavfile.read(io.BytesIO(audio_bytes))

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)

    # Resample to target SR
    if sr != TARGET_SR:
        num_samples = int(len(audio) * TARGET_SR / sr)
        audio = resample(audio, num_samples).astype(np.float32)
        sr = TARGET_SR

    return audio, sr, audio_bytes


# ─── Streamlit UI ───

def main():
    st.set_page_config(
        page_title="SwarCare — Veena Health Monitor",
        page_icon="🎵",
        layout="wide",
    )

    st.title("🎵 SwarCare — Veena Anomaly Detector")
    st.markdown(
        "Upload a WAV file to test if the Saraswati Veena audio is **healthy** or has **anomalies** "
        "(buzzing, muffled, rattling)."
    )
    st.divider()

    # Sidebar — Model info
    with st.sidebar:
        st.header("Pipeline Info")
        st.markdown("""
        **Inference Pipeline:**
        1. Audio (16kHz mono)
        2. → YAMNet TFLite (521-D scores)
        3. → PCA (64-D features)
        4. → K-means distance (anomaly score)

        **Threshold:** Score > {threshold} = Anomaly
        """.format(threshold=load_config().get("anomaly_threshold", 0.62)))

        st.divider()
        st.markdown("**Model Files:**")
        for name, path in [("YAMNet", YAMNET_PATH), ("PCA", PCA_PATH), ("K-means", KMEANS_PATH)]:
            exists = "✅" if os.path.exists(path) else "❌"
            size = f"({os.path.getsize(path) / 1024:.0f}KB)" if os.path.exists(path) else ""
            st.text(f"{exists} {name} {size}")

    # Main area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Audio")
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=["wav"],
            help="Upload a recording of Saraswati Veena (16kHz mono preferred)"
        )

        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            st.caption(f"File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

    with col2:
        if uploaded_file:
            st.subheader("🔍 Analysis Results")

            # Load and process
            audio, sr, _ = load_audio(uploaded_file)
            duration = len(audio) / sr

            st.text(f"Duration: {duration:.2f}s | Sample Rate: {sr}Hz | Samples: {len(audio)}")

            # Run inference
            with st.spinner("Running inference pipeline..."):
                t0 = time.time()

                # Step 1: YAMNet
                interpreter = load_yamnet()
                scores_avg, all_scores = yamnet_infer(interpreter, audio)
                t_yamnet = time.time() - t0

                # Step 2: PCA
                pca_mean, pca_components = load_pca()
                features = pca_transform(scores_avg, pca_mean, pca_components)
                t_pca = time.time() - t0

                # Step 3: K-means
                center = load_kmeans()
                config = load_config()
                threshold = config.get("anomaly_threshold", 0.62)
                distance, status = kmeans_predict(features, center, threshold)
                t_total = time.time() - t0

            # Display result
            if status == "healthy":
                st.success(f"✅ **HEALTHY** — Score: {distance:.4f}", icon="✅")
            elif status == "watch":
                st.warning(f"⚠️ **WATCH** — Score: {distance:.4f}", icon="⚠️")
            else:
                st.error(f"🚨 **ANOMALY DETECTED** — Score: {distance:.4f}", icon="🚨")

            # Score gauge
            st.progress(min(distance / (threshold * 2), 1.0))
            st.caption(
                f"Threshold: {threshold:.2f} | "
                f"Your score: {distance:.4f} | "
                f"{'ABOVE' if distance > threshold else 'BELOW'} threshold"
            )

            # Timing
            st.divider()
            st.markdown("**⏱ Inference Timing:**")
            tcol1, tcol2, tcol3 = st.columns(3)
            tcol1.metric("YAMNet", f"{t_yamnet * 1000:.0f}ms")
            tcol2.metric("PCA", f"{(t_pca - t_yamnet) * 1000:.1f}ms")
            tcol3.metric("Total", f"{t_total * 1000:.0f}ms")

            # Details expander
            with st.expander("📊 Detailed Results"):
                st.markdown("**YAMNet Scores (top 10 classes):**")
                # Load YAMNet class names if available
                top_indices = np.argsort(scores_avg)[::-1][:10]
                for i, idx in enumerate(top_indices):
                    st.text(f"  {i + 1}. Class {idx}: {scores_avg[idx]:.4f}")

                st.markdown(f"\n**PCA Features (64-D):** min={features.min():.3f}, max={features.max():.3f}")
                st.markdown(f"**Distance to healthy center:** {distance:.4f}")
                st.markdown(f"**Number of YAMNet windows:** {len(all_scores)}")

    # Batch testing section
    st.divider()
    st.subheader("📂 Batch Testing")
    uploaded_files = st.file_uploader(
        "Upload multiple WAV files for batch testing",
        type=["wav"],
        accept_multiple_files=True,
        key="batch"
    )

    if uploaded_files and len(uploaded_files) > 1:
        if st.button("🚀 Run Batch Analysis", type="primary"):
            interpreter = load_yamnet()
            pca_mean, pca_components = load_pca()
            center = load_kmeans()
            config = load_config()
            threshold = config.get("anomaly_threshold", 0.62)

            results = []
            progress = st.progress(0)

            for i, f in enumerate(uploaded_files):
                audio, sr, _ = load_audio(f)
                scores_avg, _ = yamnet_infer(interpreter, audio)
                features = pca_transform(scores_avg, pca_mean, pca_components)
                distance, status = kmeans_predict(features, center, threshold)
                results.append({
                    "File": f.name,
                    "Status": status.upper(),
                    "Score": f"{distance:.4f}",
                    "Duration": f"{len(audio) / sr:.1f}s",
                })
                progress.progress((i + 1) / len(uploaded_files))

            st.dataframe(results, use_container_width=True)

            # Summary
            statuses = [r["Status"] for r in results]
            healthy = statuses.count("HEALTHY")
            watch = statuses.count("WATCH")
            anomaly = statuses.count("ANOMALY")
            st.markdown(
                f"**Summary:** {healthy} healthy, {watch} watch, {anomaly} anomaly "
                f"out of {len(results)} files"
            )


if __name__ == "__main__":
    main()
