"""
SwarCare — Approach 3: Transfer Learning with YAMNet
=====================================================
Uses Google's YAMNet (pre-trained on 2M+ YouTube clips) as a feature extractor.
YAMNet already understands audio — we just teach it "what does healthy Veena sound like?"

Pipeline:
  Audio → YAMNet → 1024-D embedding → Simple anomaly detector → Score

Why this works:
  - YAMNet's 1024-D embeddings capture MEANINGFUL audio characteristics
  - Unlike raw MFE (3960 features of time×frequency), embeddings encode
    concepts like "plucked string", "resonance", "tonal vs noisy"
  - We only need a simple model on TOP of those embeddings
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

try:
    import tensorflow as tf
except ImportError:
    print("pip install tensorflow")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("pip install librosa soundfile")
    sys.exit(1)


# ─── PATHS ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YAMNET_TFLITE = os.path.join(SCRIPT_DIR, "yamnet.tflite")
YAMNET_URL = "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite"

# Data paths (both Freesound and synthetic)
FREESOUND_HEALTHY_TRAIN = os.path.join(SCRIPT_DIR, "training_data", "training", "healthy")
FREESOUND_HEALTHY_TEST = os.path.join(SCRIPT_DIR, "training_data", "testing", "healthy")
FREESOUND_ANOMALY = os.path.join(SCRIPT_DIR, "training_data", "testing", "anomaly")
SYNTH_HEALTHY = os.path.join(SCRIPT_DIR, "training_data", "synthetic", "healthy")
SYNTH_ANOMALY = os.path.join(SCRIPT_DIR, "training_data", "synthetic", "anomaly")
RAW_AUDIO = os.path.join(SCRIPT_DIR, "raw_audio")


def download_yamnet():
    """Download YAMNet TFLite model if not present."""
    if os.path.exists(YAMNET_TFLITE):
        size_mb = os.path.getsize(YAMNET_TFLITE) / 1024 / 1024
        print(f"YAMNet TFLite already downloaded ({size_mb:.1f} MB)")
        return

    print("Downloading YAMNet TFLite (~4MB)...")
    import urllib.request
    # Kaggle Models direct download
    url = "https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1?lite-format=tflite"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            with open(YAMNET_TFLITE, 'wb') as f:
                f.write(response.read())
    except Exception as e:
        print(f"\n  Auto-download failed: {e}")
        print(f"\n  MANUAL DOWNLOAD REQUIRED:")
        print(f"  1. Go to: https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1")
        print(f"  2. Click 'Download' button")
        print(f"  3. Save the .tflite file as: {YAMNET_TFLITE}")
        print(f"  4. Re-run this script")
        sys.exit(1)
    size_mb = os.path.getsize(YAMNET_TFLITE) / 1024 / 1024
    print(f"Downloaded: {size_mb:.1f} MB")


def load_yamnet():
    """Load YAMNet TFLite model."""
    download_yamnet()
    interpreter = tf.lite.Interpreter(model_path=YAMNET_TFLITE)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"YAMNet TFLite loaded!")
    print(f"  Input:  {input_details[0]['shape']} {input_details[0]['dtype']}")
    for i, od in enumerate(output_details):
        print(f"  Output {i}: {od['shape']} {od['dtype']}")

    return interpreter


def extract_embedding(interpreter, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Run YAMNet TFLite on audio and return embedding.

    YAMNet TFLite takes 15600 samples (0.975s at 16kHz).
    For longer audio, we process in windows and average.
    """
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    audio = audio.astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected_len = input_details[0]['shape'][-1]  # Usually 15600

    # Process audio in windows of expected_len
    all_scores = []
    hop = expected_len // 2  # 50% overlap

    for start in range(0, len(audio) - expected_len + 1, hop):
        chunk = audio[start:start + expected_len]
        chunk = chunk.reshape(1, -1).astype(np.float32)

        interpreter.resize_tensor_input(input_details[0]['index'], chunk.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], chunk)
        interpreter.invoke()

        # Get the scores (521 classes) — this IS our embedding
        scores = interpreter.get_tensor(output_details[0]['index'])
        all_scores.append(scores.flatten())

    if not all_scores:
        # Audio too short — pad it
        padded = np.zeros(expected_len, dtype=np.float32)
        padded[:len(audio)] = audio[:expected_len]
        padded = padded.reshape(1, -1)

        interpreter.resize_tensor_input(input_details[0]['index'], padded.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], padded)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[0]['index'])
        all_scores.append(scores.flatten())

    # Average across all windows
    return np.mean(all_scores, axis=0)


def extract_embeddings_from_folder(interpreter, folder: str, max_files: int = None) -> tuple:
    """Extract embeddings for all WAV files in a folder."""
    if not os.path.exists(folder):
        print(f"  [SKIP] Folder not found: {folder}")
        return np.array([]), []

    files = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])
    if max_files:
        files = files[:max_files]

    embeddings = []
    filenames = []
    for f in files:
        filepath = os.path.join(folder, f)
        audio, sr = librosa.load(filepath, sr=16000, mono=True)
        emb = extract_embedding(interpreter, audio, sr)
        embeddings.append(emb)
        filenames.append(f)

    if embeddings:
        return np.array(embeddings), filenames
    return np.array([]), []


class SimpleAnomalyDetector:
    """
    Simple anomaly detector based on Mahalanobis-like distance.

    Learns the mean and covariance of healthy embeddings.
    New samples are scored by their distance from the healthy distribution.

    This is essentially what GMM with 1 component does,
    but we control it properly without Edge Impulse bugs.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.threshold = None

    def fit(self, healthy_embeddings: np.ndarray):
        """Learn the healthy distribution."""
        self.mean = np.mean(healthy_embeddings, axis=0)  # (1024,)
        self.std = np.std(healthy_embeddings, axis=0) + 1e-8  # (1024,) avoid div by 0

        # Compute distances of training data to set threshold
        train_distances = self._compute_distances(healthy_embeddings)
        # Threshold = mean + 2*std of training distances (95th percentile)
        self.threshold = np.mean(train_distances) + 2 * np.std(train_distances)

        print(f"  Trained on {len(healthy_embeddings)} samples")
        print(f"  Embedding dimension: {len(self.mean)}")
        print(f"  Training distance: mean={np.mean(train_distances):.2f}, std={np.std(train_distances):.2f}")
        print(f"  Threshold: {self.threshold:.2f}")

    def _compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute normalized distance from healthy mean."""
        # Z-score each dimension, then compute Euclidean distance
        z_scores = (embeddings - self.mean) / self.std
        distances = np.sqrt(np.mean(z_scores ** 2, axis=1))
        return distances

    def predict(self, embeddings: np.ndarray) -> list:
        """
        Predict anomaly scores and labels.
        Returns list of (distance, is_anomaly, status).
        """
        distances = self._compute_distances(embeddings)
        results = []
        for d in distances:
            if d < self.threshold * 0.7:
                status = "healthy"
            elif d < self.threshold:
                status = "watch"
            else:
                status = "anomaly"
            results.append((float(d), d > self.threshold, status))
        return results


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SwarCare — Transfer Learning with YAMNet               ║")
    print("║  Pre-trained audio AI → Veena anomaly detection          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ─── 1. LOAD YAMNET ───
    interpreter = load_yamnet()

    # ─── 2. EXTRACT EMBEDDINGS FROM ALL DATA ───
    print("\n" + "=" * 60)
    print("  Extracting YAMNet embeddings from audio files...")
    print("=" * 60)

    # Freesound healthy (training)
    print("\n  Freesound healthy (training):")
    fs_healthy_emb, fs_healthy_names = extract_embeddings_from_folder(
        interpreter, FREESOUND_HEALTHY_TRAIN)
    print(f"    → {len(fs_healthy_emb)} embeddings")

    # Freesound healthy (test)
    print("\n  Freesound healthy (test):")
    fs_healthy_test_emb, fs_healthy_test_names = extract_embeddings_from_folder(
        interpreter, FREESOUND_HEALTHY_TEST)
    print(f"    → {len(fs_healthy_test_emb)} embeddings")

    # Freesound anomaly
    print("\n  Freesound anomaly (test):")
    fs_anomaly_emb, fs_anomaly_names = extract_embeddings_from_folder(
        interpreter, FREESOUND_ANOMALY)
    print(f"    → {len(fs_anomaly_emb)} embeddings")

    # Synthetic healthy
    print("\n  Synthetic healthy:")
    syn_healthy_emb, syn_healthy_names = extract_embeddings_from_folder(
        interpreter, SYNTH_HEALTHY)
    print(f"    → {len(syn_healthy_emb)} embeddings")

    # Synthetic anomaly
    print("\n  Synthetic anomaly:")
    syn_anomaly_emb, syn_anomaly_names = extract_embeddings_from_folder(
        interpreter, SYNTH_ANOMALY)
    print(f"    → {len(syn_anomaly_emb)} embeddings")

    # ─── 3. TRAIN ANOMALY DETECTOR ───
    print("\n" + "=" * 60)
    print("  Training anomaly detector on Freesound healthy embeddings...")
    print("=" * 60)

    detector = SimpleAnomalyDetector()
    detector.fit(fs_healthy_emb)

    # ─── 4. TEST ON ALL DATA ───
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    def test_set(name, embeddings, filenames, expected):
        if len(embeddings) == 0:
            return
        results = detector.predict(embeddings)
        statuses = [r[2] for r in results]

        healthy_count = statuses.count("healthy")
        watch_count = statuses.count("watch")
        anomaly_count = statuses.count("anomaly")

        print(f"\n  {name} ({len(embeddings)} samples, expected: {expected}):")
        print(f"    Healthy: {healthy_count} ({100*healthy_count/len(statuses):.0f}%)")
        print(f"    Watch:   {watch_count} ({100*watch_count/len(statuses):.0f}%)")
        print(f"    Anomaly: {anomaly_count} ({100*anomaly_count/len(statuses):.0f}%)")

        # Show a few examples
        distances = [r[0] for r in results]
        print(f"    Distance: min={min(distances):.2f}, max={max(distances):.2f}, mean={np.mean(distances):.2f}")

        # Accuracy
        if expected == "healthy":
            correct = healthy_count + watch_count
        else:
            correct = anomaly_count
        print(f"    Accuracy: {100*correct/len(statuses):.0f}%")

    # Test on each dataset
    test_set("Freesound Healthy (TEST)", fs_healthy_test_emb, fs_healthy_test_names, "healthy")

    # Split anomalies by type
    detuned = [(e, n) for e, n in zip(fs_anomaly_emb, fs_anomaly_names) if "detuned" in n]
    buzzing = [(e, n) for e, n in zip(fs_anomaly_emb, fs_anomaly_names) if "buzzing" in n]
    muffled = [(e, n) for e, n in zip(fs_anomaly_emb, fs_anomaly_names) if "muffled" in n]

    if detuned:
        embs = np.array([e for e, n in detuned])
        test_set("Freesound DETUNED", embs, [n for e, n in detuned], "anomaly")
    if buzzing:
        embs = np.array([e for e, n in buzzing])
        test_set("Freesound BUZZING", embs, [n for e, n in buzzing], "anomaly")
    if muffled:
        embs = np.array([e for e, n in muffled])
        test_set("Freesound MUFFLED", embs, [n for e, n in muffled], "anomaly")

    test_set("Synthetic Healthy", syn_healthy_emb, syn_healthy_names, "healthy")
    test_set("Synthetic Anomaly", syn_anomaly_emb, syn_anomaly_names, "anomaly")

    # ─── 5. FULL RECORDING TEST ───
    print("\n" + "=" * 60)
    print("  Full Recording Embeddings")
    print("=" * 60)
    if os.path.exists(RAW_AUDIO):
        for f in sorted(os.listdir(RAW_AUDIO)):
            if f.endswith('.wav'):
                audio, sr = librosa.load(os.path.join(RAW_AUDIO, f), sr=16000, mono=True)
                emb = extract_embedding(interpreter, audio, sr)
                result = detector.predict(emb.reshape(1, -1))[0]
                print(f"  {f[:50]:50s} → distance={result[0]:.2f} [{result[2]}]")

    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    print("  If Freesound healthy → mostly 'healthy'")
    print("  AND anomalies → mostly 'anomaly'")
    print("  → Transfer learning WORKS. YAMNet embeddings are useful.")
    print("=" * 60)

    # ─── 6. FIT PCA AND SAVE MODELS ───
    print("\n" + "=" * 60)
    print("  Fitting PCA (521 → 64) and saving models for deployment...")
    print("=" * 60)

    if len(fs_healthy_emb) == 0:
        print("  [ERROR] No healthy training embeddings! Cannot fit PCA.")
        print("  Make sure training_data/training/healthy/ has WAV files.")
        print("  Run: python 03_prepare_training_data.py")
        return

    PCA_COMPONENTS = 64

    # PCA: compute mean and top eigenvectors
    X = np.array(fs_healthy_emb)
    pca_mean = np.mean(X, axis=0)
    X_centered = X - pca_mean
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    pca_components = eigenvectors[:, :PCA_COMPONENTS].T  # (64, 521)

    explained = np.sum(sorted(eigenvalues, reverse=True)[:PCA_COMPONENTS]) / np.sum(eigenvalues)
    print(f"  PCA explained variance: {explained * 100:.1f}%")
    print(f"  PCA shape: mean={pca_mean.shape}, components={pca_components.shape}")

    # Save PCA model
    output_dir = os.path.join(SCRIPT_DIR, "yamnet_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    pca_path = os.path.join(output_dir, "pca_model.npz")
    np.savez(pca_path, mean=pca_mean, components=pca_components, n_components=PCA_COMPONENTS)
    print(f"  Saved: {pca_path} ({os.path.getsize(pca_path)/1024:.1f} KB)")

    # Transform training data to 64-D and save CSV for export_models.py
    pca_features = X_centered @ pca_components.T  # (N, 64)
    print(f"  Training features shape: {pca_features.shape}")

    # Helper to save CSVs for a set of embeddings
    col_names = [f"pca_{i}" for i in range(PCA_COMPONENTS)]

    def save_csvs(embeddings, filenames, subfolder, label):
        if len(embeddings) == 0:
            return 0
        centered = embeddings - pca_mean
        features = centered @ pca_components.T
        csv_dir = os.path.join(output_dir, "edge_impulse_upload", subfolder)
        os.makedirs(csv_dir, exist_ok=True)
        for i, name in enumerate(filenames):
            csv_name = f"{label}.{name.replace('.wav', '')}.csv"
            csv_path = os.path.join(csv_dir, csv_name)
            with open(csv_path, 'w') as f:
                f.write(','.join(col_names) + '\n')
                f.write(','.join([f"{v:.6f}" for v in features[i]]) + '\n')
        return len(filenames)

    # Save CSVs for ALL categories
    n1 = save_csvs(fs_healthy_emb, fs_healthy_names, "training/healthy", "healthy")
    n2 = save_csvs(fs_healthy_test_emb, fs_healthy_test_names, "testing/healthy", "healthy")
    n3 = save_csvs(fs_anomaly_emb, fs_anomaly_names, "testing/anomaly", "anomaly")
    n4 = save_csvs(syn_healthy_emb, syn_healthy_names, "synthetic/healthy", "healthy")
    n5 = save_csvs(syn_anomaly_emb, syn_anomaly_names, "synthetic/anomaly", "anomaly")

    print(f"  Saved CSV embeddings:")
    print(f"    training/healthy:  {n1} files")
    print(f"    testing/healthy:   {n2} files")
    print(f"    testing/anomaly:   {n3} files")
    print(f"    synthetic/healthy: {n4} files")
    print(f"    synthetic/anomaly: {n5} files")

    print(f"\n{'='*60}")
    print(f"  MODELS READY FOR EXPORT")
    print(f"{'='*60}")
    print(f"  Run: python export_models.py")
    print(f"  It will package yamnet.tflite + pca_model.npz + compute kmeans center")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
