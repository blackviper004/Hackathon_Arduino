"""
SwarCare Pipeline Validation - Step 2: Segment Audio
=====================================================
PURPOSE:
    Take the downloaded Veena WAV files and slice them into
    2-second windows suitable for Edge Impulse training.

WHY 2 SECONDS?
    - Edge Impulse anomaly detection works best with fixed-length windows
    - 2 seconds captures a full pluck + decay cycle of a Veena string
    - Short enough for real-time inference on Arduino UNO Q
    - Long enough to contain meaningful spectral information

WHAT IT DOES:
    1. Loads each WAV file from raw_audio/
    2. Resamples to 16kHz mono (Edge Impulse standard for audio)
    3. Slices into non-overlapping 2-second windows
    4. Filters out silence (segments below energy threshold)
    5. Saves as individual WAV files in segments/

OUTPUT:
    segments/
    ├── veena_murthy_001.wav  (2s, 16kHz, mono)
    ├── veena_murthy_002.wav
    ├── ...
    ├── veena_classical_001.wav
    ├── ...
    └── veena_pickup_001.wav  (contact pickup recording)
"""

import os
import sys
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install librosa soundfile numpy")
    sys.exit(1)

# === CONFIGURATION ===
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_audio")
SEG_DIR = os.path.join(os.path.dirname(__file__), "segments")
TARGET_SR = 16000       # 16kHz — Edge Impulse standard for audio models
WINDOW_SEC = 2.0        # 2-second windows
SILENCE_THRESHOLD = 0.01  # RMS below this = silence (skip)

# Map filenames to short prefixes for output naming
# Handles both manual download names AND Freesound API download names
FILE_PREFIX_MAP = {
    # Manual download names (from 01 instructions)
    "Veena-Murthy-19-4-2011.wav": "veena_murthy",
    "veena_classical.wav": "veena_classical",
    "Veena_Recording.wav": "veena_pickup",
    # Freesound API download names (ID__username__title format)
    "125655__xserra__veena-murthy-19-4-2011.wav": "veena_murthy",
    "380750__anjds__veena_classical.wav": "veena_classical",
    "216060__iskweldog__veena_recording.wav": "veena_pickup",
}


def compute_rms(audio_segment: np.ndarray) -> float:
    """Compute Root Mean Square energy of a segment."""
    return float(np.sqrt(np.mean(audio_segment ** 2)))


def segment_file(filepath: str, prefix: str) -> int:
    """
    Load a WAV file, resample to 16kHz mono, and slice into 2s windows.
    Returns count of valid (non-silent) segments saved.
    """
    filename = os.path.basename(filepath)
    print(f"\n[PROCESSING] {filename}")

    # Load audio (librosa auto-converts to mono, resamples)
    audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s | Sample Rate: {sr}Hz | Samples: {len(audio)}")

    # Calculate window parameters
    window_samples = int(WINDOW_SEC * sr)
    total_windows = len(audio) // window_samples
    print(f"  Window size: {window_samples} samples ({WINDOW_SEC}s)")
    print(f"  Total possible windows: {total_windows}")

    saved_count = 0
    skipped_silence = 0

    for i in range(total_windows):
        start = i * window_samples
        end = start + window_samples
        segment = audio[start:end]

        # Skip silent segments
        rms = compute_rms(segment)
        if rms < SILENCE_THRESHOLD:
            skipped_silence += 1
            continue

        # Save segment
        saved_count += 1
        out_filename = f"{prefix}_{saved_count:03d}.wav"
        out_path = os.path.join(SEG_DIR, out_filename)
        sf.write(out_path, segment, sr)

    print(f"  Saved: {saved_count} segments | Skipped (silence): {skipped_silence}")
    return saved_count


def main():
    print("SwarCare Pipeline Validation - Audio Segmentation")
    print("=" * 55)

    # Check raw_audio directory
    if not os.path.exists(RAW_DIR):
        print(f"\n[ERROR] raw_audio/ directory not found: {RAW_DIR}")
        print("Run 01_download_audio.py first.")
        sys.exit(1)

    # Find WAV files
    wav_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".wav")]
    if not wav_files:
        print(f"\n[ERROR] No WAV files found in: {RAW_DIR}")
        print("Download the Veena audio files first (see 01_download_audio.py)")
        sys.exit(1)

    print(f"\nFound {len(wav_files)} WAV file(s) in raw_audio/")
    print(f"Target: {TARGET_SR}Hz mono, {WINDOW_SEC}s windows")

    # Create output directory
    os.makedirs(SEG_DIR, exist_ok=True)

    # Process each file
    total_segments = 0
    for wav_file in wav_files:
        filepath = os.path.join(RAW_DIR, wav_file)

        # Determine prefix
        prefix = FILE_PREFIX_MAP.get(wav_file, wav_file.replace(".wav", "").lower())

        count = segment_file(filepath, prefix)
        total_segments += count

    # Summary
    print("\n" + "=" * 55)
    print(f"TOTAL SEGMENTS CREATED: {total_segments}")
    print(f"Output directory: {SEG_DIR}")
    print(f"\nEach segment: {WINDOW_SEC}s, {TARGET_SR}Hz, mono WAV")
    print(f"\nNext step: python 03_prepare_training_data.py")


if __name__ == "__main__":
    main()
