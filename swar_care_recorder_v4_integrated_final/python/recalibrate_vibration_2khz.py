"""recalibrate_vibration_2khz.py — regenerate the vibration baseline at 2 kHz.

The piezo hardware is confirmed at 2 kHz (engine.PIEZO_SAMPLE_RATE_HZ = 2000).
The previously deployed kmeans_vibration_center/std.npy were synthesized at a
higher rate, which made every 2 kHz recording score as a spurious ANOMALY
(rate-dependent ZCR alone inflated distances from ~3.5 to ~8+).

This script rebuilds those files AT 2000 Hz using model.py's own
calibrate_vibration_baseline_multi(), with a synthetic "known-healthy" piezo
distribution (guitar-like plucks across the low strings, pick transients and a
soft noise floor) spread wide enough that every feature has a measured std.

Run it from this folder after editing model.py:

    .venv/bin/python recalibrate_vibration_2khz.py

For best production results, replace SYNTHETIC health clips with ~10-15 real
2 kHz recordings of your hardware playing normally (e.g. load them from the
saved *_piezo.csv files) and pass those waveforms in instead.
"""

import os
import sys

import numpy as np

# Run as a script even when invoked from a different CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import calibrate_vibration_baseline_multi, extract_dsp_features  # noqa: E402

VIBRATION_SAMPLE_RATE_HZ = 2000  # must match engine.PIEZO_SAMPLE_RATE_HZ

# --- Synthetic "healthy playing" model of a 2 kHz piezo tap ---
STRINGS_HZ = [82.41, 110.0, 146.83, 164.81, 196.0, 220.0, 246.94, 329.63]
# Detune ratios (0.0 = exactly fundamental) -> relative harmonic gain.
HARMONICS = {0.0: 1.0, 0.85: 0.45, 0.95: 0.22, 0.75: 0.15, 0.9: 0.08}


def synthetic_healthy_clip(duration_s: float = 2.0, sr: int = VIBRATION_SAMPLE_RATE_HZ) -> np.ndarray:
    rng = np.random
    n = int(duration_s * sr)
    t = np.linspace(0.0, duration_s, n, endpoint=False)
    wav = np.zeros(n)

    n_notes = int(rng.choice([2, 3, 2, 3, 4]))
    notes = rng.choice(STRINGS_HZ, size=n_notes, replace=False)
    amp = float(rng.uniform(350, 900))

    for f in notes:
        env = np.exp(-t / float(rng.uniform(0.4, 1.0)))
        for detune, gain in HARMONICS.items():
            partial_hz = f * (1.0 + detune + rng.uniform(-0.004, 0.004))
            wav += np.sin(2 * np.pi * partial_hz * t) * gain * amp * env

    # Pick-attack transients: short impulsive bursts on each rep.
    n_bursts = int(duration_s * 2) + int(rng.choice([0, 1]))
    for b in range(n_bursts):
        i0 = int(b * 1000 + rng.uniform(0, 400))
        for j in range(25):
            ix = min(i0 + j, n - 1)
            wav[ix] += rng.uniform(-amp * 1.2, amp * 1.2)

    wav += rng.normal(0, rng.uniform(10, 30), n)
    # Keep the NATIVE ADC amplitude scale (raw piezo readings ~ hundreds). The
    # RMS feature is amplitude-sensitive, so calibration clips must be at the
    # same absolute amplitude the real 2 kHz hardware produces or distances
    # blow up. Do NOT normalize to unit peak.
    return wav - wav.mean()


def main() -> None:
    clips = [synthetic_healthy_clip() for _ in range(14)]
    center, std = calibrate_vibration_baseline_multi(clips, sr=VIBRATION_SAMPLE_RATE_HZ)

    names = ["rms", "zcr_sec", "centroid_hz", "high_freq_ratio", "peak_freq_hz", "fundamental_hz"]
    print("\n=== Rebuilt 2 kHz vibration baseline ===")
    for name, c, s in zip(names, center, std):
        print(f"  {name:16s} center={float(c):10.3f} std={float(s):10.3f}")

    # --- Smoke-test discrimination with the freshly deployed model ---
    from model import run_vibration_pipeline
    print("\n=== Smoke test (distance; threshold 4.5552) ===")
    for label, wav in [
        ("healthy in-dist", synthetic_healthy_clip()),
        ("healthy in-dist 2", synthetic_healthy_clip()),
        ("loud 440 Hz drone", _drone(440.0, 1200.0)),
        ("hard percussive hit", _impulse()),
        ("idle noise floor", _idle()),
    ]:
        res = run_vibration_pipeline(wav.astype(np.float32), sr=VIBRATION_SAMPLE_RATE_HZ)
        if res is None:
            print(f"  {label:24s} <no vibration model>")
            continue
        d = res["score"]
        verdict = "ANOMALY" if d > res["threshold"] else ("watch" if d > res["threshold"] * 0.5 else "ok")
        print(f"  {label:24s} distance={d:10.2f}  -> {verdict}")


def _drone(hz: float, amp: float) -> np.ndarray:
    t = np.linspace(0, 2.0, VIBRATION_SAMPLE_RATE_HZ * 2, endpoint=False)
    return np.sin(2 * np.pi * hz * t) * amp


def _impulse() -> np.ndarray:
    n = VIBRATION_SAMPLE_RATE_HZ * 2
    wav = np.zeros(n)
    i0 = int(np.random.uniform(300, 700))
    for j in range(120):
        ix = min(i0 + j, n - 1)
        wav[ix] = float(np.random.uniform(-2200, 2200))
    return wav


def _idle() -> np.ndarray:
    n = VIBRATION_SAMPLE_RATE_HZ * 2
    t = np.linspace(0, 2.0, n, endpoint=False)
    return np.random.normal(0, 6, n) + np.sin(2 * np.pi * 50 * t) * 20


if __name__ == "__main__":
    main()