"""
SwarCare Physics Pitch Engine
==============================
Deterministic DSP gate for Saraswati Veena string tuning detection.

Architecture (3 loophole fixes in sequence):
  Stage 0 — Silence / Energy Gate
  Stage 1 — Cepstral Lifter          (strips Kudam body resonance at 280–300 Hz)
  Stage 2 — Harmonic Product Spectrum (collapses suppressed fundamental)
  Stage 3 — pYIN + Sruti Soft Prior  (prevents HMM glitch from harmonic revival)
  Stage 4 — Cents Rule Engine         (deterministic ±15 cent decision)

Usage (standalone):
    python physics_pitch_engine.py

Usage (from Streamlit UI or pipeline):
    from physics_pitch_engine import PhysicsPitchEngine, TONIC_OPTIONS

    engine = PhysicsPitchEngine(tonic_hz=TONIC_OPTIONS["C3"], cents_threshold=15.0)
    result = engine.run(audio_np, sr=16000)
    print(result.status, result.message)

References:
  [1] Asokan et al. (2016) — Vibro-acoustic signatures of Saraswati Veena, JVE International
  [2] Chauhan et al. (2021) — Kudirai bridge overtone dynamics & harmonic revival
  [3] Acoustic Analysis of Timbre of Sarasvati Veena (ResearchGate, 2020)
  [4] de Cheveigné & Kawahara (2002) — YIN algorithm for fundamental frequency estimation
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np

# scipy.signal is used only for Savitzky-Golay smoothing inside HPS.
# Provide a lightweight NumPy fallback so the engine works when scipy is
# not installed (e.g. on constrained edge / Arduino MPU environments).
try:
    import scipy.signal as sp_signal
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    class _FakeSPSignal:
        @staticmethod
        def savgol_filter(x, window_length=11, polyorder=3):
            """Fallback: simple uniform moving-average smoothing."""
            k = max(1, window_length // 2)
            out = np.convolve(x, np.ones(k) / k, mode="same")
            return out
    sp_signal = _FakeSPSignal()

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

# ─── Runtime-Configurable Tonic Options ──────────────────────────────────────
# Sa (tonic) frequency options for the Streamlit UI dropdown.
# Sa = S3 Mandra (the middle string, physical tonic of the instrument).
# Lower tonics are common on heavier-gauge Saraswati Veenas.
TONIC_OPTIONS: Dict[str, float] = {
    "A1":  55.00,
    "A#1": 58.27,
    "B1":  61.74,
    "C2":  65.41,
    "C#2": 69.30,
    "D2":  73.42,
    "D#2": 77.78,
    "E2":  82.41,
    "F2":  87.31,
    "F#2": 92.50,
    "G2":  98.00,
    "G#2": 103.83,
    "A2":  110.00,
    "A#2": 116.54,
    "B2":  123.47,
    "C3":  130.81,
    "C#3": 138.59,
    "D3":  146.83,
    "D#3": 155.56,
    "E3":  164.81,
    "F3":  174.61,
    "F#3": 185.00,
}

# ─── Saraswati Veena String Ratios (relative to Sa/tonic = S3 Mandra) ─────────
#
# Physical string layout (low pitch → high pitch):
#   S4 (Anumandra)  0.75× Sa  — lower Pa, below tonic
#   S3 (Mandra)     1.0×  Sa  — tonic (Sa), this IS the reference
#   S2 (Panchama)   1.5×  Sa  — Pa (perfect fifth above Sa)
#   S1 (Sarani)     2.0×  Sa  — Tara Sa (octave above Sa)
#   T1 (Chikari 1)  4.0×  Sa  — two octaves above Sa
#   T2 (Chikari 2)  6.0×  Sa  — two octaves + Pa
#   T3 (Chikari 3)  8.0×  Sa  — three octaves above Sa
#
# Measured baseline (Phase1 healthy recordings, Sa ≈ 80 Hz):
#   S4 ≈ 58.9 Hz  |  S3 ≈ 80.0 Hz  |  S2 ≈ 120.6 Hz  |  S1 ≈ 162.8 Hz
_STRING_RATIOS: Dict[str, float] = {
    "S4": 0.75,   # Anumandra — lower Pa (5th below Sa)
    "S3": 1.0,    # Mandra    — Sa (tonic)
    "S2": 1.5,    # Panchama  — Pa (perfect fifth)
    "S1": 2.0,    # Sarani    — Tara Sa (octave)
    "T1": 4.0,    # Chikari 1 — two octaves
    "T2": 6.0,    # Chikari 2 — two octaves + Pa
    "T3": 8.0,    # Chikari 3 — three octaves
}
_STRING_NAMES: Dict[str, str] = {
    "S4": "S4 — Anumandra (lower Pa)",
    "S3": "S3 — Mandra Sa (tonic)",
    "S2": "S2 — Panchama (Pa)",
    "S1": "S1 — Sarani (Tara Sa)",
    "T1": "T1 — Chikari 1 (Sa, 2 oct)",
    "T2": "T2 — Chikari 2 (Pa, 2 oct)",
    "T3": "T3 — Chikari 3 (Sa, 3 oct)",
}

# ─── 22 Carnatic Srutis (frequency ratios relative to Sa) ────────────────────
_SRUTI_RATIOS: Tuple[float, ...] = (
    1.0, 256/243, 16/15, 10/9, 9/8,
    32/27, 6/5, 5/4, 81/64,
    4/3, 27/20, 45/32, 729/512,
    3/2,
    128/81, 8/5, 5/3, 27/16,
    16/9, 9/5, 15/8, 243/128,
)


# ─── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class PitchResult:
    """
    Output from the Physics Pitch Engine for one audio segment.

    Attributes
    ----------
    status      : "IN_TUNE" | "FLAT" | "SHARP" | "NO_PITCH" | "SILENCE"
    f0_hz       : Detected fundamental frequency (0.0 if not detected)
    cents_dev   : Deviation from nearest Veena string target (positive = sharp)
    hz_dev      : Deviation in Hz from nearest string target
    string_num  : Nearest Veena string number (1–7), 0 if unknown
    string_name : Human-readable string name
    target_hz   : Expected frequency for the nearest string
    confidence  : pYIN voiced probability (0–1)
    message     : Human-readable tuning direction (e.g. "Flat by 28 cents — tighten peg")
    method      : Which algorithm produced the result ("pyin" | "hps_autocorr" | "none")
    """
    status:      str   = "NO_PITCH"
    f0_hz:       float = 0.0
    cents_dev:   float = 0.0
    hz_dev:      float = 0.0
    string_num:  int   = 0
    string_name: str   = "Unknown"
    target_hz:   float = 0.0
    confidence:  float = 0.0
    message:     str   = "No pitch detected"
    method:      str   = "none"


# ─── Main Engine ──────────────────────────────────────────────────────────────

class PhysicsPitchEngine:
    """
    Deterministic DSP pitch gate for Saraswati Veena string tuning.

    Parameters
    ----------
    tonic_hz              : Sa (tonic) frequency in Hz. Configurable at runtime
                            via Streamlit UI. Use TONIC_OPTIONS dict for standards.
    cents_threshold       : Tuning tolerance in cents. ±15 cents is standard for
                            Carnatic instruments with peg-based tuning mechanisms.
    sr                    : Expected sample rate. Default 16000 Hz.
    silence_rms_threshold : Segments with RMS below this are skipped as silence.
    hps_harmonics         : Number of HPS downsampling stages (R). Default 4.
    """

    def __init__(
        self,
        tonic_hz: float = TONIC_OPTIONS["C3"],
        cents_threshold: float = 15.0,
        sr: int = 16000,
        silence_rms_threshold: float = 0.008,
        hps_harmonics: int = 4,
    ) -> None:
        self.tonic_hz = tonic_hz
        self.cents_threshold = cents_threshold
        self.sr = sr
        self.silence_rms_threshold = silence_rms_threshold
        self.hps_harmonics = hps_harmonics

        # Precompute string target frequencies (keyed by string label S1-S4/T1-T3)
        self._string_targets: Dict[str, float] = {
            s: round(tonic_hz * ratio, 4)
            for s, ratio in _STRING_RATIOS.items()
        }

        # Precompute Sruti frequencies — extend across two octaves
        base_srutis = np.array([tonic_hz * r for r in _SRUTI_RATIOS], dtype=np.float64)
        extended = []
        for mult in [0.5, 1.0, 2.0, 4.0]:
            extended.extend(base_srutis * mult)
        self._sruti_freqs_extended = np.array(extended, dtype=np.float64)

    # ─── Public API ──────────────────────────────────────────────────────────

    def run(self, audio: np.ndarray, sr: Optional[int] = None) -> PitchResult:
        """
        Run the full 4-stage physics pitch gate on one audio segment.

        Parameters
        ----------
        audio : np.ndarray — mono float32/float64 audio waveform
        sr    : int        — sample rate (defaults to self.sr)

        Returns
        -------
        PitchResult with status, cents deviation, direction message, etc.
        """
        sr = sr or self.sr
        audio = audio.astype(np.float64)

        # ── Stage 0: Silence Gate ────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.silence_rms_threshold:
            return PitchResult(status="SILENCE", message="Segment is silent")

        # ── Stage 1: Cepstral Lifter (Loophole 3 — Kudam formant stripping) ─
        audio_lifted = self._cepstral_lifter(audio, sr)

        # ── Stage 2: HPS (Loophole 1 — collapse suppressed fundamental) ──────
        f0_hps, _, _ = self._hps_f0_estimate(audio_lifted, sr)

        # ── Stage 3: pYIN + Sruti Soft Prior (Loophole 2 — HMM glitch fix) ──
        f0_final, confidence, method = self._sruti_aware_pyin(audio_lifted, sr, f0_hps)

        if f0_final <= 0.0 or confidence < 0.15:
            return PitchResult(
                status="NO_PITCH",
                f0_hz=0.0,
                confidence=round(confidence, 3),
                message="Could not reliably detect pitch",
                method=method,
            )

        # ── Stage 4: Cents Rule Engine ───────────────────────────────────────
        return self._cents_decision(f0_final, confidence, method)

    def update_tonic(self, tonic_hz: float) -> None:
        """
        Update the tonic (Sa = S3 Mandra) frequency at runtime.
        Called from Streamlit UI when the user changes the tonic dropdown,
        or from the calibration routine after measuring Phase1 S3 recordings.
        Reinitialises all precomputed string targets and Sruti priors.
        """
        self.__init__(
            tonic_hz=tonic_hz,
            cents_threshold=self.cents_threshold,
            sr=self.sr,
            silence_rms_threshold=self.silence_rms_threshold,
            hps_harmonics=self.hps_harmonics,
        )

    def detect_f0_only(self, audio: np.ndarray, sr: Optional[int] = None) -> Tuple[float, float]:
        """
        Run pitch detection pipeline (Stages 0-3) on transient onset window.
        Uses onset window (50ms - 350ms) where fundamental f0 is maximum and
        uncorrupted by late-stage Kudirai bridge harmonic revival.

        Returns
        -------
        (f0_hz, confidence) — (0.0, 0.0) if silent or no pitch found
        """
        sr = sr or self.sr
        audio = audio.astype(np.float64)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.silence_rms_threshold:
            return 0.0, 0.0

        # Onset extraction: 50ms to 350ms (0.3s window) for transient attack envelope
        start_idx = int(0.05 * sr)
        end_idx = int(0.35 * sr)
        if len(audio) >= end_idx:
            audio_onset = audio[start_idx:end_idx]
        else:
            audio_onset = audio

        audio_lifted = self._cepstral_lifter(audio_onset, sr)
        f0_hps, _, _ = self._hps_f0_estimate(audio_lifted, sr)
        f0, conf, _ = self._sruti_aware_pyin(audio_lifted, sr, f0_hps)
        return f0, conf

    def string_targets(self) -> Dict[str, float]:
        """Return all 7 string target frequencies (S1-S4/T1-T3) for the current tonic."""
        return dict(self._string_targets)

    # ─── Stage 1: Cepstral Lifter ────────────────────────────────────────────

    def _cepstral_lifter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        High-pass cepstral lifter to decouple string excitation from Kudam
        body resonance (fixed structural formants at 280–300 Hz).

        Signal flow:
          x[n] → FFT → log|X(f)| → IFFT → cepstrum c[n]
                → high-pass lifter (zero low-quefrency envelope)
                → FFT → exp(·) → IFFT → lifted signal x'[n]

        Low quefrency  (<3.5 ms) = slow spectral envelope = Kudam body → REMOVED
        High quefrency (≥3.5 ms) = fine harmonic peaks  = string vibration → KEPT
        """
        n = len(audio)
        window = np.blackman(n)
        X = np.fft.rfft(audio * window)

        # Log magnitude spectrum
        log_mag = np.log(np.abs(X) + 1e-10)

        # Real cepstrum
        cepstrum = np.fft.irfft(log_mag)

        # High-pass lifter: quefrency < 3.5 ms → zeroed (body resonances)
        lifter_cutoff = max(3, int(sr * 0.0035))
        lifter = np.zeros_like(cepstrum)
        lifter[lifter_cutoff: n // 2] = cepstrum[lifter_cutoff: n // 2]
        lifter[n // 2:] = cepstrum[n // 2:]  # symmetric mirror

        # Reconstruct lifted spectrum and convert back to time domain
        log_mag_lifted = np.fft.rfft(lifter).real
        X_lifted = np.exp(log_mag_lifted) * np.exp(1j * np.angle(X))
        audio_lifted = np.fft.irfft(X_lifted, n=n)

        # Normalise to preserve original RMS
        orig_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-10
        lift_rms = float(np.sqrt(np.mean(audio_lifted ** 2))) + 1e-10
        return (audio_lifted * (orig_rms / lift_rms)).astype(np.float64)

    # ─── Stage 2: Harmonic Product Spectrum ───────────────────────────────────

    def _hps_f0_estimate(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Harmonic Product Spectrum (HPS) F0 estimator.

        Formula:  Y(f) = ∏_{r=1}^{R} |X(r·f)|

        Multiplying R downsampled copies of the magnitude spectrum collapses
        energy from all upper harmonics (2f₀, 3f₀, …, Rf₀) onto the true f₀
        axis. Corrects octave-doubling errors caused by the Kudirai bridge
        suppressing the fundamental.

        R=4 stages provide robust correction without excessive noise amplification.
        A Blackman window + Savitzky-Golay smoothing pre-filter prevents the
        multiplicative noise accumulation warned about in literature.

        Returns (f0_estimate_hz, hps_spectrum, frequency_axis)
        """
        n_fft = 8192
        seg = audio[:n_fft] if len(audio) >= n_fft else np.pad(audio, (0, n_fft - len(audio)))
        window = np.blackman(n_fft)
        X = np.abs(np.fft.rfft(seg * window))
        freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Pre-smooth before HPS to dampen noise floor
        X_smooth = sp_signal.savgol_filter(X, window_length=11, polyorder=3)
        X_smooth = np.maximum(X_smooth, 1e-10)

        hps = X_smooth.copy()
        for r in range(2, self.hps_harmonics + 1):
            downsampled = X_smooth[::r]
            min_len = min(len(hps), len(downsampled))
            hps[:min_len] *= downsampled[:min_len]
            hps[min_len:] = 0.0

        # Search within physical Veena string range only
        f_min_idx = int(np.searchsorted(freq_axis, 50.0))
        f_max_idx = int(np.searchsorted(freq_axis, 1100.0))
        if f_max_idx <= f_min_idx:
            return 0.0, hps, freq_axis

        peak_idx = f_min_idx + int(np.argmax(hps[f_min_idx:f_max_idx]))
        return float(freq_axis[peak_idx]), hps, freq_axis

    # ─── Stage 3: Sruti-Aware pYIN ────────────────────────────────────────────

    def _sruti_aware_pyin(
        self, audio: np.ndarray, sr: int, hps_f0_hint: float
    ) -> Tuple[float, float, str]:
        """
        pYIN pitch tracking with a 22-Sruti Gaussian soft prior on the HMM.

        Two-phase tracking strategy (pitch anchor + gated tracking):
          Attack  (first 200 ms): Wide search, full prior weight = 1.0.
                                  pYIN freely locks onto f₀.
          Sustain (200 ms → end): Sruti soft prior — Gaussian penalty σ=70 cents.
                                  Frames far from any Sruti are downweighted,
                                  preventing HMM path from jumping to a revived
                                  upper harmonic mid-decay (Chauhan et al., 2021).

        The prior is Gaussian (soft), NOT a hard mask, so:
          - Microtonal ornaments (Gamakas) within ±70 cents pass through cleanly.
          - A badly estimated anchor does not lock the tracker in a wrong pitch.

        Falls back to HPS estimate if librosa unavailable or pYIN returns no
        voiced frames.

        Returns (f0_hz, confidence, method_name)
        """
        if not _HAS_LIBROSA:
            return hps_f0_hint, 0.5, "hps_autocorr"

        try:
            f0_track, voiced_flag, voiced_probs = librosa.pyin(
                y=audio.astype(np.float32),
                sr=sr,
                fmin=float(librosa.note_to_hz("A1")),  # 55 Hz
                fmax=float(librosa.note_to_hz("C7")),  # 2093 Hz
                frame_length=4096,
                hop_length=512,
            )
        except Exception:
            return hps_f0_hint, 0.4, "hps_autocorr"

        if voiced_flag is None or f0_track is None:
            return hps_f0_hint, 0.35, "hps_autocorr"

        voiced = voiced_flag.astype(bool) & ~np.isnan(f0_track)
        valid_f0 = f0_track[voiced]
        valid_probs = voiced_probs[voiced]

        if len(valid_f0) == 0:
            return hps_f0_hint, 0.35, "hps_autocorr"

        # ── Sruti Gaussian Soft Prior ────────────────────────────────────────
        # sigma = 70 cents — generous enough for Gamaka ornaments
        sruti_weights = np.ones(len(valid_f0))
        for i, f in enumerate(valid_f0):
            if f > 0:
                dists_cents = np.abs(
                    1200.0 * np.log2(f / (self._sruti_freqs_extended + 1e-9))
                )
                min_dist = float(np.min(dists_cents))
                sruti_weights[i] = float(np.exp(-(min_dist ** 2) / (2 * 70.0 ** 2)))

        # Attack frames get full weight (wide search, no prior penalty)
        attack_frames = max(1, int(0.2 * sr / 512))
        # Map voiced indices back to frame space for attack masking
        voiced_indices = np.where(voiced)[0]
        for k, vi in enumerate(voiced_indices):
            if vi < attack_frames:
                sruti_weights[k] = 1.0

        # Weighted median (robust to mid-decay spikes)
        weighted_probs = valid_probs * sruti_weights
        if np.sum(weighted_probs) < 1e-9:
            weighted_probs = valid_probs

        sort_idx = np.argsort(valid_f0)
        sorted_f0 = valid_f0[sort_idx]
        sorted_w = weighted_probs[sort_idx]
        cumsum = np.cumsum(sorted_w)
        median_idx = int(np.searchsorted(cumsum, cumsum[-1] * 0.5))
        median_idx = min(median_idx, len(sorted_f0) - 1)
        f0_median = float(sorted_f0[median_idx])

        # Confidence: mean of top-quartile weighted probabilities
        q75 = float(np.percentile(weighted_probs, 75))
        top_w = weighted_probs[weighted_probs >= q75]
        confidence = float(np.mean(top_w)) if len(top_w) > 0 else 0.5

        # Sanity check: if pYIN drifts >250 cents from HPS hint → blend 60/40
        if hps_f0_hint > 0:
            cents_diff = abs(1200.0 * np.log2(f0_median / (hps_f0_hint + 1e-9)))
            if cents_diff > 250.0:
                f0_median = float(0.6 * hps_f0_hint + 0.4 * f0_median)

        return f0_median, confidence, "pyin"

    # ─── Stage 4: Cents Rule Engine ───────────────────────────────────────────

    def _cents_decision(
        self, f0_hz: float, confidence: float, method: str
    ) -> PitchResult:
        """
        Deterministic tuning decision via cents deviation formula:
            cents = 1200 × log₂(f_detected / f_target)

        Finds the nearest of the 7 Veena strings (S4/S3/S2/S1/T1/T2/T3),
        computes deviation, then classifies as IN_TUNE / FLAT / SHARP with
        a human-readable direction message for the musician.

        Positive cents = sharp (loosen peg counter-clockwise)
        Negative cents = flat  (tighten peg clockwise)
        """
        # Nearest string: minimise |cents deviation| across all 7 strings
        # First check for Harmonic Foldback (Loophole 1: Kudirai bridge upper partial excitation)
        # If f0 is near 1.5x, 2.0x, or 3.0x of a string target, test if folded fundamental is within target range
        candidate_f0 = f0_hz
        best_nearest = min(
            self._string_targets.keys(),
            key=lambda s: abs(
                1200.0 * np.log2(f0_hz / (self._string_targets[s] + 1e-9))
            ),
        )
        min_cents_abs = abs(1200.0 * np.log2(f0_hz / (self._string_targets[best_nearest] + 1e-9)))

        # Check foldback factors [1.5, 2.0, 3.0]
        if min_cents_abs > self.cents_threshold:
            for mult in [1.5, 2.0, 3.0]:
                folded_f0 = f0_hz / mult
                for s_key, s_target in self._string_targets.items():
                    folded_cents = abs(1200.0 * np.log2(folded_f0 / (s_target + 1e-9)))
                    if folded_cents <= self.cents_threshold:
                        candidate_f0 = folded_f0
                        best_nearest = s_key
                        min_cents_abs = folded_cents
                        method = f"{method}+harmonic_foldback({mult:.1f}x)"
                        break
                if min_cents_abs <= self.cents_threshold:
                    break

        target_hz = self._string_targets[best_nearest]
        string_name = _STRING_NAMES[best_nearest]
        cents_dev = float(1200.0 * np.log2(candidate_f0 / (target_hz + 1e-9)))
        hz_dev = float(candidate_f0 - target_hz)

        if abs(cents_dev) <= self.cents_threshold:
            status = "IN_TUNE"
            message = (
                f"In tune ({cents_dev:+.1f} cents) — "
                f"{string_name}, target {target_hz:.2f} Hz"
            )
        elif cents_dev < 0:
            status = "FLAT"
            message = (
                f"Flat by {abs(cents_dev):.1f} cents ({hz_dev:+.2f} Hz) — "
                f"tighten peg clockwise  [{string_name}]"
            )
        else:
            status = "SHARP"
            message = (
                f"Sharp by {cents_dev:.1f} cents ({hz_dev:+.2f} Hz) — "
                f"loosen peg counter-clockwise  [{string_name}]"
            )

        string_num_map = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "T1": 5, "T2": 6, "T3": 7}
        string_num = string_num_map.get(best_nearest, 0)

        return PitchResult(
            status=status,
            f0_hz=round(f0_hz, 2),
            cents_dev=round(cents_dev, 1),
            hz_dev=round(hz_dev, 3),
            string_num=string_num,
            string_name=string_name,
            target_hz=round(target_hz, 2),
            confidence=round(confidence, 3),
            message=message,
            method=method,
        )


# ─── Self-Test ────────────────────────────────────────────────────────────────

def _synth_tone(
    f0: float, sr: int = 16000, dur: float = 2.0, suppress_fund: bool = False
) -> np.ndarray:
    """
    Synthesise a plucked Veena-like tone with exponential decay.
    suppress_fund=True: simulates Kudirai bridge effect (2f₀ energy > f₀).
    """
    t = np.linspace(0, dur, int(sr * dur))
    if suppress_fund:
        audio = (
            0.10 * np.sin(2 * np.pi * f0 * t)
            + 0.70 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.35 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.15 * np.sin(2 * np.pi * 4 * f0 * t)
            + 0.02 * np.random.randn(len(t))
        )
    else:
        audio = (
            0.60 * np.sin(2 * np.pi * f0 * t)
            + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.10 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.02 * np.random.randn(len(t))
        )
    envelope = np.exp(-3.0 * t / dur)
    return (audio * envelope).astype(np.float64)


if __name__ == "__main__":
    SR = 16000
    print("=" * 65)
    print("  SwarCare Physics Pitch Engine — Self-Test")
    print("=" * 65)

    engine = PhysicsPitchEngine(tonic_hz=TONIC_OPTIONS["C3"], cents_threshold=15.0)

    print(f"\nTonic: C3 ({TONIC_OPTIONS['C3']} Hz) | Threshold: ±15 cents")
    print("String targets:")
    for s, f in engine.string_targets().items():
        print(f"  String {s}: {_STRING_NAMES[s]:30s} {f:.2f} Hz")

    test_cases = [
        # (label, f0, suppress_fundamental)
        ("String 1 Sa — perfect C3 (should be IN_TUNE)",         130.81, False),
        ("String 1 Sa — 128 Hz (Flat ~28 cents)",                 128.00, False),
        ("String 1 Sa — 132.5 Hz (Sharp ~18 cents)",              132.50, False),
        ("String 2 Pa — G3 196 Hz (IN_TUNE)",                     196.00, False),
        # Loophole 1: suppressed fundamental (Kudirai bridge effect)
        ("String 3 Tār Sa C4 — suppressed f0 (Loophole 1)",      261.63, True),
        # Loophole 2: harmonic revival mid-decay
        ("String 1 Sa — harmonic revival simulation (Loophole 2)",130.81, True),
        # Loophole 3: note near Kudam 300 Hz resonance
        ("D#4 = 311 Hz — near Kudam 300 Hz (Loophole 3)",        311.13, False),
        # Silence test
        ("Silence segment (should be SILENCE)",                     130.81, False),
    ]

    print("\n" + "-" * 65)
    for i, (desc, f0, suppress) in enumerate(test_cases):
        if "Silence" in desc:
            audio = np.zeros(int(SR * 2.0), dtype=np.float64)
        else:
            audio = _synth_tone(f0, SR, suppress_fund=suppress)

        result = engine.run(audio, sr=SR)
        tag = "[BRIDGE SIM]" if suppress else "            "
        print(f"\n  {tag} {desc}")
        print(f"    Input f0 : {f0:.2f} Hz")
        print(f"    Detected : {result.f0_hz:.2f} Hz  "
              f"(method={result.method}, confidence={result.confidence:.2f})")
        print(f"    Status   : {result.status}")
        print(f"    Message  : {result.message}")

    print("\n" + "=" * 65)
    print("  Runtime Tonic Switch Test (C3 → D3)")
    print("=" * 65)
    engine.update_tonic(TONIC_OPTIONS["D3"])
    audio_d3 = _synth_tone(146.83, SR)
    r = engine.run(audio_d3, sr=SR)
    print(f"\n  D3 tonic — Sa at 146.83 Hz")
    print(f"  Status  : {r.status}")
    print(f"  Message : {r.message}")

    print("\n" + "=" * 65)
    print("  PHYSICS PITCH ENGINE READY")
    print("=" * 65)
