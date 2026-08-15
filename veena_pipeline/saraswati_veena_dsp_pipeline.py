"""
Saraswati Veena Vibro-Acoustic DSP & Machine Learning Pipeline
===============================================================
Author: Advanced Audio DSP & ML Research Team
Reference Physics & Literature:
  - Asokan et al. (2016): Structural Resonator Modes & Fundamental Tuning Targets
  - Chauhan et al. (2021): Extended Kudirai Bridge Overtone Dynamics & Harmonic Revival

This module provides a modular, production-grade audio feature extraction and 
classification pipeline tailored specifically to the structural acoustics of the 
Saraswati Veena.
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import librosa

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore', category=UserWarning)


class VeenaAcousticDSPPipeline:
    """
    Physics-Anchored Audio Signal Processing Engine for Saraswati Veena Acoustics.
    
    Attributes:
        target_sr (int): Target sampling rate in Hz (default 16,000 Hz).
        f0_target (float): Standardized target fundamental frequency for S1 Sarani string (150.0 Hz).
        kudam_modes (Tuple[float, float]): Natural resonator resonance frequencies (280.0 Hz, 300.0 Hz).
        upper_cutoff_hz (float): Upper partial frequency threshold for overtone analysis (2800.0 Hz).
    """

    def __init__(
        self,
        target_sr: int = 16000,
        f0_target: float = 150.0,
        kudam_modes: Tuple[float, float] = (280.0, 300.0),
        upper_cutoff_hz: float = 2800.0
    ):
        self.target_sr = target_sr
        self.f0_target = f0_target
        self.kudam_modes = kudam_modes
        self.upper_cutoff_hz = upper_cutoff_hz

    # ─── 1. STRUCTURAL BASELINES & PYIN PITCH TRACKING (Asokan et al., 2016) ───

    def extract_pyin_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract dynamic F0 pitch trajectory using probabilistic YIN (pYIN) and calculate
        Cents deviation from target F0 (150.0 Hz) and Kudam structural penalties.
        
        Returns:
            Dict containing F0 median, Cents deviation, voiced probability, and penalty metrics.
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=audio,
            fmin=45,
            fmax=400,
            sr=self.target_sr,
            frame_length=2048,
            hop_length=512
        )

        valid_f0 = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else np.array([])
        mean_voiced_prob = float(np.mean(voiced_probs)) if voiced_probs is not None else 0.0

        if len(valid_f0) > 0:
            median_f0 = float(np.median(valid_f0))
            std_f0 = float(np.std(valid_f0))
            
            # Cents deviation relative to 150.0 Hz target
            # Cents = 1200 * log2(F0 / F0_target)
            cents_dev = float(1200.0 * np.log2(median_f0 / self.f0_target))
            
            # Hard structural penalty: Significant negative pitch drift (-10 to -50 Cents)
            # Indicates peg slippage / tension loss (Biridai fault)
            if -50.0 <= cents_dev <= -10.0:
                structural_detune_penalty = float(np.abs(cents_dev) / 50.0)
            elif cents_dev < -50.0:
                # Severe detuning or octave drop
                structural_detune_penalty = 1.0
            else:
                structural_detune_penalty = 0.0
        else:
            median_f0 = 0.0
            std_f0 = 0.0
            cents_dev = -1200.0
            structural_detune_penalty = 1.0

        return {
            'pyin_f0_median': median_f0,
            'pyin_f0_std': std_f0,
            'pyin_voiced_ratio': float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0,
            'pyin_voiced_prob_mean': mean_voiced_prob,
            'cents_deviation_150hz': cents_dev,
            'structural_detune_penalty': structural_detune_penalty
        }

    def extract_kudam_resonance_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract bandpass energy ratios around the natural Kudam resonator modes (280 Hz and 300 Hz).
        """
        n = len(audio)
        fft_mag = np.abs(np.fft.rfft(audio))
        fft_freqs = np.fft.rfftfreq(n, 1.0 / self.target_sr)
        total_energy = float(np.sum(fft_mag ** 2) + 1e-8)

        # 280 Hz band (265 Hz - 295 Hz)
        mode1_mask = (fft_freqs >= 265.0) & (fft_freqs <= 295.0)
        mode1_energy = float(np.sum(fft_mag[mode1_mask] ** 2)) / total_energy

        # 300 Hz band (295 Hz - 315 Hz)
        mode2_mask = (fft_freqs >= 295.0) & (fft_freqs <= 315.0)
        mode2_energy = float(np.sum(fft_mag[mode2_mask] ** 2)) / total_energy

        # Coupled Kudam resonator band (260 Hz - 320 Hz)
        kudam_combined_energy = mode1_energy + mode2_energy

        return {
            'kudam_280hz_ratio': mode1_energy,
            'kudam_300hz_ratio': mode2_energy,
            'kudam_combined_ratio': kudam_combined_energy
        }

    # ─── 2. TIME-VARYING OVERTONE ACOUSTICS (Chauhan et al., 2021) ───

    def compute_hnr(self, audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
        """
        Compute Harmonic-to-Noise Ratio (HNR) via frame autocorrelation to isolate
        non-linear chatter / buzzing artifacts on the Kudirai bridge.
        """
        num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
        hnr_values = []

        for i in range(num_frames):
            frame = audio[i * hop_length : i * hop_length + frame_length]
            if len(frame) < frame_length or np.std(frame) < 1e-5:
                continue

            r = signal.correlate(frame, frame, mode='full')
            r = r[len(r) // 2 :] # Positive lags

            min_lag = int(self.target_sr / 400.0) # ~40 samples
            max_lag = int(self.target_sr / 45.0)  # ~355 samples

            if len(r) > max_lag:
                r_max = np.max(r[min_lag:max_lag])
                r_zero = r[0]
                if r_zero > r_max and r_max > 0:
                    # HNR = 10 * log10(R_max / (R_0 - R_max))
                    hnr = 10.0 * np.log10(r_max / max(1e-6, r_zero - r_max))
                    hnr_values.append(hnr)

        return float(np.mean(hnr_values)) if hnr_values else 0.0

    def compute_time_series_spectral_flux(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute Time-Series Spectral Flux for upper partial overtones (up to 2800 Hz)
        to capture transient decay and overtone dynamics over time.
        """
        S = np.abs(librosa.stft(y=audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.target_sr, n_fft=2048)

        # Mask frequencies up to 2800 Hz
        cutoff_mask = freqs <= self.upper_cutoff_hz
        S_upper = S[cutoff_mask, :]

        # Normalize per frame to observe shape evolution
        S_norm = S_upper / (np.sum(S_upper, axis=0, keepdims=True) + 1e-8)

        # Spectral Flux: Positive magnitude difference between consecutive frames
        diff = np.diff(S_norm, axis=1)
        flux = np.sum(np.maximum(0, diff), axis=0)

        # Divide signal into Attack Transient (first 30% frames) and Decay Phase (remaining 70%)
        n_frames = len(flux)
        attack_idx = int(0.3 * n_frames)

        attack_flux_mean = float(np.mean(flux[:attack_idx])) if attack_idx > 0 else 0.0
        decay_flux_mean = float(np.mean(flux[attack_idx:])) if attack_idx < n_frames else 0.0
        flux_decay_ratio = attack_flux_mean / (decay_flux_mean + 1e-6)

        return {
            'spectral_flux_mean_2800hz': float(np.mean(flux)),
            'spectral_flux_std_2800hz': float(np.std(flux)),
            'spectral_flux_attack': attack_flux_mean,
            'spectral_flux_decay': decay_flux_mean,
            'spectral_flux_ratio': flux_decay_ratio
        }

    def compute_temporal_envelope_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract physical temporal waveform envelope features capturing string tension attack dynamics.
        
        Physics Rationale (User Observation & Vibro-Acoustic Coupling):
          - Healthy String (High Tension): Sharp 1-fraction-of-a-second attack spike followed by 
            smooth exponential dissipation (1/x curve).
          - Detuned String (Slack Tension): Slack transverse flexing causes prolonged plectrum/bridge 
            clinging (extended peak plateau) followed by non-linear bridge collision (abrupt energy crash).
        """
        hop_len = 128
        frame_len = 512
        rms_env = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        
        if len(rms_env) == 0 or np.max(rms_env) < 1e-4:
            return {
                'env_peak_duration_ms': 0.0,
                'env_crest_factor': 0.0,
                'env_attack_plateau_ratio': 0.0,
                'env_decay_abruptness': 0.0
            }

        peak_val = np.max(rms_env)
        rms_mean = np.mean(rms_env) + 1e-6
        crest_factor = float(peak_val / rms_mean)

        # 1. Prolonged Attack Peak Duration (ms above 75% peak)
        above_75_frames = np.sum(rms_env >= 0.75 * peak_val)
        peak_duration_ms = float((above_75_frames * hop_len / self.target_sr) * 1000.0)

        # 2. Attack Plateau Ratio: Energy in [50ms, 250ms] vs [0ms, 50ms]
        f_50ms = max(1, int(0.05 * self.target_sr / hop_len))
        f_250ms = max(f_50ms + 1, int(0.25 * self.target_sr / hop_len))
        
        e_early = float(np.sum(rms_env[:f_50ms])) + 1e-6
        e_plateau = float(np.sum(rms_env[f_50ms:f_250ms])) + 1e-6
        attack_plateau_ratio = float(e_plateau / e_early)

        # 3. Decay Knee Abruptness: Ratio of early vs late decay drop rate
        peak_idx = np.argmax(rms_env)
        decay_seg = rms_env[peak_idx:]
        
        if len(decay_seg) >= 20:
            d1 = decay_seg[:10]
            d2 = decay_seg[10:20]
            drop1 = (d1[0] - d1[-1]) / (d1[0] + 1e-6)
            drop2 = (d2[0] - d2[-1]) / (d2[0] + 1e-6)
            decay_abruptness = float(drop2 / (drop1 + 1e-6))
        else:
            decay_abruptness = 0.0

        return {
            'env_peak_duration_ms': peak_duration_ms,
            'env_crest_factor': crest_factor,
            'env_attack_plateau_ratio': attack_plateau_ratio,
            'env_decay_abruptness': decay_abruptness
        }

    def compute_overtone_revival_features(self, audio: np.ndarray, f0_val: float) -> Dict[str, float]:
        """
        Track non-linear energy revival of 2nd (2*F0) and 3rd (3*F0) harmonics over time.
        (Chauhan et al., 2021 proved extended bridge geometry induces temporal overtone revival).
        """
        if f0_val < 30.0:
            return {'h2_h1_energy_ratio': 0.0, 'h3_h1_energy_ratio': 0.0, 'overtone_revival_index': 0.0}

        S = np.abs(librosa.stft(y=audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.target_sr, n_fft=2048)

        def get_harmonic_energy_time_series(h_freq):
            mask = (freqs >= h_freq * 0.92) & (freqs <= h_freq * 1.08)
            return np.sum(S[mask, :], axis=0) if np.any(mask) else np.zeros((S.shape[1],))

        h1_ts = get_harmonic_energy_time_series(f0_val)
        h2_ts = get_harmonic_energy_time_series(2.0 * f0_val)
        h3_ts = get_harmonic_energy_time_series(3.0 * f0_val)

        h1_total = np.sum(h1_ts) + 1e-8
        h2_ratio = float(np.sum(h2_ts) / h1_total)
        h3_ratio = float(np.sum(h3_ts) / h1_total)

        # revival index: ratio of 2nd/3rd harmonic energy in late decay vs early attack
        n = len(h1_ts)
        early = int(0.2 * n)
        late = int(0.6 * n)

        early_overtone = np.mean(h2_ts[:early] + h3_ts[:early]) if early > 0 else 1e-6
        late_overtone = np.mean(h2_ts[late:]) if late < n else 0.0
        revival_index = float(late_overtone / (early_overtone + 1e-6))

        return {
            'h2_h1_energy_ratio': h2_ratio,
            'h3_h1_energy_ratio': h3_ratio,
            'overtone_revival_index': revival_index
        }

    # ─── 3. VISUAL DOMAIN MEL SPECTROGRAM MATRIX ───

    def extract_2d_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_mels: int = 128,
        fmax: float = 8000.0
    ) -> np.ndarray:
        """
        Generate 2D Log-Mel Spectrogram matrix mapping time vs log-frequency scale
        to capture microtonal pitch bends (Gamakas) and Swara note transitions.
        
        Returns:
            np.ndarray: 2D array of shape (n_mels, time_steps).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            fmin=30.0,
            fmax=fmax
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec.astype(np.float32)

    # ─── 4. MASTER FEATURE VECTOR EXTRACTION ───

    def extract_master_feature_vector(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive physics-anchored feature vector combining pYIN pitch,
        Kudam resonances, HNR, Spectral Flux, and Overtone revival dynamics.
        """
        features = {}

        # 1. pYIN Pitch & Penalty
        pyin_feats = self.extract_pyin_pitch_features(audio)
        features.update(pyin_feats)

        # 2. Kudam Resonances
        kudam_feats = self.extract_kudam_resonance_features(audio)
        features.update(kudam_feats)

        # 3. HNR
        features['hnr_db'] = self.compute_hnr(audio)

        # 4. Time-Series Spectral Flux (<= 2800 Hz)
        flux_feats = self.compute_time_series_spectral_flux(audio)
        features.update(flux_feats)

        # 5. Temporal Waveform Envelope Features (User Observation: Attack Spike vs Prolonged Plateau & Abrupt Crash)
        env_feats = self.compute_temporal_envelope_features(audio)
        features.update(env_feats)

        # 6. Overtone Revival Dynamics
        f0_val = pyin_feats['pyin_f0_median']
        revival_feats = self.compute_overtone_revival_features(audio, f0_val)
        features.update(revival_feats)

        # 6. Standard Complementary Spectral Features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
        for idx, val in enumerate(np.mean(mfcc, axis=1)):
            features[f'mfcc_mean_{idx+1}'] = float(val)

        cent = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
        features['spectral_centroid_mean'] = float(np.mean(cent))
        features['spectral_centroid_std'] = float(np.std(cent))

        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = float(np.mean(zcr))

        return features


class VeenaClassifier:
    """
    Enterprise-Grade Machine Learning Pipeline with Class-Balancing Weights
    and File-Level GroupKFold Cross-Validation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.class_names = ['Healthy', 'Detuned_Strings', 'Quality_Issues']

    def train_and_evaluate_group_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Union[float, Dict]]:
        """
        Run 5-Fold GroupKFold Cross Validation using balanced class weights.
        """
        gkf = GroupKFold(n_splits=n_splits)

        models = {
            "SVM (RBF Kernel Balanced)": SVC(kernel='rbf', C=15, class_weight='balanced', probability=True, random_state=self.random_state),
            "Gradient Boosting (Balanced)": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=self.random_state),
            "Random Forest (Balanced)": RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=self.random_state)
        }

        eval_summary = {}

        for m_name, model in models.items():
            y_true_all, y_pred_all = [], []
            acc_scores, f1_scores = [], []

            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_tr_norm = scaler.fit_transform(X_tr)
                X_te_norm = scaler.transform(X_te)

                if m_name == "Gradient Boosting (Balanced)":
                    weights = compute_sample_weight('balanced', y_tr)
                    model.fit(X_tr_norm, y_tr, sample_weight=weights)
                else:
                    model.fit(X_tr_norm, y_tr)

                y_pred = model.predict(X_te_norm)

                acc = accuracy_score(y_te, y_pred)
                f1_m = f1_score(y_te, y_pred, average='macro', zero_division=0)

                acc_scores.append(acc)
                f1_scores.append(f1_m)

                y_true_all.extend(y_te)
                y_pred_all.extend(y_pred)

            report = classification_report(y_true_all, y_pred_all, target_names=self.class_names, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_true_all, y_pred_all)

            eval_summary[m_name] = {
                'mean_accuracy': float(np.mean(acc_scores)),
                'std_accuracy': float(np.std(acc_scores)),
                'mean_f1_macro': float(np.mean(f1_scores)),
                'std_f1_macro': float(np.std(f1_scores)),
                'report': report,
                'confusion_matrix': cm.tolist()
            }

        return eval_summary


# ─── PIPELINE SELF-TEST & DEMONSTRATION ───

if __name__ == "__main__":
    print("=" * 70)
    print("  Saraswati Veena Vibro-Acoustic DSP Pipeline Module Self-Test")
    print("=" * 70)

    pipeline = VeenaAcousticDSPPipeline()

    # Generate synthetic 2.0s test waveform (150 Hz fundamental + harmonics)
    t = np.linspace(0, 2.0, int(16000 * 2.0))
    synthetic_audio = (
        0.5 * np.sin(2 * np.pi * 150.0 * t) +
        0.3 * np.sin(2 * np.pi * 300.0 * t) +
        0.2 * np.sin(2 * np.pi * 450.0 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)

    # 1. Feature Extraction Test
    feats = pipeline.extract_master_feature_vector(synthetic_audio)
    print(f"\n[SUCCESS] Extracted {len(feats)} physics-anchored DSP features:")
    for k in list(feats.keys())[:10]:
        print(f"  - {k}: {feats[k]:.4f}")

    # 2. 2D Mel Spectrogram Test
    log_mel = pipeline.extract_2d_mel_spectrogram(synthetic_audio)
    print(f"\n[SUCCESS] Generated 2D Log-Mel Spectrogram Matrix shape: {log_mel.shape} (128 Mels x {log_mel.shape[1]} Time frames)")

    print("\n" + "=" * 70)
    print("DSP PIPELINE MODULE READY FOR INTEGRATION!")
    print("=" * 70)
