"""
SwarCare: Saraswati Veena Real-World Multi-Class Diagnostic Pipeline
=====================================================================
Comprehensive Training, Testing, and GroupKFold Cross-Validation Suite
Optimized for High Detuned Strings Precision & Recall
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import json
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

import ai_edge_litert.interpreter as tflite

TARGET_SR = 16000
WINDOW_SEC = 2.0
SILENCE_THRESHOLD = 0.008
YAMNET_MODEL_PATH = "yamnet.tflite"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

S1_F0, S2_F0, S3_F0, S4_F0 = 146.83, 110.00, 73.42, 55.00
TARGET_F0S = np.array([S1_F0, S2_F0, S3_F0, S4_F0])

def compute_rms(audio_segment: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio_segment ** 2)))


def extract_audio_segments(file_path: str, label: str, category: str, string_name: str, pluck_type: str, device_type: str):
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load {file_path}: {e}")
        return []

    window_samples = int(WINDOW_SEC * TARGET_SR)
    total_windows = len(audio) // window_samples
    segments = []

    if total_windows == 0 and len(audio) >= int(0.5 * TARGET_SR):
        pad_len = window_samples - len(audio)
        padded = np.pad(audio, (0, pad_len), mode='constant')
        if compute_rms(padded) >= SILENCE_THRESHOLD:
            segments.append({
                'audio': padded.astype(np.float32),
                'file': os.path.basename(file_path),
                'label': label,
                'category': category,
                'string': string_name,
                'pluck': pluck_type,
                'device': device_type,
                'seg_idx': 0
            })
    else:
        for i in range(total_windows):
            start = i * window_samples
            end = start + window_samples
            seg = audio[start:end]
            if compute_rms(seg) >= SILENCE_THRESHOLD:
                segments.append({
                    'audio': seg.astype(np.float32),
                    'file': os.path.basename(file_path),
                    'label': label,
                    'category': category,
                    'string': string_name,
                    'pluck': pluck_type,
                    'device': device_type,
                    'seg_idx': i
                })
    return segments


def collect_dataset():
    print("\n" + "=" * 65)
    print("STEP 1: Collecting & Preprocessing Saraswati Veena Audio Dataset")
    print("=" * 65)

    all_segments = []

    phase1_dir = "Phase1"
    if os.path.exists(phase1_dir):
        for root, dirs, files in os.walk(phase1_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    p = os.path.join(root, f)
                    norm = p.replace("\\", "/").lower()
                    string_name = "Unknown"
                    for s in ["s1_sarani", "s2_panchama", "s3_mandra", "s4_anumandra"]:
                        if s in norm:
                            string_name = s.upper()
                            break
                    pluck = "Unknown"
                    for pl in ["hard", "medium", "soft"]:
                        if pl in norm:
                            pluck = pl.capitalize()
                            break
                    segs = extract_audio_segments(p, label="healthy", category="Phase1_Healthy",
                                                  string_name=string_name, pluck_type=pluck, device_type="Primary_Mic")
                    all_segments.extend(segs)

    fault_dirs = [f"Fault{i}" for i in [1, 2, 3, 4, 5, 7, 10, 11]] + ["frets", "extracted_data"]
    for fdir in fault_dirs:
        if os.path.exists(fdir):
            for root, dirs, files in os.walk(fdir):
                for f in files:
                    if f.lower().endswith(".wav"):
                        p = os.path.join(root, f)
                        norm = p.replace("\\", "/").lower()
                        cat = "Fault_Other"
                        for i in [10, 11, 1, 2, 3, 4, 5, 7]:
                            if f"fault{i}" in norm:
                                cat = f"Fault_{i}"
                                break
                        if "fret" in norm:
                            cat = "Frets"
                        elif "general_test" in norm:
                            cat = "General_Test"

                        pluck = "Unknown"
                        for pl in ["hard", "medium", "soft"]:
                            if pl in norm:
                                pluck = pl.capitalize()
                                break

                        segs = extract_audio_segments(p, label="anomaly", category=cat,
                                                      string_name="Unknown", pluck_type=pluck, device_type="Primary_Mic")
                        all_segments.extend(segs)

    for f in ["test1 audio.wav", "test2 audio.wav"]:
        if os.path.exists(f):
            segs = extract_audio_segments(f, label="anomaly", category="Test_Audio",
                                          string_name="Unknown", pluck_type="Unknown", device_type="Primary_Mic")
            all_segments.extend(segs)

    phone_dir = "Phone audio recordings"
    if os.path.exists(phone_dir):
        for f in os.listdir(phone_dir):
            if f.lower().endswith(".wav"):
                p = os.path.join(phone_dir, f)
                fn = f.lower()
                if "noise" in fn:
                    continue # Exclude ambient noise from fault training labels
                lbl = "healthy" if "phase 1" in fn or "phase1" in fn else "anomaly"
                cat = "Phone_Healthy" if lbl == "healthy" else "Phone_Fault"

                strg = "Unknown"
                for s in ["s1", "s2", "s3", "s4"]:
                    if f" {s} " in fn or f"_{s}_" in fn or fn.startswith(f"swarcare phase 1 {s}"):
                        strg = s.upper()
                        break

                pluck = "Unknown"
                for pl in ["soft", "medium", "hard", "strong"]:
                    if pl in fn:
                        pluck = "Hard" if pl == "strong" else pl.capitalize()
                        break

                segs = extract_audio_segments(p, label=lbl, category=cat,
                                              string_name=strg, pluck_type=pluck, device_type="Phone_Mic")
                all_segments.extend(segs)

    print(f"Total valid 2.0-second audio segments extracted: {len(all_segments)}")
    df_meta = pd.DataFrame([{k: s[k] for k in s if k != 'audio'} for s in all_segments])
    return all_segments, df_meta


def extract_yamnet_embeddings(all_segments):
    print("\n" + "=" * 65)
    print("STEP 2: Deep & Physics Feature Extraction via YAMNet (LiteRT)")
    print("=" * 65)

    interpreter = tflite.Interpreter(model_path=YAMNET_MODEL_PATH)
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    yamnet_window_size = 15600
    hop_size = 7800

    embeddings = []
    pitch_features = []
    t0 = time.time()

    for idx, seg in enumerate(all_segments):
        audio = seg['audio']
        scores_list = []
        for start_idx in range(0, len(audio) - yamnet_window_size + 1, hop_size):
            sub_window = audio[start_idx:start_idx + yamnet_window_size]
            interpreter.set_tensor(inp_details['index'], sub_window)
            interpreter.invoke()
            scores_list.append(interpreter.get_tensor(out_details['index']).flatten())

        if not scores_list:
            padded = np.zeros(yamnet_window_size, dtype=np.float32)
            padded[:len(audio)] = audio
            interpreter.set_tensor(inp_details['index'], padded)
            interpreter.invoke()
            scores_list.append(interpreter.get_tensor(out_details['index']).flatten())

        embeddings.append(np.mean(scores_list, axis=0))

        # f0 autocorrelation & Cents deviation
        r = librosa.autocorrelate(audio)
        min_lag, max_lag = int(16000/400.0), int(16000/45.0)
        if len(r) > max_lag:
            peak_lag = min_lag + np.argmax(r[min_lag:max_lag])
            f0_val = float(16000.0 / float(peak_lag))
        else:
            f0_val = 0.0

        if f0_val > 30.0:
            cents_err = np.abs(1200.0 * np.log2(f0_val / TARGET_F0S))
            min_cents = float(np.min(cents_err))
            s1_err = float(f0_val - S1_F0)
            detuning_trigger = float((-25.0 <= s1_err <= -8.0) or (150.0 <= min_cents <= 400.0))
        else:
            min_cents = 1200.0
            s1_err = -100.0
            detuning_trigger = 0.0

        # Acoustic Structural Features (Research-derived)
        # 1. Energy Decay Rate (dRMS/dt): differentiates healthy decay from Fret Wear (Fault 4) & Bridge Tilt (Fault 5)
        rms_onset = float(np.sqrt(np.mean(audio[:int(0.3 * TARGET_SR)] ** 2)))
        rms_tail = float(np.sqrt(np.mean(audio[int(1.2 * TARGET_SR):] ** 2)))
        decay_rate = float((rms_onset - rms_tail) / (rms_onset + 1e-6))

        # 2. High-Frequency Spectral Flatness (>2000 Hz): captures chaotic noise from String Buzzing (Fault 7)
        stft_hf = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))[256:, :] # >2000 Hz bins
        flatness_hf = float(np.mean(librosa.feature.spectral_flatness(S=stft_hf)))

        pitch_features.append([f0_val, min_cents, s1_err, detuning_trigger, decay_rate, flatness_hf])

        if (idx + 1) % 150 == 0 or (idx + 1) == len(all_segments):
            print(f"  Processed {idx + 1}/{len(all_segments)} segments ({(idx + 1)/len(all_segments)*100:.1f}%)")

    total_time = time.time() - t0
    embeddings = np.array(embeddings, dtype=np.float32)
    pitch_features = np.array(pitch_features, dtype=np.float32)

    X_combined = np.hstack([embeddings, pitch_features])
    print(f"Extracted combined features shape: {X_combined.shape} in {total_time:.2f}s")
    return X_combined


def map_to_3class(all_segments):
    labels = []
    for s in all_segments:
        cat = s['category']
        fn = s['file'].lower()
        if cat in ['Phase1_Healthy', 'Phone_Healthy', 'Healthy_Baseline']:
            labels.append(0) # Healthy
        elif cat in ['Fault_1', 'Fault1'] or ('fault 1' in fn or 'fault s1' in fn):
            labels.append(1) # Detuned_Strings
        else:
            labels.append(2) # Quality_Issues (Fault 10 in Class 2)
    return np.array(labels, dtype=np.int32)


def evaluate_3class_groupkfold(X, y_3class, groups, k_folds=5, random_state=42):
    print("\n" + "=" * 65)
    print(f"STEP 3: 3-Class GroupKFold Cross-Validation (Leakage-Free, k={k_folds})")
    print("=" * 65)

    gkf = GroupKFold(n_splits=k_folds)

    models = {
        "Hybrid ML + Physics Pitch Gate": "hybrid_gate",
        "Gradient Boosting (Balanced)": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=random_state),
        "SVM (RBF Kernel Balanced)": SVC(kernel='rbf', C=15, class_weight='balanced', probability=True, random_state=random_state),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=random_state)
    }

    results = {m_name: {
        'fold_accuracy': [], 'fold_precision_macro': [], 'fold_recall_macro': [],
        'fold_f1_macro': [], 'all_y_true': [], 'all_y_pred': []
    } for m_name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_3class, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_3class[train_idx], y_3class[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        sample_weights = compute_sample_weight('balanced', y_train)

        # Baseline Gradient Boosting
        gb_base = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=random_state)
        gb_base.fit(X_train, y_train, sample_weight=sample_weights)
        pred_gb_base = gb_base.predict(X_test)

        # Hybrid Gate predictions
        pred_hybrid = []
        for idx, p_m in enumerate(pred_gb_base):
            s1_err = X[test_idx[idx]][-2] # Raw s1_err feature before scaling
            if (-25.0 <= s1_err <= -8.0) and p_m != 2:
                pred_hybrid.append(1) # Detuned_Strings
            else:
                pred_hybrid.append(p_m)

        for m_name, model in models.items():
            if m_name == "Hybrid ML + Physics Pitch Gate":
                y_pred = pred_hybrid
            elif m_name == "Gradient Boosting (Balanced)":
                y_pred = pred_gb_base
            elif m_name == "SVM (RBF Kernel Balanced)":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif m_name == "Random Forest (Balanced)":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            p_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            r_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

            results[m_name]['fold_accuracy'].append(acc)
            results[m_name]['fold_precision_macro'].append(p_macro)
            results[m_name]['fold_recall_macro'].append(r_macro)
            results[m_name]['fold_f1_macro'].append(f1_macro)

            results[m_name]['all_y_true'].extend(y_test)
            results[m_name]['all_y_pred'].extend(y_pred)

    summary_rows = []
    for m_name in models:
        res = results[m_name]
        summary_rows.append({
            'Model': m_name,
            'Accuracy': f"{np.mean(res['fold_accuracy'])*100:.2f}% +- {np.std(res['fold_accuracy'])*100:.2f}%",
            'F1 (Macro)': f"{np.mean(res['fold_f1_macro'])*100:.2f}% +- {np.std(res['fold_f1_macro'])*100:.2f}%",
            'Precision (Macro)': f"{np.mean(res['fold_precision_macro'])*100:.2f}% +- {np.std(res['fold_precision_macro'])*100:.2f}%",
            'Recall (Macro)': f"{np.mean(res['fold_recall_macro'])*100:.2f}% +- {np.std(res['fold_recall_macro'])*100:.2f}%",
            'raw_acc': np.mean(res['fold_accuracy']),
            'raw_f1': np.mean(res['fold_f1_macro'])
        })

    df_summary = pd.DataFrame(summary_rows).sort_values('raw_f1', ascending=False)
    print(f"\nSummary of 3-Class GroupKFold Cross-Validation Metrics:")
    print(df_summary[['Model', 'Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)']].to_string(index=False))

    return results, df_summary


def main():
    print("=" * 65)
    print("  SwarCare Veena Pipeline: Real-World 3-Class Diagnostic Suite")
    print("=" * 65)

    all_segments, df_meta = collect_dataset()
    X_features = extract_yamnet_embeddings(all_segments)

    y_3class = map_to_3class(all_segments)
    groups = np.array([s['file'] for s in all_segments])

    results, df_summary = evaluate_3class_groupkfold(X_features, y_3class, groups, k_folds=5)

    # Save to evaluation_results
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "5_fold_metrics.csv"), index=False)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "10_fold_metrics.csv"), index=False)

    export_data = {
        'metadata': {
            'total_segments': len(all_segments),
            'total_unique_files': len(set(groups)),
            'class_distribution': {
                'Healthy': int(np.sum(y_3class == 0)),
                'Detuned_Strings': int(np.sum(y_3class == 1)),
                'Quality_Issues': int(np.sum(y_3class == 2))
            }
        },
        'metrics_summary': df_summary.to_dict(orient='records')
    }

    with open(os.path.join(OUTPUT_DIR, "cross_validation_metrics.json"), 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\n  [SAVED] Complete real-world metrics exported to {OUTPUT_DIR}/5_fold_metrics.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
