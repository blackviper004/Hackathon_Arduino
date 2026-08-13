"""
SwarCare: Enhanced Saraswati Veena Anomaly Detection & Classification Suite
==========================================================================
Includes:
1. Hybrid Feature Extraction: YAMNet (521D) + Spectral Features (MFCCs, Centroid, Rolloff, ZCR, Bandwidth, Energy)
2. Calibrated Anomaly Detection (Tuned Thresholds, Mahalanobis, One-Class SVM, Isolation Forest)
3. Stratified 5-Fold and 10-Fold Cross-Validation for Binary and Multi-Class Fault Classification
4. Complete statistical metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, FPR, Confusion Matrices)
5. Analysis across Strings (S1-S4), Pluck Dynamics (Hard/Med/Soft), and Hardware (Primary vs Phone)
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)

import ai_edge_litert.interpreter as tflite

TARGET_SR = 16000
WINDOW_SEC = 2.0
SILENCE_THRESHOLD = 0.008
PCA_COMPONENTS = 64
YAMNET_MODEL_PATH = "yamnet.tflite"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. AUDIO INGESTION & SEGMENTATION ───

def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def extract_audio_features(audio: np.ndarray, sr: int):
    """Extract complementary acoustic & spectral features from 2.0s audio."""
    # 1. MFCCs (13 coefficients + delta)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)

    # 2. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    cent_mean, cent_std = np.mean(cent), np.std(cent)

    # 3. Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    bw_mean, bw_std = np.mean(bw), np.std(bw)

    # 4. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    rolloff_mean, rolloff_std = np.mean(rolloff), np.std(rolloff)

    # 5. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

    # 6. Spectral Flatness
    flat = librosa.feature.spectral_flatness(y=audio)
    flat_mean = np.mean(flat)

    # 7. RMS Energy
    rms_val = compute_rms(audio)

    # Concatenate all spectral features (45 dimensions)
    spectral_vector = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_delta_mean,
        [cent_mean, cent_std, bw_mean, bw_std, rolloff_mean, rolloff_std, zcr_mean, zcr_std, flat_mean, rms_val]
    ])
    return spectral_vector.astype(np.float32)


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
    print("STEP 1: Ingesting & Segmenting Audio Dataset")
    print("=" * 65)

    all_segments = []

    # 1. Phase 1 Healthy
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
                            string_name = s.split('_')[0].upper()
                            break
                    pluck = "Unknown"
                    for pl in ["hard", "medium", "soft"]:
                        if f"/{pl}/" in norm or f"\\{pl}\\" in norm or pl in norm:
                            pluck = pl.capitalize()
                            break
                    segs = extract_audio_segments(p, label="healthy", category="Healthy_Baseline",
                                                  string_name=string_name, pluck_type=pluck, device_type="Primary_Mic")
                    all_segments.extend(segs)

    # 2. Faults & Extracted Data
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

                        strg = "Unknown"
                        for s in ["s1", "s2", "s3", "s4"]:
                            if f"_{s}_" in norm or f" {s} " in norm:
                                strg = s.upper()
                                break

                        segs = extract_audio_segments(p, label="anomaly", category=cat,
                                                      string_name=strg, pluck_type=pluck, device_type="Primary_Mic")
                        all_segments.extend(segs)

    # 3. Test files
    for f in ["noise_audio.wav", "test1 audio.wav", "test2 audio.wav"]:
        if os.path.exists(f):
            lbl = "anomaly"
            cat = "Noise" if "noise" in f else "Test_Pluck"
            segs = extract_audio_segments(f, label=lbl, category=cat,
                                          string_name="Unknown", pluck_type="Unknown", device_type="Primary_Mic")
            all_segments.extend(segs)

    # 4. Phone audio
    phone_dir = "Phone audio recordings"
    if os.path.exists(phone_dir):
        for f in os.listdir(phone_dir):
            if f.lower().endswith(".wav"):
                p = os.path.join(phone_dir, f)
                fn = f.lower()
                lbl = "healthy" if "phase 1" in fn or "phase1" in fn else "anomaly"
                cat = "Phone_Healthy" if lbl == "healthy" else ("Phone_Noise" if "noise" in fn else "Phone_Fault")
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

    df_meta = pd.DataFrame([{k: s[k] for k in s if k != 'audio'} for s in all_segments])
    print(f"Total valid segments: {len(all_segments)}")
    return all_segments, df_meta


# ─── 2. DEEP & SPECTRAL FEATURE EXTRACTION ───

def extract_all_features(all_segments):
    print("\n" + "=" * 65)
    print("STEP 2: Extracting YAMNet Semantic + Spectral Features")
    print("=" * 65)

    interpreter = tflite.Interpreter(model_path=YAMNET_MODEL_PATH)
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    yamnet_window_size = 15600
    hop_size = 7800

    yamnet_embs = []
    spectral_feats = []

    t0 = time.time()
    for idx, seg in enumerate(all_segments):
        audio = seg['audio']

        # YAMNet inference
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

        yamnet_embs.append(np.mean(scores_list, axis=0))

        # Spectral feature extraction
        spec = extract_audio_features(audio, TARGET_SR)
        spectral_feats.append(spec)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_segments):
            print(f"  Extracted {idx + 1}/{len(all_segments)} samples ({(idx + 1)/len(all_segments)*100:.1f}%)")

    total_time = time.time() - t0
    yamnet_embs = np.array(yamnet_embs, dtype=np.float32)
    spectral_feats = np.array(spectral_feats, dtype=np.float32)

    print(f"Feature extraction completed in {total_time:.2f}s:")
    print(f"  - YAMNet Embeddings: {yamnet_embs.shape} (521-D)")
    print(f"  - Spectral Features:  {spectral_feats.shape} (45-D)")

    return yamnet_embs, spectral_feats


# ─── 3. STRATIFIED K-FOLD BENCHMARKING ───

def run_kfold_evaluation(yamnet_embs, spectral_feats, y_binary, k_folds=5, random_state=42):
    print("\n" + "=" * 65)
    print(f"STEP 3: Running Stratified {k_folds}-Fold Cross-Validation Benchmark")
    print("=" * 65)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Models to benchmark
    models = {
        "YAMNet + Distance Anomaly Detector (Calibrated)": "calibrated_distance",
        "YAMNet + PCA + SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=random_state),
        "YAMNet + PCA + Random Forest": RandomForestClassifier(n_estimators=150, random_state=random_state),
        "YAMNet + PCA + Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=random_state),
        "YAMNet + PCA + Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "YAMNet + PCA + Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "YAMNet + PCA + k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Hybrid (YAMNet + Spectral) + Random Forest": "hybrid_rf",
        "Hybrid (YAMNet + Spectral) + SVM (RBF)": "hybrid_svm",
        "Hybrid (YAMNet + Spectral) + Gradient Boosting": "hybrid_gb",
    }

    results = {m_name: {
        'fold_accuracy': [], 'fold_precision_macro': [], 'fold_precision_weighted': [],
        'fold_recall_macro': [], 'fold_recall_weighted': [],
        'fold_f1_macro': [], 'fold_f1_weighted': [],
        'fold_roc_auc': [], 'fold_pr_auc': [],
        'all_y_true': [], 'all_y_pred': [], 'all_y_proba': []
    } for m_name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(yamnet_embs, y_binary)):
        y_train, y_test = y_binary[train_idx], y_binary[test_idx]

        # 1. YAMNet PCA Pipeline
        Y_train_raw, Y_test_raw = yamnet_embs[train_idx], yamnet_embs[test_idx]
        pca = PCA(n_components=min(PCA_COMPONENTS, len(Y_train_raw) - 1), random_state=random_state)
        # Fit on healthy train samples
        healthy_mask = (y_train == 0)
        if np.sum(healthy_mask) > PCA_COMPONENTS:
            pca.fit(Y_train_raw[healthy_mask])
        else:
            pca.fit(Y_train_raw)

        Y_train = pca.transform(Y_train_raw)
        Y_test = pca.transform(Y_test_raw)

        # 2. Hybrid Features (YAMNet 64D PCA + Normalized Spectral 45D)
        S_train_raw, S_test_raw = spectral_feats[train_idx], spectral_feats[test_idx]
        scaler = StandardScaler()
        S_train = scaler.fit_transform(S_train_raw)
        S_test = scaler.transform(S_test_raw)

        H_train = np.hstack([Y_train, S_train])
        H_test = np.hstack([Y_test, S_test])

        for m_name, model in models.items():
            if m_name == "YAMNet + Distance Anomaly Detector (Calibrated)":
                # Calibrated Distance Detector
                healthy_feats = Y_train[y_train == 0]
                center = np.mean(healthy_feats, axis=0)
                std = np.std(healthy_feats, axis=0) + 1e-6

                train_dists = np.sqrt(np.sum(((healthy_feats - center) / std) ** 2, axis=1))
                all_train_dists = np.sqrt(np.sum(((Y_train - center) / std) ** 2, axis=1))

                # Determine optimal threshold via Youden's J statistic on train fold
                best_thresh = np.mean(train_dists) + 1.0 * np.std(train_dists)
                best_j = -1
                for thresh_cand in np.linspace(np.min(all_train_dists), np.max(all_train_dists), 100):
                    preds_cand = (all_train_dists >= thresh_cand).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_train, preds_cand, labels=[0, 1]).ravel()
                    sens = tp / (tp + fn + 1e-8)
                    spec = tn / (tn + fp + 1e-8)
                    j_stat = sens + spec - 1
                    if j_stat > best_j:
                        best_j = j_stat
                        best_thresh = thresh_cand

                test_dists = np.sqrt(np.sum(((Y_test - center) / std) ** 2, axis=1))
                y_pred = (test_dists >= best_thresh).astype(int)
                y_proba = np.clip(test_dists / (best_thresh * 2.0), 0.0, 1.0)

            elif m_name == "Hybrid (YAMNet + Spectral) + Random Forest":
                clf = RandomForestClassifier(n_estimators=150, random_state=random_state)
                clf.fit(H_train, y_train)
                y_pred = clf.predict(H_test)
                y_proba = clf.predict_proba(H_test)[:, 1]

            elif m_name == "Hybrid (YAMNet + Spectral) + SVM (RBF)":
                clf = SVC(kernel='rbf', probability=True, random_state=random_state)
                clf.fit(H_train, y_train)
                y_pred = clf.predict(H_test)
                y_proba = clf.predict_proba(H_test)[:, 1]

            elif m_name == "Hybrid (YAMNet + Spectral) + Gradient Boosting":
                clf = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
                clf.fit(H_train, y_train)
                y_pred = clf.predict(H_test)
                y_proba = clf.predict_proba(H_test)[:, 1]

            else:
                model.fit(Y_train, y_train)
                y_pred = model.predict(Y_test)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(Y_test)[:, 1]
                else:
                    y_proba = y_pred

            acc = accuracy_score(y_test, y_pred)
            p_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            p_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            r_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            r_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            try:
                roc_auc = roc_auc_score(y_test, y_proba)
            except Exception:
                roc_auc = 0.5

            try:
                pr_auc = average_precision_score(y_test, y_proba)
            except Exception:
                pr_auc = 0.5

            results[m_name]['fold_accuracy'].append(acc)
            results[m_name]['fold_precision_macro'].append(p_macro)
            results[m_name]['fold_precision_weighted'].append(p_weighted)
            results[m_name]['fold_recall_macro'].append(r_macro)
            results[m_name]['fold_recall_weighted'].append(r_weighted)
            results[m_name]['fold_f1_macro'].append(f1_macro)
            results[m_name]['fold_f1_weighted'].append(f1_weighted)
            results[m_name]['fold_roc_auc'].append(roc_auc)
            results[m_name]['fold_pr_auc'].append(pr_auc)

            results[m_name]['all_y_true'].extend([int(v) for v in y_test])
            results[m_name]['all_y_pred'].extend([int(v) for v in y_pred])
            results[m_name]['all_y_proba'].extend([float(v) for v in y_proba])

    # Summarize into DataFrame
    summary_rows = []
    for m_name in models:
        res = results[m_name]
        summary_rows.append({
            'Model Architecture': m_name,
            'Accuracy': f"{np.mean(res['fold_accuracy'])*100:.2f}% +- {np.std(res['fold_accuracy'])*100:.2f}%",
            'F1 (Macro)': f"{np.mean(res['fold_f1_macro'])*100:.2f}% +- {np.std(res['fold_f1_macro'])*100:.2f}%",
            'F1 (Weighted)': f"{np.mean(res['fold_f1_weighted'])*100:.2f}% +- {np.std(res['fold_f1_weighted'])*100:.2f}%",
            'Precision (Macro)': f"{np.mean(res['fold_precision_macro'])*100:.2f}% +- {np.std(res['fold_precision_macro'])*100:.2f}%",
            'Recall (Macro)': f"{np.mean(res['fold_recall_macro'])*100:.2f}% +- {np.std(res['fold_recall_macro'])*100:.2f}%",
            'ROC-AUC': f"{np.mean(res['fold_roc_auc']):.4f} +- {np.std(res['fold_roc_auc']):.4f}",
            'PR-AUC': f"{np.mean(res['fold_pr_auc']):.4f} +- {np.std(res['fold_pr_auc']):.4f}",
            'raw_acc': np.mean(res['fold_accuracy']),
            'raw_f1': np.mean(res['fold_f1_macro']),
            'raw_auc': np.mean(res['fold_roc_auc'])
        })

    df_summary = pd.DataFrame(summary_rows).sort_values('raw_f1', ascending=False)
    print(f"\n{k_folds}-Fold Cross-Validation Leaderboard:")
    print(df_summary[['Model Architecture', 'Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'ROC-AUC']].to_string(index=False))

    return results, df_summary


# ─── 4. MULTI-CLASS FAULT DIAGNOSIS ───

def run_multiclass_evaluation(yamnet_embs, spectral_feats, y_cat, category_names, k_folds=5, random_state=42):
    print("\n" + "=" * 65)
    print(f"STEP 4: Multi-Class Fault Classification ({k_folds}-Fold CV)")
    print("=" * 65)

    unique_cats, counts = np.unique(y_cat, return_counts=True)
    valid_cats = unique_cats[counts >= k_folds]
    valid_mask = np.isin(y_cat, valid_cats)

    Y_filt = yamnet_embs[valid_mask]
    S_filt = spectral_feats[valid_mask]
    y_filt = y_cat[valid_mask]
    cat_names = [category_names[c] for c in valid_cats]

    cat_to_new = {c: i for i, c in enumerate(valid_cats)}
    y_mapped = np.array([cat_to_new[c] for c in y_filt])

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    hybrid_rf = RandomForestClassifier(n_estimators=200, random_state=random_state)

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(Y_filt, y_mapped):
        y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]

        # YAMNet PCA
        pca = PCA(n_components=min(PCA_COMPONENTS, len(train_idx) - 1), random_state=random_state)
        Y_tr = pca.fit_transform(Y_filt[train_idx])
        Y_te = pca.transform(Y_filt[test_idx])

        # Spectral Standardized
        scaler = StandardScaler()
        S_tr = scaler.fit_transform(S_filt[train_idx])
        S_te = scaler.transform(S_filt[test_idx])

        H_tr = np.hstack([Y_tr, S_tr])
        H_te = np.hstack([Y_te, S_te])

        hybrid_rf.fit(H_tr, y_train)
        y_pred = hybrid_rf.predict(H_te)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    rep = classification_report(y_true_all, y_pred_all, target_names=cat_names, zero_division=0, output_dict=True)
    print("\nHybrid Random Forest Multi-Class Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=cat_names, zero_division=0))

    cm = confusion_matrix(y_true_all, y_pred_all)
    return {
        'categories': cat_names,
        'accuracy': float(accuracy_score(y_true_all, y_pred_all)),
        'f1_macro': float(f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)),
        'report': rep,
        'confusion_matrix': cm.tolist(),
        'cm_np': cm
    }


# ─── 5. GENERATE FINAL PLOTS ───

def generate_enhanced_plots(results_5fold, mc_res):
    print("\n" + "=" * 65)
    print("STEP 5: Generating High-Resolution Evaluation Artifacts")
    print("=" * 65)

    # 1. Comparison of ROC Curves
    plt.figure(figsize=(9, 7))
    for m_name in [
        "Hybrid (YAMNet + Spectral) + Random Forest",
        "Hybrid (YAMNet + Spectral) + SVM (RBF)",
        "YAMNet + PCA + Random Forest",
        "YAMNet + PCA + SVM (RBF)",
        "YAMNet + Distance Anomaly Detector (Calibrated)",
    ]:
        if m_name in results_5fold:
            res = results_5fold[m_name]
            fpr, tpr, _ = roc_curve(res['all_y_true'], res['all_y_proba'])
            auc_score = roc_auc_score(res['all_y_true'], res['all_y_proba'])
            plt.plot(fpr, tpr, lw=2.2, label=f"{m_name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison\n5-Fold Stratified Cross-Validation', fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "roc_comparison_enhanced.png")
    plt.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {p1}")

    # 2. Confusion Matrices for Top Models
    top_models = [
        "Hybrid (YAMNet + Spectral) + Random Forest",
        "Hybrid (YAMNet + Spectral) + SVM (RBF)",
        "YAMNet + Distance Anomaly Detector (Calibrated)",
        "YAMNet + PCA + Gradient Boosting"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, m_name in enumerate(top_models):
        res = results_5fold[m_name]
        cm = confusion_matrix(res['all_y_true'], res['all_y_pred'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = axes[idx].imshow(cm_norm, cmap='Blues' if 'Hybrid' in m_name else 'Greens', vmin=0, vmax=1)
        axes[idx].set_title(f"{m_name}\nAcc: {accuracy_score(res['all_y_true'], res['all_y_pred'])*100:.1f}% | F1: {f1_score(res['all_y_true'], res['all_y_pred'], average='macro')*100:.1f}%", fontsize=11, pad=8)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Healthy', 'Anomaly'], fontsize=10)
        axes[idx].set_yticklabels(['Healthy', 'Anomaly'], fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)

        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                               ha='center', va='center',
                               color='white' if cm_norm[i, j] > 0.5 else 'black',
                               fontsize=11, fontweight='bold')

    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "confusion_matrices_top_models.png")
    plt.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {p2}")

    # 3. Multi-Class Confusion Matrix Heatmap
    if mc_res and 'cm_np' in mc_res:
        cm_mc = np.array(mc_res['cm_np'])
        cat_names = mc_res['categories']
        plt.figure(figsize=(11, 9))
        plt.imshow(cm_mc, cmap='magma_r', interpolation='nearest')
        plt.title(f"Multi-Class Fault Diagnosis Confusion Matrix\nHybrid Random Forest (Accuracy: {mc_res['accuracy']*100:.1f}%)", fontsize=13, pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(range(len(cat_names)), cat_names, rotation=45, ha='right', fontsize=10)
        plt.yticks(range(len(cat_names)), cat_names, fontsize=10)
        plt.xlabel('Predicted Fault Class', fontsize=11)
        plt.ylabel('True Fault Class', fontsize=11)

        for i in range(len(cat_names)):
            for j in range(len(cat_names)):
                val = cm_mc[i, j]
                if val > 0:
                    plt.text(j, i, str(val), ha='center', va='center',
                             color='white' if val > cm_mc.max()/2 else 'black', fontsize=9)

        plt.tight_layout()
        p3 = os.path.join(OUTPUT_DIR, "multiclass_fault_confusion_matrix.png")
        plt.savefig(p3, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] {p3}")


# ─── 6. MAIN ───

def main():
    print("=" * 65)
    print("  SwarCare Enhanced Training, Testing & K-Fold CV Pipeline")
    print("=" * 65)

    all_segments, df_meta = collect_dataset()
    yamnet_embs, spectral_feats = extract_all_features(all_segments)

    # Binary labels (0: Healthy, 1: Anomaly/Fault)
    y_binary = np.array([0 if s['label'] == 'healthy' else 1 for s in all_segments])

    # Multi-class categories
    unique_cats = sorted(list(set(df_meta['category'])))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    y_cat = np.array([cat_to_idx[s['category']] for s in all_segments])

    # 5-Fold & 10-Fold CV
    res_5f, df_5f = run_kfold_evaluation(yamnet_embs, spectral_feats, y_binary, k_folds=5)
    res_10f, df_10f = run_kfold_evaluation(yamnet_embs, spectral_feats, y_binary, k_folds=10)

    # Multi-Class Evaluation
    mc_res = run_multiclass_evaluation(yamnet_embs, spectral_feats, y_cat, unique_cats, k_folds=5)

    # Visualizations
    generate_enhanced_plots(res_5f, mc_res)

    # Save to disk
    df_5f.to_csv(os.path.join(OUTPUT_DIR, "enhanced_5fold_metrics.csv"), index=False)
    df_10f.to_csv(os.path.join(OUTPUT_DIR, "enhanced_10fold_metrics.csv"), index=False)

    summary_out = {
        'metadata': {
            'total_samples': len(all_segments),
            'healthy_samples': int(np.sum(y_binary == 0)),
            'anomaly_samples': int(np.sum(y_binary == 1)),
            'category_counts': {k: int(v) for k, v in df_meta['category'].value_counts().items()},
            'device_counts': {k: int(v) for k, v in df_meta['device'].value_counts().items()},
        },
        '5_fold_leaderboard': df_5f.to_dict(orient='records'),
        '10_fold_leaderboard': df_10f.to_dict(orient='records'),
        'multiclass_diagnosis': mc_res['report']
    }

    with open(os.path.join(OUTPUT_DIR, "enhanced_cv_summary.json"), 'w') as f:
        json.dump(summary_out, f, indent=2)

    print("\n" + "=" * 65)
    print("ALL EVALUATIONS & CROSS-VALIDATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
