"""
SwarCare: Saraswati Veena Anomaly Detection & Classification Pipeline
=====================================================================
Comprehensive Training, Testing, and K-Fold Cross-Validation Suite
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import glob
import json
import time
import zipfile
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

import ai_edge_litert.interpreter as tflite

# === CONFIGURATION ===
TARGET_SR = 16000
WINDOW_SEC = 2.0
SILENCE_THRESHOLD = 0.008
PCA_COMPONENTS = 64
YAMNET_MODEL_PATH = "yamnet.tflite"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. AUDIO INGESTION & SEGMENTATION ───

def compute_rms(audio_segment: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio_segment ** 2)))


def extract_audio_segments(file_path: str, label: str, category: str, string_name: str, pluck_type: str, device_type: str):
    """Load WAV, resample to 16kHz mono, and slice into 2.0s segments."""
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load {file_path}: {e}")
        return []

    window_samples = int(WINDOW_SEC * TARGET_SR)
    total_windows = len(audio) // window_samples

    segments = []
    # If file is shorter than 2s but > 0.5s, pad with zeros
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
    """Scan and catalog all audio recordings across Phase1, Fault1..11, frets, phone audio, etc."""
    print("\n" + "=" * 65)
    print("STEP 1: Collecting & Preprocessing Saraswati Veena Audio Dataset")
    print("=" * 65)

    all_segments = []

    # 1. Phase 1 (Healthy baseline)
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
                        if f"/{pl}/" in norm or f"\\{pl}\\" in norm or pl in norm:
                            pluck = pl.capitalize()
                            break
                    segs = extract_audio_segments(p, label="healthy", category="Phase1_Healthy",
                                                  string_name=string_name, pluck_type=pluck, device_type="Primary_Mic")
                    all_segments.extend(segs)

    # 2. Fault directories (Fault1, Fault2, etc. and extracted zips)
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
                                cat = f"Fault{i}"
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

    # 3. Root test/noise WAV files
    for f in ["noise_audio.wav", "test1 audio.wav", "test2 audio.wav"]:
        if os.path.exists(f):
            lbl = "noise" if "noise" in f else "anomaly"
            cat = "Noise" if "noise" in f else "Test_Audio"
            segs = extract_audio_segments(f, label=lbl, category=cat,
                                          string_name="Unknown", pluck_type="Unknown", device_type="Primary_Mic")
            all_segments.extend(segs)

    # 4. Phone audio recordings (Mobile domain)
    phone_dir = "Phone audio recordings"
    if os.path.exists(phone_dir):
        for f in os.listdir(phone_dir):
            if f.lower().endswith(".wav"):
                p = os.path.join(phone_dir, f)
                fn = f.lower()
                lbl = "healthy" if "phase 1" in fn or "phase1" in fn else "anomaly"
                if "noise" in fn:
                    lbl = "noise"
                    cat = "Phone_Noise"
                elif lbl == "healthy":
                    cat = "Phone_Healthy"
                else:
                    cat = "Phone_Fault"

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
    print("\nDataset Segmentation Summary by Label:")
    print(df_meta['label'].value_counts())
    print("\nDataset Breakdown by Category:")
    print(df_meta['category'].value_counts())
    print("\nDataset Breakdown by Recording Device:")
    print(df_meta['device'].value_counts())

    return all_segments, df_meta


# ─── 2. YAMNET FEATURE EXTRACTION ───

def extract_yamnet_embeddings(all_segments):
    """Extract 521-D semantic acoustic embeddings using LiteRT YAMNet."""
    print("\n" + "=" * 65)
    print("STEP 2: Deep Feature Extraction via YAMNet (LiteRT)")
    print("=" * 65)

    interpreter = tflite.Interpreter(model_path=YAMNET_MODEL_PATH)
    interpreter.allocate_tensors()
    inp_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    yamnet_window_size = 15600  # 0.975s at 16kHz
    hop_size = 7800            # 50% overlap

    embeddings = []
    t0 = time.time()

    for idx, seg in enumerate(all_segments):
        audio = seg['audio']
        scores_list = []
        for start_idx in range(0, len(audio) - yamnet_window_size + 1, hop_size):
            sub_window = audio[start_idx:start_idx + yamnet_window_size]
            interpreter.set_tensor(inp_details['index'], sub_window)
            interpreter.invoke()
            scores = interpreter.get_tensor(out_details['index']).flatten()
            scores_list.append(scores)

        if not scores_list:
            padded = np.zeros(yamnet_window_size, dtype=np.float32)
            padded[:len(audio)] = audio
            interpreter.set_tensor(inp_details['index'], padded)
            interpreter.invoke()
            scores_list.append(interpreter.get_tensor(out_details['index']).flatten())

        avg_score = np.mean(scores_list, axis=0)
        embeddings.append(avg_score)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_segments):
            print(f"  Processed {idx + 1}/{len(all_segments)} segments ({(idx + 1)/len(all_segments)*100:.1f}%)")

    total_time = time.time() - t0
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Extracted YAMNet embeddings shape: {embeddings.shape} in {total_time:.2f}s ({total_time/len(all_segments)*1000:.1f}ms/seg)")
    return embeddings


# ─── 3. EVALUATION FUNCTIONS & METRICS ───

def evaluate_binary_classifiers(X, y, k_folds=5, random_state=42):
    """
    Run Stratified K-Fold Cross-Validation across multiple classifiers.
    """
    print("\n" + "=" * 65)
    print(f"STEP 3: Running Stratified {k_folds}-Fold Cross-Validation (Binary Anomaly Detection)")
    print("=" * 65)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    models = {
        "Distance-Based Anomaly Detector": "custom_distance",
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=random_state),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "k-Nearest Neighbors (k-NN)": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    }

    results = {m_name: {
        'fold_accuracy': [], 'fold_precision_macro': [], 'fold_precision_weighted': [],
        'fold_recall_macro': [], 'fold_recall_weighted': [],
        'fold_f1_macro': [], 'fold_f1_weighted': [],
        'fold_roc_auc': [], 'fold_pr_auc': [],
        'all_y_true': [], 'all_y_pred': [], 'all_y_proba': []
    } for m_name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit PCA on training fold healthy samples
        pca = PCA(n_components=min(PCA_COMPONENTS, len(X_train_raw) - 1), random_state=random_state)
        healthy_train_idx = np.where(y_train == 0)[0]
        if len(healthy_train_idx) > PCA_COMPONENTS:
            pca.fit(X_train_raw[healthy_train_idx])
        else:
            pca.fit(X_train_raw)

        X_train = pca.transform(X_train_raw)
        X_test = pca.transform(X_test_raw)

        for m_name, model in models.items():
            if m_name == "Distance-Based Anomaly Detector":
                healthy_feats = X_train[y_train == 0]
                center = np.mean(healthy_feats, axis=0)
                std = np.std(healthy_feats, axis=0) + 1e-6

                train_dists = np.sqrt(np.sum(((healthy_feats - center) / std) ** 2, axis=1))
                threshold = np.mean(train_dists) + 2.0 * np.std(train_dists)

                test_dists = np.sqrt(np.sum(((X_test - center) / std) ** 2, axis=1))
                y_pred = (test_dists >= threshold).astype(int)
                y_proba = np.clip(test_dists / (threshold * 2.0), 0.0, 1.0)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
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

            results[m_name]['all_y_true'].extend(y_test)
            results[m_name]['all_y_pred'].extend(y_pred)
            results[m_name]['all_y_proba'].extend(y_proba)

    # Summarize results
    summary_rows = []
    for m_name in models:
        res = results[m_name]
        summary_rows.append({
            'Model': m_name,
            'Accuracy': f"{np.mean(res['fold_accuracy'])*100:.2f}% +- {np.std(res['fold_accuracy'])*100:.2f}%",
            'F1 (Macro)': f"{np.mean(res['fold_f1_macro'])*100:.2f}% +- {np.std(res['fold_f1_macro'])*100:.2f}%",
            'F1 (Weighted)': f"{np.mean(res['fold_f1_weighted'])*100:.2f}% +- {np.std(res['fold_f1_weighted'])*100:.2f}%",
            'Precision (Macro)': f"{np.mean(res['fold_precision_macro'])*100:.2f}% +- {np.std(res['fold_precision_macro'])*100:.2f}%",
            'Recall (Macro)': f"{np.mean(res['fold_recall_macro'])*100:.2f}% +- {np.std(res['fold_recall_macro'])*100:.2f}%",
            'ROC-AUC': f"{np.mean(res['fold_roc_auc']):.4f} +- {np.std(res['fold_roc_auc']):.4f}",
            'PR-AUC': f"{np.mean(res['fold_pr_auc']):.4f} +- {np.std(res['fold_pr_auc']):.4f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print(f"\nSummary of {k_folds}-Fold Cross-Validation Metrics:")
    print(df_summary[['Model', 'Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'ROC-AUC']].to_string(index=False))

    return results, df_summary


# ─── 4. MULTI-CLASS FAULT CLASSIFICATION ───

def evaluate_multiclass_fault_classification(X, y_cat, category_names, k_folds=5, random_state=42):
    """
    Stratified K-Fold Cross-Validation for Multi-Class Fault Diagnosis
    """
    print("\n" + "=" * 65)
    print(f"STEP 4: Multi-Class Fault Classification ({k_folds}-Fold CV)")
    print("=" * 65)

    unique_cats, counts = np.unique(y_cat, return_counts=True)
    valid_cats = unique_cats[counts >= k_folds]
    valid_mask = np.isin(y_cat, valid_cats)

    X_filtered = X[valid_mask]
    y_filtered = y_cat[valid_mask]
    filtered_cat_names = [category_names[c] for c in valid_cats]

    cat_to_new_idx = {c: i for i, c in enumerate(valid_cats)}
    y_mapped = np.array([cat_to_new_idx[c] for c in y_filtered])

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=150, random_state=random_state)
    svm = SVC(kernel='rbf', probability=True, random_state=random_state)

    y_true_all, y_pred_rf_all, y_pred_svm_all = [], [], []

    for train_idx, test_idx in skf.split(X_filtered, y_mapped):
        X_train_raw, X_test_raw = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]

        pca = PCA(n_components=min(PCA_COMPONENTS, len(X_train_raw) - 1), random_state=random_state)
        X_train = pca.fit_transform(X_train_raw)
        X_test = pca.transform(X_test_raw)

        rf.fit(X_train, y_train)
        svm.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        y_pred_svm = svm.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_rf_all.extend(y_pred_rf)
        y_pred_svm_all.extend(y_pred_svm)

    print("\nRandom Forest Multi-Class Classification Report:")
    report_rf = classification_report(y_true_all, y_pred_rf_all, target_names=filtered_cat_names, zero_division=0)
    print(report_rf)

    print("\nSVM Multi-Class Classification Report:")
    report_svm = classification_report(y_true_all, y_pred_svm_all, target_names=filtered_cat_names, zero_division=0)
    print(report_svm)

    cm_rf = confusion_matrix(y_true_all, y_pred_rf_all)
    return {
        'filtered_categories': filtered_cat_names,
        'rf_accuracy': float(accuracy_score(y_true_all, y_pred_rf_all)),
        'rf_f1_macro': float(f1_score(y_true_all, y_pred_rf_all, average='macro', zero_division=0)),
        'rf_report': report_rf,
        'svm_accuracy': float(accuracy_score(y_true_all, y_pred_svm_all)),
        'svm_f1_macro': float(f1_score(y_true_all, y_pred_svm_all, average='macro', zero_division=0)),
        'svm_report': report_svm,
        'confusion_matrix': cm_rf,
        'y_true': [int(v) for v in y_true_all],
        'y_pred_rf': [int(v) for v in y_pred_rf_all]
    }


# ─── 5. GENERATE PLOTS & CHARTS ───

def generate_visualizations(binary_results, multiclass_results, embeddings, df_meta):
    """Generate high-resolution plots for confusion matrix, ROC curve, and distance distribution."""
    print("\n" + "=" * 65)
    print("STEP 5: Generating Visualization Artifacts")
    print("=" * 65)

    # 1. Confusion Matrix (Binary Anomaly Detection)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_svm = confusion_matrix(binary_results['SVM (RBF Kernel)']['all_y_true'],
                              binary_results['SVM (RBF Kernel)']['all_y_pred'])
    cm_dist = confusion_matrix(binary_results['Distance-Based Anomaly Detector']['all_y_true'],
                               binary_results['Distance-Based Anomaly Detector']['all_y_pred'])

    im0 = axes[0].imshow(cm_svm, cmap='Blues', interpolation='nearest')
    axes[0].set_title('SVM Classifier (RBF Kernel)\nAggregated K-Fold Confusion Matrix', fontsize=12, pad=10)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Healthy', 'Anomaly'], fontsize=11)
    axes[0].set_yticklabels(['Healthy', 'Anomaly'], fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    for i in range(2):
        for j in range(2):
            val = cm_svm[i, j]
            axes[0].text(j, i, f"{val}\n({val/np.sum(cm_svm[i]):.1%})",
                         ha='center', va='center', color='white' if val > cm_svm.max()/2 else 'black', fontsize=11, fontweight='bold')

    im1 = axes[1].imshow(cm_dist, cmap='Greens', interpolation='nearest')
    axes[1].set_title('Distance-Based Anomaly Detector\nAggregated K-Fold Confusion Matrix', fontsize=12, pad=10)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Healthy', 'Anomaly'], fontsize=11)
    axes[1].set_yticklabels(['Healthy', 'Anomaly'], fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    for i in range(2):
        for j in range(2):
            val = cm_dist[i, j]
            axes[1].text(j, i, f"{val}\n({val/np.sum(cm_dist[i]):.1%})",
                         ha='center', va='center', color='white' if val > cm_dist.max()/2 else 'black', fontsize=11, fontweight='bold')

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices_binary.png")
    plt.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Binary confusion matrices: {cm_path}")

    # 2. Multi-Class Confusion Matrix
    if multiclass_results and 'confusion_matrix' in multiclass_results:
        cm_mc = multiclass_results['confusion_matrix']
        cat_names = multiclass_results['filtered_categories']
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_mc, cmap='Purples', interpolation='nearest')
        plt.title('Multi-Class Fault Diagnosis Confusion Matrix (Random Forest)', fontsize=13, pad=12)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(range(len(cat_names)), cat_names, rotation=45, ha='right', fontsize=10)
        plt.yticks(range(len(cat_names)), cat_names, fontsize=10)
        plt.xlabel('Predicted Fault Type', fontsize=11)
        plt.ylabel('True Fault Type', fontsize=11)

        for i in range(len(cat_names)):
            for j in range(len(cat_names)):
                val = cm_mc[i, j]
                plt.text(j, i, str(val), ha='center', va='center',
                         color='white' if val > cm_mc.max()/2 else 'black', fontsize=9)

        plt.tight_layout()
        mc_path = os.path.join(OUTPUT_DIR, "confusion_matrix_multiclass.png")
        plt.savefig(mc_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] Multi-class confusion matrix: {mc_path}")

    # 3. ROC Curves
    plt.figure(figsize=(8, 6))
    for m_name, res in binary_results.items():
        y_true = np.array(res['all_y_true'])
        y_proba = np.array(res['all_y_proba'])
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{m_name} (AUC = {roc_val:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison (K-Fold CV)', fontsize=12, pad=10)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves_comparison.png")
    plt.savefig(roc_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] ROC comparison curve: {roc_path}")


# ─── 6. MAIN EXECUTION PIPELINE ───

def main():
    print("=" * 65)
    print("  SwarCare Veena Pipeline: Training, Testing & K-Fold CV Suite")
    print("=" * 65)

    # Step 1: Collect and segment audio
    all_segments, df_meta = collect_dataset()

    # Step 2: Extract YAMNet deep embeddings
    embeddings = extract_yamnet_embeddings(all_segments)

    # Prepare labels
    binary_labels = []
    for s in all_segments:
        if s['label'] == 'healthy':
            binary_labels.append(0)
        else:
            binary_labels.append(1)  # 1 = Anomaly/Fault
    y_binary = np.array(binary_labels)

    # Categories for multi-class
    unique_cats = sorted(list(set(df_meta['category'])))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    y_cat = np.array([cat_to_idx[s['category']] for s in all_segments])

    # Step 3: Run 5-Fold and 10-Fold Cross Validation
    results_5fold, df_5fold = evaluate_binary_classifiers(embeddings, y_binary, k_folds=5)
    results_10fold, df_10fold = evaluate_binary_classifiers(embeddings, y_binary, k_folds=10)

    # Step 4: Run Multi-Class Fault Classification
    mc_results = evaluate_multiclass_fault_classification(embeddings, y_cat, unique_cats, k_folds=5)

    # Step 5: Generate Visualizations
    generate_visualizations(results_5fold, mc_results, embeddings, df_meta)

    # Step 6: Export Full Metrics Summary to JSON & CSV
    export_data = {
        'dataset_summary': {
            'total_segments': len(all_segments),
            'total_healthy_segments': int(np.sum(y_binary == 0)),
            'total_anomaly_segments': int(np.sum(y_binary == 1)),
            'categories': {cat: int(count) for cat, count in df_meta['category'].value_counts().items()},
            'devices': {dev: int(count) for dev, count in df_meta['device'].value_counts().items()},
        },
        'k_fold_5_metrics': df_5fold.to_dict(orient='records'),
        'k_fold_10_metrics': df_10fold.to_dict(orient='records'),
        'multiclass_metrics': {
            'categories': mc_results['filtered_categories'],
            'rf_accuracy': mc_results['rf_accuracy'],
            'rf_f1_macro': mc_results['rf_f1_macro'],
            'svm_accuracy': mc_results['svm_accuracy'],
            'svm_f1_macro': mc_results['svm_f1_macro'],
        }
    }

    metrics_json_path = os.path.join(OUTPUT_DIR, "cross_validation_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"\n  [SAVED] Complete metrics exported to {metrics_json_path}")

    df_5fold.to_csv(os.path.join(OUTPUT_DIR, "5_fold_metrics.csv"), index=False)
    df_10fold.to_csv(os.path.join(OUTPUT_DIR, "10_fold_metrics.csv"), index=False)

    print("\n" + "=" * 65)
    print("PIPELINE TRAINING & CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
