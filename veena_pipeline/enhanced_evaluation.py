"""
SwarCare: Enterprise Saraswati Veena Multi-Class Evaluation Suite
==================================================================
Reference Physics:
  - Asokan et al. (2016): Structural Resonator Modes & Fundamental Tuning Targets
  - Chauhan et al. (2021): Extended Kudirai Bridge Overtone Dynamics & Harmonic Revival
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

from saraswati_veena_dsp_pipeline import VeenaAcousticDSPPipeline, VeenaClassifier

TARGET_SR_AUDIO = 16000
WINDOW_SEC = 2.0
ACTIVE_RMS_THRESH = 0.02
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def extract_audio_segments(file_path: str, label: str, category: str, device_type: str):
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR_AUDIO, mono=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load {file_path}: {e}")
        return []

    window_samples = int(WINDOW_SEC * TARGET_SR_AUDIO)
    total_windows = len(audio) // window_samples
    segments = []

    for i in range(total_windows):
        start = i * window_samples
        end = start + window_samples
        seg = audio[start:end]
        rms = compute_rms(seg)
        if rms >= ACTIVE_RMS_THRESH:
            segments.append({
                'audio': seg.astype(np.float32),
                'file': os.path.basename(file_path),
                'label': label,
                'category': category,
                'device': device_type,
                'seg_idx': i
            })
    return segments


def collect_audio_dataset():
    print("\n" + "=" * 70)
    print(f"STEP 1: Ingesting Active Audio Segments (RMS >= {ACTIVE_RMS_THRESH})")
    print("=" * 70)

    all_segments = []

    phase1_dir = "Phase1"
    if os.path.exists(phase1_dir):
        for root, dirs, files in os.walk(phase1_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    p = os.path.join(root, f)
                    segs = extract_audio_segments(p, label="healthy", category="Healthy_Baseline", device_type="Primary_Mic")
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

                        segs = extract_audio_segments(p, label="anomaly", category=cat, device_type="Primary_Mic")
                        all_segments.extend(segs)

    for f in ["test1 audio.wav", "test2 audio.wav"]:
        if os.path.exists(f):
            segs = extract_audio_segments(f, label="anomaly", category="Test_Pluck", device_type="Primary_Mic")
            all_segments.extend(segs)

    phone_dir = "Phone audio recordings"
    if os.path.exists(phone_dir):
        for f in os.listdir(phone_dir):
            if f.lower().endswith(".wav"):
                p = os.path.join(phone_dir, f)
                fn = f.lower()
                if "noise" in fn: continue
                lbl = "healthy" if "phase 1" in fn or "phase1" in fn else "anomaly"
                cat = "Phone_Healthy" if lbl == "healthy" else "Phone_Fault"
                segs = extract_audio_segments(p, label=lbl, category=cat, device_type="Phone_Mic")
                all_segments.extend(segs)

    print(f"Total active audio segments ingested: {len(all_segments)}")
    return all_segments


def map_to_3class_taxonomy(all_segments):
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


def main():
    print("=" * 70)
    print("  SwarCare Enterprise Veena Vibro-Acoustic Diagnostic Suite")
    print("=" * 70)

    # 1. Ingest audio
    all_segments = collect_audio_dataset()

    # 2. Extract physics-anchored DSP features
    print("\n" + "=" * 70)
    print("STEP 2: Extracting Physics-Anchored DSP Features (pYIN, HNR, Flux, Kudam)")
    print("=" * 70)

    dsp_engine = VeenaAcousticDSPPipeline(target_sr=TARGET_SR_AUDIO)
    feature_list = []
    
    t0 = time.time()
    for idx, s in enumerate(all_segments):
        audio = s['audio']
        f_vec = dsp_engine.extract_master_feature_vector(audio)
        feature_list.append(f_vec)
        if (idx + 1) % 150 == 0 or (idx + 1) == len(all_segments):
            print(f"  Extracted {idx + 1}/{len(all_segments)} samples ({(idx + 1)/len(all_segments)*100:.1f}%)")

    total_time = time.time() - t0
    df_features = pd.DataFrame(feature_list)
    X_matrix = df_features.values
    
    print(f"\nExtracted Feature Matrix shape: {X_matrix.shape} in {total_time:.2f}s")
    print(f"Features: {list(df_features.columns)}")

    # 3. Targets and Grouping
    y_targets = map_to_3class_taxonomy(all_segments)
    groups = np.array([s['file'] for s in all_segments])

    # 4. Balanced Machine Learning Classification with GroupKFold
    print("\n" + "=" * 70)
    print("STEP 3: 5-Fold GroupKFold Cross-Validation (Balanced Weights)")
    print("=" * 70)

    clf_suite = VeenaClassifier(random_state=42)
    eval_results = clf_suite.train_and_evaluate_group_cv(X_matrix, y_targets, groups, n_splits=5)

    summary_rows = []
    for m_name, res in eval_results.items():
        summary_rows.append({
            'Model Architecture': m_name,
            'Accuracy': f"{res['mean_accuracy']*100:.2f}% +- {res['std_accuracy']*100:.2f}%",
            'F1 (Macro)': f"{res['mean_f1_macro']*100:.2f}% +- {res['std_f1_macro']*100:.2f}%",
            'raw_acc': res['mean_accuracy'],
            'raw_f1': res['mean_f1_macro']
        })

    df_summary = pd.DataFrame(summary_rows).sort_values('raw_f1', ascending=False)
    print(f"\n3-Class Physics-Anchored Audio GroupKFold Leaderboard:")
    print(df_summary[['Model Architecture', 'Accuracy', 'F1 (Macro)']].to_string(index=False))

    top_model_name = df_summary.iloc[0]['Model Architecture']
    print(f"\nTop Model Classification Report ({top_model_name}):")
    top_rep = eval_results[top_model_name]['report']
    for cname in ['Healthy', 'Detuned_Strings', 'Quality_Issues']:
        print(f"  - {cname:16s} -> Precision: {top_rep[cname]['precision']:.2f} | Recall: {top_rep[cname]['recall']:.2f} | F1: {top_rep[cname]['f1-score']:.2f}")

    # 5. Generate & Save 3-Panel Confusion Matrix Comparison
    class_names = ['Healthy', 'Detuned_Strings', 'Quality_Issues']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Saraswati Veena Diagnostic 3-Class Confusion Matrices\nFile-Level GroupKFold Cross-Validation (Asokan & Chauhan Physics DSP Pipeline)', fontsize=14, fontweight='bold', y=1.03)

    model_items = list(eval_results.items())

    for idx, (m_name, m_data) in enumerate(model_items):
        ax = axes[idx]
        cm = np.array(m_data['confusion_matrix'])
        
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(f"{m_name}\nAcc: {m_data['mean_accuracy']*100:.2f}% | Macro F1: {m_data['mean_f1_macro']*100:.2f}%", fontsize=11, pad=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(class_names, fontsize=9, rotation=15)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Predicted Class', fontsize=10)
        if idx == 0:
            ax.set_ylabel('True Class', fontsize=10)

        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                row_sum = np.sum(cm[i])
                pct = (val / row_sum * 100) if row_sum > 0 else 0
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha='center', va='center',
                        color='white' if val > cm.max()/2 else 'black', fontsize=11, fontweight='bold')

    plt.tight_layout()
    cm_path1 = os.path.join(OUTPUT_DIR, "multiclass_fault_confusion_matrix.png")
    cm_path2 = "/Users/alen/.gemini/antigravity-ide/brain/ede50de6-58ce-402a-8d24-dca36e982904/multiclass_fault_confusion_matrix.png"

    plt.savefig(cm_path1, dpi=200, bbox_inches='tight')
    plt.savefig(cm_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [SAVED] 3-Panel confusion matrix heatmap plot to:\n   - {cm_path1}\n   - {cm_path2}")

    # 6. Save summary JSON & CSV
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "5_fold_metrics.csv"), index=False)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "3_class_group_cv_metrics.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "enhanced_cv_summary.json"), 'w') as f:
        json.dump({
            'metadata': {
                'total_segments': len(all_segments),
                'total_unique_files': len(set(groups)),
                'class_distribution': {
                    'Healthy': int(np.sum(y_targets == 0)),
                    'Detuned_Strings': int(np.sum(y_targets == 1)),
                    'Quality_Issues': int(np.sum(y_targets == 2))
                }
            },
            'evaluation_results': eval_results
        }, f, indent=2)

    print(f"  [SAVED] Enhanced CV summary JSON: {OUTPUT_DIR}/enhanced_cv_summary.json")

    print("\n" + "=" * 70)
    print("ENTERPRISE VEENA ACOUSTIC DIAGNOSTIC PIPELINE EXECUTED CLEANLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
