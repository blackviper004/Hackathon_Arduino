"""
SwarCare Full Hybrid Pipeline Evaluation (Physics Gate + ML Classifier)
========================================================================
Performs 5-Fold GroupKFold Cross-Validation across all dataset recordings.

Pipeline Architecture:
1. Stage 1: Physics Pitch Engine Gate (evaluate open-string tuning vs ±15 cents)
   - If FLAT or SHARP -> Class 1: Detuned_Strings
2. Stage 2: ML Classifier (YamNet embeddings + RandomForest/SVM)
   - Evaluates remaining samples for Class 0: Healthy vs Class 2: Quality_Issues

Classes:
  0: Healthy
  1: Detuned_Strings (Fault 1)
  2: Quality_Issues (Fault 2 - 11, Frets, General)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Ensure CWD is pipeline directory
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from physics_pitch_engine import PhysicsPitchEngine
from train_and_evaluate import collect_dataset, extract_yamnet_embeddings

SR = 16000
CENTS_THRESH = 15.0

# Per-string calibrated targets measured from Phase1 healthy baseline
CALIBRATED_TARGETS = {
    "S4": 58.90,   # Anumandra
    "S3": 80.06,   # Mandra Sa (Tonic)
    "S2": 120.56,  # Panchama
    "S1": 162.80,  # Sarani
    "T1": 320.24,
    "T2": 480.36,
    "T3": 640.48,
}

class PerStringPhysicsGate:
    def __init__(self, targets=CALIBRATED_TARGETS, cents_thresh=CENTS_THRESH):
        self.targets = targets
        self.thresh = cents_thresh
        self.dsp = PhysicsPitchEngine(tonic_hz=targets["S3"], cents_threshold=cents_thresh, sr=SR)

    def is_detuned(self, audio):
        f0, conf = self.dsp.detect_f0_only(audio, SR)
        if f0 <= 30.0 or conf < 0.15:
            return False
        # Nearest string
        nearest = min(self.targets.keys(), key=lambda s: abs(1200.0 * np.log2(f0 / (self.targets[s] + 1e-9))))
        target_hz = self.targets[nearest]
        cents = 1200.0 * np.log2(f0 / (target_hz + 1e-9))
        return abs(cents) > self.thresh

def main():
    print("=" * 65)
    print("  SwarCare End-to-End Hybrid Pipeline Evaluation")
    print("  (Physics Pitch Gate + ML Classifier)")
    print("=" * 65)

    # 1. Collect segments & extract features
    all_segments, df_meta = collect_dataset()
    X_features = extract_yamnet_embeddings(all_segments)

    # 2. Extract physics gate predictions for all segments
    print("\n[STAGE 1] Running Physics Pitch Gate on all segments...")
    gate = PerStringPhysicsGate()
    physics_is_detuned = []
    for idx, seg in enumerate(all_segments):
        is_det = gate.is_detuned(seg['audio'])
        physics_is_detuned.append(is_det)
    physics_is_detuned = np.array(physics_is_detuned, dtype=bool)

    # 3. Ground truth 3-class labels
    # 0: Healthy, 1: Detuned_Strings, 2: Quality_Issues
    y_true = []
    for s in all_segments:
        cat = s['category']
        fn = s['file'].lower()
        if cat in ['Phase1_Healthy', 'Phone_Healthy', 'Healthy_Baseline']:
            y_true.append(0)
        elif cat in ['Fault_1', 'Fault1'] or ('fault 1' in fn or 'fault s1' in fn):
            y_true.append(1)
        else:
            y_true.append(2)
    y_true = np.array(y_true, dtype=int)

    groups = np.array([s['file'] for s in all_segments])

    # 4. Run GroupKFold (5-fold) for ML stage & Full Pipeline evaluation
    gkf = GroupKFold(n_splits=5)
    
    y_pipeline_preds = []
    y_true_all = []

    # Filter out Fault 1 from ML training set so ML only learns Healthy (0) vs Quality (2)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_features, y_true, groups=groups)):
        # ML training: exclude Detuned (Class 1) from ML classifier training
        train_mask_ml = (y_true[train_idx] != 1)
        train_idx_ml = train_idx[train_mask_ml]

        X_train, y_train = X_features[train_idx_ml], y_true[train_idx_ml]
        X_test, y_test = X_features[test_idx], y_true[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        weights = compute_sample_weight('balanced', y_train)

        # Train ML Classifier on Healthy vs Quality Issues
        clf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
        clf.fit(X_train_scaled, y_train, sample_weight=weights)

        # Predict ML for test fold
        ml_preds = clf.predict(X_test_scaled)

        # Combine Stage 1 (Physics Gate) and Stage 2 (ML Classifier)
        fold_pipeline_preds = []
        for i, global_idx in enumerate(test_idx):
            if physics_is_detuned[global_idx]:
                # Physics gate flagged as Detuned
                fold_pipeline_preds.append(1)
            else:
                # Physics gate passed -> use ML prediction (Healthy vs Quality)
                fold_pipeline_preds.append(ml_preds[i])

        y_pipeline_preds.extend(fold_pipeline_preds)
        y_true_all.extend(y_test)

    y_true_all = np.array(y_true_all)
    y_pipeline_preds = np.array(y_pipeline_preds)

    # 5. Compute Detailed Confusion Matrix & Metrics
    class_names = ["Healthy (Class 0)", "Detuned Strings (Class 1)", "Quality Issues (Class 2)"]
    cm = confusion_matrix(y_true_all, y_pipeline_preds, labels=[0, 1, 2])

    print("\n" + "=" * 65)
    print("  FULL HYBRID PIPELINE CONFUSION MATRIX")
    print("=" * 65)
    
    print("\nConfusion Matrix (Absolute Segment Counts):")
    df_cm = pd.DataFrame(cm, index=[f"True: {c}" for c in class_names],
                             columns=[f"Pred: {c}" for c in class_names])
    print(df_cm.to_string())

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
    print("\nNormalized Confusion Matrix (Row % - Recall per class):")
    df_cm_norm = pd.DataFrame(cm_norm.round(1), index=[f"True: {c}" for c in class_names],
                                             columns=[f"Pred: {c}" for c in class_names])
    print(df_cm_norm.to_string())

    acc = accuracy_score(y_true_all, y_pipeline_preds)
    p_macro = precision_score(y_true_all, y_pipeline_preds, average='macro', zero_division=0)
    r_macro = recall_score(y_true_all, y_pipeline_preds, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_all, y_pipeline_preds, average='macro', zero_division=0)

    print("\n" + "─" * 65)
    print(f"  Overall Pipeline Accuracy  : {acc * 100:.2f}%")
    print(f"  Macro Precision            : {p_macro * 100:.2f}%")
    print(f"  Macro Recall               : {r_macro * 100:.2f}%")
    print(f"  Macro F1 Score             : {f1_macro * 100:.2f}%")
    print("─" * 65)

    print("\nDetailed Per-Class Performance:")
    rep = classification_report(y_true_all, y_pipeline_preds, target_names=class_names, digits=4, zero_division=0)
    print(rep)

    # Export JSON
    os.makedirs("evaluation_results", exist_ok=True)
    out_dict = {
        "confusion_matrix_absolute": cm.tolist(),
        "confusion_matrix_percentage": cm_norm.round(2).tolist(),
        "classes": class_names,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(p_macro, 4),
            "recall_macro": round(r_macro, 4),
            "f1_macro": round(f1_macro, 4)
        }
    }
    with open("evaluation_results/full_hybrid_pipeline_metrics.json", "w") as f:
        json.dump(out_dict, f, indent=2)

    print("  [SAVED] Results exported to evaluation_results/full_hybrid_pipeline_metrics.json")
    print("=" * 65)

if __name__ == "__main__":
    main()
