"""
SwarCare Parallel Hybrid Pipeline Evaluation
============================================
Runs Physics Pitch Engine and ML Classifier in PARALLEL.

Parallel Pipeline Output:
1. Tuning Result (Physics Engine):
   - Pitch Deviation (cents)
   - Tuning Status: IN_TUNE vs DETUNED (FLAT/SHARP)
   - Peg Guidance Instruction

2. Structural Quality Result (ML Classifier):
   - Healthy vs Quality Issue (Faults 2 - 11)
   - ML Model trained on Healthy vs Quality Faults (excluding Fault 1 detuning)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from physics_pitch_engine import PhysicsPitchEngine
from train_and_evaluate import collect_dataset, extract_yamnet_embeddings

SR = 16000
CENTS_THRESH = 15.0

CALIBRATED_TARGETS = {
    "S4": 58.90,   # Anumandra
    "S3": 80.06,   # Mandra Sa (Tonic)
    "S2": 120.56,  # Panchama
    "S1": 162.80,  # Sarani
    "T1": 320.24,
    "T2": 480.36,
    "T3": 640.48,
}

class PerStringPhysicsEngine:
    def __init__(self, targets=CALIBRATED_TARGETS, cents_thresh=CENTS_THRESH):
        self.targets = targets
        self.thresh = cents_thresh
        self.dsp = PhysicsPitchEngine(tonic_hz=targets["S3"], cents_threshold=cents_thresh, sr=SR)

    def evaluate_tuning(self, audio):
        f0, conf = self.dsp.detect_f0_only(audio, SR)
        if f0 <= 30.0 or conf < 0.15:
            return "NO_PITCH", 0.0
        nearest = min(self.targets.keys(), key=lambda s: abs(1200.0 * np.log2(f0 / (self.targets[s] + 1e-9))))
        target_hz = self.targets[nearest]
        cents = 1200.0 * np.log2(f0 / (target_hz + 1e-9))
        status = "IN_TUNE" if abs(cents) <= self.thresh else ("FLAT" if cents < 0 else "SHARP")
        return status, round(cents, 1)

def main():
    print("=" * 65)
    print("  SwarCare Parallel Hybrid Pipeline Evaluation")
    print("  (Physics Pitch Engine & ML Classifier Running Simultaneously)")
    print("=" * 65)

    all_segments, df_meta = collect_dataset()
    X_features = extract_yamnet_embeddings(all_segments)

    # 1. Physics Engine (Tuning Evaluation)
    print("\n[PHYSICS ENGINE] Evaluating open-string tuning for all segments...")
    phys_engine = PerStringPhysicsEngine()
    phys_results = [phys_engine.evaluate_tuning(s['audio']) for s in all_segments]
    phys_detuned = np.array([res[0] in ("FLAT", "SHARP") for res in phys_results])

    # 2. Binary Quality ML Ground Truth
    # 0: Healthy, 1: Quality Issue (Faults 2 - 11)
    # Filter out Fault 1 (pure detuned) for ML quality training/testing
    y_quality_true = []
    valid_indices = []

    for idx, s in enumerate(all_segments):
        cat = s['category']
        fn = s['file'].lower()
        if cat in ['Phase1_Healthy', 'Phone_Healthy', 'Healthy_Baseline']:
            y_quality_true.append(0) # Healthy
            valid_indices.append(idx)
        elif cat in ['Fault_1', 'Fault1'] or ('fault 1' in fn or 'fault s1' in fn):
            continue # Exclude pure detuning from structural quality classifier
        else:
            y_quality_true.append(1) # Quality Issue (Faults 2-11)
            valid_indices.append(idx)

    y_quality_true = np.array(y_quality_true)
    valid_indices = np.array(valid_indices)

    X_quality = X_features[valid_indices]
    groups_quality = np.array([all_segments[i]['file'] for i in valid_indices])

    # 3. 5-Fold GroupKFold Cross Validation for ML Quality Classifier
    gkf = GroupKFold(n_splits=5)
    ml_quality_preds = np.zeros(len(y_quality_true), dtype=int)

    for train_idx, test_idx in gkf.split(X_quality, y_quality_true, groups=groups_quality):
        X_tr, y_tr = X_quality[train_idx], y_quality_true[train_idx]
        X_te = X_quality[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        weights = compute_sample_weight('balanced', y_tr)
        clf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
        clf.fit(X_tr_s, y_tr, sample_weight=weights)

        ml_quality_preds[test_idx] = clf.predict(X_te_s)

    # 4. Display Independent Metrics
    print("\n" + "=" * 65)
    print("  PARALLEL ENGINE RESULTS")
    print("=" * 65)

    # A. Physics Engine Metrics on Detuned Strings (Fault 1)
    f1_indices = [i for i, s in enumerate(all_segments) if s['category'] in ['Fault_1', 'Fault1'] or 'fault 1' in s['file'].lower()]
    f1_detected = sum(1 for i in f1_indices if phys_detuned[i])
    print(f"\n1. Physics Engine (Tuning Detection):")
    print(f"   • Detuned Strings (Fault 1) Recall : {f1_detected}/{len(f1_indices)} ({f1_detected/len(f1_indices)*100:.1f}%)")

    # B. ML Quality Classifier Metrics (Healthy vs Quality Issues)
    cm_ml = confusion_matrix(y_quality_true, ml_quality_preds, labels=[0, 1])
    acc_ml = accuracy_score(y_quality_true, ml_quality_preds)
    prec_ml = precision_score(y_quality_true, ml_quality_preds)
    rec_ml = recall_score(y_quality_true, ml_quality_preds)
    f1_ml = f1_score(y_quality_true, ml_quality_preds)

    print(f"\n2. ML Classifier (Healthy vs Structural Quality Issues):")
    print(f"   • Accuracy               : {acc_ml * 100:.2f}%")
    print(f"   • Precision              : {prec_ml * 100:.2f}%")
    print(f"   • Recall (Quality Issues): {rec_ml * 100:.2f}%")
    print(f"   • F1 Score               : {f1_ml * 100:.2f}%")

    print("\n   ML Quality Confusion Matrix:")
    df_cm = pd.DataFrame(cm_ml, index=["True: Healthy", "True: Quality Issue"],
                                columns=["Pred: Healthy", "Pred: Quality Issue"])
    print(df_cm.to_string())

    rep = classification_report(y_quality_true, ml_quality_preds, target_names=["Healthy", "Quality Issue"], digits=4)
    print("\n" + rep)
    print("=" * 65)

if __name__ == "__main__":
    main()
