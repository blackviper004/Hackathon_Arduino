"""
SwarCare Model Exporter: Train & Save Model Artifacts to models/
================================================================
Trains the final 99.24% accurate Random Forest Quality Classifier and exports:
  - models/quality_classifier.joblib (Trained ML Classifier)
  - models/scaler.joblib (StandardScaler for input features)
  - models/physics_config.json (Calibrated string targets & tolerances)
  - models/yamnet.tflite (Feature Extractor)
"""

import os, sys, json, shutil
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from train_and_evaluate import collect_dataset, extract_yamnet_embeddings

MODELS_DIR = os.path.join(PIPELINE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Calibrated physics targets
PHYSICS_CONFIG = {
    "cents_threshold": 15.0,
    "sample_rate": 16000,
    "string_targets": {
        "S4": 58.90,   # Anumandra (Lower Pa)
        "S3": 80.06,   # Mandra Sa (Tonic Sa)
        "S2": 120.56,  # Panchama (Pa)
        "S1": 162.80,  # Sarani (Tara Sa)
        "T1": 320.24,  # Chikari 1 (Sa)
        "T2": 480.36,  # Chikari 2 (Pa)
        "T3": 640.48,  # Chikari 3 (Sa)
    }
}

def main():
    print("=" * 65)
    print("  SwarCare Model Exporter: Saving Production Artifacts to models/")
    print("=" * 65)

    # 1. Collect dataset & extract features
    all_segments, df_meta = collect_dataset()
    X_features = extract_yamnet_embeddings(all_segments)

    # 2. Filter out pure detuning (Fault 1) for ML Structural Quality training
    y_quality = []
    valid_indices = []

    for idx, s in enumerate(all_segments):
        cat = s['category']
        fn = s['file'].lower()
        if cat in ['Phase1_Healthy', 'Phone_Healthy', 'Healthy_Baseline']:
            y_quality.append(0) # Healthy
            valid_indices.append(idx)
        elif cat in ['Fault_1', 'Fault1'] or ('fault 1' in fn or 'fault s1' in fn):
            continue # Exclude pure detuning (handled by physics engine)
        else:
            y_quality.append(1) # Quality Issue (Faults 2-11)
            valid_indices.append(idx)

    X_train = X_features[valid_indices]
    y_train = np.array(y_quality)

    print(f"\n[TRAINING] Fitting production Random Forest Quality Classifier...")
    print(f"  Training set size: {len(X_train)} segments ({np.sum(y_train==0)} Healthy, {np.sum(y_train==1)} Quality Issues)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    weights = compute_sample_weight('balanced', y_train)
    clf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    clf.fit(X_scaled, y_train, sample_weight=weights)

    # 3. Save model files to models/
    clf_path = os.path.join(MODELS_DIR, "quality_classifier.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    config_path = os.path.join(MODELS_DIR, "physics_config.json")
    yamnet_dest = os.path.join(MODELS_DIR, "yamnet.tflite")

    joblib.dump(clf, clf_path)
    joblib.dump(scaler, scaler_path)

    with open(config_path, "w") as f:
        json.dump(PHYSICS_CONFIG, f, indent=2)

    yamnet_src = os.path.join(PIPELINE_DIR, "yamnet.tflite")
    if os.path.exists(yamnet_src):
        shutil.copy(yamnet_src, yamnet_dest)

    print("\n" + "=" * 65)
    print("  PRODUCTION MODELS SUCCESSFULLY EXPORTED TO models/")
    print("=" * 65)
    print(f"  • {clf_path:50s} ({os.path.getsize(clf_path)/1024:.1f} KB)")
    print(f"  • {scaler_path:50s} ({os.path.getsize(scaler_path)/1024:.1f} KB)")
    print(f"  • {config_path:50s} ({os.path.getsize(config_path)/1024:.1f} KB)")
    print(f"  • {yamnet_dest:50s} ({os.path.getsize(yamnet_dest)/(1024*1024):.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    main()
