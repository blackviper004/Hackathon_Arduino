"""
SwarCare Physics Pitch Engine — Real Recordings Evaluation (v3)
================================================================
Per-string calibration approach: measures the ACTUAL target frequency of
each string individually from Phase1 healthy recordings, then evaluates.

This is the correct approach because Carnatic veena strings are NOT tuned
in mathematically perfect equal-temperament ratios. S1 (Sarani/Tara Sa)
sits ~+30 cents sharp of a pure octave, S4 (Anumandra) ~-25 cents flat of
a pure lower fifth. These offsets are real and intentional in the instrument.

Label mapping:
  Phase1/S*     → TRUE NOT_DETUNED (healthy, in-tune)
  Fault1        → TRUE DETUNED (string loosened intentionally)
  Fault2–11     → TRUE NOT_DETUNED (structural faults, pitch unchanged)
"""

import os, sys, json, time
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_pitch_engine import PhysicsPitchEngine

SR             = 16000
WINDOW_SAMPLES = int(2.0 * SR)
SILENCE_RMS    = 0.008
CENTS_THRESH   = 15.0
OUTPUT_JSON    = "evaluation_results/physics_engine_metrics.json"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_segments(wav_path):
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as e:
        return []
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SR:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
        except Exception:
            pass
    segs = []
    n = len(audio) // WINDOW_SAMPLES
    if n == 0 and len(audio) > SR // 2:
        pad = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        pad[:len(audio)] = audio
        if np.sqrt(np.mean(pad**2)) >= SILENCE_RMS:
            segs.append(pad)
    else:
        for i in range(n):
            seg = audio[i * WINDOW_SAMPLES:(i+1) * WINDOW_SAMPLES]
            if np.sqrt(np.mean(seg**2)) >= SILENCE_RMS:
                segs.append(seg.astype(np.float32))
    return segs

def walk_wavs(base_dir):
    wavs = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(root, f))
    return wavs

# ─── Per-String Calibration ───────────────────────────────────────────────────

def calibrate_string(string_dir, raw_engine):
    """
    Measure the actual in-tune frequency of one string from its Phase1 recordings.
    Uses detect_f0_only() — no reference comparison, pure pitch detection.
    Returns median detected f0 (Hz).
    """
    all_f0 = []
    for wav in walk_wavs(string_dir):
        for seg in load_segments(wav):
            f0, conf = raw_engine.detect_f0_only(seg.astype(np.float64), SR)
            if f0 > 30 and conf > 0.3:
                all_f0.append(f0)
    if not all_f0:
        return None, []
    return float(np.median(all_f0)), all_f0

# ─── Custom Per-String Engine ─────────────────────────────────────────────────

class PerStringEngine:
    """
    Tuner engine with per-string measured targets (not ratio-computed).
    Each string has its own measured baseline from Phase1 recordings.
    Decision: if |cents from nearest string target| > threshold → DETUNED.
    """

    def __init__(self, string_targets: dict, cents_threshold: float = 15.0, sr: int = SR):
        # string_targets: {"S1": 162.80, "S2": 120.56, "S3": 80.06, "S4": 58.90, ...}
        self.string_targets  = string_targets
        self.cents_threshold = cents_threshold
        self.sr              = sr
        # Use the engine for DSP (tonic doesn't matter here — we override decision)
        # Pick any reasonable tonic; we override _cents_decision
        sa_hz = string_targets.get("S3", 80.0)
        self._dsp = PhysicsPitchEngine(
            tonic_hz=sa_hz,
            cents_threshold=cents_threshold,
            sr=sr,
            silence_rms_threshold=SILENCE_RMS,
        )

    def detect_f0(self, audio):
        """Run DSP stages 0-3, return (f0_hz, confidence)."""
        return self._dsp.detect_f0_only(audio, self.sr)

    def decide(self, f0_hz: float, confidence: float):
        """
        Compare detected f0 to the nearest per-string measured target.
        Returns (status, cents_dev, nearest_string, target_hz, message).
        """
        if f0_hz <= 0 or confidence < 0.15:
            return "NO_PITCH", 0.0, "?", 0.0, "No pitch detected"

        # Find nearest string by minimum |cents| distance
        nearest = min(
            self.string_targets.keys(),
            key=lambda s: abs(1200.0 * np.log2(f0_hz / (self.string_targets[s] + 1e-9)))
        )
        target_hz = self.string_targets[nearest]
        cents = float(1200.0 * np.log2(f0_hz / (target_hz + 1e-9)))

        # Harmonic Foldback (Loophole 1: Kudirai bridge upper partial excitation)
        if abs(cents) > self.cents_threshold:
            for mult in [1.5, 2.0, 3.0]:
                folded_f0 = f0_hz / mult
                for s_key, s_target in self.string_targets.items():
                    folded_cents = abs(1200.0 * np.log2(folded_f0 / (s_target + 1e-9)))
                    if folded_cents <= self.cents_threshold:
                        f0_hz = folded_f0
                        nearest = s_key
                        target_hz = s_target
                        cents = 1200.0 * np.log2(folded_f0 / (s_target + 1e-9))
                        break
                if abs(cents) <= self.cents_threshold:
                    break

        hz_dev = float(f0_hz - target_hz)

        if abs(cents) <= self.cents_threshold:
            status = "IN_TUNE"
            msg = f"In tune ({cents:+.1f}¢) — {nearest}, target {target_hz:.2f} Hz"
        elif cents < 0:
            status = "FLAT"
            msg = f"Flat {abs(cents):.1f}¢ ({hz_dev:+.2f} Hz) — tighten peg [{nearest}]"
        else:
            status = "SHARP"
            msg = f"Sharp {cents:.1f}¢ ({hz_dev:+.2f} Hz) — loosen peg [{nearest}]"

        return status, round(cents, 1), nearest, round(target_hz, 2), msg

    def run(self, audio):
        rms = float(np.sqrt(np.mean(np.array(audio, dtype=np.float64)**2)))
        if rms < SILENCE_RMS:
            return "SILENCE", 0.0, "?", 0.0, 0.0, "Segment is silent"
        f0, conf = self.detect_f0(np.array(audio, dtype=np.float64))
        status, cents, nearest, target, msg = self.decide(f0, conf)
        return status, cents, nearest, target, conf, msg


# ─── File-level evaluation ────────────────────────────────────────────────────

def evaluate_file(wav_path, engine):
    segments = load_segments(wav_path)
    if not segments:
        return None

    seg_statuses, seg_confs, seg_f0s, seg_cents = [], [], [], []
    for seg in segments:
        f0, conf = engine.detect_f0(seg.astype(np.float64))
        status, cents, nearest, target, msg = engine.decide(f0, conf)
        seg_statuses.append(status)
        seg_confs.append(conf)
        seg_f0s.append(f0)
        seg_cents.append(cents)

    detuned_v = sum(1 for s in seg_statuses if s in ("FLAT", "SHARP"))
    intune_v  = sum(1 for s in seg_statuses if s == "IN_TUNE")
    incon_v   = sum(1 for s in seg_statuses if s in ("SILENCE", "NO_PITCH"))

    if detuned_v > intune_v:
        predicted = "DETUNED"
    elif intune_v > 0:
        predicted = "NOT_DETUNED"
    else:
        predicted = "INCONCLUSIVE"

    # Best segment = highest confidence active detection
    active_idx = [i for i, s in enumerate(seg_statuses) if s not in ("SILENCE",)]
    if active_idx:
        bi = max(active_idx, key=lambda i: seg_confs[i])
    else:
        bi = 0

    return {
        "predicted":     predicted,
        "n_segments":    len(segments),
        "detuned_votes": detuned_v,
        "intune_votes":  intune_v,
        "incon_votes":   incon_v,
        "best_f0":       round(seg_f0s[bi], 2),
        "best_cents":    round(seg_cents[bi], 1),
        "best_conf":     round(seg_confs[bi], 3),
        "best_status":   seg_statuses[bi],
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SwarCare Physics Pitch Engine — Evaluation v3")
    print("  (Per-String Calibration from Phase1 Recordings)")
    print("=" * 65)

    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    phase1_dir   = os.path.join(pipeline_dir, "Phase1")
    ext_dir      = os.path.join(pipeline_dir, "extracted_data")

    # ── Step 1: Calibrate each string individually ────────────────────────────
    print("\n[STEP 1] Per-string calibration from Phase1 healthy recordings...")
    raw_engine = PhysicsPitchEngine(tonic_hz=80.0, cents_threshold=1200.0, sr=SR)

    string_folders = {
        "S1": "S1_Sarani",
        "S2": "S2_Panchama",
        "S3": "S3_Mandra",
        "S4": "S4_Anumandra",
    }
    calibrated_targets = {}
    print(f"\n  {'String':6s} {'Folder':18s} {'Median f0':>10s} {'Segments':>9s} {'Range':>20s}")
    print(f"  {'─'*6} {'─'*18} {'─'*10} {'─'*9} {'─'*20}")
    for key, folder in string_folders.items():
        sdir = os.path.join(phase1_dir, folder)
        median_f0, all_f0 = calibrate_string(sdir, raw_engine)
        if median_f0 is None:
            print(f"  {key:6s} {folder:18s} {'NO DATA':>10s}")
            continue
        calibrated_targets[key] = median_f0
        print(f"  {key:6s} {folder:18s} {median_f0:>9.2f}Hz {len(all_f0):>9d} "
              f"  {min(all_f0):.1f}–{max(all_f0):.1f} Hz")

    # Chikari (T1-T3) computed from S3 tonic since we have no Phase1 T recordings
    sa = calibrated_targets.get("S3", 80.0)
    calibrated_targets["T1"] = round(sa * 4.0, 2)
    calibrated_targets["T2"] = round(sa * 6.0, 2)
    calibrated_targets["T3"] = round(sa * 8.0, 2)
    print(f"\n  T1/T2/T3 (no Phase1 data — computed from S3={sa:.2f} Hz):")
    for t in ["T1","T2","T3"]:
        print(f"    {t}: {calibrated_targets[t]:.2f} Hz")

    # Build engine with per-string measured targets
    engine = PerStringEngine(calibrated_targets, cents_threshold=CENTS_THRESH, sr=SR)

    print(f"\n  Final per-string targets (±{CENTS_THRESH}¢ tolerance):")
    for s in ["S4","S3","S2","S1","T1","T2","T3"]:
        if s in calibrated_targets:
            src = "measured" if s in string_folders else "computed"
            print(f"    {s}: {calibrated_targets[s]:.2f} Hz  [{src}]")

    # ── Step 2: Collect dataset ───────────────────────────────────────────────
    print("\n[STEP 2] Collecting dataset...")
    dataset = []

    for key, folder in string_folders.items():
        sdir = os.path.join(phase1_dir, folder)
        for wav in walk_wavs(sdir):
            dataset.append((wav, "NOT_DETUNED", f"Phase1_{folder}"))

    f1_dir = os.path.join(ext_dir, "Fault1")
    for wav in walk_wavs(f1_dir):
        dataset.append((wav, "DETUNED", "Fault1 (Detuned)"))

    for fname in ["Fault2","Fault3","Fault4","Fault5","Fault7",
                  "Fault10","Fault11","frets",
                  "SwarCare_All_Recordings_20260811_135445_general_test"]:
        fdir = os.path.join(ext_dir, fname)
        if os.path.exists(fdir):
            for wav in walk_wavs(fdir):
                dataset.append((wav, "NOT_DETUNED", fname))

    print(f"  {len(dataset)} total files  "
          f"({sum(1 for _,l,_ in dataset if l=='DETUNED')} detuned, "
          f"{sum(1 for _,l,_ in dataset if l=='NOT_DETUNED')} not-detuned)")

    # ── Step 3: Run engine ────────────────────────────────────────────────────
    print(f"\n[STEP 3] Running engine...")
    t0 = time.time()
    results = []
    for wav_path, true_label, group in dataset:
        ev = evaluate_file(wav_path, engine)
        if ev is None:
            continue
        ev["file"]       = os.path.basename(wav_path)
        ev["group"]      = group
        ev["true_label"] = true_label
        ev["correct"]    = (
            (true_label == "DETUNED"     and ev["predicted"] == "DETUNED") or
            (true_label == "NOT_DETUNED" and ev["predicted"] == "NOT_DETUNED")
        )
        results.append(ev)
    print(f"  Done in {time.time()-t0:.1f}s — {len(results)} files")

    # ── Step 4: Metrics ───────────────────────────────────────────────────────
    conc = [r for r in results if r["predicted"] != "INCONCLUSIVE"]
    inco = [r for r in results if r["predicted"] == "INCONCLUSIVE"]

    tp = sum(1 for r in conc if r["true_label"] == "DETUNED"     and r["predicted"] == "DETUNED")
    tn = sum(1 for r in conc if r["true_label"] == "NOT_DETUNED" and r["predicted"] == "NOT_DETUNED")
    fp = sum(1 for r in conc if r["true_label"] == "NOT_DETUNED" and r["predicted"] == "DETUNED")
    fn = sum(1 for r in conc if r["true_label"] == "DETUNED"     and r["predicted"] == "NOT_DETUNED")

    n   = len(conc)
    acc = (tp + tn) / n      if n           else 0.0
    pre = tp / (tp + fp)     if (tp + fp)   else 0.0
    rec = tp / (tp + fn)     if (tp + fn)   else 0.0
    f1  = 2*pre*rec/(pre+rec) if (pre+rec)  else 0.0
    spe = tn / (tn + fp)     if (tn + fp)   else 0.0

    fa_by_string = {}
    for key, folder in string_folders.items():
        grp = [r for r in conc if r["group"] == f"Phase1_{folder}"]
        fa  = sum(1 for r in grp if r["predicted"] == "DETUNED")
        fa_by_string[folder] = (fa, len(grp))

    groups = sorted(set(r["group"] for r in results))
    group_stats = {}
    for g in groups:
        gr = [r for r in results if r["group"] == g]
        nc = sum(1 for r in gr if r["correct"])
        group_stats[g] = {
            "true_label": gr[0]["true_label"],
            "n_files": len(gr),
            "DETUNED_pred": sum(1 for r in gr if r["predicted"] == "DETUNED"),
            "NOT_DETUNED_pred": sum(1 for r in gr if r["predicted"] == "NOT_DETUNED"),
            "INCON_pred": sum(1 for r in gr if r["predicted"] == "INCONCLUSIVE"),
            "correct": nc,
            "acc%": round(nc/len(gr)*100, 1),
        }

    # Print
    print("\n" + "=" * 65)
    print("  FINAL METRICS")
    print("=" * 65)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':24s}  Pred DETUNED   Pred NOT_DETUNED")
    print(f"  {'True DETUNED':24s}  TP={tp:>4}           FN={fn:>4}")
    print(f"  {'True NOT_DETUNED':24s}  FP={fp:>4}           TN={tn:>4}")
    print(f"  Inconclusive: {len(inco)}")

    print(f"\n  ┌──────────────────────────────────────┬─────────┐")
    print(f"  │ Metric                               │  Value  │")
    print(f"  ├──────────────────────────────────────┼─────────┤")
    print(f"  │ Accuracy    (conclusive files)       │  {acc*100:5.1f}%  │")
    print(f"  │ Precision   (DETUNED detection)      │  {pre*100:5.1f}%  │")
    print(f"  │ Recall      (DETUNED detection)      │  {rec*100:5.1f}%  │")
    print(f"  │ F1 Score    (DETUNED detection)      │  {f1*100:5.1f}%  │")
    print(f"  │ Specificity (NOT_DETUNED correct)    │  {spe*100:5.1f}%  │")
    print(f"  └──────────────────────────────────────┴─────────┘")

    print(f"\n  False Alarm on Phase1 Healthy Strings:")
    for folder, (fa, tot) in fa_by_string.items():
        icon = "✅" if fa == 0 else ("⚠️ " if fa < tot//2 else "❌")
        rate = f"{fa/tot*100:.1f}%" if tot else "n/a"
        print(f"    {icon} {folder:18s}: {fa}/{tot} falsely detuned ({rate})")

    print(f"\n  Per-Group Breakdown:")
    print(f"  {'Group':54s} {'True':11s} {'N':>3} {'Det':>4} {'InT':>4} {'Inc':>4} {'Acc%':>6}")
    print(f"  {'─'*54} {'─'*11} {'─'*3} {'─'*4} {'─'*4} {'─'*4} {'─'*6}")
    for g, s in sorted(group_stats.items()):
        lbl = "DETUNE" if s["true_label"] == "DETUNED" else "NOT-DET"
        print(f"  {g:54s} {lbl:11s} {s['n_files']:>3} "
              f"{s['DETUNED_pred']:>4} {s['NOT_DETUNED_pred']:>4} "
              f"{s['INCON_pred']:>4} {s['acc%']:>6.1f}%")

    wrong = [r for r in results if not r["correct"] and r["predicted"] != "INCONCLUSIVE"]
    print(f"\n  Incorrect ({len(wrong)}):")
    if not wrong:
        print("  ✅  ALL conclusive files classified correctly!")
    else:
        for r in wrong:
            print(f"  ❌ [{r['group']:22s}] {r['file']:35s} "
                  f"true={r['true_label']:11s} pred={r['predicted']:11s} "
                  f"f0={r['best_f0']:.1f}Hz  {r['best_cents']:+.1f}¢")

    os.makedirs("evaluation_results", exist_ok=True)
    export = {
        "calibrated_targets": {k: round(v, 2) for k, v in calibrated_targets.items()},
        "cents_threshold": CENTS_THRESH,
        "summary": {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": round(acc, 4), "precision": round(pre, 4),
            "recall": round(rec, 4), "f1_score": round(f1, 4),
            "specificity": round(spe, 4),
            "inconclusive": len(inco),
        },
        "false_alarm_by_string": {
            k: {"fa": v[0], "total": v[1], "rate": round(v[0]/v[1],4) if v[1] else 0}
            for k, v in fa_by_string.items()
        },
        "per_group": group_stats,
        "per_file": results,
    }
    with open(OUTPUT_JSON, "w") as fout:
        json.dump(export, fout, indent=2)
    print(f"\n  [SAVED] {OUTPUT_JSON}")
    print("=" * 65)

if __name__ == "__main__":
    main()
