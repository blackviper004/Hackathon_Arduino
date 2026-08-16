import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_DIR = "/Users/alen/.gemini/antigravity-ide/brain/f42c51f4-ae95-4fe6-be71-299b4e087a90"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Physics Engine File-Level Confusion Matrix (Open-String Tuning Task)
# True In-Tune (Phase1 Healthy: 56 files): 52 Pred In-Tune, 4 Pred Detuned
# True Detuned (Fault 1: 11 files): 11 Pred Detuned, 0 Pred In-Tune
cm_phys_file = np.array([[52, 4],
                         [0, 11]])
labels_tuning = ['In-Tune\n(Healthy Strings)', 'Detuned\n(Flat / Sharp)']

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
im = ax.imshow(cm_phys_file, interpolation='nearest', cmap=plt.cm.Greens)
plt.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(labels_tuning)))
ax.set_yticks(np.arange(len(labels_tuning)))
ax.set_xticklabels(labels_tuning, fontsize=11, weight='bold')
ax.set_yticklabels(labels_tuning, fontsize=11, weight='bold')

cm_norm = cm_phys_file.astype('float') / cm_phys_file.sum(axis=1)[:, np.newaxis] * 100

for i in range(2):
    for j in range(2):
        count = cm_phys_file[i, j]
        pct = cm_norm[i, j]
        color = "white" if count > 30 else "black"
        ax.text(j, i - 0.1, f"{count} files", ha='center', va='center', color=color, fontsize=18, weight='bold')
        ax.text(j, i + 0.15, f"({pct:.1f}%)", ha='center', va='center', color=color, fontsize=11, weight='semibold')

ax.set_title("Physics Pitch Engine: Open-String Tuning Matrix\nOverall Tuning Accuracy: 94.0% (63/67 files)", fontsize=13, weight='bold', pad=15)
ax.set_xlabel("Predicted Tuning Status", fontsize=12, weight='bold', labelpad=10)
ax.set_ylabel("True Ground Truth Tuning Status", fontsize=12, weight='bold', labelpad=10)
plt.tight_layout()

img_path = os.path.join(ARTIFACT_DIR, "physics_tuning_confusion_matrix.png")
plt.savefig(img_path, dpi=300)
plt.close()
print(f"Saved: {img_path}")
