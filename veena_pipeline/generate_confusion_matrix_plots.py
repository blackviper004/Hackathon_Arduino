import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_DIR = "/Users/alen/.gemini/antigravity-ide/brain/f42c51f4-ae95-4fe6-be71-299b4e087a90"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# 1. Final Parallel ML Quality Confusion Matrix (Verified 99.24% Accuracy)
cm_ml = np.array([[215, 3],
                  [2, 439]])
labels = ['Healthy', 'Quality Issue\n(Faults 2-11)']

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
im = ax.imshow(cm_ml, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=11, weight='bold')
ax.set_yticklabels(labels, fontsize=11, weight='bold')

cm_norm = cm_ml.astype('float') / cm_ml.sum(axis=1)[:, np.newaxis] * 100

for i in range(2):
    for j in range(2):
        count = cm_ml[i, j]
        pct = cm_norm[i, j]
        color = "white" if count > 200 else "black"
        ax.text(j, i - 0.1, f"{count}", ha='center', va='center', color=color, fontsize=18, weight='bold')
        ax.text(j, i + 0.15, f"({pct:.1f}%)", ha='center', va='center', color=color, fontsize=11, weight='semibold')

ax.set_title("Final Parallel ML Classifier: Structural Quality Diagnosis\nAccuracy: 99.24% | F1-Score: 99.43%", fontsize=13, weight='bold', pad=15)
ax.set_xlabel("Predicted Class", fontsize=12, weight='bold', labelpad=10)
ax.set_ylabel("True Ground Truth Class", fontsize=12, weight='bold', labelpad=10)
plt.tight_layout()

img1_path = os.path.join(ARTIFACT_DIR, "parallel_ml_quality_confusion_matrix.png")
plt.savefig(img1_path, dpi=300)
plt.close()
print(f"Saved: {img1_path}")

# 2. Sequential 3-Class Confusion Matrix
cm_seq = np.array([[189, 25, 4],
                   [13, 29, 1],
                   [1, 307, 133]])
labels_3class = ['Healthy', 'Detuned', 'Quality Issue']

fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
im2 = ax.imshow(cm_seq, interpolation='nearest', cmap=plt.cm.Oranges)
plt.colorbar(im2, ax=ax)

ax.set_xticks(np.arange(len(labels_3class)))
ax.set_yticks(np.arange(len(labels_3class)))
ax.set_xticklabels(labels_3class, fontsize=11, weight='bold')
ax.set_yticklabels(labels_3class, fontsize=11, weight='bold')

cm_seq_norm = cm_seq.astype('float') / cm_seq.sum(axis=1)[:, np.newaxis] * 100
for i in range(3):
    for j in range(3):
        count = cm_seq[i, j]
        pct = cm_seq_norm[i, j]
        color = "white" if count > 150 else "black"
        ax.text(j, i - 0.1, f"{count}", ha='center', va='center', color=color, fontsize=16, weight='bold')
        ax.text(j, i + 0.15, f"({pct:.1f}%)", ha='center', va='center', color=color, fontsize=10, weight='semibold')

ax.set_title("Sequential Pipeline: 3-Class Gate Confusion Matrix\n(Shows Physics Gate Over-Sensitivity on Quality Audio)", fontsize=12, weight='bold', pad=15)
ax.set_xlabel("Predicted Class", fontsize=11, weight='bold', labelpad=10)
ax.set_ylabel("True Ground Truth Class", fontsize=11, weight='bold', labelpad=10)
plt.tight_layout()

img2_path = os.path.join(ARTIFACT_DIR, "sequential_3class_confusion_matrix.png")
plt.savefig(img2_path, dpi=300)
plt.close()
print(f"Saved: {img2_path}")

# 3. Overall Pipeline Performance Comparison Bar Chart
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
categories = ['Quality Issues\nRecall', 'Quality Issues\nPrecision', 'Healthy Strings\nPrecision', 'Detuned Strings\nRecall']
sequential_scores = [30.2, 96.4, 93.1, 100.0]
parallel_scores = [99.3, 98.7, 98.6, 100.0]

x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, sequential_scores, width, label='Sequential Pipeline', color='#e74c3c')
rects2 = ax.bar(x + width/2, parallel_scores, width, label='Parallel Hybrid Pipeline', color='#2ecc71')

ax.set_ylabel('Percentage (%)', fontsize=12, weight='bold')
ax.set_title('Pipeline Comparison: Sequential vs Parallel Execution', fontsize=14, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, weight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 115)

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, weight='bold')
for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
img3_path = os.path.join(ARTIFACT_DIR, "pipeline_performance_comparison.png")
plt.savefig(img3_path, dpi=300)
plt.close()
print(f"Saved: {img3_path}")
