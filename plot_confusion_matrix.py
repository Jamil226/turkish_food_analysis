"""
plot_confusion_matrix.py
Generates a high-quality, light-theme confusion matrix from outputs/<model>/predictions.csv

Usage:
  python plot_confusion_matrix.py --model efficientnet
  python plot_confusion_matrix.py --model inception
  python plot_confusion_matrix.py --model resnet50
"""

import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

matplotlib.rcParams["font.family"] = "DejaVu Sans"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="efficientnet", help="Model subfolder name")
args = parser.parse_args()

MODEL = args.model

# =====================================================
# PATHS
# =====================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs", MODEL)

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
OUTPUT_PNG      = os.path.join(OUTPUT_DIR, "confusion_matrix_hq.png")

# =====================================================
# LOAD PREDICTIONS
# =====================================================

df = pd.read_csv(PREDICTIONS_CSV)

true_labels = df["true_class"].tolist()
pred_labels = df["predicted_class"].tolist()
classes = sorted(set(true_labels))

print(f"Classes  : {len(classes)}")
print(f"Samples  : {len(true_labels)}")

# =====================================================
# COMPUTE CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(true_labels, pred_labels, labels=classes)

# Normalize row-wise (recall per class) for better readability
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# =====================================================
# PLOT
# =====================================================

N = len(classes)
fig, ax = plt.subplots(figsize=(22, 20))

# Light theme — white background, Blues colormap
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.tick_params(labelsize=11)
cbar.set_label("Recall (row-normalized)", fontsize=12, labelpad=10)

# Tick labels
ax.set_xticks(np.arange(N))
ax.set_yticks(np.arange(N))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9.5)
ax.set_yticklabels(classes, fontsize=9.5)

# Grid lines between cells
ax.set_xticks(np.arange(N + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
ax.grid(which="minor", color="lightgrey", linewidth=0.4)
ax.tick_params(which="minor", length=0)

# Annotate cells with raw counts
for i in range(N):
    for j in range(N):
        val = cm[i, j]
        text_color = "white" if cm_norm[i, j] > 0.60 else "black"
        ax.text(
            j, i,
            str(val),
            ha="center", va="center",
            fontsize=7.5,
            color=text_color,
            fontweight="bold" if i == j else "normal",
        )

# Labels & title
ax.set_xlabel("Predicted Class", fontsize=13, labelpad=12)
ax.set_ylabel("True Class", fontsize=13, labelpad=12)
ax.set_title(
    f"Confusion Matrix — Turkish Food Classification ({MODEL.replace('_', '-').title()})",
    fontsize=14,
    fontweight="bold",
    pad=16,
)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUTPUT_PNG}")
plt.close()
