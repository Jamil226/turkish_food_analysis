"""
generate_report.py
Generates a classification report with metrics formatted as percentages (e.g. 91.23%)
from outputs/<model>/predictions.csv

Usage:
  python generate_report.py --model efficientnet
  python generate_report.py --model inception
  python generate_report.py --model resnet50
"""

import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
OUTPUT_TXT      = os.path.join(OUTPUT_DIR, "classification_report_pct.txt")

df = pd.read_csv(PREDICTIONS_CSV)
true_labels = df["true_class"].tolist()
pred_labels = df["predicted_class"].tolist()
classes     = sorted(set(true_labels))

# =====================================================
# COMPUTE
# =====================================================

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, pred_labels, labels=classes, zero_division=0
)

macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="macro", zero_division=0
)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    true_labels, pred_labels, average="weighted", zero_division=0
)

accuracy = accuracy_score(true_labels, pred_labels)
total    = len(true_labels)

# =====================================================
# FORMAT
# =====================================================

col_w = max(len(c) for c in classes) + 2

header = (
    f"{'Class':<{col_w}}  {'Precision':>10}  {'Recall':>10}  {'F1-Score':>10}  {'Support':>8}"
)
sep    = "-" * len(header)

lines = [header, sep]

for cls, p, r, f, s in zip(classes, precision, recall, f1, support):
    lines.append(
        f"{cls:<{col_w}}  {p*100:>9.2f}%  {r*100:>9.2f}%  {f*100:>9.2f}%  {int(s):>8}"
    )

lines.append(sep)
lines.append(f"{'Accuracy':<{col_w}}  {'':>10}  {'':>10}  {accuracy*100:>9.2f}%  {total:>8}")
lines.append(f"{'Macro avg':<{col_w}}  {macro_p*100:>9.2f}%  {macro_r*100:>9.2f}%  {macro_f1*100:>9.2f}%  {total:>8}")
lines.append(f"{'Weighted avg':<{col_w}}  {weighted_p*100:>9.2f}%  {weighted_r*100:>9.2f}%  {weighted_f1*100:>9.2f}%  {total:>8}")

report = "\n".join(lines)

# =====================================================
# PRINT & SAVE
# =====================================================

print(report)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(report + "\n")

print(f"\nSaved → {OUTPUT_TXT}")
