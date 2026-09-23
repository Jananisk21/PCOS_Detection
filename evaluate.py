"""
Evaluates the trained ensemble on the held-out test split saved by
train.py, reporting the same metrics the base paper uses to compare
approaches in Table 1 (accuracy, AUC-ROC), plus a confusion matrix and
full classification report.

Usage:
    python evaluate.py --model_path saved_models/pcos_effidensegenop_model.keras \
        --test_split saved_models/test_split.npz
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

from src.dataset import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saved_models/pcos_effidensegenop_model.keras")
    parser.add_argument("--test_split", default="saved_models/test_split.npz")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    data = np.load(args.test_split)
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"Loaded {len(X_test)} held-out test images.")

    y_prob = model.predict(X_test, batch_size=16).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nTest Accuracy : {acc * 100:.2f}%")
    print(f"Test AUC-ROC  : {auc * 100:.2f}%")

    print("\nClassification report:")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    with open(os.path.join(args.output_dir, "evaluation_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Test AUC-ROC: {auc * 100:.2f}%\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(str(cm) + "\n")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
    ax2.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=150)

    print(f"\nSaved evaluation_report.txt, confusion_matrix.png, roc_curve.png to {args.output_dir}/")


if __name__ == "__main__":
    main()
