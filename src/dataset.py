"""
Dataset utilities.

Real usage
----------
Point this at a directory structured as:

    data/raw/
        PCOS/       *.png / *.jpg  (positive ovarian USG images)
        notPCOS/    *.png / *.jpg  (negative ovarian USG images)

This matches the structure of the public Kaggle "PCOS Detection using
Ultrasound Images" dataset referenced by the base paper (Table 1 / base
study: "Kaggle ovary USG dataset, upsampled to 8000 images").

Demo usage
----------
Because this sandbox has no internet access to Kaggle, `generate_synthetic_dataset.py`
creates a small synthetic dataset of ultrasound-like grayscale images
(speckle-textured ovary blobs, with follicle-like ring clusters for the
PCOS class) with the exact same folder layout, so the full pipeline
(preprocessing -> fuzzy enhancement -> GA-tuned ensemble training ->
evaluation -> Grad-CAM -> app) can be run and demonstrated end-to-end.
Swap in the real Kaggle images by dropping them into data/raw/PCOS and
data/raw/notPCOS with the same folder names -- no code changes required.
"""

import glob
import os

import numpy as np

from src.preprocessing import preprocess_array, simple_augment

CLASS_NAMES = ["notPCOS", "PCOS"]  # label 0, 1


def list_image_paths(raw_dir: str):
    paths, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(raw_dir, class_name)
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in glob.glob(os.path.join(class_dir, ext)):
                paths.append(p)
                labels.append(label_idx)
    return paths, np.array(labels)


def load_dataset(raw_dir: str, img_size: int = 128, use_fuzzy_enhancement: bool = True,
                  augment: bool = False, seed: int = 42):
    """Loads every image under raw_dir/{PCOS,notPCOS}, applies the fuzzy
    enhancement + resize pipeline, and returns (X, y)."""
    import cv2

    paths, labels = list_image_paths(raw_dir)
    if len(paths) == 0:
        raise RuntimeError(
            f"No images found under {raw_dir}/PCOS or {raw_dir}/notPCOS. "
            "Run generate_synthetic_dataset.py for a demo dataset, or add real images."
        )

    rng = np.random.default_rng(seed)
    X = np.zeros((len(paths), img_size, img_size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        arr = preprocess_array(img, img_size=img_size, use_fuzzy_enhancement=use_fuzzy_enhancement)
        if augment:
            arr = simple_augment(arr, rng)
        X[i] = arr

    return X, labels


def train_val_test_split(X, y, val_frac=0.15, test_frac=0.15, seed=42):
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_frac + test_frac), stratify=y, random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
