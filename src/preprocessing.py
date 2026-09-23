"""
Preprocessing pipeline for PCOS ultrasound images.

Pipeline (matches paper Sections III-B/C/F):
    1. Read image, convert to RGB.
    2. Apply the Fuzzy-Inference-System noise-aware enhancement module.
    3. Resize to the target CNN input size.
    4. Normalize to [0, 1] (per-backbone preprocessing is applied later,
       right before it enters EfficientNetB7 / DenseNet201).
"""

import cv2
import numpy as np

from src.fuzzy_enhancement import enhance_image

IMG_SIZE_DEFAULT = 128  # kept modest for CPU-only demo; use 224+ for real training


def load_and_preprocess(path: str, img_size: int = IMG_SIZE_DEFAULT,
                         use_fuzzy_enhancement: bool = True) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return preprocess_array(image, img_size=img_size, use_fuzzy_enhancement=use_fuzzy_enhancement)


def preprocess_array(image_bgr: np.ndarray, img_size: int = IMG_SIZE_DEFAULT,
                      use_fuzzy_enhancement: bool = True) -> np.ndarray:
    if use_fuzzy_enhancement:
        image_bgr = enhance_image(image_bgr)
    image_bgr = cv2.resize(image_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32)


def simple_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation: horizontal flip, small rotation, brightness jitter.
    Kept simple and dependency-free (no external augmentation library)."""
    img = image.copy()

    if rng.random() < 0.5:
        img = np.fliplr(img)

    if rng.random() < 0.5:
        angle = rng.uniform(-15, 15)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    if rng.random() < 0.5:
        factor = rng.uniform(0.85, 1.15)
        img = np.clip(img * factor, 0, 255)

    return img.astype(np.float32)
