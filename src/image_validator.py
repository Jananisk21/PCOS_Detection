"""
Ultrasound image validation module.

Detects whether an uploaded/provided image resembles a genuine medical ultrasound
sonography scan (grayscale B-mode scan) or is an out-of-distribution (OOD) image
such as a colored diagram, photo, chart, or synthetic graphic.
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np


def validate_ultrasound_image(image_bgr: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates whether the provided BGR image has the physical and visual characteristics
    of a medical B-mode ultrasound sonography scan.

    Returns:
        is_valid (bool): True if image passes ultrasound sanity checks.
        message (str): Human-readable explanation if rejected or confirmed.
        metrics (dict): Extracted image metrics for diagnostic display.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, "Empty or invalid image data.", {}

    h, w = image_bgr.shape[:2]
    if h < 32 or w < 32:
        return False, f"Image dimensions too small ({w}x{h}).", {}

    # 1. Color and Saturation Analysis (B-mode ultrasound is strictly grayscale)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_sat = float(np.mean(saturation))
    high_sat_ratio = float(np.mean(saturation > 40))

    # Channel variance across R, G, B channels
    channel_std = float(np.mean(np.std(image_bgr.astype(np.float32), axis=2)))

    # 2. Contrast and Brightness Distribution
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))
    mean_brightness = float(np.mean(gray))

    metrics = {
        "mean_saturation": round(mean_sat, 2),
        "high_saturation_ratio": round(high_sat_ratio * 100, 2),
        "channel_variance": round(channel_std, 2),
        "contrast_std": round(contrast, 2),
        "mean_brightness": round(mean_brightness, 2),
    }

    # Strict check: High color content (diagrams, color photos, slides)
    if channel_std > 10.0 or mean_sat > 20.0 or high_sat_ratio > 0.08:
        return (
            False,
            "The uploaded image contains prominent colors or graphics and is not an ultrasound scan. "
            "Ovarian sonography scans are grayscale (B-mode) medical images.",
            metrics,
        )

    # Check for blank / washed out / solid color images
    if contrast < 8.0:
        return (
            False,
            "The uploaded image lacks sufficient texture/contrast and appears blank or uniform.",
            metrics,
        )

    if mean_brightness > 245.0:
        return (
            False,
            "The image is almost completely white/blank (e.g. document/canvas).",
            metrics,
        )

    if mean_brightness < 5.0:
        return (
            False,
            "The image is almost completely black/empty.",
            metrics,
        )

    return True, "Valid ultrasound sonography scan format.", metrics
