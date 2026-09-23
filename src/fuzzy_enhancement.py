"""
Fuzzy Inference System (FIS) for noise-aware ultrasound image enhancement.

This mirrors the "Image Enhancement" module described in the base paper's
EffiDenseGenOp framework (Section III-F): a Gaussian-Triangular membership
based Fuzzy Inference System that estimates the local noise level of an
ultrasound image and adaptively decides how much smoothing vs. sharpening
to apply, so that speckle noise is suppressed without destroying follicle
boundaries.

Design (Mamdani-style FIS, implemented directly with NumPy so the project
has no extra fuzzy-logic dependency):

Input variable : local noise level (estimated via local variance of the
                 Laplacian, normalized to [0, 1])
    - LOW    : Gaussian membership function centered at 0.0
    - MEDIUM : Triangular membership function centered at 0.5
    - HIGH   : Gaussian membership function centered at 1.0

Output variable: enhancement action, a blend weight in [0, 1] between a
                 denoising (Gaussian/bilateral) filtered image and a
                 sharpened (unsharp-mask) image.
    - LOW noise  -> favor sharpening (output close to 1 = mostly sharpened)
    - MEDIUM     -> balanced blend
    - HIGH noise -> favor denoising (output close to 0 = mostly denoised)

Rules:
    R1: IF noise IS LOW    THEN action IS SHARPEN
    R2: IF noise IS MEDIUM THEN action IS BALANCE
    R3: IF noise IS HIGH   THEN action IS DENOISE

Defuzzification: weighted average (centroid of singletons at 1.0, 0.5, 0.0)
"""

import cv2
import numpy as np


def _gaussian_mf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def _triangular_mf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    left = (x - a) / (b - a + 1e-8)
    right = (c - x) / (c - b + 1e-8)
    return np.clip(np.minimum(left, right), 0, 1)


def estimate_local_noise_map(gray_img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Estimate a normalized (0-1) local noise map using the local variance
    of the Laplacian of the image. High local variance in a supposedly
    smooth ultrasound region indicates speckle noise.
    """
    lap = cv2.Laplacian(gray_img.astype(np.float32), ddepth=cv2.CV_32F, ksize=3)
    lap_sq = lap ** 2
    local_mean_sq = cv2.blur(lap_sq, (ksize, ksize))
    noise_map = np.sqrt(np.maximum(local_mean_sq, 0))
    # normalize to [0, 1]
    denom = (noise_map.max() - noise_map.min()) + 1e-8
    noise_map = (noise_map - noise_map.min()) / denom
    return noise_map


def fuzzy_action_map(noise_map: np.ndarray) -> np.ndarray:
    """
    Apply the Mamdani-style FIS rules per-pixel on the noise map and
    return the defuzzified action map (0 = pure denoise, 1 = pure sharpen).
    """
    low = _gaussian_mf(noise_map, mean=0.0, sigma=0.25)
    high = _gaussian_mf(noise_map, mean=1.0, sigma=0.25)
    medium = _triangular_mf(noise_map, a=0.15, b=0.5, c=0.85)

    # Rule outputs are singleton consequents: SHARPEN=1.0, BALANCE=0.5, DENOISE=0.0
    numerator = low * 1.0 + medium * 0.5 + high * 0.0
    denominator = low + medium + high + 1e-8
    action = numerator / denominator
    return action


def enhance_image(image: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Full fuzzy-inference-based enhancement pipeline for one ultrasound
    image (BGR or grayscale, uint8).

    Steps:
      1. Convert to grayscale for noise estimation (keep color for output
         if the input was color).
      2. Estimate the local noise map.
      3. Run the FIS to get a per-pixel action map (denoise <-> sharpen).
      4. Compute a denoised version (bilateral filter, edge-preserving)
         and a sharpened version (unsharp masking).
      5. Blend the two per-pixel using the action map.

    Returns an enhanced image with the same shape/dtype as the input.
    """
    is_color = image.ndim == 3 and image.shape[-1] == 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image.copy()

    noise_map = estimate_local_noise_map(gray)
    action_map = fuzzy_action_map(noise_map)

    if verbose:
        print(f"Mean estimated noise: {noise_map.mean():.3f} | "
              f"Mean action (1=sharpen,0=denoise): {action_map.mean():.3f}")

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        channel_f = channel.astype(np.float32)
        denoised = cv2.bilateralFilter(channel, d=7, sigmaColor=50, sigmaSpace=50).astype(np.float32)
        blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(channel, 1.5, blurred, -0.5, 0).astype(np.float32)
        blended = action_map * sharpened + (1 - action_map) * denoised
        return np.clip(blended, 0, 255).astype(np.uint8)

    if is_color:
        channels = cv2.split(image)
        out_channels = [_process_channel(c) for c in channels]
        enhanced = cv2.merge(out_channels)
    else:
        enhanced = _process_channel(image)

    return enhanced


def add_synthetic_speckle_noise(image: np.ndarray, amount: float = 0.15) -> np.ndarray:
    """Utility used for PSNR-based validation of the FIS enhancement (paper
    Section III-F evaluates the enhancement using images with artificially
    generated noise of varying levels, measured via PSNR)."""
    noisy = image.astype(np.float32)
    speckle = np.random.randn(*image.shape) * amount * 255
    noisy = noisy + noisy * (speckle / 255.0)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def psnr(original: np.ndarray, processed: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))
