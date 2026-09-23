"""
Generates a small synthetic ovarian-ultrasound-like demo dataset so the
full pipeline can be run and demonstrated without internet access to the
real Kaggle PCOS ultrasound dataset used in the base paper.

Each image is a grayscale "ovary" blob on a dark background with
realistic speckle noise:
  - notPCOS images contain 0-3 large, sparse follicles.
  - PCOS images contain many (>=8) small follicles arranged near the
    periphery -- mimicking the "string of pearls" polycystic morphology
    that is the actual diagnostic criterion ultrasound scans look for.

This is ONLY a stand-in for demonstrating that the code runs end-to-end.
For real results, replace data/raw/PCOS and data/raw/notPCOS with the
actual Kaggle "PCOS Detection using Ultrasound Images" dataset (or any
equivalent ovarian ultrasound dataset), keeping the same folder layout.
"""

import os

import cv2
import numpy as np


def _draw_ovary_base(size, rng):
    img = np.zeros((size, size), dtype=np.float32)
    cy, cx = size // 2 + rng.integers(-10, 10), size // 2 + rng.integers(-10, 10)
    axes = (rng.integers(size // 3, size // 2), rng.integers(size // 4, size // 3))
    angle = rng.integers(0, 180)
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color=90, thickness=-1)
    return img, (cx, cy), axes


def _add_speckle(img, rng, amount=0.35):
    speckle = rng.normal(0, 1, img.shape).astype(np.float32) * amount * img
    noisy = img + speckle
    return np.clip(noisy, 0, 255)


def make_non_pcos_image(size, rng):
    img, (cx, cy), axes = _draw_ovary_base(size, rng)
    n_follicles = rng.integers(0, 4)
    for _ in range(n_follicles):
        fy = cy + rng.integers(-axes[1] // 2, axes[1] // 2)
        fx = cx + rng.integers(-axes[0] // 2, axes[0] // 2)
        r = rng.integers(8, 16)
        cv2.circle(img, (fx, fy), r, color=20, thickness=-1)
        cv2.circle(img, (fx, fy), r, color=150, thickness=1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = _add_speckle(img, rng)
    return img.astype(np.uint8)


def make_pcos_image(size, rng):
    img, (cx, cy), axes = _draw_ovary_base(size, rng)
    n_follicles = rng.integers(9, 16)
    ring_radius = min(axes) * 0.75
    for i in range(n_follicles):
        theta = 2 * np.pi * i / n_follicles + rng.uniform(-0.15, 0.15)
        fx = int(cx + ring_radius * np.cos(theta) * rng.uniform(0.8, 1.05))
        fy = int(cy + ring_radius * 0.6 * np.sin(theta) * rng.uniform(0.8, 1.05))
        r = rng.integers(4, 8)
        cv2.circle(img, (fx, fy), r, color=15, thickness=-1)
        cv2.circle(img, (fx, fy), r, color=170, thickness=1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = _add_speckle(img, rng)
    return img.astype(np.uint8)


def generate(out_dir="data/raw", n_per_class=120, size=256, seed=42):
    rng = np.random.default_rng(seed)
    pcos_dir = os.path.join(out_dir, "PCOS")
    non_dir = os.path.join(out_dir, "notPCOS")
    os.makedirs(pcos_dir, exist_ok=True)
    os.makedirs(non_dir, exist_ok=True)

    for i in range(n_per_class):
        img = make_pcos_image(size, rng)
        cv2.imwrite(os.path.join(pcos_dir, f"pcos_{i:04d}.png"), img)

    for i in range(n_per_class):
        img = make_non_pcos_image(size, rng)
        cv2.imwrite(os.path.join(non_dir, f"notpcos_{i:04d}.png"), img)

    print(f"Wrote {n_per_class} PCOS images to {pcos_dir}")
    print(f"Wrote {n_per_class} notPCOS images to {non_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--n_per_class", type=int, default=120)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    generate(args.out_dir, args.n_per_class, args.size)
