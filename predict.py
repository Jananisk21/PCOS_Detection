"""
Run inference on a single ultrasound image with the trained ensemble,
printing the predicted class + confidence and saving a Grad-CAM overlay
that highlights the follicle regions driving the decision.

Usage:
    python predict.py --image path/to/scan.png \
        --model_path saved_models/pcos_effidensegenop_model.keras
"""

import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from src.dataset import CLASS_NAMES
from src.fuzzy_enhancement import enhance_image
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.image_validator import validate_ultrasound_image


def predict_image(model, image_bgr: np.ndarray, img_size: int):
    enhanced = enhance_image(image_bgr)
    resized = cv2.resize(enhanced, (img_size, img_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    batch = np.expand_dims(rgb, axis=0)

    prob = float(model.predict(batch, verbose=0)[0][0])
    pred_class = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1 - prob

    heatmap = make_gradcam_heatmap(model, batch)
    overlay = overlay_heatmap(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                               if resized.shape[-1] == 3 else resized, heatmap)

    return {
        "predicted_class": pred_class,
        "pcos_probability": prob,
        "confidence": confidence,
        "enhanced_image": enhanced,
        "gradcam_overlay": overlay,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_path", default="saved_models/pcos_effidensegenop_model.keras")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--force", action="store_true", help="Force prediction on non-ultrasound images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    # Image validation
    is_valid, reason, metrics = validate_ultrasound_image(image_bgr)
    if not is_valid:
        print(f"\n[WARNING] Non-ultrasound image detected: {reason}")
        print(f"Metrics: {metrics}")
        if not args.force:
            print("Aborting. Use --force to proceed anyway.")
            sys.exit(1)

    model = tf.keras.models.load_model(args.model_path)
    result = predict_image(model, image_bgr, args.img_size)

    print(f"Predicted class : {result['predicted_class']}")
    print(f"PCOS probability: {result['pcos_probability']:.4f}")
    print(f"Confidence      : {result['confidence'] * 100:.2f}%")

    base = os.path.splitext(os.path.basename(args.image))[0]
    enhanced_path = os.path.join(args.output_dir, f"{base}_enhanced.png")
    gradcam_path = os.path.join(args.output_dir, f"{base}_gradcam.png")
    cv2.imwrite(enhanced_path, result["enhanced_image"])
    cv2.imwrite(gradcam_path, result["gradcam_overlay"])
    print(f"Saved enhanced image to {enhanced_path}")
    print(f"Saved Grad-CAM overlay to {gradcam_path}")


if __name__ == "__main__":
    main()
