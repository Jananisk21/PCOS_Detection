"""
Grad-CAM explainability for the ensemble model, matching Section III-F of
the base paper ("using Class Activation Maps or Grad-CAM to emphasize
follicles that the AI used to determine a PCOS diagnosis").

Because the model has two parallel backbones, Grad-CAM is computed against
the last convolutional feature map of the EfficientNetB7 branch (the
branch the base paper's ensemble weights most heavily), then upsampled and
overlaid on the input image.
"""

import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(sub_model):
    for layer in reversed(sub_model.layers):
        if len(layer.output.shape) == 4:  # (batch, H, W, C)
            return layer.name
    raise ValueError("No 4D (conv) layer found in sub-model.")


def _find_backbone_layer(model, keyword="efficientnet"):
    """The EfficientNet backbone sub-model keeps its Keras-applications
    default name (e.g. 'efficientnetb7', 'efficientnetb0') since Keras
    functional Model names cannot be reassigned after construction; find
    it by keyword instead of relying on a fixed name."""
    for layer in model.layers:
        if keyword in layer.name.lower():
            return layer
    raise ValueError(f"No layer containing '{keyword}' found in model. "
                      f"Available layers: {[l.name for l in model.layers]}")


def make_gradcam_heatmap(model, image_batch, backbone_name=None):
    """
    Args:
        model: the compiled ensemble Keras model.
        image_batch: array of shape (1, H, W, 3), NOT preprocessed
            (preprocessing is applied inside the model already).
        backbone_name: optional explicit sub-model layer name to explain;
            if None, the EfficientNet backbone is located automatically.

    Returns:
        heatmap (H, W) float array normalized to [0, 1].
    """
    backbone = model.get_layer(backbone_name) if backbone_name else _find_backbone_layer(model)
    backbone_name = backbone.name
    last_conv_name = _find_last_conv_layer(backbone)

    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_name).output, backbone.output],
    )

    # Route the raw image through the same preprocessing the full model uses.
    if "efficientnet" in backbone_name:
        preprocessed = tf.keras.applications.efficientnet.preprocess_input(image_batch.copy())
    else:
        preprocessed = tf.keras.applications.densenet.preprocess_input(image_batch.copy())

    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(preprocessed)
        # Use the mean of pooled backbone features as the scalar target,
        # since the final sigmoid depends on the fused representation from
        # both backbones jointly, not this backbone's output alone.
        target = tf.reduce_mean(backbone_output)

    grads = tape.gradient(target, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_bgr: np.ndarray, heatmap: np.ndarray, alpha=0.4) -> np.ndarray:
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
