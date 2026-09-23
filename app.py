"""
Simple Streamlit interface to demonstrate the trained PCOS-detection
ensemble: upload an ovarian ultrasound image, see the fuzzy-enhanced
version, the predicted class + confidence, and a Grad-CAM overlay
showing which regions of the image drove the prediction.

Includes automatic validation to filter out non-ultrasound images (e.g.
diagrams, charts, color photos, documents).

Run with:
    streamlit run app.py
"""

import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from src.dataset import CLASS_NAMES
from src.fuzzy_enhancement import enhance_image
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.image_validator import validate_ultrasound_image

MODEL_PATH = "saved_models/pcos_effidensegenop_model.keras"
IMG_SIZE = 96

st.set_page_config(page_title="PCOS Ultrasound Detection Demo", layout="wide")
st.title("PCOS Detection from Ultrasound Sonography Images")
st.caption(
    "Demo implementation of the EffiDenseGenOp pipeline: Fuzzy-Inference-System "
    "enhancement -> EfficientNet + DenseNet ensemble (GA/PSO/GWO-tuned) "
    "-> Grad-CAM explainability."
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

if model is None:
    st.warning(
        f"No trained model found at `{MODEL_PATH}`. Run `python train.py` first "
        "(optionally after `python generate_synthetic_dataset.py` for a quick demo run)."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an ovarian ultrasound sonography image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Error: Failed to decode the uploaded image file.")
        st.stop()

    # Step 0: Validate if image is a genuine ultrasound scan
    is_valid, validation_msg, metrics = validate_ultrasound_image(image_bgr)

    if not is_valid:
        st.error(f"❌ **Invalid Image Type Detected**\n\n{validation_msg}")
        st.info(
            "💡 **Why did this happen?**\n"
            "This model is trained specifically for **B-mode grayscale ovarian ultrasound sonography scans**. "
            "Diagrams, general photos, infographics, or non-medical images are not supported for diagnostic prediction."
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
        with col2:
            st.subheader("Image Metrics Analysis")
            st.write(metrics)
            force_run = st.checkbox("⚠️ Force prediction anyway (Not recommended)")
        
        if not force_run:
            st.stop()

    enhanced = enhance_image(image_bgr)
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    batch = np.expand_dims(rgb, axis=0)

    with st.spinner("Running ensemble inference & Grad-CAM..."):
        prob = float(model.predict(batch, verbose=0)[0][0])
        pred_class = CLASS_NAMES[int(prob >= 0.5)]
        confidence = prob if prob >= 0.5 else 1 - prob
        heatmap = make_gradcam_heatmap(model, batch)
        overlay = overlay_heatmap(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), heatmap)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Fuzzy-Enhanced")
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col3:
        st.subheader("Grad-CAM (model focus)")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.divider()
    if pred_class == "PCOS":
        st.error(f"Prediction: **{pred_class}** — confidence {confidence * 100:.1f}%")
    else:
        st.success(f"Prediction: **{pred_class}** — confidence {confidence * 100:.1f}%")
    st.progress(prob, text=f"Raw PCOS probability: {prob:.3f}")

    st.info(
        "This is a research/demo tool trained on ovarian ultrasound data for demonstration "
        "purposes only. It is **not** a medical diagnostic device."
    )
else:
    st.info("Upload an ultrasound image (e.g. from `data/raw/PCOS/` or `data/raw/notPCOS/`) to run detection.")
