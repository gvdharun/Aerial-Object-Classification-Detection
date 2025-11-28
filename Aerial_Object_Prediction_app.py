# app_mobilenetv2.py

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 1. Page config
st.set_page_config(page_title="Aerial Object Classification & Detection", layout="centered")
st.title("🕊️ Bird vs 🚁 Drone Classifier")
st.write("Upload an aerial image to classify it as **Bird** or **Drone** using a fine‑tuned MobileNetV2 model.")

# 2. Load trained model (path from your training script)
try:
    model = load_model("models/best_mobilenetv2.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


CLASS_NAMES = {0: "Bird", 1: "Drone"}

# 3. Image preprocessing (must match training: 224x224, /255)
IMG_SIZE = 224

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr

# 4. File uploader
uploaded_file = st.file_uploader("Upload image (.jpg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Predict"):
        inp = preprocess_image(image)
        prob = model.predict(inp)[0][0]  # sigmoid output

        pred_class = 1 if prob >= 0.5 else 0
        confidence = prob if pred_class == 1 else 1 - prob

        st.markdown(f"### Prediction: **{CLASS_NAMES[pred_class]}**")
        st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")

        st.info("Threshold: 0.5 (≥ 0.5 → Drone, < 0.5 → Bird)")

else:
    st.write("👆 Upload an image to get started.")

st.caption("Model: MobileNetV2 (ImageNet weights) fine‑tuned on Bird vs Drone aerial dataset.")
