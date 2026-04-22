import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from collections import Counter
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚗 YOLO Detection App",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
import os

@st.cache_resource
def load_model():
    model_path = "yolov8n.pt"

    # Delete corrupted file if exists
    if os.path.exists(model_path):
        os.remove(model_path)

    return YOLO(model_path)  # forces fresh download

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5
)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image to detect objects using YOLOv8.")

# ---------------- HEADER ----------------
st.title("🚗 YOLO Object Detection")
st.write("AI-powered real-time detection of vehicles and objects")

st.markdown("---")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    img = np.array(image)

    # ---------------- PREDICTION ----------------
    with st.spinner("🔍 Detecting objects..."):
        results = model(img, conf=confidence)

    result_img = results[0].plot()

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(result_img, use_container_width=True)

    # ---------------- STATS ----------------
    st.markdown("### 📊 Detection Summary")

    if results[0].boxes is not None:
        classes = results[0].boxes.cls.tolist()
        names = model.names

        detected = [names[int(c)] for c in classes]

        if detected:
            count = Counter(detected)
            cols = st.columns(3)

            for i, (obj, num) in enumerate(count.items()):
                cols[i % 3].metric(obj.upper(), num)
        else:
            st.warning("No objects detected.")
    else:
        st.warning("No objects detected.")

    # ---------------- DOWNLOAD ----------------
    _, buffer = cv2.imencode(".png", result_img)

    st.download_button(
        label="📥 Download Result Image",
        data=buffer.tobytes(),
        file_name="result.png",
        mime="image/png"
    )

else:
    st.info("👆 Upload an image to start detection")
