import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚗 YOLO Detection App",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    color: #00ADB5;
}
.stButton>button {
    border-radius: 10px;
    background-color: #00ADB5;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # or "best.pt"

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5
)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image to detect objects using YOLOv8.")

# ---------------- HEADER ----------------
st.markdown("<h1>🚗 YOLO Object Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>AI-powered real-time detection of vehicles and objects</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    img = np.array(image)

    # Prediction
    with st.spinner("🔍 Detecting objects..."):
        results = model(img, conf=confidence)

    result_img = results[0].plot()

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(result_img, use_column_width=True)

    # ---------------- STATS ----------------
    st.markdown("### 📊 Detection Summary")

    classes = results[0].boxes.cls.tolist()
    names = model.names

    detected = [names[int(c)] for c in classes]

    if detected:
        count = Counter(detected)

        col1, col2, col3 = st.columns(3)

        for i, (obj, num) in enumerate(count.items()):
            if i % 3 == 0:
                col1.metric(obj.upper(), num)
            elif i % 3 == 1:
                col2.metric(obj.upper(), num)
            else:
                col3.metric(obj.upper(), num)
    else:
        st.warning("No objects detected.")

    # ---------------- DOWNLOAD ----------------
    st.markdown("---")
    st.download_button(
        label="📥 Download Result Image",
        data=result_img.tobytes(),
        file_name="result.png",
        mime="image/png"
    )

else:
    st.info("👆 Upload an image to start detection")
