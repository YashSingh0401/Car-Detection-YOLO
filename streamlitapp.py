import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import tempfile
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Detection AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #00BFFF;
}
.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 30px;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚗 Smart Car Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8 • Image | Video | Webcam</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Mode", ["Image", "Video", "Webcam"])
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

vehicle_classes = [2, 3, 5, 7]

# ---------------- IMAGE MODE ----------------
if mode == "Image":
    file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        col1, col2 = st.columns(2)

        image = Image.open(file)
        img = np.array(image)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("🔍 Detecting vehicles..."):
            results = model(img, conf=confidence)
            output = results[0].plot()

        # Count vehicles
        count = 0
        for box in results[0].boxes:
            if int(box.cls[0]) in vehicle_classes:
                count += 1

        with col2:
            st.image(output, caption="Detected Output", use_column_width=True)

        st.markdown(f"""
        <div class="card">
        <h3>📊 Vehicles Detected: {count}</h3>
        </div>
        """, unsafe_allow_html=True)

# ---------------- VIDEO MODE ----------------
elif mode == "Video":
    file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        progress = st.progress(0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            frame = results[0].plot()

            # Count vehicles
            count = sum(
                1 for box in results[0].boxes
                if int(box.cls[0]) in vehicle_classes
            )

            stframe.image(frame, channels="BGR")

            frame_num += 1
            progress.progress(min(frame_num / total_frames, 1.0))

        cap.release()
        st.success("✅ Video processing complete!")

# ---------------- WEBCAM MODE ----------------
elif mode == "Webcam":
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence)
        frame = results[0].plot()

        # Count vehicles
        count = sum(
            1 for box in results[0].boxes
            if int(box.cls[0]) in vehicle_classes
        )

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
