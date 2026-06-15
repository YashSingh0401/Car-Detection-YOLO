import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import torch
import subprocess

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YOLO Detection App",
    layout="wide",
    page_icon="🚀"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 38px;
            font-weight: 700;
            color: #ff4b91;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #4a4a4a;
            margin-bottom: 30px;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            font-size: 16px;
            background-color: #ff4b91;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">YOLO Object Detection App 🚀</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect objects in images and videos</div>', unsafe_allow_html=True)

# ---------------- DEVICE SELECTION ----------------
if torch.cuda.is_available():
    device = "cuda"
    device_name = "GPU (CUDA) ⚡"
else:
    device = "cpu"
    device_name = "CPU 💻"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.info(f"Running on: **{device_name}**")

    source = st.selectbox("Select Input Type", ["Image", "Video"])

    model_name = st.selectbox("Select Model", [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt"
    ])

    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    iou = st.slider("IoU Threshold", 0.0, 1.0, 0.45)

    classes = st.multiselect(
        "Select Classes",
        ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"],
        default=["person", "car"]
    )

    video_speed = st.selectbox(
        "Video Processing Speed (CPU Optimization)",
        ["Normal (Process all frames)", "Fast (Skip 1 frame)", "Very Fast (Skip 2 frames)"],
        index=0
    )

    uploaded_file = st.file_uploader(
        "Upload File",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

    start = st.button("🚀 Start Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_name)

# ---------------- CLASS MAPPING ----------------
class_map = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "traffic light": 9,
    "stop sign": 11
}

class_ids = None
if classes:
    class_ids = [class_map[c] for c in classes if c in class_map]

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------------- IMAGE DETECTION ----------------
if start and source == "Image" and uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    results = model(image_np, conf=conf, iou=iou, classes=class_ids, device=device)
    annotated = results[0].plot()

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(annotated, use_column_width=True)

    st.success("✅ Image processed successfully!")

# ---------------- VIDEO DETECTION ----------------
elif start and source == "Video" and uploaded_file is not None:

    # Determine frame skipping rate
    skip_frames = 1
    if video_speed == "Fast (Skip 1 frame)":
        skip_frames = 2
    elif video_speed == "Very Fast (Skip 2 frames)":
        skip_frames = 3

    # Save uploaded video to temp file and close to flush buffers
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output file setup
    out_tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_tfile.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_tfile.name, fourcc, fps // skip_frames, (width, height))

    frame_placeholder1 = col1.empty()
    frame_placeholder2 = col2.empty()

    st.info("Processing video... Please wait ⏳")
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            results = model(frame, conf=conf, iou=iou, classes=class_ids, device=device)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            frame_placeholder1.image(frame_rgb, channels="RGB", use_column_width=True)
            frame_placeholder2.image(annotated_rgb, channels="RGB", use_column_width=True)

        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        status_text.text(f"Processing frame {frame_count}/{total_frames}...")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    # Convert the processed video to H.264 format using ffmpeg for HTML5 browser compatibility
    h264_tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h264_tfile.close()

    st.info("Encoding video for browser display... 🎬")
    cmd = f'ffmpeg -y -i "{out_tfile.name}" -vcodec libx264 -pix_fmt yuv420p "{h264_tfile.name}"'
    res = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if res.returncode == 0:
        with open(h264_tfile.name, "rb") as f:
            video_bytes = f.read()
        st.success("✅ Video processed and encoded successfully!")
        
        st.markdown("### 🎬 Processed Video Player")
        st.video(video_bytes)
        
        st.download_button(
            label="📥 Download Annotated Video",
            data=video_bytes,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )
    else:
        with open(out_tfile.name, "rb") as f:
            video_bytes = f.read()
        st.warning("⚠️ Could not encode video for inline playback, but you can still download the processed file below.")
        st.download_button(
            label="📥 Download Annotated Video (Raw MP4)",
            data=video_bytes,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )

# ---------------- WARNING ----------------
elif start and uploaded_file is None:
    st.warning("⚠️ Please upload a file first!")