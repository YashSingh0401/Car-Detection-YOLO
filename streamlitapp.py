import os
import traceback
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import subprocess
from pathlib import Path
from typing import Optional

st.set_page_config(page_title="YOLO Car Detection", layout="wide", page_icon="🚗")

st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 38px; font-weight: 700; color: #ff4b91; }
        .subtitle { text-align: center; font-size: 18px; color: #4a4a4a; }</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 YOLO Car Detection</div>', unsafe_allow_html=True)

device: str = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 100, 100), 1: (255, 200, 50), 2: (50, 200, 255),
    3: (255, 150, 50), 5: (50, 255, 100), 7: (100, 100, 255),
    9: (0, 255, 255), 11: (255, 0, 255),
}

CAR_CLASS_IDS: set[int] = {2, 5, 7}


def apply_clahe(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] >= 3:
        lab: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


def run_detection(
    model: YOLO, img: np.ndarray, conf: float, iou: float,
    class_ids: Optional[list[int]], device: str, imgsz: int,
    use_tta: bool, use_preprocessing: bool
) -> list[tuple]:
    proc: np.ndarray = img.copy()
    if use_preprocessing:
        proc = apply_clahe(proc)

    results = model(proc, conf=conf, iou=iou, classes=class_ids, device=device, imgsz=imgsz, augment=use_tta)

    boxes_data: list[tuple] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box, score, cls_id in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int)
        ):
            if int(cls_id) in CAR_CLASS_IDS:
                score = min(score * 1.3, 1.0)
            if score >= conf:
                boxes_data.append((*box, score, cls_id))

    return boxes_data


def draw_detections(img: np.ndarray, boxes_data: list[tuple], names: dict) -> np.ndarray:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    for x1, y1, x2, y2, score, cls_id in boxes_data:
        color: tuple[int, int, int] = CLASS_COLORS.get(int(cls_id), (0, 255, 0))
        label: str = f"{names[int(cls_id)]} {score:.2f}"
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_w: int = x2 - x1
        box_h: int = y2 - y1
        diag: float = (box_w ** 2 + box_h ** 2) ** 0.5
        font_scale: float = max(0.6, min(1.6, diag / 180.0))
        thickness: int = max(1, int(diag / 150.0))
        box_thickness: int = max(2, int(diag / 120.0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
        pad: int = 6
        lx: int = x1
        ly: int = y1 - 8
        if ly - th < 0:
            ly = y1 + th + 8
        cv2.rectangle(img, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad), color, -1)
        text_color: tuple[int, int, int] = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
        cv2.putText(img, label, (lx, ly), font, font_scale, text_color, thickness)
    return img


with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"Device: **{'GPU ⚡' if device == 'cuda' else 'CPU 💻'}**")

    source: str = st.selectbox("Input Type", ["Image", "Video"])

    model_name: str = st.selectbox("Model", [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    ], index=0)

    sensitivity: str = st.select_slider("Detection Sensitivity",
        options=["Max Cars", "High", "Balanced", "Precise"],
        value="High")

    imgsz: int = st.select_slider("Resolution", options=[320, 480, 640, 800, 960, 1280], value=640)

    classes: list[str] = st.multiselect("Classes",
        ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"],
        default=["car", "bus", "truck", "motorcycle", "person"])

    use_tta: bool = st.checkbox("TTA (Test-Time Augmentation)", value=True)

    use_preprocessing: bool = st.checkbox("CLAHE Preprocessing", value=True)

    uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    start: bool = st.button("🚀 Start Detection")

SENS_MAP: dict[str, tuple[float, float]] = {
    "Max Cars": (0.08, 0.50),
    "High": (0.15, 0.45),
    "Balanced": (0.25, 0.45),
    "Precise": (0.40, 0.50),
}
conf: float
iou: float
conf, iou = SENS_MAP[sensitivity]

model: Optional[YOLO] = None
with st.spinner(f"Loading {model_name}..."):
    try:
        model = YOLO(model_name)
        st.sidebar.success(f"{model_name} loaded | conf={conf}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()

class_map: dict[str, int] = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "traffic light": 9, "stop sign": 11}
class_ids: Optional[list[int]] = [class_map[c] for c in classes if c in class_map] if classes else None

if start and uploaded_file is not None:
    try:
        if source == "Image":
            image: Image.Image = Image.open(uploaded_file)
            image_np: np.ndarray = np.array(image)

            with st.spinner("Detecting..."):
                boxes_data: list[tuple] = run_detection(
                    model, image_np, conf, iou, class_ids,
                    device, imgsz, use_tta, use_preprocessing
                )

            annotated: np.ndarray = image_np.copy()
            draw_detections(annotated, boxes_data, model.names)

            tab1, tab2 = st.tabs(["🎯 Detection Result", "📷 Original"])
            with tab1:
                st.image(annotated, use_container_width=True)
            with tab2:
                st.image(image, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Cars Found", len(boxes_data))
            if boxes_data:
                scores: list[float] = [s for _, _, _, _, s, _ in boxes_data]
                col2.metric("Avg Confidence", f"{np.mean(scores):.0%}")
                col3.metric("Settings", f"conf={conf}, imgsz={imgsz}")

        elif source == "Video":
            tfile: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            tfile.close()

            cap: cv2.VideoCapture = cv2.VideoCapture(tfile.name)
            fps: int = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_tfile: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            out_tfile.close()
            out: cv2.VideoWriter = cv2.VideoWriter(out_tfile.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            progress: object = st.progress(0)
            status: object = st.empty()
            total_obj: int = 0
            fcount: int = 0

            try:
                while True:
                    ret: bool
                    frame: np.ndarray
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_dets: list[tuple] = run_detection(
                        model, frame_rgb, conf, iou, class_ids,
                        device, imgsz, use_tta, use_preprocessing
                    )
                    annotated_frame: np.ndarray = frame.copy()
                    draw_detections(annotated_frame, frame_dets, model.names)
                    out.write(annotated_frame)
                    total_obj += len(frame_dets)
                    fcount += 1
                    progress.progress(min(fcount / total_frames, 1.0))
                    status.text(f"Frame {fcount}/{total_frames}")
            finally:
                cap.release()
                out.release()

            col_v1, col_v2, col_v3 = st.columns(3)
            col_v1.metric("Frames", fcount)
            col_v2.metric("Total Objects", total_obj)
            col_v3.metric("Per Frame", f"{total_obj / max(fcount, 1):.1f}")

            h264_tfile: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            h264_tfile.close()
            cmd: list[str] = ["ffmpeg", "-y", "-i", out_tfile.name, "-vcodec", "libx264", "-pix_fmt", "yuv420p", h264_tfile.name]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                with st.expander("🎬 Processed Video", expanded=True):
                    with open(h264_tfile.name, "rb") as f:
                        st.video(f.read())
            except subprocess.CalledProcessError:
                st.warning("⚠️ ffmpeg not found. Install ffmpeg for browser-compatible video encoding.")
                with st.expander("📥 Raw Video (Download)", expanded=True):
                    with open(out_tfile.name, "rb") as f:
                        st.video(f.read())
            finally:
                try:
                    os.unlink(h264_tfile.name)
                except (NameError, OSError):
                    pass

            st.success(f"Done! {total_obj} objects detected")

    except Exception as e:
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())

elif start and uploaded_file is None:
    st.warning("Please upload a file first")
