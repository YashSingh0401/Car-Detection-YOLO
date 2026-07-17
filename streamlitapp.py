import os
import io
import csv
import json
import logging
import traceback
import tempfile
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Optional
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log: logging.Logger = logging.getLogger("car_detection")

st.set_page_config(page_title="YOLO Car Detection", layout="wide", page_icon="🚗")

st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 38px; font-weight: 700; color: #ff4b91; }
        .subtitle { text-align: center; font-size: 18px; color: #4a4a4a; }</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 YOLO Car Detection</div>', unsafe_allow_html=True)

import torch
device: str = "cuda" if torch.cuda.is_available() else "cpu"
fp16: bool = device == "cuda"
if fp16:
    log.info("GPU detected — FP16 inference enabled")

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


@st.cache_resource
def load_model(model_name: str) -> YOLO:
    log.info(f"Loading model: {model_name}")
    model_path = Path(__file__).parent / "models" / model_name
    if not model_path.exists():
        model_path = Path(model_name)
    m: YOLO = YOLO(str(model_path))
    return m


def run_detection(
    model: YOLO, img: np.ndarray, conf: float, iou: float,
    class_ids: Optional[list[int]], imgsz: int,
    use_tta: bool, use_preprocessing: bool, use_tracking: bool,
) -> list[tuple]:
    proc: np.ndarray = img.copy()
    if use_preprocessing:
        proc = apply_clahe(proc)

    if use_tracking:
        results = model.track(proc, conf=conf, iou=iou, classes=class_ids,
                              device=device, imgsz=imgsz, augment=use_tta,
                              persist=True, tracker="bytetrack.yaml", half=fp16)
    else:
        results = model(proc, conf=conf, iou=iou, classes=class_ids,
                        device=device, imgsz=imgsz, augment=use_tta, half=fp16)

    boxes_data: list[tuple] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        track_ids = r.boxes.id.cpu().numpy().astype(int) if (use_tracking and r.boxes.id is not None) else None
        for idx, (box, score, cls_id) in enumerate(zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        )):
            if score >= conf:
                track_id: int = track_ids[idx] if track_ids is not None else -1
                boxes_data.append((*box, score, cls_id, track_id))

    return boxes_data


def draw_detections(img: np.ndarray, boxes_data: list[tuple], names: dict) -> np.ndarray:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    for item in boxes_data:
        x1, y1, x2, y2, score, cls_id = item[:6]
        track_id: int = item[6] if len(item) > 6 else -1
        color: tuple[int, int, int] = CLASS_COLORS.get(int(cls_id), (0, 255, 0))
        label: str = f"{names[int(cls_id)]} {score:.2f}"
        if track_id >= 0:
            label += f" ID:{track_id}"
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


def boxes_to_dataframe(boxes_data: list[tuple], names: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for item in boxes_data:
        x1, y1, x2, y2, score, cls_id = item[:6]
        track_id: int = item[6] if len(item) > 6 else -1
        rows.append({
            "class": names[int(cls_id)],
            "class_id": int(cls_id),
            "confidence": round(float(score), 4),
            "x1": round(float(x1), 1),
            "y1": round(float(y1), 1),
            "x2": round(float(x2), 1),
            "y2": round(float(y2), 1),
            "width": round(float(x2 - x1), 1),
            "height": round(float(y2 - y1), 1),
            "track_id": track_id,
        })
    return pd.DataFrame(rows)


def process_video_frames(
    model: YOLO, cap: cv2.VideoCapture, out: cv2.VideoWriter,
    conf: float, iou: float, class_ids: Optional[list[int]],
    imgsz: int, use_tta: bool, use_preprocessing: bool, use_tracking: bool,
    total_frames: int, progress, status_text,
) -> tuple[int, int]:
    total_obj: int = 0
    fcount: int = 0
    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_dets: list[tuple] = run_detection(
            model, frame_rgb, conf, iou, class_ids,
            imgsz, use_tta, use_preprocessing, use_tracking,
        )
        annotated_frame: np.ndarray = frame.copy()
        draw_detections(annotated_frame, frame_dets, model.names)
        out.write(annotated_frame)
        total_obj += len(frame_dets)
        fcount += 1
        progress.progress(min(fcount / total_frames, 1.0))
        status_text.text(f"Frame {fcount}/{total_frames}")
    return fcount, total_obj


if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

available_models: list[str] = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
]
custom_models_dir: Path = Path(__file__).parent / "models"
if custom_models_dir.exists():
    custom_pts = sorted([p.name for p in custom_models_dir.glob("*.pt")])
    for pt in reversed(custom_pts):
        if pt not in available_models:
            available_models.insert(0, pt)

with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"Device: **{'GPU ⚡' if device == 'cuda' else 'CPU 💻'}**")

    source: str = st.selectbox("Input Type", ["Image", "Video", "Webcam"])

    # Determine default model index (prefer custom model, then large yolo11x, then fallback to index 0)
    default_idx = 0
    if "car_detection_best.pt" in available_models:
        default_idx = available_models.index("car_detection_best.pt")
    elif "yolo11x.pt" in available_models:
        default_idx = available_models.index("yolo11x.pt")

    model_name: str = st.selectbox("Model", available_models, index=default_idx)

    model: Optional[YOLO] = None
    with st.spinner(f"Loading {model_name}..."):
        try:
            model = load_model(model_name)
            st.sidebar.success(f"{model_name} loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            st.stop()

    sensitivity: str = st.select_slider("Detection Sensitivity",
        options=["Max Cars", "High", "Balanced", "Precise"], value="Max Cars")

    imgsz: int = st.select_slider("Resolution", options=[320, 480, 640, 800, 960, 1280], value=640)

    model_classes = list(model.names.values())
    default_classes = [c for c in ["car", "bus", "truck", "motorcycle", "person"] if c in model_classes]
    if not default_classes and model_classes:
        default_classes = [model_classes[0]]

    classes: list[str] = st.multiselect("Classes", options=model_classes, default=default_classes)

    use_tta: bool = st.checkbox("TTA (Test-Time Augmentation)", value=True)
    use_preprocessing: bool = st.checkbox("CLAHE Preprocessing", value=True)
    use_tracking: bool = False
    if source == "Video":
        use_tracking = st.checkbox("Object Tracking (ByteTrack)", value=False,
                                   help="Tracks objects across frames with consistent IDs")

    uploaded_files = None
    if source in ("Image", "Video"):
        types: list[str] = ["jpg", "jpeg", "png", "mp4", "avi", "mov"] if source == "Video" else ["jpg", "jpeg", "png"]
        uploaded_files = st.file_uploader("Upload File(s)", type=types, accept_multiple_files=(source == "Image"))

    start: bool = False
    if source == "Webcam":
        col1, col2 = st.columns(2)
        if col1.button("🟢 Start Webcam"):
            st.session_state.webcam_running = True
        if col2.button("⏹ Stop Webcam"):
            st.session_state.webcam_running = False
    else:
        start = st.button("🚀 Start Detection")

SENS_MAP: dict[str, tuple[float, float]] = {
    "Max Cars": (0.08, 0.50), "High": (0.15, 0.45),
    "Balanced": (0.25, 0.45), "Precise": (0.40, 0.50),
}
conf: float
iou: float
conf, iou = SENS_MAP[sensitivity]

class_map: dict[str, int] = {v: k for k, v in model.names.items()}
class_ids: Optional[list[int]] = [class_map[c] for c in classes if c in class_map] if classes else None

# ---- Image Detection ----
if start and source == "Image" and uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            image: Image.Image = Image.open(uploaded_file)
            image_np: np.ndarray = np.array(image)

            with st.spinner("Detecting..."):
                boxes_data: list[tuple] = run_detection(
                    model, image_np, conf, iou, class_ids,
                    imgsz, use_tta, use_preprocessing, use_tracking=False,
                )

            annotated: np.ndarray = image_np.copy()
            draw_detections(annotated, boxes_data, model.names)

            st.subheader(f"📷 {uploaded_file.name}")
            tab1, tab2, tab3 = st.tabs(["🎯 Result", "📷 Original", "📊 Data"])
            with tab1:
                st.image(annotated, use_container_width=True)
            with tab2:
                st.image(image, use_container_width=True)
            with tab3:
                df: pd.DataFrame = boxes_to_dataframe(boxes_data, model.names)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if not df.empty:
                    csv_buf: io.StringIO = io.StringIO()
                    df.to_csv(csv_buf, index=False)
                    st.download_button("📥 Download CSV", csv_buf.getvalue(),
                                       file_name=f"{Path(uploaded_file.name).stem}_detections.csv",
                                       mime="text/csv")
                    st.download_button("📥 Download JSON", df.to_json(orient="records", indent=2),
                                       file_name=f"{Path(uploaded_file.name).stem}_detections.json",
                                       mime="application/json")

            col1, col2 = st.columns(2)
            col1.metric("Objects Found", len(boxes_data))
            if boxes_data:
                scores = [s for _, _, _, _, s, _, _ in boxes_data]
                col2.metric("Avg Confidence", f"{np.mean(scores):.0%}")

            log.info(f"Image {uploaded_file.name}: {len(boxes_data)} objects detected")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            log.error(f"Image processing failed: {e}", exc_info=True)

# ---- Video Detection ----
elif start and source == "Video" and uploaded_files:
    uploaded_file = uploaded_files
    tfile: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()

    try:
        cap: cv2.VideoCapture = cv2.VideoCapture(tfile.name)
        fps: int = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_tfile: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_tfile.close()
        out: cv2.VideoWriter = cv2.VideoWriter(
            out_tfile.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
        )

        progress: object = st.progress(0)
        status_text: object = st.empty()
        log.info(f"Processing video: {total_frames} frames, {fps} fps")

        fcount: int
        total_obj: int
        fcount, total_obj = process_video_frames(
            model, cap, out, conf, iou, class_ids,
            imgsz, use_tta, use_preprocessing, use_tracking,
            total_frames, progress, status_text,
        )

        cap.release()
        out.release()

        st.success(f"Done! {fcount} frames processed, {total_obj} objects detected")
        log.info(f"Video done: {fcount} frames, {total_obj} detections")

        # H.264 encoding
        h264_tfile: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        h264_tfile.close()
        import subprocess
        cmd: list[str] = ["ffmpeg", "-y", "-i", out_tfile.name, "-vcodec", "libx264", "-pix_fmt", "yuv420p", h264_tfile.name]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            with st.expander("🎬 Processed Video", expanded=True):
                with open(h264_tfile.name, "rb") as f:
                    st.video(f.read())
        except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
            st.warning(f"⚠️ FFmpeg post-processing failed or FFmpeg is not installed ({e}). Displaying raw video...")
            with st.expander("📥 Raw Video (Download)", expanded=True):
                with open(out_tfile.name, "rb") as f:
                    st.video(f.read())
        finally:
            try:
                os.unlink(h264_tfile.name)
            except (NameError, OSError):
                pass

    except Exception as e:
        st.error(f"Video processing failed: {e}")
        log.error(f"Video processing error: {e}", exc_info=True)
        st.code(traceback.format_exc())

# ---- Webcam Detection ----
elif source == "Webcam" and st.session_state.webcam_running:
    st.warning("Webcam mode: point your camera at vehicles")
    FRAME_WINDOW: object = st.image([])
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state.webcam_running:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
        frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes_data = run_detection(
            model, frame_rgb, conf, iou, class_ids,
            imgsz, use_tta, use_preprocessing, use_tracking=False,
        )
        draw_detections(frame_rgb, boxes_data, model.names)
        FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    log.info("Webcam detection stopped")

elif start and uploaded_files is None and source != "Webcam":
    st.warning("Please upload a file first")
