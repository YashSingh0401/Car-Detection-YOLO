"""FastAPI REST API for car detection."""
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO
import io

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

app: FastAPI = FastAPI(title="Car Detection API", version="1.0.0", lifespan=lifespan)
model: Optional[YOLO] = None


def get_model() -> YOLO:
    global model
    if model is None:
        custom_best = Path(__file__).parent / "models" / "car_detection_best.pt"
        if custom_best.exists():
            model = YOLO(str(custom_best))
        else:
            model = YOLO("yolo11m.pt")
    return model


@app.get("/")
async def root() -> dict:
    return {"service": "Car Detection API", "status": "running"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
) -> Response:
    contents: bytes = await file.read()
    nparr: np.ndarray = np.frombuffer(contents, np.uint8)
    img: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    m: YOLO = get_model()
    results = m(img_rgb, conf=conf, iou=iou, imgsz=imgsz)

    detections: list[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        for box, score, cls_id in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        ):
            detections.append({
                "bbox": [float(x) for x in box],
                "confidence": float(score),
                "class_id": int(cls_id),
                "class_name": m.names.get(int(cls_id), "unknown"),
            })

    return JSONResponse({
        "detections": detections,
        "count": len(detections),
    })


@app.post("/detect-annotated")
async def detect_annotated(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
) -> Response:
    contents: bytes = await file.read()
    nparr: np.ndarray = np.frombuffer(contents, np.uint8)
    img: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    m: YOLO = get_model()
    results = m(img_rgb, conf=conf, iou=iou)
    annotated: np.ndarray = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
