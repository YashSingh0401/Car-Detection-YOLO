"""Export trained YOLO model to ONNX for faster CPU inference."""
from pathlib import Path
from ultralytics import YOLO


def export_onnx(model_path: str = "yolo11m.pt", imgsz: int = 640, half: bool = False) -> str:
    model: YOLO = YOLO(model_path)
    out_path: str = model.export(format="onnx", imgsz=imgsz, half=half, simplify=True)
    print(f"ONNX model exported to: {out_path}")
    return out_path


def export_all_models(half: bool = False) -> None:
    models: list[str] = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    ]
    for m in models:
        try:
            export_onnx(m, half=half)
        except Exception as e:
            print(f"Failed to export {m}: {e}")


if __name__ == "__main__":
    import sys
    model = "yolo11m.pt"
    half = False
    args = sys.argv[1:]
    if args:
        if "--gpu" in args:
            half = True
            args.remove("--gpu")
        if args:
            model = args[0]
    export_onnx(model, half=half)
