"""Evaluate car detection models and compare accuracy."""
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import pandas as pd
from typing import Optional

ROOT: Path = Path(__file__).parent
MODELS_DIR: Path = ROOT / "models"


def benchmark_models() -> pd.DataFrame:
    models_to_test: list[str] = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    ]

    custom_models: list[Path] = list(MODELS_DIR.glob("*.pt"))
    results: list[dict] = []

    for model_name in models_to_test:
        try:
            model: YOLO = YOLO(model_name)
            metrics = model.val(data="coco8.yaml", split="val", device="cpu", imgsz=640)

            results.append({
                "Model": model_name,
                "mAP50": metrics.box.map50,
                "mAP50-95": metrics.box.map,
                "Precision": metrics.box.p,
                "Recall": metrics.box.r,
                "Parameters (M)": sum(p.numel() for p in model.model.parameters()) / 1e6,
            })

            print(f"{model_name:20s} | mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")
        except Exception as e:
            print(f"{model_name:20s} | Error: {e}")

    df: pd.DataFrame = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Model Benchmark Results:")
    print(df.to_string(index=False))

    if not df.empty:
        best_idx: int = df["mAP50"].idxmax()
        print(f"\nBest model: {df.iloc[best_idx]['Model']} (mAP50: {df.iloc[best_idx]['mAP50']:.4f})")

    return df


def evaluate_custom_model(model_path: str, dataset_yaml: Optional[Path] = None) -> Optional[object]:
    if dataset_yaml is None:
        dataset_yaml = ROOT / "data" / "car_dataset" / "dataset.yaml"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return None

    model: YOLO = YOLO(str(model_path))

    print(f"\nEvaluating {model_path}...")

    metrics = model.val(
        data=str(dataset_yaml) if dataset_yaml.exists() else "coco8.yaml",
        device="cpu",
        imgsz=640,
    )

    print(f"\nResults for {Path(model_path).name}:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p:.4f}")
    print(f"  Recall:    {metrics.box.r:.4f}")

    return metrics


def compare_models(model_paths: Optional[list[str]] = None) -> None:
    if model_paths is None:
        model_paths = [
            "yolov8n.pt",
            "yolov8s.pt",
            "yolo11n.pt",
            "yolo11s.pt",
        ]
        custom_best: Path = MODELS_DIR / "car_detection_best.pt"
        if custom_best.exists():
            model_paths.append(str(custom_best))

    for mp in model_paths:
        try:
            evaluate_custom_model(mp)
        except Exception as e:
            print(f"Error evaluating {mp}: {e}")


if __name__ == "__main__":
    benchmark_models()
