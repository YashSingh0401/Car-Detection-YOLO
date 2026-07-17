"""Evaluate car detection models and compare accuracy."""
from pathlib import Path
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
    import yaml

    if dataset_yaml is None:
        dataset_yaml = ROOT / "data" / "car_dataset" / "dataset.yaml"

    is_local = Path(model_path).exists()
    
    try:
        model: YOLO = YOLO(str(model_path))
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    val_dataset = "coco8.yaml"
    if dataset_yaml.exists() and is_local:
        try:
            with open(dataset_yaml) as f:
                data_cfg = yaml.safe_load(f)
            if len(model.names) == data_cfg.get("nc", 80):
                val_dataset = str(dataset_yaml)
            else:
                print(f"Class count mismatch (Model: {len(model.names)}, Dataset: {data_cfg.get('nc')}). Evaluating {model_path} on coco8.yaml instead.")
        except Exception as e:
            print(f"Error parsing dataset YAML: {e}. Defaulting to coco8.yaml")
    else:
        if not is_local:
            print(f"{model_path} is a pretrained model. Evaluating on coco8.yaml.")

    print(f"\nEvaluating {model_path} on {val_dataset}...")

    metrics = model.val(
        data=val_dataset,
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
