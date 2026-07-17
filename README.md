# YOLO Car Detection

A production-ready vehicle detection system using YOLOv8/YOLO11 with a Streamlit web UI, training pipeline, dataset preparation tools, and model benchmarking.

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run streamlitapp.py
```

Open http://localhost:8501

## Features

- **Web UI** — Upload images/videos, adjust sensitivity, resolution, TTA, CLAHE preprocessing
- **Multi-model support** — YOLOv8n/s/m/l/x and YOLO11n/s/m/l/x, auto-downloaded from Ultralytics Hub
- **Car confidence boost** — +30% score boost for car/bus/truck detections
- **Video processing** — Full frame-by-frame detection with ffmpeg H.264 encoding
- **Training pipeline** — Fine-tune YOLO on COCO vehicle subset (car, motorcycle, bus, truck)
- **Dataset tools** — Auto-download COCO val2017, filter vehicle classes, convert to YOLO format
- **Benchmarking** — Compare all YOLO models on mAP50/mAP50-95/precision/recall

## CLI Pipeline

```bash
python run.py prepare      # Download COCO and prepare vehicle dataset
python run.py train        # Train with CPU-friendly defaults
python run.py train-gpu    # Train with GPU-optimized settings
python run.py train-large  # Train YOLO11x with high epochs
python run.py evaluate     # Benchmark all available models
python run.py app          # Launch Streamlit UI
python run.py all          # Full pipeline
```

## Training

```bash
python train.py --model yolo11m.pt --epochs 100 --batch 8
```

Training config is managed via the `TrainingConfig` dataclass (`train.py`):

```python
from train import TrainingConfig, train_model

cfg = TrainingConfig(
    model_name="yolo11m.pt",
    epochs=200,
    batch=16,
    imgsz=640,
    device="cuda",
    optimizer="AdamW",
)
train_model(cfg)
```

The best model is saved to `models/car_detection_best.pt`.

## Dataset

The dataset pipeline downloads COCO val2017 (5,000 images), filters for vehicle classes (car, motorcycle, bus, truck), and splits into train (80%) / val (20%) in YOLO format under `data/car_dataset/`.

```python
from prepare_data import prepare_coco_val_subset
yaml_path = prepare_coco_val_subset()
```

## Project Structure

```
├── streamlitapp.py          # Streamlit web UI
├── train.py                 # Training pipeline + TrainingConfig dataclass
├── prepare_data.py          # COCO dataset download + YOLO conversion
├── evaluate.py              # Model benchmarking
├── run.py                   # CLI orchestrator
├── requirements.txt         # Pinned dependencies
├── runtime.txt              # Python version for Streamlit Cloud
├── packages.txt             # System packages (ffmpeg)
├── pyproject.toml           # Project metadata
├── .pre-commit-config.yaml  # Linting hooks
├── tests/                   # Unit tests
│   ├── test_prepare_data.py
│   ├── test_streamlitapp.py
│   └── test_train.py
├── input/                   # Uploaded test media
├── output/                  # Detection output images
├── data/                    # Datasets (COCO, car_dataset)
└── models/                  # Trained model weights
```

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Connect repo at https://streamlit.io/cloud
3. Set Python version in `runtime.txt`
4. `ffmpeg` is installed via `packages.txt`

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | Ultralytics YOLOv8 / YOLO11 |
| Web UI | Streamlit |
| Computer Vision | OpenCV |
| Dataset | COCO val2017 + pycocotools |
| Augmentation | Albumentations |
| Language | Python (3.10+) |

## Author

**Yashwardhan Singh Sengar**
