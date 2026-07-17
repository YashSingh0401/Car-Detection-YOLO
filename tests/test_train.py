from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import TrainingConfig, accuracy_preset, balanced_preset, fast_preset, PRESETS


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.model_name == "yolo11m.pt"
    assert cfg.imgsz == 640
    assert cfg.batch == 8
    assert cfg.epochs == 100
    assert cfg.optimizer == "AdamW"
    assert cfg.device == "cpu"
    assert cfg.patience == 20
    assert cfg.seed == 42


def test_training_config_custom():
    cfg = TrainingConfig(
        model_name="yolov8n.pt",
        imgsz=320,
        batch=16,
        epochs=50,
        device="cuda",
    )
    assert cfg.model_name == "yolov8n.pt"
    assert cfg.imgsz == 320
    assert cfg.batch == 16
    assert cfg.epochs == 50
    assert cfg.device == "cuda"


def test_training_config_optional_fields():
    cfg = TrainingConfig(freeze=10, dropout=0.2)
    assert cfg.freeze == 10
    assert cfg.dropout == 0.2


def test_accuracy_preset():
    cfg = accuracy_preset(device="cuda")
    assert cfg.model_name == "yolo11x.pt"
    assert cfg.imgsz == 800
    assert cfg.epochs == 300
    assert cfg.patience == 50
    assert cfg.multi_scale == True
    assert cfg.hsv_h == 0.02
    assert cfg.degrees == 10.0
    assert cfg.mixup == 0.2
    assert cfg.copy_paste == 0.2


def test_balanced_preset():
    cfg = balanced_preset(device="cpu")
    assert cfg.model_name == "yolo11m.pt"
    assert cfg.imgsz == 640
    assert cfg.epochs == 200
    assert cfg.patience == 30
    assert cfg.optimizer == "AdamW"


def test_fast_preset():
    cfg = fast_preset(device="cpu")
    assert cfg.model_name == "yolov8n.pt"
    assert cfg.epochs == 50
    assert cfg.mosaic == 0.5


def test_presets_dict():
    assert "accuracy" in PRESETS
    assert "balanced" in PRESETS
    assert "fast" in PRESETS
    assert callable(PRESETS["accuracy"])
    assert callable(PRESETS["balanced"])
    assert callable(PRESETS["fast"])
