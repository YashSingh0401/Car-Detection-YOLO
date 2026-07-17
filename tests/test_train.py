from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import TrainingConfig


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
