"""Train YOLO for maximum car detection accuracy."""
import os
import yaml
import logging
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log: logging.Logger = logging.getLogger("car_train")

ROOT: Path = Path(__file__).parent
DATA_DIR: Path = ROOT / "data"
MODELS_DIR: Path = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfig:
    model_name: str = "yolo11m.pt"
    dataset_yaml: Optional[Path] = None
    imgsz: int = 640
    batch: int = 8
    epochs: int = 100
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    erasing: float = 0.4
    crop_fraction: float = 1.0
    optimizer: str = "AdamW"
    device: str = "cpu"
    project: str = "car_detection"
    name: str = "experiment"
    exist_ok: bool = False
    pretrained: bool = True
    freeze: Optional[int] = None
    dropout: float = 0.0
    val: bool = True
    plots: bool = True
    save: bool = True
    save_period: int = 10
    workers: int = 0
    patience: int = 20
    seed: int = 42
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    cos_lr: bool = True
    multi_scale: bool = False
    nbs: int = 64


# ----- Accuracy-optimized presets -----

def accuracy_preset(device: str = "cuda") -> TrainingConfig:
    """Maximum accuracy: large model, full dataset, strong augmentation."""
    return TrainingConfig(
        model_name="yolo11x.pt",
        imgsz=800,
        batch=16,
        epochs=300,
        lr0=0.0005,
        lrf=0.005,
        optimizer="AdamW",
        device=device,
        name="car_accuracy",
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.2,
        scale=0.7,
        shear=5.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        erasing=0.5,
        weight_decay=0.0003,
        warmup_epochs=5,
        patience=50,
        multi_scale=True,
        deterministic=False,
        workers=8,
        nbs=128,
    )


def balanced_preset(device: str = "cpu") -> TrainingConfig:
    """Good accuracy with moderate compute requirements."""
    return TrainingConfig(
        model_name="yolo11m.pt",
        imgsz=640,
        batch=8,
        epochs=200,
        lr0=0.001,
        lrf=0.01,
        optimizer="AdamW",
        device=device,
        name="car_balanced",
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        flipud=0.05,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.4,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=30,
        workers=4,
        nbs=64,
    )


def fast_preset(device: str = "cpu") -> TrainingConfig:
    """Quick experiments: small model, fewer epochs."""
    return TrainingConfig(
        model_name="yolov8n.pt",
        imgsz=640,
        batch=8,
        epochs=50,
        lr0=0.001,
        optimizer="AdamW",
        device=device,
        name="car_fast",
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
        workers=2,
        patience=15,
    )


PRESETS: dict[str, callable] = {
    "accuracy": accuracy_preset,
    "balanced": balanced_preset,
    "fast": fast_preset,
}


def get_dataset() -> Path:
    possible_yamls: list[Path] = [
        DATA_DIR / "car_dataset" / "dataset.yaml",
        *list(Path(".").rglob("dataset.yaml")),
    ]
    for yml in possible_yamls:
        if yml.exists():
            return yml

    log.info("Dataset not found. Preparing COCO vehicle subset...")
    import prepare_data
    return prepare_data.prepare_coco_val_subset()


def train_model(cfg: TrainingConfig) -> object:
    dataset_yaml: Path = cfg.dataset_yaml or get_dataset()
    log.info("Using dataset: %s", dataset_yaml)

    with open(dataset_yaml) as f:
        data_cfg: dict = yaml.safe_load(f)
    log.info("Classes: %s", data_cfg.get("names", "N/A"))

    model: YOLO = YOLO(cfg.model_name)
    log.info("Loaded model: %s", cfg.model_name)

    device: str = cfg.device
    batch: int = cfg.batch
    if device == "cpu" or not torch.cuda.is_available():
        device = "cpu"
        if batch > 8:
            log.warning("CPU detected, reducing batch size to 8")
            batch = min(batch, 8)

    log.info("Training config: model=%s device=%s imgsz=%d batch=%d epochs=%d lr=%.4f",
             cfg.model_name, device, cfg.imgsz, batch, cfg.epochs, cfg.lr0)

    results = model.train(
        data=str(dataset_yaml),
        epochs=cfg.epochs,
        patience=cfg.patience,
        batch=batch,
        imgsz=cfg.imgsz,
        save=cfg.save,
        save_period=cfg.save_period,
        val=cfg.val,
        plots=cfg.plots,
        device=device,
        workers=cfg.workers,
        project=cfg.project,
        name=cfg.name,
        exist_ok=cfg.exist_ok,
        pretrained=cfg.pretrained,
        optimizer=cfg.optimizer,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
        single_cls=cfg.single_cls,
        rect=cfg.rect,
        cos_lr=cfg.cos_lr,
        multi_scale=cfg.multi_scale,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        box=cfg.box,
        cls=cfg.cls,
        dfl=cfg.dfl,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        erasing=cfg.erasing,
        crop_fraction=cfg.crop_fraction,
        dropout=cfg.dropout,
        freeze=cfg.freeze,
        nbs=cfg.nbs,
    )

    best_model_path: Path = Path(cfg.project) / cfg.name / "weights" / "best.pt"
    if best_model_path.exists():
        import shutil
        final_path: Path = MODELS_DIR / "car_detection_best.pt"
        shutil.copy(best_model_path, final_path)
        log.info("Best model saved to: %s", final_path)

    log.info("Training complete!")
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO for car detection")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default=None,
                        help="Training preset: accuracy, balanced, or fast")
    parser.add_argument("--model", type=str, default=None, help="Base model override")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size override")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--lr0", type=float, default=None, help="Learning rate override")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer override")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze backbone layers")
    parser.add_argument("--name", type=str, default=None, help="Experiment name override")
    parser.add_argument("--full-data", action="store_true", help="Use full COCO train2017 dataset")
    args = parser.parse_args()

    if args.full_data:
        log.info("Preparing full COCO dataset (train2017 + val2017)...")
        import prepare_data
        yaml_path = prepare_data.prepare_coco_full()
        args.data = str(Path(args.data) if args.data else yaml_path)

    if args.preset:
        preset_fn = PRESETS[args.preset]
        cfg = preset_fn(device=args.device)
        if args.model:
            cfg.model_name = args.model
        if args.data:
            cfg.dataset_yaml = Path(args.data)
        if args.imgsz is not None:
            cfg.imgsz = args.imgsz
        if args.batch is not None:
            cfg.batch = args.batch
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.lr0 is not None:
            cfg.lr0 = args.lr0
        if args.optimizer:
            cfg.optimizer = args.optimizer
        if args.freeze is not None:
            cfg.freeze = args.freeze
        if args.name:
            cfg.name = args.name
        cfg.device = args.device
    else:
        cfg = TrainingConfig(
            model_name=args.model or "yolo11m.pt",
            dataset_yaml=Path(args.data) if args.data else None,
            imgsz=args.imgsz or 640,
            batch=args.batch or 8,
            epochs=args.epochs or 100,
            lr0=args.lr0 or 0.001,
            optimizer=args.optimizer or "AdamW",
            device=args.device,
            freeze=args.freeze,
            name=args.name or "car_experiment",
        )

    train_model(cfg)


if __name__ == "__main__":
    main()
