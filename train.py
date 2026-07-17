"""Train YOLO for maximum car detection accuracy."""
import os
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import YOLO

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


def get_dataset() -> Path:
    possible_yamls: list[Path] = [
        DATA_DIR / "car_dataset" / "dataset.yaml",
        *list(Path(".").rglob("dataset.yaml")),
    ]
    for yml in possible_yamls:
        if yml.exists():
            return yml

    print("Dataset not found. Preparing COCO vehicle subset...")
    import prepare_data
    return prepare_data.prepare_coco_val_subset()


def train_model(cfg: TrainingConfig) -> object:
    dataset_yaml: Path = cfg.dataset_yaml or get_dataset()
    print(f"Using dataset: {dataset_yaml}")

    with open(dataset_yaml) as f:
        data_cfg: dict = yaml.safe_load(f)
    print(f"Classes: {data_cfg.get('names', 'N/A')}")

    model: YOLO = YOLO(cfg.model_name)
    print(f"Loaded model: {cfg.model_name}")

    device: str = cfg.device
    batch: int = cfg.batch
    if device == "cpu" or not torch.cuda.is_available():
        device = "cpu"
        if batch > 8:
            print("CPU detected, reducing batch size to 8")
            batch = min(batch, 8)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Device: {device}")
    print(f"  Image Size: {cfg.imgsz}")
    print(f"  Batch Size: {batch}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Optimizer: {cfg.optimizer}")
    print(f"  Learning Rate: {cfg.lr0}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"{'='*60}\n")

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
        print(f"\nBest model saved to: {final_path}")

    print("\nTraining complete!")
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO for car detection")
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--name", type=str, default="car_experiment")
    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name=args.model,
        dataset_yaml=Path(args.data) if args.data else None,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        lr0=args.lr0,
        optimizer=args.optimizer,
        device=args.device,
        freeze=args.freeze,
        name=args.name,
    )
    train_model(cfg)


if __name__ == "__main__":
    main()
