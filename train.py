"""Train YOLO for maximum car detection accuracy."""
import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def get_dataset():
    """Find or prepare dataset."""
    # Try to find existing dataset
    possible_yamls = [
        DATA_DIR / "car_dataset" / "dataset.yaml",
        *list(Path(".").rglob("dataset.yaml")),
    ]
    for yml in possible_yamls:
        if yml.exists():
            return yml
    
    # Prepare dataset if not found
    print("Dataset not found. Preparing COCO vehicle subset...")
    import prepare_data
    return prepare_data.prepare_coco_val_subset()


def train_model(
    model_name="yolo11m.pt",
    dataset_yaml=None,
    imgsz=640,
    batch=8,
    epochs=100,
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    erasing=0.4,
    crop_fraction=1.0,
    optimizer="AdamW",
    device="cpu",
    project="car_detection",
    name="experiment",
    exist_ok=False,
    pretrained=True,
    freeze=None,
    dropout=0.0,
    val=True,
    plots=True,
    save=True,
    save_period=10,
    workers=0,
    patience=20,
    seed=42,
    deterministic=True,
    single_cls=False,
    rect=False,
    cos_lr=True,
    multi_scale=False,
    nbs=64,
):
    """Train YOLO model with optimal hyperparameters for car detection."""
    
    dataset_yaml = dataset_yaml or get_dataset()
    print(f"Using dataset: {dataset_yaml}")
    
    # Load dataset info
    with open(dataset_yaml) as f:
        data_cfg = yaml.safe_load(f)
    print(f"Classes: {data_cfg.get('names', 'N/A')}")
    
    # Load model
    if pretrained and model_name.startswith("yolo"):
        model = YOLO(model_name)
        print(f"Loaded pre-trained model: {model_name}")
    else:
        model = YOLO(model_name)
        print(f"Loaded model: {model_name}")
    
    # Auto-compute optimal batch size if possible
    if device == "cpu" or not torch.cuda.is_available():
        device = "cpu"
        if batch > 8:
            print("CPU detected, reducing batch size to 8")
            batch = min(batch, 8)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Image Size: {imgsz}")
    print(f"  Batch Size: {batch}")
    print(f"  Epochs: {epochs}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Learning Rate: {lr0}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"{'='*60}\n")
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        save=save,
        save_period=save_period,
        val=val,
        plots=plots,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=exist_ok,
        pretrained=pretrained,
        optimizer=optimizer,
        seed=seed,
        deterministic=deterministic,
        single_cls=single_cls,
        rect=rect,
        cos_lr=cos_lr,
        multi_scale=multi_scale,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        box=box,
        cls=cls,
        dfl=dfl,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        erasing=erasing,
        crop_fraction=crop_fraction,
        dropout=dropout,
        freeze=freeze,
        nbs=nbs,
    )
    
    # Get best model path
    best_model_path = Path(project) / name / "weights" / "best.pt"
    if best_model_path.exists():
        final_path = MODELS_DIR / "car_detection_best.pt"
        import shutil
        shutil.copy(best_model_path, final_path)
        print(f"\nBest model saved to: {final_path}")
        print(f"You can now use this model in streamlitapp.py by selecting it or setting it as default.")
    
    print("\nTraining complete!")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLO for car detection")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base model")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze backbone layers")
    parser.add_argument("--name", type=str, default="car_experiment", help="Experiment name")
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        dataset_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        lr0=args.lr0,
        optimizer=args.optimizer,
        device=args.device,
        freeze=args.freeze,
        name=args.name,
    )


if __name__ == "__main__":
    main()
