import os
import yaml
import shutil
from pathlib import Path

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "car_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

YOLO_CLASSES = ["car", "motorcycle", "bus", "truck"]
CLASS_MAP = {2: 0, 3: 1, 5: 2, 7: 3}


def download_coco_subset():
    """Download COCO validation set and filter for vehicle classes."""
    from ultralytics.utils.downloads import download
    from ultralytics.data.converter import coco80_to_coco91_class
    
    print("Downloading COCO validation images...")
    coco_dir = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = coco_dir / "images" / "val2017"
    labels_dir = coco_dir / "labels" / "val2017"
    
    if not images_dir.exists():
        download("http://images.cocodataset.org/zips/val2017.zip", dir=coco_dir / "images")
    if not labels_dir.exists():
        download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", dir=coco_dir)
    
    print("COCO dataset ready")


def convert_coco_to_yolo():
    """Convert COCO annotations to YOLO format, filtering for vehicles only."""
    from pycocotools.coco import COCO
    
    coco_dir = DATA_DIR / "coco"
    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    
    if not ann_file.exists():
        print("COCO annotations not found. Run download_coco_subset() first.")
        return
    
    coco = COCO(str(ann_file))
    
    cat_ids = coco.getCatIds(catNms=list(VEHICLE_CLASSES.values()))
    img_ids = coco.getImgIds(catIds=cat_ids)
    
    print(f"Found {len(img_ids)} images with vehicles")
    
    for split in ["train", "val"]:
        split_dir = IMAGES_DIR / split
        label_split_dir = LABELS_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        label_split_dir.mkdir(parents=True, exist_ok=True)
    
    train_count = 0
    val_count = 0
    
    for i, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        
        vehicle_anns = [a for a in anns if a['category_id'] in CLASS_MAP]
        if not vehicle_anns:
            continue
        
        is_train = i % 5 != 0
        split = "train" if is_train else "val"
        
        src_img = coco_dir / "images" / "val2017" / img_info['file_name']
        dst_img = IMAGES_DIR / split / img_info['file_name']
        shutil.copy2(src_img, dst_img)
        
        label_file = LABELS_DIR / split / (Path(img_info['file_name']).stem + ".txt")
        with open(label_file, 'w') as f:
            for ann in vehicle_anns:
                cls_id = CLASS_MAP[ann['category_id']]
                x, y, w, h = ann['bbox']
                img_w, img_h = img_info['width'], img_info['height']
                
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        if is_train:
            train_count += 1
        else:
            val_count += 1
    
    print(f"Train: {train_count}, Val: {val_count}")
    return train_count, val_count


def create_dataset_yaml():
    """Create dataset.yaml for YOLO training."""
    yaml_content = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(YOLO_CLASSES),
        'names': YOLO_CLASSES,
    }
    
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created {yaml_path}")
    return yaml_path


def verify_dataset():
    """Verify dataset structure and show statistics."""
    for split in ["train", "val"]:
        img_dir = IMAGES_DIR / split
        lbl_dir = LABELS_DIR / split
        
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))
        
        print(f"\n{split.upper()}: {len(images)} images, {len(labels)} labels")
        
        class_counts = {c: 0 for c in YOLO_CLASSES}
        for lbl in labels[:100]:
            with open(lbl) as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    class_counts[YOLO_CLASSES[cls_id]] += 1
        
        print("Class distribution (sample):", class_counts)


def main():
    print("=" * 50)
    print("CAR DETECTION DATASET PREPARATION")
    print("=" * 50)
    
    DATA_DIR.mkdir(exist_ok=True)
    
    download_coco_subset()
    convert_coco_to_yolo()
    create_dataset_yaml()
    verify_dataset()
    
    print("\nDataset ready! Use dataset.yaml for training.")


if __name__ == "__main__":
    main()