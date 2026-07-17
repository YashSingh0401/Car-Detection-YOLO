"""Download COCO vehicle subset and prepare for YOLO training."""
import os
import yaml
import shutil
import urllib.request
import zipfile
import json
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "car_dataset"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
CLASS_MAP = {2: 0, 3: 1, 5: 2, 7: 3}


def download_file(url, dest_path):
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"{dest_path.name} already exists, skipping download")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded {dest_path.name}")


def prepare_coco_val_subset():
    """Download COCO val2017 images and annotations, filter for vehicles."""
    coco_dir = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    img_zip = coco_dir / "val2017.zip"
    ann_zip = coco_dir / "annotations_trainval2017.zip"
    
    download_file(
        "http://images.cocodataset.org/zips/val2017.zip",
        img_zip
    )
    download_file(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ann_zip
    )
    
    img_dir = coco_dir / "val2017"
    ann_dir = coco_dir / "annotations"
    
    if not img_dir.exists():
        print("Extracting val2017 images...")
        with zipfile.ZipFile(img_zip, 'r') as z:
            z.extractall(coco_dir)
    
    ann_file = ann_dir / "instances_val2017.json"
    if not ann_file.exists():
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as z:
            z.extractall(coco_dir)
    
    print("Loading COCO annotations...")
    with open(ann_file) as f:
        coco = json.load(f)
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories'] if cat['id'] in VEHICLE_CLASSES}
    vehicle_cat_ids = list(cat_id_to_name.keys())
    
    img_id_to_info = {img['id']: img for img in coco['images']}
    
    img_vehicle_anns = {}
    for ann in coco['annotations']:
        if ann['category_id'] in vehicle_cat_ids and ann.get('iscrowd', 0) == 0:
            img_id = ann['image_id']
            if img_id not in img_vehicle_anns:
                img_vehicle_anns[img_id] = []
            img_vehicle_anns[img_id].append(ann)
    
    print(f"Found {len(img_vehicle_anns)} images containing vehicles")
    
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    img_ids = list(img_vehicle_anns.keys())
    split_idx = int(len(img_ids) * 0.8)
    
    for i, img_id in enumerate(tqdm(img_ids, desc="Processing images")):
        img_info = img_id_to_info[img_id]
        src_img = img_dir / img_info['file_name']
        if not src_img.exists():
            continue
        
        split = "train" if i < split_idx else "val"
        dst_img = DATASET_DIR / "images" / split / img_info['file_name']
        
        try:
            shutil.copy2(src_img, dst_img)
        except:
            continue
        
        label_lines = []
        for ann in img_vehicle_anns[img_id]:
            cls_id = CLASS_MAP[ann['category_id']]
            x, y, w, h = ann['bbox']
            img_w, img_h = img_info['width'], img_info['height']
            
            x_center = ((x + w / 2) / img_w)
            y_center = ((y + h / 2) / img_h)
            w_norm = w / img_w
            h_norm = h / img_h
            
            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        if label_lines:
            label_file = DATASET_DIR / "labels" / split / (Path(img_info['file_name']).stem + ".txt")
            with open(label_file, 'w') as f:
                f.write("\n".join(label_lines))
    
    yaml_content = {
        'path': str(DATASET_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    
    with open(DATASET_DIR / "dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    train_count = len([p for p in (DATASET_DIR / "images" / "train").glob("*") if p.suffix in ('.jpg', '.png')])
    val_count = len([p for p in (DATASET_DIR / "images" / "val").glob("*") if p.suffix in ('.jpg', '.png')])
    
    print(f"\nDataset ready!")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Config: {DATASET_DIR / 'dataset.yaml'}")
    
    return DATASET_DIR / "dataset.yaml"


if __name__ == "__main__":
    prepare_coco_val_subset()
