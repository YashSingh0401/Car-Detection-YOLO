"""Download COCO vehicle subset and prepare for YOLO training."""
import logging
import yaml
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log: logging.Logger = logging.getLogger("car_prep")

DATA_DIR: Path = Path("data")
DATASET_DIR: Path = DATA_DIR / "car_dataset"

VEHICLE_CLASSES: dict[int, str] = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}
CLASS_NAMES: list[str] = ["car", "motorcycle", "bus", "truck"]
CLASS_MAP: dict[int, int] = {3: 0, 4: 1, 6: 2, 8: 3}

# Use COCO train2017 for training (118k images) and val2017 for validation
COCO_TRAIN_URL: str = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_URL: str = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL: str = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


class DownloadProgress:
    def __init__(self, desc: str) -> None:
        self.pbar: Optional[tqdm] = None
        self.desc: str = desc

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=self.desc)
        downloaded: int = block_num * block_size
        if self.pbar is not None:
            self.pbar.update(downloaded - self.pbar.n)
            if downloaded >= self.pbar.total:
                self.pbar.close()
                self.pbar = None


def download_file(url: str, dest_path: Path) -> None:
    dest_path = Path(dest_path)
    if dest_path.exists():
        log.info("%s already exists, skipping download", dest_path.name)
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    desc: str = f"Downloading {dest_path.name}"
    urllib.request.urlretrieve(url, dest_path, reporthook=DownloadProgress(desc))
    log.info("Downloaded %s", dest_path.name)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    dest_dir: Path = extract_dir / Path(zip_path.stem).stem
    if dest_dir.exists():
        log.info("%s already exists, skipping extraction", dest_dir.name)
        return
    log.info("Extracting %s...", zip_path.name)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)


def verify_dataset() -> None:
    for split in ["train", "val"]:
        img_dir: Path = DATASET_DIR / "images" / split
        lbl_dir: Path = DATASET_DIR / "labels" / split
        images: list[Path] = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels: list[Path] = list(lbl_dir.glob("*.txt"))
        log.info("%s: %d images, %d labels", split.upper(), len(images), len(labels))
        class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
        for lbl in labels[:200]:
            with open(lbl) as f:
                for line in f:
                    cls_id: int = int(line.split()[0])
                    class_counts[CLASS_NAMES[cls_id]] += 1
        log.info("Class distribution (sample): %s", class_counts)


def create_dataset_yaml(train_split: str = "images/train", val_split: str = "images/val") -> Path:
    yaml_content: dict = {
        'path': str(DATASET_DIR.resolve()),
        'train': train_split,
        'val': val_split,
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    yaml_path: Path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return yaml_path


def _filter_coco_annotations(ann_file: Path, coco_dir: Path, img_dir: Path, use_as_val: bool) -> int:
    """Filter COCO annotations for vehicle classes and write YOLO labels."""
    from pycocotools.coco import COCO
    coco: COCO = COCO(str(ann_file))
    cat_ids: list[int] = coco.getCatIds(catNms=list(VEHICLE_CLASSES.values()))
    all_img_ids_set = set()
    for cat_id in cat_ids:
        all_img_ids_set.update(coco.getImgIds(catIds=[cat_id]))
    all_img_ids: list[int] = sorted(list(all_img_ids_set))
    print(f"Found {len(all_img_ids)} images containing vehicles")

    split: str = "val" if use_as_val else "train"
    out_img_dir: Path = DATASET_DIR / "images" / split
    out_lbl_dir: Path = DATASET_DIR / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    processed: int = 0
    for img_id in tqdm(all_img_ids, desc=f"Processing {split} images"):
        img_info: dict = coco.loadImgs(img_id)[0]
        src_img: Path = img_dir / img_info['file_name']
        if not src_img.exists():
            continue
        dst_img: Path = out_img_dir / img_info['file_name']
        try:
            shutil.copy2(src_img, dst_img)
        except OSError:
            continue

        ann_ids: list[int] = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns: list[dict] = coco.loadAnns(ann_ids)

        label_lines: list[str] = []
        for ann in anns:
            if ann['category_id'] not in CLASS_MAP:
                continue
            cls_id: int = CLASS_MAP[ann['category_id']]
            x, y, w, h = ann['bbox']
            img_w: int = img_info['width']
            img_h: int = img_info['height']
            x_center: float = (x + w / 2) / img_w
            y_center: float = (y + h / 2) / img_h
            w_norm: float = w / img_w
            h_norm: float = h / img_h
            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if label_lines:
            label_file: Path = out_lbl_dir / (Path(img_info['file_name']).stem + ".txt")
            with open(label_file, 'w') as f:
                f.write("\n".join(label_lines))
        processed += 1
    return processed


def prepare_coco_val_subset() -> Path:
    """Quick dataset: COCO val2017 only (~5k images), split 80/20 train/val."""
    from pycocotools.coco import COCO

    coco_dir: Path = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    download_file(COCO_VAL_URL, coco_dir / "val2017.zip")
    download_file(COCO_ANN_URL, coco_dir / "annotations_trainval2017.zip")

    img_dir: Path = coco_dir / "val2017"
    extract_zip(coco_dir / "val2017.zip", coco_dir)

    ann_dir: Path = coco_dir / "annotations"
    ann_file: Path = ann_dir / "instances_val2017.json"
    if not ann_file.exists():
        extract_zip(coco_dir / "annotations_trainval2017.zip", coco_dir)

    coco: COCO = COCO(str(ann_file))
    cat_ids: list[int] = coco.getCatIds(catNms=list(VEHICLE_CLASSES.values()))
    all_img_ids_set = set()
    for cat_id in cat_ids:
        all_img_ids_set.update(coco.getImgIds(catIds=[cat_id]))
    all_img_ids: list[int] = sorted(list(all_img_ids_set))
    log.info("Found %d images containing vehicles", len(all_img_ids))

    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_idx: int = int(len(all_img_ids) * 0.8)
    for i, img_id in enumerate(tqdm(all_img_ids, desc="Processing images")):
        img_info: dict = coco.loadImgs(img_id)[0]
        src_img: Path = img_dir / img_info['file_name']
        if not src_img.exists():
            continue

        split: str = "train" if i < split_idx else "val"
        dst_img: Path = DATASET_DIR / "images" / split / img_info['file_name']
        try:
            shutil.copy2(src_img, dst_img)
        except OSError:
            continue

        ann_ids: list[int] = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns: list[dict] = coco.loadAnns(ann_ids)

        label_lines: list[str] = []
        for ann in anns:
            if ann['category_id'] not in CLASS_MAP:
                continue
            cls_id: int = CLASS_MAP[ann['category_id']]
            x, y, w, h = ann['bbox']
            img_w, img_h = img_info['width'], img_info['height']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if label_lines:
            label_file: Path = DATASET_DIR / "labels" / split / (Path(img_info['file_name']).stem + ".txt")
            with open(label_file, 'w') as f:
                f.write("\n".join(label_lines))

    create_dataset_yaml()
    train_count: int = len([p for p in (DATASET_DIR / "images" / "train").glob("*") if p.suffix in ('.jpg', '.png')])
    val_count: int = len([p for p in (DATASET_DIR / "images" / "val").glob("*") if p.suffix in ('.jpg', '.png')])

    log.info("Dataset ready (quick mode)! train=%d val=%d classes=%s",
             train_count, val_count, CLASS_NAMES)
    verify_dataset()
    return DATASET_DIR / "dataset.yaml"


def prepare_coco_full() -> Path:
    """Full dataset: COCO train2017 (~118k images) + val2017 (~5k images)."""
    coco_dir: Path = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    # Download
    download_file(COCO_TRAIN_URL, coco_dir / "train2017.zip")
    download_file(COCO_VAL_URL, coco_dir / "val2017.zip")
    download_file(COCO_ANN_URL, coco_dir / "annotations_trainval2017.zip")

    # Extract
    train_img_dir: Path = coco_dir / "train2017"
    extract_zip(coco_dir / "train2017.zip", coco_dir)
    val_img_dir: Path = coco_dir / "val2017"
    extract_zip(coco_dir / "val2017.zip", coco_dir)

    ann_dir: Path = coco_dir / "annotations"
    train_ann: Path = ann_dir / "instances_train2017.json"
    val_ann: Path = ann_dir / "instances_val2017.json"
    if not train_ann.exists() or not val_ann.exists():
        extract_zip(coco_dir / "annotations_trainval2017.zip", coco_dir)

    log.info("Processing training set (COCO train2017)...")
    train_count: int = _filter_coco_annotations(train_ann, coco_dir, train_img_dir, use_as_val=False)

    log.info("Processing validation set (COCO val2017)...")
    val_count: int = _filter_coco_annotations(val_ann, coco_dir, val_img_dir, use_as_val=True)

    create_dataset_yaml()

    log.info("Full dataset ready! train=%d val=%d classes=%s",
             train_count, val_count, CLASS_NAMES)
    verify_dataset()
    return DATASET_DIR / "dataset.yaml"


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if mode == "full":
        prepare_coco_full()
    else:
        prepare_coco_val_subset()
