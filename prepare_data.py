"""Download COCO vehicle subset and prepare for YOLO training."""
import os
import yaml
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm

DATA_DIR: Path = Path("data")
DATASET_DIR: Path = DATA_DIR / "car_dataset"

VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CLASS_NAMES: list[str] = ["car", "motorcycle", "bus", "truck"]
CLASS_MAP: dict[int, int] = {2: 0, 3: 1, 5: 2, 7: 3}


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
        print(f"{dest_path.name} already exists, skipping download")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    desc: str = f"Downloading {dest_path.name}"
    urllib.request.urlretrieve(url, dest_path, reporthook=DownloadProgress(desc))
    print(f"Downloaded {dest_path.name}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        print(f"{extract_dir.name} already exists, skipping extraction")
        return
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)


def verify_dataset() -> None:
    for split in ["train", "val"]:
        img_dir: Path = DATASET_DIR / "images" / split
        lbl_dir: Path = DATASET_DIR / "labels" / split
        images: list[Path] = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels: list[Path] = list(lbl_dir.glob("*.txt"))
        print(f"\n{split.upper()}: {len(images)} images, {len(labels)} labels")
        class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
        for lbl in labels[:100]:
            with open(lbl) as f:
                for line in f:
                    cls_id: int = int(line.split()[0])
                    class_counts[CLASS_NAMES[cls_id]] += 1
        print("Class distribution (sample):", class_counts)


def create_dataset_yaml() -> Path:
    yaml_content: dict = {
        'path': str(DATASET_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    yaml_path: Path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return yaml_path


def prepare_coco_val_subset() -> Path:
    from pycocotools.coco import COCO

    coco_dir: Path = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    img_zip: Path = coco_dir / "val2017.zip"
    ann_zip: Path = coco_dir / "annotations_trainval2017.zip"

    download_file("http://images.cocodataset.org/zips/val2017.zip", img_zip)
    download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)

    img_dir: Path = coco_dir / "val2017"
    extract_zip(img_zip, coco_dir)

    ann_dir: Path = coco_dir / "annotations"
    ann_file: Path = ann_dir / "instances_val2017.json"
    if not ann_file.exists():
        extract_zip(ann_zip, coco_dir)

    coco: COCO = COCO(str(ann_file))

    cat_ids: list[int] = coco.getCatIds(catNms=list(VEHICLE_CLASSES.values()))
    all_img_ids: list[int] = coco.getImgIds(catIds=cat_ids)
    print(f"Found {len(all_img_ids)} images containing vehicles")

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
            img_w: int = img_info['width']
            img_h: int = img_info['height']
            x_center: float = (x + w / 2) / img_w
            y_center: float = (y + h / 2) / img_h
            w_norm: float = w / img_w
            h_norm: float = h / img_h
            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        if label_lines:
            label_file: Path = DATASET_DIR / "labels" / split / (Path(img_info['file_name']).stem + ".txt")
            with open(label_file, 'w') as f:
                f.write("\n".join(label_lines))

    yaml_path: Path = create_dataset_yaml()

    train_count: int = len([p for p in (DATASET_DIR / "images" / "train").glob("*") if p.suffix in ('.jpg', '.png')])
    val_count: int = len([p for p in (DATASET_DIR / "images" / "val").glob("*") if p.suffix in ('.jpg', '.png')])

    print(f"\nDataset ready!")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Config: {DATASET_DIR / 'dataset.yaml'}")
    verify_dataset()

    return DATASET_DIR / "dataset.yaml"


if __name__ == "__main__":
    prepare_coco_val_subset()
