from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_data import CLASS_NAMES, CLASS_MAP, VEHICLE_CLASSES, create_dataset_yaml, verify_dataset


def test_class_mapping():
    assert len(CLASS_NAMES) == 4
    assert CLASS_NAMES == ["car", "motorcycle", "bus", "truck"]
    assert CLASS_MAP[3] == 0
    assert CLASS_MAP[4] == 1
    assert CLASS_MAP[6] == 2
    assert CLASS_MAP[8] == 3


def test_vehicle_classes():
    assert 3 in VEHICLE_CLASSES
    assert VEHICLE_CLASSES[3] == "car"
    assert VEHICLE_CLASSES[8] == "truck"


def test_create_dataset_yaml(tmp_path):
    import shutil
    import yaml
    from prepare_data import DATASET_DIR, DATA_DIR

    original_data_dir = DATA_DIR
    original_dataset_dir = DATASET_DIR

    try:
        import prepare_data
        prepare_data.DATA_DIR = tmp_path
        prepare_data.DATASET_DIR = tmp_path / "car_dataset"
        (tmp_path / "car_dataset" / "images" / "train").mkdir(parents=True)
        (tmp_path / "car_dataset" / "images" / "val").mkdir(parents=True)
        (tmp_path / "car_dataset" / "labels" / "train").mkdir(parents=True)
        (tmp_path / "car_dataset" / "labels" / "val").mkdir(parents=True)

        yaml_path = prepare_data.create_dataset_yaml()

        assert yaml_path.exists()
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["nc"] == 4
        assert cfg["names"] == ["car", "motorcycle", "bus", "truck"]
        assert cfg["train"] == "images/train"
        assert cfg["val"] == "images/val"
    finally:
        prepare_data.DATA_DIR = original_data_dir
        prepare_data.DATASET_DIR = original_dataset_dir
