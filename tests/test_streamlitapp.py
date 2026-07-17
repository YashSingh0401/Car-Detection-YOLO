from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlitapp import apply_clahe, CLASS_COLORS, CAR_CLASS_IDS


def test_apply_clahe_rgb():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = apply_clahe(img)
    assert result.shape == img.shape
    assert result.dtype == img.dtype


def test_apply_clahe_grayscale():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = apply_clahe(img)
    assert result.shape == img.shape


def test_class_colors_keys():
    expected = {0, 1, 2, 3, 5, 7, 9, 11}
    assert set(CLASS_COLORS.keys()) == expected


def test_car_class_ids():
    assert CAR_CLASS_IDS == {2, 5, 7}


def test_apply_clahe_identity():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    result = apply_clahe(img)
    assert result.shape == (50, 50, 3)
