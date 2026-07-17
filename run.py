"""
Car Detection YOLO - Complete Pipeline
=======================================
Usage:
  python run.py prepare          - Quick dataset (COCO val2017, ~5k images)
  python run.py prepare-full     - Full dataset (COCO train2017, ~118k images, ~18GB)
  python run.py train            - Train with balanced defaults
  python run.py train-balance    - Train with balanced preset (good accuracy)
  python run.py train-accuracy   - Train with max accuracy preset (GPU recommended)
  python run.py train-gpu        - Train with GPU-optimized settings
  python run.py train-large      - Train YOLO11x with high epochs
  python run.py evaluate         - Evaluate trained model
  python run.py app              - Launch Streamlit app
  python run.py all              - Full pipeline (quick dataset + balanced train)
"""
import sys
import subprocess
from pathlib import Path
from typing import NoReturn

ROOT: Path = Path(__file__).parent


def run_script(script_name: str, *args: str) -> None:
    python: str = sys.executable
    cmd: list[str] = [python, str(ROOT / script_name)] + list(args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command: str = sys.argv[1]

    if command == "prepare":
        run_script("prepare_data.py")

    elif command == "prepare-full":
        run_script("prepare_data.py", "full")

    elif command == "train":
        run_script("train.py", "--preset", "balanced", "--name", "car_v1")

    elif command == "train-accuracy":
        run_script("train.py", "--preset", "accuracy",
            "--full-data", "--name", "car_accuracy_v1")

    elif command == "train-gpu":
        run_script("train.py", "--preset", "balanced",
            "--device", "0", "--batch", "32", "--epochs", "200", "--name", "car_v1_gpu")

    elif command == "train-large":
        run_script("train.py",
            "--model", "yolo11x.pt",
            "--epochs", "300",
            "--batch", "16",
            "--imgsz", "640",
            "--lr0", "0.0005",
            "--optimizer", "AdamW",
            "--device", "0",
            "--name", "car_v1_large"
        )

    elif command == "train-balance":
        run_script("train.py", "--preset", "balanced", "--name", "car_balanced_v1")

    elif command == "evaluate":
        run_script("evaluate.py")

    elif command == "app":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "streamlitapp.py")])

    elif command == "all":
        run_script("prepare_data.py")
        run_script("train.py", "--preset", "balanced", "--name", "car_v1")
        run_script("evaluate.py")
        print("\nPipeline complete! Run 'python run.py app' to launch the detection app.")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
