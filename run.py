"""
Car Detection YOLO - Complete Pipeline
=======================================
1. Prepare COCO vehicle dataset
2. Train model with optimized hyperparameters
3. Evaluate model
4. Run app

Usage:
  python run.py prepare     - Download and prepare dataset
  python run.py train       - Train model (CPU-friendly defaults)
  python run.py train-gpu   - Train with GPU-optimized settings
  python run.py evaluate    - Evaluate trained model
  python run.py app         - Launch Streamlit app
  python run.py all         - Run full pipeline
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

    elif command == "train":
        run_script("train.py", "--epochs", "50", "--batch", "8", "--name", "car_v1")

    elif command == "train-gpu":
        run_script("train.py",
            "--model", "yolo11m.pt",
            "--epochs", "200",
            "--batch", "32",
            "--imgsz", "640",
            "--lr0", "0.001",
            "--optimizer", "AdamW",
            "--device", "0",
            "--name", "car_v1_gpu"
        )

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

    elif command == "evaluate":
        run_script("evaluate.py")

    elif command == "app":
        streamlit: str = str(ROOT / "venv" / "Scripts" / "streamlit")
        if not Path(streamlit).exists():
            streamlit = "streamlit"
        subprocess.run([streamlit, "run", str(ROOT / "streamlitapp.py")])

    elif command == "all":
        run_script("prepare_data.py")
        run_script("train.py", "--epochs", "100", "--batch", "8", "--name", "car_v1")
        run_script("evaluate.py")
        print("\nTraining complete! Run 'python run.py app' to launch the detection app.")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
