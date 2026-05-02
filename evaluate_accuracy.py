#!/usr/bin/env python3
"""
Evaluate overall classification accuracy (top-1 / top-5) on validation splits.

Usage (from repo root):
  myenv\\Scripts\\python.exe evaluate_accuracy.py

Optional:
  myenv\\Scripts\\python.exe evaluate_accuracy.py --model backend\\yolo11n-cls.pt
  myenv\\Scripts\\python.exe evaluate_accuracy.py --data dataset --data dataset_original_split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "backend" / "yolo11n-cls.pt"
DEFAULT_DATASETS: list[tuple[str, Path]] = [
    ("dataset", ROOT / "dataset"),
    ("dataset_original_split", ROOT / "dataset_original_split"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Print YOLO-cls validation accuracy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to .pt weights (default: backend/yolo11n-cls.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Dataset root with train/ and val/ (can pass multiple). Default: dataset + dataset_original_split",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1

    if args.data:
        datasets: list[tuple[str, Path]] = []
        for p in args.data:
            rp = p.resolve()
            if not rp.is_dir():
                print(f"Error: data path not found: {rp}", file=sys.stderr)
                return 1
            datasets.append((rp.name, rp))
    else:
        datasets = [(name, path.resolve()) for name, path in DEFAULT_DATASETS]
        for name, path in datasets:
            if not path.is_dir():
                print(f"Error: default dataset missing: {path}", file=sys.stderr)
                return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("Error: ultralytics not installed. Use the project venv:", file=sys.stderr)
        print("  myenv\\Scripts\\python.exe evaluate_accuracy.py", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    print(f"Model: {model_path}")
    print()

    model = YOLO(str(model_path))

    for label, data_root in datasets:
        val_dir = data_root / "val"
        if not val_dir.is_dir():
            print(f"[{label}] skip: no val folder at {val_dir}")
            continue
        results = model.val(data=str(data_root), split="val", verbose=False)
        top1 = float(results.top1)
        top5 = float(results.top5)
        print(f"[{label}]")
        print(f"  Top-1 accuracy: {top1 * 100:.2f}%")
        print(f"  Top-5 accuracy: {top5 * 100:.2f}%")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
