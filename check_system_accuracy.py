#!/usr/bin/env python3
"""
Check overall validation accuracy and record system / training context.

Reports:
  - Host environment (OS, Python, PyTorch, device, Ultralytics)
  - Checkpoint metadata when present (last training epoch, date, train_args)
  - Optional training log summary from Ultralytics results.csv (best epoch / last row)
  - Top-1 and Top-5 accuracy on each dataset root (train/ + val/ layout)

Usage (from repo root):
  python check_system_accuracy.py
  python check_system_accuracy.py --model backend/yolo11n-cls.pt
  python check_system_accuracy.py --data path/to/dataset
  python check_system_accuracy.py --results-csv runs/classify/train/results.csv
  python check_system_accuracy.py --json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "backend" / "yolo11n-cls.pt"
DEFAULT_DATASETS: list[tuple[str, Path]] = [
    ("dataset", ROOT / "dataset"),
    ("dataset_original_split", ROOT / "dataset_original_split"),
]


def print_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None

    print("=== System ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()
    return info


def load_checkpoint_meta(model_path: Path) -> dict[str, Any]:
    """Best-effort read of Ultralytics / PyTorch checkpoint dict."""
    out: dict[str, Any] = {"weights": str(model_path)}
    try:
        import torch
    except ImportError:
        out["error"] = "torch not installed"
        return out

    try:
        kwargs: dict[str, Any] = {"map_location": "cpu"}
        try:
            ckpt = torch.load(model_path, weights_only=False, **kwargs)  # type: ignore[call-arg]
        except TypeError:
            ckpt = torch.load(model_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out

    if not isinstance(ckpt, dict):
        out["note"] = "checkpoint is not a dict; no epoch metadata"
        return out

    if "epoch" in ckpt:
        out["epoch"] = ckpt["epoch"]
    if "best_fitness" in ckpt:
        out["best_fitness"] = ckpt["best_fitness"]
    if "date" in ckpt:
        out["date"] = ckpt["date"]
    ta = ckpt.get("train_args")
    if ta is not None:
        if hasattr(ta, "items"):
            out["train_args"] = dict(ta.items()) if hasattr(ta, "items") else str(ta)
        else:
            out["train_args"] = str(ta)
    return out


def print_checkpoint_meta(meta: dict[str, Any]) -> None:
    print("=== Weights / training checkpoint ===")
    for k, v in meta.items():
        if k == "train_args" and isinstance(v, dict):
            print(f"  {k}:")
            for ak, av in list(v.items())[:12]:
                print(f"    {ak}: {av}")
            if len(v) > 12:
                print(f"    ... ({len(v) - 12} more keys)")
        else:
            print(f"  {k}: {v}")
    print()


def summarize_results_csv(csv_path: Path) -> dict[str, Any] | None:
    """Summarize last row of Ultralytics results.csv (epoch + common metric columns)."""
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]
    # normalize keys (strip BOM from first column name)
    last = {k.lstrip("\ufeff"): v for k, v in last.items()}
    epoch_raw = last.get("epoch", "")
    try:
        epoch = int(float(epoch_raw))
    except (TypeError, ValueError):
        epoch = None
    summary: dict[str, Any] = {
        "results_csv": str(csv_path.resolve()),
        "rows": len(rows),
        "last_epoch": epoch,
        "last_row_sample": {k: last[k] for k in list(last.keys())[:16]},
    }
    for key in last:
        lk = key.lower()
        if "top1" in lk or "accuracy_top1" in lk or lk.endswith("acc"):
            if last[key] not in ("", None):
                summary[f"last_{key}"] = last[key]
    return summary


def discover_results_csv(repo_root: Path) -> Path | None:
    candidates = sorted(repo_root.glob("runs/classify/**/results.csv"))
    if not candidates:
        candidates = sorted(repo_root.glob("runs/detect/**/results.csv"))
    return candidates[-1] if candidates else None


def run_validation(model_path: Path, datasets: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rows_out: list[dict[str, Any]] = []
    for label, data_root in datasets:
        val_dir = data_root / "val"
        row: dict[str, Any] = {"dataset": label, "root": str(data_root)}
        if not val_dir.is_dir():
            row["skipped"] = f"no val folder at {val_dir}"
            rows_out.append(row)
            continue
        results = model.val(data=str(data_root), split="val", verbose=False)
        top1 = float(results.top1)
        top5 = float(results.top5)
        row["top1"] = top1
        row["top5"] = top5
        row["top1_percent"] = round(top1 * 100, 4)
        row["top5_percent"] = round(top5 * 100, 4)
        rows_out.append(row)
    return rows_out


def print_validation(rows: list[dict[str, Any]]) -> None:
    print("=== Validation accuracy (Ultralytics val, split=val) ===")
    for row in rows:
        label = row["dataset"]
        if "skipped" in row:
            print(f"  [{label}] {row['skipped']}")
            continue
        print(f"  [{label}]")
        print(f"    Top-1: {row['top1_percent']:.2f}%")
        print(f"    Top-5: {row['top5_percent']:.2f}%")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="System info + checkpoint epoch + validation accuracy.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to .pt weights")
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Dataset root with train/ and val/ (repeatable). Defaults: dataset + dataset_original_split",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Ultralytics results.csv from a training run (epoch curves).",
    )
    parser.add_argument(
        "--no-discover-csv",
        action="store_true",
        help="Do not auto-pick runs/**/results.csv when --results-csv is omitted.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object with all sections (no human banners).",
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
                datasets = [(n, p) for n, p in datasets if p.is_dir()]
                if not datasets:
                    print(
                        "Error: no default dataset folders found. Pass --data <root> or create dataset/val.",
                        file=sys.stderr,
                    )
                    return 1
                break

    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as exc:
        print("Error: ultralytics not installed.", file=sys.stderr)
        print("  python -m pip install -r backend/requirements.txt", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    system = print_system_info() if not args.json else _system_dict()

    csv_path = args.results_csv
    if csv_path is None and not args.no_discover_csv:
        csv_path = discover_results_csv(ROOT)
    csv_summary: dict[str, Any] | None = None
    if csv_path is not None:
        csv_summary = summarize_results_csv(csv_path.resolve())

    ckpt_meta = load_checkpoint_meta(model_path)

    if args.json:
        val_rows = run_validation(model_path, datasets)
        payload = {
            "system": system if isinstance(system, dict) else _system_dict(),
            "checkpoint": ckpt_meta,
            "training_log": csv_summary,
            "validation": val_rows,
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print_checkpoint_meta(ckpt_meta)

    if csv_summary:
        print("=== Training run log (results.csv) ===")
        for k, v in csv_summary.items():
            if k == "last_row_sample":
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk}: {sv}")
            else:
                print(f"  {k}: {v}")
        print()
    elif args.results_csv is not None:
        print(f"=== Training run log ===\n  (missing file) {args.results_csv}\n")

    val_rows = run_validation(model_path, datasets)
    print_validation(val_rows)

    return 0


def _system_dict() -> dict[str, Any]:
    out: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        out["torch"] = None
    try:
        import ultralytics

        out["ultralytics"] = ultralytics.__version__
    except ImportError:
        out["ultralytics"] = None
    return out


if __name__ == "__main__":
    raise SystemExit(main())
