"""Batch runner for EB2B on simulated degradation datasets."""

from __future__ import annotations

import argparse
import csv
import json
import dataclasses
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

if __package__ is None or __package__ == "":
    import sys

    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root))

    from trainer import EBTrainer, EBTrainingConfig  # type: ignore
    from utils import ensure_dir, calculate_metrics  # type: ignore
    from main import select_device  # type: ignore
    from losses import SSIMLoss  # type: ignore
else:
    from .trainer import EBTrainer, EBTrainingConfig
    from .utils import ensure_dir, calculate_metrics
    from .losses import SSIMLoss
    from .main import select_device

CONFIG_DEFAULT = Path("EB2B/Config/superres_datasets.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EB2B on simulated degradation datasets")
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT, help="Path to dataset JSON config")
    parser.add_argument("--dataset", nargs="*", help="Subset of dataset names to process")
    parser.add_argument("--kernels", nargs="*", help="Subset of kernel directories to process (e.g., kernel_01)")
    parser.add_argument("--device", type=str, default="auto", help="Computation device: auto|cpu|cuda")
    parser.add_argument("--no-save-output", action="store_true", help="Disable saving reconstructed images")
    parser.add_argument("--output-root", type=Path, default=None, help="Override output root directory")
    return parser


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gather_kernel_dirs(dataset_root: Path, kernels_filter: Optional[Iterable[str]]) -> List[Path]:
    if kernels_filter:
        allowed = {str(k) for k in kernels_filter}
        dirs = [dataset_root / k for k in allowed]
    else:
        dirs = sorted(p for p in dataset_root.glob("kernel_*") if p.is_dir())
    return [d for d in dirs if d.is_dir()]


def compute_metrics(device: torch.device, ssim_metric: SSIMLoss, sr_img: np.ndarray, hr_path: Optional[Path], border: int) -> Dict[str, str]:
    if hr_path is None:
        return {"psnr_rgb": "", "psnr_y": "", "mse_rgb": "", "mse_y": "", "ssim": ""}

    hr_uint8 = np.array(Image.open(hr_path).convert("RGB"), dtype=np.uint8)
    metrics = calculate_metrics(hr_uint8, sr_img, border=border)

    hr_tensor = torch.from_numpy(hr_uint8.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    sr_tensor = torch.from_numpy(sr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        ssim_val = float(ssim_metric(sr_tensor, hr_tensor).item())

    metrics.update({
        "psnr_rgb": f"{metrics['psnr_rgb']:.4f}",
        "psnr_y": f"{metrics['psnr_y']:.4f}",
        "mse_rgb": f"{metrics['mse_rgb']:.6f}",
        "mse_y": f"{metrics['mse_y']:.6f}",
        "ssim": f"{ssim_val:.4f}"
    })
    return metrics


def run_from_config(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    data_root = Path(cfg.get("simdata_root", "/mnt/DATA/DegradedResolution")).expanduser()
    if not data_root.exists():
        raise FileNotFoundError(f"Simulated data root not found: {data_root}")

    output_root = args.output_root.expanduser() if args.output_root else Path(cfg.get("output_root", "outputs")).expanduser()
    ensure_dir(output_root)

    save_results = not args.no_save_output
    device = select_device(args.device)
    ssim_metric = SSIMLoss().to(device)
    ssim_metric.eval()

    default_params = cfg.get("default_params", {})
    artifact_cfg = cfg.get("artifact_layout", {})
    lr_dir_name = artifact_cfg.get("lr_dir", "LR")
    hr_dir_name = artifact_cfg.get("hr_dir", "HR")
    recon_dir_name = artifact_cfg.get("recon_dir", "Recon")

    config_fields = {field.name for field in dataclasses.fields(EBTrainingConfig)}

    dataset_entries = cfg.get("datasets", [])
    if not dataset_entries:
        print("No dataset entries found in configuration; nothing to run.")
        return

    requested_datasets = set(args.dataset) if args.dataset else None
    requested_kernels = set(args.kernels) if args.kernels else None

    for entry in dataset_entries:
        name = entry.get("name")
        if requested_datasets and name not in requested_datasets:
            continue

        sim_dataset_root = data_root / (entry.get("output_subdir") or name or "")
        if not sim_dataset_root.exists():
            print(f"[WARN] Simulated dataset directory missing: {sim_dataset_root}")
            continue

        kernel_dirs = gather_kernel_dirs(sim_dataset_root, requested_kernels)
        if not kernel_dirs:
            print(f"[WARN] No kernel directories found under {sim_dataset_root}")
            continue

        dataset_output_root = output_root / (entry.get("output_subdir") or name or sim_dataset_root.name)
        ensure_dir(dataset_output_root)

        for kernel_dir in kernel_dirs:
            manifest_path = kernel_dir / "manifest.csv"
            if not manifest_path.exists():
                print(f"  [WARN] Manifest missing for {kernel_dir}")
                continue

            results_dir = dataset_output_root / kernel_dir.name
            lr_output_dir = results_dir / lr_dir_name
            recon_output_dir = results_dir / recon_dir_name
            hr_output_dir = results_dir / hr_dir_name

            if save_results:
                for path in (lr_output_dir, recon_output_dir, hr_output_dir):
                    ensure_dir(path)

            entry_params = dict(default_params)
            entry_params.update(entry.get("params", {}))
            entry_params["save_output"] = False
            config_params = {k: entry_params[k] for k in entry_params if k in config_fields}

            stats_rows: List[Dict[str, str]] = []
            processed = 0

            with manifest_path.open("r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    hr_rel = row.get("hr_path")
                    lr_rel = row.get("lr_path")
                    if not hr_rel or not lr_rel:
                        continue

                    hr_path = (kernel_dir / hr_rel).resolve()
                    lr_path = (kernel_dir / lr_rel).resolve()
                    if not hr_path.exists() or not lr_path.exists():
                        print(f"    [WARN] Missing HR/LR for row {row}")
                        continue

                    try:
                        scale = int(row.get("scale", config_params.get("scale_factor", 1)))
                    except (TypeError, ValueError):
                        print(f"    [WARN] Invalid scale in row: {row}")
                        continue

                    params = dict(config_params)
                    params["scale_factor"] = scale
                    trainer_config = EBTrainingConfig(**params)

                    trainer = EBTrainer(lr_path, hr_path=hr_path, device=device, config=trainer_config)
                    trainer.save_output = False
                    print(f"Processing {name}/{kernel_dir.name}: {row.get('pair_id')} (x{scale})")
                    sr_img, kernel = trainer.train()

                    metrics = compute_metrics(
                        device,
                        ssim_metric,
                        sr_img,
                        hr_path if entry.get("use_hr", True) else None,
                        border=scale,
                    )

                    sr_rel = ""
                    if save_results:
                        sr_path = recon_output_dir / f"{row.get('pair_id')}_x{scale}_SR.png"
                        Image.fromarray(sr_img).save(sr_path)
                        sr_rel = str(sr_path.relative_to(results_dir))

                    stats_rows.append(
                        {
                            "dataset": name,
                            "kernel": kernel_dir.name,
                            "pair_id": row.get("pair_id", ""),
                            "scale": str(scale),
                            "psnr_rgb": metrics["psnr_rgb"],
                            "psnr_y": metrics["psnr_y"],
                            "mse_rgb": metrics["mse_rgb"],
                            "mse_y": metrics["mse_y"],
                            "ssim": metrics["ssim"],
                            "sr_path": sr_rel,
                            "hr_path": hr_rel,
                            "lr_path": lr_rel,
                        }
                    )

                    if save_results:
                        lr_dest = lr_output_dir / Path(lr_rel).name
                        if not lr_dest.exists():
                            shutil.copy2(lr_path, lr_dest)
                        hr_dest = hr_output_dir / Path(hr_rel).name
                        if not hr_dest.exists():
                            shutil.copy2(hr_path, hr_dest)

                    processed += 1

            if processed == 0:
                print(f"  [WARN] No samples processed for kernel {kernel_dir.name}")
                continue

            summary_path = results_dir / "results_summary.csv"
            ensure_dir(results_dir)
            with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "dataset",
                    "kernel",
                    "pair_id",
                    "scale",
                    "psnr_rgb",
                    "psnr_y",
                    "mse_rgb",
                    "mse_y",
                    "ssim",
                    "sr_path",
                    "hr_path",
                    "lr_path",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stats_rows)
                psnr_values = [float(row["psnr_rgb"]) for row in stats_rows if row["psnr_rgb"]]
                psnr_y_values = [float(row["psnr_y"]) for row in stats_rows if row["psnr_y"]]
                mse_values = [float(row["mse_rgb"]) for row in stats_rows if row["mse_rgb"]]
                mse_y_values = [float(row["mse_y"]) for row in stats_rows if row["mse_y"]]
                ssim_values = [float(row["ssim"]) for row in stats_rows if row["ssim"]]
                writer.writerow(
                    {
                        "dataset": name,
                        "kernel": kernel_dir.name,
                        "pair_id": "MEAN",
                        "scale": "",
                        "psnr_rgb": f"{np.mean(psnr_values):.4f}" if psnr_values else "",
                        "psnr_y": f"{np.mean(psnr_y_values):.4f}" if psnr_y_values else "",
                        "mse_rgb": f"{np.mean(mse_values):.6f}" if mse_values else "",
                        "mse_y": f"{np.mean(mse_y_values):.6f}" if mse_y_values else "",
                        "ssim": f"{np.mean(ssim_values):.4f}" if ssim_values else "",
                        "sr_path": "",
                        "hr_path": "",
                        "lr_path": "",
                    }
                )

            print(f"  Completed kernel {kernel_dir.name} ({processed} samples)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_from_config(args)


if __name__ == "__main__":
    main()
