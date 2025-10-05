#!/usr/bin/env python3
"""Evaluate PSNR/SSIM for reconstructed images stored under Results/Set5."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    package_root = Path(__file__).resolve().parent / "EB2B"
    sys.path.insert(0, str(package_root))

    from utils import calculate_metrics  # type: ignore
    from losses import SSIMLoss  # type: ignore
else:
    from .EB2B.utils import calculate_psnr  # type: ignore
    from .EB2B.losses import SSIMLoss  # type: ignore


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0 or img.shape[0] <= 2 * border or img.shape[1] <= 2 * border:
        return img
    return img[border:-border, border:-border]


def parse_scale(name: str) -> int:
    if "_x" in name:
        try:
            return int(name.split("_x")[-1].split("_")[0])
        except ValueError:
            return 1
    return 1


def evaluate_kernel(kernel_dir: Path, device: torch.device, ssim_metric: SSIMLoss) -> List[Dict[str, str]]:
    recon_dir = kernel_dir / "Recon"
    hr_dir = kernel_dir / "HR"
    entries: List[Dict[str, str]] = []

    for sr_path in sorted(recon_dir.glob("*_SR.png")):
        name = sr_path.stem
        pair_id = name.split("_SR")[0]
        scale = parse_scale(pair_id)
        hr_path = hr_dir / f"{pair_id}.png"
        if not hr_path.exists():
            hr_path = hr_dir / f"{pair_id}.jpg"
        if not hr_path.exists():
            print(f"[WARN] HR not found for {sr_path}")
            continue

        sr_img = np.array(Image.open(sr_path).convert("RGB"), dtype=np.float32) / 255.0
        hr_img = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32) / 255.0

        if sr_img.shape != hr_img.shape:
            sr_tensor = torch.from_numpy(sr_img.transpose(2, 0, 1)).unsqueeze(0).float()
            target_size = (hr_img.shape[0], hr_img.shape[1])
            sr_tensor = F.interpolate(sr_tensor, size=target_size, mode="bicubic", align_corners=False)
            sr_img_aligned = sr_tensor.squeeze().permute(1, 2, 0).numpy()
        else:
            sr_img_aligned = sr_img

        sr_uint8 = (np.clip(sr_img_aligned, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        hr_uint8 = (np.clip(hr_img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        metrics = calculate_metrics(hr_uint8, sr_uint8, border=scale)

        sr_tensor = torch.from_numpy(sr_uint8.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
        hr_tensor = torch.from_numpy(hr_uint8.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
        if scale > 0:
            border = min(scale, sr_tensor.shape[-1] // 2, sr_tensor.shape[-2] // 2)
            if border > 0:
                sr_tensor = sr_tensor[:, :, border:-border, border:-border]
                hr_tensor = hr_tensor[:, :, border:-border, border:-border]
        with torch.no_grad():
            ssim_val = float(ssim_metric(sr_tensor, hr_tensor).item())

        entries.append(
            {
                "kernel": kernel_dir.name,
                "pair_id": pair_id,
                "scale": str(scale),
                "psnr_rgb": f"{metrics['psnr_rgb']:.4f}",
                "psnr_y": f"{metrics['psnr_y']:.4f}",
                "mse_rgb": f"{metrics['mse_rgb']:.6f}",
                "mse_y": f"{metrics['mse_y']:.6f}",
                "ssim": f"{ssim_val:.4f}",
            }
        )

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM for results in Results/Set5")
    parser.add_argument("--results-root", type=Path, default=Path("Results/Set5"), help="Root directory containing kernel outputs")
    parser.add_argument("--output", type=Path, default=None, help="Path to save evaluation CSV")
    parser.add_argument("--device", type=str, default="cpu", help="Device for SSIM computation (cpu|cuda|auto)")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    ssim_metric = SSIMLoss().to(device)
    ssim_metric.eval()

    all_entries: List[Dict[str, str]] = []
    results_root = args.results_root

    for kernel_dir in sorted(results_root.glob("kernel_*")):
        if not kernel_dir.is_dir():
            continue
        entries = evaluate_kernel(kernel_dir, device, ssim_metric)
        if not entries:
            continue

        psnr_vals = [float(e["psnr_rgb"]) for e in entries if e["psnr_rgb"]]
        psnr_y_vals = [float(e["psnr_y"]) for e in entries if e["psnr_y"]]
        mse_vals = [float(e["mse_rgb"]) for e in entries if e["mse_rgb"]]
        mse_y_vals = [float(e["mse_y"]) for e in entries if e["mse_y"]]
        ssim_vals = [float(e["ssim"]) for e in entries if e["ssim"]]
        mean_entry = {
            "kernel": kernel_dir.name,
            "pair_id": "MEAN",
            "scale": "",
            "psnr_rgb": f"{np.mean(psnr_vals):.4f}" if psnr_vals else "",
            "psnr_y": f"{np.mean(psnr_y_vals):.4f}" if psnr_y_vals else "",
            "mse_rgb": f"{np.mean(mse_vals):.6f}" if mse_vals else "",
            "mse_y": f"{np.mean(mse_y_vals):.6f}" if mse_y_vals else "",
            "ssim": f"{np.mean(ssim_vals):.4f}" if ssim_vals else "",
        }
        entries.append(mean_entry)
        all_entries.extend(entries)

    if not all_entries:
        print("No reconstructions found to evaluate.")
        return

    output_path = args.output or (results_root / "evaluation_summary.csv")
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "kernel",
            "pair_id",
            "scale",
            "psnr_rgb",
            "psnr_y",
            "mse_rgb",
            "mse_y",
            "ssim",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
