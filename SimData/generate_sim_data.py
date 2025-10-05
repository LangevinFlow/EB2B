#!/usr/bin/env python3
"""Generate simulated LR images by applying random blur kernels to HR datasets.

This script adapts the kernel generation utilities from DKP/DIPDKP/model/kernel_generate.py
and applies the generated kernels to the HR images listed in the CSV manifests that live under
`/mnt/DATA/SuperResolution/`.

For each selected dataset we sample a number of random kernels, save each kernel as a `.npy`
file, and convolve/downsample every HR image to produce synthetic LR observations. The LR
outputs are organised under `SimData/<dataset>/kernel_<id>/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import center_of_mass

CONFIG_DEFAULT = Path("EB2B/Config/superres_datasets.json")
SIMDATA_ROOT_DEFAULT = Path("SimData")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetEntry:
    name: str
    manifest: Path
    use_hr: bool = True
    output_subdir: Optional[str] = None
    params: Dict[str, float] = field(default_factory=dict)
    scales: Optional[Iterable[int]] = None


@dataclass
class KernelSpec:
    data: np.ndarray
    kind: str
    params: Dict[str, float]


def kernel_move(kernel: np.ndarray, move_x: float, move_y: float) -> np.ndarray:
    """Shift kernel centre to a target location."""
    current_center = center_of_mass(kernel)
    shift_vec = (move_x - current_center[0], move_y - current_center[1])
    return nd_shift(kernel, shift_vec, mode="nearest")


def gen_kernel_fixed(
    k_size: np.ndarray,
    scale_factor: int,
    lambda_1: float,
    lambda_2: float,
    theta: float,
    noise: np.ndarray,
    move_x: float,
    move_y: float,
) -> np.ndarray:
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    X, Y = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    raw_kernel_moved = kernel_move(raw_kernel, move_x, move_y)
    kernel = raw_kernel_moved / np.sum(raw_kernel_moved)
    return kernel


def gen_kernel_random_gaussian(
    scale_factor: int,
    rng: np.random.Generator,
    min_var: float,
    max_var: float,
    noise_level: float = 0.0,
) -> KernelSpec:
    kernel_size = min(scale_factor * 4 + 3, 21)
    k_size = np.array([kernel_size, kernel_size])
    lambda_1 = rng.uniform(min_var, max_var)
    lambda_2 = rng.uniform(min_var, max_var)
    theta = rng.uniform(0, math.pi)
    noise = rng.uniform(-noise_level, noise_level, size=k_size) if noise_level > 0 else np.zeros(k_size)
    centre = (kernel_size - 1) / 2
    shift_range = 0.5
    move_x = centre + rng.uniform(-shift_range, shift_range)
    move_y = centre + rng.uniform(-shift_range, shift_range)
    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise, move_x, move_y)
    params = {
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "theta": float(theta),
        "move_x": float(move_x),
        "move_y": float(move_y),
        "kernel_size": int(kernel_size),
    }
    return KernelSpec(data=kernel.astype(np.float32), kind="gaussian", params=params)


def gen_kernel_random_motion(scale_factor: int, rng: np.random.Generator) -> KernelSpec:
    kernel_size = min(scale_factor * 4 + 3, 21)
    k_size = kernel_size
    M = int((scale_factor * 3 + 3) / 2)
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    length = rng.integers(max(1, scale_factor), max(2, scale_factor * 2))
    kernel[M, M - length : M + length + 1] = 1.0
    theta = rng.uniform(0, 360)
    kernel = nd_rotate(kernel, angle=theta, reshape=False, order=1)
    kernel = np.clip(kernel, 0, None)
    kernel /= kernel.sum()
    params = {"theta": float(theta), "length": int(length), "kernel_size": int(kernel_size)}
    return KernelSpec(data=kernel.astype(np.float32), kind="motion", params=params)


def apply_kernel_and_downsample(image: np.ndarray, kernel: np.ndarray, scale: int) -> np.ndarray:
    h, w, c = image.shape
    target_h = (h // scale) * scale
    target_w = (w // scale) * scale
    image = image[:target_h, :target_w, :]

    img_t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    kernel_t = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pad = kernel_t.shape[-1] // 2
        padded = F.pad(img_t, (pad, pad, pad, pad), mode="reflect")
        blurred = F.conv2d(padded, kernel_t.expand(c, 1, -1, -1), groups=c)
        downsampled = blurred[:, :, ::scale, ::scale]

    output = downsampled.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return (output * 255.0 + 0.5).astype(np.uint8)


def load_image(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def save_kernel_metadata(kernel_root: Path, kernel_spec: KernelSpec) -> None:
    np.save(kernel_root / "kernel.npy", kernel_spec.data)
    metadata = {"type": kernel_spec.kind, "params": kernel_spec.params}
    with (kernel_root / "kernel.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(kernel_spec.data, cmap="viridis")
    ax.axis("off")
    ax.set_title(f"{kernel_spec.kind} kernel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(kernel_root / "kernel.png")
    plt.close(fig)


def parse_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_entries(cfg: Dict, datasets_filter: Optional[List[str]], data_root: Path) -> List[DatasetEntry]:
    entries: List[DatasetEntry] = []
    for entry in cfg.get("datasets", []):
        name = entry.get("name")
        if datasets_filter and name not in datasets_filter:
            continue
        manifest = Path(entry.get("manifest"))
        if not manifest.is_absolute():
            manifest = data_root / manifest
        entries.append(
            DatasetEntry(
                name=name,
                manifest=manifest,
                use_hr=entry.get("use_hr", True),
                output_subdir=entry.get("output_subdir"),
                params=entry.get("params", {}),
                scales=entry.get("scales"),
            )
        )
    return entries


def select_kernel_generator(kind: str, rng: np.random.Generator):
    def gaussian(scale: int) -> KernelSpec:
        sf = max(1, scale)
        min_var = 0.175 * sf
        max_var = min(1.5 * sf, 10)
        return gen_kernel_random_gaussian(sf, rng, min_var, max_var)

    def motion(scale: int) -> KernelSpec:
        sf = max(1, scale)
        return gen_kernel_random_motion(sf, rng)

    if kind == "gaussian":
        return gaussian
    if kind == "motion":
        return motion

    def both(scale: int, _toggle=[0]) -> KernelSpec:  # mutable default to alternate
        _toggle[0] ^= 1
        return (gaussian if _toggle[0] else motion)(scale)

    return both


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated LR data using random blur kernels.")
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT, help="Path to dataset JSON config.")
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset names to process")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data root directory (folder containing HR images)")
    parser.add_argument("--simdata-root", type=Path, default=SIMDATA_ROOT_DEFAULT, help="Destination root directory")
    parser.add_argument("--num-kernels", type=int, default=5, help="Number of kernels per dataset")
    parser.add_argument("--kernel-type", choices=["gaussian", "motion", "both"], default="gaussian")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing kernel directories")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images per dataset (debug)")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    cfg = parse_config(args.config)
    if args.data_root is not None:
        data_root = args.data_root.expanduser()
    else:
        data_root = Path(cfg["data_root"]).expanduser()

    if args.simdata_root == SIMDATA_ROOT_DEFAULT and "simdata_root" in cfg:
        output_root = Path(cfg["simdata_root"]).expanduser()
    else:
        output_root = args.simdata_root.expanduser()
    ensure_dir(output_root)

    rng = np.random.default_rng(args.seed)
    kernel_generator = select_kernel_generator(args.kernel_type, rng)

    datasets = build_dataset_entries(cfg, args.datasets, data_root)
    if not datasets:
        print("No matching datasets found in configuration.")
        return

    for dataset in datasets:
        dataset_root = output_root / (dataset.output_subdir or dataset.name or dataset.manifest.stem)
        ensure_dir(dataset_root)

        print(f"Dataset: {dataset.name}")
        rows: List[Dict[str, str]]
        if dataset.manifest.exists():
            with dataset.manifest.open("r", newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))
        else:
            source_dir = data_root / (dataset.output_subdir or dataset.name or "")
            if not source_dir.exists():
                print(f"  [WARN] Neither manifest nor source directory found for {dataset.name}")
                continue
            rows = []
            for img_path in sorted(source_dir.glob("*")):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    continue
                rel = img_path.relative_to(data_root)
                rows.append(
                    {
                        "dataset": dataset.name,
                        "split": "",
                        "scale": "1",
                        "pair_id": img_path.stem,
                        "hr_path": str(rel),
                        "lr_path": "",
                    }
                )
        if not rows:
            print(f"  [WARN] No images found for dataset: {dataset.name}")
            continue

        if dataset.scales:
            scale_filter = {int(s) for s in dataset.scales}
        else:
            scale_filter = None

        for kernel_idx in range(1, args.num_kernels + 1):
            kernel_root = dataset_root / f"kernel_{kernel_idx:02d}"
            lr_dir = kernel_root / "LR"
            hr_dir = kernel_root / "HR"

            if kernel_root.exists():
                if args.overwrite:
                    shutil.rmtree(kernel_root)
                else:
                    print(f"  Kernel {kernel_idx:02d} exists; skipping (use --overwrite to regenerate)")
                    continue

            ensure_dir(lr_dir)
            ensure_dir(hr_dir)

            kernel_spec: Optional[KernelSpec] = None
            processed = 0
            manifest_rows: List[Dict[str, str]] = []
            for row in rows:
                if args.limit is not None and processed >= args.limit:
                    break

                try:
                    scale = int(row.get("scale", 0)) or int(dataset.params.get("scale_factor", 1))
                except (TypeError, ValueError):
                    print(f"    [WARN] Invalid scale for row: {row}")
                    continue

                if scale_filter and scale not in scale_filter:
                    continue

                lr_rel = row.get("lr_path")
                hr_rel = row.get("hr_path")
                if not hr_rel:
                    continue

                hr_path = (data_root / hr_rel).expanduser()
                if not hr_path.exists():
                    print(f"    [WARN] HR image missing: {hr_path}")
                    continue

                if kernel_spec is None:
                    kernel_spec = kernel_generator(scale)
                    save_kernel_metadata(kernel_root, kernel_spec)
                kernel = kernel_spec.data

                try:
                    hr_image = load_image(hr_path)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"    [WARN] Failed to load {hr_path}: {exc}")
                    continue

                lr_image = apply_kernel_and_downsample(hr_image, kernel, scale)

                pair_id = row.get("pair_id") or Path(lr_rel).stem
                lr_filename = f"{pair_id}_x{scale}.png"
                lr_output_path = lr_dir / lr_filename
                Image.fromarray(lr_image).save(lr_output_path)

                hr_filename = f"{pair_id}_x{scale}.png"
                hr_output_path = hr_dir / hr_filename
                if not hr_output_path.exists():
                    Image.fromarray((hr_image * 255.0 + 0.5).astype(np.uint8)).save(hr_output_path)

                manifest_rows.append(
                    {
                        "dataset": dataset.name,
                        "split": row.get("split", ""),
                        "scale": str(scale),
                        "pair_id": pair_id,
                        "hr_path": str(Path("HR") / hr_filename),
                        "lr_path": str(Path("LR") / lr_filename),
                        "kernel_path": "kernel.npy",
                        "kernel_json": "kernel.json",
                        "kernel_png": "kernel.png",
                    }
                )

                processed += 1

            if processed == 0:
                print(f"  [WARN] No images processed for kernel {kernel_idx:02d}; removing directory.")
                shutil.rmtree(kernel_root, ignore_errors=True)
            else:
                manifest_path = kernel_root / "manifest.csv"
                with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "dataset",
                        "split",
                        "scale",
                        "pair_id",
                        "hr_path",
                        "lr_path",
                        "kernel_path",
                        "kernel_json",
                        "kernel_png",
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(manifest_rows)

                print(f"  Kernel {kernel_idx:02d}: {processed} images")


if __name__ == "__main__":
    main()
