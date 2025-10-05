"""Utility helpers for the EB2B reimplementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().clamp(0.0, 1.0)
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = (tensor * 255.0 + 0.5).astype(np.uint8)
    return tensor


def save_image(image: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(image).save(path)


def get_noise(channels: int, height: int, width: int, device: torch.device, *, sigma: float = 1.0) -> torch.Tensor:
    noise = torch.randn(1, channels, height, width, device=device)
    return noise * sigma


def _to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64)


def _align_to_min(reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_h = min(reference.shape[0], target.shape[0])
    min_w = min(reference.shape[1], target.shape[1])
    if reference.ndim == 3:
        reference = reference[:min_h, :min_w, :]
    else:
        reference = reference[:min_h, :min_w]
    if target.ndim == 3:
        target = target[:min_h, :min_w, :]
    else:
        target = target[:min_h, :min_w]
    return reference, target


def _crop_border(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    h, w = image.shape[:2]
    border = min(border, h // 2, w // 2)
    if image.ndim == 3:
        return image[border:h - border, border:w - border, :]
    return image[border:h - border, border:w - border]


def _to_y_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def calculate_metrics(reference: np.ndarray, target: np.ndarray, border: int = 0) -> dict:
    ref = _to_float(reference)
    tgt = _to_float(target)

    ref, tgt = _align_to_min(ref, tgt)

    ref = _crop_border(ref, border)
    tgt = _crop_border(tgt, border)

    ref_y = _crop_border(_to_y_channel(ref), 0)
    tgt_y = _crop_border(_to_y_channel(tgt), 0)

    mse_rgb = np.mean((ref - tgt) ** 2)
    mse_y = np.mean((ref_y - tgt_y) ** 2)

    psnr_rgb = float("inf") if mse_rgb == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse_rgb))
    psnr_y = float("inf") if mse_y == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse_y))

    return {
        "psnr_rgb": psnr_rgb,
        "psnr_y": psnr_y,
        "mse_rgb": mse_rgb,
        "mse_y": mse_y,
    }


def calculate_psnr(reference: np.ndarray, target: np.ndarray, border: int = 0) -> float:
    return calculate_metrics(reference, target, border)["psnr_rgb"]


def calculate_psnr_y(reference: np.ndarray, target: np.ndarray, border: int = 0) -> float:
    return calculate_metrics(reference, target, border)["psnr_y"]


def calculate_mse(reference: np.ndarray, target: np.ndarray, border: int = 0) -> float:
    return calculate_metrics(reference, target, border)["mse_rgb"]


def calculate_mse_y(reference: np.ndarray, target: np.ndarray, border: int = 0) -> float:
    return calculate_metrics(reference, target, border)["mse_y"]


def blur_downsample(image: torch.Tensor, kernel: torch.Tensor, scale_factor: int) -> torch.Tensor:
    pad = kernel.size(-1) // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    groups = image.shape[1]
    blurred = F.conv2d(padded, kernel.expand(groups, -1, -1, -1), groups=groups)
    return blurred[:, :, ::scale_factor, ::scale_factor]


def match_tensor_to_size(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    mode: str = "reflect",
) -> torch.Tensor:
    """Center crop or pad a tensor (NCHW) to the requested spatial size."""

    _, _, h, w = tensor.shape
    if h == target_h and w == target_w:
        return tensor

    dh = target_h - h
    dw = target_w - w

    out = tensor

    if dh < 0 or dw < 0:
        top = max(-dh // 2, 0)
        bottom = top + target_h
        left = max(-dw // 2, 0)
        right = left + target_w
        out = out[:, :, top:bottom, left:right]

    if dh > 0 or dw > 0:
        pad_top = dh // 2 if dh > 0 else 0
        pad_bottom = dh - pad_top if dh > 0 else 0
        pad_left = dw // 2 if dw > 0 else 0
        pad_right = dw - pad_left if dw > 0 else 0
        out = F.pad(out, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)

    return out


def save_kernel(kernel: np.ndarray, output_dir: Path, stem: str) -> None:
    ensure_dir(output_dir)
    mat_path = output_dir / f"{stem}.mat"
    png_path = output_dir / f"{stem}_kernel.png"

    sio.savemat(mat_path, {"Kernel": kernel})

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    im = ax.imshow(kernel, vmin=0.0, vmax=kernel.max())
    fig.colorbar(im, ax=ax)
    ax.set_title("Estimated Kernel")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)

def make_gradient_filter(device: torch.device) -> torch.Tensor:
    filters = np.zeros([4, 3, 3], dtype=np.float32)
    filters[0] = np.array([[0, -1, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
    filters[1] = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
    filters[2] = np.array([[0, 0, 0],
                           [-1, 1, 0],
                           [0, 0, 0]])
    filters[3] = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [-1, 0, 0]])
    return torch.from_numpy(filters).to(device)


def gradient_magnitude(image: torch.Tensor, grad_filters: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    filters = grad_filters.to(image.device).unsqueeze(1)
    grads = F.conv2d(padded, filters)
    return grads.abs()

def save_sr_kernel_figure(sr_image: np.ndarray, kernel: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(sr_image)
    axes[0].axis("off")
    axes[0].set_title("Deblurred SR")

    im = axes[1].imshow(kernel, cmap="viridis")
    axes[1].axis("off")
    axes[1].set_title("Estimated Kernel")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
