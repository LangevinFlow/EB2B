"""Utility helpers for the EB2B reimplementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

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


def calculate_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    target = target.astype(np.float64)
    mse = np.mean((reference - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


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
