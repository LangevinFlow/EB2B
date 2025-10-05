"""Empirical Bayes super-resolution trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    import sys

    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root))

    from empirical_bayes import EmpiricalBayesConfig, EmpiricalBayesKernelEstimator  # type: ignore
    from dip import skip  # type: ignore
    from losses import SSIMLoss  # type: ignore
    from utils import (  # type: ignore
        blur_downsample,
        calculate_psnr,
        calculate_metrics,
        ensure_dir,
        get_noise,
        image_to_tensor,
        load_image,
        make_gradient_filter,
        save_image,
        save_kernel,
        tensor_to_image,
    )
else:
    from .empirical_bayes import EmpiricalBayesConfig, EmpiricalBayesKernelEstimator
    from .dip import skip
    from .losses import SSIMLoss
    from .utils import (
        blur_downsample,
        calculate_psnr,
        calculate_metrics,
        ensure_dir,
        get_noise,
        image_to_tensor,
        load_image,
        make_gradient_filter,
        save_image,
        save_kernel,
        tensor_to_image,
    )


@dataclass
class EBTrainingConfig:
    scale_factor: int = 4
    max_iters: int = 1000
    eb_steps: int = 25
    eb_lr: float = 1e-4
    eb_prior_weight: float = 1e-4
    dip_lr: float = 5e-3
    log_every: int = 50
    noise_sigma: float = 1.0
    kernel_size: Optional[int] = None
    I_loop_x: int = 5
    I_loop_k: int = 3
    grad_loss_lr: float = 1e-3
    image_disturbance: float = 0.0
    save_output: bool = True

    @property
    def print_iteration(self) -> int:
        return max(1, (self.max_iters * self.I_loop_x) // 5)

    def resolved_kernel_size(self) -> int:
        if self.kernel_size is not None:
            return self.kernel_size
        return min(self.scale_factor * 4 + 3, 21)


class EBTrainer:
    def __init__(
        self,
        lr_path: Path,
        *,
        hr_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        config: Optional[EBTrainingConfig] = None,
    ) -> None:
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or EBTrainingConfig()

        self.lr_tensor = image_to_tensor(load_image(lr_path), self.device)
        self.hr_tensor = None
        self.hr_image_np = None
        if hr_path is not None and hr_path.exists():
            self.hr_image_np = load_image(hr_path)
            self.hr_tensor = image_to_tensor(self.hr_image_np, self.device)

        _, _, h_lr, w_lr = self.lr_tensor.shape

        if self.hr_image_np is not None:
            target_h, target_w = self.hr_image_np.shape[:2]
        else:
            target_h = h_lr * self.config.scale_factor
            target_w = w_lr * self.config.scale_factor

        self.target_hr_shape = (target_h, target_w)
        self.target_lr_shape = (h_lr, w_lr)

        self.generator = skip(
            num_input_channels=self.lr_tensor.shape[1],
            num_output_channels=self.lr_tensor.shape[1],
            num_channels_down=[128, 128, 128, 128, 128],
            num_channels_up=[128, 128, 128, 128, 128],
            num_channels_skip=[16, 16, 16, 16, 16],
            upsample_mode='bilinear',
            need_sigmoid=True,
            need_bias=True,
            pad='reflection',
            act_fun='LeakyReLU'
        ).to(self.device)
        self.optimizer_dip = torch.optim.Adam(self.generator.parameters(), lr=self.config.dip_lr)

        self.noise = get_noise(
            self.lr_tensor.shape[1],
            self.target_hr_shape[0],
            self.target_hr_shape[1],
            self.device,
            sigma=self.config.noise_sigma,
        )
        self.noise.requires_grad = False

        eb_conf = EmpiricalBayesConfig(
            learning_rate=self.config.eb_lr,
            prior_weight=self.config.eb_prior_weight,
            anneal_iters=self.config.max_iters,
            num_steps=self.config.eb_steps,
        )
        self.eb_kernel = EmpiricalBayesKernelEstimator(
            kernel_size=self.config.resolved_kernel_size(),
            scale_factor=self.config.scale_factor,
            config=eb_conf,
            device=self.device,
            dtype=self.lr_tensor.dtype,
        )

        self.ssimloss = SSIMLoss().to(self.device)
        self.mse = torch.nn.MSELoss().to(self.device)
        self.grad_filters = make_gradient_filter(self.device)
        self.noise2_mean = 1.0
        self.num_pixels = self.lr_tensor.numel()
        self.ssim_iterations = min(80, self.config.max_iters)
        self.print_iteration = self.config.print_iteration
        self.log_dir: Optional[Path] = None

    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        best_psnr = -float("inf")
        best_sr: Optional[np.ndarray] = None
        best_kernel: Optional[np.ndarray] = None
        best_loss = float("inf")

        total_inner = self.config.I_loop_x
        pad = self.config.resolved_kernel_size() // 2

        for iteration in range(self.config.max_iters):
            sr = self.generator(self.noise)
            sr_pad = F.pad(sr, mode='circular', pad=(pad, pad, pad, pad))

            kernel = self.eb_kernel.optimise(
                sr.detach(),
                self.lr_tensor,
                num_steps=self.config.eb_steps,
                current_iter=iteration,
            )
            kernel.requires_grad = True

            ac_loss_k = torch.tensor(0.0, device=self.device, requires_grad=True)

            for i_p in range(total_inner):
                current_kernel = kernel.clone().detach().requires_grad_(True)

                self.optimizer_dip.zero_grad()

                sr = self.generator(self.noise)
                sr_pad = F.pad(sr, mode='circular', pad=(pad, pad, pad, pad))

                out_x = F.conv2d(sr_pad, current_kernel.expand(3, -1, -1, -1), groups=3)
                out_x = out_x[:, :, :: self.config.scale_factor, :: self.config.scale_factor]

                disturb = np.random.normal(0, np.random.uniform(0, self.config.image_disturbance), out_x.shape)
                disturb_tc = torch.from_numpy(disturb).type(torch.FloatTensor).to(self.device)

                if iteration <= self.ssim_iterations:
                    loss_x = 1 - self.ssimloss(out_x, self.lr_tensor + disturb_tc)
                else:
                    loss_x = self.mse(out_x, self.lr_tensor + disturb_tc)

                self.im_HR_est = sr
                grad_abs = self.calculate_grad_abs()
                grad_loss = self.config.grad_loss_lr * self.noise2_mean * 0.20 * torch.pow(grad_abs + 1e-8, 0.67).sum() / self.num_pixels

                loss_x_update = loss_x + grad_loss
                loss_x_update.backward(retain_graph=True)
                self.optimizer_dip.step()
                loss_x_update = loss_x_update.detach()

                if self.hr_image_np is None:
                    loss_value = float(loss_x_update.item())
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_sr = tensor_to_image(sr.detach())
                        best_kernel = kernel.clone().detach().squeeze().cpu().numpy()

                kernel_for_loss = kernel.clone().detach().requires_grad_(True)
                out_k = F.conv2d(sr_pad.clone().detach(), kernel_for_loss.expand(3, -1, -1, -1), groups=3)
                out_k = out_k[:, :, :: self.config.scale_factor, :: self.config.scale_factor]

                if iteration <= self.ssim_iterations:
                    loss_k = 1 - self.ssimloss(out_k, self.lr_tensor)
                else:
                    loss_k = self.mse(out_k, self.lr_tensor)

                ac_loss_k = ac_loss_k + loss_k

                global_step = iteration * total_inner + i_p + 1
                if global_step % self.config.I_loop_k == 0:
                    if ac_loss_k.requires_grad:
                        ac_loss_k.backward(retain_graph=True)
                    ac_loss_k = torch.tensor(0.0, device=self.device, requires_grad=True)

                if (global_step % self.print_iteration == 0) or global_step == 1:
                    self._print_and_output(sr, kernel, loss_x, iteration, i_p, global_step)

            kernel = kernel.detach()

            if self.hr_image_np is not None:
                with torch.no_grad():
                    sr_img = tensor_to_image(sr)
                    psnr_value = calculate_psnr(self.hr_image_np, sr_img)
                    if psnr_value > best_psnr:
                        best_psnr = psnr_value
                        best_sr = sr_img
                        best_kernel = kernel.squeeze().cpu().numpy()

        if best_sr is None or best_kernel is None:
            with torch.no_grad():
                final_sr = tensor_to_image(self.generator(self.noise))
                final_kernel = self.eb_kernel.current_kernel().squeeze().cpu().numpy()
            return final_sr, final_kernel

        return best_sr, best_kernel

    def _print_and_output(
        self,
        sr: torch.Tensor,
        kernel: torch.Tensor,
        loss_x: torch.Tensor,
        iteration: int,
        inner_idx: int,
        global_step: int,
    ) -> None:
        sr_img = tensor_to_image(sr)
        kernel_np = kernel.squeeze().detach().cpu().numpy()

        if self.hr_image_np is not None and self.hr_tensor is not None:
            metrics = calculate_metrics(self.hr_image_np, sr_img)
            psnr_val = metrics["psnr_rgb"]
            psnr_y = metrics["psnr_y"]
            mse_rgb = metrics["mse_rgb"]
            mse_y = metrics["mse_y"]
            ssim_val = self.ssimloss(sr, self.hr_tensor).item()
        else:
            psnr_val = -1.0
            psnr_y = -1.0
            mse_rgb = -1.0
            mse_y = -1.0
            ssim_val = -1.0

        print(
            f" Iter {iteration:03d}, loss: {loss_x.item():.6f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, "
            f"PSNR_Y: {psnr_y:.2f}, MSE_RGB: {mse_rgb:.6f}, MSE_Y: {mse_y:.6f}"
        )

        if self.log_dir is not None:
            ensure_dir(self.log_dir)
            step_str = f"{global_step:05d}"
            save_image(sr_img, self.log_dir / f"{self.lr_path.stem}_{step_str}.png")
            save_kernel(kernel_np, self.log_dir, f"{self.lr_path.stem}_{step_str}")

    def calculate_grad_abs(self, padding_mode: str = "reflect") -> torch.Tensor:
        hr_pad = F.pad(self.im_HR_est, mode=padding_mode, pad=(1,) * 4)
        filters = self.grad_filters.unsqueeze(1).unsqueeze(1)
        out = F.conv3d(
            hr_pad.expand(self.grad_filters.shape[0], -1, -1, -1).unsqueeze(0),
            filters,
            stride=1,
            groups=self.grad_filters.shape[0],
        )
        return torch.abs(out.squeeze(0))

    def run_and_save(self, output_dir: Path, log_dir: Optional[Path] = None) -> None:
        self.log_dir = None
        if self.config.save_output:
            self.log_dir = log_dir or (output_dir / "logs")

        sr_img, kernel = self.train()

        stem = self.lr_path.stem
        output_path = output_dir / self.lr_path.name
        save_image(sr_img, output_path)

        if self.log_dir is not None:
            save_kernel(kernel, self.log_dir, stem)
