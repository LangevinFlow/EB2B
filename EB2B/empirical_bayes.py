"""Empirical Bayes kernel estimation utilities.

This module re-implements the Empirical Bayes optimiser that appears in
``DKP/DIPDKP/model/kernel_generate.py`` but in an isolated, reusable form.

The estimator learns the parameters of an anisotropic Gaussian blur kernel
(rotation, eigenvalues and sub-pixel shifts) by minimising the discrepancy
between a high-resolution estimate and the observed low-resolution input.
"""

from dataclasses import dataclass
from typing import Optional, Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class EmpiricalBayesConfig:
    """Hyper-parameters controlling the Empirical Bayes optimiser.

    Attributes
    ----------
    learning_rate:
        Step size used by the Adam optimiser. Default 5e-2 matches DKP.
    prior_weight:
        Strength of the inverse-eigenvalue penalty that discourages degenerate
        kernels. Default 1e-4 matches DKP.
    eigen_lower:
        Minimum eigenvalue (in pixels) after the softplus re-parameterisation.
    eigen_scale:
        Softplus output is multiplied by this factor to control the reachable
        range of eigenvalues.
    clip_grad_norm:
        Maximum norm for gradient clipping; set to ``None`` to disable.
    temp_start, temp_end:
        Temperature annealing bounds used to smoothly transition from a
        conservative to an aggressive data-fit objective.
    anneal_iters:
        Number of outer iterations used for the linear annealing schedule.
    shift_range:
        Optional override for the maximum absolute sub-pixel shift.  When
        ``None`` we restrict the shift to half the down-sampling stride.
    pad_mode:
        Padding mode used before convolution. ``"reflect"`` matches the DKP
        implementation.
    optimiser_cls:
        Optimiser class to instantiate. It must follow the signature
        ``optimiser_cls(parameters: Iterable[nn.Parameter], lr=learning_rate)``.
    """

    # Defaults match DKP's actual behavior
    learning_rate: float = 5e-2
    prior_weight: float = 1e-4
    eigen_lower: float = 0.1
    eigen_scale: float = 4.0
    num_steps: int = 25
    clip_grad_norm: Optional[float] = 0.5
    temp_start: float = 0.5
    temp_end: float = 1.0
    anneal_iters: int = 100
    shift_range: Optional[float] = None
    pad_mode: str = "reflect"
    optimiser_cls: Type[torch.optim.Optimizer] = torch.optim.Adam


class EmpiricalBayesKernelEstimator(nn.Module):
    """Estimate an anisotropic Gaussian blur kernel via Empirical Bayes.

    Parameters
    ----------
    kernel_size:
        Size of the (square) blur kernel. The size is forced to be odd.
    scale_factor:
        Down-sampling factor applied after blurring.
    config:
        Hyper-parameters for the optimisation procedure.
    device:
        Device on which both the parameters and buffers are stored.
    dtype:
        Floating point precision for the learnable parameters.
    """

    def __init__(
        self,
        kernel_size: int,
        scale_factor: int,
        config: EmpiricalBayesConfig | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be a positive integer")

        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1  # enforce odd size for symmetric padding

        self.kernel_size = kernel_size
        self.scale_factor = int(scale_factor)
        self.config = config or EmpiricalBayesConfig()

        device = device or torch.device("cpu")
        dtype = dtype or torch.get_default_dtype()

        # Learnable raw parameters. They are initialised close to zero which
        # maps to symmetric kernels with soft shifts.
        self.raw_eigen = nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
        self.raw_theta = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self.raw_shift = nn.Parameter(torch.zeros(2, device=device, dtype=dtype))

        # Pre-compute coordinate grid and centre position.
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(kernel_size, device=device, dtype=dtype),
                torch.arange(kernel_size, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        )  # [k, k, 2]
        self.register_buffer("coords", coords, persistent=False)
        centre = torch.tensor((kernel_size - 1) / 2.0, device=device, dtype=dtype)
        self.register_buffer("centre", centre, persistent=False)

        # Lazily built optimiser (created on first call to ``build_optimizer``).
        self._optimiser: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Optimiser helpers
    # ------------------------------------------------------------------
    def build_optimizer(self) -> torch.optim.Optimizer:
        """Instantiate (or return the existing) optimiser."""
        if self._optimiser is None:
            self._optimiser = self.config.optimiser_cls(
                self.parameters(), lr=self.config.learning_rate
            )
        return self._optimiser

    # ------------------------------------------------------------------
    # Parameter transforms
    # ------------------------------------------------------------------
    def eigenvalues(self) -> Tensor:
        """Return positive eigenvalues with a configurable lower bound."""
        return self.config.eigen_lower + self.config.eigen_scale * F.softplus(
            self.raw_eigen
        )

    def rotation_matrix(self) -> Tensor:
        """Compute a 2x2 rotation matrix from ``raw_theta``."""
        theta = self.raw_theta[0]
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        return torch.stack(
            (torch.stack((cos_theta, -sin_theta)), torch.stack((sin_theta, cos_theta)))
        )

    def shifts(self) -> Tensor:
        """Return bounded sub-pixel shifts in x and y."""
        max_shift = (
            self.config.shift_range
            if self.config.shift_range is not None
            else max((self.scale_factor - 1) / 2.0, 0.0)
        )
        if max_shift == 0:
            return torch.zeros_like(self.raw_shift)
        return max_shift * torch.tanh(self.raw_shift)

    # ------------------------------------------------------------------
    # Kernel generation and application
    # ------------------------------------------------------------------
    def generate_kernel(self) -> Tensor:
        """Create the current blur kernel as a `[1, 1, k, k]` tensor."""
        eigvals = self.eigenvalues()
        rotation = self.rotation_matrix()
        shifts = self.shifts()

        # Compute the inverse covariance using the eigenvalue decomposition.
        inv_lambda = torch.diag(1.0 / eigvals)
        inv_sigma = rotation @ inv_lambda @ rotation.T  # [2, 2]

        mu = torch.stack((self.centre + shifts[0], self.centre + shifts[1]))  # [2]
        zz = self.coords - mu  # [k, k, 2]
        zz_unsq = zz.unsqueeze(-2)  # [k, k, 1, 2]
        quad_form = (zz_unsq @ inv_sigma @ zz_unsq.transpose(-1, -2)).squeeze((-1, -2))
        raw_kernel = torch.exp(-0.5 * quad_form)

        # Normalise to ensure the kernel sums to one.
        kernel = raw_kernel / raw_kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def apply_kernel(self, image: Tensor, kernel: Tensor | None = None) -> Tensor:
        """Blur ``image`` with ``kernel`` (or the current kernel) and down-sample."""
        if image.ndim != 4:
            raise ValueError("image must be a 4D NCHW tensor")
        if image.size(0) != 1:
            raise ValueError("batch size must be 1 for this lightweight implementation")

        padding = self.kernel_size // 2
        padded = F.pad(image, (padding, padding, padding, padding), mode=self.config.pad_mode)

        kernel = kernel if kernel is not None else self.generate_kernel()
        groups = image.size(1)
        blurred = F.conv2d(padded, kernel.expand(groups, -1, -1, -1), groups=groups)
        return blurred[:, :, :: self.scale_factor, :: self.scale_factor]

    # ------------------------------------------------------------------
    # Optimisation loop
    # ------------------------------------------------------------------
    def optimise(
        self,
        source: Tensor,
        target: Tensor,
        *,
        num_steps: int | None = None,
        current_iter: int = 0,
        keep_graph: bool = False,
    ) -> Tensor:
        """Run ``num_steps`` of Empirical Bayes optimisation.

        Parameters
        ----------
        source:
            High-resolution estimate (``[1, C, H, W]``) that is blurred and
            down-sampled to compare against ``target``.
        target:
            Observed low-resolution tensor. The spatial size must match the
            down-sampled output of ``source``.
        num_steps:
            Number of gradient updates to perform.
        current_iter:
            Index of the outer loop iteration; used for temperature annealing.
        keep_graph:
            Whether to retain the computation graph after optimisation.

        Returns
        -------
        torch.Tensor
            The final kernel detached from the computation graph.
        """

        steps = num_steps if num_steps is not None else self.config.num_steps
        if steps <= 0:
            raise ValueError("num_steps must be positive")

        optimiser = self.build_optimizer()

        device = self.raw_eigen.device
        if source.device != device or target.device != device:
            raise ValueError(
                "source and target must live on the same device as the estimator's parameters"
            )

        # Temperature annealing schedule (linear in ``current_iter``).
        progress = min(max(current_iter, 0) / max(self.config.anneal_iters, 1), 1.0)
        temp = self.config.temp_start + (self.config.temp_end - self.config.temp_start) * progress

        for _ in range(steps):
            optimiser.zero_grad(set_to_none=True)

            kernel = self.generate_kernel()
            blurred = self.apply_kernel(source, kernel=kernel)

            if blurred.shape[-2:] != target.shape[-2:]:
                min_h = min(blurred.shape[-2], target.shape[-2])
                min_w = min(blurred.shape[-1], target.shape[-1])
                blurred_cmp = blurred[..., :min_h, :min_w]
                target_cmp = target[..., :min_h, :min_w]
            else:
                blurred_cmp = blurred
                target_cmp = target

            mse_loss = F.mse_loss(blurred_cmp, target_cmp) * temp

            if blurred.shape != target.shape:
                blurred = blurred[..., :target.shape[-2], :target.shape[-1]]

            eigvals = self.eigenvalues()
            inv_penalty = self.config.prior_weight * torch.sum(1.0 / (eigvals ** 2))
            loss = mse_loss + inv_penalty

            loss.backward(retain_graph=keep_graph)

            if self.config.clip_grad_norm is not None:
                # Only clip theta and shift parameters, not eigenvalues (matches DKP)
                nn.utils.clip_grad_norm_(
                    [self.raw_theta, self.raw_shift], self.config.clip_grad_norm
                )

            optimiser.step()

        return self.generate_kernel().detach()

    # Backwards-compatible alias used by the original DKP implementation.
    def optimize(
        self,
        source: Tensor,
        target: Tensor,
        *,
        num_steps: int | None = None,
        current_iter: int = 0,
        keep_graph: bool = False,
    ) -> Tensor:
        return self.optimise(
            source,
            target,
            num_steps=num_steps,
            current_iter=current_iter,
            keep_graph=keep_graph,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Reset learnable parameters and optimiser state."""
        with torch.no_grad():
            self.raw_eigen.zero_()
            self.raw_theta.zero_()
            self.raw_shift.zero_()
        if self._optimiser is not None:
            self._optimiser.state.clear()
            self._optimiser = None

    def current_kernel(self) -> Tensor:
        """Return the current kernel detached from the computation graph."""
        return self.generate_kernel().detach()

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, scale_factor={self.scale_factor}, "
            f"learning_rate={self.config.learning_rate}"
        )


__all__ = [
    "EmpiricalBayesConfig",
    "EmpiricalBayesKernelEstimator",
]
