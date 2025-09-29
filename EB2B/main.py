"""Command-line entry point for the EB2B Empirical Bayes pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    import sys

    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root))

    from trainer import EBTrainer, EBTrainingConfig  # type: ignore
    from utils import ensure_dir  # type: ignore
else:
    from .trainer import EBTrainer, EBTrainingConfig
    from .utils import ensure_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Empirical Bayes super-resolution (EB2B)")
    parser.add_argument("--dataset", type=str, default="butterfly", help="Dataset name (for default directory layout)")
    parser.add_argument("--sf", type=int, default=4, help="Scale factor")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing low-resolution inputs")
    parser.add_argument("--hr-dir", type=Path, default=None, help="Directory containing high-resolution references")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to store outputs")
    parser.add_argument("--image", type=str, default=None, help="Process a single image (filename)")
    parser.add_argument("--max-iters", type=int, default=1000, help="Number of outer iterations (will be divided by I_loop_x=5)")
    parser.add_argument("--eb-steps", type=int, default=25, help="Inner Empirical Bayes iterations per outer step")
    parser.add_argument("--eb-lr", type=float, default=5e-2, help="Empirical Bayes learning rate (DKP default: 5e-2)")
    parser.add_argument("--eb-prior-weight", type=float, default=1e-4, help="Empirical Bayes prior weight")
    parser.add_argument("--dip-lr", type=float, default=5e-3, help="Learning rate for the DIP generator")
    parser.add_argument("--kernel-size", type=int, default=None, help="Override kernel size (defaults to min(sf*4+3, 21))")
    parser.add_argument("--log-every", type=int, default=50, help="Logging frequency (iterations)")
    parser.add_argument("--noise-sigma", type=float, default=1.0, help="Input noise standard deviation")
    parser.add_argument("--I-loop-x", type=int, default=5, help="Number of inner DIP iterations per outer iteration")
    parser.add_argument("--I-loop-k", type=int, default=3, help="Frequency of kernel gradient accumulation")
    parser.add_argument("--grad-loss-lr", type=float, default=1e-3, help="Gradient loss weight")
    parser.add_argument("--device", type=str, default="auto", help="Computation device: auto|cpu|cuda")
    parser.add_argument("--real", action="store_true", help="Treat input as real data without ground truth")
    parser.add_argument("--no-save-output", action="store_true", help="Disable saving intermediate/final figures")
    return parser


def infer_default_paths(dataset: str, sf: int) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "DKP" / "data" / "datasets"
    input_dir = data_root / dataset / f"DIPDKP_lr_x{sf}"
    hr_dir = data_root / dataset / "HR"
    output_dir = repo_root / "EB2B" / "outputs" / f"{dataset}_x{sf}"
    return input_dir, hr_dir, output_dir


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    default_input, default_hr, default_output = infer_default_paths(args.dataset, args.sf)

    input_dir = args.input_dir or default_input
    hr_dir = args.hr_dir or default_hr
    output_dir = args.output_dir or default_output

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.real:
        hr_dir = None
    elif hr_dir is not None and not hr_dir.exists():
        hr_dir = None

    ensure_dir(output_dir)

    device = select_device(args.device)

    config = EBTrainingConfig(
        scale_factor=args.sf,
        max_iters=args.max_iters,
        eb_steps=args.eb_steps,
        eb_lr=args.eb_lr,
        eb_prior_weight=args.eb_prior_weight,
        dip_lr=args.dip_lr,
        kernel_size=args.kernel_size,
        log_every=args.log_every,
        noise_sigma=args.noise_sigma,
        I_loop_x=args.I_loop_x,
        I_loop_k=args.I_loop_k,
        grad_loss_lr=args.grad_loss_lr,
        save_output=not args.no_save_output,
    )

    lr_images = sorted(input_dir.glob("*.png"))
    if args.image is not None:
        candidate = input_dir / args.image
        if not candidate.exists():
            raise FileNotFoundError(f"Specific image not found: {candidate}")
        lr_images = [candidate]

    if not lr_images:
        raise RuntimeError(f"No PNG images found in {input_dir}")

    for lr_path in lr_images:
        hr_path = None
        if hr_dir is not None:
            candidate = hr_dir / lr_path.name
            if candidate.exists():
                hr_path = candidate
        print(f"Processing {lr_path.name} (device={device})")
        trainer = EBTrainer(lr_path, hr_path=hr_path, device=device, config=config)
        trainer.run_and_save(output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
