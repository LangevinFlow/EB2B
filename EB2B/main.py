"""Command-line entry point for the EB2B Empirical Bayes pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import dataclasses
from pathlib import Path

import torch
from PIL import Image

if __package__ is None or __package__ == "":
    import sys

    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root))

    from trainer import EBTrainer, EBTrainingConfig  # type: ignore
    from utils import ensure_dir, save_sr_kernel_figure  # type: ignore
else:
    from .trainer import EBTrainer, EBTrainingConfig
    from .utils import ensure_dir, save_sr_kernel_figure


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Empirical Bayes super-resolution (EB2B)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (for direct single-image run or to filter config entries)")
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
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON configuration for batch processing")
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

    if args.config is not None:
        run_from_config(args)
        return

    if args.dataset is None:
        parser.error("--dataset is required when --config is not provided")

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


def run_from_config(args) -> None:
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    save_results = not args.no_save_output
    device = select_device(args.device)

    data_root = Path(cfg["data_root"]).expanduser()
    output_root = Path(cfg.get("output_root", "outputs")).expanduser()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if save_results:
        ensure_dir(output_root)
    default_params = cfg.get("default_params", {})
    artifact_cfg = cfg.get("artifact_layout", {})

    lr_dir_name = artifact_cfg.get("lr_dir", "LR")
    hr_dir_name = artifact_cfg.get("hr_dir", "HR")
    recon_dir_name = artifact_cfg.get("recon_dir", "Recon")

    dataset_entries = cfg.get("datasets", [])
    if not dataset_entries:
        print("No dataset entries found in configuration; nothing to run.")
        return

    config_fields = {field.name for field in dataclasses.fields(EBTrainingConfig)}

    selected_name = args.dataset

    for entry in dataset_entries:
        name = entry.get("name")
        if selected_name and name != selected_name:
            continue

        manifest_value = entry.get("manifest")
        if manifest_value is None:
            print(f"[WARN] Dataset entry '{name}' missing 'manifest'; skipping")
            continue

        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = data_root / manifest_path
        if not manifest_path.exists():
            print(f"[WARN] Manifest not found for dataset '{name}': {manifest_path}")
            continue

        dataset_output_dir = output_root / (entry.get("output_subdir") or name or manifest_path.stem)
        lr_output_dir = dataset_output_dir / lr_dir_name
        recon_output_dir = dataset_output_dir / recon_dir_name
        hr_output_dir = dataset_output_dir / hr_dir_name if entry.get("use_hr", True) else None

        if save_results:
            ensure_dir(dataset_output_dir)
            ensure_dir(lr_output_dir)
            ensure_dir(recon_output_dir)
            if hr_output_dir is not None:
                ensure_dir(hr_output_dir)

        entry_params = dict(default_params)
        entry_params.update(entry.get("params", {}))
        entry_params["save_output"] = False  # disable intermediate saving during batch runs

        scales_filter = entry.get("scales")
        if scales_filter:
            scales_filter = {int(s) for s in scales_filter}
        splits_filter = entry.get("splits")
        if splits_filter:
            splits_filter = {str(s).lower() for s in splits_filter}

        processed = 0

        with manifest_path.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                split_value = row.get("split", "").lower()
                if splits_filter and split_value not in splits_filter:
                    continue

                try:
                    scale = int(row.get("scale", entry_params.get("scale_factor", 1)))
                except (TypeError, ValueError):
                    print(f"[WARN] Invalid scale in row: {row}")
                    continue

                if scales_filter and scale not in scales_filter:
                    continue

                lr_rel = row.get("lr_path")
                if not lr_rel:
                    continue
                lr_path = (data_root / lr_rel).expanduser()
                if not lr_path.exists():
                    print(f"[WARN] LR image missing: {lr_path}")
                    continue

                hr_path = None
                if entry.get("use_hr", True):
                    hr_rel = row.get("hr_path")
                    if not hr_rel:
                        print(f"[WARN] HR path missing for entry: {row}")
                        continue
                    hr_path_candidate = (data_root / hr_rel).expanduser()
                    if not hr_path_candidate.exists():
                        print(f"[WARN] HR image missing: {hr_path_candidate}")
                        continue
                    hr_path = hr_path_candidate

                pair_id = row.get("pair_id") or Path(lr_rel).stem
                output_basename = f"{pair_id}_x{scale}"

                if save_results:
                    lr_dest = lr_output_dir / f"{output_basename}_LR{lr_path.suffix}"
                    if not lr_dest.exists():
                        ensure_dir(lr_dest.parent)
                        shutil.copy2(lr_path, lr_dest)

                    if hr_output_dir is not None and hr_path is not None:
                        hr_dest = hr_output_dir / f"{output_basename}_HR{hr_path.suffix}"
                        if not hr_dest.exists():
                            ensure_dir(hr_dest.parent)
                            shutil.copy2(hr_path, hr_dest)

                params = dict(entry_params)
                params["scale_factor"] = scale
                filtered_params = {k: params[k] for k in params if k in config_fields}
                trainer_config = EBTrainingConfig(**filtered_params)

                trainer = EBTrainer(lr_path, hr_path=hr_path, device=device, config=trainer_config)
                trainer.save_output = False
                print(f"Processing {name or manifest_path.stem}: {output_basename} (scale x{scale})")
                sr_img, kernel = trainer.train()

                if save_results:
                    recon_output_dir.mkdir(parents=True, exist_ok=True)
                    sr_path = recon_output_dir / f"{output_basename}_SR.png"
                    Image.fromarray(sr_img).save(sr_path)

                processed += 1

        if processed == 0:
            print(f"[WARN] No samples processed for dataset '{name}'")
        else:
            print(f"Completed dataset '{name}' ({processed} samples)")


if __name__ == "__main__":  # pragma: no cover
    main()
