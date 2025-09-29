# Empirical Bayes Blur Kernel Estimation

EB2B provides a standalone reimplementation of the Empirical Bayes (EB)
super-resolution pipeline from the DKP codebase. The estimator optimises an
anisotropic Gaussian blur kernel—rotation, eigenvalues and sub-pixel shifts—by
minimising the discrepancy between a high-resolution estimate produced by a
Deep Image Prior (DIP) generator and the observed low-resolution input.

**Key Improvements Over DKP:**
- ✅ **Properly respects command line arguments** (DKP ignores `--eb-lr` and `--eb-steps`)
- ✅ **Fully configurable** parameters (I_loop_x, I_loop_k, grad_loss_lr, etc.)
- ✅ **Correct defaults** that match DKP's actual hardcoded behavior
- ✅ **Clean, modular code** with type hints and documentation

## Quick start

```python
import torch
from EB2B import EmpiricalBayesKernelEstimator

# High-resolution prediction from your image generator (N=1 expected)
sr = torch.rand(1, 3, 128, 128)

# Observed low-resolution image
lr = torch.rand(1, 3, 64, 64)

estimator = EmpiricalBayesKernelEstimator(kernel_size=15, scale_factor=2)

# Refine the kernel parameters to align the two observations
kernel = estimator.optimise(sr, lr, num_steps=50)
```

The resulting `kernel` tensor has shape `[1, 1, kernel_size, kernel_size]` and
sums to one. You can reuse the estimator instance across iterations by calling
`optimise` multiple times, or reset it with `reset_state()` when starting from
scratch.

See `tests/test_empirical_bayes.py` for additional, self-contained examples.

## Command-line workflow

The original DKP training script couples the EB optimiser with an MCMC kernel
prior. EB2B recreates only the EB branch, providing its own runner that shares
the same dataset layout:

### Default usage (matches DKP behavior):
```bash
conda activate basedl
python -m EB2B.main --dataset butterfly --sf 4
```

This uses the correct defaults that match DKP's actual behavior:
- `--max-iters 1000` → 1000÷5 = 200 outer iterations
- `--eb-steps 25` → 25 EB optimization steps per outer iteration
- `--eb-lr 5e-2` → EB Adam learning rate (DKP's hardcoded value)
- `--I-loop-x 5` → 5 inner DIP updates per outer iteration

### Custom parameters (now actually work!):
```bash
python -m EB2B.main --dataset butterfly --sf 4 \
  --max-iters 100 \
  --eb-steps 50 \
  --eb-lr 1e-2 \
  --I-loop-x 3
```

Unlike DKP which ignores these parameters, EB2B actually uses them!

### Full parameter list:
```bash
python -m EB2B.main \
  --dataset butterfly \
  --sf 4 \
  --max-iters 1000         # Total iterations (divided by I-loop-x)
  --eb-steps 25            # EB optimization steps per call
  --eb-lr 5e-2             # EB learning rate (default: 5e-2, not 1e-4!)
  --eb-prior-weight 1e-4   # Inverse eigenvalue penalty
  --dip-lr 5e-3            # DIP network learning rate
  --I-loop-x 5             # Inner iterations per outer
  --I-loop-k 3             # Kernel gradient accumulation frequency
  --grad-loss-lr 1e-3      # Gradient regularization weight
```

By default the command expects LR/HR pairs under
`DKP/data/datasets/<dataset>/DIPDKP_lr_x<sf>` and `.../HR`. Outputs are written
to `EB2B/outputs/<dataset>_x<sf>`. Override the directories with
`--input-dir`, `--hr-dir`, or `--output-dir` if your data lives elsewhere. The
`--real` flag skips HR evaluation when no reference is available.

## Important Notes

**DKP Bug**: The original DKP code has bugs where command line arguments
`--eb-lr` and `--eb-steps` are stored but never used. DKP hardcodes `lr=5e-2`
and uses a default `steps=25` regardless of what you specify. EB2B fixes this
by properly using the command line arguments, with defaults that match DKP's
actual hardcoded behavior.

See `FIXES_SUMMARY.md` for detailed information about the issues found and fixed.

## Tests

Run the unit tests inside the conda environment:

```bash
conda activate basedl
pytest -q
```
