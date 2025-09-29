# Testing Guide: Verifying EB2B Matches and Improves Upon DKP

## Overview

EB2B now:
1. ✅ Matches DKP's default behavior exactly
2. ✅ **Properly respects command line arguments** (unlike DKP which ignores them!)
3. ✅ Allows full configurability of all parameters

## Quick Test: Default Behavior

To verify that EB2B matches DKP's default behavior:

### Run DKP (from DKP/DIPDKP directory):
```bash
cd DKP/DIPDKP
python main.py --kernel_source eb --dataset butterfly
```

### Run EB2B (from EB2B directory):
```bash
cd EB2B
python main.py --dataset butterfly --sf 4
```

Both should produce **identical results** because EB2B's defaults match DKP's hardcoded values.

## Test: Command Line Arguments

Here's where EB2B is **better than DKP**:

### DKP with custom args (IGNORED!):
```bash
cd DKP/DIPDKP
python main.py \
  --kernel_source eb \
  --eb_steps 15 \
  --eb_lr 5e-4 \
  --max_iters 100 \
  --dataset butterfly
```
**DKP behavior**: Ignores `--eb_steps 15` and `--eb_lr 5e-4`, uses hardcoded values (25 and 5e-2).

### EB2B with custom args (RESPECTED!):
```bash
cd EB2B
python main.py \
  --dataset butterfly \
  --sf 4 \
  --eb-steps 15 \
  --eb-lr 5e-4 \
  --max-iters 100
```
**EB2B behavior**: Actually uses `--eb-steps 15` and `--eb-lr 5e-4` as specified!

## What to Compare

### 1. Default Run Comparison

When both use defaults, they should match exactly:

**Iteration Count:**
- Outer iterations: 1000 // 5 = **200**
- Inner iterations per outer: **5**
- Total updates: 200 × 5 = **1000**

**EB Kernel Parameters:**
- Learning rate: **5e-2**
- Steps per call: **25**
- Prior weight: **1e-4**

**Training Behavior:**
- SSIM loss for first: **80 // 5 = 16** outer iterations
- Then switches to MSE loss

### 2. Custom Arguments Test

To verify EB2B properly handles custom args:

```bash
# EB2B with custom parameters
python main.py --dataset butterfly --sf 4 \
  --max-iters 100 \
  --eb-steps 30 \
  --eb-lr 1e-2 \
  --I-loop-x 3
```

Expected behavior:
- Outer iterations: 100 // 3 = **33** (uses custom I-loop-x!)
- EB steps per call: **30** (not default 25)
- EB learning rate: **1e-2** (not default 5e-2)

### 3. Output Comparison

Compare final outputs from default runs:
- **DKP**: `DKP/data/log_DIPDKP/butterfly_DIPDKP_lr_x4_4_DIPDKP/butterfly.png`
- **EB2B**: `EB2B/outputs/butterfly_x4/butterfly.png`

Also compare kernels:
- **DKP**: `*.mat` file with kernel
- **EB2B**: `butterfly.mat` and `butterfly_kernel.png`

## Expected Results

### Default Runs Should Match:
- ✅ Same number of iterations
- ✅ Similar loss values at each iteration (within 1e-6)
- ✅ Similar final PSNR/SSIM (within 0.1 dB / 0.001)
- ✅ Visually identical output images
- ✅ Nearly identical kernel estimates

### Custom Args Tests:
- ✅ **EB2B respects the arguments** (check iteration counts, loss behavior)
- ✅ **DKP ignores the arguments** (uses hardcoded values regardless)

## Configuration Options in EB2B

EB2B now supports full configuration:

```bash
python main.py \
  --dataset butterfly \
  --sf 4 \
  --max-iters 100         # Outer iters (divided by I-loop-x)
  --eb-steps 25           # EB optimization steps per call
  --eb-lr 5e-2            # EB Adam learning rate
  --eb-prior-weight 1e-4  # Inverse eigenvalue penalty weight
  --dip-lr 5e-3           # DIP network learning rate
  --I-loop-x 5            # Inner DIP updates per outer iteration
  --I-loop-k 3            # Kernel gradient accumulation frequency
  --grad-loss-lr 1e-3     # Gradient regularization weight
  --noise-sigma 1.0       # Input noise std for DIP
  --kernel-size 19        # Override kernel size (default: sf*4+3)
```

All parameters are now actually used (unlike DKP)!

## Debugging Mismatches

If default runs don't match, check:

1. **Random seeds**: Both use seed=0, verify identical initialization
2. **Image preprocessing**: Ensure images are loaded identically
3. **SSIM threshold**: Should be 80 // I_loop_x = 16
4. **Temperature annealing**: Should use actual_max_iters (200), not 1000

If custom args don't work in EB2B:
1. Check that config values are passed to EmpiricalBayesConfig
2. Verify actual_max_iters property uses I_loop_x correctly
3. Check that trainer uses self.config values, not hardcoded

## Success Criteria

### For Default Behavior Matching:
- ✅ Same total iterations (1000)
- ✅ Same loss trajectory
- ✅ Same final PSNR/SSIM (within tolerance)
- ✅ Identical visual quality

### For Improved Functionality:
- ✅ Custom `--eb-steps` changes EB optimization steps
- ✅ Custom `--eb-lr` changes EB learning rate
- ✅ Custom `--I-loop-x` changes outer/inner loop ratio
- ✅ All parameters are actually used (not ignored like DKP)

## Example Test Session

```bash
# Test 1: Default behavior (should match DKP)
cd EB2B
python main.py --dataset butterfly --sf 4
# Check: 200 outer iterations, PSNR similar to DKP

# Test 2: Custom EB parameters (should use them!)
python main.py --dataset butterfly --sf 4 \
  --eb-steps 50 --eb-lr 1e-2 --max-iters 50
# Check: 50//5=10 outer iterations, 50 EB steps per call (not 25!)

# Test 3: Custom loop structure
python main.py --dataset butterfly --sf 4 \
  --I-loop-x 10 --max-iters 100
# Check: 100//10=10 outer iterations, 10 inner per outer

# Test 4: Verify logs
cat outputs/butterfly_x4/*.png  # Should have multiple checkpoints
```

## Advanced: Parameter Sensitivity

Test different parameters to explore behavior:

```bash
# Aggressive EB optimization
python main.py --dataset butterfly --sf 4 \
  --eb-steps 100 --eb-lr 0.1

# Conservative EB optimization  
python main.py --dataset butterfly --sf 4 \
  --eb-steps 10 --eb-lr 1e-3

# Different inner/outer ratios
python main.py --dataset butterfly --sf 4 \
  --I-loop-x 10  # Fewer outer, more inner per outer
```

These experiments are now possible in EB2B but not in DKP!