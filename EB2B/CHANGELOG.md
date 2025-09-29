# EB2B Changelog

## Phase 2: Made Parameters Configurable (Latest)

### Summary
Fixed hardcoded parameters in EB2B to make them properly configurable via command line arguments, while keeping defaults that match DKP's actual behavior. EB2B is now superior to DKP!

### Changes

#### 1. `main.py` - Updated Argument Defaults and Added Parameters
- **Fixed `--eb-lr` default**: Changed from `1e-4` to `5e-2` (matches DKP's hardcoded value)
- **Added `--I-loop-x`**: Controls inner DIP iterations per outer iteration (default: 5)
- **Added `--I-loop-k`**: Controls kernel gradient accumulation frequency (default: 3)
- **Added `--grad-loss-lr`**: Controls gradient regularization weight (default: 1e-3)
- **Updated help text**: Clarified that max-iters is divided by I-loop-x

#### 2. `trainer.py` - Removed Hardcoded Values
- **Before**: Used hardcoded `learning_rate=5e-2`, `num_steps=25`, `prior_weight=1e-4`
- **After**: Uses `self.config.eb_lr`, `self.config.eb_steps`, `self.config.eb_prior_weight`
- **Result**: Parameters are now actually configurable via command line!

#### 3. `empirical_bayes.py` - Updated Documentation
- Removed notes about DKP ignoring parameters (no longer relevant)
- Simplified docstrings to reflect that defaults match DKP
- Kept gradient clipping fix (only clips theta and shift, not eigenvalues)

#### 4. Documentation Updates
- **README.md**: Added examples of configurable parameters, noted DKP bugs
- **FIXES_SUMMARY.md**: Complete history of all issues found and fixes applied
- **TESTING_GUIDE.md**: Instructions for verifying behavior and testing new configurability

### Benefits

**Before Phase 2:**
- ✅ Matched DKP's default behavior
- ❌ Had hardcoded values like DKP
- ❌ Command line args were ignored (mirroring DKP's bugs)

**After Phase 2:**
- ✅ Matches DKP's default behavior
- ✅ **Properly respects command line arguments**
- ✅ **Fully configurable** (better than DKP!)
- ✅ Can tune parameters for experimentation

### Usage Examples

#### Default behavior (matches DKP):
```bash
python main.py --dataset butterfly --sf 4
```

#### Custom tuning (now works!):
```bash
python main.py --dataset butterfly --sf 4 \
  --max-iters 100 \
  --eb-steps 50 \
  --eb-lr 1e-2 \
  --I-loop-x 3 \
  --grad-loss-lr 5e-4
```

#### Match original DKP command intention:
```bash
# What user intended with DKP (but DKP ignored these):
python main.py --dataset butterfly --sf 4 \
  --eb-steps 15 \
  --eb-lr 5e-4 \
  --max-iters 100
# EB2B actually uses these values!
```

### Verification

All defaults match DKP's hardcoded behavior:
- `eb_lr`: 5e-2 ✓
- `eb_steps`: 25 ✓
- `eb_prior_weight`: 1e-4 ✓
- `I_loop_x`: 5 ✓
- `I_loop_k`: 3 ✓
- `grad_loss_lr`: 1e-3 ✓

But now they're all configurable!

---

## Phase 1: Fixed Core Mismatches

### Summary
Initial fixes to make EB2B match DKP's actual behavior (including DKP's bugs).

### Issues Fixed

1. **Iteration count bug**: Fixed max_iters not being divided by I_loop_x
   - Was running 5x too many iterations
   - Added `actual_max_iters` property

2. **SSIM threshold bug**: Fixed SSIM iterations not being divided by I_loop_x
   - Was using wrong threshold (80 instead of 16)
   - Added `ssim_threshold` property

3. **Gradient clipping bug**: Fixed to only clip rotation and shift parameters
   - Was clipping all parameters including eigenvalues
   - Now matches DKP behavior

4. **Discovered DKP bugs**: Found that DKP ignores command line args
   - DKP hardcodes `lr=5e-2` (ignores `--eb-lr`)
   - DKP uses default `steps=25` (ignores `--eb-steps`)
   - Initially mirrored this behavior to match DKP

### Result
EB2B produced identical results to DKP with default settings.

---

## Combined Impact

**Original EB2B Issues:**
- ❌ Wrong iteration count (5x too many)
- ❌ Wrong SSIM threshold
- ❌ Wrong gradient clipping
- ❌ Wrong default parameters

**After All Fixes:**
- ✅ Correct iteration count matching DKP
- ✅ Correct SSIM threshold matching DKP
- ✅ Correct gradient clipping matching DKP
- ✅ Correct defaults matching DKP
- ✅ **Bonus: Fully configurable (better than DKP!)**

EB2B is now a **superior implementation** that matches DKP's behavior by default but allows full control over all parameters!
