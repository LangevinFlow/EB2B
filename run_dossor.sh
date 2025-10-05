#!/bin/bash

# DOSSR (Deep Omni Super-Resolution) Inference Script
# Usage: ./run_dossor.sh

# Configuration
INPUT_DIR="/mnt/DATA/EB2B_Image_Sup_Res_Project/DIV2K/EB2B_REC/"           # Path to input low-resolution images
CONFIG_FILE="DoSSR/configs/model/cldm_v21.yaml"  # Model configuration file
CKPT_FILE="/mnt/DATA/LargeModelCheckpoints/dossr/dossr_default.ckpt"     # Path to model checkpoint
OUTPUT_DIR="/mnt/DATA/EB2B_Image_Sup_Res_Project/DIV2K/EB2B_DoSSR_REC/"         # Output directory for results
STEPS=5                                     # Number of diffusion steps
SR_SCALE=1                                 # Super-resolution scale factor
COLOR_FIX="wavelet"                         # Color fix type: wavelet, adain, or none
DEVICE="cuda"                               # Device: cuda or cpu

# GPU Selection (uncomment ONE method below)
# Method 1: Use specific GPU by ID (recommended)
# export CUDA_VISIBLE_DEVICES=0              # Use GPU 0
export CUDA_VISIBLE_DEVICES=0             # Use GPU 1 (ACTIVE - GPU 1 is free)
# export CUDA_VISIBLE_DEVICES=2              # Use GPU 2
# export CUDA_VISIBLE_DEVICES=0,1            # Use GPUs 0 and 1

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Method 2: Let the script auto-select (uses default GPU)
# No export needed, just leave DEVICE="cuda"

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Run DOSSR inference
# Use PYTHONPATH to prioritize DoSSR's ldm module over any system ldm package
cd DoSSR

# Handle both absolute and relative paths
if [[ "$INPUT_DIR" = /* ]]; then
    INPUT_PATH="$INPUT_DIR"
else
    INPUT_PATH="../$INPUT_DIR"
fi

if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_PATH="$OUTPUT_DIR"
else
    OUTPUT_PATH="../$OUTPUT_DIR"
fi

PYTHONPATH="$(pwd):$PYTHONPATH" python inference.py \
    --input "$INPUT_PATH" \
    --config "configs/model/cldm_v21.yaml" \
    --ckpt "$CKPT_FILE" \
    --steps "$STEPS" \
    --sr_scale "$SR_SCALE" \
    --color_fix_type "$COLOR_FIX" \
    --output "$OUTPUT_PATH" \
    --device "$DEVICE" \
    --tiled \
    --tile_size 512 \
    --tile_stride 256
cd ..

echo "DOSSR inference completed. Results saved to: $OUTPUT_DIR"

