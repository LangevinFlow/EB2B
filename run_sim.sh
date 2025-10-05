#!/usr/bin/env bash
# Generate simulated LR datasets for Set5, Set14, BSD100, and Urban100. Set14 BSD100 Urban100 

set -euo pipefail

conda activate basedl >/dev/null 2>&1 || source activate basedl

python SimData/generate_sim_data.py \
  --config EB2B/Config/superres_datasets.json \
  --data-root /mnt/DATA/BlindSuperRes \
  --simdata-root /mnt/DATA/DegradedResolution \
  --datasets  IXIT1 \
  --num-kernels 5 \
  --kernel-type gaussian \
  --seed 0 \
  --overwrite

