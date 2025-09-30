#!/usr/bin/env bash
# Generate simulated LR datasets for Set5, Set14, BSD100, and Urban100.

set -euo pipefail

python SimData/generate_sim_data.py \
  --config EB2B/Config/superres_datasets.json \
  --datasets Set5 \
  --num-kernels 3 \
  --kernel-type gaussian \
  --seed 0 \
  --overwrite \
  --simdata-root /mnt/DATA/DegradedResolution


#Set14 BSD100 Urban100 