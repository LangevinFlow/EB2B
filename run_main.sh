#!/usr/bin/env bash
# Run EB2B on simulated degraded datasets defined in the config.

set -euo pipefail

conda activate basedl >/dev/null 2>&1 || source activate basedl

#python -m EB2B.main_sim --config EB2B/Config/superres_datasets.json


# Run on the formatted dataset
python -m EB2B.main_sim \
  --config EB2B/Config/superres_datasets.json \
  --dataset IXIT1  


# Run directly on a directory for DIV2K

# python -m EB2B.main --dataset DIV2K --sf 2 \
#   --input-dir /mnt/DATA/EB2B_Image_Sup_Res_Project/DIV2K/DIV2K_LR_Patches/ \
#   --output-dir /mnt/DATA/EB2B_Image_Sup_Res_Project/DIV2K/EB2B_REC/ \
#   --output-log-dir /mnt/DATA/EB2B_Image_Sup_Res_Project/OLI2MSI/Recon_log/ \
#   --real \
#   --max-iters 500 \
#   --eb-steps 1 \
#   --kernel-size 6 \
#   --eb-lr 0.002 \
#   --dip-lr 0.002



# Run directly on a directory for OLI2MSI

# python -m EB2B.main \
#   --dataset OLI2MSI \
#   --sf 3 \
#   --input-dir /mnt/DATA/OLI2MSI/OLI2MSI_Reshape/test_lr \
#   --output-dir /mnt/DATA/EB2B_Image_Sup_Res_Project/OLI2MSI/ \
#   --output-log-dir /mnt/DATA/EB2B_Image_Sup_Res_Project/OLI2MSI/Recon_log/ \
#   --real   --max-iters 1000   --eb-steps 1   --kernel-size 6 \
#   --eb-lr 0.0022   --dip-lr 0.003