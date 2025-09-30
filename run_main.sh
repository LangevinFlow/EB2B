#!/usr/bin/env bash
# Run EB2B on simulated degraded datasets defined in the config.

set -euo pipefail

conda activate basedl >/dev/null 2>&1 || source activate basedl

#python -m EB2B.main_sim --config EB2B/Config/superres_datasets.json


python -m EB2B.main_sim \
  --config EB2B/Config/superres_datasets.json
