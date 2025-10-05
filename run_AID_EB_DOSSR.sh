#!/usr/bin/env bash
set -Eeuo pipefail

export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export SDL_VIDEODRIVER=dummy
unset DISPLAY

############################################
#               EB2B 配置
############################################
INPUT_ROOT="/mnt/DATA/AID/AID_LR_128"
EB2B_OUT_ROOT="/mnt/DATA/EB2B_Image_Sup_Res_Project/AID"
LOG_ROOT="/mnt/DATA/EB2B_Image_Sup_Res_Project/AID/Recon_log"
mkdir -p "$EB2B_OUT_ROOT" "$LOG_ROOT"

# EB2B 参数
EB2B_DATASET="DIV2K"
EB2B_SF=2
EB2B_MAX_ITERS=500
EB2B_EB_STEPS=2
EB2B_KERNEL=4
EB2B_EB_LR=0.002
EB2B_DIP_LR=0.002
EB2B_REAL_FLAG="--real"   # 不需要时可设为空串 ""

############################################
#               DoSSR 配置
############################################
DOSSR_REPO_DIR="DoSSR"
DOSSR_CONFIG="configs/model/cldm_v21.yaml"
DOSSR_CKPT="/mnt/DATA/LargeModelCheckpoints/dossr/dossr_default.ckpt"
DOSSR_STEPS=5
DOSSR_SR_SCALE=2
DOSSR_COLOR_FIX="wavelet"
DOSSR_DEVICE="cuda"

DOSSR_OUT_ROOT="/mnt/DATA/EB2B_Image_Sup_Res_Project/AID_DoSSR_EB2B_REC"
mkdir -p "$DOSSR_OUT_ROOT"

# GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############################################
#               工具函数
############################################
run_eb2b() {
  local in_dir="$1"     # 原始 LR 子目录
  local out_dir="$2"    # EB2B 输出子目录
  local name="$3"       # 子目录名
  local log_file="$LOG_ROOT/EB2B_${name}.log"

  echo "=== [EB2B] Processing: $name"
  mkdir -p "$out_dir"
  python -m EB2B.main \
      --dataset "$EB2B_DATASET" \
      --sf "$EB2B_SF" \
      --input-dir "$in_dir" \
      --output-dir "$out_dir" \
      --output-log-dir "$LOG_ROOT" \
      ${EB2B_REAL_FLAG:+$EB2B_REAL_FLAG} \
      --max-iters "$EB2B_MAX_ITERS" \
      --eb-steps "$EB2B_EB_STEPS" \
      --kernel-size "$EB2B_KERNEL" \
      --eb-lr "$EB2B_EB_LR" \
      --dip-lr "$EB2B_DIP_LR" 2>&1 | tee "$log_file"
}

run_dossr() {
  local in_dir="$1"     # EB2B 输出作为 DoSSR 输入
  local out_dir="$2"    # DoSSR 输出子目录
  local name="$3"       # 子目录名
  local log_file="$LOG_ROOT/DoSSR_${name}.log"

  echo "=== [DoSSR] Processing EB2B output: $name"
  mkdir -p "$out_dir"

  # 确认仓库存在
  if [[ ! -d "$DOSSR_REPO_DIR" ]]; then
    echo "[Error] DoSSR repo dir not found: $DOSSR_REPO_DIR" >&2
    return 1
  fi

  pushd "$DOSSR_REPO_DIR" >/dev/null
  PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}" python inference.py \
    --input "$in_dir" \
    --config "$DOSSR_CONFIG" \
    --ckpt "$DOSSR_CKPT" \
    --steps "$DOSSR_STEPS" \
    --sr_scale "$DOSSR_SR_SCALE" \
    --color_fix_type "$DOSSR_COLOR_FIX" \
    --output "$out_dir" \
    --device "$DOSSR_DEVICE" \
    --tiled \
    --tile_size 512 \
    --tile_stride 256 2>&1 | tee "$log_file"
  popd >/dev/null
}

############################################
#               主流程（两阶段）
############################################

# 收集需要处理的子目录
folders=()
for sub in "$INPUT_ROOT"/*; do
  [[ -d "$sub" ]] || continue
  folders+=("$sub")
done

# 阶段 1：全部跑 EB2B
for sub in "${folders[@]}"; do
  folder_name="$(basename "$sub")"
  eb2b_out_dir="$EB2B_OUT_ROOT/$folder_name"

  echo "-------------------------------------------"
  echo "[Stage 1/2 EB2B] Folder: $folder_name"
  echo "EB2B in  : $sub"
  echo "EB2B out : $eb2b_out_dir"
  echo "-------------------------------------------"

  # 若 EB2B 输出已存在且非空，可自行决定是否跳过
  if [[ -d "$eb2b_out_dir" && -n "$(ls -A "$eb2b_out_dir" 2>/dev/null || true)" ]]; then
    echo "[Skip] EB2B output already exists for $folder_name -> $eb2b_out_dir"
  else
    run_eb2b "$sub" "$eb2b_out_dir" "$folder_name"
  fi
done

# 阶段 2：全部跑 DoSSR（基于 EB2B 输出）
for sub in "${folders[@]}"; do
  folder_name="$(basename "$sub")"
  eb2b_out_dir="$EB2B_OUT_ROOT/$folder_name"
  dossr_out_dir="$DOSSR_OUT_ROOT/$folder_name"

  echo "-------------------------------------------"
  echo "[Stage 2/2 DoSSR] Folder: $folder_name"
  echo "DoSSR in  : $eb2b_out_dir"
  echo "DoSSR out : $dossr_out_dir"
  echo "-------------------------------------------"

  # 若 DoSSR 输出已存在且非空，则跳过
  if [[ -d "$dossr_out_dir" && -n "$(ls -A "$dossr_out_dir" 2>/dev/null || true)" ]]; then
    echo "[Skip] DoSSR output already exists for $folder_name -> $dossr_out_dir"
    continue
  fi

  # 没有 EB2B 输出时给出清晰提示
  if [[ ! -d "$eb2b_out_dir" || -z "$(ls -A "$eb2b_out_dir" 2>/dev/null || true)" ]]; then
    echo "[Warn] EB2B output missing/empty for $folder_name -> $eb2b_out_dir; skipping DoSSR."
    continue
  fi

  run_dossr "$eb2b_out_dir" "$dossr_out_dir" "$folder_name"
done

echo "All done."
