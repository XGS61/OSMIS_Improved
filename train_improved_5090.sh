#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u verify_5090.py

EXP_NAME="${1:-rendered_us_test2_full_v2_5090}"
IMAGE_PATH="${IMAGE_PATH:-datasets/rendered_us_test2_source/image/00000.jpg}"
MASK_PATH="${MASK_PATH:-datasets/rendered_us_test2_source/mask/00000.png}"
DATASET_NAME="${DATASET_NAME:-rendered_us_test2_full_guidance}"
SOURCE_CROP_TOP="${SOURCE_CROP_TOP:-20}"
GLOBAL_DISPLACEMENT_FRAC="${GLOBAL_DISPLACEMENT_FRAC:-0.028}"
REGIONAL_DISPLACEMENT_FRAC="${REGIONAL_DISPLACEMENT_FRAC:-0.012}"
NUM_VARIANTS="${NUM_VARIANTS:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

python -u prepare_anatomy_dataset.py \
  --image "${IMAGE_PATH}" \
  --mask "${MASK_PATH}" \
  --output "datasets/${DATASET_NAME}" \
  --num-variants "${NUM_VARIANTS}" \
  --crop-top "${SOURCE_CROP_TOP}" \
  --global-displacement-frac "${GLOBAL_DISPLACEMENT_FRAC}" \
  --regional-displacement-frac "${REGIONAL_DISPLACEMENT_FRAC}" \
  --overwrite

python -u validate_guidance_dataset.py --dataset "datasets/${DATASET_NAME}"

mkdir -p "run_logs/${EXP_NAME}"
python -u train.py \
  --exp_name "${EXP_NAME}" \
  --dataset_name "${DATASET_NAME}" \
  --num_epochs "${NUM_EPOCHS}" \
  --max_size 330 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --use_kornia_augm \
  --prob_augm 0.35 \
  --prob_FA_con 0.15 \
  --prob_FA_lay 0.0 \
  --lambda_DR 0.05 \
  --lambda_seg 10.0 \
  --lambda_boundary 2.0 \
  --lambda_lowfreq 0.0 \
  --lambda_structure 4.0 \
  --lambda_texture 1.0 \
  --style_dim 64 \
  --sean_blocks 3 \
  --freq_print 1000 \
  --freq_save_loss 1000 \
  --freq_save_ckpt 1000 \
  2>&1 | tee "run_logs/${EXP_NAME}/train.log"
