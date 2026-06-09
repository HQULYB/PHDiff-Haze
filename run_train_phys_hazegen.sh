#!/usr/bin/env bash
set -e

cd /data/users/gaoyin/2024_LYB/PhysHazeDiffusion

GPU_IDS=0,1
NUM_PROCESSES=2
PYTHON_BIN=/data/users/gaoyin/miniconda/envs/LHTD/bin/python
LOG_DIR=logs
DETACH=${DETACH:-true}

CONFIG=configs/train/phys_stage.yaml
SD_PATH=weights/v2-1_512-ema-pruned.ckpt
EXP_DIR=./experiment/phys_hazegen

SYNTHETIC_CLEAN_DIR=/data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/gt
SYNTHETIC_HAZY_DIR=/data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/hazy
SYNTHETIC_DEPTH_DIR=/data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/depth
REAL_CLEAR_DIR=/data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/gt
REAL_CLEAR_DEPTH_DIR=/data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/depth
REAL_HAZY_DIR=/data/users/gaoyin/datasets/dehaze/URHI
EXCLUDE_NAME_KEYWORDS=dense,Dense_Haze,Dense_Haze_NTIRE19

STAGE=both
USE_DEPTH_CONDITION=true
DEPTH_CONDITION_SCALE=0.25
DEPTH_CONDITION_ZERO_CENTER=true
DEPTH_CONDITION_DROPOUT=0.25
INVERT_DEPTH_CONDITION=false

BATCH_SIZE=2
STAGE1_STEPS=30000
STAGE2_STEPS=15000
LR=1e-5
STAGE2_LR=
LAMBDA_AIRLIGHT=0.1
LAMBDA_CLIP_DIRECTION=0.05
LAMBDA_AIRLIGHT_STAGE2=0.02

PROMPT="dense gray-white foggy haze, thick realistic mist, low visibility, natural color."
CLEAR_PROMPT="sharp clear clean image, natural color, high visibility"
STAGE2_TEACHER_CKPT=
SWANLAB_RUN_NAME=phys_hazegen_density_airlight

CMD=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --num_processes "${NUM_PROCESSES}"
  train_phys_hazegen.py
  --config "${CONFIG}"
  --stage "${STAGE}"
  --sd_path "${SD_PATH}"
  --exp_dir "${EXP_DIR}"
  --synthetic_clean_dir "${SYNTHETIC_CLEAN_DIR}"
  --synthetic_hazy_dir "${SYNTHETIC_HAZY_DIR}"
  --synthetic_depth_dir "${SYNTHETIC_DEPTH_DIR}"
  --real_clear_dir "${REAL_CLEAR_DIR}"
  --real_clear_depth_dir "${REAL_CLEAR_DEPTH_DIR}"
  --real_hazy_dir "${REAL_HAZY_DIR}"
  --exclude_name_keywords "${EXCLUDE_NAME_KEYWORDS}"
  --prompt "${PROMPT}"
  --clear_prompt "${CLEAR_PROMPT}"
  --batch_size "${BATCH_SIZE}"
  --stage1_steps "${STAGE1_STEPS}"
  --stage2_steps "${STAGE2_STEPS}"
  --lr "${LR}"
  --depth_condition_scale "${DEPTH_CONDITION_SCALE}"
  --depth_condition_dropout "${DEPTH_CONDITION_DROPOUT}"
  --lambda_airlight "${LAMBDA_AIRLIGHT}"
  --lambda_clip_direction "${LAMBDA_CLIP_DIRECTION}"
  --lambda_airlight_stage2 "${LAMBDA_AIRLIGHT_STAGE2}"
  --swanlab_run_name "${SWANLAB_RUN_NAME}"
)

if [[ -n "${STAGE2_LR}" ]]; then
  CMD+=(--stage2_lr "${STAGE2_LR}")
fi
if [[ -n "${STAGE2_TEACHER_CKPT}" ]]; then
  CMD+=(--stage2_teacher_ckpt "${STAGE2_TEACHER_CKPT}")
fi
if [[ "${USE_DEPTH_CONDITION}" == "true" ]]; then
  CMD+=(--use_depth_condition)
else
  CMD+=(--no-use_depth_condition)
fi
if [[ "${DEPTH_CONDITION_ZERO_CENTER}" == "true" ]]; then
  CMD+=(--depth_condition_zero_center)
fi
if [[ "${INVERT_DEPTH_CONDITION}" == "true" ]]; then
  CMD+=(--invert_depth_condition)
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/phys_hazegen_${STAGE}_$(date +%Y%m%d_%H%M%S).log"

if [[ "${DETACH}" == "true" ]]; then
  setsid nohup env \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${CMD[@]}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started PhysHazeDiffusion training. PID: $!"
  echo "Log: ${LOG_FILE}"
else
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
fi
