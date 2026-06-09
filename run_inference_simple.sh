#!/usr/bin/env bash
set -e

cd /data/users/gaoyin/2024_LYB/PhysHazeDiffusion

CUDA_VISIBLE_DEVICES=0 /data/users/gaoyin/miniconda/envs/LHTD/bin/python inference_phys_hazegen.py \
  --config configs/train/phys_stage.yaml \
  --sd_path weights/v2-1_512-ema-pruned.ckpt \
  --controlnet_path experiment/phys_hazegen_rerun_20260606_144840/stage2_real_adapt/checkpoints/stage2_final.pt \
  --input /data/users/gaoyin/datasets/dehaze/OTS/test/clear \
  --output_dir outputs/SOTS_rtts \
  --device cuda:0 \
  --no-use_depth_condition \
  --no-skip_existing
