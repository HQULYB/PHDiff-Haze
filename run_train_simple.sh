#!/usr/bin/env bash
set -e

cd /data/users/gaoyin/2024_LYB/PhysHazeDiffusion

CUDA_VISIBLE_DEVICES=0 /data/users/gaoyin/miniconda/envs/LHTD/bin/python train_phys_hazegen.py \
  --stage both \
  --sd_path weights/v2-1_512-ema-pruned.ckpt \
  --exp_dir experiment/phys_hazegen_simple \
  --synthetic_clean_dir /data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/gt \
  --synthetic_hazy_dir /data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/hazy \
  --synthetic_depth_dir /data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/depth \
  --real_clear_dir /data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/gt \
  --real_clear_depth_dir /data/users/gaoyin/datasets/dehaze/Haze_Gen/Stage1/Train/depth \
  --real_hazy_dir /data/users/gaoyin/datasets/dehaze/URHI
