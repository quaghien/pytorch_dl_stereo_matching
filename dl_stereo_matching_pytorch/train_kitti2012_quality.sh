#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hqh

python -m dl_stereo_matching_pytorch.train \
  --data-root /home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data \
  --util-root /home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data \
  --model-dir /home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/model_win37_kitti2012_quality \
  --data-version kitti2012 \
  --net-type win37_dep9 \
  --patch-size 37 \
  --disp-range 256 \
  --optimizer adam \
  --weight-decay 5e-4 \
  --learning-rate 1e-3 \
  --num-tr-img 160 \
  --num-val-img 34 \
  --num-val-loc 5000 \
  --train-samples-per-epoch 50000 \
  --batch-size 128 \
  --num-iter 40000 \
  --eval-every 100
