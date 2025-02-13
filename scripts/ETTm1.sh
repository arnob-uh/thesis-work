#!/bin/bash

export PYTHONPATH=../

# Define parameters
PRED_LENS=("96" "168" "336" "720")
# NORM_TYPES=("RevIN" "SAN" "FAN" "DAIN")
NORM_TYPES=("ZScore")

# Loop over prediction lengths and normalization types
for pred_len in "${PRED_LENS[@]}"; do
  for norm_type in "${NORM_TYPES[@]}"; do
    echo "Running SCINet with pred_len=${pred_len} and norm_type=${norm_type}"

    # Additional argument for FAN normalization
    # FAN_CONFIG=""
    # if [ "$norm_type" == "FAN" ]; then
    #   FAN_CONFIG="--norm_config=\"{freq_topk:4}\""
    # fi

    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 ../experiments/SCINet.py \
       --dataset_type="ETTm1" \
       --split_type="popular" \
       --norm_type="$norm_type" \
       --device="cuda:0" \
       --batch_size=32 \
       --horizon=1 \
       --pred_len="$pred_len" \
       --windows=96 \
       --epochs=100 \
       runs --seeds='[3]'
  done
done