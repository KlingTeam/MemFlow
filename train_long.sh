#!/bin/bash

# Project path and config
CONFIG=configs/train_long.yaml
LOGDIR=logs/train_long
mkdir -p $LOGDIR
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=2 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --no-one-logger