# #!/bin/bash

# Project path and config
CONFIG=configs/train_init.yaml
LOGDIR=logs/train_init

mkdir -p $LOGDIR
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=2 \
  --master_port=29500 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --no-one-logger 2>&1 | tee $LOGDIR/log.txt \