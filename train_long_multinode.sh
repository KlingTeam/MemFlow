#!/bin/bash

# --- User-configurable section ---
NNODES=2                               # Total number of machines
NPROC_PER_NODE=8                       # Number of GPUs per machine
MASTER_ADDR="10.82.141.18"             # IP address of the master node (must be accessible by all nodes)
MASTER_PORT=29500                      # Port on the master node
NODE_RANK=$1                           # Rank of the current node (starting from 0), passed via command-line argument

# --- Script arguments ---
CONFIG=configs/train_long.yaml
LOGDIR=logs/train_long
mkdir -p $LOGDIR
WANDB_SAVE_DIR=wandb
echo "CONFIG=$CONFIG"

# --- Argument check ---
if [ -z "$NODE_RANK" ]; then
    echo "Error: Please provide the node rank (NODE_RANK) as the first argument."
    echo "Usage: bash run_multi_node.sh 0  (on the master node)"
    echo "       bash run_multi_node.sh 1  (on the first worker node)"
    exit 1
fi

# --- Print configuration ---
echo "========================================"
echo "Starting multi-node training..."
echo "Total number of nodes (NNODES): $NNODES"
echo "Processes per node (NPROC_PER_NODE): $NPROC_PER_NODE"
echo "Master node address (MASTER_ADDR): $MASTER_ADDR:$MASTER_PORT"
echo "Current node rank (NODE_RANK): $NODE_RANK"
echo "========================================"

# --- Build and launch the torchrun command ---
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --rdzv_id=job_123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --node_rank=$NODE_RANK \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --no-one-logger 2>&1 | tee $LOGDIR/train_init_node_${NODE_RANK}.txt