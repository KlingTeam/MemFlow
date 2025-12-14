CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=29501 \
  inference.py \
  --config_path configs/inference.yaml