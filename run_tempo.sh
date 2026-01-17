#!/bin/bash
#SBATCH --job-name=tempo_8a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:6
#SBATCH --time=7-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/tempo_8a100_%j.out
#SBATCH --error=logs/tempo_8a100_%j.err

export NCCL_TIMEOUT=1200
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # ← ADD THIS!

source /home/groups/ChangLab/govindsa/miniconda3/etc/profile.d/conda.sh
conda activate vfi_env

cd /home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO
mkdir -p logs

echo "=== Starting Training (8x A100) ==="
torchrun \
  --nproc_per_node=6 \
  train_tempo_mixed.py \
  --data_root datasets_atlas_specimen_ch0_ch1_80_20_split/Video_vimeo_triplet \
  --x4k_root datasets_atlas_specimen_ch0_ch1_80_20_split/Extreme \
  --x4k_step 1 3 \
  --x4k_crop 448 \
  --vimeo_ratio 0.5 \
  --batch_size 4 \
  --epochs 100 \
  --base_channels 64 \
  --temporal_channels 64 \
  --encoder_depths 3 3 12 3 \
  --decoder_depths 3 3 3 3 \
  --amp --amp_dtype bf16 \
  --distributed \
  --lr 1.4e-4 \
--resume /home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/checkpoint_all/checkpoints/checkpoint_step_595000.pth \
  --exp_name "tempo_8a100_resumed" \
  --use_wandb

echo "=== Training Complete ==="