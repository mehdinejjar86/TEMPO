#!/bin/bash
#SBATCH --partition legacygpu
#SBATCH --output=err_logs/tempo_eval_%j.out
#SBATCH --error=err_logs/tempo_eval_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=4:00:00
#SBATCH --job-name tempo_eval

# Paths
CHECKPOINT_PATH="/home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO/checkpoint_step_685000.pth"
DATA_PATH="/home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO/datasets_atlas_specimen_ch0_ch1_80_20_split/vimeo_triplet"

# Model architecture
BASE_CHANNELS=64
TEMPORAL_CHANNELS=64
ENCODER_DEPTHS="3 3 12 3"
DECODER_DEPTHS="3 3 3 3"

echo "=== TEMPO Evaluation (VFIMamba Match) ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data: $DATA_PATH"
echo "=========================================="

# Activate conda
source /home/groups/ChangLab/govindsa/miniconda3/etc/profile.d/conda.sh
conda activate vfi_env

# Run evaluation
cd /home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO

python tempo_vfi_match.py \
    --model_path "$CHECKPOINT_PATH" \
    --data_path "$DATA_PATH" \
    --device cuda \
    --base_channels $BASE_CHANNELS \
    --temporal_channels $TEMPORAL_CHANNELS \
    --encoder_depths $ENCODER_DEPTHS \
    --decoder_depths $DECODER_DEPTHS \
    --use_cross_scale

echo "=== Evaluation Complete ==="