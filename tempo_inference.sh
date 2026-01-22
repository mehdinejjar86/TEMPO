#!/bin/bash
#SBATCH --partition legacygpu
#SBATCH --output=err_logs/inffft%j.out        ### File in which to store job output
#SBATCH --error=err_logs/inffft%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=336:00:00
#SBATCH --job-name inffft

#!/bin/bash
# run_tempo_eval.sh - Evaluate TEMPO on confocal data

# Set paths (modify these for your setup)
CHECKPOINT_PATH="/home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO/checkpoint_step_685000.pth"
DATA_PATH="/home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO/datasets_atlas_specimen_ch0_ch1_80_20_split/vimeo_triplet"

# Model architecture (should match your training config)
BASE_CHANNELS=64
TEMPORAL_CHANNELS=64
ENCODER_DEPTHS="3 3 12 3"
DECODER_DEPTHS="3 3 3 3"

echo "=== TEMPO Confocal Evaluation ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data: $DATA_PATH"
echo "=================================="

# Activate conda environment
source /home/groups/ChangLab/govindsa/miniconda3/etc/profile.d/conda.sh
conda activate vfi_env

# Run evaluation
cd /home/groups/ChangLab/govindsa/confocal_project/TEMPO/code/TEMPO

python tempo_inference.py \
  --model_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --device cuda \
  --base_channels $BASE_CHANNELS \
  --temporal_channels $TEMPORAL_CHANNELS \
  --encoder_depths $ENCODER_DEPTHS \
  --decoder_depths $DECODER_DEPTHS \
  --use_cross_scale

echo "=== Evaluation Complete ==="