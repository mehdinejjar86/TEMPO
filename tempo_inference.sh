#!/bin/bash
#SBATCH --partition legacygpu
#SBATCH --output=err_logs/inffft%j.out        ### File in which to store job output
#SBATCH --error=err_logs/inffft%j.err          ### File in which to store job error messages
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=336:00:00
#SBATCH --job-name inffft

# Activate conda environment
# source /home/groups/ChangLab/govindsa/miniconda3/bin/activate vfi_env


# inputs=(
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/input/AMTEC_001_ON_HRD_RS_2_ch0_resized"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/input/AMTEC_001_ON_HRD_RS_2_ch1_resized"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/input/AMTEC_001_ON_HRD_RS_2_ch2_resized"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/input/AMTEC_001_ON_HRD_RS_2_ch3_resized"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/input/AMTEC_001_ON_HRD_RS_2_ch4_resized"
# )

# outputs=(
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/output/AMTEC_001_ON_HRD_RS_2_ch0_resized_STEP3_Fusion"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/output/AMTEC_001_ON_HRD_RS_2_ch1_resized_STEP3_Fusion"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/output/AMTEC_001_ON_HRD_RS_2_ch2_resized_STEP3_Fusion"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/output/AMTEC_001_ON_HRD_RS_2_ch3_resized_STEP3_Fusion"
# "/home/exacloud/gscratch/ChangLab/govindsa/confocal_project/dataset/gordonImages/output/AMTEC_001_ON_HRD_RS_2_ch4_resized_STEP3_Fusion"
# )

# MODEL_PATH="/home/groups/ChangLab/govindsa/confocal_project/fusionfft/runs/2025-10-05-12-46-36/checkpoints/best_val_model.pth"
# for i in "${!inputs[@]}"; do
#     echo "Processing ${inputs[$i]}"
#     python /home/groups/ChangLab/govindsa/confocal_project/fusionfft/fusion_inference.py \
#     --model_path "$MODEL_PATH" \
#     --input_dir "${inputs[$i]}" \
#     --output_dir "${outputs[$i]}" \
#     --step 2 
#     echo "Completed ${inputs[$i]}"
# done
# echo "All jobs completed successfully!"

python tempo_inference.py 