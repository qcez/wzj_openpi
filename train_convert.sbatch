#!/bin/bash

#SBATCH --job-name=wzj_convert
#SBATCH --nodes=1

#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=train_convert_torch_robotwin.out


nvidia-smi

module purge
#module load slurm
#module avail cuda
module load slurm
#module load cuda12.2/toolkit/12.2.2
#module avail tensorrt
# source /home/congcong/miniconda3/bin/activate RoboTwin
source /home/rouyangaa/anaconda3/bin/activate RoboTwin-pi0-1

#nvidia-smi
#nvcc -V
export PYTHONUNBUFFERED=1

export XDG_CACHE_HOME=/project/peilab/wzj/.cache

export WANDB_API_KEY=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
# export WANDB_API_KEY=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
# export WANDB_PROJECT="openpi"
# export WANDB_ENTITY=""

#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
#bash vla-server.bash

#bash eval_distributed.sh adjust_bottle demo_clean demo_clean 50 0 0 
#bash finetune.sh pi0_base_piper real 0,1,2,3
#bash quick_eval.bash


#  uv run python scripts/train_pytorch.py pi0_base_aloha_robotwin_full_torch \
#  --exp_name robotwin_aloha_lerobot \
#  --save_interval 10000 
#  --resume

# uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_full_torch

bash generate.sh /project/peilab/wzj/RoboTwin/policy/openpi_test/training_data/robotwin_50_task_clean robotwin_aloha_lerobot_repo

uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_full_torch

sbatch train_torch_robotwin.sbatch