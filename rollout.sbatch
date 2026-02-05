#!/bin/bash

#SBATCH --job-name=wzj_rollout
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=rollout_beat_block_hammer.out


nvidia-smi

module purge
#module load slurm
#module avail cuda
module load slurm
module load cuda12.2/toolkit/12.2.2
#module avail tensorrt
source /home/congcong/miniconda3/bin/activate RoboTwin-pi0-1

#nvidia-smi
#nvcc -V
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=/project/peilab/wzj/.cache
#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
#bash vla-server.bash

#bash eval_distributed.sh adjust_bottle demo_clean demo_clean 50 0 0 
#bash finetune.sh pi0_base_piper real 0,1,2,3

TASK=(
    "beat_block_hammer"
)
bash eval_distributed.sh "${TASK[@]}" demo_clean pi0_base_aloha_robotwin_full_torch robotwin_aloha_lerobot \
    0 0,1 \
    2 \
    16 \
    rollout_1.pkl \
    100000 100001

bash eval_distributed.sh "${TASK[@]}" demo_clean pi0_base_aloha_robotwin_full_torch robotwin_aloha_lerobot \
    0 0,1 \
    2 \
    16 \
    rollout_2.pkl \
    100002 100003 
    
bash eval_distributed.sh "${TASK[@]}" demo_clean pi0_base_aloha_robotwin_full_torch robotwin_aloha_lerobot \
    0 0,1 \
    2 \
    16 \
    rollout_3.pkl \
    100004 100005

bash eval_distributed.sh "${TASK[@]}" demo_clean pi0_base_aloha_robotwin_full_torch robotwin_aloha_lerobot \
    0 0,1 \
    2 \
    16 \
    rollout_4.pkl \
    100006 100007