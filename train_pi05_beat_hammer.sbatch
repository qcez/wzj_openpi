#!/bin/bash


#SBATCH --job-name=wzj_pi05_train
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=train_pi05_beat_hammer.out


nvidia-smi

module purge
#module load slurm
#module avail cuda
module load slurm
#module load cuda12.2/toolkit/12.2.2
#module avail tensorrt
# source /home/congcong/miniconda3/bin/activate RoboTwin
source /home/rouyangaa/anaconda3/bin/activate RoboTwin-pi0-1


export WANDB_API_KEY=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
# export WANDB_API_KEY=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
# export WANDB_PROJECT="openpi"
# export WANDB_ENTITY=""

#nvidia-smi
#nvcc -V
export PYTHONUNBUFFERED=1
#/project/peilab/wzj/.cache/huggingface/lerobot/folding_cloth-1_8-50_repo/data/chunk-000/episode_000028.parquet
#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
#bash vla-server.bash
export XDG_CACHE_HOME=/project/peilab/wzj/.cache
#bash eval_distributed.sh adjust_bottle demo_clean demo_clean 50 0 0 

# bash /project/peilab/wzj/RoboTwin/policy/openpi_test/finetune.sh pi0_base_aloha_folding_full fold_cloth_1_14 0,1,2,3,4,5,6,7

bash finetune.sh pi05_base_aloha_robotwin_lora beat_hammer_pi05_bs32 0

#resume
# bash finetune_resume.sh pi05_base_aloha_robotwin_lora beat_hammer_pi05 0 --resume

#bash quick_eval.bash
#uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_base_torch_full --exp_name pytorch_handover_block