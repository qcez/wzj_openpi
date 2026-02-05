#!/bin/bash


#SBATCH --job-name=wzj_rjax
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=eval_jax_hammer_replay.out


nvidia-smi

module purge
#module load slurm
#module avail cuda
module load slurm
#module load cuda12.2/toolkit/12.2.2
#module avail tensorrt
# source /home/congcong/miniconda3/bin/activate RoboTwin-pi0-1
source /home/rouyangaa/anaconda3/bin/activate RoboTwin-pi0-1
#nvidia-smi
#nvcc -V
export PYTHONUNBUFFERED=1
#/project/peilab/wzj/.cache/huggingface/lerobot/folding_cloth-1_8-50_repo/data/chunk-000/episode_000028.parquet
#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
#bash vla-server.bash
export XDG_CACHE_HOME=/project/peilab/wzj/.cache
#bash eval_distributed.sh adjust_bottle demo_clean demo_clean 50 0 0 

# ckpt_path like: policy/pi0/checkpoints/pi0_base_aloha_robotwin_full/demo_clean/30000
# bash eval.sh ${task_name} ${task_config} ${train_config_name} ${model_name} ${seed} ${gpu_id}

# bash eval.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_lora beat_block_hammer_jax 0 0
bash eval_replay.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_lora beat_block_hammer_jax 0 0

# bash eval.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_full demo_clean 0 0
# This command trains the policy using the `demo_clean` setting ($model_name)
# and evaluates it using the same `demo_clean` setting ($task_config).

# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized pi0_base_aloha_robotwin_full demo_clean 0 0