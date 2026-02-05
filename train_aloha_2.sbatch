#!/bin/bash


#SBATCH --job-name=wzj_2
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=train_aloha_test_1_8_2.out


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
#/project/peilab/wzj/.cache/huggingface/lerobot/folding_cloth-1_8-50_repo/data/chunk-000/episode_000028.parquet
#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
#bash vla-server.bash
export XDG_CACHE_HOME=/project/peilab/wzj/.cache
#bash eval_distributed.sh adjust_bottle demo_clean demo_clean 50 0 0 
bash /project/peilab/wzj/RoboTwin/policy/openpi_test/finetune.sh pi0_base_aloha_folding_full_test fold_cloth_2 0,1
#bash quick_eval.bash
#uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_base_torch_full --exp_name pytorch_handover_block