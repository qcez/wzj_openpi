#!/bin/bash


#SBATCH --job-name=yanzhengyang
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=train_piper.out


nvidia-smi

module purge
#module load slurm
#module avail cuda
module load slurm
#module load cuda12.2/toolkit/12.2.2
#module avail tensorrt
source /home/congcong/miniconda3/bin/activate RoboTwin

#nvidia-smi
#nvcc -V
export PYTHONUNBUFFERED=1

#python -u pipeline.py
#python -u pipeline-scenario-2.py #world_model_config.py
#bash vla-server.bash

#bash eval_distributed.sh adjust_bottle demo_clean demo_clean 50 0 0 
bash finetune.sh pi0_base_piper real 0,1,2,3
#bash quick_eval.bash
#uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_base_torch_full --exp_name pytorch_handover_block