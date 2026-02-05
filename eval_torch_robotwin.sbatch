#!/bin/bash

#SBATCH --job-name=wzj_etorch
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --account=peilab
#SBATCH --output=eval_torch_robotwin.out

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
# export XDG_CACHE_HOME=/scratch/peilab/wzj/.cache

bash eval_50_tasks.sh

