#!/bin/bash

# Wrapper script for multi_gpu_eval_rollout.py, compatible with eval_distributed.sh and eval_replay.sh styles
# Usage: ./multi_gpu_eval_rollout_wrapper.sh <task_names> <task_config> <train_config_name> <model_name> <seed> <gpu_ids> <gpu_num> <batch_size_per_gpu> <file_name> <seed_base> <use_saved_instructions> <saved_instructions_path> <generate_rollout> <env_seeds>

policy_name=openpi_test

# Parameters (matching eval_distributed.sh)
task_names=${1}          # Task names as space-separated string, e.g., "task1 task2"
task_config=${2}         # Task config name, e.g., "demo_clean"
train_config_name=${3}   # Train config name, e.g., "pi0_base_aloha_robotwin_full_torch"
model_name=${4}          # Model name, e.g., "robotwin_aloha_lerobot"
seed=${5}                # Random seed, e.g., 0
gpu_ids=${6}             # GPU IDs as comma-separated, e.g., "0,1,2,3"
gpu_num=${7}             # Number of GPUs, e.g., 4
batch_size_per_gpu=${8}  # Batch size per GPU, e.g., 16
file_name=${9:-rollout.pkl}  # Output file name, default "rollout.pkl"
seed_base=${10}          # Seed base, e.g., 100000
use_saved_instructions=${11}  # Use saved instructions
saved_instructions_path=${12}    # Path to saved instructions
generate_rollout=${13}         # Generate rollout
env_seeds=${14}                # Environment seeds, comma-separated

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${gpu_ids}

echo -e "\033[33mGPU IDs (to use): ${gpu_ids}\033[0m"
echo "Policy: ${policy_name}"

source .venv/bin/activate
cd ../..  # Move to root

PYTHONWARNINGS=ignore::UserWarning \
uv run script/multi_gpu_eval_rollout.py \
    --config policy/${policy_name}/deploy_policy.yml \
    --task_names ${task_names} \
    --gpu_num ${gpu_num} \
    --batch_size_per_gpu ${batch_size_per_gpu} \
    --file_name ${file_name} \
    --seed_base ${seed_base} \
    --env_seed ${env_seeds} \
    --use_saved_instructions ${use_saved_instructions} \
    --saved_instructions_path "${saved_instructions_path}" \
    --generate_rollout ${generate_rollout} \
    --overrides \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name}