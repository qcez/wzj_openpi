#!/bin/bash

policy_name=openpi_test
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo ${policy_name}
source .venv/bin/activate
cd ../.. # move to root
PYTHONWARNINGS=ignore::UserWarning \
# uv run script/eval_policy_replay.py --config policy/$policy_name/deploy_policy_pi05.yml \
#     --use-saved-instructions  \
#     --saved-instructions-path /project/peilab/wzj/RoboTwin/eval_result/beat_block_hammer/openpi_test/demo_clean/beat_block_hammer_jax/2026-01-11_18:32:25/test_instructions.json \
#     --overrides \
#     --task_name ${task_name} \
#     --task_config ${task_config} \
#     --train_config_name ${train_config_name} \
#     --model_name ${model_name} \
#     --ckpt_setting ${model_name} \
#     --seed ${seed} \
#     --policy_name ${policy_name} 

uv run script/eval_policy.py --config policy/$policy_name/deploy_policy_pi05.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 



#bash eval_distributed.sh beat_block_hammer demo_clean pi0_base_torch_full pytorch_beat_block_hammer 0 0
#bash eval.sh handover_block demo_clean pi0_base_aloha_robotwin_lora demo_clean 0 0
#bash eval.sh beat_block_hammer demo_clean pi0_base_torch_lora beat_hammer 0 0
#bash eval.sh beat_block_hammer demo_clean pi0_torch_from_jax beat_block_hammer 0 0