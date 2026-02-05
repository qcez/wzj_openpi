
#!/bin/bash
policy_name=openpi_test
# task_name=${1}
task_names=${1}  # 拼接所有传入参数为任务列表字符串，如 "adjust_bottle beat_block_hammer"
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
gpu_num=${7}
batch_size_per_gpu=${8}
file_name=${9:-rollout.pkl}
shift 9  # 移除前7个参数
env_seed="$*"  # 剩余参数作为 env_seed

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

# PYTHONWARNINGS=ignore::UserWarning \
# uv run script/distributed_loop_collect_fm.py --config policy/$policy_name/deploy_policy.yml \
#     --overrides \
#     --task_name ${task_name} \
#     --task_config ${task_config} \
#     --train_config_name ${train_config_name} \
#     --model_name ${model_name} \
#     --ckpt_setting ${model_name} \
#     --seed ${seed} \
#     --policy_name ${policy_name} 

PYTHONWARNINGS=ignore::UserWarning \
uv run script/distributed_loop_collect_fm_wzj.py --config policy/$policy_name/deploy_policy.yml \
    --file_name ${file_name} \
    --task_names ${task_names} \
    --env_seed ${env_seed} \
    --gpu_num ${gpu_num} \
    --batch_size_per_gpu ${batch_size_per_gpu} \
    --overrides \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name}

#bash eval.sh beat_block_hammer demo_clean pi0_base_torch_full pytorch_beat_block_hammer 0 0
#bash eval.sh handover_block demo_clean pi0_base_aloha_robotwin_lora demo_clean 0 0
#bash eval.sh beat_block_hammer demo_clean pi0_base_torch_lora beat_hammer 0 0
#bash eval.sh beat_block_hammer demo_clean pi0_torch_from_jax beat_block_hammer 0 0
#bash eval_distributed.sh beat_block_hammer demo_clean pytorch_beat_block_hammer demo_clean 0 0


#!/bin/bash
policy_name=openpi_test
task_names="$1"  # 修复：只取第一个参数
task_config="$2"
train_config_name="$3"
model_name="$4"
seed="$5"
gpu_id="$6"
file_name="${7:-rollout.pkl}"

