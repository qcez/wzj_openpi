eval "$(/root/anaconda3/bin/conda shell.bash hook)"

#git clone RoboTwin
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
#download dataset
# https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset
wget https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/dataset/adjust_bottle/aloha-agilex_clean_50.zip

#容器>镜像
docker commit robotwin-container XXX/robotwin-env-pi0:12_15
#镜像>dockerhub
docker push XXX/robotwin-env-pi0:12_15
#dockerhub>镜像

docker pull XXX/robotwin-env-pi0:12_15
#创建容器，挂载gpu，本地文件

docker run --name robotwin-container --gpus all -it -v /root/RoboTwin:/workspace/RoboTwin XXX/robotwin-env-pi0:12_15 /bin/bash
#启动已存在容器
docker start robotwin-container
#进入已存在容器
docker exec -it robotwin-container /bin/bash
#关闭容器 -t x x秒后kill(默认10s)
docker stop robotwin-container
#强制关闭容器
docker kill robotwin-container
#容器列表
docker ps -a

#镜像加标签
docker tag robotwin-env:latest your-dockerhub-username/robotwin-env:latest
#容器改名
docker rename <old_name> <new_name>

# pi0
conda activate RoboTwin
# Install uv
pip install uv

cd policy/pi0
# Install prequisites in uv environment
GIT_LFS_SKIP_SMUDGE=1 uv sync

mkdir processed_data && mkdir training_data
#bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num} 数据>HDF5
bash process_data_pi0.sh adjust_bottle demo_clean 50

cp processed_data/ training_data/

#bash generate.sh ${hdf5_path} ${repo_id} generate the LerobotDataset format data for pi0
bash generate.sh training_data/demo_clean/adjust_bottle-demo_clean-50 demo_clean_repo

#training
#改src/openpi/training/config.py 
# fsdp_devices gpu数
# batch_size  
# pi0_base_aloha_robotwin_lora / pi0_fast_aloha_robotwin_lora / pi0_base_aloha_robotwin_full / pi0_fast_aloha_robotwin_full

# compute norm_stat for dataset
uv run scripts/compute_norm_stats.py --config-name pi0_fast_aloha_robotwin_lora
#bash finetune.sh ${train_config_name} ${model_name}(training_data/下一级目录) ${gpu_use} 
bash finetune.sh pi0_fast_aloha_robotwin_lora adjust_bottle-demo_clean-50 0
# gpu少 XLA_PYTHON_CLIENT_PREALLOCATE=false

#fast lora
uv run scripts/compute_norm_stats.py --config-name pi0_fast_aloha_robotwin_lora
bash finetune.sh pi0_fast_aloha_robotwin_lora adjust_bottle-demo_clean-50 0

#base lora
uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_lora
bash finetune.sh pi0_base_aloha_robotwin_lora adjust_bottle-demo_clean-50 0

bash eval_pi05.sh beat_block_hammer demo_clean pi05_base_aloha_robotwin_lora beat_hammer_pi05 0 0

#Superpod
#查询时间
sacct -j <ID> --format=JobID,Submit,Eligible,Start,End
#预计开始时间
squeue --start -j <ID>
#详细信息
scontrol show job <ID>
#显示在跑进程的gpu占用
srun --jobid <ID> --overlap nvidia-smi

export XDG_CACHE_HOME=

export HF_TOKEN=""
export GITHUB_TOKEN=""

eval "$(/home/xxx/anaconda3/bin/conda shell.bash hook)"
conda activate RoboTwin-pi0-1
