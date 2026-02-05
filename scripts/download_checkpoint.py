from huggingface_hub import snapshot_download
import os

os.environ['HF_ENDPOINT']='https://hf-mirror.com'           # 中国加速（可选）
os.environ['HF_XET_HIGH_PERFORMANCE']='1'                   # 高性能模式（更快，但进度条仍正常）

# 模型 ID
# repo_id = "qcez/beat-block-hammer-jax-lora-checkpoint-30000"

# # 本地保存路径（可以自定义）
# local_dir = "/project/peilab/wzj/RoboTwin/policy/openpi_test/checkpoints/pi0_base_aloha_robotwin_lora/beat_block_hammer_jax/30000"

# print(f"正在开始下载模型 {repo_id}...")

# try:
#     snapshot_download(
#         repo_id=repo_id,
#         local_dir=local_dir,
#         local_dir_use_symlinks=False, # 直接下载到文件夹，不使用符号链接
#         resume_download=True,         # 支持断点续传
#         # endpoint="https://hf-mirror.com" # 如果在国内网络环境下载慢，请取消此行的注释
#     )
#     print(f"下载完成！模型已保存至: {local_dir}")
# except Exception as e:
#     print(f"下载过程中出错: {e}")


# 下载整个仓库到指定文件夹（推荐大文件夹）
snapshot_download(
    repo_id="qcez/folding_clothes_1_14",
    repo_type="dataset",
    local_dir="/project/peilab/wzj/folding_clothes_1_14",
    local_dir_use_symlinks=False,   # 避免符号链接问题
    resume_download=True,
    # revision="main"               # 默认就是 main，可省略
)

print("下载完成!")