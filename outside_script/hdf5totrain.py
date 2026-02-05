import os
import shutil
import json
import re
from pathlib import Path


# 源目录和目标目录
source_dir = "/project/peilab/wzj/folding_clothes_1_14"
target_dir = "/project/peilab/wzj/RoboTwin/policy/openpi_test/training_data/fold_cloth/fold_cloth-1_14-200"

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

# 获取所有 hdf5 文件
hdf5_files = [f for f in os.listdir(source_dir) if f.endswith('.hdf5')]

# 为每个 hdf5 文件创建对应的 episode 目录
for hdf5_file in hdf5_files:
    # 从文件名中提取数字
    match = re.search(r'episode_(\d+)\.hdf5', hdf5_file)
    if match:
        episode_num = match.group(1)
        
        # 创建 episode 目录
        episode_dir = os.path.join(target_dir, f"episode_{episode_num}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # 复制 hdf5 文件
        source_path = os.path.join(source_dir, hdf5_file)
        target_hdf5_path = os.path.join(episode_dir, f"episode_{episode_num}.hdf5")
        shutil.copy2(source_path, target_hdf5_path)
        
        # 创建 instructions.json 文件
        instructions_data = {
            "instructions": ["fold the cloth"]
        }
        instructions_path = os.path.join(episode_dir, "instructions.json")
        with open(instructions_path, 'w', encoding='utf-8') as f:
            json.dump(instructions_data, f, indent=3)
        
        print(f"已处理: {hdf5_file} -> episode_{episode_num}/")
    else:
        print(f"警告: 无法解析文件名 {hdf5_file}")

print(f"\n完成！")
print(f"目标目录: {target_dir}")