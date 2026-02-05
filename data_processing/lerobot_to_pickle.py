import pickle
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
from features import FEATURESV30
import torch
from collections import defaultdict
import gc

import pyarrow.parquet as pq
import pyarrow as pa

def clear_memory():
    """强制清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def lerobot_to_pickle(repo_id, output_path, chunk_size=50, home_lerobot=None, batch_size=32, num_workers=0, save_interval=10, max_episodes=10):
    """
    将lerobot数据集转换为pickle文件（优化版本 - 分批保存并彻底清理内存）
    
    参数:
        repo_id: lerobot数据集的repo_id
        output_path: 输出pickle文件的路径
        chunk_size: 动作序列的长度（默认为50，当前未使用）
        home_lerobot: lerobot数据集的本地路径（如果为None，则使用默认路径）
        batch_size: 批处理大小（默认32）
        num_workers: 并行加载数据的进程数（已废弃，现在直接遍历数据集）
        save_interval: 每处理多少个样本后检查并保存已完成的episode（默认10，实际间隔为save_interval*batch_size）
        max_episodes: 最多提取的episode数量（默认10，只提取前10个episode）
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 创建数据集
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(os.path.join(home_lerobot, repo_id)),
        revision="main",
        download_videos=False,
        force_cache_sync=False,
        video_backend="pyav"
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"只提取前 {max_episodes} 个episode")
    
    # 按 episode 组织数据
    episode_data = defaultdict(list)
    saved_episodes = set()  # 记录已保存的episode
    seen_episodes = set()    # 记录所有见过的episode
    episode_complete = set()  # 记录已完成的episode（已保存且不再有新数据）
    episode_part_count = defaultdict(int)  # 记录每个episode已保存的分块数
    processed_count = 0
    
    # 直接遍历数据集，不使用DataLoader以避免内存问题
    # 这样可以提前退出，一旦收集到前max_episodes个episode就停止
    
    def save_episode(episode_idx):
        """保存单个episode的剩余数据到分块文件（不合并，直接使用分块文件）
        
        参数:
            episode_idx: episode索引
        """
        # 如果内存中没有数据，返回False
        if episode_idx not in episode_data or len(episode_data[episode_idx]) == 0:
            return False
        
        # 直接保存内存中的数据到新的分块文件
        part_num = episode_part_count[episode_idx]
        episode_part_file = os.path.join(output_path, f'episode_{episode_idx}_part{part_num}.pkl')
        
        # 保存当前内存中的数据
        with open(episode_part_file, 'wb') as f:
            pickle.dump(episode_data[episode_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        episode_part_count[episode_idx] += 1
        
        # 清理内存
        for sample in episode_data[episode_idx]:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    del value
                elif isinstance(value, np.ndarray):
                    del value
        del episode_data[episode_idx]
        
        # 标记为已保存（所有数据都在分块文件中）
        saved_episodes.add(episode_idx)
        
        # 强制垃圾回收
        clear_memory()
        
        return True
    
    # 直接遍历数据集，逐个处理样本
    prev_episode_idx = None
    consecutive_skips = 0  # 连续跳过的样本数（用于判断是否可以提前退出）
    max_consecutive_skips = 1000  # 连续跳过这么多样本后，检查是否可以退出
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="处理数据"):
        # 获取单个样本
        sample = dataset[idx]
        episode_idx = sample['episode_index'].item()
        
        # 只处理前max_episodes个episode
        if episode_idx >= max_episodes:
            # 如果遇到超出范围的episode，先保存当前正在处理的episode（如果有）
            if prev_episode_idx is not None and prev_episode_idx < max_episodes:
                if save_episode(prev_episode_idx):
                    print(f"遇到超出范围的episode，保存当前 episode_{prev_episode_idx}.pkl")
            
            consecutive_skips += 1
            # 检查是否可以提前退出：如果前max_episodes个episode都已保存
            if len(saved_episodes) >= max_episodes:
                if all(ep_idx in saved_episodes for ep_idx in range(max_episodes)):
                    print(f"\n已收集到前 {max_episodes} 个episode的所有数据，提前退出")
                    break
            # 如果连续跳过很多样本，也检查一次（作为额外保险）
            elif consecutive_skips >= max_consecutive_skips:
                if len(saved_episodes) >= max_episodes:
                    if all(ep_idx in saved_episodes for ep_idx in range(max_episodes)):
                        print(f"\n已收集到前 {max_episodes} 个episode的所有数据，提前退出")
                        break
            del sample
            continue
        
        consecutive_skips = 0
        seen_episodes.add(episode_idx)
        
        # 如果遇到新的episode，立即保存上一个episode的剩余数据
        if prev_episode_idx is not None and episode_idx != prev_episode_idx:
            if save_episode(prev_episode_idx):
                part_count = episode_part_count[prev_episode_idx]
                print(f"episode_{prev_episode_idx} 完成，共 {part_count} 个分块文件 (共 {len(saved_episodes)} 个episode已保存)")
                episode_complete.add(prev_episode_idx)
        
        prev_episode_idx = episode_idx
        
        # 处理样本，直接保存 tensor（不转换为 numpy）
        # 优化：减少不必要的clone，只在必要时创建副本
        processed_sample = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    # 直接从GPU移到CPU并clone，避免中间变量
                    processed_sample[key] = value.detach().cpu().clone()
                else:
                    # 使用detach().clone()确保完全独立
                    processed_sample[key] = value.detach().clone()
            elif isinstance(value, np.ndarray):
                if value.ndim > 0:
                    # 创建numpy数组的副本，避免引用原始数据
                    processed_sample[key] = value.copy()
                else:
                    processed_sample[key] = value
            else:
                processed_sample[key] = value
        
        episode_data[episode_idx].append(processed_sample)
        processed_count += 1
        
        # 每处理10个样本就清理一次sample引用（更频繁的清理）
        if processed_count % 10 == 0:
            del sample
            clear_memory()
        
        # 更频繁地保存和清理内存：每处理一定数量的样本就检查
        # 1. 如果当前episode的样本数超过阈值，强制保存分块（防止单个episode太大）
        max_samples_per_episode = 500  # 使用更大的分块（500个样本），避免过多分块文件
        if len(episode_data[episode_idx]) >= max_samples_per_episode:
            # 使用分块保存，避免加载大文件
            part_num = episode_part_count[episode_idx]
            episode_part_file = os.path.join(output_path, f'episode_{episode_idx}_part{part_num}.pkl')
            
            # 直接保存，不创建副本（pickle会处理序列化）
            # 注意：保存后立即清理，避免内存累积
            with open(episode_part_file, 'wb') as f:
                pickle.dump(episode_data[episode_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
            
            episode_part_count[episode_idx] += 1
            
            # 立即清理已保存的数据（在del之前）
            for sample in episode_data[episode_idx]:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        del value
                    elif isinstance(value, np.ndarray):
                        del value
            episode_data[episode_idx].clear()  # 清空列表但保留列表对象
            
            print(f"episode_{episode_idx} 样本数达到 {max_samples_per_episode}，保存分块 {part_num}")
            # 立即清理内存
            clear_memory()
        
        # 2. 定期保存已完成的episode（更频繁的检查）
        if processed_count % 30 == 0:  # 每30个样本检查一次，更频繁
            # 检查并保存已完成的episode（当前episode之前的所有episode）
            episodes_to_save = []
            for ep_idx in list(episode_data.keys()):
                if ep_idx >= max_episodes:
                    continue
                if ep_idx < episode_idx and ep_idx not in saved_episodes:
                    episodes_to_save.append(ep_idx)
            
            # 保存这些已完成的episode
            for ep_idx in sorted(episodes_to_save):
                if save_episode(ep_idx):
                    part_count = episode_part_count[ep_idx]
                    print(f"定期保存 episode_{ep_idx} 的剩余数据（共 {part_count} 个分块，{len(saved_episodes)} 个episode已保存）")
                    episode_complete.add(ep_idx)
            
            # 显示当前内存中的episode数量和样本数
            if len(episode_data) > 0:
                total_samples_in_memory = sum(len(samples) for samples in episode_data.values())
                print(f"当前内存中有 {len(episode_data)} 个episode的数据（共 {total_samples_in_memory} 个样本），已保存 {len(saved_episodes)} 个episode")
            
            # 定期清理内存
            clear_memory()
        
        # 清理当前样本的引用（如果还没清理）
        if processed_count % 10 != 0:  # 如果上面已经清理过，就不重复
            del sample
    
    # 保存最后一个episode（如果还没有保存，且在前max_episodes个范围内）
    if prev_episode_idx is not None and prev_episode_idx < max_episodes:
        if prev_episode_idx not in saved_episodes:
            if save_episode(prev_episode_idx):
                part_count = episode_part_count[prev_episode_idx]
                print(f"已保存最后一个 episode_{prev_episode_idx} 的剩余数据（共 {part_count} 个分块）")
    
    # 保存所有剩余的episode（确保没有遗漏，只保存前max_episodes个）
    if len(episode_data) > 0:
        remaining_episodes = [ep_idx for ep_idx in episode_data.keys() if ep_idx < max_episodes and ep_idx not in saved_episodes]
        if remaining_episodes:
            print(f"\n保存剩余的 {len(remaining_episodes)} 个episode...")
            for episode_idx in sorted(remaining_episodes):
                if save_episode(episode_idx):
                    part_count = episode_part_count[episode_idx]
                    print(f"已保存 episode_{episode_idx} 的剩余数据（共 {part_count} 个分块）")
    
    # 最终清理：删除所有不在前max_episodes范围内的episode数据
    episodes_to_remove = [ep_idx for ep_idx in episode_data.keys() if ep_idx >= max_episodes]
    for ep_idx in episodes_to_remove:
        del episode_data[ep_idx]
    del episode_data
    clear_memory()
    
    print(f"\n数据已保存到: {output_path}")
    print(f"总共保存了 {len(saved_episodes)} 个 episode (前{max_episodes}个)")
    print(f"总共处理了 {processed_count} 个样本")
    
    # 统计每个episode的分块数量
    print(f"\n各episode分块统计:")
    total_parts = 0
    for ep_idx in sorted(saved_episodes):
        part_count = episode_part_count[ep_idx]
        total_parts += part_count
        print(f"  episode_{ep_idx}: {part_count} 个分块文件")
    
    # 验证保存的文件（分块文件）
    saved_files = [f for f in os.listdir(output_path) if f.endswith('.pkl') and '_part' in f]
    print(f"\n实际保存的分块文件数: {len(saved_files)}")
    print(f"提示: 数据以分块形式保存（episode_X_partN.pkl），每个分块包含最多 {max_samples_per_episode} 个样本")

# 使用示例
if __name__ == "__main__":
    lerobot_to_pickle(
        "xvla-folding", 
        "tmp_data", 
        home_lerobot="/project/peilab/yanzhengyang/RoboTwin/data",
        batch_size=64,       # 仅用于计算保存间隔，不再使用DataLoader
        num_workers=0,       # 已废弃，不再使用
        save_interval=10     # 每处理save_interval*batch_size个样本后检查一次
    )