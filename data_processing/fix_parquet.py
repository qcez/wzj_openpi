#!/usr/bin/env python3
"""
修复 Parquet 文件元数据中的 'List' 特征类型为 'Sequence'
用于解决 HuggingFace datasets 库不识别 'List' 类型的问题
"""

import pyarrow.parquet as pq
import json
import os
from pathlib import Path
from typing import Any, Dict
import sys


def fix_features_metadata(features: Dict[str, Any]) -> bool:
    """递归修复 features 字典中的 List 类型为 Sequence"""
    fixed = False
    
    if isinstance(features, dict):
        # 检查是否有 _type 字段且值为 'List'
        if '_type' in features and features['_type'] == 'List':
            features['_type'] = 'Sequence'
            fixed = True
        
        # 检查是否有 feature 字段
        if 'feature' in features:
            if isinstance(features['feature'], dict):
                if '_type' in features['feature'] and features['feature']['_type'] == 'List':
                    features['feature']['_type'] = 'Sequence'
                    fixed = True
                # 递归处理 feature 内部
                if fix_features_metadata(features['feature']):
                    fixed = True
        
        # 递归处理所有值
        for key, value in features.items():
            if isinstance(value, (dict, list)):
                if fix_features_metadata(value):
                    fixed = True
    
    elif isinstance(features, list):
        for item in features:
            if isinstance(item, dict):
                if fix_features_metadata(item):
                    fixed = True
    
    return fixed


def fix_parquet_metadata(parquet_file_path: str) -> bool:
    """修复单个 parquet 文件的元数据"""
    try:
        # 读取 parquet 文件
        table = pq.read_table(parquet_file_path)
        
        # 获取当前元数据
        metadata = table.schema.metadata
        
        if metadata is None:
            return False
        
        # 检查是否有 huggingface 元数据
        if b'huggingface' not in metadata:
            return False
        
        # 解析 huggingface 元数据
        try:
            hf_metadata_str = metadata[b'huggingface'].decode('utf-8')
            hf_metadata = json.loads(hf_metadata_str)
        except Exception as e:
            print(f"  ✗ Error parsing metadata: {e}")
            return False
        
        # 修复 features 中的 List 类型
        fixed = False
        if 'info' in hf_metadata and 'features' in hf_metadata['info']:
            if fix_features_metadata(hf_metadata['info']['features']):
                fixed = True
        
        if not fixed:
            return False
        
        # 更新元数据
        new_metadata = dict(metadata)
        new_metadata[b'huggingface'] = json.dumps(hf_metadata).encode('utf-8')
        
        # 创建新的 schema
        new_schema = table.schema.with_metadata(new_metadata)

        # 写入到临时文件后再原子替换目标文件，避免因写入中断导致文件丢失
        tmp_path = parquet_file_path + ".tmp"
        pq.write_table(table, tmp_path, schema=new_schema)
        os.replace(tmp_path, parquet_file_path)

        return True
        
    except Exception as e:
        print(f"  ✗ Error processing file: {e}")
        return False


def fix_all_parquet_files(data_dir: str):
    """修复指定目录下所有 parquet 文件的元数据"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    parquet_files = sorted(list(data_path.glob("*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files in {data_dir}")
    print("=" * 70)
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, parquet_file in enumerate(parquet_files, 1):
        print(f"[{i}/{len(parquet_files)}] Processing {parquet_file.name}...", end=" ")
        
        try:
            if fix_parquet_metadata(str(parquet_file)):
                print("✓ Fixed")
                fixed_count += 1
            else:
                print("- No changes needed")
                skipped_count += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            error_count += 1
    
    print("=" * 70)
    print(f"Summary:")
    print(f"  Total files: {len(parquet_files)}")
    print(f"  Fixed: {fixed_count}")
    print(f"  No changes needed: {skipped_count}")
    print(f"  Errors: {error_count}")


if __name__ == "__main__":
    # 目标目录
    target_dir = "/project/peilab/wzj/.cache/huggingface/lerobot/robotwin_aloha_lerobot_repo/data/chunk-001"
    
    print("=" * 70)
    print("Parquet Metadata Fix Script")
    print("Fixing 'List' feature type to 'Sequence' in HuggingFace metadata")
    print("=" * 70)
    print(f"Target directory: {target_dir}")
    print()
    
    # 确认
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    print()
    fix_all_parquet_files(target_dir)
    print()
    print("Done!")