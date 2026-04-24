#!/usr/bin/env python3
"""比较两个结果目录中的样本差异"""
import os
from pathlib import Path


def get_sample_ids_from_json_dir(dir_path):
    """从包含JSON文件的目录中提取样本ID"""
    sample_ids = set()
    for file in os.listdir(dir_path):
        if file.endswith('.json'):
            # 提取数字ID，例如 "123.json" -> "123"
            sample_id = file.replace('.json', '')
            sample_ids.add(sample_id)
    return sample_ids


def get_sample_ids_from_subdir(dir_path):
    """从包含子目录的目录中提取样本ID"""
    sample_ids = set()
    for subdir in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, subdir)):
            # 提取数字ID，例如 "val_123" -> "123"
            if subdir.startswith('val_'):
                sample_id = subdir.replace('val_', '')
                sample_ids.add(sample_id)
    return sample_ids


def compare_directories(dir1, dir2):
    """比较两个目录的样本差异"""
    print(f"正在比较目录:")
    print(f"  目录1: {dir1}")
    print(f"  目录2: {dir2}")
    print("=" * 80)
    
    # 获取两个目录中的样本ID
    print("\n正在提取样本ID...")
    samples1 = get_sample_ids_from_json_dir(dir1)
    samples2 = get_sample_ids_from_subdir(dir2)
    
    print(f"目录1 (final_results) 中的样本数: {len(samples1)}")
    print(f"目录2 (final_visual_results) 中的样本数: {len(samples2)}")
    
    # 找出差异
    only_in_dir1 = samples1 - samples2
    only_in_dir2 = samples2 - samples1
    common = samples1 & samples2
    
    print("\n" + "=" * 80)
    print(f"共同样本数: {len(common)}")
    print(f"仅在目录1中的样本数: {len(only_in_dir1)}")
    print(f"仅在目录2中的样本数: {len(only_in_dir2)}")
    print("=" * 80)
    
    if only_in_dir1:
        print(f"\n仅在目录1 (final_results) 中存在的样本 (共 {len(only_in_dir1)} 个):")
        sorted_ids = sorted([int(x) for x in only_in_dir1])
        print(sorted_ids)
    
    if only_in_dir2:
        print(f"\n仅在目录2 (final_visual_results) 中存在的样本 (共 {len(only_in_dir2)} 个):")
        sorted_ids = sorted([int(x) for x in only_in_dir2])
        print(sorted_ids)
    
    return {
        'total_dir1': len(samples1),
        'total_dir2': len(samples2),
        'common': len(common),
        'only_in_dir1': sorted([int(x) for x in only_in_dir1]) if only_in_dir1 else [],
        'only_in_dir2': sorted([int(x) for x in only_in_dir2]) if only_in_dir2 else []
    }


if __name__ == "__main__":
    dir1 = "<DIR1>"
    dir2 = "<DIR2>"
    
    result = compare_directories(dir1, dir2)


