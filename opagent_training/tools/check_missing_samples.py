#!/usr/bin/env python3
"""检查缺失的样本ID"""
import os
from pathlib import Path


def get_sample_ids_from_json_dir(dir_path):
    """从包含JSON文件的目录中提取样本ID"""
    sample_ids = set()
    for file in os.listdir(dir_path):
        if file.endswith('.json'):
            sample_id = int(file.replace('.json', ''))
            sample_ids.add(sample_id)
    return sample_ids


def get_sample_ids_from_subdir(dir_path):
    """从包含子目录的目录中提取样本ID"""
    sample_ids = set()
    for subdir in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, subdir)):
            if subdir.startswith('val_'):
                sample_id = int(subdir.replace('val_', ''))
                sample_ids.add(sample_id)
    return sample_ids


def check_missing_samples(dir1, dir2, expected_total=812):
    """检查缺失的样本"""
    print(f"检查缺失的样本 (期望总数: {expected_total})")
    print("=" * 80)
    
    # 获取样本ID
    samples1 = get_sample_ids_from_json_dir(dir1)
    samples2 = get_sample_ids_from_subdir(dir2)
    
    print(f"目录1中的样本数: {len(samples1)}")
    print(f"目录2中的样本数: {len(samples2)}")
    
    # 找出应该存在的所有样本ID（0 到 expected_total-1）
    expected_samples = set(range(expected_total))
    
    missing_from_dir1 = expected_samples - samples1
    missing_from_dir2 = expected_samples - samples2
    
    print("\n" + "=" * 80)
    print(f"目录1中缺失的样本ID (共 {len(missing_from_dir1)} 个):")
    if missing_from_dir1:
        print(sorted(missing_from_dir1))
    
    print(f"\n目录2中缺失的样本ID (共 {len(missing_from_dir2)} 个):")
    if missing_from_dir2:
        print(sorted(missing_from_dir2))
    
    # 找出在两个目录中都缺失的样本
    missing_in_both = missing_from_dir1 & missing_from_dir2
    print(f"\n在两个目录中都缺失的样本ID (共 {len(missing_in_both)} 个):")
    if missing_in_both:
        print(sorted(missing_in_both))
    
    # 找出只在其中一个目录中缺失的样本
    only_missing_in_dir1 = missing_from_dir1 - missing_from_dir2
    only_missing_in_dir2 = missing_from_dir2 - missing_from_dir1
    
    if only_missing_in_dir1:
        print(f"\n仅在目录1中缺失的样本ID (共 {len(only_missing_in_dir1)} 个):")
        print(sorted(only_missing_in_dir1))
    
    if only_missing_in_dir2:
        print(f"\n仅在目录2中缺失的样本ID (共 {len(only_missing_in_dir2)} 个):")
        print(sorted(only_missing_in_dir2))
    
    print("=" * 80)


if __name__ == "__main__":
    dir1 = "<RESULTS_JSON_DIR>"
    dir2 = "<RESULTS_VISUAL_DIR>"
    
    check_missing_samples(dir1, dir2, expected_total=812)


