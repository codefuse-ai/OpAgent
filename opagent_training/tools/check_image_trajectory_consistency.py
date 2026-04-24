#!/usr/bin/env python3
"""检查每个样本的图像数量和trajectory中的轨迹数量是否一致"""
import os
import json
from pathlib import Path
from collections import defaultdict


def check_consistency(results_dir):
    """检查图像数量和轨迹数量的一致性"""
    results_dir = Path(results_dir)
    
    inconsistent_samples = []
    consistent_count = 0
    total_samples = 0
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    
    print("正在检查每个样本的图像和轨迹一致性...")
    print("=" * 100)
    
    # 遍历所有样本目录
    for sample_dir in sorted(results_dir.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith('val_'):
            continue
        
        total_samples += 1
        sample_id = sample_dir.name.replace('val_', '')
        
        # 统计图像数量（检查images子目录）
        image_files = []
        images_dir = sample_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            for file in images_dir.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    image_files.append(file.name)
        image_count = len(image_files)
        
        # 读取trajectory.json
        trajectory_file = sample_dir / "trajectory.json"
        trajectory_count = 0
        trajectory_items = []
        
        if trajectory_file.exists():
            try:
                with open(trajectory_file, 'r', encoding='utf-8') as f:
                    trajectory_data = json.load(f)
                
                # 统计trajectory中的observation条目数量（每个observation对应一张截图）
                for item in trajectory_data:
                    if isinstance(item, dict):
                        if item.get("type") == "observation":
                            trajectory_count += 1
                            trajectory_items.append(item.get("type", "unknown"))
                
            except Exception as e:
                print(f"错误: 读取 {trajectory_file} 失败: {e}")
                continue
        else:
            print(f"警告: {sample_dir.name} 没有 trajectory.json")
            continue
        
        # 检查是否一致
        if image_count != trajectory_count:
            inconsistent_samples.append({
                'sample_id': sample_id,
                'sample_name': sample_dir.name,
                'image_count': image_count,
                'trajectory_count': trajectory_count,
                'difference': image_count - trajectory_count,
                'image_files': sorted(image_files)[:5],  # 只显示前5个
                'trajectory_types': trajectory_items[:5]  # 只显示前5个
            })
        else:
            consistent_count += 1
    
    # 输出结果
    print(f"\n总样本数: {total_samples}")
    print(f"一致的样本数: {consistent_count}")
    print(f"不一致的样本数: {len(inconsistent_samples)}")
    print(f"一致率: {consistent_count/total_samples*100:.2f}%")
    
    if inconsistent_samples:
        print("\n" + "=" * 100)
        print(f"发现 {len(inconsistent_samples)} 个样本的图像数量和轨迹数量不一致:")
        print("=" * 100)
        
        # 统计差异分布
        difference_distribution = defaultdict(int)
        for item in inconsistent_samples:
            difference_distribution[item['difference']] += 1
        
        print("\n差异分布 (图像数 - 轨迹数):")
        for diff in sorted(difference_distribution.keys()):
            count = difference_distribution[diff]
            print(f"  差异 {diff:+3d}: {count} 个样本")
        
        print("\n" + "-" * 100)
        print("详细信息 (显示前20个不一致的样本):")
        print("-" * 100)
        
        for i, item in enumerate(inconsistent_samples[:20], 1):
            print(f"\n{i}. 样本 {item['sample_name']} (ID: {item['sample_id']}):")
            print(f"   图像数量: {item['image_count']}")
            print(f"   轨迹数量: {item['trajectory_count']}")
            print(f"   差异: {item['difference']:+d}")
            if item['image_count'] > 0:
                print(f"   图像示例: {', '.join(item['image_files'])}")
            if item['trajectory_types']:
                print(f"   轨迹类型: {', '.join(item['trajectory_types'])}")
        
        if len(inconsistent_samples) > 20:
            print(f"\n... 还有 {len(inconsistent_samples) - 20} 个不一致的样本未显示")
    else:
        print("\n✅ 所有样本的图像数量和轨迹数量都一致！")
    
    print("\n" + "=" * 100)
    
    return {
        'total_samples': total_samples,
        'consistent_count': consistent_count,
        'inconsistent_count': len(inconsistent_samples),
        'inconsistent_samples': inconsistent_samples
    }


if __name__ == "__main__":
    results_dir = "<RESULTS_DIR>"
    
    result = check_consistency(results_dir)
    
    # 保存详细结果
    output_file = Path(results_dir).parent / "image_trajectory_consistency_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {output_file}")

