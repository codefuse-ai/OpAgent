#!/usr/bin/env python3
"""找出两个目录中分数不一致的样本"""
import os
import json
from pathlib import Path


def get_scores_from_final_results(dir_path):
    """从final_results目录中获取所有样本的分数"""
    scores = {}
    for file in os.listdir(dir_path):
        if file.endswith('.json'):
            sample_id = file.replace('.json', '')
            with open(os.path.join(dir_path, file), 'r') as f:
                data = json.load(f)
                score = data[-1]['score']
                scores[sample_id] = score
    return scores


def get_scores_from_final_visual_results(dir_path):
    """从final_visual_results目录中获取所有样本的分数"""
    scores = {}
    for subdir in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, subdir)) and subdir.startswith('val_'):
            sample_id = subdir.replace('val_', '')
            trajectory_file = os.path.join(dir_path, subdir, 'trajectory.json')
            if os.path.exists(trajectory_file):
                with open(trajectory_file, 'r') as f:
                    data = json.load(f)
                    score = None
                    for item in data:
                        if isinstance(item, dict) and item.get("type") == "evaluation":
                            score = item.get("score", 0.0)
                            break
                    if score is not None:
                        scores[sample_id] = score
    return scores


def compare_scores(dir1, dir2):
    """比较两个目录中的分数"""
    print("正在提取分数...")
    scores1 = get_scores_from_final_results(dir1)
    scores2 = get_scores_from_final_visual_results(dir2)
    
    print(f"目录1 (final_results) 样本数: {len(scores1)}")
    print(f"目录2 (final_visual_results) 样本数: {len(scores2)}")
    
    # 统计总分
    total_score1 = sum(scores1.values())
    total_score2 = sum(scores2.values())
    
    print(f"\n目录1总分: {total_score1}")
    print(f"目录2总分: {total_score2}")
    print(f"总分差异: {abs(total_score1 - total_score2)}")
    
    # 统计成功样本数（score > 0）
    success_count1 = sum(1 for s in scores1.values() if s > 0)
    success_count2 = sum(1 for s in scores2.values() if s > 0)
    
    print(f"\n目录1成功样本数 (score > 0): {success_count1}")
    print(f"目录2成功样本数 (score > 0): {success_count2}")
    
    # 找出分数不一致的样本
    common_ids = set(scores1.keys()) & set(scores2.keys())
    different_scores = []
    
    for sample_id in common_ids:
        if scores1[sample_id] != scores2[sample_id]:
            different_scores.append({
                'sample_id': sample_id,
                'score_in_dir1': scores1[sample_id],
                'score_in_dir2': scores2[sample_id],
                'difference': scores1[sample_id] - scores2[sample_id]
            })
    
    if different_scores:
        print(f"\n发现 {len(different_scores)} 个样本的分数不一致:")
        print("=" * 80)
        for item in sorted(different_scores, key=lambda x: int(x['sample_id'])):
            print(f"样本 {item['sample_id']}:")
            print(f"  final_results 中的分数: {item['score_in_dir1']}")
            print(f"  final_visual_results 中的分数: {item['score_in_dir2']}")
            print(f"  差异: {item['difference']}")
            print("-" * 80)
    else:
        print("\n所有共同样本的分数都一致！")
    
    # 检查非0非1的分数
    print("\n检查非0/1的分数:")
    non_binary_scores1 = {k: v for k, v in scores1.items() if v not in [0.0, 1.0]}
    non_binary_scores2 = {k: v for k, v in scores2.items() if v not in [0.0, 1.0]}
    
    if non_binary_scores1:
        print(f"\n目录1中有 {len(non_binary_scores1)} 个非0/1分数:")
        for sample_id, score in sorted(non_binary_scores1.items(), key=lambda x: int(x[0])):
            print(f"  样本 {sample_id}: {score}")
    
    if non_binary_scores2:
        print(f"\n目录2中有 {len(non_binary_scores2)} 个非0/1分数:")
        for sample_id, score in sorted(non_binary_scores2.items(), key=lambda x: int(x[0])):
            print(f"  样本 {sample_id}: {score}")


if __name__ == "__main__":
    dir1 = "<DIR1>"
    dir2 = "<DIR2>"
    
    compare_scores(dir1, dir2)


