import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def calculate_accuracy(results_dir: str) -> Dict:
    """
    统计指定目录下所有样本的准确率
    
    Args:
        results_dir: 结果目录路径,例如 '/path/to/final_visual_results'
    
    Returns:
        包含统计信息的字典,包括:
        - total_samples: 总样本数
        - successful_samples: 成功样本数(score > 0)
        - accuracy: 准确率
        - average_score: 平均分数
        - score_distribution: 分数分布
        - failed_samples: 失败样本列表
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"目录不存在: {results_dir}")
    
    total_samples = 0
    successful_samples = 0
    total_score = 0.0
    scores = []
    failed_samples = []
    sample_details = []
    
    # 遍历所有子目录
    for sample_dir in sorted(results_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
            
        trajectory_file = sample_dir / "trajectory.json"
        
        # 检查trajectory.json是否存在
        if not trajectory_file.exists():
            print(f"警告: {sample_dir.name} 下没有找到 trajectory.json")
            continue
        
        try:
            # 读取trajectory.json
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
            
            # 查找评估结果
            score = None
            for item in trajectory_data:
                if isinstance(item, dict) and item.get("type") == "evaluation":
                    score = item.get("score", 0.0)
                    break
            
            if score is None:
                print(f"警告: {sample_dir.name} 中没有找到评估分数")
                continue
            
            # 统计
            total_samples += 1
            total_score += score
            scores.append(score)
            
            sample_info = {
                "sample_name": sample_dir.name,
                "score": score
            }
            sample_details.append(sample_info)
            
            if score > 0:
                successful_samples += 1
            else:
                failed_samples.append(sample_dir.name)
                
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析 {trajectory_file}: {e}")
            continue
        except Exception as e:
            print(f"错误: 处理 {sample_dir.name} 时出错: {e}")
            continue
    
    # 计算统计信息
    accuracy = (successful_samples / total_samples * 100) if total_samples > 0 else 0.0
    average_score = (total_score / total_samples) if total_samples > 0 else 0.0
    
    # 计算分数分布
    score_distribution = {}
    for score in scores:
        score_key = f"{score:.1f}"
        score_distribution[score_key] = score_distribution.get(score_key, 0) + 1
    
    results = {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples_count": len(failed_samples),
        "accuracy": round(accuracy, 2),
        "average_score": round(average_score, 4),
        "score_distribution": dict(sorted(score_distribution.items())),
        "failed_samples": failed_samples[:10],  # 只显示前10个失败样本
        "sample_details": sample_details[:5]  # 只显示前5个样本详情作为示例
    }
    
    return results


def print_accuracy_report(results_dir: str):
    """
    打印格式化的准确率报告
    
    Args:
        results_dir: 结果目录路径
    """
    results = calculate_accuracy(results_dir)
    
    print("=" * 60)
    print("样本准确率统计报告")
    print("=" * 60)
    print(f"结果目录: {results_dir}")
    print("-" * 60)
    print(f"总样本数: {results['total_samples']}")
    print(f"成功样本数: {results['successful_samples']}")
    print(f"失败样本数: {results['failed_samples_count']}")
    print(f"准确率: {results['accuracy']}%")
    print(f"平均分数: {results['average_score']}")
    print("-" * 60)
    print("分数分布:")
    for score, count in results['score_distribution'].items():
        percentage = (count / results['total_samples'] * 100)
        print(f"  分数 {score}: {count} 个样本 ({percentage:.2f}%)")
    print("-" * 60)
    
    if results['failed_samples']:
        print(f"失败样本示例(显示前10个):")
        for sample in results['failed_samples']:
            print(f"  - {sample}")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # 示例用法
    results_dir = "./final_visual_results"
    
    try:
        results = print_accuracy_report(results_dir)
        
        # 也可以保存结果到文件
        output_file = Path(results_dir).parent / "accuracy_report.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: {e}")
    import os
    import json
    res_path = './final_results'
    res_jsons = os.listdir(res_path)
    score = 0
    for res in res_jsons:
        with open(os.path.join(res_path, res), 'r') as f:
            data = json.load(f)
        score+=data[-1]['score']
    print(f'{score}:{score/812}')

