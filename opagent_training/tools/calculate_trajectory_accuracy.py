#!/usr/bin/env python3
"""
统计trajectory_data目录下样本的得分
规则：
1. 每个样本（由intent和start_url共同标识，忽略IP差异）可能测试多次
2. 只要有一次final_score为1，该样本就算对
3. 只有final_score等于1的才算对
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import argparse
from urllib.parse import urlparse
import random
import numpy as np

SITE_PORT_MAP = {
    "7770": "shopping",
    "7780": "shopping_admin",
    "9999": "reddit",
    "8023": "gitlab",
    "8888": "wikipedia",
    "3000": "map",
}
def calculate_pass_at_k(sample_results, k, min_samples=None, num_iterations=1000, debug=False):
    """
    计算 pass@k 指标
    
    参数:
    - sample_results: dict, key为(intent, normalized_url), value为结果列表
    - k: int, 采样数量
    - min_samples: int, 最小样本数要求（默认为k）
    - num_iterations: int, 采样迭代次数(默认1000)
    - debug: bool, 是否输出调试信息
    
    返回:
    - float: pass@k 的值
    - int: 有效样本数
    """
    if min_samples is None:
        min_samples = k
    
    from math import comb
    
    total_pass = 0
    valid_samples = 0
    debug_samples = []
    
    for (intent, normalized_url), results in sample_results.items():
        n = len(results)
        
        # 如果样本数量少于最小要求，跳过该样本
        if n < min_samples:
            continue
        
        valid_samples += 1
        
        # 计算该样本中有多少个正确结果
        correct_count = sum(1 for r in results if r['final_score'] == 1 and r['eval_type'] != 'webjudge')
        fail_count = n - correct_count
        
        # 如果正确数为0，则pass@k为0
        if correct_count == 0:
            if debug and len(debug_samples) < 5:
                debug_samples.append({
                    'n': n, 'c': correct_count, 'k': k,
                    'prob': 0.0, 'method': 'zero_correct'
                })
            continue
        
        # 如果失败次数少于k，不可能全抽到失败，概率为1
        # 这是唯一能保证 pass@k = 1.0 的情况
        if fail_count < k:
            total_pass += 1
            if debug and len(debug_samples) < 5:
                debug_samples.append({
                    'n': n, 'c': correct_count, 'k': k,
                    'prob': 1.0, 'method': 'fail_count<k',
                    'note': f'失败{fail_count}次 < k={k}'
                })
            continue
        
        # 使用数学公式计算: 1 - C(n-c, k) / C(n, k)
        # 其中 n 是总样本数, c 是正确样本数, k 是采样数
        # 这个公式计算的是"从n次中抽k次，至少有一次成功"的概率
        # 等价于 1 - "从n次中抽k次，全部失败"的概率
        
        try:
            # 标准公式（此时已保证 fail_count >= k）
            prob = 1 - comb(fail_count, k) / comb(n, k)
            total_pass += prob
            
            if debug and len(debug_samples) < 5:
                debug_samples.append({
                    'n': n, 'c': correct_count, 'k': k,
                    'prob': prob, 'method': 'formula',
                    'formula': f"1 - C({fail_count},{k})/C({n},{k})",
                    'detail': f"1 - {comb(fail_count, k)}/{comb(n, k)} = {prob:.6f}"
                })
                
        except (ValueError, ZeroDivisionError) as e:
            # 如果计算失败，使用采样估计
            sample_pass = 0
            for _ in range(num_iterations):
                sampled = random.sample(results, k)
                if any(r['final_score'] == 1 and r['eval_type'] != 'webjudge' for r in sampled):
                    sample_pass += 1
            prob = sample_pass / num_iterations
            total_pass += prob
            
            if debug and len(debug_samples) < 5:
                debug_samples.append({
                    'n': n, 'c': correct_count, 'k': k,
                    'prob': prob, 'method': 'sampling',
                    'error': str(e)
                })
    
    if debug and debug_samples:
        print(f"\n[DEBUG] Pass@{k} 前5个样本的计算详情:")
        for i, s in enumerate(debug_samples, 1):
            print(f"  样本{i}: n={s['n']}, 成功={s['c']}, 失败={s['n']-s['c']}, k={s['k']}")
            print(f"         概率={s['prob']:.6f}, 方法={s['method']}")
            if 'formula' in s:
                print(f"         公式: {s['formula']}")
            if 'detail' in s:
                print(f"         计算: {s['detail']}")
            if 'note' in s:
                print(f"         说明: {s['note']}")
    
    if valid_samples == 0:
        return 0.0, 0
    
    return total_pass / valid_samples, valid_samples


def normalize_url(url):
    """
    标准化URL，忽略IP地址/域名的差异，只保留端口和路径
    支持处理包含 |AND| 分隔符的多个URL
    例如：
    <WEBARENA_EXAMPLE_URL_1>
    <WEBARENA_EXAMPLE_URL_2>
    都会被标准化为：:7770/photosmart-plus-b209.html
    
    对于多URL的情况：
    http://ip1:7770/page1.html |AND| http://ip2:7770/page2.html
    会被标准化为排序后的形式（忽略IP差异）
    """
    if not url:
        return url
    
    # 检查是否包含 |AND| 分隔符
    if ' |AND| ' in url or '|AND|' in url:
        # 分割多个URL
        urls = [u.strip() for u in url.replace(' |AND| ', '|AND|').split('|AND|')]
        # 标准化每个URL
        normalized_urls = []
        for u in urls:
            try:
                parsed = urlparse(u)
                port = f":{parsed.port}" if parsed.port else ""
                path = parsed.path or "/"
                query = f"?{parsed.query}" if parsed.query else ""
                fragment = f"#{parsed.fragment}" if parsed.fragment else ""
                normalized = f"{port}{path}{query}{fragment}"
                normalized_urls.append(normalized)
            except Exception:
                normalized_urls.append(u)
        
        # 排序后用 |AND| 连接，确保顺序一致
        return ' |AND| '.join(sorted(normalized_urls))
    else:
        # 单个URL的情况
        try:
            parsed = urlparse(url)
            # 只保留端口和路径（以及query和fragment如果有的话）
            port = f":{parsed.port}" if parsed.port else ""
            path = parsed.path or "/"
            query = f"?{parsed.query}" if parsed.query else ""
            fragment = f"#{parsed.fragment}" if parsed.fragment else ""
            normalized = f"{port}{path}{query}{fragment}"
            return normalized
        except Exception as e:
            # 如果解析失败，返回原始URL
            return url


def load_evaluation_data(json_file):
    """加载评估数据JSON文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"警告: 无法读取文件 {json_file}: {e}")
        return None


def analyze_test_config(config_dir):
    """分析测试配置文件目录，返回所有样本的集合"""
    config_path = Path(config_dir)
    
    if not config_path.exists():
        print(f"警告: 配置目录不存在: {config_dir}")
        return set(), {}
    
    config_samples = set()
    config_sample_details = {}
    
    json_files = sorted(list(config_path.glob('*.json')), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            intent = data.get('intent', '')
            start_url = data.get('start_url', '')
            
            if intent and start_url:
                normalized_url = normalize_url(start_url)
                sample_key = (intent, normalized_url)
                config_samples.add(sample_key)
                
                if sample_key not in config_sample_details:
                    config_sample_details[sample_key] = {
                        'intent': intent,
                        'normalized_url': normalized_url,
                        'config_files': [],
                        'original_urls': []
                    }
                
                config_sample_details[sample_key]['config_files'].append(json_file.name)
                config_sample_details[sample_key]['original_urls'].append(start_url)
        except Exception as e:
            print(f"警告: 无法处理配置文件 {json_file}: {e}")
    
    return config_samples, config_sample_details


def analyze_trajectory_data(trajectory_dir, config_dir=None, k_values=None, debug=False):
    """分析trajectory_data目录下的所有样本"""
    if k_values is None:
        k_values = [1, 5]
    trajectory_path = Path(trajectory_dir)
    
    if not trajectory_path.exists():
        print(f"错误: 目录不存在: {trajectory_dir}")
        return
    
    # 如果提供了配置目录，先分析原始测试集
    config_samples = set()
    config_sample_details = {}
    if config_dir:
        print(f"\n正在分析原始测试集: {config_dir}")
        config_samples, config_sample_details = analyze_test_config(config_dir)
        print(f"原始测试集包含 {len(config_samples)} 个不同的样本（按 intent+标准化URL 区分）")
        print("=" * 80)
    
    # 用于存储每个(intent, start_url)组合的所有测试结果
    sample_results = defaultdict(list)
    
    # 遍历所有子目录
    subdirs = [d for d in trajectory_path.iterdir() if d.is_dir() and 'val_' in d.name]
    print(f"找到 {len(subdirs)} 个子目录 {subdirs[0].name}")
    
    processed_count = 0
    skipped_count = 0
    
    for subdir in subdirs:
        json_file = subdir / "evaluation_data.json"
        
        if not json_file.exists():
            print(f"警告: 未找到文件 {json_file}")
            skipped_count += 1
            continue
        
        data = load_evaluation_data(json_file)
        if data is None:
            skipped_count += 1
            continue
        
        task_id = data.get('task_id', 'unknown')
        intent = data.get('intent', 'unknown')
        start_url = data.get('start_url', 'unknown')
        final_score = data.get('final_score', 0)
        eval_type = data.get('eval_type', 'unknown')
        timestamp = data.get('timestamp', '')
        training_step = data.get('training_step', '')
        
        # 标准化start_url，忽略IP地址差异
        normalized_url = normalize_url(start_url)
        
        # 使用(intent, normalized_url)作为key来标识唯一样本
        sample_key = (intent, normalized_url)
        
        sample_results[sample_key].append({
            'task_id': task_id,
            'original_start_url': start_url,  # 保留原始URL用于显示
            'final_score': final_score,
            'timestamp': timestamp,
            'training_step': training_step,
            'eval_type': eval_type,
            'subdir': subdir.name
        })
        
        processed_count += 1
    
    print(f"\n成功处理: {processed_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")
    print(f"=" * 80)
    
    # 分析结果
    total_samples = len(sample_results)
    correct_samples = 0
    
    # 存储详细结果
    correct_sample_details = []
    incorrect_sample_details = []
    
    for (intent, normalized_url), results in sample_results.items():
        # 检查是否有任何一次final_score为1
        has_correct = any(result['final_score'] == 1 and result['eval_type'] != 'webjudge' for result in results)
        
        max_score = max(result['final_score'] for result in results)
        test_count = len(results)
        
        # 收集所有使用过的原始URL（用于显示）
        original_urls = list(set([r['original_start_url'] for r in results]))
        
        if has_correct:
            correct_samples += 1
            correct_sample_details.append({
                'intent': intent,
                'normalized_url': normalized_url,
                'original_urls': original_urls,
                'test_count': test_count,
                'max_score': max_score,
                'results': results
            })
        else:
            incorrect_sample_details.append({
                'intent': intent,
                'normalized_url': normalized_url,
                'original_urls': original_urls,
                'test_count': test_count,
                'max_score': max_score,
                'results': results
            })
    
    # 分析实际测试的样本集合
    tested_samples = set(sample_results.keys())
    
    # 计算 Pass@k 指标
    pass_k_results = compute_and_display_pass_at_k(sample_results, k_values, debug=debug)
    
    # 打印基础统计结果
    print("\n" + "=" * 80)
    print("统计结果（忽略start_url的IP差异）")
    print("=" * 80)
    print(f"实际测试的样本数（不同intent+标准化URL组合）: {len(tested_samples)}")
    print(f"总样本数（包含重复测试）: {total_samples}")
    print(f"正确样本数（至少一次得分为1）: {correct_samples}")
    print(f"错误样本数（从未得分为1）: {total_samples - correct_samples}")
    print(f"准确率: {correct_samples / total_samples * 100:.2f}%" if total_samples > 0 else "准确率: N/A")
    print("=" * 80)


def compute_and_display_pass_at_k(sample_results, k_values, debug=False):
    """
    计算并显示多个k值的pass@k指标
    
    Args:
        sample_results: dict, 样本测试结果
        k_values: list of int, 要计算的k值列表
        debug: bool, 是否显示调试信息
    
    Returns:
        dict: 包含所有k值的pass@k结果
    """
    if not k_values:
        k_values = [1, 5]
    
    k_values = sorted(k_values)
    max_k = max(k_values)
    
    print(f"\n正在计算 Pass@k 指标 (k={k_values})...")
    
    # 对每个k值，分别统计（min_samples=k）
    results_separate = {}
    for k in k_values:
        pass_k, valid = calculate_pass_at_k(sample_results, k=k, min_samples=k, debug=False)
        results_separate[k] = (pass_k, valid)
        print(f"  Pass@{k}: {pass_k * 100:.2f}% (基于 {valid} 个测试次数>={k}的样本)")
    
    # 在相同样本集上计算所有k值（只统计测试次数>=max_k的样本，确保可比性）
    print(f"\n在相同样本集上计算 Pass@k (只统计测试次数>={max_k}的样本)...")
    results_same = {}
    for k in k_values:
        show_debug = debug and k == k_values[0]  # 只在第一个k显示debug
        pass_k, valid = calculate_pass_at_k(sample_results, k=k, min_samples=max_k, debug=show_debug)
        results_same[k] = (pass_k, valid)
        print(f"  Pass@{k}: {pass_k * 100:.2f}% (基于 {valid} 个样本)")
    
    # 验证单调性
    print("\n验证 Pass@k 单调性...")
    all_valid = True
    for i in range(len(k_values) - 1):
        k1, k2 = k_values[i], k_values[i + 1]
        p1, p2 = results_same[k1][0], results_same[k2][0]
        diff = p2 - p1
        if p2 >= p1 - 1e-9:  # 允许浮点误差
            print(f"  ✓ Pass@{k2} >= Pass@{k1} (差值: +{diff*100:.2f}%)")
        else:
            print(f"  ⚠ Pass@{k2} < Pass@{k1} (差值: {diff*100:.2f}%, 不符合预期！)")
            all_valid = False
    
    if not all_valid:
        print("\n  注意: 发现单调性违反！这可能是浮点精度问题或数据异常")
    
    return {
        'separate': results_separate,
        'same_set': results_same,
        'k_values': k_values,
        'max_k': max_k
    }
    
    # 如果有原始测试集，显示对比信息
    if config_samples:
        print("\n" + "-" * 80)
        print("与原始测试集对比:")
        print("-" * 80)
        print(f"原始测试集样本数: {len(config_samples)}")
        print(f"实际测试的样本数: {len(tested_samples)}")
        print(f"测试覆盖率: {len(tested_samples) / len(config_samples) * 100:.2f}%" if config_samples else "N/A")
        
        # 找出被测试的样本
        tested_in_config = tested_samples & config_samples
        print(f"在原始测试集中的样本: {len(tested_in_config)}")
        
        # 找出未被测试的样本
        not_tested = config_samples - tested_samples
        print(f"未被测试的样本数: {len(not_tested)}")
        
        # 找出不在原始测试集中的样本（可能是额外的测试）
        extra_tested = tested_samples - config_samples
        if extra_tested:
            print(f"不在原始测试集中的样本数: {len(extra_tested)}")
            print("\n不在原始测试集中的样本详情:")
            for i, (intent, normalized_url) in enumerate(sorted(extra_tested), 1):
                print(f"\n  {i}. Intent: {intent[:100]}{'...' if len(intent) > 100 else ''}")
                print(f"     标准化URL: {normalized_url}")
                # 找出这个样本的原始URL
                if (intent, normalized_url) in sample_results:
                    original_urls = list(set([r['original_start_url'] for r in sample_results[(intent, normalized_url)]]))
                    print(f"     原始URL: {original_urls[0] if original_urls else 'N/A'}")
    
    print("=" * 80)
    
    # 统计测试次数分布
    test_count_distribution = defaultdict(int)
    for results in sample_results.values():
        test_count = len(results)
        test_count_distribution[test_count] += 1
    
    print("\n测试次数分布:")
    for test_count in sorted(test_count_distribution.keys()):
        count = test_count_distribution[test_count]
        print(f"  测试 {test_count} 次: {count} 个样本")
    
    print("\n" + "-" * 80)
    print("Pass@k 指标说明:")
    print("-" * 80)
    print("Pass@k 表示从每个样本的多次测试中随机采样 k 个，至少有一个成功的概率")
    print()
    print("两种统计方式:")
    max_k = max(k_values)
    print(f"1. 分别统计: Pass@k 只统计测试次数>=k的样本")
    print(f"   这种方式可能出现 Pass@k1 > Pass@k2 的情况（因为样本集不同）")
    print(f"2. 相同样本集: 都只统计测试次数>={max_k}的样本，确保可比性")
    print(f"   这种方式下 Pass@k 随k单调递增（理论保证）")
    
    # # 可选：显示详细信息
    # print("\n" + "=" * 80)
    # print("正确样本详情（前10个）:")
    # print("=" * 80)
    # for i, detail in enumerate(correct_sample_details[:10], 1):
    #     print(f"\n{i}. Intent: {detail['intent']}")
    #     print(f"   标准化URL: {detail['normalized_url']}")
    #     if len(detail['original_urls']) > 1:
    #         print(f"   原始URLs ({len(detail['original_urls'])}个不同IP): {', '.join(detail['original_urls'][:3])}" + 
    #               (' ...' if len(detail['original_urls']) > 3 else ''))
    #     else:
    #         print(f"   原始URL: {detail['original_urls'][0]}")
    #     print(f"   测试次数: {detail['test_count']}, 最高分: {detail['max_score']}")
    #     for j, result in enumerate(detail['results'], 1):
    #         status = "✓" if result['final_score'] == 1 else "✗"
    #         print(f"   [{status}] 测试{j}: Task ID={result['task_id']}, 得分={result['final_score']}, "
    #               f"时间={result['timestamp']}, 目录={result['subdir']}")
    
    # print("\n" + "=" * 80)
    # print("错误样本详情（前10个）:")
    # print("=" * 80)
    # for i, detail in enumerate(incorrect_sample_details[:10], 1):
    #     print(f"\n{i}. Intent: {detail['intent']}")
    #     print(f"   标准化URL: {detail['normalized_url']}")
    #     if len(detail['original_urls']) > 1:
    #         print(f"   原始URLs ({len(detail['original_urls'])}个不同IP): {', '.join(detail['original_urls'][:3])}" + 
    #               (' ...' if len(detail['original_urls']) > 3 else ''))
    #     else:
    #         print(f"   原始URL: {detail['original_urls'][0]}")
    #     print(f"   测试次数: {detail['test_count']}, 最高分: {detail['max_score']}")
    #     for j, result in enumerate(detail['results'], 1):
    #         print(f"   [✗] 测试{j}: Task ID={result['task_id']}, 得分={result['final_score']}, "
    #               f"时间={result['timestamp']}, 目录={result['subdir']}")
    
    # 保存详细结果到JSON文件
    output_file = Path(trajectory_dir) / "accuracy_report.json"
    
    # 构建pass@k指标的字典
    pass_k_metrics = {}
    for k in pass_k_results['k_values']:
        pass_k_metrics[f'pass_at_{k}_separate'] = pass_k_results['separate'][k][0]
        pass_k_metrics[f'pass_at_{k}_separate_valid_samples'] = pass_k_results['separate'][k][1]
        pass_k_metrics[f'pass_at_{k}_same_set'] = pass_k_results['same_set'][k][0]
        pass_k_metrics[f'pass_at_{k}_same_set_valid_samples'] = pass_k_results['same_set'][k][1]
    
    report = {
        'summary': {
            'tested_unique_samples': len(tested_samples),
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'incorrect_samples': total_samples - correct_samples,
            'accuracy': correct_samples / total_samples if total_samples > 0 else 0,
            **pass_k_metrics,  # 解包所有pass@k指标
            'processed_files': processed_count,
            'skipped_files': skipped_count
        },
        'test_count_distribution': dict(test_count_distribution),
        'correct_samples': correct_sample_details,
        'incorrect_samples': incorrect_sample_details
    }
    
    # 如果有原始测试集信息，添加到报告中
    if config_samples:
        tested_in_config = tested_samples & config_samples
        not_tested = config_samples - tested_samples
        extra_tested = tested_samples - config_samples
        
        report['summary']['original_test_set_size'] = len(config_samples)
        report['summary']['tested_from_original'] = len(tested_in_config)
        report['summary']['coverage_rate'] = len(tested_samples) / len(config_samples) if config_samples else 0
        report['summary']['not_tested_count'] = len(not_tested)
        report['summary']['extra_tested_count'] = len(extra_tested)
        
        # 保存未测试的样本列表
        report['not_tested_samples'] = [
            {'intent': intent, 'normalized_url': url} 
            for intent, url in sorted(not_tested)
        ]
    
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告将保存到: {output_file}")
    print("（当前已注释保存功能，取消注释第219-220行可启用）")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='统计trajectory_data目录下样本的得分，并与原始测试集对比',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python calculate_trajectory_accuracy.py /path/to/trajectory_data
  python calculate_trajectory_accuracy.py --dir /path/to/trajectory_data --config /path/to/config_dir
        """
    )
    exp_name = "Qwen3-VL-32B-Thinking_Webarena_Webjudge_online_rl_global_step_80_Async_Stepwise_klconv4__ConSch_wPRALL4_Prompt1012_SuffleFalse_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasettest_webarena_conflict_RewardModel0912"
    #exp_name = "Qwen3-VL-32B-Thinking_test_Async_Stepwise_klconv4__ConSch_wPRALL4_Prompt1012_SuffleFalse_MeanFormatReward_64_ObserTypeimage_Gpuh20-3ehigh_Datasettest_webarena_conflict_RewardModel0912"
    parser.add_argument(
        'trajectory_dir',
        nargs='?',
        default=f'<TRAJECTORY_DATA_DIR_TEMPLATE>',
        help='trajectory_data目录的路径（默认为预设路径）'
    )
    
    parser.add_argument(
        '--dir',
        dest='trajectory_dir_alt',
        help='trajectory_data目录的路径（替代参数）'
    )
    
    parser.add_argument(
        '--config',
        dest='config_dir',
        default='<DATASET_CONFIG_DIR>',
        help='原始测试集配置文件目录（默认为 test_webarena_conflict）'
    )
    
    parser.add_argument(
        '--k',
        dest='k_values',
        type=int,
        nargs='+',
        default=[1, 2, 3, 5],
        help='要计算的k值列表（默认为1和5），例如: --k 1 3 5 10'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='显示详细的调试信息'
    )
    
    args = parser.parse_args()
    
    # 使用--dir参数（如果提供）或位置参数
    trajectory_dir = args.trajectory_dir_alt if args.trajectory_dir_alt else args.trajectory_dir
    config_dir = args.config_dir
    k_values = args.k_values
    debug = args.debug
    
    print("=" * 80)
    print("样本得分统计工具（忽略start_url的IP差异）")
    print("=" * 80)
    print(f"分析目录: {trajectory_dir}")
    if config_dir:
        print(f"原始测试集: {config_dir}")
    print(f"计算 Pass@k，k值: {k_values}")
    print("注意：相同intent+端口+路径但不同IP的样本将被视为同一样本")
    print("=" * 80)
    
    analyze_trajectory_data(trajectory_dir, config_dir=config_dir, k_values=k_values, debug=debug)


if __name__ == "__main__":
    main()
