#!/usr/bin/env python3
"""
对比两个实验在各个子网站上的准确率
根据URL端口号识别网站类型
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse
import csv

# 端口到网站的映射
SITE_PORT_MAP = {
    "7770": "shopping",
    "7780": "shopping_admin",
    "9999": "reddit",
    "8023": "gitlab",
    "8888": "wikipedia",
    "3000": "map",
}

def normalize_url(url):
    """
    标准化URL，忽略IP地址/域名的差异，只保留端口和路径
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
        
        # 排序后用 |AND| 连接
        return ' |AND| '.join(sorted(normalized_urls))
    else:
        # 单个URL的情况
        try:
            parsed = urlparse(url)
            # 只保留端口和路径
            port = f":{parsed.port}" if parsed.port else ""
            path = parsed.path or "/"
            query = f"?{parsed.query}" if parsed.query else ""
            fragment = f"#{parsed.fragment}" if parsed.fragment else ""
            normalized = f"{port}{path}{query}{fragment}"
            return normalized
        except Exception:
            return url

def extract_port_from_url(url):
    """从URL中提取端口号"""
    try:
        # 处理多个URL的情况（用 |AND| 分隔）
        if '|AND|' in url:
            # 取第一个URL的端口
            first_url = url.split('|AND|')[0].strip()
            parsed = urlparse(first_url)
        else:
            parsed = urlparse(url)
        
        if parsed.port:
            return str(parsed.port)
        return None
    except Exception:
        return None

def get_site_from_url(url):
    """根据URL获取网站名称"""
    port = extract_port_from_url(url)
    if port:
        return SITE_PORT_MAP.get(port, "unknown")
    return "unknown"

def load_evaluation_data(json_file):
    """加载评估数据JSON文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"警告: 无法读取文件 {json_file}: {e}")
        return None

def calculate_pass_at_k(sample_results, k, use_fixed_denominator=False):
    """
    计算 pass@k 指标
    
    参数:
    - sample_results: dict, key为(intent, normalized_url), value为结果列表
    - k: int, 采样数量
    - use_fixed_denominator: bool, 是否使用固定分母812（用于计算总体平均分）
    
    返回:
    - float: pass@k 的值
    - int: 有效样本数
    """
    from math import comb
    
    total_pass = 0
    valid_samples = 0
    
    for (intent, normalized_url), results in sample_results.items():
        n = len(results)
        
        # 如果样本数量少于k，跳过该样本
        if n < k:
            continue
        
        valid_samples += 1
        
        # 计算该样本中有多少个正确结果
        correct_count = sum(1 for r in results if r['final_score'] == 1 and r['eval_type'] != 'webjudge')
        fail_count = n - correct_count
        
        # 如果正确数为0，则pass@k为0
        if correct_count == 0:
            continue
        
        # 如果失败次数少于k，不可能全抽到失败，概率为1
        if fail_count < k:
            total_pass += 1
            continue
        
        # 使用数学公式计算: 1 - C(n-c, k) / C(n, k)
        try:
            prob = 1 - comb(fail_count, k) / comb(n, k)
            total_pass += prob
        except (ValueError, ZeroDivisionError):
            # 如果计算失败，跳过
            valid_samples -= 1
    
    # 根据参数决定使用固定分母还是实际有效样本数
    if use_fixed_denominator:
        # 用于计算总体Average时，固定除以812
        FIXED_TOTAL_SAMPLES = 812
        denominator = FIXED_TOTAL_SAMPLES
    else:
        # 用于计算各个子网站时，使用实际的有效样本数
        denominator = valid_samples if valid_samples > 0 else 1
    
    if denominator == 0:
        return 0.0, valid_samples
    
    return total_pass / denominator, valid_samples

def analyze_trajectory_by_site(trajectory_dir, k_values=[1, 2, 3, 5]):
    """
    分析trajectory_data目录，按网站统计pass@k指标
    
    Returns:
        dict: {site: {f'pass@{k}': float, ...}}
    """
    trajectory_path = Path(trajectory_dir)
    
    if not trajectory_path.exists():
        print(f"错误: 目录不存在: {trajectory_dir}")
        return {}
    
    # 用于存储每个(intent, start_url)组合的所有测试结果，按网站分类
    site_sample_results = defaultdict(lambda: defaultdict(list))
    
    # 遍历所有子目录
    subdirs = [d for d in trajectory_path.iterdir() if d.is_dir() and 'val_' in d.name]
    print(f"  找到 {len(subdirs)} 个子目录")
    
    processed_count = 0
    skipped_count = 0
    
    for subdir in subdirs:
        json_file = subdir / "evaluation_data.json"
        
        if not json_file.exists():
            skipped_count += 1
            continue
        
        data = load_evaluation_data(json_file)
        if data is None:
            skipped_count += 1
            continue
        
        intent = data.get('intent', 'unknown')
        start_url = data.get('start_url', 'unknown')
        final_score = data.get('final_score', 0)
        eval_type = data.get('eval_type', 'unknown')
        
        # 获取网站类型
        site = get_site_from_url(start_url)
        
        # 标准化start_url
        normalized_url = normalize_url(start_url)
        
        # 使用(intent, normalized_url)作为key来标识唯一样本
        sample_key = (intent, normalized_url)
        
        site_sample_results[site][sample_key].append({
            'final_score': final_score,
            'eval_type': eval_type,
        })
        
        processed_count += 1
    
    print(f"  成功处理: {processed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")
    
    # 计算每个网站的pass@k指标
    site_stats = {}
    
    for site, sample_results in site_sample_results.items():
        site_stats[site] = {}
        
        for k in k_values:
            pass_k, valid = calculate_pass_at_k(sample_results, k)
            site_stats[site][f'pass@{k}'] = pass_k
            site_stats[site][f'pass@{k}_valid'] = valid
    
    return site_stats, site_sample_results

def main():
    # 两个实验的配置
    exp1_name = "Qwen3-VL-32B-Thinking_Webarena_Webjudge_online_rl_global_step_80_Async_Stepwise_klconv4__ConSch_wPRALL4_Prompt1012_SuffleFalse_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasettest_webarena_conflict_RewardModel0912"
    exp2_name = "Qwen3-VL-32B-Thinking_test_Async_Stepwise_klconv4__ConSch_wPRALL4_Prompt1012_SuffleFalse_MeanFormatReward_64_ObserTypeimage_Gpuh20-3ehigh_Datasettest_webarena_conflict_RewardModel0912"
    
    base_dir = "<BASE_DIR>"
    
    exp1_trajectory_dir = os.path.join(base_dir, exp1_name, "trajectory_data")
    exp2_trajectory_dir = os.path.join(base_dir, exp2_name, "trajectory_data")
    
    k_values = [1, 2, 3, 4, 5]
    
    print("=" * 80)
    print("对比两个实验在各个子网站上的Pass@k指标")
    print("=" * 80)
    
    # 分析实验1
    print(f"\n分析实验1: {exp1_name}")
    exp1_stats, exp1_all_samples = analyze_trajectory_by_site(exp1_trajectory_dir, k_values)
    
    # 分析实验2
    print(f"\n分析实验2: {exp2_name}")
    exp2_stats, exp2_all_samples = analyze_trajectory_by_site(exp2_trajectory_dir, k_values)
    
    # 计算总体pass@k（合并所有网站的样本）
    def calculate_total_pass_k(all_samples, k_values):
        all_sample_results = {}
        for site_samples in all_samples.values():
            all_sample_results.update(site_samples)
        
        total_stats = {}
        for k in k_values:
            # 计算总体平均时使用固定分母812
            pass_k, valid = calculate_pass_at_k(all_sample_results, k, use_fixed_denominator=True)
            total_stats[f'pass@{k}'] = pass_k
        
        return total_stats
    
    exp1_total_stats = calculate_total_pass_k(exp1_all_samples, k_values)
    exp2_total_stats = calculate_total_pass_k(exp2_all_samples, k_values)
    
    # 按照指定顺序输出：Average, Shopping, CMS (shopping_admin), Reddit, GitLab, Maps
    site_order = ["shopping", "shopping_admin", "reddit", "gitlab", "map"]
    site_display_names = {
        "shopping": "Shopping",
        "shopping_admin": "CMS",
        "reddit": "Reddit",
        "gitlab": "GitLab",
        "map": "Maps"
    }
    
    # 打印结果
    print("\n" + "=" * 80)
    print("统计结果 (Pass@k):")
    print("=" * 80)
    
    for k in k_values:
        print(f"\n=== Pass@{k} ===")
        print(f"{'网站':<15} {'实验1':<15} {'实验2':<15}")
        print("-" * 50)
        print(f"{'Average':<15} {exp1_total_stats[f'pass@{k}']*100:>13.2f}% {exp2_total_stats[f'pass@{k}']*100:>13.2f}%")
        
        for site in site_order:
            display_name = site_display_names.get(site, site)
            exp1_data = exp1_stats.get(site, {})
            exp2_data = exp2_stats.get(site, {})
            
            exp1_pass = exp1_data.get(f'pass@{k}', 0)
            exp2_pass = exp2_data.get(f'pass@{k}', 0)
            
            print(f"{display_name:<15} {exp1_pass*100:>13.2f}% {exp2_pass*100:>13.2f}%")
    
    # 保存到CSV文件（每个实验的每个pass@k占一行）
    output_dir = "<OUTPUT_DIR>"
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv_path = os.path.join(output_dir, "site_comparison.csv")
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 表头：Experiment, Average, Shopping, CMS, Reddit, GitLab, Maps
        header = ['Experiment', 'Average', 'Shopping', 'CMS', 'Reddit', 'GitLab', 'Maps']
        writer.writerow(header)
        
        # 写入实验1的各个pass@k
        for k in k_values:
            row = [f'Exp1_Pass@{k}']
            # Average
            row.append(f'{exp1_total_stats[f"pass@{k}"]*100:.2f}')
            # 各网站
            for site in site_order:
                exp1_data = exp1_stats.get(site, {})
                exp1_pass = exp1_data.get(f'pass@{k}', 0)
                row.append(f'{exp1_pass*100:.2f}')
            writer.writerow(row)
        
        # 写入实验2的各个pass@k
        for k in k_values:
            row = [f'Exp2_Pass@{k}']
            # Average
            row.append(f'{exp2_total_stats[f"pass@{k}"]*100:.2f}')
            # 各网站
            for site in site_order:
                exp2_data = exp2_stats.get(site, {})
                exp2_pass = exp2_data.get(f'pass@{k}', 0)
                row.append(f'{exp2_pass*100:.2f}')
            writer.writerow(row)
    
    print(f"\n结果已保存到: {output_csv_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
