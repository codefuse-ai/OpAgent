import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置学术风格的字体 - 使用 Liberation Serif (Times New Roman 的开源替代)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'Times', 'serif']
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 1.2
rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.unicode_minus'] = False

def load_domain_classification(csv_path):
    """加载领域分类CSV文件"""
    df = pd.read_csv(csv_path)
    # 创建 intent -> domain_category 的映射
    intent_to_domain = dict(zip(df['intent'], df['domain_category']))
    return intent_to_domain

def extract_scores_from_trajectory(trajectory_dir, intent_to_domain):
    """
    从 trajectory_data 目录提取分数，并根据 intent 分配领域
    
    Returns:
        dict: {domain: [scores]}
    """
    domain_scores = {}
    processed_count = 0
    
    # 遍历所有 val_unknown 开头的目录
    for dirname in os.listdir(trajectory_dir):
        if dirname.startswith("val_unknown"):
            dir_path = os.path.join(trajectory_dir, dirname)
            json_file = os.path.join(dir_path, "evaluation_data.json")
            
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        intent = data.get("intent", "")
                        final_score = data.get("final_score")
                        
                        if intent and final_score is not None:
                            # 根据 intent 查找对应的领域
                            domain = intent_to_domain.get(intent, "Other")
                            
                            if domain not in domain_scores:
                                domain_scores[domain] = []
                            
                            domain_scores[domain].append(float(final_score))
                            processed_count += 1
                            
                except Exception as e:
                    print(f"错误: 处理文件 {json_file} 时发生错误: {e}")
    
    print(f"  处理了 {processed_count} 个样本")
    return domain_scores

def calculate_domain_averages(domain_scores):
    """计算每个领域的平均分（剔除负分）"""
    domain_averages = {}
    for domain, scores in domain_scores.items():
        if scores:
            # 过滤掉负分
            non_negative_scores = [s for s in scores if s >= 0]
            if non_negative_scores:
                avg = np.mean(non_negative_scores)
                domain_averages[domain] = avg
            else:
                # 如果全是负分，则不计入该领域
                print(f"  警告: 领域 '{domain}' 的所有分数都是负分，已跳过")
    return domain_averages

def translate_domain_to_english(chinese_domain):
    """将中文领域名翻译成英文"""
    translation = {
        "教育": "Education",
        "娱乐": "Entertainment",
        "购物/电商": "E-commerce",
        "新闻/资讯": "News & Info",
        "社交": "Social Media",
        "旅游/出行": "Travel",
        "科技/科普": "Sci-Tech",
        "政府/公共服务": "Government",
        "金融": "Finance",
        "医疗健康": "Healthcare",
        "生活服务/餐饮": "Lifestyle",
        "数码/电子产品": "Electronics",
        "汽车": "Automotive",
        "其他": "Others"
    }
    return translation.get(chinese_domain, chinese_domain)

def plot_comparison(exp1_averages, exp2_averages, exp1_name, exp2_name, output_path):
    """
    绘制对比柱状图
    
    Args:
        exp1_averages: 实验1的领域平均分
        exp2_averages: 实验2的领域平均分
        exp1_name: 实验1名称
        exp2_name: 实验2名称
        output_path: 输出图片路径
    """
    # 获取所有领域（合并两个实验的领域）
    all_domains = set(exp1_averages.keys()) | set(exp2_averages.keys())
    
    # 排除"其他"类别，单独处理
    main_domains = [d for d in all_domains if d != "其他" and d != "Other"]
    
    # 计算差值（实验1 - 实验2），按差值降序排序
    domain_differences = {}
    for domain in main_domains:
        score1 = exp1_averages.get(domain, 0)
        score2 = exp2_averages.get(domain, 0)
        domain_differences[domain] = score1 - score2
    
    # 按差值降序排序（差值越大越靠前）
    sorted_domains = sorted(main_domains, key=lambda d: domain_differences[d], reverse=True)
    
    # 如果有"其他"类别，添加到最后
    if "其他" in all_domains:
        sorted_domains.append("其他")
    elif "Other" in all_domains:
        sorted_domains.append("Other")
    
    # 翻译成英文
    english_domains = [translate_domain_to_english(d) for d in sorted_domains]
    
    # 准备数据
    exp1_scores = [exp1_averages.get(d, 0) for d in sorted_domains]
    exp2_scores = [exp2_averages.get(d, 0) for d in sorted_domains]
    
    # 设置图形大小
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 设置柱状图位置
    x = np.arange(len(english_domains))
    width = 0.38
    
    # 使用学术论文常用的配色方案
    # 蓝色系和橙色系，色盲友好
    color1 = '#4472C4'  # 专业蓝色
    color2 = '#ED7D31'  # 专业橙色
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, exp1_scores, width, label=exp1_name, 
                   color=color1, alpha=0.85, edgecolor='black', linewidth=0.7)
    bars2 = ax.bar(x + width/2, exp2_scores, width, label=exp2_name,
                   color=color2, alpha=0.85, edgecolor='black', linewidth=0.7)
    
    # 在柱子上方添加数值标签（更小的字体，避免拥挤）
    def add_value_labels(bars, offset=0):
        for bar in bars:
            height = bar.get_height()
            if height > 0.2:  # 只显示较高的柱子的标签，避免拥挤
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8, fontweight='normal')
    
    add_value_labels(bars1, offset=0.08)
    add_value_labels(bars2, offset=0.08)
    
    # 设置标签和标题
    ax.set_xlabel('Domain', fontsize=15, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=15, fontweight='bold')
    ax.set_title('Domain-wise Performance Comparison', fontsize=17, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(english_domains, rotation=45, ha='right', fontsize=11)
    
    # 优化图例位置和样式
    legend = ax.legend(fontsize=11, loc='upper right', framealpha=0.95, 
                      edgecolor='black', fancybox=False, shadow=False)
    legend.get_frame().set_linewidth(0.8)
    
    # 添加网格线
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.6)
    ax.set_axisbelow(True)
    
    # 设置y轴范围和刻度
    y_max = max(max(exp1_scores), max(exp2_scores))
    ax.set_ylim(0, y_max * 1.2)
    ax.set_yticks(np.arange(0, y_max * 1.2 + 0.5, 1.0))
    
    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 同时保存为 PDF 格式（学术论文常用）
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"PDF 版本已保存到: {pdf_path}")
    
    plt.close()

def main():
    # 实验1配置
    exp1_config = {
        "name": "Qwen2.5-VL-72B-RL-HybridReward",
        "directory_path": "<EXPERIMENT_ROOT_1>",
        "exp_name": "augvis_rollout_cot_Stepwise_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppuhigh_Datasetph_xl_hz_1112_test_webarena_RewardModel0912"
    }
    
    # 实验2配置
    exp2_config = {
        "name": "Qwen2.5-VL-72B",
        "directory_path": "<EXPERIMENT_ROOT_2>",
        "exp_name": "Qwen2.5-VL-72B-Instruct_wPRALL3_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasetwa_ali_test_webarena_RewardModel0912"
    }
    
    # 领域分类CSV路径
    domain_csv_path = "<DOMAIN_CLASSIFICATION_CSV>"
    
    # 输出路径
    output_dir = "<OUTPUT_DIR>"
    output_image_path = os.path.join(output_dir, "domain_comparison_chart.png")
    output_stats_path = os.path.join(output_dir, "domain_comparison_stats.csv")
    
    print("=" * 50)
    print("开始对比实验...")
    print("=" * 50)
    
    # 加载领域分类
    print("\n加载领域分类数据...")
    intent_to_domain = load_domain_classification(domain_csv_path)
    print(f"已加载 {len(intent_to_domain)} 个样本的领域分类")
    
    # 处理实验1
    print(f"\n处理实验1: {exp1_config['name']}")
    exp1_trajectory_dir = os.path.join(exp1_config['directory_path'], exp1_config['exp_name'], 'trajectory_data')
    exp1_domain_scores = extract_scores_from_trajectory(exp1_trajectory_dir, intent_to_domain)
    exp1_averages = calculate_domain_averages(exp1_domain_scores)
    
    # 处理实验2
    print(f"\n处理实验2: {exp2_config['name']}")
    exp2_trajectory_dir = os.path.join(exp2_config['directory_path'], exp2_config['exp_name'], 'trajectory_data')
    exp2_domain_scores = extract_scores_from_trajectory(exp2_trajectory_dir, intent_to_domain)
    exp2_averages = calculate_domain_averages(exp2_domain_scores)
    
    # 打印统计结果
    print("\n" + "=" * 50)
    print("各领域平均分对比:")
    print("=" * 50)
    print(f"{'领域':<20} {exp1_config['name']:<25} {exp2_config['name']:<25} {'差值':<10}")
    print("-" * 90)
    
    all_domains = sorted(set(exp1_averages.keys()) | set(exp2_averages.keys()))
    comparison_data = []
    
    for domain in all_domains:
        score1 = exp1_averages.get(domain, 0)
        score2 = exp2_averages.get(domain, 0)
        diff = score2 - score1
        english_domain = translate_domain_to_english(domain)
        print(f"{domain:<20} {score1:<25.4f} {score2:<25.4f} {diff:+.4f}")
        
        comparison_data.append({
            "Domain (CN)": domain,
            "Domain (EN)": english_domain,
            exp1_config['name']: score1,
            exp2_config['name']: score2,
            "Difference": diff
        })
    
    # 保存统计数据到CSV
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(output_stats_path, index=False, encoding='utf-8-sig')
    print(f"\n统计数据已保存到: {output_stats_path}")
    
    # 绘制对比图
    print("\n生成对比图...")
    plot_comparison(exp1_averages, exp2_averages, 
                   exp1_config['name'], exp2_config['name'], 
                   output_image_path)
    
    print("\n" + "=" * 50)
    print("对比分析完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
