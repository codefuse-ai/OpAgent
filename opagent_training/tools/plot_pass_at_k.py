#!/usr/bin/env python3
"""
根据 site_comparison.csv 绘制 Pass@k 折线图
"""

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

def plot_pass_at_k_comparison(csv_path, output_path):
    """
    绘制 Pass@k 对比折线图
    
    Args:
        csv_path: CSV 文件路径
        output_path: 输出 PDF 文件路径
    """
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 提取 k 值和 Average 列数据
    k_values = []
    exp1_scores = []
    exp2_scores = []
    
    for _, row in df.iterrows():
        exp_name = row['Experiment']
        average_score = float(row['Average'])
        
        # 提取 k 值
        if 'Exp1_Pass@' in exp_name:
            k = int(exp_name.replace('Exp1_Pass@', ''))
            k_values.append(k)
            exp1_scores.append(average_score)
        elif 'Exp2_Pass@' in exp_name:
            k = int(exp_name.replace('Exp2_Pass@', ''))
            exp2_scores.append(average_score)
    
    # 按 k 值排序
    sorted_indices = np.argsort(k_values)
    k_values = [k_values[i] for i in sorted_indices]
    exp1_scores = [exp1_scores[i] for i in sorted_indices]
    exp2_scores = [exp2_scores[i] for i in sorted_indices]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用学术论文常用的配色和样式
    color1 = '#4472C4'  # 专业蓝色
    color2 = '#ED7D31'  # 专业橙色
    
    # 绘制折线
    line1 = ax.plot(k_values, exp1_scores, 
                    marker='o', markersize=8, linewidth=2.5, 
                    color=color1, label='Qwen3-VL-Thinking-RL-HybridReward-Zero',
                    markeredgecolor='white', markeredgewidth=1.5)
    
    line2 = ax.plot(k_values, exp2_scores, 
                    marker='s', markersize=8, linewidth=2.5, 
                    color=color2, label='Qwen3-VL-Thinking',
                    markeredgecolor='white', markeredgewidth=1.5)
    
    # 在每个数据点上标注数值
    for i, (k, score) in enumerate(zip(k_values, exp1_scores)):
        ax.text(k, score + 0.8, f'{score:.2f}', 
               ha='center', va='bottom', fontsize=10, 
               color=color1, fontweight='bold')
    
    for i, (k, score) in enumerate(zip(k_values, exp2_scores)):
        ax.text(k, score - 1.2, f'{score:.2f}', 
               ha='center', va='top', fontsize=10, 
               color=color2, fontweight='bold')
    
    # 设置标签和标题
    ax.set_xlabel('k', fontsize=15, fontweight='bold')
    ax.set_ylabel('Average Pass@k (%)', fontsize=15, fontweight='bold')
    ax.set_title('Pass@k Performance Comparison', fontsize=17, fontweight='bold', pad=20)
    
    # 设置 x 轴刻度
    ax.set_xticks(k_values)
    ax.set_xticklabels([f'{k}' for k in k_values], fontsize=12)
    
    # 设置 y 轴范围和刻度
    y_min = 10
    y_max = 45
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(10, 50, 5))
    
    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax.set_axisbelow(True)
    
    # 优化图例位置和样式
    legend = ax.legend(fontsize=11, loc='lower right', framealpha=0.95, 
                      edgecolor='black', fancybox=False, shadow=False)
    legend.get_frame().set_linewidth(0.8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PDF
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"折线图已保存为 PDF: {output_path}")
    
    # 同时保存PNG版本
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"折线图已保存为 PNG: {png_path}")
    
    plt.close()
    
    # 打印统计信息
    print("\n统计信息:")
    print("=" * 60)
    print(f"{'k':<10} {'Exp1 (%)':<20} {'Exp2 (%)':<20} {'差值 (%)':<15}")
    print("-" * 60)
    for k, s1, s2 in zip(k_values, exp1_scores, exp2_scores):
        diff = s1 - s2
        print(f"{k:<10} {s1:<20.2f} {s2:<20.2f} {diff:<15.2f}")
    print("=" * 60)

if __name__ == "__main__":
    csv_path = "<SITE_COMPARISON_CSV>"
    output_path = "<OUTPUT_PDF_PATH>"
    
    plot_pass_at_k_comparison(csv_path, output_path)
