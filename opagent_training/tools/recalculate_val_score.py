import os
import json
import csv

def analyze_and_generate_csv(base_dir):
    """
    在指定目录中搜索 'val_' 开头的目录，分析其中的 'evaluation_data.json' 文件，
    计算 final_score (5分制) 的统计数据，并生成一份包含非负分数量的 CSV 报告。

    Args:
        base_dir (str): 要搜索的根目录路径。
    """
    # 1. 定义路径和常量
    output_csv_path = os.path.join(base_dir, "results_report")
    if not os.path.isdir(output_csv_path):
        os.makedirs(output_csv_path)
    output_csv_path = os.path.join(output_csv_path, "results_report.csv")
    target_filename = "evaluation_data.json"
    
    # 更新表头，加入“非负分数量”
    csv_header = ['最高分', '最低分', '平均分（5分制除去负分）', '非负分数量', '负分数量', '总样本数', '去重后样本数']
    
    intent_score_dict = {}  # 使用字典存储 intent -> score 的映射，自动去重
    processed_files_count = 0
    total_samples = 0  # 统计总样本数（去重前）
    
    print(f"开始在目录中搜索: {base_dir}")
    print(f"筛选条件: 只处理位于 'val_unknown' 开头目录下的 '{target_filename}' 文件。")

    # 2. 遍历目录树，查找并处理符合条件的文件
    for root, _, files in os.walk(base_dir):
        if target_filename in files and os.path.basename(root).startswith("val_unknown"):
            file_path = os.path.join(root, target_filename)
            processed_files_count += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    score = data.get("final_score")
                    intent = data.get("intent")
                    
                    if score is not None and intent is not None:
                        total_samples += 1
                        # 只保留第一次遇到的 intent 的分数
                        if intent not in intent_score_dict:
                            intent_score_dict[intent] = float(score)
                        # 如果 intent 已存在，则跳过（去重）
                    else:
                        if score is None:
                            print(f"警告: 文件 {file_path} 中未找到 'final_score' 键")
                        if intent is None:
                            print(f"警告: 文件 {file_path} 中未找到 'intent' 键")
            except Exception as e:
                print(f"错误: 处理文件 {file_path} 时发生错误: {e}")

    # 从字典中提取去重后的分数列表
    all_scores = list(intent_score_dict.values())
    unique_sample_count = len(all_scores)  # 去重后的样本数
    
    print(f"\n搜索完成。共处理了 {processed_files_count} 个符合条件的文件。")
    print(f"总样本数（去重前）: {total_samples}")
    print(f"去重后样本数: {unique_sample_count}")
    print(f"重复样本数: {total_samples - unique_sample_count}")
    
    # 3. 分离分数并直接进行统计计算
    non_negative_scores = [s for s in all_scores if s >= 0]
    negative_score_count = len(all_scores) - len(non_negative_scores)

    max_score = 'N/A'
    min_score = 'N/A'
    avg_score = 'N/A'
    non_negative_score_count = len(non_negative_scores) # 计算非负分数的数量

    if non_negative_scores:
        max_score = max(non_negative_scores)
        min_score = min(non_negative_scores)
        avg_score = sum(non_negative_scores) / len(non_negative_scores)

    # 4. 准备写入 CSV 的数据行，加入“非负分数量”
    data_row = [
        max_score,
        min_score,
        f"{avg_score:.4f}" if isinstance(avg_score, float) else avg_score,
        non_negative_score_count,  # 非负分数量
        negative_score_count,      # 负分数量
        total_samples,             # 总样本数（去重前）
        unique_sample_count        # 去重后样本数
    ]

    # 5. 将结果写入 CSV 文件
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header) # 写入表头
            writer.writerow(data_row)   # 写入数据行
            
        print("\n" + "="*20)
        print(f"CSV 报告已成功生成: {output_csv_path}")
        print("="*20 + "\n")
        
        print("--- CSV 内容预览 ---")
        print(','.join(map(str, csv_header)))
        print(','.join(map(str, data_row)))

    except Exception as e:
        print(f"错误: 无法将报告写入 CSV 文件 {output_csv_path}: {e}")

# --- 主程序执行部分 ---
if __name__ == "__main__":
    # 请将此路径替换为您的实际目录路径
    directory_path = "<EXPERIMENT_ROOT>"
    exp_name = "augvis_rollout_cot_Stepwise_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppuhigh_Datasetph_xl_hz_1112_test_webarena_RewardModel0912"
    #
    #exp_name = "Qwen2.5-VL-72B_online_rl_global_step_120_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_ali_test_webarena_RewardModel0912"
    #exp_name = "Qwen2.5-VL-72B_online_rl_global_step_200_wPRALL3_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_ali_test_webarena_RewardModel0912"
    
    directory_path = "<EXPERIMENT_ROOT>"
    exp_name = "Qwen2.5-VL-72B-Instruct_wPRALL3_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasetwa_ali_test_webarena_RewardModel0912"
    #exp_name = ""
    #exp_name = "webagent_online"
    directory_path = f"{directory_path}/{exp_name}"
    if not os.path.isdir(directory_path):
        print(f"错误：指定的目录不存在 -> {directory_path}")
    else:
        analyze_and_generate_csv(directory_path)

