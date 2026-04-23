import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from datetime import datetime
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
methods_dir = os.path.join(current_dir, 'methods')
sys.path.append(current_dir + '/webjudge/src')
sys.path.append(current_dir + '/webjudge/src/methods')
from src import vllm_run
import asyncio

# --- 1. 定义你的模型评估函数 ---
# 这是你需要根据你的实际模型替换的部分。
# 它应该接受一个样本作为输入，并返回 True (正确) 或 False (错误)。

def evaluate_with_my_model(sample: dict) -> bool:
    """
    一个模拟的模型评估函数。
    
    真实场景下，这里会包含调用你的模型API、运行本地模型等的代码。
    为了演示，我们创建一个有一定概率出错的模拟模型：
    - 对于包含“正确”的句子，有 90% 的概率返回 True。
    - 对于其他句子，有 80% 的概率返回 False。
    """
    if 'task' in sample:
        if type(sample['task']) == list:
            sample['task'] = sample['task'][0]
    
    output_result = asyncio.run(vllm_run.simple_eval("", sample))
    with open("data/evaluate_results/details_v1.jsonl", 'a') as f:
        f.write(json.dumps(output_result, ensure_ascii=False) + '\n')
    print(output_result)
    score = float(output_result['predicted_label'])
    if score > 0.5:
        return True
    else:
        return False

# -------------------------------------


def load_samples_from_jsonl(file_path: str) -> list:
    """从 JSONL 文件加载数据样本。"""
    samples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return []

def run_evaluation(positive_file: str, negative_file: str, model_func, output_file: str):
    """
    运行完整的评估流程，并将报告打印到控制台和指定的输出文件。
    """
    # 使用 'w' 模式打开文件，如果文件已存在则会覆盖
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            
            def log_and_print(message=""):
                """一个辅助函数，同时打印到控制台和写入文件。"""
                print(message)
                f.write(message + '\n')

            # 1. 加载数据集
            positive_samples = load_samples_from_jsonl(positive_file)
            negative_samples = load_samples_from_jsonl(negative_file)

            if not positive_samples and not negative_samples:
                log_and_print("错误: 两个样本集都为空，无法进行评估。")
                return

            # 真实标签
            y_true = [1] * len(positive_samples) + [0] * len(negative_samples)
            all_samples = positive_samples + negative_samples
            
            # 2. 获取模型预测结果
            print(f"正在评估 {len(all_samples)} 个样本...")
            y_pred = [1 if model_func(sample) else 0 for sample in all_samples]
            print("评估完成。")
            
            # 3. 计算评估指标
            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            except ValueError: # 如果所有预测都一样，ravel()可能会失败
                # 简单处理，更复杂场景需要更细致的逻辑
                cm = confusion_matrix(y_true, y_pred)
                if len(cm) == 1: # 只有一个类别被预测
                    if y_true[0] == 0: # 全是负类
                        tn, fp, fn, tp = cm[0][0], 0, 0, 0
                    else: # 全是正类
                        tn, fp, fn, tp = 0, 0, 0, cm[0][0]
                else: # 这种情况不应该发生，但作为保险
                    tn, fp, fn, tp = 0,0,0,0

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

            # 4. 生成并输出结果报告
            log_and_print("\n" + "="*40)
            log_and_print(f"      模型评估报告")
            log_and_print(f"      评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            log_and_print("="*40)
            log_and_print(f"正样本文件: '{positive_file}' ({len(positive_samples)} 个)")
            log_and_print(f"负样本文件: '{negative_file}' ({len(negative_samples)} 个)")
            log_and_print(f"总样本数量: {len(all_samples)}")
            log_and_print("-" * 40)
            
            # 混淆矩阵
            conf_matrix_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=['真实: 错误 (N)', '真实: 正确 (P)'],
                columns=['预测: 错误 (N)', '预测: 正确 (P)']
            )
            log_and_print("混淆矩阵:")
            # 将DataFrame转换为字符串以便写入文件
            conf_matrix_str = conf_matrix_df.to_string()
            log_and_print(conf_matrix_str)
            log_and_print("-" * 40)
            
            log_and_print("核心指标:")
            log_and_print(f"  准确率 (Accuracy):   {accuracy:.2%}")
            log_and_print(f"  精确率 (Precision):  {precision:.2%}")
            log_and_print(f"  召回率 (Recall/TPR): {recall:.2%}")
            log_and_print(f"  F1 分数 (F1-Score):  {f1:.2f}")
            log_and_print(f"  真阴性率 (TNR):      {true_negative_rate:.2%}")
            log_and_print("="*40 + "\n")
            
            log_and_print("指标解释:")
            log_and_print(f"  - 精确率: 在所有被模型预测为“正确”的样本中，有 {precision:.0%} 确实是正确的。")
            log_and_print(f"  - 召回率: 在所有真实为“正确”的样本中，模型成功找出了 {recall:.0%}。")
            log_and_print(f"  - 真阴性率: 在所有真实为“错误”的样本中，模型成功拒绝了 {true_negative_rate:.0%}。")

        print(f"\n评估报告已成功保存到文件: '{output_file}'")

    except IOError as e:
        print(f"错误: 无法写入文件 '{output_file}'. 原因: {e}")
    # except Exception as e:
    #     print(f"评估过程中发生未知错误: {e}")


if __name__ == "__main__":
    # python -m eval.evaluate_model
    # 定义你的样本文件路径
    POSITIVE_SAMPLES_FILE = "data/eval_data_pos_v1.jsonl"
    NEGATIVE_SAMPLES_FILE = "data/eval_data_neg_v1.jsonl"
    
    # 定义报告输出文件路径
    REPORT_OUTPUT_FILE = "data/evaluate_results/evaluation_report_v1.txt"
    
    # 运行评估
    run_evaluation(
        positive_file=POSITIVE_SAMPLES_FILE,
        negative_file=NEGATIVE_SAMPLES_FILE,
        model_func=evaluate_with_my_model,
        output_file=REPORT_OUTPUT_FILE
    )

# if __name__ == "__main__":
#     # python -m recipe.webagent_fully_async_policy.evaluation_harness.webjudge.eval.evaluate_model
#     # 定义你的样本文件路径
#     POSITIVE_SAMPLES_FILE = "agent_r1/tool/tools/evaluation_harness/webjudge/data/eval_data_pos.jsonl"
#     NEGATIVE_SAMPLES_FILE = "agent_r1/tool/tools/evaluation_harness/webjudge/data/eval_data_neg.jsonl"
    
#     # 运行评估
#     run_evaluation(
#         positive_file=POSITIVE_SAMPLES_FILE,
#         negative_file=NEGATIVE_SAMPLES_FILE,
#         model_func=evaluate_with_my_model # 传入你的模型评估函数
#     )
