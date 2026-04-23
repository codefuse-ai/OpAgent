import json
import os
import random
from collections import defaultdict

def sample_by_complexity_custom_order(directory_path, sample_size=10):
    """
    扫描指定目录下的所有 .json 文件，按 'complexity' 类别分组，
    并从每个类别中随机抽取指定数量的样本。
    最后按照 ['easy', 'medium', 'hard'] 的指定顺序，打印出样本的
    完整路径、行号、intent 和 start_url。

    Args:
        directory_path (str): 要扫描的目录路径。
        sample_size (int): 每个类别要抽取的样本数量。
    """
    # 1. 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误: 目录不存在 -> '{directory_path}'")
        return

    print(f"--- 开始扫描目录: {directory_path} ---")

    # 2. 按类别收集所有记录信息
    categorized_records = defaultdict(list)
    total_lines_processed = 0
    total_errors = 0

    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    if not json_files:
        print("未找到任何 .json 文件。")
        return

    # 3. 遍历所有文件和行，收集详细数据
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines_processed += 1
                    clean_line = line.strip()
                    if not clean_line:
                        continue

                    try:
                        data = json.loads(clean_line)
                        complexity = data.get('labels', {}).get('complexity')
                        intent = data.get('intent', 'N/A')
                        start_url = data.get('start_url', 'N/A')

                        if complexity:
                            record_info = {
                                "file_path": file_path,
                                "line_num": line_num,
                                "intent": intent,
                                "start_url": start_url
                            }
                            categorized_records[complexity].append(record_info)
                        else:
                            total_errors += 1
                    except json.JSONDecodeError:
                        total_errors += 1
        except Exception as e:
            print(f"错误: 读取文件 {filename} 时发生严重错误: {e}")

    print(f"--- 数据收集完成 ---")
    print(f"总计处理行数: {total_lines_processed}")
    print(f"格式错误或缺少键的行数: {total_errors}")
    
    found_categories = list(categorized_records.keys())
    print(f"发现的类别: {found_categories}")

    # 4. 按照指定顺序进行抽样和打印
    print("\n" + "="*20 + " 随机抽样结果 (按指定顺序) " + "="*20)

    # --- 这里是核心改动 ---
    # 定义期望的输出顺序
    output_order = ['easy', 'medium', 'hard']
    
    # 获取所有在数据中找到的类别
    found_categories_set = set(found_categories)
    
    # 找出不在指定顺序中的“其他”类别，并对它们进行排序以保证输出一致
    other_categories = sorted(list(found_categories_set - set(output_order)))
    
    # 最终的打印顺序 = 指定顺序 + 其他类别
    final_processing_order = output_order + other_categories
    # --- 改动结束 ---

    for category in final_processing_order:
        # 只处理在数据中实际存在的类别
        if category not in categorized_records:
            continue

        records = categorized_records[category]
        num_records = len(records)
        
        print(f"\n[ 类别: {category} ] (共 {num_records} 条)")

        actual_sample_size = min(num_records, sample_size)

        if actual_sample_size == 0:
            print("  -> 该类别下没有可抽样的样本。")
            continue
            
        print(f"  -> 随机抽取 {actual_sample_size} 条样本的详细信息:")

        selected_samples = random.sample(records, actual_sample_size)

        for i, sample in enumerate(selected_samples, 1):
            print("-" * 60)
            print(f"  样本 {i}:")
            print(f"    文件路径: {sample['file_path']}")
            print(f"    行号:      {sample['line_num']}")
            print(f"    Start URL: {sample['start_url']}")
            print(f"    Intent:    {sample['intent']}")
        print("-" * 60)

if __name__ == "__main__":
    # 目标目录
    target_directory = '<DATASET_DIR>'
    
    # 每个类别抽取的样本数量
    samples_to_draw = 20
    
    sample_by_complexity_custom_order(target_directory, sample_size=samples_to_draw)
