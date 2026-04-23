import os
import json
import glob

def process_and_extract_data(directory_path: str):
    """
    遍历指定目录下的所有JSON文件，提取数据并报告异常。

    提取逻辑:
    1. 优先获取 `eval` -> `reference_answer_raw_annotation` 的值。
    2. 如果上述键不存在，则回退获取 `eval` -> `reference_url` 的值。
    3. 如果两者都不存在或文件处理出错，则记录为异常样本。

    Args:
        directory_path: 包含JSON文件的目录路径。
    """
    if not os.path.isdir(directory_path):
        print(f"错误：目录不存在 '{directory_path}'")
        return

    search_pattern = os.path.join(directory_path, '*.json')
    json_files = glob.glob(search_pattern)

    if not json_files:
        print(f"在目录 '{directory_path}' 中没有找到任何JSON文件。")
        return

    print(f"找到 {len(json_files)} 个JSON文件。开始处理...")
    print("-" * 50)
    
    # 用于收集所有异常样本的信息
    exception_samples = []

    # --- 开始遍历文件 ---
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 安全地获取 'eval' 字典
            eval_dict = data.get('eval')
            if not isinstance(eval_dict, dict):
                # 如果 'eval' 键不存在或不是一个字典，记录为异常
                raise KeyError("未找到 'eval' 键或其格式不正确")

            # 优先获取 'reference_answer_raw_annotation'
            value_to_print = eval_dict.get('reference_answers')
            if "fuzzy_match" in value_to_print:
                print("fuzzy_match: ", file_path)
            # 如果主目标不存在，则回退到 'reference_url'
            if value_to_print is None:
                value_to_print = eval_dict.get('reference_url')

            # 如果回退后仍然没有值，也记录为异常
            # if value_to_print is None:
            #     raise ValueError("主键和回退键 ('reference_answer_raw_annotation', 'reference_url') 均未找到")
            
            # 成功提取到值，打印它
            print(value_to_print)

        except Exception as e:
            # 捕获所有可能的错误（文件打不开、JSON解析失败、键不存在等）
            # 将文件名和错误信息作为一个元组添加到异常列表中
            exception_samples.append((file_path, str(e)))


    # --- 循环结束后，打印所有收集到的异常样本 ---
    print("-" * 50)
    if exception_samples:
        print(f"\n发现 {len(exception_samples)} 个异常样本：")
        for file_name, error_message in exception_samples:
            print(f"  - 文件: {file_name}, 原因: {error_message}")
    else:
        print("\n所有文件处理成功，未发现异常样本。")


if __name__ == "__main__":
    # 在这里设置你的目标目录
    target_dir = "<TARGET_DIR>"
    process_and_extract_data(target_dir)

