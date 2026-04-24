import os
import json
from pathlib import Path
from typing import List
def check_all_json_for_key(directory_path: str) -> None:
    """
    检查指定目录下的所有JSON文件是否都包含一个特定的嵌套键。

    Args:
        directory_path (str): 要检查的目录路径。
    """
    target_dir = Path(directory_path)

    # 1. 检查目录是否存在
    if not target_dir.is_dir():
        print(f"错误：目录不存在 -> {directory_path}")
        return

    # 2. 获取目录下所有的.json文件
    json_files = list(target_dir.glob('*.json'))

    if not json_files:
        print(f"信息：在目录 '{directory_path}' 中没有找到任何 .json 文件。")
        return

    print(f"开始检查目录: {directory_path}")
    print(f"共找到 {len(json_files)} 个 .json 文件进行分析...\n")

    # 3. 准备一个列表来存储缺少关键字的文件
    files_missing_key = []
    
    # 4. 遍历所有JSON文件
    for file_path in json_files:
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 5. 安全地检查嵌套的关键字
                # 使用 try-except 块来处理任何层级的KeyError或TypeError
                try:
                    _ = data['eval']['reference_url']
                except Exception as e:
                    # 如果 'config_dict' 或 'eval' 不存在，或它们不是字典类型，
                    # 或者 'reference_url' 不存在，都会触发异常。
                    print(f"{e}")
                    files_missing_key.append(file_path.name)

        except json.JSONDecodeError:
            print(f"警告：文件 '{file_path.name}' 不是一个有效的JSON文件，已跳过。")
        except Exception as e:
            print(f"处理文件 '{file_path.name}' 时发生未知错误: {e}")
            files_missing_key.append(file_path.name) # 将出错文件也视为不合格

    # 6. 输出最终结论
    print("-" * 30)
    print("检查完成！")
    
    if not files_missing_key:
        print("\n结论：✅ 是的，所有样本都包含 'config_dict['eval']['reference_url']' 关键字。")
    else:
        print(f"\n结论：❌ 不是，有 {len(files_missing_key)} 个样本缺少 'config_dict['eval']['reference_url']' 关键字。")
        print("\n缺少该关键字的文件列表如下：")
        for filename in files_missing_key:
            print(f"- {filename}")
    print(f"len(files_missing_key) {len(files_missing_key)}")
# --- 主程序 ---

def find_intents_with_image(directory_path: str) -> List[str]:
    """
    遍历指定目录下的所有JSON文件，找出'intent'字段包含'<image>'字符串的文件。

    Args:
        directory_path (str): 要搜索的目录的路径。

    Returns:
        List[str]: 一个列表，包含所有满足条件的JSON文件名。
    """
    # 将字符串路径转换为更易于操作的 Path 对象
    p = Path(directory_path)

    # 检查路径是否存在且是否为目录
    if not p.is_dir():
        print(f"错误：目录 '{directory_path}' 不存在或不是一个有效的目录。")
        return []

    files_with_image_intent = []
    
    # 使用 glob('*.json') 高效地遍历目录下所有 .json 文件
    print(f"开始在目录 '{directory_path}' 中搜索...")
    for json_file in p.glob('*.json'):
        try:
            # 使用 'with' 语句安全地打开文件，并指定 utf-8 编码
            with json_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 使用 .get() 方法安全地获取 'intent' 字段
                # 如果 'intent' 键不存在，.get()会返回 None（或指定的默认值），避免了 KeyError
                intent_text = data.get("intent")
                
                # 检查 intent_text 是否存在且包含 '<image>'
                if "image" in intent_text:
                    # 如果满足条件，将文件名（不含路径）添加到列表中
                    files_with_image_intent.append(json_file.name)

        except json.JSONDecodeError:
            print(f"警告：文件 '{json_file.name}' 不是有效的JSON格式，已跳过。")
        except Exception as e:
            print(f"处理文件 '{json_file.name}' 时发生未知错误: {e}")
    print(f"files_with_image_intent {files_with_image_intent}")
    print(f"len(files_with_image_intent) {len(files_with_image_intent)}")   
    return files_with_image_intent

if __name__ == "__main__":
    # 指定要检查的目录
    # TARGET_DIRECTORY = '<TARGET_DIRECTORY_EXAMPLE>'
    
    # # 运行检查函数
    # check_all_json_for_key(TARGET_DIRECTORY)
    input_directory = '<INPUT_DIRECTORY>'
    find_intents_with_image(input_directory)

