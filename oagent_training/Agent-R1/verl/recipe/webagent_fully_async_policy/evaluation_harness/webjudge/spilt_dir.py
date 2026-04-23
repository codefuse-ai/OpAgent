import os
import shutil
import json
from tqdm import tqdm  # 需要安装 tqdm 库：pip install tqdm


def split_folders(src_dir, target_dir1, target_dir2, target_dir3):
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir1, exist_ok=True)
    os.makedirs(target_dir2, exist_ok=True)
    os.makedirs(target_dir3, exist_ok=True)

    output_json_path = os.environ.get("WEBJUDGE_RESULT_PATH", "./data/webjudge/example_25_result/WebJudge_Online_Mind2Web_eval_Qwen2.5-VL-72B-Instruct_score_threshold_1_auto_eval_results.json")
    already_ids = []
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            already_data = f.read()
        already_tasks = already_data.splitlines()
        for item in already_tasks:
            try:
                item = json.loads(item)
                already_ids.append(item["task_id"])
            except Exception as e:
                print("error item:", item)

    print(f"The number of already done tasks: {len(already_ids)}")

    # 获取所有子文件夹（排除隐藏目录）
    subfolders = [
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, f)) and not f.startswith('.') and f not in already_ids
    ]

    total = len(subfolders)
    half = total // 3

    print(f"总文件夹数: {total}")
    print(f"拆分点: 前 {half} 个到 {target_dir1},  {half} 个到 {target_dir2}, 剩余{total - 2 * half} 到  {target_dir3}")

    # 移动前半部分
    for path in tqdm(subfolders[:half], desc="Moving to target_dir1"):
        try:
            shutil.move(path, target_dir1)
        except Exception as e:
            print(f"移动失败 {path}: {e}")

    # 移动后半部分
    for path in tqdm(subfolders[half: 2 * half], desc="Moving to target_dir2"):
        try:
            shutil.move(path, target_dir2)
        except Exception as e:
            print(f"移动失败 {path}: {e}")

    # 移动后半部分
    for path in tqdm(subfolders[2 * half:], desc="Moving to target_dir3"):
        try:
            shutil.move(path, target_dir3)
        except Exception as e:
            print(f"移动失败 {path}: {e}")

    print("拆分完成！")


if __name__ == "__main__":
    # 替换为你的实际路径
    original_dir = os.environ.get("WEBJUDGE_SPLIT_SRC", "./data/webjudge/example_25")
    target_dir1 = os.environ.get("WEBJUDGE_SPLIT_DST1", "./data/webjudge/example_26")
    target_dir2 = os.environ.get("WEBJUDGE_SPLIT_DST2", "./data/webjudge/example_27")
    target_dir3 = os.environ.get("WEBJUDGE_SPLIT_DST3", "./data/webjudge/example_28")

    split_folders(original_dir, target_dir1, target_dir2, target_dir3)
