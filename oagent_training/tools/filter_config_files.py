import os
import json
import shutil

def filter_and_copy_configs(source_dir, dest_dir_name, excluded_domains):
    """
    读取指定目录下的JSON配置文件，过滤掉start_url包含排除列表中的任何域名的文件，
    并将符合条件的文件复制到新的目标目录。

    Args:
        source_dir (str): 包含JSON配置文件的源目录路径。
        dest_dir_name (str): 用于存放过滤后文件的目标目录名称。
        excluded_domains (list): 一个包含要排除的域名字符串的列表。
    """
    # 基于源目录的父目录来创建目标目录的完整路径
    parent_dir = os.path.dirname(source_dir)
    dest_dir = os.path.join(parent_dir, dest_dir_name)

    # 1. 如果目标目录不存在，则创建它
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"成功创建目录: '{dest_dir}'")

    # 用于统计的计数器
    total_files = 0
    copied_files = 0
    skipped_files = 0

    print(f"\n开始从 '{source_dir}' 读取文件...")
    print(f"将要排除的域名: {excluded_domains}")

    # 2. 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        # 只处理.json文件
        if filename.endswith(".json"):
            total_files += 1
            source_file_path = os.path.join(source_dir, filename)
            
            try:
                # 打开并解析JSON文件
                with open(source_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 3. 检查 'start_url' 字段
                start_url = data.get("start_url")

                # 核心改动：检查start_url是否包含排除列表中的任何一个域名
                # a) `start_url and ...` 确保start_url不为None或空字符串
                # b) `any(...)` 会在找到第一个匹配项时立即返回True，非常高效
                is_excluded = start_url and any(domain in start_url for domain in excluded_domains)

                if is_excluded:
                    # 为了更清晰地输出，找到具体是哪个域名匹配了
                    matched_domain = next((domain for domain in excluded_domains if domain in start_url), "未知")
                    print(f"[-] 跳过文件: {filename} (URL: {start_url} - 匹配到排除项: '{matched_domain}')")
                    skipped_files += 1
                else:
                    # 4. 如果不包含任何排除的域名，则复制文件到目标目录
                    dest_file_path = os.path.join(dest_dir, filename)
                    shutil.copy2(source_file_path, dest_file_path)
                    # print(f"[+] 复制文件: {filename}") # 可以取消注释以查看每个复制的文件
                    copied_files += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[!] 处理文件 {filename} 时出错: {e}。已跳过。")
                skipped_files += 1
            except Exception as e:
                print(f"[!] 发生未知错误处理 {filename}: {e}。已跳过。")
                skipped_files += 1

    # 打印最终的统计结果
    print("\n--- 任务完成 ---")
    print(f"总共处理JSON文件数: {total_files}")
    print(f"成功复制的文件数: {copied_files}")
    print(f"跳过的文件数 (因匹配排除项或错误): {skipped_files}")
    print(f"过滤后的文件已保存至: '{dest_dir}'")


# --- 使用方法 ---
if __name__ == "__main__":
    # ====================== 在这里修改您要排除的网站列表 ======================
    # 您可以随时在这里增加或删除域名，例如 ["douban.com", "weibo.com"]
    excluded_sites = [
        "zhihu.com",
    ]
    # ========================================================================

    # 源目录（根据您的终端信息）
    source_directory = "<SOURCE_CONFIG_DIR>"
    
    # 新目录的名称
    destination_directory_name = "gemini_gen_intent_task_v4_cookied_auth_filter"

    # 调用函数，将排除列表作为参数传入
    filter_and_copy_configs(source_directory, destination_directory_name, excluded_sites)
