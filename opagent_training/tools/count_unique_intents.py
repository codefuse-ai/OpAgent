#!/usr/bin/env python3
"""
统计配置文件目录下不同intent的数量
"""

import json
import os
from pathlib import Path
from collections import Counter
import argparse


def count_intents(config_dir):
    """统计指定目录下所有配置文件中的不同intent"""
    config_path = Path(config_dir)
    
    if not config_path.exists():
        print(f"错误: 目录不存在: {config_dir}")
        return
    
    intents = []
    intent_files = {}
    skipped_files = []
    
    # 遍历所有JSON文件
    json_files = list(config_path.glob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in sorted(json_files, key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf')):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                intent = data.get('intent', '')
                if intent:
                    intents.append(intent)
                    if intent not in intent_files:
                        intent_files[intent] = []
                    intent_files[intent].append(json_file.name)
                else:
                    skipped_files.append(json_file.name)
        except Exception as e:
            print(f'警告: 无法读取文件 {json_file}: {e}')
            skipped_files.append(json_file.name)
    
    # 统计
    total_files = len(json_files)
    unique_intents = len(set(intents))
    intent_counts = Counter(intents)
    
    print('\n' + '=' * 80)
    print('统计结果')
    print('=' * 80)
    print(f'总文件数: {total_files}')
    print(f'成功处理: {len(intents)} 个文件')
    print(f'跳过/错误: {len(skipped_files)} 个文件')
    print(f'总intent数（包含重复）: {len(intents)}')
    print(f'不同的intent数: {unique_intents}')
    print('=' * 80)
    
    # 显示重复的intent
    duplicates = [(intent, count) for intent, count in intent_counts.items() if count > 1]
    duplicates.sort(key=lambda x: x[1], reverse=True)
    
    if duplicates:
        print(f'\nIntent重复情况（共 {len(duplicates)} 个intent有重复）:')
        print('=' * 80)
        
        # 显示前20个重复最多的intent
        for i, (intent, count) in enumerate(duplicates[:20], 1):
            print(f'\n{i}. 出现 {count} 次:')
            print(f'   Intent: {intent}')
            files_str = ', '.join(intent_files[intent][:10])
            if len(intent_files[intent]) > 10:
                files_str += ' ...'
            print(f'   文件: {files_str}')
    else:
        print('\n所有intent都是唯一的！')
    
    # 统计唯一intent（只出现一次的）
    unique_once = sum(1 for count in intent_counts.values() if count == 1)
    print(f'\n只出现一次的intent数: {unique_once}')
    print(f'出现多次的intent数: {unique_intents - unique_once}')
    
    # 保存详细报告到JSON文件
    output_file = config_path / "intent_statistics.json"
    report = {
        'summary': {
            'total_files': total_files,
            'processed_files': len(intents),
            'skipped_files': len(skipped_files),
            'total_intents': len(intents),
            'unique_intents': unique_intents,
            'unique_once': unique_once,
            'duplicate_intents': unique_intents - unique_once
        },
        'intent_counts': dict(intent_counts),
        'intent_files': intent_files,
        'skipped_files': skipped_files
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='统计配置文件目录下不同intent的数量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python count_unique_intents.py /path/to/config_dir
  python count_unique_intents.py --dir /path/to/config_dir
        """
    )
    
    parser.add_argument(
        'config_dir',
        nargs='?',
        default='<DATASET_DIR>',
        help='配置文件目录的路径（默认为预设路径）'
    )
    
    parser.add_argument(
        '--dir',
        dest='config_dir_alt',
        help='配置文件目录的路径（替代参数）'
    )
    
    args = parser.parse_args()
    
    # 使用--dir参数（如果提供）或位置参数
    config_dir = args.config_dir_alt if args.config_dir_alt else args.config_dir
    
    print("=" * 80)
    print("Intent统计工具")
    print("=" * 80)
    print(f"分析目录: {config_dir}")
    print("=" * 80)
    
    count_intents(config_dir)


if __name__ == "__main__":
    main()
