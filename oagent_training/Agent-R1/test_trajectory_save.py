#!/usr/bin/env python3
"""
测试轨迹保存功能的脚本
"""

import os
import sys

# 设置环境变量
os.environ['TRAJECTORY_SAVE_FREQ'] = '15'
os.environ['TRAJECTORY_SAVE_ENABLED'] = 'true'
os.environ['SAVE_MODEL_PATH'] = '/tmp/test_trajectory_save'

def test_environment_variables():
    """测试环境变量设置"""
    print("=== 测试环境变量设置 ===")
    print(f"TRAJECTORY_SAVE_FREQ: {os.environ.get('TRAJECTORY_SAVE_FREQ', '20')}")
    print(f"TRAJECTORY_SAVE_ENABLED: {os.environ.get('TRAJECTORY_SAVE_ENABLED', 'false')}")
    print(f"SAVE_MODEL_PATH: {os.environ.get('SAVE_MODEL_PATH', '/tmp')}")
    print()

def test_step_calculation():
    """测试step计算逻辑"""
    print("=== 测试step计算逻辑 ===")
    freq = int(os.environ.get('TRAJECTORY_SAVE_FREQ', '20'))
    enabled = os.environ.get('TRAJECTORY_SAVE_ENABLED', 'true').lower() == 'true'
    
    for step in range(1, 25):
        should_save = (enabled and 
                      step > 0 and 
                      step % freq == 0)
        print(f"Step {step:2d}: {'保存' if should_save else '跳过'} (频率={freq})")
    print()

def test_filename_generation():
    """测试文件名生成"""
    print("=== 测试文件名生成 ===")
    from datetime import datetime
    
    for step in [15, 30, 45, 60]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step_{step:06d}"
        task_id = "test_task_123"
        task_dir_name = f"{step_str}_{task_id}_{timestamp}"
        print(f"Step {step:2d}: {task_dir_name}")
    print()

if __name__ == "__main__":
    test_environment_variables()
    test_step_calculation()
    test_filename_generation()
    
    print("=== 使用说明 ===")
    print("1. 设置保存频率:")
    print("   export TRAJECTORY_SAVE_FREQ=15  # 每15步保存一次")
    print("   export TRAJECTORY_SAVE_FREQ=20  # 每20步保存一次")
    print()
    print("2. 启用/禁用保存:")
    print("   export TRAJECTORY_SAVE_ENABLED=true   # 启用保存")
    print("   export TRAJECTORY_SAVE_ENABLED=false  # 禁用保存")
    print()
    print("3. 运行训练:")
    print("   bash ./scripts/visual_webarena/run_grpo_debug.sh")
    print()
    print("4. 轨迹数据将保存到:")
    print("   ${SAVE_MODEL_PATH}/trajectory_data/")
    print("   文件名格式: step_XXXXXX_taskid_timestamp/")
