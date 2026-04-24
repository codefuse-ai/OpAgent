import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import random
from collections import defaultdict
import pypandoc

def process_single_trajectory(traj_dir):
    """处理单个轨迹目录的函数，用于多进程调用"""
    try:
        json_file_path = traj_dir / "evaluation_data.json"
        if not json_file_path.exists():
            return traj_dir.name, 0, f"未找到 evaluation_data.json"
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        task_id = data.get("task_id", "")
        task = data.get("intent", "")
        final_result_response = data.get("final_answer", {}).get('answer', '')
        
        action_history = []
        thoughts = []
        input_image_paths = []
        trajectory = data.get("trajectory_with_rewards", [])
        current_step = None

        # thoughts.append(trajectory[0].get("raw_prediction", "").split('\n</think>')[0].split('<think>\n')[-1])
        # ele_map = {}
        # # ele_map['index'] = previous_step['element_id']
        # if 'action' in trajectory[0]:
        #     action = trajectory[0]['action']
        #     ele_map['text'] = ''.join([text for text in action['text']])
        #     ele_map['url'] = action['url']
        #     ele_map['x'] = float(action['coords'][0])
        #     ele_map['y'] = float(action['coords'][1])
        #     action_str = json.dumps(ele_map, ensure_ascii=False) + ' -> ' + (str(action['action_type']))
        #     action_history.append(action_str)

        for i in range(1, len(trajectory)):
            current_step, previous_step = trajectory[i], trajectory[i-1]

            if current_step.get("type") == "action" and previous_step.get("type") == "observation":
                coords = current_step.get("action", {}).get("coords")
                
                if coords and len(coords) == 2:
                    relative_image_path = previous_step.get("image_path")
                    if not relative_image_path: continue

                    original_image_path = traj_dir / relative_image_path
                    if not original_image_path.exists(): continue
                    input_image_paths.append(str(original_image_path.resolve()))
                    thoughts.append(current_step.get("raw_prediction", "").split('</think>')[0].split('<think>')[-1])
           
                    ele_map = {}
                    # ele_map['index'] = previous_step['element_id']
                    if 'action' in current_step:
                        action = current_step['action']
                        # ele_map['text'] = ''.join([text for text in action['text']])
                        ele_map['url'] = action['url']
                        # ele_map['x'] = float(action['coords'][0])
                        # ele_map['y'] = float(action['coords'][1])
                        action_str = json.dumps(ele_map, ensure_ascii=False) + ' -> ' + (str(action['action_type']))
                        action_history.append(action_str)
        if current_step:
            last_base_name = relative_image_path.split("/")[-1]
            step_ind = int(last_base_name.split("_")[1]) + 2
            img_ind = int(last_base_name.split("_")[3].split(".")[0]) + 1
            last_original_image_path = traj_dir / f"images/step_{step_ind}_img_{img_ind}.png"
            input_image_paths.append(str(last_original_image_path.resolve()))

        human_score = 0.0

        return json.dumps({"task_id": task_id, "task": task, "final_result_response": final_result_response, "action_history": action_history, "thoughts": thoughts, "input_image_paths":input_image_paths, "human_score":human_score}, ensure_ascii=False)
    
    except Exception as e:
        print(e)
        return ""

def procsess_trajectories(root_dir: str):
    results = []
    """遍历根目录，使用多进程并发标注图片。"""
    root_path = Path(root_dir)
    trajectory_data_path = root_path / "trajectory_data"

    if not trajectory_data_path.is_dir():
        print(f"错误: 在 '{root_dir}' 中未找到 'trajectory_data' 目录。")
        return

    trajectory_dirs = [d for d in trajectory_data_path.iterdir() if d.is_dir()]
    if not trajectory_dirs:
        print(f"在 '{trajectory_data_path}' 中未找到任何轨迹目录。")
        return

    print(f"找到 {len(trajectory_dirs)} 个轨迹目录，开始处理.")
    for d in trajectory_dirs:
        ret = process_single_trajectory(d)
        results.append(ret)
    
    total_processed = sum(r[1] for r in results)
    print(f"\n图片标注完成！总共标注了 {total_processed} 张图片。\n")

if __name__ == '__main__':
    dataset_path = "<EXPERIMENTS_ROOT>"
    #model_name = "Qwen2.5-VL-7B-Instruct_JudgewoAnswer_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_VLM_EXP_NAMEWithStepScore_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-7B-Instruct_JudgewoAnswer_EC0_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_VLM_EXP_NAMEWithStepScore_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-7B-Instruct_JudgewoAnswer_woKLInReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_VLM_EXP_NAMEWithStepScore_ProcessRewaedFalse_catid"
    model_name = "Qwen2.5-VL-7B-Instruct_GSPO_JudgewoAnswer_EC0.001_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_gemini_gen_intent_task_v3_ProcessRewaedFalse_catid"
    # 构造模型路径
    model_path = os.path.join(dataset_path, model_name)
    # procsess_trajectories(model_path, )
    pos_trajectories = [
        'step_000030_c4b899eb-a15b-42f7-8ae8-2a964d045930_20250910_104610',
        'step_000030_695fee76-3b19-4147-93f5-1af83a0840a6_20250910_104511',
        'step_000030_8a89a49d-7fce-4272-b726-d608dc285f95_20250910_104431',
        'step_000020_d87433ca-e4f1-4645-96b3-8d6d4d6f2230_20250910_063908', #最新资讯
        'step_000020_ff9b9ffe-9870-4f5f-bd7e-43cc7ffb9e5b_20250910_063910',
        'step_000010_23c88d6a-1a36-4d4e-a99f-e1ba3cd351f2_20250910_023350',
        'step_000010_ad8beaee-016d-4e54-93d2-e6c62f671d7f_20250910_023510',
        'step_000010_d4a7977c-72ae-48fd-af89-29270bd0a5b6_20250910_023505'
    ]
    neg_trajectories = [
        'step_000040_cd680c06-98f1-4218-a8a8-6334dd3c6b86_20250910_151908',
        'step_000040_56690d4f-0982-4b97-8ec6-d7896314377e_20250910_152010', # 播放量最高
        'step_000040_de534060-f1c1-4ab4-a859-4a197c41c5ce_20250910_152107', # 筛选黑进搜索词
        'step_000040_8f45fd8c-5cf9-45ff-a724-1d4711b104e8_20250910_152207',
        'step_000030_a0808a78-3f7f-4166-9055-e6dc1733a156_20250910_104507',
        'step_000030_bb3bdf36-265d-4b21-9223-7b8e4da67056_20250910_104606',
        'step_000030_79452c5a-3336-4bc8-ab58-bafba1c6ed0b_20250910_104432',
        'step_000030_83eb1f89-d5bb-4af6-8727-4d06cd2ef4d9_20250910_104712',  # 没法排序，次优
        'step_000030_f9c2f5c1-5970-4170-ad15-853b7139cebd_20250910_104506',
        'step_000020_b8bf4741-e4f3-499f-9223-1705981656a9_20250910_063819', # 本田思域论坛
        'step_000020_4bebbb14-a703-4b27-b3ed-efd114e1c0a3_20250910_063905', # 知乎黑进搜索
        'step_000020_5c2f59bb-4091-4bf8-9ed7-dbf69fab0f31_20250910_063819',
        'step_000010_a0038429-2cfb-4bd6-981a-1657df450103_20250910_023516', # 黑进搜索
        'step_000010_8d2b9d26-50a1-442c-9dd1-515f2b08c48c_20250910_023510', # 找工作找到了杭州
        'step_000010_f5ac5fa3-cbe9-4ce9-b7c3-cef53404e591_20250910_023515'
    ]
    with open('<WEBJUDGE_POS_OUTPUT_JSONL>', 'w') as f:
        for pt in pos_trajectories:
            d = process_single_trajectory(Path(os.path.join(model_path,f"trajectory_data/{pt}")))
            f.write(d + '\n')
    with open('<WEBJUDGE_NEG_OUTPUT_JSONL>', 'w') as f:
        for nt in neg_trajectories:
            d = process_single_trajectory(Path(os.path.join(model_path,f"trajectory_data/{nt}")))
            f.write(d + '\n')
        

    # print(process_single_trajectory(Path(os.path.join(model_path, "trajectory_data/step_000040_1a7101a4-9b17-441a-a9ce-dfa6d4e3ddae_20250910_152010"))))

    # dataset_path_output_dir = os.path.join(model_path, "webjudge_eval_data")