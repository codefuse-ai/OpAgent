import sys
sys.path.append("<AGENT_R1_ROOT>/")
# export PYTHONPATH="<SET_YOUR_PYTHONPATH_HERE>"

import os
os.environ["PYTHONPATH"] = "<AGENT_R1_ROOT>/:$PYTHONPATH"
import json
from pathlib import Path
from PIL import Image, ImageDraw
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import random
from collections import defaultdict
from recipe.webagent_fully_async_policy.browser_env.utils import draw_image_with_coords
import math
#import pypandoc
# ==============================================================================
# 函数 1: 标注图片 (已修改为缩放0.5)
# ==============================================================================
def process_single_trajectory(traj_dir, root_path):
    """处理单个轨迹目录的函数，用于多进程调用"""
    try:
        json_file_path = traj_dir / "evaluation_data.json"
        if not json_file_path.exists():
            return traj_dir.name, 0, f"未找到 evaluation_data.json"
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trajectory = data.get("trajectory_with_rewards", [])
        processed_count = 0
        resize_ratio = 0.5
        for i in range(1, len(trajectory)):
            current_step, previous_step = trajectory[i], trajectory[i-1]

            if current_step.get("type") == "action" and previous_step.get("type") == "observation":
                coords = current_step.get("action", {}).get("coords")
                
                if coords and len(coords) == 2:
                    relative_image_path = previous_step.get("image_path")
                    if not relative_image_path: continue

                    original_image_path = traj_dir / relative_image_path
                    if not original_image_path.exists(): continue

                    output_image_path = original_image_path.with_name(f"{original_image_path.stem}_annotated.png")
                    
                    with Image.open(original_image_path) as img:

                        img = draw_image_with_coords(img, coords)
                        # 缩放图片
                        new_size = (int(img.width * resize_ratio), int(img.height * resize_ratio))
                        img = img.resize(new_size, Image.LANCZOS) # 使用高质量的缩放算法
                        
                        img.save(output_image_path)
                        processed_count += 1
        # 处理最后一张图片
        last_base_name = relative_image_path.split("/")[-1]
        step_ind = int(last_base_name.split("_")[1]) + 2
        img_ind = int(last_base_name.split("_")[3].split(".")[0]) + 1
        last_original_image_path = traj_dir / f"images/step_{step_ind}_img_{img_ind}.png"
        with Image.open(last_original_image_path) as img:
            new_size = (int(img.width * resize_ratio), int(img.height * resize_ratio))
            img = img.resize(new_size, Image.LANCZOS) # 使用高质量的缩放算法
            output_image_path = last_original_image_path.with_name(f"{last_original_image_path.stem}_annotated.png")
            img.save(output_image_path)

        return traj_dir.name, processed_count, None 
        
    except Exception as e:
        return traj_dir.name, 0, f"处理出错: {str(e)}"

def annotate_trajectories(root_dir: str, num_processes: int = None):
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

    print(f"找到 {len(trajectory_dirs)} 个轨迹目录，开始进行图片标注...")
    if num_processes is None: num_processes = min(mp.cpu_count(), len(trajectory_dirs))
    
    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(process_single_trajectory, root_path=root_path)
        results = list(tqdm(pool.imap(process_func, trajectory_dirs), total=len(trajectory_dirs), desc="标注图片"))
    
    total_processed = sum(r[1] for r in results)
    print(f"\n图片标注完成！总共标注了 {total_processed} 张图片。\n")


# ==============================================================================
# 函数 2: 生成 Markdown 报告 (已更新为分文件生成和每个样本的webjudge_details，并折叠样本)
# ==============================================================================
def generate_markdown_reports(root_dir, output_dir):
    """
    为每个训练步骤和每个分数生成一个独立的、带折叠功能的Markdown报告。
    """
    print(f"开始生成 Markdown 报告，将保存在目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    abs_root_dir = os.path.abspath(root_dir)
    trajectory_data_path = os.path.join(abs_root_dir, 'trajectory_data')

    if not os.path.isdir(trajectory_data_path):
        print(f"错误: 在 '{root_dir}' 中未找到 'trajectory_data' 目录。")
        return

    # 数据收集部分保持不变
    grouped_data = defaultdict(lambda: defaultdict(list))
    total_samples = 0
    for sample_dir_name in os.listdir(trajectory_data_path):
        sample_path = os.path.join(trajectory_data_path, sample_dir_name)
        if not os.path.isdir(sample_path): continue
        json_path = os.path.join(sample_path, 'evaluation_data.json')
        if not os.path.exists(json_path): continue
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sample_info = {
                'intent': data.get('intent'), 
                'trajectory': data.get('trajectory_with_rewards'), 
                'sample_dir_path': sample_path,
                'webjudge_details': data.get('webjudge_details', {}), # 新增webjudge_details
                'final_answer': data.get('final_answer', {})
            }
            grouped_data[data.get('training_step')][f"{data.get('final_score'):.3g}"].append(sample_info)
            total_samples += 1
        except Exception as e:
            print(f"警告: 解析 {json_path} 时出错: {e}")
            continue

    print(f"成功解析 {total_samples} 个样本。")
    if not grouped_data:
        print("未找到有效数据，无法生成报告。")
        return

    # --- 核心修改：为每个step和每个score创建一个文件 ---
    for step in sorted(grouped_data.keys(), reverse=True):
        step_data = grouped_data[step]
        for score in sorted(step_data.keys(), reverse=True):
            # 构造每个报告的文件名，使用0填充保证排序正确
            report_filename = f"report_step_{step:06d}_score_{score}.md"
            report_filepath = os.path.join(output_dir, report_filename)
            
            print(f"  - 正在生成报告: {report_filepath}")

            with open(report_filepath, 'w', encoding='utf-8') as md_file:
                md_file.write(f"# 训练步骤: {step} - 分数: {score} - 轨迹报告\n\n")
                
                samples = step_data[score]
                # 决定展示多少个样本，可以根据需要调整
                samples_to_show = random.sample(samples, min(len(samples), 20)) 
                
                for i, sample in enumerate(samples_to_show):
                    md_file.write(f"<details>\n<summary><h2>样本 {i+1}: {sample['intent']}</h2></summary>\n\n") # 样本折叠

                    # 添加webjudge_details.response内容到每个样本下
                    webjudge_response = sample.get('webjudge_details', {}).get('response', 'N/A')
                    webjudge_response = webjudge_response.replace('```', '')
                    md_file.write("#### WebJudge Details for this Sample:\n")
                    md_file.write("```text\n")
                    md_file.write(webjudge_response)
                    md_file.write("\n```\n\n")
                    base_name = ""
                    trajectory_steps = sample.get('trajectory', [])
                    
                    if len(trajectory_steps) == 0:
                        print("sample is 0:", sample)
                    # 遍历观察和行动对
                    for j in range(0, len(trajectory_steps), 2):
                        if j + 1 < len(trajectory_steps): # 确保有对应的action
                            obs = trajectory_steps[j]
                            action = trajectory_steps[j+1]

                            image_path_raw = obs.get('image_path')
                            if not image_path_raw: continue

                            base_name, _ = os.path.splitext(image_path_raw)
                            annotated_image_abs_path = os.path.join(sample['sample_dir_path'], f"{base_name}_annotated.png")
                            
                            if os.path.exists(annotated_image_abs_path):
                                step_number = (j // 2) + 1
                                raw_prediction = action.get('raw_prediction', 'N/A')
                                think_str = raw_prediction.split('</think>')[0]
                                think_str = think_str.replace('<think>', '').replace('</think>', '')
                                md_file.write(f"<details>\n<summary><b>步骤 {step_number}: {think_str}</b></summary>\n\n")
                                
                                # 这里调整图片路径，使其相对于报告文件
                                # os.path.dirname(report_filepath) 获取当前报告文件的目录
                                relative_image_path = os.path.relpath(annotated_image_abs_path, start=os.path.dirname(report_filepath)).replace(os.sep, '/')
                                md_file.write(f"![Annotated Image]({relative_image_path})\n\n")
                                
                                md_file.write("#### Raw Prediction:\n")
                                
                                md_file.write("```\n")
                                md_file.write(raw_prediction)
                                md_file.write("\n```\n\n")
                                
                                md_file.write("</details>\n\n")

                    final_answer = sample.get('final_answer').get('answer', 'N/A')
                    md_file.write(f"<details>\n<summary><b>最后答案: {final_answer}</b></summary>\n\n")
                    
                    # 这里调整图片路径，使其相对于报告文件
                    # os.path.dirname(report_filepath) 获取当前报告文件的目录
                    if len(base_name) > 0:
                        step_ind = int(base_name.split("_")[1]) + 2
                        img_ind = int(base_name.split("_")[-1]) + 1
                        last_annotated_image_abs_path = os.path.join(sample['sample_dir_path'], f"images/step_{step_ind}_img_{img_ind}_annotated.png")
                        relative_image_path = os.path.relpath(last_annotated_image_abs_path, start=os.path.dirname(report_filepath)).replace(os.sep, '/')
                        md_file.write(f"![Annotated Image]({relative_image_path})\n\n")
                        
                    md_file.write("</details>\n\n")

                    md_file.write("</details>\n\n") # 结束样本折叠标签
                    md_file.write("---\n\n")
    
    print(f"\n报告生成完毕！所有报告文件已保存在 '{output_dir}' 目录中。")

# ==============================================================================
# 函数 3: 将 Markdown 转换为 PDF (新增)
# ==============================================================================
def convert_reports_to_pdf(md_dir, pdf_output_dir):
    """
    将指定目录下的所有 Markdown 文件转换为 PDF。
    """
    print(f"开始将 Markdown 报告转换为 PDF，将保存在目录: {pdf_output_dir}")

    # 检查 pandoc 是否安装
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("\n错误: Pandoc 未找到。请确保 Pandoc 已经安装并存在于系统的 PATH 中。")
        print("访问 https://pandoc.org/installing.html 进行安装。")
        return

    # 确保 PDF 输出目录存在
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    md_files = list(Path(md_dir).glob('*.md'))
    if not md_files:
        print("未找到任何 Markdown 文件进行转换。")
        return

    for md_file in tqdm(md_files, desc="转换 PDF"):
        output_pdf_path = Path(pdf_output_dir) / f"{md_file.stem}.pdf"
        try:
            pypandoc.convert_file(
                str(md_file),
                'pdf',
                outputfile=str(output_pdf_path),
                # extra_args=['--pdf-engine=pdflatex', '-V', 'geometry:margin=1in'] # 可以添加额外参数调整样式
            )
        except Exception as e:
            print(f"\n转换文件 {md_file.name} 时出错: {e}")
            print("提示: 常见的错误是缺少 LaTeX 发行版 (如 MiKTeX, TeX Live)。请确保已正确安装。")
            # 遇到一个错误后可以选择继续或停止
            # break 

    print(f"\nPDF 转换完成！所有 PDF 文件已保存在 '{pdf_output_dir}' 目录中。")

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    dataset_path = "<EXPERIMENTS_ROOT>"
    #model_name = "Qwen2.5-VL-7B-Instruct_JudgewoAnswer_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_VLM_EXP_NAMEWithStepScore_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-7B-Instruct_JudgewoAnswer_EC0_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_VLM_EXP_NAMEWithStepScore_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-7B-Instruct_JudgewoAnswer_woKLInReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_VLM_EXP_NAMEWithStepScore_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-7B-Instruct_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_VLM_EXP_NAMEWithStepScore_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-7B-Instruct_GSPO_JudgewoAnswer_EC0.001_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_gemini_gen_intent_task_v3_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-7B-Instruct_GSPO_JudgewoAnswer_EC0.001_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_gemini_gen_intent_task_v3_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-72B-Instruct_JudgewoAnswer_EC0.001_woKLInReward_WithToolUsageReward_OnlyEasy_SuffleTrue_MeanFormatReward_64_ObserTypeimage_Gpuppu-810ehigh_Datasetwa_gemini_gen_intent_task_v3_ProcessRewaedFalse_catid"
    #model_name = "Qwen2.5-VL-72B-Instruct_JudgewoAnswer_EC0.001_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_gemini_gen_intent_task_v4_cookied_auth_filter_RewardModel0912_old"
    #model_name = "debug_vis"
    #model_name = "Qwen2.5-VL-72B-Instruct_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_huaizhu_0910_peiheng_v4_0918_RewardModel0912"
    #model_name = "debug_vis_icon_async_test_yuyu"
    #model_name = "Qwen2.5-VL-72B-Instruct_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppuhigh_Datasetwa_huaizhu_0910_peiheng_v4_0918_RewardModel0912"
    #model_name = "Qwen2.5-VL-72B-Instruct_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppulow_Datasetwa_huaizhu_0910_peiheng_v4_0918_RewardModel0912"
    #model_name = "Qwen2.5-VL-72B-Instruct_wCap_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppuhigh_Datasetwa_huaizhu_0910_peiheng_v4_0918_RewardModel0912"
    #model_name = "Qwen2.5-VL-72B-Instruct_wPRALL_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuppuhigh_Datasetwa_huaizhu_0910_peiheng_v4_0918_RewardModel0912"
    #model_name = "Qwen2.5-VL-72B-Instruct_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasetdatasets_OAgent_RewardModel0912"
    #model_name = "Qwen2.5-VL-72B-Instruct_Stepwise_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasetph_xl_hz_1031_test_webarena_RewardModel0912"
    #model_name = "checkpoint-150_Stepwise_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasetph_xl_hz_1031_test_webarena_RewardModel0912"
    #model_name = "Qwen2.5-VL-72B-Instruct_Stepwise_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleTrue_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasetph_xl_hz_1031_test_webarena_RewardModel0912"
    #model_name = "111_checkpoint-150_ConSch_Stepwise_wPRALL4_Prompt1012_klcov_wRewardMask_JudgewoAnswer_EC0_woKLInReward_SuffleFalse_MeanFormatReward_32_ObserTypeimage_Gpuh20-3ehigh_Datasettest_webarena_conflict_RewardModel0912"
    #model_name = "Qwen3-VL-32B-Thinking_Async_Stepwise_klconv4__ConSch_wPRALL4_Prompt1012_SuffleFalse_MeanFormatReward_64_ObserTypeimage_Gpuh20-3ehigh_Datasettest_webarena_conflict_webjudge_RewardModel0912"
    model_name = "Qwen3-VL-32B-Thinking_Async_Stepwise_klconv4__ConSch_Prompt1012_SuffleFalse_MeanFormatReward_64_ObserTypeimage_Gpuh20-3ehigh_Datasetantmonitor_0126_repeat_3k_RewardModel0912"
    # 构造模型路径
    model_path = os.path.join(dataset_path, model_name)

    # 1. 标注所有轨迹图片
    annotate_trajectories(model_path, num_processes=8)

    # 2. 生成多个 Markdown 报告，并将它们保存在模型目录下的 "reports" 子文件夹内
    report_output_dir = os.path.join(model_path, "trajectory_reports")
    generate_markdown_reports(model_path, output_dir=report_output_dir)
    
    # 3. 将生成的 Markdown 报告转换为 PDF (新增步骤)
    # pdf_report_dir = os.path.join(model_path, "pdf_reports")
    # convert_reports_to_pdf(md_dir=report_output_dir, pdf_output_dir=pdf_report_dir)

    print("\n所有任务处理完毕！")
