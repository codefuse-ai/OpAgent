import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
methods_dir = os.path.join(current_dir, 'methods')
# print(methods_dir)
sys.path.append(methods_dir)

# from methods.agenttrek_eval import *
# from methods.automomous_eval import *
# from methods.webjudge_general_eval import *

from src.methods.webjudge_online_mind2web import *
# from methods.webvoyager_eval import *
from src.utils import extract_predication
from openai import OpenAI
# from vllm_model import VLLMModelEngine
import json
import copy
import asyncio
import multiprocessing
import traceback
from src.qwen_vlm_request import send_chat_completion_request_shangshu, getJsonObject
from PIL import Image
import re
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# GLOBALNUM = 10
# def draw_image(img, coords):
#     # global GLOBALNUM  # 声明 GLOBALNUM 是全局变量
#     if coords is None or coords == (None, None):
#         return img
#     draw = ImageDraw.Draw(img)
#     x, y = coords
#     radius = 15
#     draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="red", width=3)
#     draw.line([x - radius/2, y, x + radius/2, y], fill="red", width=2)
#     draw.line([x, y - radius/2, x, y + radius/2], fill="red", width=2)
#     # savepath = f'/tmp/{GLOBALNUM}.png'
#     # print(savepath)
#     # img.save(f'/tmp/{GLOBALNUM}.png')
#     # GLOBALNUM += 1

#     return img

# def extract_coordinates(text):
#     # 使用正则表达式匹配 JSON 部分
#     match = re.search(r'\{.*?\}', text)
#     if match:
#         json_str = match.group(0)
#         try:
#             # 将 JSON 字符串解析为字典
#             data = json.loads(json_str)
#             x = data.get('x')
#             y = data.get('y')
#             return x, y
#         except json.JSONDecodeError as e:
#             print("JSON 解析失败:", e)
#     return None, None

def parse_scores_from_string(text_blob: str) -> list[int]:
    """
    从给定的多行字符串中按顺序解析评估分数。

    该函数通过正则表达式查找 "**得分：**" 或 "**得分:**" 模式，
    并提取其后紧跟的数字。

    Args:
        text_blob: 包含评估分析和分数的字符串。

    Returns:
        一个包含按顺序找到的所有分数的整数列表。
        如果未找到任何分数，则返回一个空列表。
    """
    # 正则表达式模式详解:
    # \*\*得分   - 匹配字面量 "**得分" (星号需转义)
    # [：:]    - 匹配一个字符，该字符可以是全角冒号 "：" 或半角冒号 ":"
    # \s*      - 匹配冒号后的零个或多个空白字符（如空格、换行符等）
    # (\d+)    - 捕获组：匹配并捕获一个或多个数字。这是我们真正想要提取的内容。
    pattern = r"得分[：:]\D*?(\d+)"

    # re.findall会查找所有匹配项，并返回一个仅包含所有捕获组内容的列表。
    # 在这里，它会返回一个由分数组成的字符串列表，例如 ['3', '3', '4']
    matches = re.findall(pattern, text_blob)

    # 使用列表推导式将匹配到的字符串列表转换为整数列表
    scores = [int(score) for score in matches]

    return scores

async def simple_eval(trajectory_images, result, score_threshold=1, mode='WebJudge_Online_Mind2Web_eval', model='Qwen2.5-VL-72B-Instruct'):
    screenshot_paths = []
    thoughts = None
    action_history = None
    final_result_response = None
    input_image_paths = []
    task_description = []
    coords = []
    task_id = '-'

    output_results = copy.deepcopy(result)
    task_description = result["task"]
    if "task_id" in result:
        task_id = result["task_id"]
    if "action_history" in result:
        action_history = result["action_history"]
        # for action_str in action_history:
        #     coord = extract_coordinates(action_str) # 没传入坐标没起作用
        #     coords.append(coord)
    action_size = len(coords)
    if "thoughts" in result:
        thoughts = result["thoughts"]
    if "final_result_response" in result:
        final_result_response = result["final_result_response"]
    if "input_image_paths" in result: # input_image_paths优先级最高
        input_image_paths = result["input_image_paths"]

        trajectory_images = []
        # input_image_size = len(input_image_paths)
        # for action, image_path in zip_longest(coords, input_image_paths):
        #     if action_size + 1 == input_image_size:
        #         trajectory_images.append(draw_image(Image.open(image_path), action))
        #     elif image_path:
        #         trajectory_images.append(Image.open(image_path))
        for image_path in input_image_paths:
            trajectory_images.append(Image.open(image_path))

    elif type(trajectory_images)==str:
        input_image_paths = trajectory_images

        trajectory_images = []
        file_paths = []
        # 遍历所有文件，收集符合条件的文件路径
        for root, dirs, files in os.walk(input_image_paths):
            for file in files:
                if '_annotated' in file:  # 跳过带 '_annotated' 的文件
                    continue
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        # 根据文件创建时间排序（Windows有效，Linux可能为元数据变更时间）
        file_paths.sort(key=lambda x: os.path.getctime(x))
        # 依次打开文件并添加到轨迹列表
        for image_path in input_image_paths:
            trajectory_images.append(Image.open(image_path))
    # else:
    #     input_images = []
    #     input_image_size = len(trajectory_images)
    #     for action, image in zip_longest(coords, trajectory_images):
    #         if action_size + 1 == input_image_size:
    #             print('draw image to:')
    #             input_images.append(draw_image(image, action))
    #         elif image:
    #             input_images.append(image)
    #     trajectory_images = input_images

    logger.info(f"Start evaluation for Task: {task_description}")

    if mode == "Autonomous_eval":
        messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])

    elif mode == "AgentTrek_eval":
        messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts, screenshot_paths[-1])

    elif mode == "WebVoyager_eval":
        messages, text, system_msg = WebVoyager_eval(task_description, screenshot_paths, final_result_response)

    elif mode == "WebJudge_Online_Mind2Web_eval":
        messages, text, system_msg, record, key_points = await WebJudge_Online_Mind2Web_eval(task_description, action_history, trajectory_images, model,
                                          score_threshold, final_result_response, thoughts)
        output_results["image_judge_record"] = record
        output_results["key_points"] = key_points

    elif mode == "WebJudge_general_eval":
        messages, text, system_msg, record, key_points = await WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, score_threshold)
        # messages, text, system_msg, record, key_points = WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold)
        output_results["image_judge_record"] = record
        output_results["key_points"] = key_points

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # print(f"process message: {messages}")

    openai_res = openAi_api(model, messages, temperature=0)
    # print(f"openai_res: {openai_res}")
    response = openai_res.choices[0].message.content
    #print(f"response: {response}")
    response_json = getJsonObject(response)
    #print(response_json)
    if isinstance(response_json, dict):
        score = response_json 
    else:
        score = None
    
    # 解析分数
    complete_score, action_score, traj_score = -1, -1, -1
    try:
        complete_score = int(score.get('Completion Score'))
        action_score = int(score.get('Action Score'))
        traj_score = int(score.get('Trajectory Score'))
    except Exception as e1:
        try:
            scores = parse_scores_from_string(response)
            complete_score, action_score, traj_score = map(int, scores)
        except Exception as e3:
            logger.error(f"webjudge分数解析失败:{response}")

    output_results["complete_score"] = complete_score
    output_results["action_score"] = action_score
    output_results["traj_score"] = traj_score
    ## response = model.generate(messages)[0]
    # predicted_label = extract_predication(response, mode)
    if score and int(score.get('Completion Score', -1)) >= 5:
        predicted_label = 1
    else:
        predicted_label = 0

    # Store evaluation details
    evaluation_results = {"response": response, "predicted_label": predicted_label}
    output_results["task_id"] = task_id
    output_results["input_text"] = text
    output_results["system_msg"] = system_msg
    output_results["evaluation_details"] = evaluation_results
    output_results["predicted_label"] = predicted_label

    logger.info(f"Finish evaluation for {task_description}")
    logger.info("=" * 20)

    return output_results

def auto_eval(args, task_subset, final_predicted_labels, lock):
    model_name = args.model.split("/")[-1]

    ################## get the already done task id ###############
    output_json_path = os.path.join(args.output_path,
                                    f"{args.mode}_{model_name}_score_threshold_{args.score_threshold}_auto_eval_results.json")
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
                logger.error(f"error item: {item}")

    logger.info(f"The number of already done tasks: {len(already_ids)}")

    for task_id in task_subset:
        # Skip already done task
        if task_id in already_ids:
            continue

        trajectory_images_path = os.path.join(args.trajectories_dir, task_id, "trajectory")
        screenshot_paths = []
        thoughts = None
        action_history = None
        final_result_response = None
        input_image_paths = []
        task_description = None

        # Load results
        with open(os.path.join(args.trajectories_dir, task_id, "result.json")) as f:
            result = json.load(f)
            # print(f"input info: {result}")
            output_results = copy.deepcopy(result)
            task_description = result["task"]
            if "action_history" in result:
                action_history = result["action_history"]
            if "thoughts" in result:
                thoughts = result["thoughts"]
            if "final_result_response" in result:
                final_result_response = result["final_result_response"]
            if "input_image_paths" in result:
                input_image_paths = result["input_image_paths"]

        logger.info(f"Start evaluation for Task: {task_description}")
        # Do the auto-eval
        # abs_base_path = Path(trajectory_images_path).resolve()

        # for image in sorted(os.listdir(Path(trajectory_images_path).resolve()), key=lambda x: int(re.findall(r'\d+', x)[0])):
        #     full_path = abs_base_path / image
        #     screenshot_paths.append(str(full_path.resolve()))
        # input_image_paths_json = json.load(input_image_paths)
        # screenshot_paths = input_image_paths
        for image_path in input_image_paths:
            if "ori_image_fullpage" in image_path:
                image_path = image_path.replace("ori_image_fullpage", "ori_image_fullpage_added")
            screenshot_paths.append(image_path)
        logger.info(screenshot_paths)
        # for image_path in input_image_paths:
        #     image_filename = image_path.split(r"/")[-1]

        #     # for image_tmp in trajectory_images_path:
        #     #     image_tmp_file = image_tmp.split(r"/")[-1]
        #     #     print(image_tmp_file)
        #     #     if image_filename == image_tmp_file:
        #     image_tmp = trajectory_images_path + "/" + image_filename
        #     screenshot_paths.append(image_tmp)
        try:
            if args.mode == "Autonomous_eval":
                messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])

            elif args.mode == "AgentTrek_eval":
                messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts,
                                                            screenshot_paths[-1])

            elif args.mode == "WebVoyager_eval":
                messages, text, system_msg = WebVoyager_eval(task_description, screenshot_paths, final_result_response)

            elif args.mode == "WebJudge_Online_Mind2Web_eval":
                messages, text, system_msg, record, key_points = asyncio.run(
                    WebJudge_Online_Mind2Web_eval(task_description, action_history, screenshot_paths, args.model,
                                                  args.score_threshold))
                output_results["image_judge_record"] = record
                output_results["key_points"] = key_points

            # elif args.mode == "WebJudge_general_eval":
            #     messages, text, system_msg, record, key_points = asyncio.run(WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold))
            #     # messages, text, system_msg, record, key_points = WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold)
            #     output_results["image_judge_record"] = record
            #     output_results["key_points"] = key_points

            else:
                raise ValueError(f"Unknown mode: {args.mode}")
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
            continue
        # print(f"process message: {messages}")

        # response = model.generate(messages)[0]
        openai_res = openAi_api(args.model, messages)
        # print(f"openai_res: {openai_res}")
        response = openai_res.choices[0].message.content

        predicted_label = extract_predication(response, args.mode)

        # print(f"predicted_label: {predicted_label}")

        # Store evaluation details
        evaluation_results = {"response": response, "predicted_label": predicted_label}
        output_results["task_id"] = task_id
        output_results["input_text"] = text
        output_results["system_msg"] = system_msg
        output_results["evaluation_details"] = evaluation_results
        output_results["predicted_label"] = predicted_label

        with lock:
            final_predicted_labels.append(predicted_label)

        logger.info(f"Finish evaluation for {task_description}")
        logger.info("=" * 20)
        logger.info(args.output_path)
        os.makedirs(args.output_path, exist_ok=True)
        with lock:
            output_json_path = os.path.join(args.output_path,
                                            f"{args.mode}_{model_name}_score_threshold_{args.score_threshold}_auto_eval_results.json")
            logger.info(output_json_path)
            logger.info(json.dumps(output_results))
            with open(os.path.join(args.output_path,
                                   f"{args.mode}_{model_name}_score_threshold_{args.score_threshold}_auto_eval_results.json"),
                      "a+") as f_out:
                f_out.write(json.dumps(output_results) + "\n")


def openAi(model, messages):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8030/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return chat_response

def openAi_api(model, messages, temperature=0.6):
    # Set OpenAI's API key and API base to use vLLM's API server.
    chat_response = send_chat_completion_request_shangshu(messages, model, temperature=temperature)

    return chat_response


def single_eval(args, task_subset):
    model_name = args.model.split("_")[-1]
    output_json_path = os.path.join(args.output_path,
                                    f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_single_eval_results.json")

    task_id = task_subset[0][0]
    trajectory_images_path = os.path.join(args.trajectories_dir, task_id, "trajectory")
    # print("trajectory_images_path:", trajectory_images_path)
    screenshot_paths = []
    thoughts = None
    action_history = None
    final_result_response = None
    input_image_paths = []
    task_description = []
    # Load results
    with open(os.path.join(args.trajectories_dir, task_id, "result.json")) as f:
        result = json.load(f)
        # print(f"input info: {result}")
        output_results = copy.deepcopy(result)
        task_description = result["task"]
        if "action_history" in result:
            action_history = result["action_history"]
        if "thoughts" in result:
            thoughts = result["thoughts"]
        if "final_result_response" in result:
            final_result_response = result["final_result_response"]
        if "input_image_paths" in result:
            input_image_paths = result["input_image_paths"]

    logger.info(f"Start evaluation for Task: {task_description}")
    # Do the auto-eval
    # abs_base_path = Path(trajectory_images_path).resolve()
    # for image in sorted(os.listdir(Path(trajectory_images_path).resolve()), key=lambda x: int(re.findall(r'\d+', x)[0])):
    #     full_path = abs_base_path / image
    #     screenshot_paths.append(str(full_path.resolve()))
    for image in input_image_paths:
        image_filename = os.path.basename(image)
        for image_tmp in trajectory_images_path:
            image_tmp_file = os.path.basename(image_tmp)
            if image_filename == image_tmp_file:
                screenshot_paths.append(image_tmp)

    if args.mode == "Autonomous_eval":
        messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])

    elif args.mode == "AgentTrek_eval":
        messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts, screenshot_paths[-1])

    elif args.mode == "WebVoyager_eval":
        messages, text, system_msg = WebVoyager_eval(task_description, screenshot_paths, final_result_response)

    elif args.mode == "WebJudge_Online_Mind2Web_eval":
        messages, text, system_msg, record, key_points = asyncio.run(
            WebJudge_Online_Mind2Web_eval(task_description, action_history, screenshot_paths, model,
                                          args.score_threshold))
        output_results["image_judge_record"] = record
        output_results["key_points"] = key_points

    elif args.mode == "WebJudge_general_eval":
        messages, text, system_msg, record, key_points = asyncio.run(
            WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths,
                                  model, args.score_threshold))
        # messages, text, system_msg, record, key_points = WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold)
        output_results["image_judge_record"] = record
        output_results["key_points"] = key_points

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # print(f"process message: {messages}")

    openai_res = openAi_api(args.model, messages)
    # print(f"openai_res: {openai_res}")
    response = openai_res.choices[0].message.content
    logger.info(f"response: {response}")

    # response = model.generate(messages)[0]
    predicted_label = extract_predication(response, args.mode)

    # Store evaluation details
    evaluation_results = {"response": response, "predicted_label": predicted_label}
    output_results["task_id"] = task_id
    output_results["input_text"] = text
    output_results["system_msg"] = system_msg
    output_results["evaluation_details"] = evaluation_results
    output_results["predicted_label"] = predicted_label

    logger.info(f"Finish evaluation for {task_description}")
    logger.info("=" * 20)
    os.makedirs(args.output_path, exist_ok=True)


def process_subset(task_subset, args, final_predicted_labels, lock):
    auto_eval(args, task_subset, final_predicted_labels, lock)


def parallel_eval(args, num_workers=60):
    # Evaluate in parallel based on num of works
    task_dirs = [
        d for d in sorted(os.listdir(args.trajectories_dir))
        if os.path.isdir(os.path.join(args.trajectories_dir, d))
    ]
    logger.info(f"Evaluating {len(task_dirs)} tasks in total.")
    chunk_size = max(len(task_dirs) // num_workers,1)
    task_subsets = [task_dirs[i:i + chunk_size] for i in range(0, len(task_dirs), chunk_size)]
    # Load model
    # model = VLLMModelEngine()
    # model = OpenaiEngine(model=args.model, api_key=args.api_key)
    # single_eval(args, task_subsets)

    # 并发处理
    lock = multiprocessing.Lock()
    with multiprocessing.Manager() as manager:
        final_predicted_labels = manager.list()
        processes = []
        for subset in task_subsets:
            p = multiprocessing.Process(target=process_subset, args=(subset, args, final_predicted_labels, lock))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        success_num = sum(final_predicted_labels)

    logger.info("Evaluation complete.")
    logger.info(f"The success rate is {(success_num / len(task_dirs)) * 100}.")

