# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Optional
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset

import os
import json, json_repair
import copy
import random
import asyncio
from PIL import Image
import threading
import logging
import datetime
from collections import defaultdict
from itertools import zip_longest
from typing import List, Dict
from recipe.webagent_fully_async_policy.browser_env.tool_env import ToolEnv, WebBrowserEnv
logger = logging.getLogger(__name__)
#logger.setLevel("DEBUG")
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
VLM_EXP_DEBUG = os.environ.get('VLM_EXP_DEBUG', '0')
VLM_EXP_DEBUG = str(VLM_EXP_DEBUG)
EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', "")
EXPERIMENT_NAME = str(EXPERIMENT_NAME)
DATA_SFUFFLE = os.environ.get('DATA_SFUFFLE', 'True')
DATA_SFUFFLE = str(DATA_SFUFFLE)
VAL_DATASET_PATH = os.environ.get('VAL_DATASET_PATH', '')
VAL_DATASET_PATH = str(VAL_DATASET_PATH)
BATCH_SIZE = os.environ.get('BATCH_SIZE', '32')
BATCH_SIZE = int(BATCH_SIZE)
print("VAL_DATASET_PATH: ", VAL_DATASET_PATH)
def read_json_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        outputs = json.load(f)
    return outputs

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        #tensors[key].append(val)
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)
    #     non_tensors[key].append(val)
    return {**tensors, **non_tensors,}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

class TaskScheduler:
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.conflict_groups = defaultdict(list)
        self.non_conflicting = []
    
    def load_tasks(self, input_path: str, input_list: List[str] = None) -> None:
        """加载任务"""
        task_file_names = input_list #* 5
        for task_file_name in task_file_names:
            full_path = os.path.join(input_path, task_file_name)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    task_conf = json.load(f)
                
                conflict_key = task_conf.get('conflict_key')
                
                if conflict_key and conflict_key != "":
                    self.conflict_groups[conflict_key].append({
                        'file': task_file_name,
                        'config': task_conf,
                        'conflict_key': conflict_key
                    })
                else:
                    self.non_conflicting.append({
                        'file': task_file_name,
                        'config': task_conf,
                        'conflict_key': None
                    })
            except Exception as e:
                logger.error(f"Error loading {task_file_name}: {e}")
    
    def schedule_tasks_fixed_batches_balanced(self, num_steps: int = 7) -> List[List[Dict]]:
        """固定batch数，平衡分布"""
        
        total_tasks = sum(len(tasks) for tasks in self.conflict_groups.values()) + \
                      len(self.non_conflicting)
        
        logger.info(f"\n[固定Batch调度 - 平衡策略]")
        logger.info(f"  总任务数: {total_tasks}")
        logger.info(f"  固定step数: {num_steps}")
        logger.info(f"  每batch限制: {self.batch_size} 任务")
        logger.info(f"  理想平均: {total_tasks / num_steps:.1f} 任务/batch\n")
        
        # 初始化batch
        batches = [[] for _ in range(num_steps)]
        
        # 按conflict_key的任务数排序
        conflict_items = sorted(
            self.conflict_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        logger.info(f"[冲突key分布]")
        for rank, (conflict_key, tasks) in enumerate(conflict_items[:8], 1):
            logger.info(f"  {rank}. {conflict_key:<30} {len(tasks):3} 个任务")
        if len(conflict_items) > 8:
            logger.info(f"  ... 还有 {len(conflict_items) - 8} 个key")
        
        # 构建平衡的任务队列（关键）
        max_tasks_per_key = max((len(t) for _, t in conflict_items), default=0)
        task_queue = []
        
        # 轮转取任务：position 0, 1, 2, ... 每个position从每个key取一个
        for position in range(max_tasks_per_key):
            for conflict_key, tasks in conflict_items:
                if position < len(tasks):
                    task_queue.append(tasks[position])
        
        logger.info(f"[任务队列构建]")
        logger.info(f"  任务队列长度: {len(task_queue)}")
        logger.info(f"  非冲突任务: {len(self.non_conflicting)}\n")
        
        # 贪心分配：选择任务最少的可用batch
        for task in task_queue:
            # 优先选择未满的batch
            available_batches = [i for i in range(num_steps) 
                                if len(batches[i]) < self.batch_size]
            
            if available_batches:
                best_batch_idx = min(available_batches, 
                                    key=lambda i: len(batches[i]))
            else:
                # 所有batch都满，选择最小的
                best_batch_idx = min(range(num_steps), 
                                    key=lambda i: len(batches[i]))
            
            batches[best_batch_idx].append(task)
        
        # 分配非冲突任务
        for task in self.non_conflicting:
            available_batches = [i for i in range(num_steps) 
                                if len(batches[i]) < self.batch_size]
            
            if available_batches:
                best_batch_idx = min(available_batches, 
                                    key=lambda i: len(batches[i]))
            else:
                best_batch_idx = min(range(num_steps), 
                                    key=lambda i: len(batches[i]))
            
            batches[best_batch_idx].append(task)
        
        return batches
    


class ToolRLDataset(RLHFDataset):
    """
    Dataset for tool use in RLHF
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 tool_env: ToolEnv = None,
                 use_custom_tool_format_func=False):
        self.tool_env = tool_env
        self.tools = tool_env.tool_desc
        self.use_custom_tool_format_func = use_custom_tool_format_func
        super().__init__(parquet_files, tokenizer, processor, prompt_key, image_key, max_prompt_length, filter_prompts, cache_dir, chat_template_func, return_raw_chat, truncation, filter_overlong_prompts)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """

        row_dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict.pop(self.prompt_key)

        if self.use_custom_tool_format_func:
            if chat[0]['role'] == 'system':
                chat[0]['content'] = chat[0]['content'] + self.tool_env.tools_format_func()
            else:
                system_msg = [{"role": "system", "content": self.tool_env.tools_format_func()}]
                # Convert chat to a list if it's not already one
                chat_list = chat.tolist() if hasattr(chat, 'tolist') else list(chat)
                chat = system_msg + chat_list
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, tools=self.tools, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
            position_ids = [position_ids]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
    
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], tools=self.tools, add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')


class WebAgentRLDataset(ToolRLDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 tool_env: Optional[Union[ToolEnv, WebBrowserEnv]] = None,
                 use_custom_tool_format_func=False,
                 split='train',
                 config=None):
        self.config = config
        self.tool_env = tool_env
        self.tools = tool_env.tool_desc
        self.use_custom_tool_format_func = use_custom_tool_format_func

        self.parquet_files = copy.deepcopy(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store
        self.split = split
        self.serialize_dataset = False
        self.playwright_startup_lock = threading.Lock()
        self._read_files_and_tokenize(split)


    def __len__(self):
        return len(self.dataframe)

    def get_screenshot_from_start_urls(self, row_dict, env):
        # 1. 创建循环和设置，放在 try 外部
        web_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(web_loop)
        try:
            # 2. 在 try 块中执行核心的异步逻辑
            observation, _ = web_loop.run_until_complete(
                env.areset(options={"config_file": row_dict['config_file']}, 
                           playwright_startup_lock=self.playwright_startup_lock)
            )
            obs_img = Image.fromarray(observation)
            
            # 成功获取截图后，在返回前先关闭浏览器环境。
            # 这是因为 env.aclose() 也需要事件循环来运行。
            web_loop.run_until_complete(env.aclose())
            web_loop.close()
            return obs_img

        except Exception as e:
            # 3. 如果 try 块中发生任何错误，记录日志并重新抛出
            logger.error(f"Error getting screenshot from start urls: {e}")
            # 重新抛出异常，让调用者（如 DataLoader）知道这个样本处理失败了
            raise

        finally:
            # 4. finally 块：无论成功还是失败，这里都将被执行
            # 它的核心职责是清理事件循环本身。
            
            # 在关闭循环之前，我们最后再尝试一次关闭浏览器环境。
            # 这主要处理 areset 成功但后续代码失败的情况。
            try:
                # 只有在循环还未关闭时才尝试运行
                if not web_loop.is_closed() and env:
                    web_loop.run_until_complete(env.aclose())
            except Exception as close_error:
                # 如果关闭环境也失败了，只记录日志，不让它影响最终的循环关闭。
                logger.error(f"Failed to close env in finally block: {close_error}")

            # 确保事件循环最终被关闭
            if not web_loop.is_closed():
                web_loop.close()
            
            # 好习惯：将当前线程的事件循环策略重置
            asyncio.set_event_loop(None)

    def get_background_screenshot(self,):
        fallback_img = np.zeros((self.config.tool.webbrowser.viewport_height, self.config.tool.webbrowser.viewport_width, 3), dtype=np.uint8)
        obs_img = Image.fromarray(fallback_img)
        return obs_img

    def webarena2agent(self, idx, webarena_confile_file, split='train'):
        #        idx = int(webarena_confile_file.split("/")[-1].split(".")[0])
        with open(webarena_confile_file, 'r', encoding='utf-8') as config_file:
            config_dict = json_repair.load(config_file)
        #debug 过滤
        # if VLM_EXP_DEBUG == '1':
        #     filter_debug = False
        #     debug_file_list = ['']
        #     if config_dict['start_url'] is not None:
        #         filter_debug = "bilibili" in config_dict['start_url']
        #     if not filter_debug:
        #         return None

        instruction_following = """<image>You are an excellent web agent. 
Now, you are given a user query along with the last webpage (including screenshot and other information). Besides the image, other information returned by the browser is placed in <tool_response>...</tool_response>.
You need to call the provided webpage functions multiple times to complete the user\'s request. 
Considering the last state of the webpage and the user\'s request, please give a required webpage function and its corresponding parameters for each round of conversation, and output them strictly according to the following format.
You must first conduct reasoning inside <think>...</think>. 
In the <think>...</think>, please first describe the main content of the last screenshot, and then, based on the previous screenshots, analyze what action to take to continue fulfilling the user's request.
After reasoning, you must output EXACTLY ONE action: either a tool call or a final answer.
If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer for the user\'s request, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
Please generate the next ONE action (either tool call or answer) according to instruction, action history and the last UI screenshot or image.
Instruction: %s
"""

        instruction_following_single_step = """<image>You are an excellent web agent. 
Now, you are given a user query along with the latest webpage (including screenshot and other information). Besides the image, other information returned by the browser is placed in <tool_response>...</tool_response>.
You need to call one provided webpage function to complete the user\'s request progressively. 
Considering the last state of the webpage and the user\'s request, please give a required webpage function and its corresponding parameters, and output them strictly according to the following format.
You must first conduct reasoning inside <think>...</think>. 
In the <think>...</think>, please first describe the main content of the last screenshot, and then, based on the action history, analyze what action to take to continue fulfilling the user's request.
After reasoning, you must output EXACTLY ONE action: either a tool call or a final answer.
If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
If the latest webpage screenshot and the action history are sufficient to fulfill the user's request, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
Please generate the next ONE action (either tool call or answer) according to instruction, action history and the last UI screenshot or image.
Instruction: %s
"""

        if "Stepwise".lower() in EXPERIMENT_NAME.lower():
            question = instruction_following_single_step % config_dict['intent'] 
        else:
            question = instruction_following % config_dict['intent'] 
        #question = "<image>\nQuestion: " + config_dict['intent'] 
        if 'reference_answer_raw_annotation' in config_dict['eval']:
            raw_answer = config_dict['eval']['reference_answer_raw_annotation']
        else:
            raw_answer = "The task is finished!"#config_dict['eval']['reference_url']

        # if raw_answer == "" or raw_answer is None:
        #     return None
        data = {
                "data_source": 'webarena',
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "webagent",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": raw_answer,
                    "eval_method":json.dumps(config_dict['eval'])
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': raw_answer,
                    'question': config_dict['intent'],
                    'config': json.dumps(config_dict),
                },
                'config_file': webarena_confile_file,
            }
        return data

    def get_random_sample(self, sorted_files, percentage=0.2):
        # 计算要随机选择的数量
        sample_size = int(len(sorted_files) * percentage)

        # 随机选择指定比例的文件
        random_sample = random.sample(sorted_files, sample_size)

        return random_sample

    def sorted_files(self, config_file_list):
        config_file_list = sorted(
            config_file_list,
            key=lambda x: int(x.split('/')[-1].split('.')[0]) if x.split('/')[-1].split('.')[0].isdigit() else float('inf')  # 确保如果不是数字，放到最后
        )
        return config_file_list

    def get_current_time(self):
        """
        获取当前本地时间，并以 'YYYY-MM-DD HH:MM:SS' 格式返回字符串。
        """
        # 1. 获取当前的datetime对象
        now = datetime.datetime.now()
        
        # 2. 使用 strftime 方法将其格式化为字符串
        # %Y: 四位数的年份 (e.g., 2023)
        # %m: 两位数的月份 (01-12)
        # %d: 两位数的日期 (01-31)
        # %H: 24小时制的两位数小时 (00-23)
        # %M: 两位数的分钟 (00-59)
        # %S: 两位数的秒 (00-59)
        formatted_string = now.strftime("%Y-%m-%d %H:%M:%S")
        
        return formatted_string

    def _sort_files_by_complexity_and_id(self, config_file_dir):
        """
        对目录下的文件进行多级排序：
        1. 按文件内容中的 'complexity' 字段 ('easy', 'medium', 'hard') 排序。
        2. 在相同复杂度内，按文件名中的数字ID排序。

        如果 'complexity' 缺失，则默认为 'hard'。
        """
        all_filenames = os.listdir(config_file_dir)
        
        files_with_metadata = []
        for filename in all_filenames:
            # 1. 从文件名提取任务ID
            try:
                task_id_str = filename.split('/')[-1].split('.')[0]
                task_id = int(task_id_str) if task_id_str.isdigit() else float('inf')
            except (ValueError, IndexError):
                task_id = float('inf')  # 格式不正确的文件放在最后

            # 2. 从文件内容读取 'complexity'
            complexity = 'hard'  # 默认值
            file_path = os.path.join(config_file_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 安全地获取嵌套的 'complexity' 值
                    complexity = data.get('labels', {}).get('complexity', 'hard')
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                # 如果文件无法读取或不是有效的JSON，则使用默认值 'hard'
                logger.warning(f"Warning: Could not read complexity from {filename}. Defaulting to 'hard'. Error: {e}")
            
            files_with_metadata.append({
                'filename': filename,
                'task_id': task_id,
                'complexity': complexity
            })

        # 3. 定义复杂度的排序顺序
        complexity_order = {'easy': 0, 'medium': 1, 'hard': 2}
        if "onlyeasy".lower() in EXPERIMENT_NAME.lower():
            files_with_metadata = [item for item in files_with_metadata if item['complexity'] == 'easy']

        # 4. 使用多级排序键进行排序
        sorted_files_metadata = sorted(
            files_with_metadata,
            key=lambda item: (
                complexity_order.get(item['complexity'], 2),  # 主键：复杂度
                item['task_id']                               # 次键：任务ID
            )
        )

        # 5. 返回排序后的文件名列表
        output_list = [item['filename'] for item in sorted_files_metadata]
        return output_list

    def uniform_by_start_url(self, config_files):
        """
        根据配置文件中的 'start_url' 对其进行分组，并返回一个均匀分布的列表。
        均匀分布意味着来自同一个域名的配置文件会尽可能地分散在列表中，而不是聚集在一起。

        Args:
            config_files (list): 包含配置文件路径的列表。

        Returns:
            list: 一个经过均匀排序的配置文件路径列表。
        """
        # 步骤 1: 按 domain_url 分组配置文件 (与你的原代码类似，但增加了健壮性)
        config_files_by_domain = defaultdict(list)
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                start_url = config_dict.get('start_url')
                if not start_url:
                    logger.warning(f"警告: 配置文件 '{config_file}' 中缺少 'start_url' 键，已跳过。")
                    continue

                # 从 'http://www.example.com/path' 中提取 'www.example.com'
                domain_url = start_url.split('/')[2]
                config_files_by_domain[domain_url].append(config_file)

            except FileNotFoundError:
                logger.warning(f"警告: 找不到文件 '{config_file}'，已跳过。")
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"警告: 处理文件 '{config_file}' 时出错: {e}，已跳过。")

        # 步骤 2: 交叉合并分组后的列表，实现均匀分布
        # 获取所有分组后的列表
        grouped_lists = list(config_files_by_domain.values())
        
        # 如果没有任何有效的分组，直接返回空列表
        if not grouped_lists:
            return []

        # 使用 itertools.zip_longest 进行交叉合并，就像发牌一样
        # 例如：[['a1', 'a2'], ['b1'], ['c1', 'c2', 'c3']]
        # 会变成：[('a1', 'b1', 'c1'), ('a2', None, 'c2'), (None, None, 'c3')]
        interleaved_tuples = zip_longest(*grouped_lists)

        # 展开并过滤掉 None 值
        uniform_list = [item for tpl in interleaved_tuples for item in tpl if item is not None]

        return uniform_list


    def filter_config_files(self, config_files):
        output_config_files = []
        for config_file_i in config_files:
            config_file_path = os.path.join(self.parquet_files, config_file_i)
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            eval_types = config_dict["eval"]["eval_types"]
            if "program_html" in eval_types: #url_match #program_html #string_match
                output_config_files.append(config_file_i)
        return output_config_files

    def _read_files_and_tokenize(self, split):
        # 使用新的排序函数，它会处理复杂度和ID的排序
        ttl_config_files = self._sort_files_by_complexity_and_id(self.parquet_files)
        val_config_files = []
        if len(VAL_DATASET_PATH) > 0 and os.path.exists(VAL_DATASET_PATH):
            val_config_files_name = os.listdir(VAL_DATASET_PATH)
            val_config_files = [os.path.join(VAL_DATASET_PATH, i) for i in val_config_files_name]
            #print("val_config_files: ", val_config_files)
        
        # --- 以下代码与您的原版几乎相同，只是删除了对 sorted_files 的调用 ---
        val_config_files_from_train = []
        if VLM_EXP_DEBUG == '1':
            # 筛选出start_url中包含dianping的文件
            ttl_config_files_tmp = []
            for ttl_config_files_i in ttl_config_files:
                ttl_config_files_i_path = os.path.join(self.parquet_files, ttl_config_files_i)
                with open(ttl_config_files_i_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                    #if "-W-" in config_dict.get('conflict_key', ''):
                    if True:
                        ttl_config_files_tmp.append(ttl_config_files_i)
            ttl_config_files = ttl_config_files_tmp
            if len(val_config_files) == 0:
                val_config_files_from_train = self.get_random_sample(ttl_config_files, percentage=0.2)
        else:
            if len(val_config_files) == 0:
                val_config_files_from_train = self.get_random_sample(ttl_config_files, percentage=0.01)
        # 注意：get_random_sample 会打乱顺序，如果您希望验证集也保持排序，
        # 则需要在采样后再次排序，或者使用切片代替随机采样。
        # 这里为了保持随机性，我们不再重新排序。如果需要，可以取消下面的注释。
        # val_config_files = self._sort_files_by_complexity_and_id(val_config_files) # 这需要修改函数以接受列表而非目录
        val_config_files = val_config_files
        train_config_files = []
        val_config_files_from_train = set(val_config_files_from_train) # 使用集合以提高查找效率
        for file_i in ttl_config_files:
            if file_i not in val_config_files_from_train:
                train_config_files.append(file_i)
                

        if VLM_EXP_DEBUG == '1':
            aaaa = 1
            # val_config_files = val_config_files[:17]
            # train_config_files = train_config_files[:11]
        # if VLM_EXP_DEBUG == '1':
        #     train_config_files = self.filter_config_files(train_config_files)
        #     val_config_files = self.filter_config_files(val_config_files)


        if split == 'train':
            config_files = train_config_files
            if DATA_SFUFFLE.lower() == 'true'.lower():
                logger.info("Shuffling dataset files...")   
                random.shuffle(config_files)
        else:
            # 确保验证集也是有序的（如果需要的话）
            # val_config_files会保持从ttl_config_files中采样时的相对顺序
            #config_files = [f for f in ttl_config_files if f in val_files_set]
            if len(val_config_files) == 0:
                config_files = sorted(val_config_files_from_train)
            else:
                config_files = sorted(val_config_files)

        #uniform by start url
        if "UniUrl".lower() in EXPERIMENT_NAME.lower():
            config_files = self.uniform_by_start_url(config_files)
        logger.info(f"{split} config_files: {len(config_files)}")

        if "ConSch".lower() in EXPERIMENT_NAME.lower():
            config_files_sch = []
            scheduler = TaskScheduler(batch_size=BATCH_SIZE)
            scheduler.load_tasks(input_path=self.parquet_files, input_list=config_files)
            batches = scheduler.schedule_tasks_fixed_batches_balanced(num_steps=(len(config_files) // BATCH_SIZE) + 1)
            for batch_i in batches:
                for task_i in batch_i:
                    config_files_sch.append(task_i['file'])
            config_files = config_files_sch

        self.dataframe = []
        for idx, config_file_i in enumerate(config_files):
            # read parquet files and cache
            config_file_path = os.path.join(self.parquet_files, config_file_i)
            data_dict = self.webarena2agent(idx, config_file_path, split)
            if data_dict is not None:
                self.dataframe.append(data_dict)
        logger.info(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        dataframe_tmp = []
        for idx, dataset_i in enumerate(self.dataframe):
            tokenizer_str = tokenizer.apply_chat_template(dataset_i[prompt_key], tools=self.tools, add_generation_prompt=True)
            if len(tokenizer_str) <= self.max_prompt_length:
                dataframe_tmp.append(dataset_i)
        self.dataframe = dataframe_tmp
        logger.info(f'filter dataset len: {len(self.dataframe)}')

    def try_getitem(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        chat = row_dict[self.prompt_key]
        current_time = self.get_current_time()
        #在user query中增加当前时间
        for chat_i in chat:
            if chat_i['role'] == 'user':
                chat_i['content'] = chat_i['content'] + f"Current time: {current_time}"
                break
        if self.use_custom_tool_format_func:
            if chat[0]['role'] == 'system':
                chat[0]['content'] = chat[0]['content'] + self.tool_env.tools_format_func()
            else:
                system_msg = [{"role": "system", "content": self.tool_env.tools_format_func()}]
                #system_msg = []
                #下边apply_chat_template有了tools description，所以不需要再添加system_msg
                # Convert chat to a list if it's not already one
                chat_list = chat.tolist() if hasattr(chat, 'tolist') else list(chat)
                chat = system_msg + chat_list
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, tools=self.tools, add_generation_prompt=True, tokenize=False)
        is_multi_modal = "screenshot" in self.image_key or "image" in self.image_key
        if is_multi_modal:  # expand image token
            if self.config.tool.env == "webbrowser":
                #image = self.get_screenshot_from_start_urls(row_dict, sample_env)
                image = self.get_background_screenshot()
            else:
                image = row_dict[self.image_key]
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [image]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index_image = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index_image].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index_image += 1
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
                #print("prompt_with_chat_template count: ", prompt_with_chat_template.count( self.processor.image_token))
        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        #print("input_ids[0]: ", (input_ids[0] == 151655).sum())
        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        # encode prompts without chat template
        row_dict['raw_prompt'] = chat
        row_dict['split'] = self.split
        return row_dict

    def __getitem__(self, item):
        try:
            return self.try_getitem(item)
        except Exception as e:
            # 如果是长度超出的 NotImplementedError，或者是其他异常但我们不在调试模式，则跳过该样本
            if isinstance(e, NotImplementedError) or VLM_EXP_DEBUG != '1':
                logger.error(f"Get error for item {item} {e}")
                # 尝试获取下一个样本，注意防止无限递归（简单的处理是模一下长度，或者由外部控制）
                # 这里沿用原逻辑，但需注意边界
                next_item = (item + 1) % len(self)
                return self.__getitem__(next_item)
            # 调试模式下，抛出非长度相关的异常
            raise e

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
    
 