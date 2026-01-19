"""base class for evaluation"""
# answer string match
from ast import List
import importlib
import json, json_repair
from tkinter import N, NO

import time
import collections
import html
import urllib.parse

from pathlib import Path
from typing import Any, Optional, Tuple, Union
from urllib.parse import urljoin

import evaluate  # type: ignore[import]
import requests
from beartype import beartype
from beartype.door import is_bearable
from nltk.tokenize import word_tokenize  # type: ignore
from PIL import Image
from playwright.async_api import CDPSession, Page
from demo.browser_env.utils import with_timeout_legacy, change_mainip2ecsip
from aiopath import AsyncPath # 引入 AsyncPath

from demo.browser_env.actions import Action, _id2key
from demo.browser_env.utils import StateInfo
from demo.evaluation_harness import image_utils
from tqdm import tqdm
import concurrent.futures
import traceback
from demo.browser_env.actions import _key2id
from demo.evaluation_harness.helper_functions import (
    PseudoPage,
    get_query_text,
    get_query_text_lowercase,
    gitlab_get_project_memeber_role,
    llm_fuzzy_match,
    llm_ua_match,
    reddit_get_latest_comment_content_by_username,
    reddit_get_latest_comment_obj_by_username,
    reddit_get_parent_comment_username_of_latest_comment_by_username,
    reddit_get_post_url,
    shopping_get_latest_order_url,
    shopping_get_num_reviews,
    shopping_get_order_product_name_list,
    shopping_get_order_product_option,
    shopping_get_order_product_quantity,
    shopping_get_product_attributes,
    shopping_get_product_price,
    shopping_get_rating_as_percentage,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
    shopping_get_sku_latest_review_text,
    qwen_vl_captioning_fn,
)
import uuid, os, sys
import numpy as np
import logging
import aiofiles # 导入 aiofiles
import asyncio  # 可能会用到，比如 to_thread
import shutil
from datetime import datetime
from demo.tool_utils import smart_resize, extract_answer, extract_coords_by_index, extract_tool_call_arguments
import math
from tqdm.asyncio import tqdm_asyncio
import functools # 需要导入functools

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
#logger.setLevel("DEBUG")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/webjudge')
sys.path.append(current_dir + '/webjudge/src')
sys.path.append(current_dir + '/webjudge/src/methods')
from demo.evaluation_harness.webjudge.src import vllm_run
from enum import IntEnum



Trajectory = list[Union[Action, StateInfo]]
VLM_ONLINERL_EXP_NAME = os.environ.get('VLM_ONLINERL_EXP_NAME', '')
SAVE_MODEL_PATH = os.environ.get('SAVE_MODEL_PATH', '')
TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR")
EXP_TRAJECTORY_SAVE_PATH = TENSORBOARD_DIR.replace("tensorboard", "trajectory_data")
VLM_EXP_DEBUG = os.environ.get('VLM_EXP_DEBUG', '0')
EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', "")
EXPERIMENT_NAME = str(EXPERIMENT_NAME)
print("EXP_TRAJECTORY_SAVE_PATH: ", EXP_TRAJECTORY_SAVE_PATH)
REWARD_COEFF = os.environ.get('REWARD_COEFF', 0.3)
REWARD_COEFF = float(REWARD_COEFF)


class Evaluator(object):
    def __init__(self, 
    eval_tag: str = "",
    save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
    max_saved_trajectories: int = 1000,
    REPLACE_WITH_YOUR_HOST: str = None
    ) -> None:
        self.eval_tag = eval_tag
        self.output_result = None
        self.save_path = Path(save_path) if save_path else None
        if EXP_TRAJECTORY_SAVE_PATH is not None and len(EXP_TRAJECTORY_SAVE_PATH) > 0:
            self.save_path = Path(EXP_TRAJECTORY_SAVE_PATH)

        if VLM_EXP_DEBUG == '1':
            max_saved_trajectories = 5
        self.max_saved_trajectories = max_saved_trajectories
        
        # 新增属性来存储训练信息
        self.training_step = 0
        self.should_save = False
        self.epoch = 0
        self.REPLACE_WITH_YOUR_HOST = REPLACE_WITH_YOUR_HOST
        # 如果提供了保存路径，则确保它存在
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            #logger.info(f"Trajectory data will be saved to: {self.save_path}")

    def set_training_info(self, training_step: int, should_save: Optional[bool], epoch: int, is_val: bool = False):
        """设置训练信息，控制是否保存轨迹"""
        #logger.info(f"set_training_info: {training_step}, {should_save}")
        self.training_step = training_step
        self.should_save = should_save
        self.epoch = epoch
        self.is_val = is_val

    async def save_trajectory_data(self, 
    task_id: str, 
    trajectory: Trajectory, 
    configs: dict, 
    score: float, 
    final_result_response: str,
    eval_type: str = "webjudge"):
       # --- 修改部分：根据step决定是否保存轨迹数据 ---
        if self.save_path and self.should_save:
            try:
                logger.info(f"Training step {self.training_step}: Saving trajectory data for task {task_id}")
                # 使用 await 来调用新的异步保存方法
                await self._save_trajectory_data(
                    task_id=task_id,
                    trajectory=trajectory,
                    configs=configs,
                    final_score=score,
                    #vlm_output=self.output_result,
                    vlm_output = {},
                    final_answer = {"answer": final_result_response},
                    webjudge_details = self.output_result['evaluation_details'] if self.output_result else {},
                    step_scores=[],
                    training_step=self.training_step,  # 新增：传递训练step
                    eval_type=eval_type
                )
                # _enforce_retention_policy 是同步的，如果它很快，可以直接调用
                # 如果它也涉及大量文件操作，也应该改造
                await self._enforce_retention_policy()
            except Exception as e:
                # 捕获到异常
                print(f"任务失败，捕获到异常: {e}")
                
                # 使用 traceback.format_exc() 获取完整的堆栈信息字符串
                error_stack_trace = traceback.format_exc()
                
                print("\n--- 完整的堆栈信息 ---")
                print(error_stack_trace)
                
                # 你可以返回这个字符串，或者将它写入日志文
                logger.error(f"Failed to save trajectory data for task {task_id} at step {self.training_step}: {e}")
        else:
            logger.info(f"Training step {self.training_step}: Skipping trajectory save (should_save={self.should_save})")
        # -----------------------------
    async def _save_trajectory_data(
        self,
        task_id: str,
        trajectory: Trajectory,
        configs: dict,
        final_score: float,
        vlm_output: dict,
        final_answer: dict,
        webjudge_details: dict,
        step_scores: list[float],
        training_step: int = 0,
        eval_type: str = "webjudge"
    ):
        """Saves trajectory, images, and evaluation results to a directory."""
        # 1. 创建本次任务的专属文件夹，在文件名中包含step信息
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step_{training_step:06d}" if training_step > 0 else "unknown_step"
        if self.is_val:
            step_str = f"val_{step_str}"
        task_dir_name = f"{step_str}_{task_id}_{timestamp}"
        task_path = self.save_path / task_dir_name
        images_path = task_path / "images"
        images_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving trajectory for task {task_id} to {task_path}")

        # 2. 准备要保存的结构化数据
        serializable_trajectory = []
        image_save_count = 0
        for i, step in enumerate(trajectory):
            step_data = {}
            # 处理 Action 步骤
            if 'action_type' in step:
                step_data['type'] = 'action'
                action_newdata = {}
                for k, v in step.items():
                    if isinstance(v, IntEnum):
                        action_newdata[k] = v.name
                    elif isinstance(v, np.ndarray):
                        action_newdata[k] = v.tolist()
                    else:
                        action_newdata[k] = v
                    if k == 'text':
                        action_newdata[k] = [_id2key[v_i] for v_i in v]
                step_data['action'] = action_newdata
                step_data['raw_prediction'] = step['raw_prediction']
            # 处理 State/Observation 步骤
            elif 'observation' in step:
                step_data['type'] = 'observation'
                #step_data['observation'] = {k: v for k, v in step['observation'].items() if k != 'image'}
                if 'image' in step['observation'] and isinstance(step['observation']['image'], np.ndarray):
                    try:
                        image = Image.fromarray(step['observation']['image'])
                        image_filename = f"step_{i}_img_{image_save_count}.png"
                        # 图片保存仍然是同步的，如果图片很多很大，也应该用 to_thread 包装
                        image.save(images_path / image_filename)
                        step_data['image_path'] = f"images/{image_filename}"
                        image_save_count += 1
                    except Exception as e:
                        logger.error(f"Could not save image for step {i}: {e}")
                        step_data['image_path'] = None
            else:
                step_data['type'] = 'unknown'
                step_data['content'] = str(step)

            if i < len(step_scores):
                step_data['step_reward'] = step_scores[i]
            
            serializable_trajectory.append(step_data)
        
        evaluation_data = {
            "task_id": task_id,
            "training_step": training_step,  # 新增：记录训练step
            "timestamp": timestamp,
            "intent": configs.get("intent", "N/A"),
            "start_url": configs.get("start_url", "N/A"),
            "final_score": final_score,
            "vlm_output": vlm_output,
            "trajectory_with_rewards": serializable_trajectory,
            "final_answer": final_answer,
            "webjudge_details": webjudge_details,
            "eval_type": eval_type
        }
        #logger.info(f"evaluation_data: {evaluation_data}")
        # 3. 异步写入JSON文件
        # 先将 dict 序列化为 string
        json_string = json.dumps(evaluation_data, indent=2, ensure_ascii=False)
        
        # 使用 aiofiles 异步写入
        async with aiofiles.open(task_path / "evaluation_data.json", "w", encoding="utf-8") as f:
            await f.write(json_string)

    async def _enforce_retention_policy(self):
        """
        [MODERN ASYNC VERSION] Deletes the oldest trajectories using a combination
        of aiopath for reading and asyncio.to_thread for deleting.
        """
        if not self.save_path:
            return

        save_path = AsyncPath(self.save_path)

        try:
            subdirs = [d async for d in save_path.iterdir() if await d.is_dir()]
        except FileNotFoundError:
            return

        if len(subdirs) <= self.max_saved_trajectories:
            return
            
        logger.info(f"Saved trajectories ({len(subdirs)}) exceed limit ({self.max_saved_trajectories}). Cleaning up...")
        
        stats_tasks = [d.stat() for d in subdirs]
        stats_results = await asyncio.gather(*stats_tasks)
        
        subdirs_with_stats = list(zip(subdirs, stats_results))
        subdirs_with_stats.sort(key=lambda item: item[1].st_mtime)

        #去除val开头的文件夹
        subdirs_with_stats = [d for d in subdirs_with_stats if not "val_" in d[0].name]
        num_to_delete = len(subdirs_with_stats) - self.max_saved_trajectories
        
        if num_to_delete <= 0:
            return

        dirs_to_delete = subdirs_with_stats[:num_to_delete]
        
        # --- START OF THE FIX ---
        
        # 5. 并发删除最旧的文件夹 (已修正)
        # shutil.rmtree 是一个阻塞函数，我们需要用 asyncio.to_thread 包装它。
        # aiopath.AsyncPath 对象可以像 pathlib.Path 对象一样直接传递给它。
        delete_tasks = [
            asyncio.to_thread(shutil.rmtree, d)  # `d` 是 AsyncPath 对象，shutil 会正确处理它
            for d, stat in dirs_to_delete
        ]

        # --- END OF THE FIX ---

        results = await asyncio.gather(*delete_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            dir_path = dirs_to_delete[i][0]
            if isinstance(result, Exception):
                logger.error(f"Error removing directory {dir_path.name}: {result}")
            else:
                logger.info(f"Removed old trajectory: {dir_path.name}")

    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page 
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        try:
            is_bearable(trajectory[-1], Action)
            last_action = trajectory[-1]
        except Exception:
            raise ValueError(
                "The last element of trajectory should be an action, add a fake stop action if needed"
            )

        return last_action  # type: ignore[return-value]

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        try:
            is_bearable(trajectory[-2], StateInfo)
            last_state = trajectory[-2]
        except Exception:
            raise ValueError(
                "The second last element of trajectory should be a state, add a fake stop action if needed"
            )

        return last_state  # type: ignore[return-value]


class NumericEvaluator(Evaluator):
    """Check if the numerical relationship is correct"""
    def __init__(self, 
        eval_tag: str = "",
        save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
        max_saved_trajectories: int = 1000,
        REPLACE_WITH_YOUR_HOST: str = None,
        ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)

        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False
    @staticmethod
    @beartype
    def str_2_int(s: str) -> Optional[int]:
        try:
            s = s.strip()
            if "," in s:
                s = s.replace(",", "")

            return int(s)
        except ValueError:
            # Return None if the string cannot be converted to int
            logger.error(f"[NumericEvaluator error]: Cannot convert {s} to int")
            return None

    @staticmethod
    @beartype
    def compare_inequality(
        value: Union[int, float], inequality: str, tol: float = 1e-8
    ) -> bool:
        """
        Compare a value (int or float) against an inequality string.

        Args:
        - value (int/float): The value to be compared.
        - inequality (str): Inequality in the form of "< 700", ">= 300", etc.
        - tol (float): Tolerance for floating point comparisons.

        Returns:
        - bool: True if the value satisfies the inequality, False otherwise.
        """
        # Extract the operator and the number from the inequality string
        ops = {
            "<=": lambda x, y: x <= y + tol,
            ">=": lambda x, y: x >= y - tol,
            "==": lambda x, y: abs(x - y) <= tol,
            "<": lambda x, y: x < y + tol,
            ">": lambda x, y: x > y - tol,
        }

        for op, func in ops.items():
            if op in inequality:
                _, num = inequality.split(op)
                return func(value, float(num.strip()))

        raise ValueError(f"Invalid inequality string: {inequality}")


class StringEvaluator(Evaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """
    def __init__(self, 
        eval_tag: str = "",
        save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
        max_saved_trajectories: int = 1000,
        REPLACE_WITH_YOUR_HOST: str = None,
        ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)

        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False
    @staticmethod
    @beartype
    def clean_answer(answer: str) -> str:
        answer = answer.strip()
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    @staticmethod
    @beartype
    def exact_match(ref: str, pred: str) -> float:
        return float(
            StringEvaluator.clean_answer(pred)
            == StringEvaluator.clean_answer(ref)
        )

    @staticmethod
    @beartype
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if (
            tokenize
            and len(clean_ref) == 1
            and len(word_tokenize(clean_ref)) == 1
        ):
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)

    @staticmethod
    @beartype
    def must_exclude(ref: str, pred: str) -> float:
        """Returns 1 if pred is not in ref, and 0 otherwise"""
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if len(word_tokenize(clean_ref)) == 1:
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref not in tok_pred)
        else:
            return float(clean_ref not in clean_pred)

    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(pred, ref, intent)

    @staticmethod
    @beartype
    def ua_match(ref: str, pred: str, intent: str) -> float:
        return llm_ua_match(pred, ref, intent)

    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page  | None = None
    ) -> float:

        with open(config_file, "r") as f:
            raw = f.read()
            raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
            configs = json.loads(raw)
        
        task_id = str(uuid.uuid4())
        last_action = self.get_last_action(trajectory)
        #answer = last_action["answer"]
        answer = extract_answer(solution_str)
        # logger.info(f"name StringEvaluator ... answer: {answer} solution_str: {solution_str} \n ------")
        # logger.info(f"name StringEvaluator ... reference_answers: REPLACE_WITH_YOUR_HOST {self.REPLACE_WITH_YOUR_HOST}  {configs}  \n ------")
        if answer is None:
            if REWARD_COEFF > 0:
                score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
                await self.save_trajectory_data(task_id, trajectory, configs, score, solution_str, eval_type="webjudge")
                return score
            else:
                await self.save_trajectory_data(task_id, trajectory, configs, 0.0, solution_str, eval_type="string")
                return 0.0
        pred = self.clean_answer(answer)
        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)
                case "required_values":
                    required_values = value
                    assert isinstance(required_values, list)
                    pred = NumericEvaluator.str_2_int(pred)
                    if pred is None:
                        score = 0.0
                    else:
                        for v in required_values:
                            value_or = v.split(" |OR| ")
                            score *= any(
                                [
                                    NumericEvaluator.compare_inequality(
                                        pred, value
                                    )
                                    for value in value_or
                                ]
                            )
                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                case "must_exclude":
                    assert isinstance(value, list)
                    for must_excl_value in value:
                        score *= self.must_exclude(
                            ref=must_excl_value, pred=pred
                        )
                case "one_of":
                    assert isinstance(value, list)
                    found = False
                    for one_of_value in value:
                        one_of_value = self.clean_answer(one_of_value)
                        if one_of_value in pred:
                            found = True
                            break
                    score = score * found
                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.fuzzy_match(
                                ref=reference, pred=pred, intent=intent
                            )
            # logger.info(f"step reward: {step_score}")

        if score == 0.0 and REWARD_COEFF > 0:
            score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
            await self.save_trajectory_data(task_id, trajectory, configs, score, pred, eval_type="webjudge")
        else:
            await self.save_trajectory_data(task_id, trajectory, configs, score, pred, eval_type="string")
        return score


class StringSoftEvaluator(Evaluator):
    """Use text generation metrics such as BLEU, ROUGE, etc. to evaluate the answer"""
    def __init__(self, 
        eval_tag: str = "",
        save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
        max_saved_trajectories: int = 1000,
        REPLACE_WITH_YOUR_HOST: str = None,
        ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)

        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False
    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page  | None = None
    ) -> float:
        with open(config_file, "r") as f:
            raw = f.read()
            raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
            configs = json.loads(raw)
        task_id = str(uuid.uuid4())
        last_action = self.get_last_action(trajectory)
        #pred = last_action["answer"]
        pred = extract_answer(solution_str)
        if pred is None:
            if REWARD_COEFF > 0:
                score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
                await self.save_trajectory_data(task_id, trajectory, configs, score, solution_str, eval_type="webjudge")
                return score
            else:
                await self.save_trajectory_data(task_id, trajectory, configs, 0.0, solution_str, eval_type="string_soft")
                return 0.0
        ref = configs["eval"]["reference_answers"]
        # rouge
        m = evaluate.load("rouge")
        rouge = m.compute(predictions=[pred], references=[ref])
        score = float(rouge["rouge1"])
        if score == 0.0 and REWARD_COEFF > 0:
            score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
            await self.save_trajectory_data(task_id, trajectory, configs, score, pred, eval_type="webjudge")
        else:
            await self.save_trajectory_data(task_id, trajectory, configs, score, pred, eval_type="string_soft")

        return score


class URLExactEvaluator(Evaluator):
    """Check whether the URL is exactly the same as of the reference URLs"""
    def __init__(self, 
        eval_tag: str = "",
        save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
        max_saved_trajectories: int = 1000,
        REPLACE_WITH_YOUR_HOST: str = None,
        ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)

        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False
    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page 
    ) -> float:
        with open(config_file, "r") as f:
            raw = f.read()
            raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
            configs = json.loads(raw)

        def clean_url(url: str) -> str:
            url = str(url)
            url = url.rstrip("/")
            return url

        def parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
            """Parse a URL into its base, path, and query components."""
            parsed_url = urllib.parse.urlparse(url)
            base_path = parsed_url.netloc + parsed_url.path
            query = urllib.parse.parse_qs(parsed_url.query)
            return base_path, query

        def parse_urls(
            urls: list[str],
        ) -> tuple[list[str], dict[str, set[str]]]:
            """Parse a list of URLs."""
            base_paths = []
            queries = collections.defaultdict(set)
            for url in urls:
                base_path, query = parse_url(url)
                base_paths.append(base_path)
                for k, v in query.items():
                    queries[k].update(v)
            return base_paths, queries

        pred = clean_url(page.url)
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = configs["eval"].get("url_note", "GOLD in PRED")
        logger.info(f"name ... answer: {ref_urls}\n ------")
        logger.info(f"name ... pred asnwer: {pred}  \n ------")
        score = 0.0
        if matching_rule == "GOLD in PRED":
            ref_base_paths, ref_queries = parse_urls(ref_urls)
            pred_base_paths, pred_query = parse_url(pred)

            base_score = float(
                any(
                    [
                        ref_base_path in pred_base_paths
                        for ref_base_path in ref_base_paths
                    ]
                )
            )
            query_score = 1.0
            for k, possible_values in ref_queries.items():
                query_score *= float(
                    any(
                        possible_ref_value in pred_query.get(k, [])
                        for possible_ref_value in possible_values
                    )
                )
            score = base_score * query_score
        elif matching_rule == "EXACT":
            if pred in ref_urls:
                score = 1.0
            else:
                score = 0.0
        else:
             raise ValueError(f"Unknown matching rule: {matching_rule}")
        
        task_id = str(uuid.uuid4())
        if score == 0.0 and REWARD_COEFF > 0:
            score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
            await self.save_trajectory_data(task_id, trajectory, configs, score, pred, eval_type="webjudge")
        else:
            await self.save_trajectory_data(task_id, trajectory, configs, score, pred, eval_type="url")
        return score


class HTMLContentExactEvaluator(Evaluator):
    """Check whether the contents appear in the page"""
    def __init__(self, 
        eval_tag: str = "",
        save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
        max_saved_trajectories: int = 1000,
        REPLACE_WITH_YOUR_HOST: str = None,
        ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)

        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False
    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page 
    ) -> float:
        with open(config_file, "r") as f:
            raw = f.read()
            raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
            configs = json.loads(raw)

        targets = configs["eval"]["program_html"]
        #logger.info(f"name ... answer: {targets}\n ------")
        score = 1.0
        for target in targets:
            target_url: str = target["url"]  # which url to check
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                if "shopping_get_latest_order_url" in func:
                    target_url = shopping_get_latest_order_url(self.REPLACE_WITH_YOUR_HOST)
                elif "shopping_get_sku_latest_review_author" in func:
                    new_func = func.strip()[:-1] + f", REPLACE_WITH_YOUR_HOST='{self.REPLACE_WITH_YOUR_HOST}')"
                    target_url = eval(new_func)
                elif "shopping_get_sku_latest_review_rating" in func:
                    new_func = func.strip()[:-1] + f", REPLACE_WITH_YOUR_HOST='{self.REPLACE_WITH_YOUR_HOST}')"
                    target_url = eval(new_func)
                elif "shopping_get_sku_latest_review_text" in func:
                    new_func = func.strip()[:-1] + f", REPLACE_WITH_YOUR_HOST='{self.REPLACE_WITH_YOUR_HOST}')"
                    target_url = eval(new_func)
                elif "shopping_get_sku_latest_review_title" in func:
                    new_func = func.strip()[:-1] + f", REPLACE_WITH_YOUR_HOST='{self.REPLACE_WITH_YOUR_HOST}')"
                    target_url = eval(new_func)
                elif "shopping_get_sku_product_page_url" in func:
                    new_func = func.strip()[:-1] + f", REPLACE_WITH_YOUR_HOST='{self.REPLACE_WITH_YOUR_HOST}')"
                    target_url = eval(new_func)
                else:
                    target_url = eval(func)
                
                if asyncio.iscoroutine(target_url):
                    target_url = await target_url

            locator: str = target["locator"]  # js element locator
            print("name ... target_url: ", target_url)
            print("name ... locator: ", locator)
            # navigate to that url
            if target_url != "last":
                try:
                    await page.goto(target_url, timeout=30 * 1000 * 100)
                    await page.wait_for_load_state("networkidle", timeout=30 * 1000 * 100)
                except Exception as e:
                    print(f"Navigating to {target_url} failed: {e}")
                    pass
                    

            # empty, use the full page
            if not locator.strip():
                selected_element = await page.content()
            # use JS to select the element
            elif locator.startswith("document.") or locator.startswith(
                "[...document."
            ):
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            await page.evaluate(f"() => {prep_action}")
                    except Exception:
                        pass
                try:
                    selected_element = str(await page.evaluate(f"() => {locator}"))
                    if not selected_element:
                        selected_element = ""
                except Exception as e:
                    # the page is wrong, return empty
                    print("name ... locator error: ", locator, e)
                    selected_element = ""
            elif locator.startswith("lambda:"):
                try:
                    locator = locator.lstrip("lambda:")
                    selected_element = await page.evaluate(locator)
                    if not selected_element:
                        selected_element = None
                except Exception:
                    # the page is wrong, return empty
                    selected_element = None
            # run program to call API
            elif locator.startswith("func:"):  # a helper function
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                try:
                    selected_element = eval(func)
                    if asyncio.iscoroutine(selected_element):
                        selected_element = await selected_element
                except Exception as e:
                    print("name ... func error: ", func, e)
                    # the page is wrong, return empty
                    selected_element = None
            else:
                raise ValueError(f"Unknown locator: {locator}")
            # If the selected element is None, then the page is wrong
            if selected_element is None:
                score = 0.0
                break
            
            # Unescape HTML entities
            selected_element = html.unescape(selected_element)

            print("name ... selected_element: ", selected_element, "----", "required_contents: ", target["required_contents"])
            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                print("name ... required_contents: ", required_contents)
                print("name ... selected_element: ", selected_element)
                score *= StringEvaluator.exact_match(
                    ref=required_contents, pred=selected_element
                )
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    score *= any(
                        [
                            StringEvaluator.must_include(
                                ref=content,
                                pred=selected_element,
                                tokenize=False,
                            )
                            for content in content_or
                        ]
                    )
            elif "must_exclude" in target["required_contents"]:
                required_contents = target["required_contents"]["must_exclude"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    assert " |OR| " not in content
                    score *= StringEvaluator.must_exclude(
                        content, pred=selected_element
                    )
            elif "required_values" in target["required_contents"]:
                required_values = target["required_contents"][
                    "required_values"
                ]
                assert isinstance(required_values, list)
                if isinstance(selected_element, str):
                    selected_element = NumericEvaluator.str_2_int(
                        selected_element
                    )
                if selected_element is None:
                    score = 0.0
                else:
                    for value in required_values:
                        value_or = value.split(" |OR| ")
                        score *= any(
                            [
                                NumericEvaluator.compare_inequality(
                                    selected_element, value
                                )
                                for value in value_or
                            ]
                        )
            elif "fuzzy_match" in target["required_contents"]:
                targets = target["required_contents"]["fuzzy_match"]
                assert isinstance(targets, str)
                targets = targets.split(" |OR| ")
                for target in targets:
                    score *= max(
                        [
                            StringEvaluator.fuzzy_match(
                                ref=target,
                                pred=selected_element,
                                intent="NOT USED",
                            )
                        ]
                    )
            else:
                raise ValueError(
                    f"Unknown required_contents: {target['required_contents'].keys()}"
                )

        task_id = str(uuid.uuid4())
        if score == 0.0 and REWARD_COEFF > 0:
            score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
            await self.save_trajectory_data(task_id, trajectory, configs, score, "", eval_type="webjudge")
        else:
            await self.save_trajectory_data(task_id, trajectory, configs, score, "", eval_type="html")
        return score

def normalize_string_integer(input_str, max_value):
    """
    将输入的字符串整数转换为 0 到 1 之间的归一化浮点数。

    参数:
    input_str (str): 输入的字符串整数。
    max_value (int): 最大值，用于归一化。

    返回:
    float: 归一化后的浮点数。
    """
    try:
        # 将字符串转换为整数
        value = int(input_str)
        
        # 检查输入值是否在有效范围内
        if value < 0 or value > max_value:
            raise ValueError("输入值不在有效范围内")
        
        # 计算归一化后的值
        normalized_value = value / max_value
        return normalized_value
    except ValueError as e:
        logger.error(f"输入错误: {e}")
        return None



class PageImageEvaluator(Evaluator):
    """Check whether the answer is correct by querying a vision model."""
    def __init__(self, 
        eval_tag: str = "",
        save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
        max_saved_trajectories: int = 1000,
        REPLACE_WITH_YOUR_HOST: str = None,
        ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)

        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False
    def __init__(self, captioning_fn):
        self.captioning_fn = captioning_fn
        # Default to 0.8 as the threshold for similarity to account for compression, resizing, etc
        # This might be too generous but we bias towards minimizing false negatives.
        self.ssim_threshold = 0.8

    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page  | None = None
    ) -> float:
        with open(config_file, "r") as f:
            raw = f.read()
            raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
            configs = json.loads(raw)

        for query in configs["eval"]["page_image_query"]:
            locator: str = query["eval_image_class"]
            target_url: str = query["eval_image_url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = await eval(func)

            # navigate to that url
            if target_url != "last":
                await page.goto(target_url, timeout=30 * 1000 * 100)
                time.sleep(3)  # TODO(jykoh): fix this hard-coded sleep

            # empty, use the full page
            if not locator.strip():
                images = await page.get_by_role("img").all()
            # use JS to select the element
            elif locator.startswith("."):
                # Get all img children under the locator
                elements = await page.query_selector_all(locator)
                images = []
                for element in elements:
                    is_img = await element.evaluate(
                        'element => element.tagName === "IMG"'
                    )
                    if is_img:
                        images.append(element)
                    else:
                        images.extend(await element.query_selector_all("img"))
            else:
                raise ValueError(f"Unknown locator: {locator}")

            if images == []:
                return 0.0

            all_image_pixels = []
            for image in images:
                try:
                    # Get image from URL.
                    image_url = image.get_attribute("src")
                    if not image_url.startswith(
                        ("http://", "https://", "www.")
                    ):
                        image_url = urljoin(page.url, image_url)
                    image = Image.open(
                        requests.get(image_url, stream=True).raw
                    )
                    all_image_pixels.append(image)
                except Exception as e:
                    logger.warning(f"[WARNING]: {e}")

            score = 1.0
            if all_image_pixels == []:
                return 0.0
            else:
                # Run the VQA eval on the image elements.
                eval_vqas = query.get("eval_vqa", [])
                assert (
                    len(eval_vqas) > 0 or "eval_fuzzy_image_match" in query
                ), "eval_vqa must have at least 2 questions or eval_fuzzy_image_match must be True"
                for qa in eval_vqas:
                    question, answer = qa["question"], qa["answer"]
                    prompt = f"Q: {question} A:"
                    # pred_ans = self.captioning_fn(
                    #     all_image_pixels, [prompt] * len(all_image_pixels)
                    # )
                    pred_ans = qwen_vl_captioning_fn(all_image_pixels, [prompt] * len(all_image_pixels))
                    score *= float(
                        any(
                            [answer.lower() in ans.lower() for ans in pred_ans]
                        )
                    )

                if "eval_fuzzy_image_match" in query:
                    ssim_threshold = query.get(
                        "ssim_threshold", self.ssim_threshold
                    )
                    exact_match_imgs = query["eval_fuzzy_image_match"].split(
                        " |OR| "
                    )
                    all_exact_match_pixels = []

                    for exact_match_img in exact_match_imgs:
                        if exact_match_img.startswith("http"):
                            exact_match_pixels = Image.open(
                                requests.get(exact_match_img, stream=True).raw
                            )
                        else:
                            exact_match_pixels = Image.open(exact_match_img)
                        all_exact_match_pixels.append(exact_match_pixels)

                    # Check if any of the images on the page match
                    found_exact_match = False
                    for exact_match_pixels in all_exact_match_pixels:
                        for image_pixels in all_image_pixels:
                            ssim = image_utils.get_image_ssim(
                                image_pixels, exact_match_pixels
                            )
                            if ssim > ssim_threshold:
                                found_exact_match = True
                                break
                    score *= float(found_exact_match)

        return score


class EvaluatorComb:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page 
    ) -> float:

        score = 1.0
        for evaluator in self.evaluators:
            cur_score = await evaluator(solution_str, trajectory, config_file, page)
            score *= cur_score
        return score

class WebjudgeEvaluator(Evaluator):
    """
    Check whether the answer is correct by querying a vision model.
    Additionally, saves the evaluated trajectory data and images for review.
    """

    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page  | None = None
    ) -> float:
        task_id = str(uuid.uuid4())
        trajectory_images = []
        max_image_num = 15  # 截断长度

        with open(config_file, "r") as f:
            raw = f.read()
            raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
            configs = json.loads(raw)
        result_map = {}
        result_map['task_id'] = task_id
        result_map['task'] = configs["intent"]
        final_result_response = extract_answer(solution_str)
        if "JudgewoAnswer" in EXPERIMENT_NAME:
            final_result_response = ''
        result_map['final_result_response'] = final_result_response if final_result_response is not None and final_result_response not in ('...', '-') else ''

        action_history = []
        step_score_tag = []
        image_order = 0
        act_order = 0
        last_traj = None
        for ind, traj_i in enumerate(trajectory):
            if 'action_type' in traj_i:
                ele_map = {}
                # ele_map['index'] = traj_i['element_id']
                # ele_map['text'] = ''.join([_id2key[id] for id in traj_i['text']]) if isinstance(traj_i['text'], list) and all(isinstance(x, int) for x in traj_i['text']) else traj_i['text']
                ele_map['url'] = traj_i['url']
                # ele_map['x'] = float(traj_i['coords'][0])
                # ele_map['y'] = float(traj_i['coords'][1])
                action_str = json.dumps(ele_map, ensure_ascii=False) + ' -> ' + (str(traj_i['action_type'].name) if isinstance(traj_i['action_type'], IntEnum) else str(traj_i['action_type']))
                action_history.append(action_str)
                step_score_tag.append(0)  # 0 表示action
                act_order += 1
            else:
                if 'observation' in traj_i:
                    if 'image' in traj_i['observation']:
                        if last_traj and 'observation' in last_traj and 'image' in last_traj['observation']:
                            step_score_tag.append(2)  # 2 表示连续image
                        else:
                            screenshot = traj_i['observation']['image']
                            image = Image.fromarray(screenshot)
                            trajectory_images.append(image)
                            image_order += 1
                            step_score_tag.append(1)  # 1 表示新image
                        # 注意：这里我们不再在评估期间保存图片到临时路径
                        # 图片将会在评估结束后，连同所有数据一起保存
                        continue
                step_score_tag.append(-1)
            last_traj = traj_i

        result_map['action_history'] = action_history[1-max_image_num:].copy()
        
        self.output_result = await vllm_run.simple_eval(trajectory_images[-max_image_num:], result_map)
        complete_score = float(self.output_result['complete_score'])
        action_score = float(self.output_result['action_score'])
        traj_score = float(self.output_result['traj_score'])
        if complete_score == 5.0:
            score = 5.0 
        else:
            score = complete_score * 0.8 + action_score * 0.1 + traj_score * 0.1
        if complete_score < 0:
            score_norm = -1.0
        else:
            score_norm = normalize_string_integer(max(score-1, 0), 4)

        logger.info(f"webjudge score: {score}")
        logger.info(f"webjudge score normalized: {score_norm}")
        # logger.info(f"step reward: {step_score}")
        await self.save_trajectory_data(task_id, trajectory, configs, score, final_result_response if final_result_response is not None else "", eval_type="webjudge")
        return score_norm

class BboxJudgeEvaluator(Evaluator):
    """
    判断模型最后一次工具调用的坐标是否在给定的ground truth矩形框（bounding box）内。
    """
    def __init__(self, 
        eval_tag: str = "",
        save_path: Optional[Path | str] = SAVE_MODEL_PATH+"/trajectory_data/",
        max_saved_trajectories: int = 1000,
        REPLACE_WITH_YOUR_HOST: str = None,
        ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)

        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False

    def is_point_in_bbox(self, point_xy: Tuple[int, int], bbox: list[int]) -> bool:
        """
        检查一个点 (x, y) 是否在一个矩形框 [x1, y1, x2, y2] 内部（包含边界）。

        Args:
            point_xy: (x, y) 坐标点。
            bbox: [x1, y1, x2, y2] 格式的矩形框。

        Returns:
            如果在框内则返回 True, 否则返回 False。
        """
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return False
        x, y = point_xy
        x1, y1, x2, y2 = bbox
        # 确保 x1 <= x2 和 y1 <= y2
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        return min_x <= x <= max_x and min_y <= y <= max_y

    async def __call__(
        self,
        solution_str: str,
        trajectory: Trajectory,  # 此评估器中未使用，但保留以符合接口
        config_file: Path | str,
        page: Page  | None = None  # 此评估器中未使用，但保留以符合接口
    ) -> float:
        """
        执行评估：判断最后一次操作的坐标是否在 ground truth 矩形框内。

        Args:
            solution_str: 包含模型生成内容的完整字符串。
            trajectory: 智能体执行轨迹。
            config_file: 配置文件路径，应包含 ground truth bbox。
            page: 当前页面对象。

        Returns:
            如果坐标在矩形框内，返回 1.0，否则返回 0.0。
        """
        # 1. 从配置文件加载 ground truth bounding box
        task_id = str(uuid.uuid4())
        final_result_response = extract_answer(solution_str) 
        if final_result_response is None:
            final_result_response = ""
        try:
            with open(config_file, "r") as f:
                raw = f.read()
                raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
                configs = json.loads(raw)
            # 假设 ground truth bbox 存储在 'eval' -> 'ground_truth_bbox'
            ground_truth_bbox = configs["eval"]["reference_answers"]['reference_answer_raw_annotation'] # x1, y1, x2, y2
            original_width = configs["viewsize"]["viewport_width"]
            original_height = configs["viewsize"]["viewport_height"]
            resized_height, resized_width = smart_resize(original_height, original_width, 28, 512 * 512, 2048 * 2048)
            ground_truth_bbox = [int(ground_truth_bbox[0]/original_width * resized_width)  , 
            int(ground_truth_bbox[1]/original_height * resized_height), 
            int(ground_truth_bbox[2]/original_width * resized_width), 
            int(ground_truth_bbox[3]/original_height * resized_height)] # x1, y1, x2, y2
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            return 0.0

        # 2. 从 solution_str 中提取预测的坐标
        pred_coords = extract_coords_by_index(solution_str, index=0)
        if pred_coords is None:
            return 0.0

        # 3. 检查坐标点是否在 ground truth 矩形框内
        is_inside = self.is_point_in_bbox(point_xy=pred_coords, bbox=ground_truth_bbox)
        score = float(is_inside)
        if score == 0.0 and REWARD_COEFF > 0:
            score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
            await self.save_trajectory_data(task_id, trajectory, configs, score, final_result_response, eval_type="webjudge")
        else:
            await self.save_trajectory_data(task_id, trajectory, configs, score, final_result_response, eval_type="bbox_judge")
        return score


def evaluator_router(
    config_file: Path | str, captioning_fn=None, REPLACE_WITH_YOUR_HOST: str = None
) -> EvaluatorComb:
    """Router to get the evaluator class"""

    with open(config_file, "r") as f:
            raw = f.read()
            raw = change_mainip2ecsip(raw, REPLACE_WITH_YOUR_HOST) if REPLACE_WITH_YOUR_HOST else raw
            configs = json.loads(raw)

    eval_types = configs["eval"]["eval_types"]
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
            case "url_match":
                evaluators.append(URLExactEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
            case "program_html":
                evaluators.append(HTMLContentExactEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
            case "page_image_query":
                evaluators.append(PageImageEvaluator(captioning_fn, REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
            case "webjudge":
                evaluators.append(WebjudgeEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
            case "bbox_judge":
                evaluators.append(BboxJudgeEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
            case "action_judge":
                evaluators.append(ActionJudgeEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators)


# --- Constants & Config ---
# Simplified weights. The score is a blend of "did you pick the right tool?"
# and "did you provide the right arguments?"
REWARD_WEIGHTS = {
    "action_name": 0.3,
    "arguments": 0.7,
}

# Distance decay for numerical values
MAX_REWARD_DISTANCE = 200  # pixels in resized space
DISTANCE_DECAY_RATE = 3.0 / MAX_REWARD_DISTANCE

# Default resizing parameters
DEFAULT_RESIZE_PARAMS = {
    "patch_size": 28,
    "min_res": 512 * 512,
    "max_res": 2048 * 2048
}

# --- Main Evaluator Implementation ---

class ActionJudgeEvaluator(Evaluator):
    """
    Compares predicted and GT actions with continuous rewards and unified
    argument handling. (Version 5)
    """
    def __init__(self,
        eval_tag: str = "action_judge",
        save_path: Optional[Union[Path, str]] = None,
        max_saved_trajectories: int = 1000,
        reward_weights: dict[str, float] = None,
        distance_decay_rate: float = DISTANCE_DECAY_RATE,
        resize_params: dict[str, int] = None,
        REPLACE_WITH_YOUR_HOST: str = None,
    ):
        super().__init__(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.reward_weights = reward_weights or REWARD_WEIGHTS
        self.distance_decay_rate = distance_decay_rate
        self.resize_params = resize_params or DEFAULT_RESIZE_PARAMS
        if not math.isclose(sum(self.reward_weights.values()), 1.0):
            raise ValueError("Reward weights must sum to 1.0")
        self.webjudge_evaluator = WebjudgeEvaluator(eval_tag, save_path, max_saved_trajectories, REPLACE_WITH_YOUR_HOST)
        self.webjudge_evaluator.should_save = False
        # logger.info("ActionJudgeEvaluator (v5) initialized with unified argument handling.")
        # logger.info(f"Reward weights: {self.reward_weights}")

    def _get_distance_reward(self, p1: list[float], p2: list[float]) -> float:
        if not (isinstance(p1, list) and len(p1) == 2 and isinstance(p2, list) and len(p2) == 2):
            return 0.0
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        dis_score = math.exp(-self.distance_decay_rate * distance)
        return dis_score

    def _get_numerical_reward(self, v1: float, v2: float, max_diff: float) -> float:
        diff = abs(v1 - v2)
        decay_rate = 3.0 / max_diff if max_diff > 0 else float('inf')
        num_score = math.exp(-decay_rate * diff)
        return num_score

    async def __call__(self, solution_str: str, trajectory: Trajectory, config_file: Union[Path, str], page: Optional[Union[Page, PseudoPage]] = None) -> float:
        task_id = str(uuid.uuid4())
        final_result_response = ""

        try:
            with open(config_file, "r") as f:
                raw = f.read()
                raw = change_mainip2ecsip(raw, self.REPLACE_WITH_YOUR_HOST) if self.REPLACE_WITH_YOUR_HOST else raw
                configs = json.loads(raw)

            gt_actions_original = configs["eval"]["reference_answers"]['reference_answer_raw_annotation']
            if not isinstance(gt_actions_original, list) or not gt_actions_original:
                logger.error("[ActionJudge] Ground truth actions are invalid or empty.")
                return 0.0

            # Scale GT coordinates to the resized space
            original_width = configs["viewsize"]["width"]
            original_height = configs["viewsize"]["height"]
            resized_height, resized_width = smart_resize(original_height, original_width, 28, 512 * 512, 2048 * 2048)
            gt_actions_scaled = []
            for gt_action in gt_actions_original:
                scaled_action = json.loads(json.dumps(gt_action))
                if "coords" in scaled_action.get("arguments", {}):
                    orig_coords = scaled_action["arguments"]["coords"]
                    if isinstance(orig_coords, list) and len(orig_coords) == 2:
                        scaled_action["arguments"]["coords"] = [
                            int(orig_coords[0] * resized_width / original_width),
                            int(orig_coords[1] * resized_height / original_height)
                        ]
                gt_actions_scaled.append(scaled_action)

            # Extract predicted actions
            predicted_actions = []
            for step in trajectory:
                if 'action_type' in step and 'raw_prediction' in step:
                    tool_call_str = extract_tool_call_arguments(step['raw_prediction'])
                    predicted_actions.append(json_repair.loads(tool_call_str) if tool_call_str else None)
            # Compare sequences
            num_steps = len(gt_actions_scaled)
            step_scores = []
            for i in range(num_steps):
                pred_action = predicted_actions[i] if i < len(predicted_actions) else None
                gt_action = gt_actions_scaled[i]
                step_scores.append(self._compare_single_action(i, pred_action, gt_action))

            final_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
            logger.info(f"[ActionJudge] Step-wise scores: {[f'{s:.2f}' for s in step_scores]}")
            logger.info(f"[ActionJudge] Final average score: {final_score:.4f}")

        except Exception as e:
            logger.error(f"[ActionJudge] CRITICAL ERROR: {e}", exc_info=True)
            final_score = 0.0
        if final_score == 0.0 and REWARD_COEFF > 0:
            final_score = REWARD_COEFF * await self.webjudge_evaluator(solution_str, trajectory, config_file, page)
            await self.save_trajectory_data(task_id, trajectory, configs, final_score, final_result_response, eval_type="webjudge")
        else:
            await self.save_trajectory_data(task_id, trajectory, configs, final_score, final_result_response, eval_type="action_judge")
        return final_score

    def _compare_single_action(self, index: int, pred_action: Optional[dict], gt_action: dict) -> float:
        """Compares a single action and returns a continuous score between 0 and 1."""
        if pred_action is None:
            #logger.warning(f"[ActionJudge] Step {index+1}: No valid predicted action found. Score for this step is 0.")
            return 0.0

        pred_name = pred_action.get("name")
        gt_name = gt_action.get("name")

        # 1. Score for Action Name
        name_score = 1.0 if pred_name.lower() == gt_name.lower() else 0.0
        
        # If action name is wrong, we can still give partial credit for arguments if they are coincidentally correct,
        # but the weighted score will be low.
        
        # 2. Score for Arguments
        pred_args = pred_action.get("arguments", {})
        gt_args = gt_action.get("arguments", {})
        
        avg_args_score = 1.0 # Default if GT has no arguments
        if gt_args:
            arg_scores = []
            # Iterate through all arguments defined in the ground truth
            for key, gt_value in gt_args.items():
                pred_value = pred_args.get(key)
                param_score = 0.0
                
                if key == "coords":
                    param_score = self._get_distance_reward(pred_value, gt_value)
                elif isinstance(gt_value, (int, float)) and isinstance(pred_value, (int, float)):
                    max_diff = 100 if key == 'distance' else 5 if key == 'seconds' else 3
                    param_score = self._get_numerical_reward(pred_value, gt_value, max_diff)
                elif isinstance(gt_value, str) and isinstance(pred_value, str):
                    param_score = 1.0 if gt_value.lower().strip() == pred_value.lower().strip() else 0.0
                else: # Strict equality for other types (booleans, etc.)
                    if key == 'press_enter_after' and (pred_value is None and gt_value == 0):
                        param_score = 1.0 # Treat model's default (None) as 0
                    else:
                        param_score = 1.0 if gt_value == pred_value else 0.0
                arg_scores.append(param_score)

            if arg_scores:
                avg_args_score = sum(arg_scores) / len(arg_scores)

        # 3. Final Weighted Score for this step
        weighted_score = (
            self.reward_weights["action_name"] * name_score +
            self.reward_weights["arguments"] * avg_args_score
        )

        return weighted_score


# 1. 创建一个“线程工作函数”
# 这个函数将在每个线程中被执行。它是同步的。
def run_evaluation_in_thread(result, save_path):
    """
    这个函数由线程池中的一个线程执行。
    它负责为单个 'result' 运行评估。
    """
    # 在这个函数内部，我们为这个线程创建一个独立的 Evaluator 实例
    # 这对于线程安全至关重要
    eval_i = WebjudgeEvaluator(save_path=save_path)
    eval_i.should_save = True
    eval_i.is_val = True

    # --- 关键部分 ---
    # 因为 eval_i() 是一个异步函数 (coroutine)，我们不能直接调用它。
    # 我们需要在当前线程中启动一个临时的 asyncio 事件循环来运行它。
    # asyncio.run() 正好可以完成这个任务。
    try:
        output_result = asyncio.run(eval_i(
            solution_str="",
            trajectory=result["trajectory"],
            config_file=result["config_file"],
            page=None
        ))
        return output_result
    except Exception as e:
        # 捕获异常很重要，否则线程可能会悄无声息地失败
        print(f"Error processing task with config {result.get('config_file')}: {e}")
        return None # 或者返回一个错误标记

async def offline_eval():
    # TODO: 配置您的离线评估路径
    offline_file_path = os.getenv("OFFLINE_FILE_PATH", "./models/webagent_online/")
    val_file_path = os.getenv("VAL_FILE_PATH", "./config_files/wa/weiyuan_task/")
    sample_names = os.listdir(offline_file_path) #[:2]
    val_sample_names = os.listdir(val_file_path)
    val_config_dict = {}
    for val_sample_name in val_sample_names:
        val_sample_i = os.path.join(val_file_path, val_sample_name)
        with open(val_sample_i, "r") as f:
            val_sample = json.load(f)
            val_sample["config_file"] = val_sample_i
            val_config_dict[val_sample["task_id"]] = val_sample
    results_list = []
    for sample_name in sample_names:
        if "val_" not in sample_name:
            continue
        sample_path = os.path.join(offline_file_path, sample_name)
        task_id = sample_name
        trajectory = []
        if os.path.isdir(sample_path):
            trajectory_path = os.path.join(sample_path, "trajectory.json")
            if os.path.exists(trajectory_path):
                with open(trajectory_path, "r") as f:
                    action_history = json.load(f)
                if len(action_history) == 0:
                    continue
                for item in action_history:
                    print(item)
                    if item["type"] == "action":
                        traj_action: Action = {}
                        input_action = item["action"]
                        if 'params' in input_action:
                            if 'coordinate_x' in input_action['params'] and 'coordinate_y' in input_action['params']:
                                traj_action['coords'] = np.array([input_action['params']['coordinate_x'], input_action['params']['coordinate_y']])
                            else:
                                traj_action['coords'] = np.array([0, 0])
                        else:
                            traj_action['coords'] = np.array([0, 0])
                        if 'index' in input_action['params']:
                            traj_action['nth'] = input_action['params']['index']
                        else:
                            traj_action['nth'] = 0
                        if 'text' in input_action['params']:
                            traj_action['text'] = [ _key2id[i]for i in input_action['params']['text']]
                        else:
                            traj_action['text'] = []
                        if 'url' in input_action['params']:
                            traj_action['url'] = input_action['params']['url']
                        else:
                            traj_action['url'] = ""
                        traj_action['action_type'] = input_action['action_type']
                        traj_action['raw_prediction'] = ''
                        trajectory.append(traj_action)
                    
                    elif item["type"] == "observation":
                        image_path = os.path.join(sample_path, item["image_path"])
                        image = Image.open(image_path)
                        observation = {"text": "", "image": np.array(image)}
                        state_info: StateInfo = {"observation": observation, "info": {"page": ""}}
                        trajectory.append(state_info)
                        
            task_id = int(sample_name.split("_")[-1])
            result = {
                "config_file": val_config_dict[task_id]['config_file'],
                "trajectory": trajectory
            }
            results_list.append(result)

    CONCURRENCY_LIMIT = 40
    save_path = offline_file_path + "trajectory_data/"
    all_outputs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        # 使用 executor.submit 提交所有任务
        # 这会立即返回一个 future 对象，代表未来的结果
        future_to_result = {
            executor.submit(run_evaluation_in_thread, result, save_path): result
            for result in results_list
        }
        
        print(f"已提交 {len(results_list)} 个任务到线程池，并发上限为 {CONCURRENCY_LIMIT}...")

        # 4. 使用 as_completed 和 tqdm 来获取结果并显示进度条
        # as_completed 会在任何一个 future 完成时就 yield 它
        for future in tqdm(concurrent.futures.as_completed(future_to_result), total=len(results_list), desc="Processing evaluations"):
            try:
                # 获取已完成任务的结果
                output_result = future.result()
                if output_result is not None:
                    all_outputs.append(output_result)
            except Exception as exc:
                # 如果任务本身抛出异常，.result() 会重新引发它
                config_file = future_to_result[future].get('config_file', 'Unknown')
                print(f'Task for config {config_file} generated an exception: {exc}')

    print("\n所有任务执行完毕！正在打印结果...")
    for output in all_outputs:
        print(output)
        

async def test_val_506():
    from playwright.async_api import async_playwright
    import os
    import json
    
    # Configuration content derived from the provided trajectory and user instructions
    config_content = {
      "sites": [
        "reddit"
      ],
      "task_id": 615,
      "require_login": True,
      "storage_state": "./.auth/reddit_state.json",
      "start_url": "http://REPLACE_WITH_YOUR_HOST:9999/f/pics",
      "geolocation": None,
      "intent_template": "Re-post the image of {{content}} in this page to {{subreddit}} subreddit and note \"from /f/pics\"",
      "instantiation_dict": {
        "content": "Bald Eagle",
        "subreddit": "earthporn"
      },
      "intent": "Re-post the image of Bald Eagle in this page to earthporn subreddit and note \"from /f/pics\"",
      "require_reset": False,
      "eval": {
        "eval_types": [
          "url_match",
          "program_html"
        ],
        "reference_answers": None,
        "reference_url": "http://REPLACE_WITH_YOUR_HOST:9999/f/earthporn",
        "program_html": [
          {
            "url": "func:reddit_get_post_url('__last_url__')",
            "locator": "document.querySelector('.submission__inner').outerText",
            "required_contents": {
              "must_include": [
                "from /f/pics"
              ]
            }
          },
          {
            "url": "func:reddit_get_post_url('__last_url__')",
            "locator": "[...document.querySelector('.submission__inner').querySelectorAll('[href],[src]')].map(elem => elem.getAttribute('href') || elem.getAttribute('src')).join(' ')",
            "required_contents": {
              "must_include": [
                "b02113033af32feae9ff147dbbe3764039368d67d193885bd04e65c2e6beea9c.jpg"
              ]
            }
          }
        ],
        "url_note": "GOLD in PRED"
      },
      "intent_template_id": 11,
      "conflict_key": "REDDIT-W-POST_CREATE-earthporn"
    }
    
    config_path = "temp_val_361_config.json"
    try:
        with open(config_path, "w") as f:
            json.dump(config_content, f)

        REPLACE_WITH_YOUR_HOST = "120.26.36.48"
        evaluator = URLExactEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST)
        
        # Load storage state if available to handle login
        storage_state_path = "/root/oagent_training_onlineRL/Agent-R1/log_120.26.36.48/.auth/reddit_state.json"
        context_args = {}
        if os.path.exists(storage_state_path):
            print(f"Loading storage state from {storage_state_path}")
            context_args["storage_state"] = storage_state_path
        else:
            print("Storage state not found, proceeding without login state (might fail)")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            context = await browser.new_context(**context_args)
            page = await context.new_page()
            
            target_url = "http://120.26.36.48:9999/f/EarthPorn/2/amazing-shot-of-a-blue-jay-pestering-a-bald-eagle"
            # print(f"Navigating to {target_url}...")
            try:
                start_time = time.time()
                await page.goto(target_url, timeout=10 * 60 * 1000)
                print("Navigation successful.")
                end_time = time.time()
                print(f"Navigation time: {end_time - start_time} seconds")
                #await page.wait_for_timeout(10 * 60 * 1000)
            except Exception as e:
                print(f"Navigation failed: {e}")
                
            trajectory = []
            solution_str = "<answer>The product has been added to the cart.</answer>"
            
            print("Running evaluator HTMLContentExactEvaluator...")
            try:
                score = await evaluator(solution_str, trajectory, config_path, page)
                print(f"Evaluation finished. Score: {score}")
            except Exception as e:
                print(f"Evaluation failed with error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                await browser.close()
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

# --- Main execution block for testing ---
if __name__ == '__main__':

    print("Running test_val_506...")
    asyncio.run(test_val_506())
    # evaluator = StringEvaluator(REPLACE_WITH_YOUR_HOST="11111")
    # solution_str = "I want to buy a laptop"
    # perfect_traj = [
    #     {"raw_prediction": "<tool_call>{\"name\": \"click\", \"arguments\": {\"coords\": [503, 303]}}</tool_call>", "action_type": "click"},
    #     {"raw_prediction": "<tool_call>{\"name\": \"type\", \"arguments\": {\"coords\": [402, 202], \"content\": \"hello\"}}</tool_call>", "action_type": "type"},
    #     {"raw_prediction": "<tool_call>{\"name\": \"go_back\", \"arguments\": {}}</tool_call>", "action_type": "go_back"},
    # ]
    # config_file = "/ainative/muti-modal/yuhang/438262/projects/gui_agent/visualwebarena/config_files/wa_ali/test_webarena_conflict/1.json"
    # page = None
    # score = asyncio.run(evaluator(solution_str, perfect_traj, config_file, page))
    # print(f"Score: {score}")
    #asyncio.run(offline_eval())
    
    # (Assuming mock base classes and helpers are defined for standalone testing)
    # async def run_test_v5():
    #     logger.info("\n" + "="*50)
    #     logger.info("  Running Tests for ActionJudgeEvaluator (v5 - Unified Args)")
    #     logger.info("="*50)

    #     mock_config_file = Path("mock_config_v5.json")
    #     mock_config_data = {
    #       "url": "https://example.com", "intent": "Test intent",
    #       "viewsize": {"width": 1280, "height": 720},
    #       "eval": {
    #         "eval_types": ["action_judge"],
    #         "reference_answers": {
    #           "reference_answer_raw_annotation": [
    #             {"name": "click", "arguments": {"coords": [500, 300]}},
    #             {"name": "type", "arguments": {"coords": [400, 200], "content": "hello"}},
    #             {"name": "go_back", "arguments": {}}, # Action with no arguments
    #           ]
    #         }
    #       }
    #     }
    #     with open(mock_config_file, "w", encoding="utf-8") as f:
    #         json.dump(mock_config_data, f, ensure_ascii=False)
        
    #     evaluator = ActionJudgeEvaluator()

    #     # Test Case 1: Perfect Match
    #     logger.info("\n--- Test Case 1: Perfect Match ---")
    #     perfect_traj = [
    #         {"raw_prediction": "<tool_call>{\"name\": \"click\", \"arguments\": {\"coords\": [503, 303]}}</tool_call>", "action_type": "click"},
    #         {"raw_prediction": "<tool_call>{\"name\": \"type\", \"arguments\": {\"coords\": [402, 202], \"content\": \"hello\"}}</tool_call>", "action_type": "type"},
    #         {"raw_prediction": "<tool_call>{\"name\": \"go_back\", \"arguments\": {}}</tool_call>", "action_type": "go_back"},
    #     ]
    #     score = await evaluator(solution_str="", trajectory=perfect_traj, config_file=mock_config_file, page=None)
    #     print(f"Test Case 1 (Perfect Match) Score: {score:.4f}")
    #     #assert math.isclose(score, 1.0)

    #     # Test Case 2: Wrong name, but args are coincidentally correct
    #     logger.info("\n--- Test Case 2: Wrong name, correct args ---")
    #     wrong_name_traj = [
    #         {"raw_prediction": "<tool_call>{\"name\": \"hover\", \"arguments\": {\"coords\": [503, 303]}}</tool_call>", "action_type": "hover"}, # Should be click
    #         {"raw_prediction": "<tool_call>{\"name\": \"type\", \"arguments\": {\"coords\": [402, 202], \"content\": \"hello\"}}</tool_call>", "action_type": "type"},
    #         {"raw_prediction": "<tool_call>{\"name\": \"go_back\", \"arguments\": {}}</tool_call>", "action_type": "go_back"},
    #     ]
    #     score = await evaluator(solution_str="", trajectory=wrong_name_traj, config_file=mock_config_file, page=None)
    #     print(f"Test Case 2 (Wrong Name) Score: {score:.4f}")
    #     # Step 1 score: 0.5 * (0 for name) + 0.5 * (1 for args) = 0.5
    #     ## Final score: (0.5 + 1.0 + 1.0) / 3 = 2.5 / 3
    #     #assert math.isclose(score, 2.5 / 3.0)

    #     # Test Case 3: Correct name, one arg wrong in a multi-arg action
    #     logger.info("\n--- Test Case 3: Correct name, one arg wrong ---")
    #     one_arg_wrong_traj = [
    #         {"raw_prediction": "<tool_call>{\"name\": \"click\", \"arguments\": {\"coords\": [503, 303]}}</tool_call>", "action_type": "click"},
    #         {"raw_prediction": "<tool_call>{\"name\": \"type\", \"arguments\": {\"coords\": [410, 210], \"content\": \"goodbye\"}}</tool_call>", "action_type": "type"},
    #         {"raw_prediction": "<tool_call>{\"name\": \"go_back\", \"arguments\": {}}</tool_call>", "action_type": "go_back"},
    #     ]
    #     score = await evaluator(solution_str="", trajectory=one_arg_wrong_traj, config_file=mock_config_file, page=None)
    #     print(f"Test Case 3 (One Arg Wrong) Score: {score:.4f}")
    #     # Step 2: name_score=1. Coords have slight error, text is 0.
    #     # Coords distance = sqrt(10^2+10^2) = 14.14. Reward = exp(- (3/200) * 14.14) ~= 0.8
    #     # Avg args score = (0.8 + 0.0) / 2 = 0.4
    #     # Step 2 score = 0.5 * 1 + 0.5 * 0.4 = 0.7
    #     # Final score = (1.0 + 0.7 + 1.0) / 3 = 2.7 / 3
    #     #assert math.isclose(score, 2.7 / 3.0, abs_tol=0.01)

    #     # Cleanup
    #     os.remove(mock_config_file)
    
    # To run the test, you'd need the full script with base classes etc.
    # The following line is for demonstration.
    #asyncio.run(run_test_v5())
