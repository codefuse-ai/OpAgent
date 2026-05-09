# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import asyncio
import os
import uuid
import time
from pprint import pformat
import json
import logging
import concurrent.futures
import random
from collections import defaultdict
from itertools import zip_longest
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse

import ray
import torch
import numpy as np
from ray import ObjectRef

from recipe.webagent_fully_async_policy.detach_utils import merge_rollout_sample_flattened_steps
from recipe.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
)
from verl.trainer.ppo.metric_utils import (
    process_validation_metrics,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from recipe.webagent_fully_async_policy.detach_utils import prepare_webagent_generation_data
from recipe.fully_async_policy.message_queue import MessageQueueClient
from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.reward_manager import NaiveRewardManager
from verl import DataProto
# Import the new class instead of defining FullyAsyncRollouter
from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouterBase

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR", "")
TRAJECTORY_DATA = TENSORBOARD_DIR.replace("tensorboard", "trajectory_data")

WEBARENA_AUTH_PATH = os.environ.get("WEBARENA_AUTH_PATH", "")

class WebAgentRewardManager(NaiveRewardManager):

    def __init__(self, config, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.config = config
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            reward = data_item.non_tensor_batch['reward_score']
  
            reward_extra_info = data_item.non_tensor_batch.get("extra_fields", {})['reward_extra_info']

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)

                for key, value in reward_extra_info.items():
                    print(f"[{key}]", value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

@ray.remote(num_cpus=10, max_concurrency=100)
class WebAgentFullyAsyncRollouter(FullyAsyncRollouterBase):
    """
    WebAgent specific Async Rollouter.
    Inherits from FullyAsyncRollouter and overrides the manager initialization
    to use FullyAsyncWebAgentLoopManager.
    Also overrides dataset creation to use WebAgentRLDataset.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = WebAgentRewardManager(
            config, tokenizer, num_examine=0, 
        )
        self.val_reward_fn = WebAgentRewardManager(
            config, tokenizer, num_examine=3,
        )
        
        self.config.actor_rollout_ref.rollout.agent.default_agent_loop = "async_web_agent"
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        # Re-create datasets using WebAgentRLDataset
        print("[WebAgentFullyAsyncRollouter] Re-creating datasets using WebAgentRLDataset...")

        # Import WebBrowserEnv locally to avoid circular imports if any
        from recipe.webagent_fully_async_policy.agent_rl_dataset import WebAgentRLDataset, collate_fn
        from recipe.webagent_fully_async_policy.browser_env.tool_env import WebBrowserEnv
        from verl.trainer.main_ppo import create_rl_sampler

        tool_env = WebBrowserEnv(
            with_wplaywright=False,
            headless=not self.config.tool.webbrowser.render,
            slow_mo=self.config.tool.webbrowser.slow_mo,
            observation_type=self.config.tool.webbrowser.observation_type,
            current_viewport_only=self.config.tool.webbrowser.current_viewport_only,
            viewport_size={
                "width": self.config.tool.webbrowser.viewport_width,
                "height": self.config.tool.webbrowser.viewport_height,
            },
            save_trace_enabled=self.config.tool.webbrowser.save_trace_enabled,
            sleep_after_execution=self.config.tool.webbrowser.sleep_after_execution,
            # NOTE: captioning_fn here is used for LLM + captioning baselines.
            # This can be different from the captioning model used for evals.
            captioning_fn=self.config.tool.webbrowser.caption_image_fn,
        )

        # Handle parquet_files input format. If it's a list (from config), take the first element if expected to be a dir.
        # WebAgentRLDataset expects a directory path string or list of files?
        # Looking at agent_rl_dataset.py, it now handles list input correctly (taking [0] if list).
        # So passing config.data.train_files directly is fine if agent_rl_dataset.py is updated.
        # However, user REVERTED agent_rl_dataset.py changes.
        # So I MUST fix the input here to match what the original WebAgentRLDataset expects (a directory string).

        train_files_input = self.config.data.train_files
        if isinstance(train_files_input, list) and len(train_files_input) > 0:
            train_files_input = train_files_input[0]

        val_files_input = self.config.data.val_files
        if isinstance(val_files_input, list) and len(val_files_input) > 0:
            val_files_input = val_files_input[0]

        # Re-create train dataset
        self.train_dataset = WebAgentRLDataset(
            parquet_files=train_files_input,
            tokenizer=self.tokenizer,
            processor=self.processor,
            tool_env=tool_env,
            split="train",
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.get('truncation', 'error'),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            use_custom_tool_format_func=self.config.data.get('use_custom_tool_format_func', False),
            config=self.config)

        # Re-create val dataset
        val_dataset = WebAgentRLDataset(
            parquet_files=val_files_input,
            tokenizer=self.tokenizer,
            processor=self.processor,
            tool_env=tool_env,
            split="test",
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.get('truncation', 'error'),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            use_custom_tool_format_func=self.config.data.get('use_custom_tool_format_func', False),
            config=self.config)

        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        self._validate_config()
        print(f"[WebAgentFullyAsyncRollouter] Rollouter _create_dataloader...\n{self.train_dataset}\n{val_dataset}")

        self._create_dataloader(self.train_dataset, val_dataset, collate_fn, train_sampler)
        print("[WebAgentFullyAsyncRollouter] Datasets re-created successfully.")

        # Initialize assignments
        self.task_assignments = {}

        # ==================== fully async config ====================

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[WebAgentFullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0  # 当前参数版本号，用于同步Trainer的参数
        self.total_generated_samples = 0  # 累计生成的有效样本总数 (成功放入MQ的样本)
        self.staleness_samples = 0  # 当前版本已生成的样本数 (用于并发控制，超过阈值会暂停生成，直到参数更新重置此值)
        self.dropped_stale_samples = 0  # 被丢弃的样本数 (因无效/报错/低分等原因未进入MQ的样本)
        self.processed_sample_count = 0  # 累计处理的样本总数 (Processor处理过的所有样本，包含丢弃的)
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()
        self.result_queue = asyncio.Queue()
        self.cancel_queue = asyncio.Queue()

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.webagent_fully_async_policy.agent_loop import FullyAsyncWebAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncWebAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )

    def _get_gen_batch(self, batch: DataProto, epoch: int, global_steps: int, validate: bool) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        gen_batch.non_tensor_batch["epoch"] = np.array([epoch] * len(gen_batch), dtype=object)
        gen_batch.non_tensor_batch["global_steps"] = np.array([global_steps] * len(gen_batch), dtype=object)
        gen_batch.non_tensor_batch["validate"] = np.array([validate] * len(gen_batch), dtype=object)

        return gen_batch

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing IP address."""
        try:
            parsed = urlparse(url)
            if ':' in parsed.netloc:
                port = parsed.netloc.split(':')[-1]
                return f"{parsed.scheme}://PORT_{port}"
            return f"{parsed.scheme}://{parsed.netloc}"
        except:
            return url

    def _dropip_url(self, url: str) -> str:
        """
        标准化URL，忽略IP地址/域名的差异，只保留端口和路径
        支持处理包含 |AND| 分隔符的多个URL
        例如：
        http://host1:7770/photosmart-plus-b209.html
        http://host2:7770/photosmart-plus-b209.html
        都会被标准化为：:7770/photosmart-plus-b209.html

        对于多URL的情况：
        http://host1:7770/page1.html |AND| http://host2:7770/page2.html
        会被标准化为排序后的形式（忽略IP差异）
        """
        if not url:
            return url
        
        # 检查是否包含 |AND| 分隔符
        if ' |AND| ' in url or '|AND|' in url:
            # 分割多个URL
            urls = [u.strip() for u in url.replace(' |AND| ', '|AND|').split('|AND|')]
            # 标准化每个URL
            normalized_urls = []
            for u in urls:
                try:
                    parsed = urlparse(u)
                    port = f":{parsed.port}" if parsed.port else ""
                    path = parsed.path or "/"
                    query = f"?{parsed.query}" if parsed.query else ""
                    fragment = f"#{parsed.fragment}" if parsed.fragment else ""
                    normalized = f"{port}{path}{query}{fragment}"
                    normalized_urls.append(normalized)
                except Exception:
                    normalized_urls.append(u)
            
            # 排序后用 |AND| 连接，确保顺序一致
            return ' |AND| '.join(sorted(normalized_urls))
        else:
            # 单个URL的情况
            try:
                parsed = urlparse(url)
                # 只保留端口和路径（以及query和fragment如果有的话）
                port = f":{parsed.port}" if parsed.port else ""
                path = parsed.path or "/"
                query = f"?{parsed.query}" if parsed.query else ""
                fragment = f"#{parsed.fragment}" if parsed.fragment else ""
                normalized = f"{port}{path}{query}{fragment}"
                return normalized
            except Exception as e:
                # 如果解析失败，返回原始URL
                return url

    def _get_sample_key(self, intent: str, start_url: str) -> str:
        """Create a unique key for a sample based on intent and normalized start_url."""
        normalized_url = self._dropip_url(start_url)
        return f"{intent}|||{normalized_url}"
    
    def _load_validated_samples(self, current_step: int) -> Dict[str, int]:
        """
        Load already validated samples from previous validation runs for the current step.
        
        Returns:
            Dict mapping sample_key to count of validations
        """
        validated_samples = defaultdict(int)
        
        trajectory_data_dir = Path(TRAJECTORY_DATA)
        if not trajectory_data_dir.exists():
            return validated_samples
        
        step_str = f"val_step_{current_step:06d}_"
        val_dirs = [d for d in trajectory_data_dir.iterdir() 
                   if d.is_dir() and d.name.startswith(step_str)]
        
        for val_dir in val_dirs:
            eval_data_file = val_dir / "evaluation_data.json"
            if not eval_data_file.exists():
                continue
            
            try:
                with open(eval_data_file, 'r') as f:
                    eval_data = json.load(f)
                
                intent = eval_data.get('intent', '')
                start_url = eval_data.get('start_url', '')
                
                if intent and start_url:
                    sample_key = self._get_sample_key(intent, start_url)
                    validated_samples[sample_key] += 1
                    
            except Exception as e:
                continue
        
        if validated_samples:
            logger.info(f"[Validation Resume] Loaded {len(validated_samples)} unique validated samples "
                       f"with total {sum(validated_samples.values())} validations for step {current_step}")
        
        return validated_samples
    
    def _reorder_indices_by_url_uniformly(self, dataset, indices: List[int]) -> List[int]:
        """
        Reorder indices to distribute samples from the same URL uniformly across batches.
        
        This prevents clustering of same-URL samples in one batch, which improves
        validation efficiency and resource utilization.
        
        Args:
            dataset: The dataset object to extract URL information from
            indices: List of sample indices to reorder
            
        Returns:
            Reordered list of indices with uniform URL distribution
        """
        # Group indices by domain URL
        indices_by_domain = defaultdict(list)
        
        for idx in indices:
            try:
                sample = dataset[idx]
                # Extract start_url from sample
                # Sample structure: sample['extra_info']['config'] contains the JSON string with start_url
                domain_url = 'unknown'
                
                # Try dict-like access first (most common for validation dataset)
                if isinstance(sample, dict) and 'extra_info' in sample:
                    extra_info = sample['extra_info']
                    
                    # If extra_info has 'config' field (it's a JSON string)
                    if isinstance(extra_info, dict) and 'config' in extra_info:
                        config_str = extra_info['config']
                        if isinstance(config_str, str):
                            config_dict = json.loads(config_str)
                        else:
                            config_dict = config_str
                        
                        start_url = config_dict.get('start_url', '')
                        if start_url:
                            domain_url = self._normalize_url(start_url)
                    # Fallback: check if start_url is directly in extra_info
                    elif isinstance(extra_info, dict) and 'start_url' in extra_info:
                        start_url = extra_info['start_url']
                        if start_url:
                            domain_url = self._normalize_url(start_url)
                # Fallback: Try attribute access
                elif hasattr(sample, 'extra_info') and sample.extra_info:
                    extra_info = sample.extra_info
                    if isinstance(extra_info, str):
                        config_dict = json.loads(extra_info)
                        start_url = config_dict.get('start_url', '')
                        if start_url:
                            domain_url = self._normalize_url(start_url)
                
            except Exception as e:
                logger.debug(f"Failed to extract URL from sample {idx}: {e}")
                domain_url = 'unknown'
            
            indices_by_domain[domain_url].append(idx)
        
        # If only one domain or less, no need to reorder
        if len(indices_by_domain) <= 1:
            logger.info(f"[URL Distribution] Only {len(indices_by_domain)} domain(s) found, skipping reordering")
            return indices
        
        # Interleave indices from different domains using zip_longest
        # This ensures samples from same domain are distributed evenly
        # Example: [['a1','a2'], ['b1'], ['c1','c2','c3']]
        # -> [('a1','b1','c1'), ('a2',None,'c2'), (None,None,'c3')]
        # -> ['a1', 'b1', 'c1', 'a2', 'c2', 'c3']
        grouped_indices = list(indices_by_domain.values())
        interleaved_tuples = zip_longest(*grouped_indices)
        uniform_indices = [idx for tpl in interleaved_tuples for idx in tpl if idx is not None]
        
        logger.info(f"[URL Distribution] Reordered {len(uniform_indices)} samples across {len(grouped_indices)} domains")
        domain_dist = [(domain, len(indices)) for domain, indices in list(indices_by_domain.items())[:10]]
        logger.info(f"[URL Distribution] Top domain distribution: {domain_dist}")
        if len(indices_by_domain) > 10:
            logger.info(f"[URL Distribution] ... and {len(indices_by_domain) - 10} more domains")
        else:
            logger.info(f"[URL Distribution] All domains: {list(indices_by_domain.keys())}")
        
        return uniform_indices
    
    def _create_filtered_val_dataloader(self, required_validations: int):
        """
        Create a filtered validation dataloader that only includes samples needing validation.
        
        This method filters out samples that have already been validated sufficiently.
        It's called before the validation loop starts, ensuring clean separation of concerns.
        
        Args:
            required_validations: Number of validations required per sample
            
        Returns:
            Filtered dataloader or original dataloader if filtering not needed
        """
        validated_samples = self._load_validated_samples(self.global_steps)
        
        # if not validated_samples:
        #     logger.info("[Validation Resume] No previous validation data found, validating all samples")
        #     return self.val_dataloader
        
        # Extract dataset from dataloader
        dataset = self.val_dataloader.dataset
        
        # Filter samples
        filtered_indices = []
        skipped_count = 0
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            
            # Extract intent and start_url from sample
            extra_info = sample.get('extra_info')
            if extra_info is not None:
                # Handle potential list wrapping
                if isinstance(extra_info, list) and len(extra_info) > 0:
                    extra_info = extra_info[0]
                
                config_str = extra_info.get('config', '{}') if isinstance(extra_info, dict) else '{}'
                try:
                    if isinstance(config_str, str):
                        config_dict = json.loads(config_str)
                    else:
                        config_dict = config_str
                    
                    intent = config_dict.get('intent', '')
                    start_url = config_dict.get('start_url', '')
                    
                    if intent and start_url:
                        sample_key = self._get_sample_key(intent, start_url)
                        current_validations = validated_samples.get(sample_key, 0)
                        
                        if current_validations >= required_validations:
                            skipped_count += 1
                            continue  # Skip this sample
                        else:
                            filtered_indices.append(idx)
                            if current_validations > 0:
                                logger.debug(f"Sample {idx}: {current_validations}/{required_validations} done, will continue")
                    else:
                        filtered_indices.append(idx)  # Can't determine, include it
                except Exception as e:
                    filtered_indices.append(idx)  # Error parsing, include it
            else:
                filtered_indices.append(idx)  # No extra_info, include it
        
        # Log statistics
        total_samples = len(dataset)
        samples_to_validate = len(filtered_indices)
        logger.info(f"[Validation Resume] Filtering results:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Already completed: {skipped_count}")
        logger.info(f"  Need validation: {samples_to_validate}")
        logger.info(f"  Skip rate: {skipped_count/total_samples*100:.1f}%")
        
        # If all samples are already validated, return empty iterator
        if samples_to_validate == 0:
            logger.info("[Validation Resume] All samples already validated, skipping validation")
            # Return an empty iterable that behaves like a dataloader
            return iter([])
        
        # If no samples skipped, return original dataloader
        # if skipped_count == 0:
        #     logger.info("[Validation Resume] No samples skipped, using original dataloader")
        #     return self.val_dataloader
        
        # Create filtered dataset using torch.utils.data.Subset
        from torch.utils.data import Subset
        from torchdata.stateful_dataloader import StatefulDataLoader
        
        # Reorder indices to distribute same-URL samples uniformly
        uniform_indices = self._reorder_indices_by_url_uniformly(dataset, filtered_indices)
        
        # Create filtered dataset with uniformly distributed indices
        filtered_dataset = Subset(dataset, uniform_indices)
        
        # Create new StatefulDataLoader with same parameters as original
        # This maintains consistency with the original val_dataloader type
        # Note: We keep shuffle=False for validation resume to ensure consistent ordering
        filtered_dataloader = StatefulDataLoader(
            dataset=filtered_dataset,
            batch_size=self.val_dataloader.batch_size,
            num_workers=getattr(self.val_dataloader, 'num_workers', 0),  # Match original or default to 0
            shuffle=False,  # Always False for validation to ensure reproducibility
            drop_last=False,  # Keep all samples in validation
            collate_fn=getattr(self.val_dataloader, 'collate_fn', None),  # Use original collate_fn
        )
        
        logger.info(f"[Validation Resume] Created filtered StatefulDataLoader with batch_size={filtered_dataloader.batch_size}")
        
        return filtered_dataloader

    async def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        
        # Create filtered dataloader that excludes already validated samples
        required_validations = self.config.actor_rollout_ref.rollout.val_kwargs.n
        filtered_dataloader = self._create_filtered_val_dataloader(required_validations)

        # Create progress bar for validation (count by samples, not batches)
        try:
            # Try to get total from filtered dataloader
            if hasattr(filtered_dataloader, 'dataset'):
                total_val_samples = len(filtered_dataloader.dataset)
            elif hasattr(filtered_dataloader, '__len__'):
                total_val_samples = len(filtered_dataloader)
            else:
                total_val_samples = None
        except:
            total_val_samples = None
        if total_val_samples is None or total_val_samples == 0:
            logger.info("[Validation Resume] No validation samples found, skipping validation")
            return {}
        progress_bar = tqdm(total=total_val_samples, desc="Validation Progress", unit="sample")

        # Use async pipeline processing for better throughput
        # Control max concurrent batches to avoid overwhelming the system
        # Can be configured via data.val_max_concurrent_batches
        # If not set, auto-calculate based on batch_size and available workers
        max_concurrent_batches = self.config.data.get("val_max_concurrent_batches", None)
        
        if max_concurrent_batches is None:
            # Auto-calculate: smaller batch_size allows higher concurrency
            val_batch_size = self.config.data.get("val_batch_size", 1)
            num_workers = self.config.actor_rollout_ref.rollout.agent.get("num_workers", 4)
            
            # Smart default: allow more concurrency for smaller batches
            if val_batch_size <= 1:
                max_concurrent_batches = min(num_workers // 2, 4)  # Up to 4 for single samples
            elif val_batch_size <= 2:
                max_concurrent_batches = min(num_workers // 3, 3)  # Up to 3 for small batches
            else:
                max_concurrent_batches = 2  # Conservative for larger batches
            
            max_concurrent_batches = max(1, max_concurrent_batches)  # At least 1
            logger.info(f"[Validation] Auto-set val_max_concurrent_batches={max_concurrent_batches} "
                       f"(val_batch_size={val_batch_size}, num_workers={num_workers})")
        
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        # Lock for thread-safe updates to shared data structures and progress bar
        update_lock = asyncio.Lock()
        
        async def process_batch(test_data):
            """Process a single validation batch asynchronously"""
            async with semaphore:
                return await self._process_single_validation_batch(
                    test_data,
                    sample_inputs,
                    sample_outputs,
                    sample_gts,
                    sample_scores,
                    sample_turns,
                    sample_uids,
                    reward_extra_infos_dict,
                    data_source_lst,
                    progress_bar,
                    update_lock
                )
        
        # Dynamic task pool approach - avoid blocking event loop
        # Don't create all tasks at once; instead, maintain a pool of active tasks
        pending_tasks = set()
        dataloader_iter = iter(filtered_dataloader)
        has_more_data = True
        
        # Initialize task pool with max_concurrent_batches tasks
        for _ in range(max_concurrent_batches):
            try:
                test_data = next(dataloader_iter)
                task = asyncio.create_task(process_batch(test_data))
                pending_tasks.add(task)
            except StopIteration:
                has_more_data = False
                break
        
        # Process batches dynamically: when one completes, add a new one
        while pending_tasks:
            # Wait for at least one task to complete
            done, pending_tasks = await asyncio.wait(
                pending_tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks (handle exceptions if any)
            for task in done:
                try:
                    await task  # This will raise if task had an exception
                except Exception as e:
                    logger.error(f"[Validation] Batch processing failed: {e}", exc_info=True)
            
            # Replenish task pool with new batches
            if has_more_data:
                for _ in range(len(done)):
                    try:
                        test_data = next(dataloader_iter)
                        task = asyncio.create_task(process_batch(test_data))
                        pending_tasks.add(task)
                    except StopIteration:
                        has_more_data = False
                        break
        
        # Close progress bar after validation loop
        progress_bar.close()
        
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict
    
    async def _process_single_validation_batch(
        self,
        test_data,
        sample_inputs,
        sample_outputs,
        sample_gts,
        sample_scores,
        sample_turns,
        sample_uids,
        reward_extra_infos_dict,
        data_source_lst,
        progress_bar,
        update_lock
    ):
        """Process a single validation batch - extracted for concurrent processing
        
        Args:
            update_lock: asyncio.Lock for thread-safe updates to shared data structures
        """
        test_batch = DataProto.from_single_dict(test_data)

        if "uid" not in test_batch.non_tensor_batch:
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

        # Store original batch size before repeat (for progress bar update)
        original_batch_size = len(test_batch.batch)

        # repeat test batch
        test_batch = test_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
        )

        # we only do validation on rule-based rm
        if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            return

        # Prepare data (no lock needed for local processing)
        input_ids = test_batch.batch["input_ids"]
        # TODO: Can we keep special tokens except for padding tokens?
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        
        ground_truths = [
            item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
        ]
        
        # Update shared data structures with lock
        async with update_lock:
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            sample_gts.extend(ground_truths)

        test_gen_batch = self._get_gen_batch(test_batch, epoch=0, global_steps=self.global_steps, validate=True)

        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
            "global_steps": self.global_steps,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

        # pad to be divisible by dp_size
        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
        
        if not self.async_rollout_mode:
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
        else:
            # Use async version to avoid blocking the event loop
            test_output_gen_batch_padded = await self.async_rollout_manager.generate_sequences_async(test_gen_batch_padded)

        # unpad
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

        print("validation generation end")

        # Prepare outputs (no lock needed for local processing)
        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        test_batch = test_batch.union(test_output_gen_batch)
        test_batch.meta_info["validate"] = True

        # evaluate using reward_function
        if self.val_reward_fn is None:
            raise ValueError("val_reward_fn must be provided for validation.")
        result = self.val_reward_fn(test_batch, return_dict=True)
        reward_tensor = result["reward_tensor"]
        scores = reward_tensor.sum(-1).cpu().tolist()
        
        # Collect data source and num_turns
        data_source = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
        num_turns = test_batch.non_tensor_batch.get("__num_turns__", None)

        # Update all shared data structures with lock (atomic operation)
        async with update_lock:
            sample_outputs.extend(output_texts)
            sample_scores.extend(scores)
            
            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
            
            # collect num_turns of each prompt
            if num_turns is not None:
                sample_turns.append(num_turns)
            
            data_source_lst.append(data_source)
            
            # Update progress bar by original batch size (before repeat)
            progress_bar.update(original_batch_size)


    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = (
                len(self.active_tasks)
                + self.result_queue.qsize()
                + self.cancel_queue.qsize()
                + await self.message_queue_client.get_queue_size()
            )
            timing_raw = {}
            idle_ratio = None
            if self.idle_start_time is not None and self.version_start_time is not None:
                rollout_active_time = self.idle_start_time - self.version_start_time
                rollout_version_time = time.time() - self.version_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
                timing_raw["rollouter/active_time"] = rollout_active_time
                timing_raw["rollouter/version_time"] = rollout_version_time
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.idle_start_time = None
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            val_metrics = None
            if (
                self.val_reward_fn is not None
                and self.config.rollout.test_freq > 0
                and self.current_param_version % self.config.rollout.test_freq == 0
                and self.current_param_version > 0  # don't test here in the initial parameter sync
            ) or (validate and self.val_reward_fn is not None):
                with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                    try:
                        print(f"[FullyAsyncRollouter] Start validation for version {self.current_param_version}...", flush=True)
                        val_metrics: dict = await self._validate()
                        print(f"[FullyAsyncRollouter] Validation completed for version {self.current_param_version}.", flush=True)
                    except Exception as e:
                        print(f"[FullyAsyncRollouter] Validation FAILED for version {self.current_param_version}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        # Return empty metrics to avoid crashing the sync process
                        val_metrics = {}
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version
            )
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

            self.version_start_time = time.time()

    async def _feed_samples(self):
        print(f"[WebAgentFullyAsyncRollouter] _feed_samples start")
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:

            full_batch = prepare_webagent_generation_data(batch_dict, self.config, epoch, self.global_steps, batch_dict['split'] == 'test')
            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * self.config.actor_rollout_ref.rollout.n,
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                tool_calls=[],
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[WebAgentFullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples"
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1


        # End signal
        await self.pending_queue.put("DONE")
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    # Override _consumer_worker to use merge_rollout_sample_flattened_steps
    async def _consumer_worker(self):
        """
        The consumer coroutine is responsible for obtaining the processing results
        from the result queue and putting them into the message queue
        """
        while True:
            rollout_sample = await self.result_queue.get()

            # Use the FLATTENED steps version of merge function
            rollout_sample = merge_rollout_sample_flattened_steps(self.config, self.tokenizer, rollout_sample, self.processor)
            
            if rollout_sample is None:
                logger.warning(f"[WebAgentFullyAsyncRollouter] Dropping sample due to negative answer_score.")
                self.dropped_stale_samples += 1
                self.staleness_samples -= 1
                self.result_queue.task_done()
                continue 

            # Put RolloutSample into the messagequeue
            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_stale_samples += 1
                #self.staleness_samples -= 1

            self.result_queue.task_done()

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        # Calling asynchronous generation methods
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        
        try:
            rollout_sample.agent_loop_output_list = await self.async_rollout_manager.generate_single_sample_async(
                rollout_sample.full_batch, rollout_sample.agent_loop_output_list
            )

            is_cancel = any(
                agent_loop.extra_fields.get("is_cancel", False) for agent_loop in rollout_sample.agent_loop_output_list
            )
        except (Exception, asyncio.CancelledError) as e:
            print(f"[FullyAsyncRollouter] Sample {rollout_sample.sample_id} processing failed/cancelled: {repr(e)}")
            # Treat exception as cancellation to ensure sample is requeued and worker doesn't crash.
            # Even if partial_rollout=False, we use cancel_queue to retry the failed sample from scratch.
            is_cancel = True

        if is_cancel:
            # Put in the cancel queue and wait for the generation to resume
            await self.cancel_queue.put(rollout_sample)
        else:
            # put into the result_queue
            rollout_sample.param_version = self.current_param_version
            rollout_sample.rollout_status = await self.get_statistics()
            await self.result_queue.put(rollout_sample)

        self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[WebAgentFullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine and consumer coroutine
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.processor_task = asyncio.create_task(self._processor_worker())
        self.consumer_task = asyncio.create_task(self._consumer_worker())


        try:
            # Wait for sample feed to complete
            # Use asyncio.wait to monitor all tasks. If processor/consumer exits early,
            # detect it instead of blocking on feed_task (it might be stuck on a full queue).
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task, self.consumer_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor or consumer task exited prematurely")

            print("[WebAgentFullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[WebAgentFullyAsyncRollouter] Streaming process completed")

            # Waiting for the result queue to clear
            await self.result_queue.join()
            print("[WebAgentFullyAsyncRollouter] Result queue cleared")

        except Exception as e:
            print(f"[WebAgentFullyAsyncRollouter] Streaming process exception:{e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.processor_task:
                self.processor_task.cancel()
            if self.consumer_task:
                self.consumer_task.cancel()

            await asyncio.gather(self.processor_task, self.consumer_task, return_exceptions=True)

        # Send a finish signal
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

        async with self.lock:
            self.running = False
