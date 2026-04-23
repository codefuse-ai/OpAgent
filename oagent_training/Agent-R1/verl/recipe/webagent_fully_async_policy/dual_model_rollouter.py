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

"""
DualModelWebAgentFullyAsyncRollouter: A @ray.remote actor that extends FullyAsyncRollouterBase
(NOT the @ray.remote WebAgentFullyAsyncRollouter, since Ray forbids actor inheritance)
to support two separate vLLM engine groups (planner + grounder) for dual-model inference.

Architecture:
- self.rollout_wg: Planner vLLM rollout worker group (inherited, uses rollout_pool GPUs)
- self.grounder_rollout_wg: Grounder vLLM rollout worker group (NEW, uses grounder_rollout_pool GPUs)
- Both groups are used by DualModelFullyAsyncWebAgentLoopManager for routing

GPU Layout (example with 16 GPUs per node):
  Training (shared, FSDP):  GPUs 0-7   (8 GPUs)
  Planner rollout (vLLM):   GPUs 8-11  (4 GPUs, TP=4)
  Grounder rollout (vLLM):  GPUs 12-15 (4 GPUs, TP=4)

NOTE: Ray does not allow inheriting from @ray.remote actor classes.
      WebAgentFullyAsyncRollouter is @ray.remote, so we inherit from the
      non-decorated base class FullyAsyncRollouterBase and replicate
      WebAgent-specific __init__ + override methods.
"""

import asyncio
import copy
import json
import logging
import os
from collections import Counter
from typing import List, Dict, Any

import ray
import numpy as np

from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouterBase
from recipe.webagent_fully_async_policy.fully_async_rollouter import (
    WebAgentRewardManager,
)
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.tracking import ValidationGenerationsLogger

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10, max_concurrency=100)
class DualModelWebAgentFullyAsyncRollouter(FullyAsyncRollouterBase):
    """
    Dual-model rollouter that creates separate planner and grounder vLLM engine groups.

    This class replicates WebAgentFullyAsyncRollouter's __init__ (since we cannot inherit
    from a @ray.remote actor class) and adds dual-model support.

    Requires config.grounder_rollout to be set with:
      grounder_rollout.n_gpus_per_node: Number of GPUs for grounder vLLM per node

    Optionally:
      dual_model.grounder_model_path: Path to grounder model (if different from planner)
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
        grounder_resource_pool_manager: ResourcePoolManager = None,
    ):
        # =============================================
        # Replicate WebAgentFullyAsyncRollouter.__init__
        # (Cannot call super().__init__ of the actor class)
        # =============================================
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = WebAgentRewardManager(config, tokenizer, num_examine=0)
        self.val_reward_fn = WebAgentRewardManager(config, tokenizer, num_examine=3)

        self.config.actor_rollout_ref.rollout.agent.default_agent_loop = "dual_model_async_web_agent"
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
        print("[DualModelRollouter] Creating datasets using WebAgentRLDataset...")

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
            captioning_fn=self.config.tool.webbrowser.caption_image_fn,
        )

        train_files_input = self.config.data.train_files
        if isinstance(train_files_input, list) and len(train_files_input) > 0:
            train_files_input = train_files_input[0]

        val_files_input = self.config.data.val_files
        if isinstance(val_files_input, list) and len(val_files_input) > 0:
            val_files_input = val_files_input[0]

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
            config=self.config,
        )

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
            config=self.config,
        )

        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        self._validate_config()
        print(f"[DualModelRollouter] _create_dataloader...\n{self.train_dataset}\n{val_dataset}")
        self._create_dataloader(self.train_dataset, val_dataset, collate_fn, train_sampler)
        print("[DualModelRollouter] Datasets created successfully.")

        # Initialize assignments
        self.task_assignments = {}

        # ==================== fully async config ====================
        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[DualModelRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        self.message_queue_client = None
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        self.max_queue_size = None

        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        self.dataloader_lock = asyncio.Lock()
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()
        self.result_queue = asyncio.Queue()
        self.cancel_queue = asyncio.Queue()

        # =============================================
        # Dual-model specific attributes
        # =============================================
        self.grounder_resource_pool_manager = grounder_resource_pool_manager
        self.grounder_rollout_wg = None
        self.grounder_actor_rollout_wg = None
        self._grounder_role_worker_mapping = role_worker_mapping

    def get_grounder_rollout_wg(self):
        """Get grounder rollout worker group (used by DualModelParameterSynchronizer)."""
        return self.grounder_rollout_wg

    async def init_workers(self):
        """Override to also initialize grounder rollout workers.

        Flow:
        1. Initialize planner rollout workers (base class logic)
        2. Create grounder rollout worker group
        3. Initialize DualModelFullyAsyncWebAgentLoopManager with both groups
        """
        import time as _time
        total_start = _time.time()

        t0 = _time.time()
        self._init_async_objects()
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        print(f"[DualModelRollouter] Planner rollout init: {_time.time() - t0:.2f}s")

        # Create grounder rollout worker group
        t1 = _time.time()
        await self._init_grounder_rollout_wg()
        print(f"[DualModelRollouter] Grounder rollout init: {_time.time() - t1:.2f}s")

        # Initialize agent loop manager with both worker groups
        t2 = _time.time()
        await self._init_async_rollout_manager()
        print(f"[DualModelRollouter] Async rollout manager init: {_time.time() - t2:.2f}s")

        print(f"[DualModelRollouter] Total init_workers: {_time.time() - total_start:.2f}s")

    def _create_actor_rollout_classes(self):
        """Override: only create planner rollout (same as WebAgentFullyAsyncRollouter)."""
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        """Override: same as WebAgentFullyAsyncRollouter."""
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    async def _init_grounder_rollout_wg(self):
        """Create the grounder rollout worker group on a separate GPU pool."""
        if self.grounder_resource_pool_manager is None:
            logger.warning(
                "[DualModelRollouter] No grounder resource pool manager provided. "
                "Grounder will share planner vLLM engine."
            )
            self.grounder_rollout_wg = self.rollout_wg
            self.grounder_actor_rollout_wg = self.grounder_rollout_wg
            return

        # Create the grounder rollout resource pool
        self.grounder_resource_pool_manager.create_resource_pool()

        # Prepare grounder rollout config
        dual_model_config = self.config.get("dual_model", {})
        grounder_model_path = dual_model_config.get("grounder_model_path", None)

        if grounder_model_path:
            grounder_rollout_config = copy.deepcopy(self.config.actor_rollout_ref)
            grounder_rollout_config.model.path = grounder_model_path
        else:
            grounder_rollout_config = self.config.actor_rollout_ref

        # Register grounder rollout class on grounder resource pool
        grounder_rollout_pool = list(self.grounder_resource_pool_manager.resource_pool_dict.values())[0]
        grounder_cls = RayClassWithInitArgs(
            cls=self._grounder_role_worker_mapping[Role.Rollout],
            config=grounder_rollout_config,
            role="rollout",  # Must be a valid role string
        )

        # Key MUST contain "rollout" (lowercase) so that DIRECT_ROLLOUT_METHOD
        # methods (e.g. get_zeromq_address) are bound without prefix in WorkerDict.
        # See _bind_workers_method_to_parent: `"rollout" in key` is case-sensitive.
        class_dict = {"grounder_rollout": grounder_cls}
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

        from omegaconf import OmegaConf
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        wg_dict = self.ray_worker_group_cls(
            resource_pool=grounder_rollout_pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        self.grounder_rollout_wg = spawn_wg["grounder_rollout"]
        self.grounder_rollout_wg.init_model()
        self.grounder_actor_rollout_wg = self.grounder_rollout_wg

        print(
            f"[DualModelRollouter] Grounder rollout worker group created with "
            f"{self.grounder_rollout_wg.world_size} workers"
        )

    async def _init_async_rollout_manager(self):
        """Override to create DualModelFullyAsyncWebAgentLoopManager."""
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.webagent_fully_async_policy.agent_loop.agent_loop import DualModelFullyAsyncWebAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await DualModelFullyAsyncWebAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
            grounder_worker_group=self.grounder_rollout_wg,
        )


    # =============================================
    # WebAgent-specific method overrides
    # (Replicated from WebAgentFullyAsyncRollouter since we inherit from
    #  FullyAsyncRollouterBase which has different base implementations)
    # =============================================

    def _get_gen_batch(self, batch, epoch, global_steps, validate):
        """Override: same as WebAgentFullyAsyncRollouter."""
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        gen_batch.non_tensor_batch["epoch"] = np.array([epoch] * len(gen_batch), dtype=object)
        gen_batch.non_tensor_batch["global_steps"] = np.array([global_steps] * len(gen_batch), dtype=object)
        gen_batch.non_tensor_batch["validate"] = np.array([validate] * len(gen_batch), dtype=object)
        return gen_batch

    async def _feed_samples(self):
        """Override: Use prepare_webagent_generation_data."""
        from recipe.webagent_fully_async_policy.detach_utils import prepare_webagent_generation_data
        from recipe.fully_async_policy.fully_async_rollouter import RolloutSample

        print(f"[DualModelRollouter] _feed_samples start")
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            full_batch = prepare_webagent_generation_data(
                batch_dict, self.config, epoch, self.global_steps, batch_dict['split'] == 'test'
            )

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

            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[DualModelRollouter][Feed] "
                    f"Maximum count reached, stop adding new samples "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        await self.pending_queue.put("DONE")
        print(f"[DualModelRollouter][Feed] Sample addition complete, {self.global_steps} samples added")

    async def _consumer_worker(self):
        """Override: Use merge_rollout_sample_flattened_steps."""
        from recipe.webagent_fully_async_policy.detach_utils import merge_rollout_sample_flattened_steps
        import time as _time

        while True:
            rollout_sample = await self.result_queue.get()

            consumer_start = _time.time()

            # Use the FLATTENED steps version of merge function (critical for dual-model role tags)
            merge_start = _time.time()
            rollout_sample = merge_rollout_sample_flattened_steps(
                self.config, self.tokenizer, rollout_sample, self.processor
            )
            merge_time = _time.time() - merge_start

            if rollout_sample is None:
                logger.warning(f"[DualModelRollouter] Dropping sample due to negative answer_score.")
                self.dropped_stale_samples += 1
                self.staleness_samples -= 1
                self.result_queue.task_done()
                continue

            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_stale_samples += 1

            consumer_total = _time.time() - consumer_start
            print(
                f"[DualModelRollouter] Consumer processed sample: "
                f"merge={merge_time:.2f}s, total={consumer_total:.2f}s"
            )

            self.result_queue.task_done()

    async def _process_single_sample_streaming(self, rollout_sample):
        """Override: Same as WebAgentFullyAsyncRollouter (with exception handling)."""
        import time as _time
        sample_start = _time.time()

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
            print(f"[DualModelRollouter] Sample {rollout_sample.sample_id} processing failed/cancelled: {repr(e)}")
            is_cancel = True

        sample_time = _time.time() - sample_start

        if is_cancel:
            await self.cancel_queue.put(rollout_sample)
            print(f"[DualModelRollouter] Sample {rollout_sample.sample_id} cancelled, time={sample_time:.2f}s")
        else:
            rollout_sample.param_version = self.current_param_version
            rollout_sample.rollout_status = await self.get_statistics()
            await self.result_queue.put(rollout_sample)
            print(f"[DualModelRollouter] Sample {rollout_sample.sample_id} completed, time={sample_time:.2f}s")

        self.processed_sample_count += 1

    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0):
        """Override: Same as WebAgentFullyAsyncRollouter (with validation)."""
        import time as _time
        from recipe.fully_async_policy.fully_async_rollouter import ValidateMetrics
        from verl.utils.debug import marked_timer

        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
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
                rollout_version_time = _time.time() - self.version_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
                timing_raw["rollouter/active_time"] = rollout_active_time
                timing_raw["rollouter/version_time"] = rollout_version_time
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.idle_start_time = None
            print(
                f"[DualModelRollouter][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            val_metrics = None
            if (
                self.val_reward_fn is not None
                and self.config.rollout.test_freq > 0
                and self.current_param_version % self.config.rollout.test_freq == 0
                and self.current_param_version > 0
            ) or (validate and self.val_reward_fn is not None):
                with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                    try:
                        print(f"[DualModelRollouter] Start validation for version {self.current_param_version}...", flush=True)
                        val_metrics: dict = await self._validate()
                        print(f"[DualModelRollouter] Validation completed for version {self.current_param_version}.", flush=True)
                    except Exception as e:
                        print(f"[DualModelRollouter] Validation FAILED for version {self.current_param_version}: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        val_metrics = {}
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version
            )
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))
            self.version_start_time = _time.time()

    async def _validate(self):
        """
        Validation for dual-model rollouter.
        
        Delegates to the WebAgent validation logic. We import the method bodies from
        the WebAgent rollouter module to reuse the same validation pipeline (which uses
        self.async_rollout_manager.generate_sequences_async internally).
        
        Note: This requires all the same self.* attributes as WebAgentFullyAsyncRollouter._validate,
        which are set up in our __init__ (val_dataloader, val_reward_fn, config, tokenizer, etc.)
        """
        # Import the standalone validation implementation
        from recipe.webagent_fully_async_policy._validation_impl import webagent_validate
        return await webagent_validate(self)