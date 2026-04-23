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
import logging
import os
from typing import Any, Optional, Sequence, List

import numpy as np
import ray
from omegaconf import DictConfig
import hydra

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    get_trajectory_info,
)
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
# Import base classes from fully_async_policy
from recipe.fully_async_policy.agent_loop.agent_loop import (
    FullyAsyncAgentLoopManager,
    FullyAsyncAgentLoopWorker,
    _agent_loop_registry,
    _DummyConfig,
    FullyAsyncAgentLoopWorkerBase,
)
from tensordict import TensorDict
from recipe.webagent_fully_async_policy.agent_loop.web_agent_loop import AsyncWebAgentLoop
from recipe.webagent_fully_async_policy.browser_env.utils import get_ws_endpoint_list
from recipe.webagent_fully_async_policy.browser_env.async_web_browser_envs import BrowserActor
import concurrent.futures
import random
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.experimental.agent_loop.agent_loop import _InternalAgentLoopOutput
import torch
from verl.utils.transferqueue_utils import tqbridge

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
MAX_CONCURRENT_WORKERS = int(os.environ.get('MAX_CONCURRENT_WORKERS', 32))
WEBARENA_AUTH_PATH = os.environ.get("WEBARENA_AUTH_PATH", "")

@ray.remote
class FullyAsyncWebAgentLoopWorker(FullyAsyncAgentLoopWorkerBase):
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        browser_endpoints: list[str] = None,
    ):        
        print(f"[FullyAsyncWebAgentLoopWorker] start")
        super().__init__(config, server_handles, reward_router_address)

        self.browser_actor_pool = []
        if browser_endpoints:

            
            def init_actor(ep):
                actor = BrowserActor(ep)
                actor.start()
                return actor

            self.webbrowser_thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS, thread_name_prefix="BrowserWorker")
            
            futures = {self.webbrowser_thread_executor.submit(init_actor, ep): ep for ep in browser_endpoints}
            for future in concurrent.futures.as_completed(futures):
                try:
                    actor = future.result()
                    if actor.browser_unit:
                        self.browser_actor_pool.append(actor)
                    else:
                        print(f"Failed to initialize actor for {futures[future]}")
                except Exception as e:
                    print(f"Exception initializing actor for {futures[future]}: {e}")
            
            print(f"Initialized {len(self.browser_actor_pool)}/{len(browser_endpoints)} browser actors.")


    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            
            # Filter out 'config' from kwargs to prevent OmegaConf GrammarParseError
            # The 'config' field in kwargs usually contains a JSON string which confuses OmegaConf
            hydra_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}
            
            # Also filter out 'config' inside 'extra_info' if present to avoid OmegaConf GrammarParseError
            if 'extra_info' in hydra_kwargs and isinstance(hydra_kwargs['extra_info'], dict):
                if 'config' in hydra_kwargs['extra_info']:
                    # Make a shallow copy to avoid modifying the original kwargs in place
                    extra_info = hydra_kwargs['extra_info'].copy()
                    # Removing the config key from extra_info to avoid OmegaConf parsing errors
                    extra_info.pop('config', None)
                    hydra_kwargs['extra_info'] = extra_info

            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                **hydra_kwargs
            )
            # We pass the original kwargs to run if needed, or rely on what was passed to __init__
            # But usually run() takes specific args.
            # The original code passed **kwargs to run(), let's keep that but be careful about instantiation.
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

            # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            # Handle multi-modal inputs and position_ids calculation
            # Only support Qwen2VLImageProcessor for multi-modal processing currently
            # TODO: support other multi-modal inputs
            multi_modal_inputs = None
            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = getattr(output, "multi_modal_data", {}).get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
                # because np.array() only keeps the keys for BatchFeature.
                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                position_ids = vision_position_ids # (1, 3, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            enable_async_reward = (
                self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                        "responses": response_output["input_ids"],  # [1, response_length]
                        "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                        "input_ids": input_ids,  # [1, prompt_length + response_length]
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                result = await self.reward_manager_worker.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
            )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        print(f"[WebAgentFullyAsyncWebAgentLoopWorker] generate_sequences start")
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
            # Ensure logprobs is enabled for validation to avoid None logprobs
            sampling_params["logprobs"] = config.calculate_log_probs

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["async_web_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.non_tensor_batch.get("global_steps", -1), index, batch.non_tensor_batch.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            # Pass the pool list directly
            if self.browser_actor_pool:
                kwargs["browser_actor_pool"] = self.browser_actor_pool
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)

        return output

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch.
        Override parent method to safely handle None logprobs."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        # Check if ALL inputs have response_logprobs before concatenating
        if all(input.response_logprobs is not None for input in inputs):
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys()) if reward_extra_infos else []
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> list[AgentLoopOutput]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        
        # No queue initialization here. We use the list directly.
        print(f"[WebAgentFullyAsyncWebAgentLoopWorker] generate_sequences_no_post start")
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
            # Ensure logprobs is enabled for validation to avoid None logprobs
            sampling_params["logprobs"] = config.calculate_log_probs

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.non_tensor_batch.get("global_steps", -1), index, batch.non_tensor_batch.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["output"] = partial_output_list[i]
            # Pass the pool list directly
            if self.browser_actor_pool:
                kwargs["browser_actor_pool"] = self.browser_actor_pool
            tasks.append(
                asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
            )
        return await asyncio.gather(*tasks)



    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
        try:
            with rollout_trace_attr(
                step=trajectory["step"],
                sample_index=trajectory["sample_index"],
                rollout_n=trajectory["rollout_n"],
                validate=trajectory["validate"],
                name="agent_loop",
            ):
                assert agent_name in _agent_loop_registry, (
                    f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
                )

                agent_loop_config = _agent_loop_registry[agent_name]
                
                # Filter out 'config' from kwargs to prevent OmegaConf GrammarParseError
                hydra_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}
                
                # Also filter out 'config' inside 'extra_info' if present to avoid OmegaConf GrammarParseError
                if 'extra_info' in hydra_kwargs and isinstance(hydra_kwargs['extra_info'], dict):
                    if 'config' in hydra_kwargs['extra_info']:
                        # Make a shallow copy to avoid modifying the original kwargs in place
                        extra_info = hydra_kwargs['extra_info'].copy()
                        # Removing the config key from extra_info to avoid OmegaConf parsing errors
                        extra_info.pop('config', None)
                        hydra_kwargs['extra_info'] = extra_info

                agent_loop = hydra.utils.instantiate(
                    config=agent_loop_config,
                    trainer_config=_DummyConfig(config=self.config),
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    **hydra_kwargs
                )
                return await agent_loop.run(sampling_params, cancellation_event=self.cancellation_event, **kwargs)
        except Exception as e:
            logger.exception(f"Agent_loop run failed: {e}")
            raise e

class FullyAsyncWebAgentLoopManager(FullyAsyncAgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        super().__init__(config, worker_group, rm_wg)
        self.browser_endpoints = []
        self.agent_loop_workers_class = FullyAsyncWebAgentLoopWorker

    async def _async_init(self):
        # Fetch browser endpoints here in the rollouter process
        self.browser_endpoints = get_ws_endpoint_list()
        print(f"[FullyAsyncWebAgentLoopManager] Found {len(self.browser_endpoints)} browser endpoints")
        
        await super()._async_init()

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = len(self.server_handles)
        
        # Distribute browser endpoints among workers
        endpoints_per_worker = [[] for _ in range(num_workers)]
        if self.browser_endpoints:
            for i, ep in enumerate(self.browser_endpoints):
                endpoints_per_worker[i % num_workers].append(ep)

        for i in range(num_workers):
            worker = self.agent_loop_workers_class.remote(
                config=self.config,
                server_handles=[self.server_handles[i]],
                reward_router_address=self.reward_router_address,
                browser_endpoints=endpoints_per_worker[i]
            )
            self.agent_loop_workers.append(worker)

    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Asynchronously split input batch and dispatch to agent loop workers.

        This method overrides the base class to use async/await instead of blocking ray.get(),
        preventing event loop blocking during long-running validation tasks.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        # Fix for Issue #4147: Always call wake_up() to ensure weight sync
        # The wake_up()/sleep() methods internally check free_cache_engine
        await self._async_wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        # Use asyncio.gather with Ray's async API instead of blocking ray.get()
        output_futures = [
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
        ]
        # Convert Ray ObjectRefs to asyncio futures and await them
        outputs = await asyncio.gather(*[asyncio.wrap_future(fut.future()) for fut in output_futures])
        
        output = DataProto.concat(outputs)
        # Fix for Issue #4147: Always call sleep() to ensure proper cleanup
        await self._async_sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        """Calculate performance metrics from agent loop outputs.
        
        This method is copied from the parent class AgentLoopManager for use in generate_sequences_async.
        """
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

# ========================= Dual-Model Classes =========================

@ray.remote
class DualModelFullyAsyncWebAgentLoopWorker(FullyAsyncAgentLoopWorkerBase):
    """
    Dual-model variant of FullyAsyncWebAgentLoopWorker.
    
    Key difference: accepts two sets of server_handles (planner + grounder)
    and creates a DualModelServerManager instead of AsyncLLMServerManager.
    """

    def __init__(
        self,
        config: DictConfig,
        planner_server_handles: list[ray.actor.ActorHandle],
        grounder_server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
        browser_endpoints: list[str] = None,
    ):
        print(f"[DualModelFullyAsyncWebAgentLoopWorker] Initializing...")
        # We DON'T call super().__init__ because it would create a single-model ServerManager.
        # Instead, we manually replicate the initialization but with DualModelServerManager.
        from recipe.webagent_fully_async_policy.agent_loop.dual_model_server_manager import DualModelServerManager

        # Create dual model server manager (MUST be set before anything that checks self.server_manager)
        self.server_manager = DualModelServerManager(
            config=config,
            planner_server_handles=planner_server_handles,
            grounder_server_handles=grounder_server_handles,
        )

        # ---- Replicate AgentLoopWorkerBase.__init__ ----
        self.config = config
        all_server_handles = planner_server_handles + grounder_server_handles
        self.server_handles = all_server_handles  # for compatibility

        self.reward_router_address = reward_router_address

        # model_name (used by tracing / logging)
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Tokenizer & processor (use copy_to_local like parent)
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer, hf_processor
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        try:
            self.processor = hf_processor(local_path, trust_remote_code=True)
        except Exception:
            self.processor = None

        # Load agent loop YAML configs into the registry
        from verl.experimental.agent_loop.utils import resolve_config_path
        from omegaconf import OmegaConf as _OmegaConf
        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = _OmegaConf.load(resolved_path)
            for alc in agent_loop_configs:
                _agent_loop_registry[alc.name] = alc

        # Custom chat template
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        # Reward manager worker (same as parent: ray.remote with NodeAffinity)
        from verl.experimental.reward import RewardManagerWorker
        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, self.reward_router_address)

        # Rollout trace config
        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

        # ---- Replicate FullyAsyncAgentLoopWorkerBase additions ----
        # Cancellation event
        self.cancellation_event = asyncio.Event()

        # ---- Replicate FullyAsyncWebAgentLoopWorker additions ----
        # Browser actors
        self.browser_actor_pool = []
        if browser_endpoints:
            def init_actor(ep):
                actor = BrowserActor(ep)
                actor.start()
                return actor

            self.webbrowser_thread_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_CONCURRENT_WORKERS, thread_name_prefix="BrowserWorker"
            )
            futures = {self.webbrowser_thread_executor.submit(init_actor, ep): ep for ep in browser_endpoints}
            for future in concurrent.futures.as_completed(futures):
                try:
                    actor = future.result()
                    if actor.browser_unit:
                        self.browser_actor_pool.append(actor)
                    else:
                        print(f"Failed to initialize actor for {futures[future]}")
                except Exception as e:
                    print(f"Exception initializing actor for {futures[future]}: {e}")

            print(f"[DualModel] Initialized {len(self.browser_actor_pool)}/{len(browser_endpoints)} browser actors.")

        # Event loop reference
        self.loop = asyncio.get_event_loop()

        print(f"[DualModelFullyAsyncWebAgentLoopWorker] Ready with dual model servers")

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        """Override to use dual_model_async_web_agent as default agent name."""
        # Override agent_name to use dual-model version
        if self.config.get("dual_model", {}).get("enable", False):
            agent_name = "dual_model_async_web_agent"
        
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]

            hydra_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}
            if 'extra_info' in hydra_kwargs and isinstance(hydra_kwargs['extra_info'], dict):
                if 'config' in hydra_kwargs['extra_info']:
                    extra_info = hydra_kwargs['extra_info'].copy()
                    extra_info.pop('config', None)
                    hydra_kwargs['extra_info'] = extra_info

            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                **hydra_kwargs
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

            # Reuse parent's padding/postprocess logic
            return await self._build_internal_output(output)

    async def _build_internal_output(self, output: AgentLoopOutput) -> _InternalAgentLoopOutput:
        """Build _InternalAgentLoopOutput from AgentLoopOutput (same as parent's logic)."""
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        # Handle multi-modal inputs and position_ids calculation
        multi_modal_inputs = None
        if (
            self.processor is not None
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            images = getattr(output, "multi_modal_data", {}).get("image", None)
            current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)
            multi_modal_inputs = dict(multi_modal_inputs)

            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            video_grid_thw = multi_modal_inputs.get("video_grid_thw")
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask.squeeze(0),
            ).unsqueeze(0)
            position_ids = vision_position_ids
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        enable_async_reward = (
            self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
        ) or not self.config.reward_model.enable
        if output.reward_score is None and enable_async_reward and self.reward_manager_worker:
            batch = TensorDict(
                {
                    "prompts": prompt_output["input_ids"],
                    "responses": response_output["input_ids"],
                    "attention_mask": attention_mask,
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in {}},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }
            data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            result = await self.reward_manager_worker.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    # Reuse parent's generate_sequences, _postprocess, generate_sequences_no_post
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Reuse parent's generate_sequences logic."""
        # Import to ensure dual_model_async_web_agent is registered
        import recipe.webagent_fully_async_policy.agent_loop.dual_model_web_agent_loop  # noqa: F401

        print(f"[DualModel] generate_sequences start")
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
            sampling_params["logprobs"] = config.calculate_log_probs

        agent_name = "dual_model_async_web_agent"
        # Always force dual-model agent name (override whatever the batch provides)
        batch.non_tensor_batch["agent_name"] = np.array([agent_name] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.non_tensor_batch.get("global_steps", -1), index, batch.non_tensor_batch.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k not in ("agent_name", "trace")}
            if self.browser_actor_pool:
                kwargs["browser_actor_pool"] = self.browser_actor_pool
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(
                        sampling_params, trajectory_info[i],
                        trace=trace_this_sample, agent_name=agent_name, **kwargs
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)
        return self._postprocess(outputs)

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch.

        This is a copy of FullyAsyncWebAgentLoopWorker._postprocess since DualModelFullyAsyncWebAgentLoopWorker
        inherits from FullyAsyncAgentLoopWorkerBase, not FullyAsyncWebAgentLoopWorker.
        """
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        # Check if ALL inputs have response_logprobs before concatenating
        if all(input.response_logprobs is not None for input in inputs):
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys()) if reward_extra_infos else []
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> list[AgentLoopOutput]:
        """Override to use dual-model agent loop for partial rollout / validation."""
        # Ensure dual model agent loop is registered
        import recipe.webagent_fully_async_policy.agent_loop.dual_model_web_agent_loop  # noqa: F401

        print(f"[DualModel] generate_sequences_no_post start")
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
            sampling_params["logprobs"] = config.calculate_log_probs

        # Force dual-model agent name
        agent_name = "dual_model_async_web_agent"
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array([agent_name] * len(batch), dtype=object)
        else:
            batch.non_tensor_batch["agent_name"] = np.array([agent_name] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.non_tensor_batch.get("global_steps", -1), index, batch.non_tensor_batch.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["output"] = partial_output_list[i]
            if self.browser_actor_pool:
                kwargs["browser_actor_pool"] = self.browser_actor_pool
            tasks.append(
                asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
            )
        return await asyncio.gather(*tasks)

    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
        """Override to use dual-model agent loop and filter config from kwargs."""
        # Ensure dual model agent loop is registered
        import recipe.webagent_fully_async_policy.agent_loop.dual_model_web_agent_loop  # noqa: F401

        try:
            with rollout_trace_attr(
                step=trajectory["step"],
                sample_index=trajectory["sample_index"],
                rollout_n=trajectory["rollout_n"],
                validate=trajectory["validate"],
                name="agent_loop",
            ):
                # Force dual-model agent name
                agent_name = "dual_model_async_web_agent"

                assert agent_name in _agent_loop_registry, (
                    f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
                )

                agent_loop_config = _agent_loop_registry[agent_name]

                # Filter out 'config' from kwargs to prevent OmegaConf GrammarParseError
                hydra_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}

                # Also filter out 'config' inside 'extra_info' if present
                if 'extra_info' in hydra_kwargs and isinstance(hydra_kwargs['extra_info'], dict):
                    if 'config' in hydra_kwargs['extra_info']:
                        extra_info = hydra_kwargs['extra_info'].copy()
                        extra_info.pop('config', None)
                        hydra_kwargs['extra_info'] = extra_info

                agent_loop = hydra.utils.instantiate(
                    config=agent_loop_config,
                    trainer_config=_DummyConfig(config=self.config),
                    server_manager=self.server_manager,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    **hydra_kwargs
                )
                return await agent_loop.run(sampling_params, cancellation_event=self.cancellation_event, **kwargs)
        except Exception as e:
            logger.exception(f"Agent_loop run failed: {e}")
            raise e


class DualModelFullyAsyncWebAgentLoopManager(FullyAsyncWebAgentLoopManager):
    """
    Dual-model variant of FullyAsyncWebAgentLoopManager.
    
    Key difference: initializes DualModelFullyAsyncWebAgentLoopWorker with separate
    planner and grounder server handles.
    """

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None,
                 grounder_worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        super().__init__(config, worker_group, rm_wg)
        self.grounder_worker_group = grounder_worker_group
        self.grounder_server_handles = []

    @classmethod
    async def create(cls, config: DictConfig, worker_group: RayWorkerGroup = None,
                     grounder_worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        """Override parent create() to accept grounder_worker_group."""
        instance = cls(config, worker_group, grounder_worker_group, rm_wg)
        await instance._async_init()
        return instance

    async def _async_init(self):
        """Override to also get grounder server handles and properly init reward model."""
        self.browser_endpoints = get_ws_endpoint_list()
        print(f"[DualModelManager] Found {len(self.browser_endpoints)} browser endpoints")

        # Initialize reward model manager (same as grandparent FullyAsyncAgentLoopManager)
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager
            self.reward_model_manager = RewardModelManager(self.config.reward_model, self.rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        elif self.rm_wg is not None:
            self.reward_router_address = self.rm_wg.get_router_address()
        else:
            self.reward_router_address = None

        # Initialize planner vLLM servers from the planner worker group
        await self._initialize_llm_servers_async()
        # self.server_handles is now set by _initialize_llm_servers_async()
        print(f"[DualModelManager] Got {len(self.server_handles)} planner server handles")

        # Get grounder server handles
        if self.grounder_worker_group is not None:
            # Initialize grounder vLLM servers
            await self._initialize_grounder_llm_servers_async()
            print(f"[DualModelManager] Got {len(self.grounder_server_handles)} grounder server handles")
        else:
            logger.warning("[DualModelManager] No grounder worker group, falling back to single-model")
            self.grounder_server_handles = self.server_handles

        self._init_agent_loop_workers()

    async def _initialize_grounder_llm_servers_async(self):
        """Initialize grounder vLLM servers from the grounder worker group."""
        import copy as _copy

        grounder_rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        # Check for grounder-specific TP override
        grounder_rollout_config_override = self.config.get("grounder_rollout", {})
        if grounder_rollout_config_override and grounder_rollout_config_override.get("tensor_model_parallel_size", None):
            grounder_rollout_world_size = grounder_rollout_config_override.tensor_model_parallel_size

        world_size = self.grounder_worker_group.world_size
        num_replicas = world_size // grounder_rollout_world_size

        # Construct a separate rollout config for grounder (deepcopy to avoid mutating planner's)
        rollout_config = _copy.deepcopy(self.config.actor_rollout_ref.rollout)
        rollout_config.tensor_model_parallel_size = grounder_rollout_world_size
        model_config = self.config.actor_rollout_ref.model

        # Check for grounder-specific model path
        dual_model_config = self.config.get("dual_model", {})
        grounder_model_path = dual_model_config.get("grounder_model_path", None)
        if grounder_model_path:
            import copy
            from omegaconf import OmegaConf
            model_config = copy.deepcopy(model_config)
            model_config.path = grounder_model_path

        self.grounder_rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        # Use a different server name prefix so grounder actors don't collide with planner actors
        for replica in self.grounder_rollout_replicas:
            replica.server_name_prefix = "vllm_grounder_server"

        await asyncio.gather(*[
            server.init_hybrid(self.grounder_worker_group) for server in self.grounder_rollout_replicas
        ])

        self.grounder_server_handles = [server._server_handle for server in self.grounder_rollout_replicas]

    def _init_agent_loop_workers(self):
        """Override to create DualModelFullyAsyncWebAgentLoopWorker instances."""
        self.agent_loop_workers = []
        # Use the minimum of planner/grounder server counts as number of workers
        num_planner = len(self.server_handles)
        num_grounder = len(self.grounder_server_handles)
        num_workers = min(num_planner, num_grounder) if num_grounder > 0 else num_planner

        # Distribute browser endpoints among workers
        endpoints_per_worker = [[] for _ in range(num_workers)]
        if self.browser_endpoints:
            for i, ep in enumerate(self.browser_endpoints):
                endpoints_per_worker[i % num_workers].append(ep)

        for i in range(num_workers):
            # Each worker gets one planner server and one grounder server
            planner_handles = [self.server_handles[i % num_planner]]
            grounder_handles = [self.grounder_server_handles[i % num_grounder]]

            worker = DualModelFullyAsyncWebAgentLoopWorker.remote(
                config=self.config,
                planner_server_handles=planner_handles,
                grounder_server_handles=grounder_handles,
                reward_router_address=self.reward_router_address,
                browser_endpoints=endpoints_per_worker[i],
            )
            self.agent_loop_workers.append(worker)

        print(f"[DualModelManager] Created {num_workers} dual-model agent loop workers")