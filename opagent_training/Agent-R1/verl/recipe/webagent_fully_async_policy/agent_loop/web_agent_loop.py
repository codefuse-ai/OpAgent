import asyncio
import logging
import os
import re
import copy
import random
from unittest import enterModuleContext
import torch
from typing import Any, Optional
from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, AgentData
from verl.utils.profiler import simple_timer
from recipe.webagent_fully_async_policy.browser_env.async_web_browser_envs import BrowserActor
from recipe.webagent_fully_async_policy.browser_env.tool_env import WebBrowserEnv
from recipe.webagent_fully_async_policy.browser_env.web_browser_tool import WebBrowserTool
from recipe.webagent_fully_async_policy.evaluation_harness.utils import compute_score_format
import json_repair
from recipe.webagent_fully_async_policy.browser_env.auto_login import SITE_PORT_MAP
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
logger.setLevel("INFO")



@register("async_web_agent")
class AsyncWebAgentLoop(AgentLoopBase):
    def __init__(self, trainer_config, **kwargs):
        super().__init__(trainer_config, **kwargs)
        self.max_turns = trainer_config.config.tool.max_turns
        self.response_length = trainer_config.config.actor_rollout_ref.rollout.response_length
        # Assuming default prompt/response lengths if not set
        if not hasattr(self, 'response_length'):
             self.response_length = 1024
        self.apply_chat_template_kwargs = trainer_config.config.data.get("apply_chat_template_kwargs", {})
        self.enable_partial_rollout = trainer_config.config.async_training.get("partial_rollout", False)
        # Used for response truncation/normalization
        self.tool_call_end = trainer_config.config.get("tool_call_end", "</tool_call>")
        self.trajectory_save_freq = trainer_config.config.async_training.get("trajectory_save_freq", 20)

    async def run(self, sampling_params: dict[str, Any], cancellation_event: asyncio.Event = None, **kwargs) -> AgentLoopOutput:
        param_version = kwargs.get("param_version", 0)
        agent_data = None
        state = None
        env_options = {
            "config_file": kwargs.get("config_file", None),
        }
        # 1. check whether is the partial task
        output: Optional[AgentLoopOutput] = kwargs.get("output", None)
        if output and output.extra_fields.get("is_cancel", False):
            agent_data, state = self._restore_from_output(output)
            logger.info(f"[AsyncWebAgentLoop] Resuming from {state.value}")
        else:
            if output and not output.extra_fields.get("is_cancel", False):
                # Completed, return directly
                return output
            
            agent_data, state = await self._init_new_session(sampling_params, kwargs, param_version, kwargs.get("browser_actor_pool"), env_options)
            logger.info("[AsyncWebAgentLoop] Start from scratch")

        # 2. run state machine
        state = await self._run_state_machine(agent_data, state, sampling_params, cancellation_event, agent_data.request_id)

        # 3. bulid output
        if state == AgentState.TERMINATED:
            output = await self._build_completed_output(agent_data, param_version)

            # Cleanup
            async def cleanup_op():
                if hasattr(self.browser_actor.browser_unit, 'active_envs'):
                    env: WebBrowserEnv = self.browser_actor.browser_unit.active_envs.pop(agent_data.request_id, None)
                    if env:
                        try:
                            await env.aclose()
                        except Exception as e:
                            logger.error(f"Error closing env {agent_data.request_id}: {e}")

            try:
                await asyncio.wrap_future(self.browser_actor.submit(cleanup_op))
            except Exception as e:
                logger.error(f"Failed to cleanup env {agent_data.request_id}: {e}")

            return output
        else:
            # build cancelled output
            return self._build_cancelled_output(agent_data, state)

    def _init_system_prompt(self, messages: list, env_options: dict):
        url_info = f"There are some website ports: {SITE_PORT_MAP}. You can use http://<ip>:<port> to access the website if you need access another website."
        #从config_file中获取start_url
        config_file = env_options['config_file']
        with open(config_file, 'r', encoding='utf-8') as config_file_tmp:
            config_dict = json_repair.load(config_file_tmp)
        start_url = config_dict['start_url']
        url_info += f"The target url is {start_url}. The first url has already been loaded in the initial webpage. If |AND| is in the target url, you can navigate to the second url when you need."
        messages[0]['content'] = url_info + messages[0]['content']
        return messages

    async def _init_new_session(self, sampling_params, kwargs, param_version, browser_actor_pool: list, env_options: dict):
        messages = list(copy.deepcopy(kwargs["raw_prompt"]))
        messages = self._init_system_prompt(messages, env_options)
        kwargs["raw_prompt"] = copy.deepcopy(messages)
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", []))
        metrics = {}
        request_id = "web_agent_" + str(uuid4().hex)
        
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs={},
        )
        # additional param version record
        agent_data.extra_fields["param_version_start"] = param_version
        agent_data.extra_fields["param_version_end"] = param_version
        agent_data.extra_fields["steps"] = []
        agent_data.extra_fields["current_multi_modal_inputs"] = {}
        agent_data.extra_fields["data_source"] = kwargs.get("data_source", "unknown")
        
        if not browser_actor_pool:
            # Fallback or detailed error
            msg = f"browser_actor_pool is required. kwargs keys: {list(kwargs.keys())}"
            raise ValueError(msg)
        
        browser_actor = None
        last_actor_endpoint = agent_data.extra_fields.get("browser_actor_endpoint")
        
        if last_actor_endpoint:
            for actor in browser_actor_pool:
                if actor._endpoint == last_actor_endpoint:
                    browser_actor = actor
                    logger.info(f"Resumed with same actor: {last_actor_endpoint}")
                    break
            if not browser_actor:
                logger.warning(f"Could not find previous actor {last_actor_endpoint}, picking random new one. State loss possible!")
        
        if not browser_actor:
            browser_actor = random.choice(browser_actor_pool)
            
        # Save endpoint for future resumes
        self.browser_actor = browser_actor 
        agent_data.extra_fields["browser_actor_endpoint"] = browser_actor._endpoint

        request_id = agent_data.request_id

       # Restore or Init Env on Actor
        async def setup_env_op():
            if not hasattr(browser_actor.browser_unit, 'active_envs'):
                browser_actor.browser_unit.active_envs = {}
            # Init env instance
            if request_id not in browser_actor.browser_unit.active_envs:

                config_file = kwargs.get('config_file')
                with open(config_file, 'r', encoding='utf-8') as config_file_tmp:
                    config_dict = json_repair.load(config_file_tmp)

                new_env = WebBrowserEnv(
                    context_id = request_id,
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
                    user_query=config_dict['intent']
                )
                new_env.owner_actor = browser_actor
                # areset 现在可以直接 await
                browser_actor.browser_unit.active_envs[request_id] = new_env
                current_screenshot, current_info = await new_env.areset(browser_unit=browser_actor.browser_unit, options=env_options)
                #current_screenshot: numpy image

                if hasattr(new_env, 'evaluator') and new_env.evaluator:
                    if kwargs.get('validate'):
                        should_save = True
                    else:
                        should_save = kwargs.get('global_steps', 0) % self.trajectory_save_freq == 0
                    for evaluator_i in range(len(new_env.evaluator.evaluators)):
                        if hasattr(new_env.evaluator.evaluators[evaluator_i], 'set_training_info'):
                            new_env.evaluator.evaluators[evaluator_i].set_training_info(
                                training_step=kwargs.get('global_steps', 0),
                                should_save=should_save,
                                epoch=kwargs.get('epoch', 0),
                                is_val=kwargs.get('validate', False),
                            )

                if current_screenshot is not None:
                    agent_data.image_data = [current_screenshot]
                if current_info is not None:
                    agent_data.extra_fields["current_info"] = current_info
            return None

        # We always setup env at start of run/resume
        await asyncio.wrap_future(browser_actor.submit(setup_env_op))


        return agent_data, AgentState.PENDING

    def _restore_from_output(self, output: AgentLoopOutput) -> tuple[AgentData, AgentState]:
        """restore AgentState and AgentData from output"""
        agent_data = output.extra_fields.get("agent_data", None)
        agent_state = output.extra_fields.get("agent_state", None)
        if agent_data is None or agent_state is None:
            raise ValueError(f"Unexpected situation: agent_data is {agent_data}, agent_state is {agent_state}")
        return agent_data, agent_state

    async def _run_state_machine(self, agent_data: AgentData, state: AgentState, sampling_params: dict, cancellation_event: asyncio.Event, request_id: str) -> AgentState:
        
        # State machine loop
        while state != AgentState.TERMINATED:
            if cancellation_event and cancellation_event.is_set():
                return state

            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
                
            elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)

            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, self.browser_actor)
            
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
        
        return state

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        # Store initial messages and initialize action history for stepwise generation
        agent_data.extra_fields['initial_messages'] = copy.deepcopy(agent_data.messages)
        agent_data.extra_fields['action_history'] = []

        if self.processor is not None:
            # Use processor for VLM
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs, 
                ),
            )
            # Manual expansion logic for Qwen2-VL to ensure correct tokenization
            image_inputs = self.processor.image_processor(agent_data.image_data, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            
            merge_length = self.processor.image_processor.merge_size**2
            index_image = 0
            raw_prompt_vllm = raw_prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            raw_prompt_vllm_ids = self.tokenizer.encode(raw_prompt_vllm, add_special_tokens=False)
            
            full_text_template = raw_prompt
            
            while '<image>' in full_text_template:
                full_text_template = full_text_template.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index_image].prod() // merge_length) +
                    '<|vision_end|>',
                    1,
                )
                index_image += 1
            
            full_text_template = full_text_template.replace('<|placeholder|>', self.processor.image_token)
            
            # NOTE: For vLLM inference, we use raw_prompt_vllm_ids (simple expansion).
            # However, for PPO training data collection, we MIGHT need the full expansion (full_text_template).
            # The current implementation uses raw_prompt_vllm_ids for agent_data.prompt_ids, which is sent to vLLM.
            # If PPO training requires full expansion, we might need to store it separately or vLLM output might need adjustment.
            # Based on user instruction, we use raw_prompt_vllm_ids here.
            agent_data.prompt_ids = raw_prompt_vllm_ids
            agent_data.response_mask = [0] * len(raw_prompt_vllm_ids)
            
            # Capture multi_modal_inputs for the initial step
            agent_data.extra_fields["current_multi_modal_inputs"] = {k: v.clone() if hasattr(v, 'clone') else v for k, v in image_inputs.items()}
            # Store fully expanded prompt for potential training use if needed
            agent_data.extra_fields["fully_expanded_prompt_ids"] = self.tokenizer.encode(full_text_template, add_special_tokens=False)
        else:
             raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
             agent_data.prompt_ids = raw_prompt
             agent_data.response_mask = [0] * len(raw_prompt)
             agent_data.extra_fields["current_multi_modal_inputs"] = {}

        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        # Capture state before generation
        start_prompt_ids = copy.deepcopy(agent_data.prompt_ids)
        start_fully_expanded_prompt_ids = copy.deepcopy(agent_data.extra_fields.get("fully_expanded_prompt_ids", start_prompt_ids))
        start_image_data = copy.deepcopy(agent_data.image_data)
        start_multi_modal_inputs = copy.deepcopy(agent_data.extra_fields.get("current_multi_modal_inputs", {}))
        
        # Ensure we limit the generation length for this turn
        single_turn_limit = self.config.data.get("max_response_length_single_turn", 1024)
        
        # Update sampling_params safely
        current_sampling_params = sampling_params.copy()
        current_sampling_params["max_tokens"] = single_turn_limit

        with simple_timer("generate_sequences", agent_data.metrics):
             output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=current_sampling_params,
                image_data=agent_data.image_data,
            )
        
        response_ids = output.token_ids
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [1] * len(response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        agent_data.assistant_turns += 1
        
        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids)
        )
        start_prompt_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(start_prompt_ids, skip_special_tokens=False)
        )
        # Accumulate action history (clean up potentially hallucinated tokens)

        clean_response_text = response_text.replace('<image>', '').replace("<|im_end|>", "")
        if "<think>" not in clean_response_text and "</think>" in clean_response_text:
            clean_response_text = "<think>" + clean_response_text
        agent_data.extra_fields['action_history'].append(clean_response_text)

        # Store step info
        current_step = {
            "prompts": start_fully_expanded_prompt_ids,
            "vllm_prompts": start_prompt_ids,
            "responses": response_ids,
            "image_data": start_image_data,
            "multi_modal_inputs": start_multi_modal_inputs,
            "log_probs": output.log_probs, # save log probs if available
            "tool_rewards": 0.0 # Default reward
        }
        agent_data.extra_fields["current_step"] = current_step

        # Parse Action
        action = self._parse_action(response_text)
        
        if action:
            agent_data.last_action = action
            # Append assistant message to history
            # Use truncated response text (action) to avoid hallucination in history
            agent_data.messages.append({"role": "assistant", "content": response_text}) 
            return AgentState.PROCESSING_TOOLS
        else:
            # No action found or end of conversation
            # Append the full response text
            agent_data.messages.append({"role": "assistant", "content": response_text})
            
            # Final step (no tool call), finalize it
            agent_data.extra_fields["steps"].append(current_step)
            agent_data.extra_fields["current_step"] = None
            
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData, env_actor: BrowserActor) -> AgentState:
        action_text = agent_data.last_action
        request_id = agent_data.request_id
        # Define step operation
        async def step_op():
            if not hasattr(env_actor.browser_unit, 'active_envs') or request_id not in env_actor.browser_unit.active_envs:
                raise RuntimeError(f"Env session {request_id} not found on actor!")
            
            env: WebBrowserEnv = env_actor.browser_unit.active_envs[request_id]
            
            # 1. Extract tool call
            action_data = env.extract_tool_call(action_text)
            if action_data == env.INVALID_ACTION:
                 return None, env.PENALTY_FOR_INVALID, False, {"error": "Invalid Action"}
            
            tool_name = action_data['tool']
            tool_args = action_data['args']
            
            # 2. Get tool
            if tool_name not in env.tool_map:
                 return None, env.PENALTY_FOR_INVALID, False, {"error": "Unknown Tool"}
            
            tool: WebBrowserTool = env.tool_map[tool_name]
            
            # 3. Convert to web action and execute
                 # Assuming toolargs2webaction is available on WebBrowserTool instance
            web_action, _ = tool.toolargs2webaction(tool_args, env, action_text)
            if web_action:
                screenshot, reward, done, truncated, info = await env.astep(web_action)
                # Calculate specific tool reward if needed
                reward = await tool.calculate_reward(tool_args, {"content": "", "image": screenshot}, env, env.PENALTY_FOR_INEFFECTIVE)
                return screenshot, reward, done, info
            else:
                return None, env.PENALTY_FOR_INVALID, False, {"error": "Not a web browser tool or failed conversion"}

        # Execute step
        try:
            screenshot, reward, done, info = await asyncio.wrap_future(env_actor.submit(step_op))
        except (Exception, asyncio.CancelledError) as e:
            logger.error(f"Step failed or cancelled: {e} (Type: {type(e)})")
            screenshot, reward, done = None, 0.0, False
            info = {"error": str(e)}
            # If cancelled, we might want to re-raise to stop the loop immediately

        # Update history
        agent_data.tool_rewards.append(reward)
        
        # Finalize current step
        current_step = agent_data.extra_fields.get("current_step")
        if current_step:
             current_step["tool_rewards"] = reward
             agent_data.extra_fields["steps"].append(current_step)
             agent_data.extra_fields["current_step"] = None

        image_description = ""
        if info.get('image_description') is not None:
            image_description = info.get('image_description')
            image_description = f"The Description of the Web Browser ScreenShot: {image_description}"

        current_page_num = 0
        for page_ind, page_i in enumerate(env_actor.browser_unit.active_envs[request_id].context.pages):
            if page_i == env_actor.browser_unit.active_envs[request_id].page:
                current_page_num = page_ind
                break

        # Create observation message
        content = []
        new_images = []
        if screenshot is not None:
            content.append({"type": "image"})
            new_images.append(screenshot)
            # Limit image history to the latest one to match agent_r1 logic and save memory
            #
            #agent_data.image_data = [screenshot] 
            agent_data.image_data.append(screenshot)
        # Add text observation if available (e.g. from info)
        obs_text = f"Step {env_actor.browser_unit.active_envs[request_id].current_step}. This Tab Number is {current_page_num} in the browser. \n{image_description}"
        env_actor.browser_unit.active_envs[request_id].current_step += 1
        if info.get("error"):
             obs_text = f"Error: {info['error']} " + obs_text

        if obs_text:
            content.append({"type": "text", "text": obs_text})

        agent_data.extra_fields['action_history'][-1] += f"\n<tool_response>\n{obs_text}\n</tool_response>\n"

        if not content:
             content.append({"type": "text", "text": "Action executed."})

        new_message = {"role": "user", "content": content}
        agent_data.messages.append(new_message)
        
        # Check termination
        if agent_data.assistant_turns >= self.max_turns:
            return AgentState.TERMINATED
        
        # Update prompt_ids with new tokens using Stepwise logic (Initial Prompt + History + Latest Image)
        if self.processor is not None:
            initial_messages = copy.deepcopy(agent_data.extra_fields['initial_messages'])
            action_history = agent_data.extra_fields['action_history']
            history_text = "\n".join(action_history)
            
            # Inject history into initial user message
            found_user = False
            for msg in reversed(initial_messages):
                if msg['role'] == 'user':
                    msg['content'] = msg['content'] + "\nAction History:\n" + history_text
                    found_user = True
                    break
            if not found_user:
                initial_messages.append({"role": "user", "content": "\nAction History:\n" + history_text})

            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    initial_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            
            current_multi_modal_inputs = {}
            target_images = []
            if new_images:
                target_images = new_images
            elif agent_data.image_data:
                target_images = [agent_data.image_data[-1]]

            if target_images:
                # Update agent state to only hold the current image
                agent_data.image_data = target_images
                
                # Manual expansion logic for Qwen2-VL
                image_inputs = self.processor.image_processor(target_images, return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']
                
                full_text_template = copy.deepcopy(raw_prompt)
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index_image = 0
                    while '<image>' in full_text_template:
                        full_text_template = full_text_template.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index_image].prod() // merge_length) +
                            '<|vision_end|>',
                            1,
                        )
                        index_image += 1
                    full_text_template = full_text_template.replace('<|placeholder|>',
                                                                    self.processor.image_token)
                input_ids_training = self.tokenizer.encode(full_text_template, add_special_tokens=False)
                agent_data.extra_fields["fully_expanded_prompt_ids"] = input_ids_training
                # vLLM format
                raw_prompt_vllm = raw_prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                new_ids = self.tokenizer.encode(raw_prompt_vllm, add_special_tokens=False)
                
                
                current_multi_modal_inputs = {k: v.clone() if hasattr(v, 'clone') else v for k, v in image_inputs.items()}
            else:
                # Text only case
                new_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                agent_data.extra_fields["fully_expanded_prompt_ids"] = new_ids
            
            agent_data.prompt_ids = new_ids
            agent_data.response_mask = [0] * len(new_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs = [0.0] * len(new_ids)
            
            # Save multi_modal_inputs for the NEXT step
            agent_data.extra_fields["current_multi_modal_inputs"] = current_multi_modal_inputs

        else:
             # Text only fallback
             initial_messages = copy.deepcopy(agent_data.extra_fields['initial_messages'])
             action_history = agent_data.extra_fields['action_history']
             history_text = "\n".join(action_history)
             
             found_user = False
             for msg in reversed(initial_messages):
                if msg['role'] == 'user':
                    msg['content'] = msg['content'] + "\nAction History:\n" + history_text
                    found_user = True
                    break
             if not found_user:
                 initial_messages.append({"role": "user", "content": "\nAction History:\n" + history_text})

             raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    initial_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
             agent_data.prompt_ids = raw_prompt
             agent_data.response_mask = [0] * len(raw_prompt)
             if agent_data.response_logprobs:
                agent_data.response_logprobs = [0.0] * len(raw_prompt)
             agent_data.extra_fields["current_multi_modal_inputs"] = {}
            
        return AgentState.GENERATING

    def _parse_action(self, response_text):
        # Implement parsing logic using tool_call pattern
        tool_pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(tool_pattern, response_text, re.DOTALL)
        
        if not match:
            return None # No tool call found, treat as thought/end
        
        # Truncate response to include only up to the first tool call end
        tool_call_end = getattr(self, 'tool_call_end', '</tool_call>')
        response_text = response_text.split(tool_call_end)[0] + tool_call_end
        
        return response_text

    def prime_norm(self, rewards: list[float]) -> list[float]:
        """
        Applies normalization to token-level scores.
        """
        if not rewards:
            return []
        
        # Convert to tensor for easier manipulation
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # 1. Calculate reverse cumulative sum (Returns)
        reverse_cumsum = torch.cumsum(rewards_tensor.flip(dims=[0]), dim=0).flip(dims=[0])
        
        # 2. Find normalization factor (max absolute return)
        norm_factor = reverse_cumsum.abs().max() + 1e-8
        
        # 3. Normalize
        normalized_rewards = rewards_tensor / norm_factor
        
        return normalized_rewards.tolist()

    async def _build_completed_output(self, agent_data: AgentData, param_version: int) -> AgentLoopOutput:
        final_reward = 0.0
        answer_score = 0.0
        is_bbox_evaluator = False
        assistant_blocks = []
        # for ind, msg_i in enumerate(agent_data.messages):
        #     if msg_i["role"] != "system":
        #         if msg_i["role"] == "user":
        #             #防止历史对话中的gpt对答案干扰，在用户输入中插入一个用户查询的placeholder
        #             #得加user 要不然qwen 3的chattemplate会把think丢掉
        #             assistant_blocks.append({
        #                 "role": "user",
        #                 "content": "user query placeholder"
        #             })
        #         else:
        #             assistant_blocks.append(msg_i)

        # full_sequence_str = await self.loop.run_in_executor(
        #         None,
        #         lambda: self.processor.apply_chat_template(
        #             assistant_blocks,
        #             add_generation_prompt=True,
        #             tokenize=False,
        #             **self.apply_chat_template_kwargs, 
        #         ),
        # )

        #使用 agent_data.extra_fields["steps"] 来重构完整的会话
        full_sequence_ids = []
        for step_ind, step_i in enumerate(agent_data.extra_fields["steps"]):
            start_prompt_ids = step_i["vllm_prompts"]
            response_ids = step_i["responses"]
            full_sequence_ids.extend(start_prompt_ids)
            full_sequence_ids.extend(response_ids)

        full_sequence_str = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(full_sequence_ids, skip_special_tokens=False)
        )
        if hasattr(self, 'browser_actor') and self.browser_actor:
            request_id = agent_data.request_id
            async def get_score_op():
                if hasattr(self.browser_actor.browser_unit, 'active_envs') and request_id in self.browser_actor.browser_unit.active_envs:
                    env: WebBrowserEnv = self.browser_actor.browser_unit.active_envs[request_id]
                    # AsyncScriptBrowserEnv/WebBrowserEnv get_score takes solution_str
                    score = await env.get_score(full_sequence_str)

                    is_bbox = False
                    if hasattr(env, 'evaluator') and env.evaluator.__class__.__name__ == 'BboxJudgeEvaluator':
                        is_bbox = True
                    
                    return score, is_bbox
                else:
                    logger.info(f"[AsyncWebAgentLoop] get_score_op env not found")
                    return 0.0, False
             
            try:
                answer_score, is_bbox_evaluator = await asyncio.wrap_future(self.browser_actor.submit(get_score_op))
            except Exception as e:
                logger.error(f"Failed to get final score: {e}")

        # Calculate Format Score
        format_score = compute_score_format(
            solution_str=full_sequence_str
        )

        # Determine Threshold
        EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', "")
        format_score_threshold = 1.0
        if is_bbox_evaluator \
            or "_rft".lower() in EXPERIMENT_NAME.lower() \
            or "_forRL".lower() in EXPERIMENT_NAME.lower() \
            or "checkpoint-".lower() in EXPERIMENT_NAME.lower() \
            or "stepwise".lower() in EXPERIMENT_NAME.lower():
            format_score_threshold = 0.5
        else:
            format_score_threshold = 1.0

        # Combine Scores
        # Check valid_response_length? 
        # Here we assume we have a response since it's completed_output.
        # generation.py checks: if task_data['valid_response_length'] > 0:
        # Here agent_data.assistant_turns > 0 presumably, or response_ids is not empty.
        score = 0.0
        logger.info(f"[AsyncWebAgentLoop] format_score: {format_score}, format_score_threshold: {format_score_threshold}, answer_score: {answer_score}")
        format_score_tmp = min(format_score, format_score_threshold)
        if format_score_tmp >= format_score_threshold:
            score = -format_score_threshold + format_score_tmp
            if answer_score is not None:
                score += answer_score
        else:
            score = -format_score_threshold + format_score_tmp
            
        final_reward = score
        # Combine step rewards and final reward logic from StepwiseToolGenerationManager
        steps = agent_data.extra_fields.get("steps", [])
        
        # 1. Normalize process rewards
        if steps and "_wPR" in os.environ.get('EXPERIMENT_NAME', ""):
             raw_tool_rewards = [step["tool_rewards"] for step in steps]
             normalized_process_rewards = self.prime_norm(raw_tool_rewards)
        else:
             normalized_process_rewards = [0.0] * len(steps)

        # 2. Assign token level scores and build flattened samples
        flattened_steps = []
        for i, step in enumerate(steps):
             # Calculate token_level_score
             current_reward = 0.0
             if "_wPR" in os.environ.get('EXPERIMENT_NAME', ""):
                  current_reward += 0.1 * normalized_process_rewards[i]
             
             # Add final reward to EVERY step.
             # For stepwise training where each step is a sample, we attribute the final outcome
             # to all steps in the trajectory.
             current_reward += final_reward

             response_ids = step["responses"]
             token_level_scores = torch.zeros(len(response_ids), dtype=torch.float32)
             if len(response_ids) > 0:
                 token_level_scores[-1] = current_reward
             
             # Construct sample
             sample = {
                 "prompts": torch.tensor(step["prompts"], dtype=torch.long),
                 "responses": torch.tensor(response_ids, dtype=torch.long),
                 "token_level_scores": token_level_scores,
                 "format_score": format_score,
                 "answer_score": answer_score,
                 "reward_score": final_reward,
                 "turns": agent_data.assistant_turns, # Total turns? or current? Stepwise uses 'episode_turns' which is total.
                 "multi_modal_data": {"image": step["image_data"]} if step["image_data"] else {},
                 "multi_modal_inputs": step["multi_modal_inputs"],
                 # Add other metadata as needed
                 "log_probs": step.get("log_probs")
             }
             flattened_steps.append(sample)

        # For backward compatibility, populate tool_rewards in agent_data
        # Combine step rewards and final reward for normalization (Legacy logic)
        if agent_data.tool_rewards:
            agent_data.tool_rewards[-1] += final_reward
        else:
            agent_data.tool_rewards.append(final_reward)
            
        # Normalize (Legacy)
        if "_wPR" in os.environ.get('EXPERIMENT_NAME', ""): 
             agent_data.tool_rewards = self.prime_norm(agent_data.tool_rewards)

        full_seq = agent_data.prompt_ids
        response_mask = agent_data.response_mask
        initial_prompt_len = len(full_seq) - sum(response_mask)
        response_len = sum(response_mask)
        fully_expanded_prompt_ids = agent_data.extra_fields.get("fully_expanded_prompt_ids", full_seq)
        prompt_ids = fully_expanded_prompt_ids
        pure_prompt_ids = full_seq[:initial_prompt_len]
        response_ids = full_seq[initial_prompt_len:]
        response_mask = response_mask[initial_prompt_len:]
        assert len(response_mask) == len(response_ids), f"response_mask: {len(response_mask)}, response_ids: {len(response_ids)}"

        response_logprobs = None
        if agent_data.response_logprobs:
            response_logprobs = agent_data.response_logprobs[initial_prompt_len:]

        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data else {}
        output = AgentLoopOutput(
            prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
            response_ids=torch.tensor(response_ids[: self.response_length], dtype=torch.long), # Truncate if needed
            response_mask=torch.tensor(response_mask[: self.response_length], dtype=torch.long),
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            num_turns=agent_data.assistant_turns,
            metrics=agent_data.metrics,
            reward_score=final_reward,
            extra_fields={
                "reward_extra_info": {"format_score": format_score, "answer_score": answer_score, "final_reward": final_reward}
            },
        )
        output.extra_fields.update({
            "tool_rewards": agent_data.tool_rewards,
            "is_cancel": False,
            "param_version_start": agent_data.extra_fields["param_version_start"],
            "param_version_end": param_version,
            "browser_actor_endpoint": agent_data.extra_fields.get("browser_actor_endpoint"),
            "flattened_steps": flattened_steps # Store the stepwise samples here
        })
        return output

    def _build_cancelled_output(self, agent_data: AgentData, state) -> AgentLoopOutput:
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            multi_modal_data={},
            # Return empty tensor instead of None to avoid TypeError in postprocess
            response_logprobs=[],
            num_turns=0,
            metrics=agent_data.metrics,
            extra_fields={
                "is_cancel": True,
                "agent_data": agent_data,
                "agent_state": state,
            },
        )
