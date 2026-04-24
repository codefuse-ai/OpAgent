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
Dual-Model Web Agent Loop: splits the single GENERATING state into PLANNING + GROUNDING.

- PLANNING: calls the Planner model to produce <think>...</think>
- GROUNDING: calls the Grounder model to produce <tool_call>...</tool_call>

Joint reward is computed identically to the original single-model loop.
Each step records a `role` field ("planner" or "grounder") for independent parameter updates.
"""

import asyncio
import copy
import logging
import os
import re
import time as _time

import torch
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, AgentData
from verl.utils.profiler import simple_timer

from recipe.webagent_fully_async_policy.agent_loop.web_agent_loop import AsyncWebAgentLoop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
logger.setLevel("INFO")


@register("dual_model_async_web_agent")
class DualModelAsyncWebAgentLoop(AsyncWebAgentLoop):
    """
    Dual-model variant of AsyncWebAgentLoop.
    
    Key differences from parent:
    1. PENDING -> PLANNING (instead of GENERATING)
    2. PLANNING calls planner engine -> produces think tokens -> GROUNDING
    3. GROUNDING calls grounder engine -> produces tool_call tokens -> PROCESSING_TOOLS
    4. PROCESSING_TOOLS -> PLANNING (instead of GENERATING)
    5. Each step in flattened_steps has a "role" field for training batch split
    """

    # ---- Default system prompts ------------------------------------------------
    DEFAULT_PLANNER_SYSTEM_PROMPT = (
        "You are a web navigation planning assistant. "
        "Given the current webpage screenshot, user's task, and action history, "
        "analyze the situation and reason about what action should be taken next.\n\n"
        "Output your reasoning inside <think>...</think> tags. In your reasoning:\n"
        "1. Describe the main content of the current screenshot\n"
        "2. Analyze the current state relative to the user's goal\n"
        "3. Determine what specific action to take next and why, "
        "OR determine that the task is complete and what the answer is\n\n"
        "Do NOT output any tool calls or answers directly. Only provide reasoning."
    )

    DEFAULT_GROUNDER_SYSTEM_PROMPT_TEMPLATE = (
        "You are a web navigation execution assistant.\n\n"
        "{tools_description}\n\n"
        "Based on the reasoning provided in <think>...</think> before your turn, "
        "select and execute exactly ONE action:\n"
        "- If an action is needed, output it in <tool_call>...</tool_call>\n"
        "- If the task is complete, output the answer in <answer>...</answer>"
    )

    def __init__(self, trainer_config, **kwargs):
        super().__init__(trainer_config, **kwargs)
        # Dual model configuration
        dual_model_config = trainer_config.config.get("dual_model", {})
        self.planner_stop_tokens = list(dual_model_config.get("planner_stop_tokens", ["</think>"]))
        self.grounder_stop_tokens = list(dual_model_config.get("grounder_stop_tokens", ["</tool_call>", "</answer>"]))
        self.enable_dual_model = dual_model_config.get("enable", False)

        # --- Role-specific system prompts -------------------------------------------
        self.planner_system_prompt = dual_model_config.get(
            "planner_system_prompt", None
        ) or self.DEFAULT_PLANNER_SYSTEM_PROMPT

        grounder_custom = dual_model_config.get("grounder_system_prompt", None)
        if grounder_custom:
            self.grounder_system_prompt = grounder_custom
        else:
            # Will be finalised in _handle_pending_state once we know the
            # original tool description that was injected by the dataset.
            self.grounder_system_prompt = None  # deferred

    # ---- Prompt helper utilities -----------------------------------------------

    def _extract_tools_description(self, messages: list) -> str:
        """Extract the tools description block from the original system message.

        The dataset / ``_init_system_prompt`` puts the URL info + tool description
        into ``messages[0]['content']``.  We need the tool description portion so
        that the grounder system prompt can include it.
        """
        if not messages:
            return ""
        sys_content = messages[0].get("content", "") if messages[0].get("role") == "system" else ""
        # The tool description starts with the text produced by
        # ``WebBrowserEnv.tools_format_func()`` which always begins with
        # 'You are a helpful assistant.' or '# Tools'.  We look for the
        # first occurrence of '# Tools' as the anchor.
        idx = sys_content.find("# Tools")
        if idx >= 0:
            return sys_content[idx:]
        # If no anchor found, return the full system content as fallback
        return sys_content

    def _extract_url_info(self, messages: list) -> str:
        """Extract the URL / port-map info that ``_init_system_prompt`` prepends.

        The system message content follows the pattern::

            url_info + tools_format_func()

        ``tools_format_func()`` starts with either:
        - ``"You are a helpful assistant.\\n\\n# Tools..."`` (WebBrowserEnv)
        - ``"# Tools..."`` (ToolEnv)

        We try the most-specific anchor first to avoid including the
        ``"You are a helpful assistant."`` prefix in the URL portion.
        """
        if not messages:
            return ""
        sys_content = messages[0].get("content", "") if messages[0].get("role") == "system" else ""
        # Check anchors from most specific to least specific
        for anchor in ["You are a helpful assistant.", "# Tools"]:
            idx = sys_content.find(anchor)
            if idx > 0:
                return sys_content[:idx]
        return ""

    def _build_role_messages(self, original_messages: list, system_prompt: str) -> list:
        """Replace the system message content with *system_prompt*, keeping
        everything else (user messages, images, etc.) intact."""
        msgs = copy.deepcopy(original_messages)
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = system_prompt
        else:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        return msgs

    async def _tokenize_messages_for_vllm(self, messages: list, agent_data: AgentData):
        """Tokenize *messages* into vLLM prompt ids + fully-expanded ids.

        Returns ``(vllm_prompt_ids, fully_expanded_ids, multi_modal_inputs)``.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            target_images = agent_data.image_data if agent_data.image_data else []
            multi_modal_inputs = {}

            if target_images:
                image_inputs = self.processor.image_processor(target_images, return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']

                # Fully-expanded (for training)
                full_text = copy.deepcopy(raw_prompt)
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size ** 2
                    idx_img = 0
                    while '<image>' in full_text:
                        full_text = full_text.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[idx_img].prod() // merge_length) + '<|vision_end|>',
                            1,
                        )
                        idx_img += 1
                    full_text = full_text.replace('<|placeholder|>', self.processor.image_token)

                fully_expanded_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

                # vLLM format
                raw_prompt_vllm = raw_prompt.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                vllm_ids = self.tokenizer.encode(raw_prompt_vllm, add_special_tokens=False)

                multi_modal_inputs = {k: v.clone() if hasattr(v, 'clone') else v for k, v in image_inputs.items()}
            else:
                vllm_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                fully_expanded_ids = vllm_ids

            return vllm_ids, fully_expanded_ids, multi_modal_inputs
        else:
            # Text-only fallback
            ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            return ids, ids, {}

    def _inject_action_history(self, messages: list, action_history: list) -> list:
        """Append action history text to the last user message."""
        msgs = copy.deepcopy(messages)
        if not action_history:
            return msgs
        history_text = "\nAction History:\n" + "\n".join(action_history)
        for msg in reversed(msgs):
            if msg['role'] == 'user':
                msg['content'] = msg['content'] + history_text
                return msgs
        msgs.append({"role": "user", "content": history_text})
        return msgs

    # ---- State machine ----------------------------------------------------------

    async def _run_state_machine(self, agent_data: AgentData, state: AgentState,
                                  sampling_params: dict, cancellation_event: asyncio.Event,
                                  request_id: str) -> AgentState:
        """Override state machine to handle PLANNING and GROUNDING states."""
        if not self.enable_dual_model:
            # Fallback to single-model behavior
            return await super()._run_state_machine(agent_data, state, sampling_params,
                                                     cancellation_event, request_id)

        turn_count = 0
        sm_start = _time.time()
        while state != AgentState.TERMINATED:
            if cancellation_event and cancellation_event.is_set():
                return state

            turn_start = _time.time()
            prev_state = state

            if state == AgentState.PENDING:
                state = await self._handle_pending_state_dual(agent_data, sampling_params)

            elif state == AgentState.PLANNING:
                state = await self._handle_planning_state(agent_data, sampling_params)

            elif state == AgentState.GROUNDING:
                state = await self._handle_grounding_state(agent_data, sampling_params)

            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, self.browser_actor)

            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

            turn_time = _time.time() - turn_start
            turn_count += 1
            logger.info(
                f"[DualModel] Turn {turn_count}: {prev_state} -> {state}, "
                f"time={turn_time:.2f}s"
            )

        total_sm_time = _time.time() - sm_start
        logger.info(
            f"[DualModel] State machine completed: {turn_count} turns, "
            f"total_time={total_sm_time:.2f}s"
        )
        agent_data.metrics["state_machine_total_time"] = total_sm_time
        agent_data.metrics["state_machine_turn_count"] = turn_count
        return state

    async def _handle_pending_state_dual(self, agent_data: AgentData,
                                          sampling_params: dict[str, Any]) -> AgentState:
        """Initialise dual-model session: build separate planner / grounder messages
        and tokenize the planner prompt for the first PLANNING turn."""
        pending_start = _time.time()

        # Let the parent do all the heavy lifting first (env setup, initial_messages,
        # action_history, prompt_ids, multi_modal_inputs, fully_expanded_prompt_ids).
        _next = await super()._handle_pending_state(agent_data, sampling_params)
        # _next is AgentState.GENERATING; we will override below.

        original_messages = agent_data.extra_fields['initial_messages']  # already deep-copied

        # ---- Extract URL info and tools description from the original system msg
        url_info = self._extract_url_info(original_messages)
        tools_desc = self._extract_tools_description(original_messages)

        # ---- Finalise grounder system prompt if not yet set
        if self.grounder_system_prompt is None:
            self.grounder_system_prompt = self.DEFAULT_GROUNDER_SYSTEM_PROMPT_TEMPLATE.format(
                tools_description=tools_desc,
            )

        # ---- Build role-specific initial messages
        planner_sys = url_info + self.planner_system_prompt
        grounder_sys = url_info + self.grounder_system_prompt

        planner_initial = self._build_role_messages(original_messages, planner_sys)
        grounder_initial = self._build_role_messages(original_messages, grounder_sys)

        agent_data.extra_fields['planner_initial_messages'] = planner_initial
        agent_data.extra_fields['grounder_initial_messages'] = grounder_initial

        # ---- Re-tokenize with planner prompt (first turn goes to planner)
        vllm_ids, expanded_ids, mm_inputs = await self._tokenize_messages_for_vllm(
            planner_initial, agent_data,
        )
        agent_data.prompt_ids = vllm_ids
        agent_data.response_mask = [0] * len(vllm_ids)
        agent_data.extra_fields["fully_expanded_prompt_ids"] = expanded_ids
        agent_data.extra_fields["current_multi_modal_inputs"] = mm_inputs

        logger.info(f"[DualModel] PENDING -> PLANNING  planner_prompt_len={len(vllm_ids)}, time={_time.time() - pending_start:.2f}s")
        return AgentState.PLANNING

    # ---- PLANNING state ---------------------------------------------------------

    async def _handle_planning_state(self, agent_data: AgentData,
                                      sampling_params: dict[str, Any]) -> AgentState:
        """
        PLANNING state: Call the Planner model to generate thinking tokens.
        
        The planner produces: <think>reasoning content</think>
        After planning, we transition to GROUNDING state.
        """
        # Capture state before generation (planner-specific prompt)
        start_prompt_ids = copy.deepcopy(agent_data.prompt_ids)
        start_fully_expanded_prompt_ids = copy.deepcopy(
            agent_data.extra_fields.get("fully_expanded_prompt_ids", start_prompt_ids)
        )
        start_image_data = copy.deepcopy(agent_data.image_data)
        start_multi_modal_inputs = copy.deepcopy(
            agent_data.extra_fields.get("current_multi_modal_inputs", {})
        )

        single_turn_limit = self.config.data.get("max_response_length_single_turn", 1024)

        current_sampling_params = sampling_params.copy()
        current_sampling_params["max_tokens"] = single_turn_limit
        # Add stop tokens specific to planner
        current_sampling_params["stop"] = self.planner_stop_tokens
        current_sampling_params["include_stop_str_in_output"] = True

        with simple_timer("generate_sequences_planner", agent_data.metrics):
            # Route to planner vLLM engine via DualModelServerManager
            generate_kwargs = dict(
                request_id=agent_data.request_id + "_planner",
                prompt_ids=agent_data.prompt_ids,
                sampling_params=current_sampling_params,
                image_data=agent_data.image_data,
            )
            if hasattr(self.server_manager, 'grounder_handles'):
                # DualModelServerManager supports engine_tag
                generate_kwargs["engine_tag"] = "planner"
            output = await self.server_manager.generate(**generate_kwargs)

        response_ids = output.token_ids
        agent_data.response_ids = response_ids
        # NOTE: do NOT append to agent_data.prompt_ids yet — we will rebuild
        # the grounder prompt from scratch in _handle_grounding_state.
        agent_data.response_mask += [1] * len(response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids)
        )

        # Store planner step info (for independent training)
        planner_step = {
            "prompts": start_fully_expanded_prompt_ids,
            "vllm_prompts": start_prompt_ids,
            "responses": response_ids,
            "image_data": start_image_data,
            "multi_modal_inputs": start_multi_modal_inputs,
            "log_probs": output.log_probs,
            "tool_rewards": 0.0,
            "role": "planner",  # <-- Role tag for training split
        }
        agent_data.extra_fields["current_planner_step"] = planner_step

        # Check if the planner output contains a tool_call directly
        # (shouldn't happen with proper stop tokens, but handle gracefully)
        if "</tool_call>" in response_text:
            logger.warning("Planner output contains tool_call - treating as single-model fallback")
            agent_data.prompt_ids += response_ids  # restore for combined handling
            return await self._handle_combined_output(agent_data, response_text, planner_step)

        # Check if planner terminated without producing think tags (end of conversation)
        if "</think>" not in response_text and "<tool_call>" not in response_text:
            # Planner decided to end without action
            agent_data.prompt_ids += response_ids
            agent_data.messages.append({"role": "assistant", "content": response_text})
            planner_step["tool_rewards"] = 0.0
            agent_data.extra_fields["steps"].append(planner_step)
            agent_data.extra_fields["current_planner_step"] = None
            return AgentState.TERMINATED

        # Planner produced thinking. Now transition to GROUNDING.
        logger.info(f"[DualModel] PLANNING completed, think tokens: {len(response_ids)}")
        return AgentState.GROUNDING

    # ---- GROUNDING state --------------------------------------------------------

    async def _handle_grounding_state(self, agent_data: AgentData,
                                       sampling_params: dict[str, Any]) -> AgentState:
        """
        GROUNDING state: Build a grounder-specific prompt (with tools, without
        planner system prompt) and append the planner's think tokens as the
        assistant prefix.  Then call the Grounder model to generate
        ``<tool_call>...</tool_call>`` or ``<answer>...</answer>``.
        """
        grounding_start = _time.time()
        planner_step = agent_data.extra_fields.get("current_planner_step")
        planner_response_ids = planner_step["responses"] if planner_step else []

        # ---- Build grounder prompt from grounder_initial_messages + history
        tokenize_start = _time.time()
        grounder_initial = agent_data.extra_fields['grounder_initial_messages']
        action_history = agent_data.extra_fields.get('action_history', [])
        grounder_msgs = self._inject_action_history(grounder_initial, action_history)

        grounder_vllm_ids, grounder_expanded_ids, grounder_mm = (
            await self._tokenize_messages_for_vllm(grounder_msgs, agent_data)
        )
        tokenize_time = _time.time() - tokenize_start
        agent_data.metrics["grounder_tokenize_time"] = agent_data.metrics.get("grounder_tokenize_time", 0) + tokenize_time

        # Append planner's think tokens as assistant prefix so the grounder
        # sees:  ....<|im_start|>assistant\n<think>planner reasoning</think>
        grounder_prompt_ids = grounder_vllm_ids + planner_response_ids
        grounder_expanded_prompt_ids = grounder_expanded_ids + planner_response_ids

        # Snapshot for step recording
        start_prompt_ids = copy.deepcopy(grounder_prompt_ids)
        start_fully_expanded_prompt_ids = copy.deepcopy(grounder_expanded_prompt_ids)
        start_image_data = copy.deepcopy(agent_data.image_data)
        start_multi_modal_inputs = copy.deepcopy(grounder_mm if grounder_mm else
            agent_data.extra_fields.get("current_multi_modal_inputs", {}))

        single_turn_limit = self.config.data.get("max_response_length_single_turn", 1024)

        current_sampling_params = sampling_params.copy()
        current_sampling_params["max_tokens"] = single_turn_limit
        current_sampling_params["stop"] = self.grounder_stop_tokens
        current_sampling_params["include_stop_str_in_output"] = True

        with simple_timer("generate_sequences_grounder", agent_data.metrics):
            generate_kwargs = dict(
                request_id=agent_data.request_id + "_grounder",
                prompt_ids=grounder_prompt_ids,
                sampling_params=current_sampling_params,
                image_data=agent_data.image_data,
            )
            if hasattr(self.server_manager, 'grounder_handles'):
                generate_kwargs["engine_tag"] = "grounder"
            output = await self.server_manager.generate(**generate_kwargs)

        response_ids = output.token_ids
        agent_data.response_ids = response_ids

        # Update the *shared* trajectory bookkeeping.
        # In dual-model mode the flat trajectory is secondary (per-step records
        # are used for training), but we keep prompt_ids / response_mask in sync
        # so that any utility reading them doesn't crash.
        agent_data.prompt_ids = grounder_prompt_ids + response_ids
        # Reset (not append) so mask length matches prompt_ids
        agent_data.response_mask = (
            [0] * len(grounder_vllm_ids)
            + [1] * len(planner_response_ids)
            + [1] * len(response_ids)
        )
        planner_lp = list(planner_step.get("log_probs", []) or []) if planner_step else []
        if len(planner_lp) != len(planner_response_ids):
            planner_lp = [0.0] * len(planner_response_ids)
        agent_data.response_logprobs = (
            [0.0] * len(grounder_vllm_ids)
            + planner_lp
            + (output.log_probs if output.log_probs else [0.0] * len(response_ids))
        )

        agent_data.assistant_turns += 1

        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids)
        )

        # Reconstruct the full response (planner think + grounder action) for history
        if planner_step:
            planner_response_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(planner_step["responses"])
            )
            full_response_text = planner_response_text + response_text
        else:
            full_response_text = response_text

        # Clean up for action history
        clean_response_text = full_response_text.replace('<image>', '').replace("<|im_end|>", "")
        if "<think>" not in clean_response_text and "</think>" in clean_response_text:
            clean_response_text = "<think>" + clean_response_text
        agent_data.extra_fields['action_history'].append(clean_response_text)

        # Store grounder step info (for independent training)
        grounder_step = {
            "prompts": start_fully_expanded_prompt_ids,
            "vllm_prompts": start_prompt_ids,
            "responses": response_ids,
            "image_data": start_image_data,
            "multi_modal_inputs": start_multi_modal_inputs,
            "log_probs": output.log_probs,
            "tool_rewards": 0.0,
            "role": "grounder",
        }
        agent_data.extra_fields["current_grounder_step"] = grounder_step

        # Parse action from the full response
        action = self._parse_action(full_response_text)

        if action:
            agent_data.last_action = action
            agent_data.messages.append({"role": "assistant", "content": full_response_text})
            grounding_time = _time.time() - grounding_start
            agent_data.metrics["grounding_state_time"] = agent_data.metrics.get("grounding_state_time", 0) + grounding_time
            return AgentState.PROCESSING_TOOLS
        else:
            agent_data.messages.append({"role": "assistant", "content": full_response_text})
            if planner_step:
                agent_data.extra_fields["steps"].append(planner_step)
                agent_data.extra_fields["current_planner_step"] = None
            agent_data.extra_fields["steps"].append(grounder_step)
            agent_data.extra_fields["current_grounder_step"] = None
            grounding_time = _time.time() - grounding_start
            agent_data.metrics["grounding_state_time"] = agent_data.metrics.get("grounding_state_time", 0) + grounding_time
            return AgentState.TERMINATED

    # ---- PROCESSING_TOOLS state (override) ------------------------------------

    async def _handle_processing_tools_state(self, agent_data: AgentData,
                                              env_actor) -> AgentState:
        """
        Override to:
        1. Finalize planner/grounder steps with tool reward.
        2. Delegate to parent for actual tool execution & observation.
        3. Rebuild prompt with *planner* initial messages for next PLANNING turn.
        """
        tool_exec_start = _time.time()
        grounder_step = agent_data.extra_fields.get("current_grounder_step")

        # Safety: ensure current_step is None so parent doesn't double-append
        agent_data.extra_fields["current_step"] = None

        # Call parent implementation (executes tool, updates action_history,
        # appends observation, rebuilds prompt_ids from initial_messages).
        next_state = await super()._handle_processing_tools_state(agent_data, env_actor)

        # ---- Finalize steps with tool reward
        if grounder_step is not None:
            tool_reward = 0.0
            if agent_data.tool_rewards:
                tool_reward = agent_data.tool_rewards[-1]
            grounder_step["tool_rewards"] = tool_reward

            planner_step = agent_data.extra_fields.get("current_planner_step")
            if planner_step:
                planner_step["tool_rewards"] = tool_reward
                agent_data.extra_fields["steps"].append(planner_step)
                agent_data.extra_fields["current_planner_step"] = None

            agent_data.extra_fields["steps"].append(grounder_step)
            agent_data.extra_fields["current_grounder_step"] = None

        # ---- Re-tokenize with planner prompt for the next PLANNING turn.
        # The parent rebuilt prompt_ids using ``initial_messages`` (which is the
        # *original* shared system prompt).  We need to replace it with the
        # planner-specific messages.
        if next_state == AgentState.GENERATING:
            # Map GENERATING -> PLANNING and rebuild with planner prompt
            planner_initial = agent_data.extra_fields.get('planner_initial_messages')
            if planner_initial is not None:
                action_history = agent_data.extra_fields.get('action_history', [])
                planner_msgs = self._inject_action_history(planner_initial, action_history)

                vllm_ids, expanded_ids, mm_inputs = await self._tokenize_messages_for_vllm(
                    planner_msgs, agent_data,
                )
                agent_data.prompt_ids = vllm_ids
                agent_data.response_mask = [0] * len(vllm_ids)
                if agent_data.response_logprobs:
                    agent_data.response_logprobs = [0.0] * len(vllm_ids)
                agent_data.extra_fields["fully_expanded_prompt_ids"] = expanded_ids
                agent_data.extra_fields["current_multi_modal_inputs"] = mm_inputs

            next_state = AgentState.PLANNING

        tool_exec_time = _time.time() - tool_exec_start
        agent_data.metrics["tool_execution_time"] = agent_data.metrics.get("tool_execution_time", 0) + tool_exec_time

        return next_state

    async def _handle_combined_output(self, agent_data: AgentData,
                                       response_text: str,
                                       planner_step: dict) -> AgentState:
        """
        Handle case where planner accidentally produced both think and tool_call.
        Treat as single combined step with role='planner'.
        """
        clean_response_text = response_text.replace('<image>', '').replace("<|im_end|>", "")
        if "<think>" not in clean_response_text and "</think>" in clean_response_text:
            clean_response_text = "<think>" + clean_response_text
        agent_data.extra_fields['action_history'].append(clean_response_text)

        agent_data.assistant_turns += 1

        # Use current_planner_step (NOT current_step) to avoid double-append
        # when _handle_processing_tools_state is called via parent
        agent_data.extra_fields["current_planner_step"] = planner_step
        agent_data.extra_fields["current_step"] = None  # prevent parent from also appending
        action = self._parse_action(response_text)

        if action:
            agent_data.last_action = action
            agent_data.messages.append({"role": "assistant", "content": response_text})
            return AgentState.PROCESSING_TOOLS
        else:
            agent_data.messages.append({"role": "assistant", "content": response_text})
            agent_data.extra_fields["steps"].append(planner_step)
            agent_data.extra_fields["current_planner_step"] = None
            return AgentState.TERMINATED

    async def _build_completed_output(self, agent_data: AgentData,
                                       param_version: int) -> AgentLoopOutput:
        """
        Override to ensure role tags are propagated to flattened_steps.
        The parent's _build_completed_output already reads from agent_data.extra_fields["steps"].
        We just need to ensure steps have the "role" field (which we set above).
        """
        output = await super()._build_completed_output(agent_data, param_version)

        # Verify and propagate role tags to flattened_steps
        flattened_steps = output.extra_fields.get("flattened_steps", [])
        steps = agent_data.extra_fields.get("steps", [])

        # The parent's _build_completed_output creates flattened_steps from steps.
        # We need to ensure the role field is copied over.
        step_idx = 0
        for fstep in flattened_steps:
            if step_idx < len(steps) and "role" in steps[step_idx]:
                fstep["role"] = steps[step_idx]["role"]
            else:
                fstep["role"] = "unknown"
            step_idx += 1

        output.extra_fields["flattened_steps"] = flattened_steps
        output.extra_fields["dual_model_enabled"] = True
        return output
