
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.model import compute_position_id_with_mask
from recipe.fully_async_policy.detach_utils import RolloutSample
import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
# Import functions from the base detach_utils to avoid duplication if possible,
# or just reimplement what's needed. Given we are making a separate file,
# we will copy the necessary parts to ensure independence or import if we want to reuse.
# Here we will implement the requested functions.

def prepare_webagent_generation_data(batch_dict, config, epoch, global_steps, validate) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    # batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    # non_tensor_batch_keys_to_pop=['raw_prompt_ids', 'multi_modal_data', 
    # 'multi_modal_inputs','config_file', 
    # 'index', 'raw_prompt', 
    # 'data_source', 'reward_model']

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

    full_batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )

    if full_batch.batch.is_empty():
        full_batch.batch["_dummy"] = torch.zeros(len(full_batch), 1)

    full_batch.non_tensor_batch["agent_name"] = np.array(
        ["async_web_agent"] * len(full_batch), dtype=object
    )
    full_batch.non_tensor_batch["epoch"] = np.array(
        [epoch] * len(full_batch), dtype=object
    )
    full_batch.non_tensor_batch["global_steps"] = np.array(
        [global_steps] * len(full_batch), dtype=object
    )
    full_batch.non_tensor_batch["validate"] = np.array(
        [validate] * len(full_batch), dtype=object
    )
    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)

    return full_batch

def postprocess_agent_loop_outputs_flattened_steps(rs: "RolloutSample", tokenizer, config, processor) -> DataProto:
    """
    Static method to postprocess a list of AgentLoopOutput into DataProto using flattened steps.
    This transforms trajectory-level outputs into step-level samples for PPO training.

    Args:
        rs: RolloutSample
        tokenizer: Tokenizer instance
        config: Configuration object

    Returns:
        DataProto: Processed batch data (flattened steps)
    """
    logger.info(f"[postprocess_agent_loop_outputs_flattened_steps] start")
    inputs: list[AgentLoopOutput] = rs.agent_loop_output_list
    full_batch = rs.full_batch
    
    all_steps = []
    
    for i, input_output in enumerate(inputs):
        # flattened_steps is expected to be in extra_fields
        # It was added by AsyncWebAgentLoop._build_completed_output
        flattened_steps = input_output.extra_fields.get("flattened_steps", [])
        if not flattened_steps:
            continue
            
        # Turns is the length of flattened_steps for this trajectory
        current_traj_num_turns = len(flattened_steps)

        for step_ind, step in enumerate(flattened_steps):
            # step is a dict with keys: prompts, responses, token_level_scores, etc.
            
            # Create a step dict suitable for stacking
            step_data = {}
            step_data['__num_turns__'] = current_traj_num_turns # Save turns info
            step_data['prompts'] = step['prompts']
            step_data['responses'] = step['responses']
            step_data['token_level_scores'] = step['token_level_scores']
            step_data['reward_score'] = step['reward_score']
            step_data['format_score'] = step['format_score']
            step_data['answer_score'] = step['answer_score']
            if 'log_probs' in step and step['log_probs'] is not None:
                 # log_probs might be list or tensor
                 if isinstance(step['log_probs'], list):
                     step_data['response_logprobs'] = torch.tensor(step['log_probs'])
                 else:
                     step_data['response_logprobs'] = step['log_probs']
            
            # Metadata to map back to original trajectory if needed
            step_data['original_traj_index'] = i
            
            # Dual-model role tag: "planner", "grounder", or "unknown"
            step_data['role'] = step.get('role', 'unknown')
            
            # Add multi-modal info if available in the step
            if 'multi_modal_data' in step:
                step_data['multi_modal_data'] = step['multi_modal_data']
            if 'multi_modal_inputs' in step:
                step_data['multi_modal_inputs'] = step['multi_modal_inputs']
            
            #解码prompt_ids和response_ids
            # prompt_ids_str = tokenizer.decode(step_data['prompts'], skip_special_tokens=True)
            # response_ids_str = tokenizer.decode(step_data['responses'], skip_special_tokens=True)
            
            all_steps.append(step_data)
    
    if not all_steps:
        # Return empty DataProto if no steps
        return DataProto()

    # Extract lists for padding
    prompts_list = [s['prompts'] for s in all_steps]
    responses_list = [s['responses'] for s in all_steps]
    
    # Padding using tokenizer
    tokenizer.padding_side = "left"
    outputs_p = tokenizer.pad(
        [{"input_ids": p} for p in prompts_list],
        padding="longest", # Pad to longest in batch
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_ids, prompt_attention_mask = outputs_p["input_ids"], outputs_p["attention_mask"]
    
    tokenizer.padding_side = "right"
    outputs_r = tokenizer.pad(
        [{"input_ids": r} for r in responses_list],
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    response_ids, response_attention_mask = outputs_r["input_ids"], outputs_r["attention_mask"]
    
    # Construct input_ids and attention_mask
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
    
    # Handle position_ids for Qwen2-VL
    if (
        processor is not None
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    ):
        

        # We need to compute position_ids sample by sample because they depend on image grid sizes
        position_ids_list = []
        
        # Reconstruct multi_modal_inputs list if not already in a convenient format
        multi_modal_inputs_list = [step.get('multi_modal_inputs', {}) for step in all_steps]

        # Check images source
        images_list = []
        for step in all_steps:
            mm_data = step.get('multi_modal_data')
            if mm_data and isinstance(mm_data, dict):
                images_list.append(mm_data.get('image', None))
            else:
                images_list.append(None)

        for i in range(len(input_ids)):
            current_input_ids = input_ids[i]
            current_attention_mask = attention_mask[i]
            
            # Try to get pre-computed grid info from multi_modal_inputs
            mmi = multi_modal_inputs_list[i]
            if isinstance(mmi, np.ndarray): mmi = mmi.item()
            
            image_grid_thw = mmi.get("image_grid_thw")
            video_grid_thw = mmi.get("video_grid_thw")
            second_per_grid_ts = mmi.get("second_per_grid_ts")
            
            # If grid info is missing but we have images and processor, we might need to re-process?
            # However, usually if multi_modal_inputs is present, grid info is there.
            # If it's missing, maybe it's a text-only sample or something went wrong.
            # Fallback: if image_grid_thw is None but we have images, try to re-process (expensive)
            if image_grid_thw is None and images_list[i] is not None:
                 # This path is expensive and should be avoided if possible
                 # But for robustness:
                 current_text = tokenizer.decode(current_input_ids, skip_special_tokens=True)
                 temp_inputs = processor(text=[current_text], images=images_list[i], return_tensors="pt")
                 image_grid_thw = temp_inputs.get("image_grid_thw")
                 video_grid_thw = temp_inputs.get("video_grid_thw")
                 second_per_grid_ts = temp_inputs.get("second_per_grid_ts")

            if "Qwen3VLProcessor" in processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                processor,
                input_ids=current_input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=current_attention_mask,
            ).unsqueeze(0)  # (1, 3, seq_len)

            # Combine with text position ids logic (if needed, but get_rope_index usually handles everything for Qwen2-VL)
            # The reference code used a specific combination logic:
            valid_mask = current_attention_mask.bool()
            text_position_ids = torch.ones((1, len(current_input_ids)), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            text_position_ids = text_position_ids.unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1) 
            
            position_ids_list.append(position_ids)

        if position_ids_list:
            position_ids = torch.cat(position_ids_list, dim=0) # (batch_size, 3, seq_len)
        else:
            # Fallback if list is empty (e.g. empty batch), though should be handled by earlier check
            # Create empty tensor with correct dimensions to satisfy torch.cat downstream if any
            position_ids = torch.empty((0, 3, 0), dtype=torch.long, device=input_ids.device)
    else:
        position_ids = compute_position_id_with_mask(attention_mask)
    
    # Response mask: 1 for valid response tokens, 0 for padding
    response_mask = response_attention_mask 
    
    # Token level scores padding
    scores_list = [s['token_level_scores'] for s in all_steps]
    max_response_len = response_ids.size(1)
    padded_scores = []
    for score in scores_list:
        pad_len = max_response_len - score.size(0)
        if pad_len > 0:
            padded_scores.append(torch.cat([score, torch.zeros(pad_len, dtype=score.dtype)]))
        else:
            padded_scores.append(score)
    token_level_scores = torch.stack(padded_scores)
    
    # Log probs padding
    rollout_log_probs = None
    if 'response_logprobs' in all_steps[0]:
        logprobs_list = [s.get('response_logprobs') for s in all_steps]
        # Filter None if any
        if all(lp is not None for lp in logprobs_list):
            padded_logprobs = []
            for lp in logprobs_list:
                pad_len = max_response_len - lp.size(0)
                if pad_len > 0:
                    padded_logprobs.append(torch.cat([lp, torch.zeros(pad_len, dtype=lp.dtype)]))
                else:
                    padded_logprobs.append(lp)
            rollout_log_probs = torch.stack(padded_logprobs)

    batch_dict = {
        "prompts": prompt_ids,
        "responses": response_ids,
        "prompt_mask": prompt_attention_mask, # Save prompt mask for correct padding later
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_level_scores": token_level_scores
    }
    
    if rollout_log_probs is not None:
        batch_dict["rollout_log_probs"] = rollout_log_probs

    non_tensor_batch = {}
    
    # Propagate trajectory-level non-tensors (broadcast)
    traj_indices = [s['original_traj_index'] for s in all_steps]
    
    for key, value in rs.full_batch.non_tensor_batch.items():
        if key in ['multi_modal_data', 'multi_modal_inputs']: continue # Handle separately
        if isinstance(value, (list, np.ndarray)) and len(value) == len(inputs):
            # Broadcast
            non_tensor_batch[key] = np.array([value[idx] for idx in traj_indices], dtype=object)
    
    # Handle multi-modal non-tensors and scores
    # Unconditionally add these keys to ensure consistency across batches for DataProto.concat
    
    non_tensor_batch['multi_modal_data'] = np.array([s.get('multi_modal_data') for s in all_steps], dtype=object)
    non_tensor_batch['multi_modal_inputs'] = np.array([s.get('multi_modal_inputs') for s in all_steps], dtype=object)

    non_tensor_batch['final_reward'] = np.array([s.get('final_reward') for s in all_steps], dtype=object)
    non_tensor_batch['format_score'] = np.array([s.get('format_score') for s in all_steps], dtype=object)
    non_tensor_batch['answer_score'] = np.array([s.get('answer_score') for s in all_steps], dtype=object)
    # Add turns info
    non_tensor_batch['__num_turns__'] = np.array([s.get('__num_turns__', 0) for s in all_steps], dtype=np.int32)
    # Add role info for dual-model training (planner/grounder/unknown)
    non_tensor_batch['role'] = np.array([s.get('role', 'unknown') for s in all_steps], dtype=object)

    # Metrics aggregation (broadcast)
    metrics = [inputs[idx].metrics.model_dump() for idx in traj_indices]
    
    return DataProto(batch=TensorDict(batch_dict, batch_size=len(all_steps)), 
                     non_tensor_batch=non_tensor_batch, 
                     meta_info={"metrics": metrics, "is_stepwise": True})


def merge_rollout_sample_flattened_steps(config, tokenizer, rs: "RolloutSample", processor):
    """
    Supplement and refine the RolloutSample object using flattened steps.
    """
    # Step 1: Create a DataProto from the AgentLoopOutput using flattened steps logic
    gen_batch_output = postprocess_agent_loop_outputs_flattened_steps(rs, tokenizer, config, processor)

    # Check for valid answer_scores (no negatives)
    answer_scores = gen_batch_output.non_tensor_batch.get("answer_score")
    is_valid = True
    if answer_scores is not None:
        for score in answer_scores:
            try:
                if score is not None and float(score) < 0:
                    is_valid = False
                    break
            except (ValueError, TypeError):
                pass
    
    if not is_valid:
        return None

    # Step 2: Add uid if not present (though postprocess propagates it)
    # If the original batch had uid, it's already propagated. 
    # If not, we might need to generate new ones, but usually it has.

    gen_batch_output.non_tensor_batch["uid"] = np.array([f"uid_{rs.sample_id}" for i in range(len(gen_batch_output))], dtype=object)

    # Merge meta_info
    gen_batch_output.meta_info.update(rs.full_batch.meta_info)

    # Step 3, set full_batch
    rs.full_batch = gen_batch_output
    
    # Update processing metadata (aggregating or keeping lists)
    # Since we flattened, these lists might need to match the new batch size or keep original.
    # The original logic in detach_utils kept lists of length equal to num_rollouts.
    # Here we might want to keep them as is for statistics, but they won't align 1-to-1 with the new batch.
    rs.processing_times = []
    rs.tool_calls = []
    for agent_loop in rs.agent_loop_output_list:
        rs.processing_times.append(agent_loop.metrics.generate_sequences)
        rs.tool_calls.append(agent_loop.metrics.tool_calls)
        
    rs.param_version_start = [
        agent_loop.extra_fields.get("param_version_start", 0) for agent_loop in rs.agent_loop_output_list
    ]
    rs.param_version_end = [
        agent_loop.extra_fields.get("param_version_end", 0) for agent_loop in rs.agent_loop_output_list
    ]
    
    # Step 4, clear agent_loop_output_list to save memory
    rs.agent_loop_output_list = []
    
    return rs

def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    # Filter out samples with negative answer_score
    valid_rollout_samples = []
    for rs in rollout_samples:
        answer_scores = rs.full_batch.non_tensor_batch.get("answer_score")
        if answer_scores is None:
            valid_rollout_samples.append(rs)
            continue

        # answer_scores is np.array of objects or numbers
        is_valid = True
        for score in answer_scores:
            try:
                if score is not None and float(score) < 0:
                    is_valid = False
                    break
            except (ValueError, TypeError):
                pass # Ignore non-numeric scores
        
        if is_valid:
            valid_rollout_samples.append(rs)
            
    if len(rollout_samples) != len(valid_rollout_samples):
        print(f"[BatchUtils] Filtered out {len(rollout_samples) - len(valid_rollout_samples)} samples with negative answer_score.")
    
    rollout_samples = valid_rollout_samples
    if not rollout_samples:
        # raise ValueError("All rollout samples were filtered out due to negative answer_score")
        # Return empty DataProto if no samples left
        # But assemble_batch generally expects to return something useful.
        # If we raise error, it might crash the loop. Better to maybe return empty or let downstream handle it.
        # However, return type is DataProto.
        # Let's raise error for now as it seems safer than returning broken batch.
         raise ValueError("All rollout samples were filtered out due to negative answer_score")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    processing_times = []
    tool_calls = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        rollout_samples_batch.append(rs.full_batch)
        processing_times.extend(rs.processing_times)
        tool_calls.extend(rs.tool_calls)
    
    # --- PADDING LOGIC START ---
    
    # Find global max lengths for each sequence type
    max_prompt_len = 0
    max_response_len = 0
    max_position_ids_len = 0
    
    for batch_proto in rollout_samples_batch:
        if "prompts" in batch_proto.batch.keys():
            max_prompt_len = max(max_prompt_len, batch_proto.batch["prompts"].shape[-1])
        if "responses" in batch_proto.batch.keys():
            max_response_len = max(max_response_len, batch_proto.batch["responses"].shape[-1])
        if "position_ids" in batch_proto.batch.keys():
            max_position_ids_len = max(max_position_ids_len, batch_proto.batch["position_ids"].shape[-1])
            
    # Pad each batch
    for batch_proto in rollout_samples_batch:
        bsz = batch_proto.batch.batch_size[0]
        # Handle device (might be None if batch is on cpu but device attr missing)
        device = torch.device("cpu")
        if "prompts" in batch_proto.batch.keys():
            device = batch_proto.batch["prompts"].device
        elif "input_ids" in batch_proto.batch.keys():
            device = batch_proto.batch["input_ids"].device

        prompt_pad_len = 0
        response_pad_len = 0

        # Pad Prompts and Prompt Mask (Left Padding)
        if "prompts" in batch_proto.batch.keys():
            curr_len = batch_proto.batch["prompts"].shape[-1]
            pad_len = max_prompt_len - curr_len
            prompt_pad_len = pad_len
            if pad_len > 0:
                batch_proto.batch["prompts"] = torch.cat(
                    [torch.full((bsz, pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device), batch_proto.batch["prompts"]],
                    dim=-1
                )
                if "prompt_mask" in batch_proto.batch.keys():
                    batch_proto.batch["prompt_mask"] = torch.cat(
                        [torch.zeros((bsz, pad_len), dtype=torch.long, device=device), batch_proto.batch["prompt_mask"]],
                        dim=-1
                    )
        
        # Pad Responses and related (Right Padding)
        if "responses" in batch_proto.batch.keys():
            # Check for consistency
            resp_shape = batch_proto.batch["responses"].shape
            if "response_mask" in batch_proto.batch.keys():
                rm_shape = batch_proto.batch["response_mask"].shape
                assert resp_shape == rm_shape, f"response_mask shape {rm_shape} mismatch responses shape {resp_shape}"
            if "token_level_scores" in batch_proto.batch.keys():
                tls_shape = batch_proto.batch["token_level_scores"].shape
                assert resp_shape == tls_shape, f"token_level_scores shape {tls_shape} mismatch responses shape {resp_shape}"
                
                # Check last valid token score
                rm = batch_proto.batch["response_mask"]
                tls = batch_proto.batch["token_level_scores"]
                lengths = rm.sum(dim=-1).long()

                # for i, length in enumerate(lengths):
                #     if length > 0:
                #         last_val = tls[i, length - 1].item()
                #         # if last_val == 0:
                #         #      answer_score_i = "N/A"
                #         #      if "answer_score" in batch_proto.non_tensor_batch:
                #         #          
                #         answer_score_i = batch_proto.non_tensor_batch["answer_score"][i]

            curr_len = batch_proto.batch["responses"].shape[-1]
            pad_len = max_response_len - curr_len
            response_pad_len = pad_len
            if pad_len > 0:
                # Pad responses
                batch_proto.batch["responses"] = torch.cat(
                    [batch_proto.batch["responses"], torch.full((bsz, pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device)],
                    dim=-1
                )
                
                # Pad response_mask
                if "response_mask" in batch_proto.batch.keys():
                    batch_proto.batch["response_mask"] = torch.cat(
                        [batch_proto.batch["response_mask"], torch.zeros((bsz, pad_len), dtype=torch.long, device=device)],
                        dim=-1
                    )
                
                # Pad token_level_scores
                if "token_level_scores" in batch_proto.batch.keys():
                    scores = batch_proto.batch["token_level_scores"]
                    batch_proto.batch["token_level_scores"] = torch.cat(
                        [scores, torch.zeros((bsz, pad_len), dtype=scores.dtype, device=device)],
                        dim=-1
                    )
                    
                # Pad rollout_log_probs
                if "rollout_log_probs" in batch_proto.batch.keys():
                    lp = batch_proto.batch["rollout_log_probs"]
                    batch_proto.batch["rollout_log_probs"] = torch.cat(
                        [lp, torch.zeros((bsz, pad_len), dtype=lp.dtype, device=device)],
                        dim=-1
                    )

        # 3. Re-construct input_ids and attention_mask
        if "prompts" in batch_proto.batch.keys() and "responses" in batch_proto.batch.keys():
            p = batch_proto.batch["prompts"]
            r = batch_proto.batch["responses"]
            
            # input_ids
            batch_proto.batch["input_ids"] = torch.cat([p, r], dim=-1)
            
            # Reconstruct attention_mask from prompt_mask and response_mask if available
            # This preserves original tokenizer mask logic instead of simple pad_token check
            if "prompt_mask" in batch_proto.batch.keys() and "response_mask" in batch_proto.batch.keys():
                p_mask = batch_proto.batch["prompt_mask"]
                r_mask = batch_proto.batch["response_mask"]
                batch_proto.batch["attention_mask"] = torch.cat([p_mask, r_mask], dim=-1)
            else:
                # Fallback to simple re-calculation
                p_mask = (p != tokenizer.pad_token_id).long()
                r_mask = (r != tokenizer.pad_token_id).long()
                batch_proto.batch["attention_mask"] = torch.cat([p_mask, r_mask], dim=-1)
            
            # 4. Re-calculate/Pad position_ids
            if "position_ids" in batch_proto.batch.keys():
                pos_ids = batch_proto.batch["position_ids"]
                
                # Split into prompt and response parts
                orig_prompt_len = max_prompt_len - prompt_pad_len
                pos_ids_prompt = pos_ids[..., :orig_prompt_len]
                pos_ids_response = pos_ids[..., orig_prompt_len:]
                
                # Pad Prompt (Left)
                if prompt_pad_len > 0:
                    if pos_ids.dim() == 3: # (bsz, 4, seq_len)
                        pad = torch.zeros((bsz, pos_ids.size(1), prompt_pad_len), dtype=pos_ids.dtype, device=device)
                    else: # (bsz, seq_len)
                        pad = torch.zeros((bsz, prompt_pad_len), dtype=pos_ids.dtype, device=device)
                    pos_ids_prompt = torch.cat([pad, pos_ids_prompt], dim=-1)
                
                # Pad Response (Right)
                if response_pad_len > 0:
                    if pos_ids.dim() == 3: # (bsz, 4, seq_len)
                        pad = torch.zeros((bsz, pos_ids.size(1), response_pad_len), dtype=pos_ids.dtype, device=device)
                    else: # (bsz, seq_len)
                        pad = torch.zeros((bsz, response_pad_len), dtype=pos_ids.dtype, device=device)
                    pos_ids_response = torch.cat([pos_ids_response, pad], dim=-1)
                
                batch_proto.batch["position_ids"] = torch.cat([pos_ids_prompt, pos_ids_response], dim=-1)

    # --- PADDING LOGIC END ---


    final_batch = DataProto.concat(rollout_samples_batch)

    # --- Align batch size to DP size (chunk size) ---
    # The batch size must be divisible by dp_size for distributed training (compute_log_prob)
    try:
        # dp_size calculation:
        # We should NOT use fsdp_size (which might be 8 in HSDP) because the system expects chunks equal to the number of workers (16).
        # The error "chunk 16" confirms dp_size should be 16.
        world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
        
        actor_config = config.actor_rollout_ref.actor
        actor_tp_size = 1
        if hasattr(actor_config, "get"):
             actor_tp_size = actor_config.get("tensor_model_parallel_size", 1)
        elif hasattr(actor_config, "tensor_model_parallel_size"):
             actor_tp_size = actor_config.tensor_model_parallel_size

        dp_size = world_size // actor_tp_size
        # print(f"[BatchUtils] Calculated dp_size: {dp_size} (world_size={world_size}, actor_tp_size={actor_tp_size})")
        
        # Ensure dp_size is at least 1 and valid
        if dp_size > 1:
            current_len = len(final_batch)
            if current_len > 0:
                remainder = current_len % dp_size
                if remainder > 0:
                    new_len = current_len - remainder
                    if new_len > 0:
                        final_batch = final_batch[:new_len]
                        print(f"[BatchUtils] Truncated batch from {current_len} to {new_len} to align with dp_size {dp_size}")
                    else:
                        # Rare case: batch smaller than dp_size? 
                        pass 
    except Exception as e:
        print(f"[BatchUtils] Warning: Failed to align batch size: {e}")
    # ----------------------------------------------

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    # Collect statistics
    param_versions = [rs.param_version for rs in rollout_samples]
    trajectorys_param_versions = [version for rs in rollout_samples for version in rs.param_version_end]

    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if tool_calls:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }

    # Turns Statistics
    turns_stats = {}
    all_turns = []
    for rs in rollout_samples:
        if '__num_turns__' in rs.full_batch.non_tensor_batch:
            # Get turns from the first step of the sample (since it's broadcasted/same for all steps in a traj)
            # rs.full_batch.non_tensor_batch['__num_turns__'] is an array
            turns_arr = rs.full_batch.non_tensor_batch['__num_turns__']
            if len(turns_arr) > 0:
                all_turns.append(turns_arr[0])
    
    if all_turns:
        turns_stats = {
            "fully_async/web_agent_turns/max": np.max(all_turns),
            "fully_async/web_agent_turns/min": np.min(all_turns),
            "fully_async/web_agent_turns/mean": np.mean(all_turns),
        }

    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}
    
    param_version_start = [v for rs in rollout_samples for v in rs.param_version_start]
    param_version_end = [v for rs in rollout_samples for v in rs.param_version_end]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    final_batch.meta_info.update(
        {
            "rollout_param_versions": param_versions,
            "param_version_diversity": len(set(param_versions)) if param_versions else 0,
            "trajectory_param_versions": trajectorys_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
            **turns_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch

# Re-export other necessary classes/functions if they are needed by importers of this file
# Since we are in a recipe, we assume the base classes are available.
# If RolloutSample needs to be imported here, we assume it's available or define it.
# But RolloutSample is defined in recipe.fully_async_policy.detach_utils.
# We are just defining new functions. The type hint "RolloutSample" is forward ref.