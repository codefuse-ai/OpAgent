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
Shared validation implementation for WebAgent-based rollouters.

This module extracts the validation logic from WebAgentFullyAsyncRollouter into
standalone functions so that DualModelWebAgentFullyAsyncRollouter can reuse
them without needing to inherit from the @ray.remote decorated class.

All functions take `rollouter` (the rollouter instance) as first argument,
which must provide: config, tokenizer, processor, val_dataloader, val_reward_fn,
global_steps, async_rollout_mode, actor_rollout_wg, async_rollout_manager,
_get_gen_batch,
_maybe_log_val_generations, _dump_generations.
"""

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urlparse

import numpy as np
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics

logger = logging.getLogger(__name__)

TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR", "")
TRAJECTORY_DATA = TENSORBOARD_DIR.replace("tensorboard", "trajectory_data")


def _normalize_url(url: str) -> str:
    """Normalize URL by removing IP address to handle IP differences."""
    try:
        parsed = urlparse(url)
        if ':' in parsed.netloc:
            port = parsed.netloc.split(':')[-1]
            return f"{parsed.scheme}://PORT_{port}"
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return url


def _dropip_url(url: str) -> str:
    """Normalize URL, ignoring IP/domain differences, keeping only port and path.
    Supports URLs joined with |AND| separator."""
    if not url:
        return url

    if ' |AND| ' in url or '|AND|' in url:
        urls = [u.strip() for u in url.replace(' |AND| ', '|AND|').split('|AND|')]
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
        return ' |AND| '.join(sorted(normalized_urls))
    else:
        try:
            parsed = urlparse(url)
            port = f":{parsed.port}" if parsed.port else ""
            path = parsed.path or "/"
            query = f"?{parsed.query}" if parsed.query else ""
            fragment = f"#{parsed.fragment}" if parsed.fragment else ""
            return f"{port}{path}{query}{fragment}"
        except Exception:
            return url


def _get_sample_key(intent: str, start_url: str) -> str:
    """Create a unique key for a sample based on intent and normalized start_url."""
    normalized_url = _dropip_url(start_url)
    return f"{intent}|||{normalized_url}"


def _load_validated_samples(global_steps: int) -> Dict[str, int]:
    """Load already validated samples from previous validation runs for the current step."""
    validated_samples = defaultdict(int)

    trajectory_data_dir = Path(TRAJECTORY_DATA)
    if not trajectory_data_dir.exists():
        return validated_samples

    step_str = f"val_step_{global_steps:06d}_"
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
                sample_key = _get_sample_key(intent, start_url)
                validated_samples[sample_key] += 1
        except Exception:
            continue

    if validated_samples:
        logger.info(f"[Validation Resume] Loaded {len(validated_samples)} unique validated samples "
                     f"with total {sum(validated_samples.values())} validations for step {global_steps}")

    return validated_samples


def _reorder_indices_by_url_uniformly(dataset, indices: List[int]) -> List[int]:
    """Reorder indices to distribute samples from the same URL uniformly across batches."""
    indices_by_domain = defaultdict(list)

    for idx in indices:
        try:
            sample = dataset[idx]
            domain_url = 'unknown'
            if isinstance(sample, dict) and 'extra_info' in sample:
                extra_info = sample['extra_info']
                if isinstance(extra_info, dict) and 'config' in extra_info:
                    config_str = extra_info['config']
                    if isinstance(config_str, str):
                        config_dict = json.loads(config_str)
                    else:
                        config_dict = config_str
                    start_url = config_dict.get('start_url', '')
                    if start_url:
                        domain_url = _normalize_url(start_url)
                elif isinstance(extra_info, dict) and 'start_url' in extra_info:
                    start_url = extra_info['start_url']
                    if start_url:
                        domain_url = _normalize_url(start_url)
            elif hasattr(sample, 'extra_info') and sample.extra_info:
                extra_info = sample.extra_info
                if isinstance(extra_info, str):
                    config_dict = json.loads(extra_info)
                    start_url = config_dict.get('start_url', '')
                    if start_url:
                        domain_url = _normalize_url(start_url)
        except Exception as e:
            logger.debug(f"Failed to extract URL from sample {idx}: {e}")
            domain_url = 'unknown'

        indices_by_domain[domain_url].append(idx)

    if len(indices_by_domain) <= 1:
        logger.info(f"[URL Distribution] Only {len(indices_by_domain)} domain(s) found, skipping reordering")
        return indices

    grouped_indices = list(indices_by_domain.values())
    interleaved_tuples = zip_longest(*grouped_indices)
    uniform_indices = [idx for tpl in interleaved_tuples for idx in tpl if idx is not None]

    logger.info(f"[URL Distribution] Reordered {len(uniform_indices)} samples across {len(grouped_indices)} domains")
    domain_dist = [(domain, len(idxs)) for domain, idxs in list(indices_by_domain.items())[:10]]
    logger.info(f"[URL Distribution] Top domain distribution: {domain_dist}")

    return uniform_indices


def _create_filtered_val_dataloader(rollouter, required_validations: int):
    """Create a filtered validation dataloader that only includes samples needing validation."""
    validated_samples = _load_validated_samples(rollouter.global_steps)

    dataset = rollouter.val_dataloader.dataset

    filtered_indices = []
    skipped_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        extra_info = sample.get('extra_info')
        if extra_info is not None:
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
                    sample_key = _get_sample_key(intent, start_url)
                    current_validations = validated_samples.get(sample_key, 0)
                    if current_validations >= required_validations:
                        skipped_count += 1
                        continue
                    else:
                        filtered_indices.append(idx)
                else:
                    filtered_indices.append(idx)
            except Exception:
                filtered_indices.append(idx)
        else:
            filtered_indices.append(idx)

    total_samples = len(dataset)
    samples_to_validate = len(filtered_indices)
    logger.info(f"[Validation Resume] Filtering: total={total_samples}, completed={skipped_count}, "
                f"need={samples_to_validate}, skip_rate={skipped_count / total_samples * 100:.1f}%")

    if samples_to_validate == 0:
        logger.info("[Validation Resume] All samples already validated, skipping validation")
        return iter([])

    from torch.utils.data import Subset
    from torchdata.stateful_dataloader import StatefulDataLoader

    uniform_indices = _reorder_indices_by_url_uniformly(dataset, filtered_indices)
    filtered_dataset = Subset(dataset, uniform_indices)

    filtered_dataloader = StatefulDataLoader(
        dataset=filtered_dataset,
        batch_size=rollouter.val_dataloader.batch_size,
        num_workers=getattr(rollouter.val_dataloader, 'num_workers', 0),
        shuffle=False,
        drop_last=False,
        collate_fn=getattr(rollouter.val_dataloader, 'collate_fn', None),
    )

    logger.info(f"[Validation Resume] Created filtered StatefulDataLoader "
                f"with batch_size={filtered_dataloader.batch_size}")
    return filtered_dataloader


async def _process_single_validation_batch(
    rollouter,
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
    update_lock,
):
    """Process a single validation batch asynchronously."""
    test_batch = DataProto.from_single_dict(test_data)

    if "uid" not in test_batch.non_tensor_batch:
        test_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
        )

    original_batch_size = len(test_batch.batch)

    test_batch = test_batch.repeat(
        repeat_times=rollouter.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
    )

    if rollouter.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
        return

    input_ids = test_batch.batch["input_ids"]
    input_texts = [rollouter.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

    ground_truths = [
        item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
    ]

    # NOTE: Don't add sample_inputs here - only add them after successful processing
    # to avoid length mismatch if processing fails
    uids_to_add = test_batch.non_tensor_batch["uid"]
    gts_to_add = ground_truths

    test_gen_batch = rollouter._get_gen_batch(test_batch, epoch=0, global_steps=rollouter.global_steps, validate=True)

    test_gen_batch.meta_info = {
        "eos_token_id": rollouter.tokenizer.eos_token_id,
        "pad_token_id": rollouter.tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": rollouter.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
        "validate": True,
        "global_steps": rollouter.global_steps,
    }

    size_divisor = (
        rollouter.actor_rollout_wg.world_size
        if not rollouter.async_rollout_mode
        else rollouter.config.actor_rollout_ref.rollout.agent.num_workers
    )
    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

    if not rollouter.async_rollout_mode:
        test_output_gen_batch_padded = rollouter.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    else:
        test_output_gen_batch_padded = await rollouter.async_rollout_manager.generate_sequences_async(
            test_gen_batch_padded
        )

    test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

    output_ids = test_output_gen_batch.batch["responses"]
    output_texts = [rollouter.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

    test_batch = test_batch.union(test_output_gen_batch)
    test_batch.meta_info["validate"] = True

    if rollouter.val_reward_fn is None:
        raise ValueError("val_reward_fn must be provided for validation.")
    result = rollouter.val_reward_fn(test_batch, return_dict=True)
    reward_tensor = result["reward_tensor"]
    scores = reward_tensor.sum(-1).cpu().tolist()

    data_source = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
    num_turns = test_batch.non_tensor_batch.get("__num_turns__", None)

    # Only add to sample lists after successful processing
    async with update_lock:
        sample_inputs.extend(input_texts)
        sample_uids.extend(uids_to_add)
        sample_gts.extend(gts_to_add)
        sample_outputs.extend(output_texts)
        sample_scores.extend(scores)
        reward_extra_infos_dict["reward"].extend(scores)
        if "reward_extra_info" in result:
            for key, lst in result["reward_extra_info"].items():
                reward_extra_infos_dict[key].extend(lst)
        if num_turns is not None:
            sample_turns.append(num_turns)
        data_source_lst.append(data_source)
        progress_bar.update(original_batch_size)


async def webagent_validate(rollouter):
    """
    Full WebAgent validation pipeline.

    This is a standalone function equivalent to WebAgentFullyAsyncRollouter._validate().
    It takes the rollouter instance and uses its attributes to perform validation.

    Args:
        rollouter: The rollouter instance (WebAgent or DualModel), must have:
            config, tokenizer, processor, val_dataloader, val_reward_fn,
            global_steps, async_rollout_mode, actor_rollout_wg, async_rollout_manager,
            _get_gen_batch,
            _maybe_log_val_generations, _dump_generations
    """
    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    sample_inputs = []
    sample_outputs = []
    sample_gts = []
    sample_scores = []
    sample_turns = []
    sample_uids = []

    required_validations = rollouter.config.actor_rollout_ref.rollout.val_kwargs.n
    filtered_dataloader = _create_filtered_val_dataloader(rollouter, required_validations)

    try:
        if hasattr(filtered_dataloader, 'dataset'):
            total_val_samples = len(filtered_dataloader.dataset)
        elif hasattr(filtered_dataloader, '__len__'):
            total_val_samples = len(filtered_dataloader)
        else:
            total_val_samples = None
    except Exception:
        total_val_samples = None

    if total_val_samples is None or total_val_samples == 0:
        logger.info("[Validation Resume] No validation samples found, skipping validation")
        return {}

    progress_bar = tqdm(total=total_val_samples, desc="Validation Progress", unit="sample")

    max_concurrent_batches = rollouter.config.data.get("val_max_concurrent_batches", None)
    if max_concurrent_batches is None:
        val_batch_size = rollouter.config.data.get("val_batch_size", 1)
        num_workers = rollouter.config.actor_rollout_ref.rollout.agent.get("num_workers", 4)
        if val_batch_size <= 1:
            max_concurrent_batches = min(num_workers // 2, 4)
        elif val_batch_size <= 2:
            max_concurrent_batches = min(num_workers // 3, 3)
        else:
            max_concurrent_batches = 2
        max_concurrent_batches = max(1, max_concurrent_batches)
        logger.info(f"[Validation] Auto-set val_max_concurrent_batches={max_concurrent_batches}")

    semaphore = asyncio.Semaphore(max_concurrent_batches)
    update_lock = asyncio.Lock()

    async def process_batch(test_data):
        async with semaphore:
            return await _process_single_validation_batch(
                rollouter,
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
                update_lock,
            )

    pending_tasks = set()
    dataloader_iter = iter(filtered_dataloader)
    has_more_data = True

    for _ in range(max_concurrent_batches):
        try:
            test_data = next(dataloader_iter)
            task = asyncio.create_task(process_batch(test_data))
            pending_tasks.add(task)
        except StopIteration:
            has_more_data = False
            break

    while pending_tasks:
        done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                await task
            except Exception as e:
                logger.error(f"[Validation] Batch processing failed: {e}", exc_info=True)
        if has_more_data:
            for _ in range(len(done)):
                try:
                    test_data = next(dataloader_iter)
                    task = asyncio.create_task(process_batch(test_data))
                    pending_tasks.add(task)
                except StopIteration:
                    has_more_data = False
                    break

    progress_bar.close()

    rollouter._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

    val_data_dir = rollouter.config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        rollouter._dump_generations(
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
