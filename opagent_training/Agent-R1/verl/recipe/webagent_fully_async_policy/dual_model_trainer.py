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
DualModelWebAgentFullyAsyncTrainer: Extends WebAgentFullyAsyncTrainer to support
two independent models (Planner + Grounder) with:
- Shared GRPO trajectory-level advantage (computed BEFORE role split)
- Independent parameter updates (each model has its own FSDP worker group + optimizer)
- Independent reference log_prob computation
- Joint reward computation (unchanged from original)
"""

import copy
import time
import torch
import ray
import numpy as np

from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainerBase
from recipe.webagent_fully_async_policy.detach_utils import assemble_batch_from_rollout_samples
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.utils.debug import marked_timer
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from omegaconf import OmegaConf

import os
import logging

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10)
class DualModelWebAgentFullyAsyncTrainer(FullyAsyncTrainerBase):
    """
    Dual-model trainer with independent Planner and Grounder parameter updates.
    
    Architecture:
    - self.actor_wg / self.actor_rollout_wg: Planner FSDP worker group (inherited)
    - self.grounder_actor_wg: Grounder FSDP worker group (new)
    - self.ref_policy_wg: Planner reference policy (inherited)
    - self.grounder_ref_policy_wg: Grounder reference policy (new)
    
    Training flow:
    1. Collect batch from message queue (same as parent)
    2. Compute joint reward (same as parent, reward computation unchanged)
    3. Compute GRPO advantage on the FULL batch (before role split)
    4. Split batch by role: planner_batch, grounder_batch
    5. Compute old_log_prob independently for each role
    6. Compute ref_log_prob independently for each role
    7. Update planner with planner_batch
    8. Update grounder with grounder_batch
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
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=device_name,
        )
        # Grounder-specific attributes (will be initialized in _init_models)
        self.grounder_actor_wg = None
        self.grounder_ref_policy_wg = None

    def _create_actor_rollout_classes(self):
        """Override to register both planner and grounder actor worker classes.
        
        Both are colocated on the same resource pool (same GPUs).
        With FSDP + CPU offload, they share GPU memory efficiently.
        """
        # Planner actor (same as parent)
        super()._create_actor_rollout_classes()

        # Grounder actor - same worker class but potentially different model config
        dual_model_config = self.config.get("dual_model", {})
        grounder_model_path = dual_model_config.get("grounder_model_path", None)

        if grounder_model_path:
            grounder_actor_config = copy.deepcopy(self.config.actor_rollout_ref)
            grounder_actor_config.model.path = grounder_model_path
        else:
            # Same model as planner (will diverge during training)
            grounder_actor_config = self.config.actor_rollout_ref

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        grounder_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=grounder_actor_config,
            role="actor",  # Must be a valid role string; "GrounderActor" is only the dict key
        )
        self.resource_pool_to_cls[resource_pool]["GrounderActor"] = grounder_cls
        print(f"[DualModelTrainer] Registered GrounderActor on trainer pool")

    def _create_reference_policy_class(self):
        """Override to also create grounder reference policy."""
        super()._create_reference_policy_class()

        if self.use_reference_policy:
            dual_model_config = self.config.get("dual_model", {})
            grounder_model_path = dual_model_config.get("grounder_model_path", None)

            if grounder_model_path:
                grounder_ref_config = copy.deepcopy(self.config.actor_rollout_ref)
                grounder_ref_config.model.path = grounder_model_path
            else:
                grounder_ref_config = self.config.actor_rollout_ref

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            grounder_ref_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=grounder_ref_config,
                role="ref",  # Must be a valid role string; "GrounderRefPolicy" is only the dict key
            )
            self.resource_pool_to_cls[resource_pool]["GrounderRefPolicy"] = grounder_ref_cls
            print(f"[DualModelTrainer] Registered GrounderRefPolicy on trainer pool")

    def _init_models(self):
        """Initialize both planner and grounder model worker groups."""
        # Initialize planner models (use parent's logic)
        super()._init_models()
        # self.actor_wg is now the planner
        # self.actor_rollout_wg is self.actor_wg (planner)

        # Initialize grounder actor
        grounder_actor_key = "GrounderActor"
        if grounder_actor_key in self.all_wg:
            self.grounder_actor_wg = self.all_wg[grounder_actor_key]
            self.grounder_actor_wg.init_model()
            print(f"[DualModelTrainer] Grounder actor initialized. Available keys: {list(self.all_wg.keys())}")
        else:
            logger.warning(
                f"[DualModelTrainer] '{grounder_actor_key}' not found in all_wg "
                f"(keys: {list(self.all_wg.keys())}). Falling back to single-model."
            )
            self.grounder_actor_wg = self.actor_wg

        # Initialize grounder reference policy
        grounder_ref_key = "GrounderRefPolicy"
        if self.use_reference_policy and not self.ref_in_actor:
            if grounder_ref_key in self.all_wg:
                self.grounder_ref_policy_wg = self.all_wg[grounder_ref_key]
                self.grounder_ref_policy_wg.init_model()
                print("[DualModelTrainer] Grounder reference policy initialized")
            else:
                self.grounder_ref_policy_wg = self.grounder_actor_wg
                print("[DualModelTrainer] Using grounder actor as grounder ref policy")

    def _get_dp_size(self):
        """Compute data-parallel size for the actor worker group.

        dp_size = total_training_gpus / ulysses_sequence_parallel_size.
        Used to pad sub-batches so that dispatch_nd_compute produces equal
        chunks across all DP ranks, preventing NCCL collective divergence.
        """
        n_training_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        sp_size = self.config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
        return max(1, n_training_gpus // sp_size)

    def get_actor_wg(self):
        """Get planner actor worker group (used by ParameterSynchronizer)."""
        return self.actor_wg

    def get_grounder_actor_wg(self):
        """Get grounder actor worker group (used by DualModelParameterSynchronizer)."""
        return self.grounder_actor_wg

    def _split_batch_by_role(self, batch: DataProto):
        """
        Split a batch into planner and grounder sub-batches based on the 'role' field
        in non_tensor_batch.
        
        Returns:
            planner_batch: DataProto or None
            grounder_batch: DataProto or None
            planner_indices: np.ndarray
            grounder_indices: np.ndarray
        """
        roles = batch.non_tensor_batch.get('role', None)
        if roles is None:
            # No role info, treat everything as planner (backward compatible)
            return batch, None, np.arange(len(batch)), np.array([])

        planner_mask = np.array([r == 'planner' for r in roles])
        grounder_mask = np.array([r == 'grounder' for r in roles])

        planner_indices = np.where(planner_mask)[0]
        grounder_indices = np.where(grounder_mask)[0]

        planner_batch = batch.select_idxs(planner_indices) if len(planner_indices) > 0 else None
        grounder_batch = batch.select_idxs(grounder_indices) if len(grounder_indices) > 0 else None

        # Enable auto-padding so sub-batches can be chunked across dp workers
        # even when the sub-batch size is not evenly divisible by dp_size.
        # Copy meta_info to avoid polluting the parent batch's meta_info.
        from verl.protocol import DataProtoConfig
        if planner_batch is not None:
            planner_batch.meta_info = dict(planner_batch.meta_info)
            planner_batch.meta_info[DataProtoConfig.auto_padding_key] = True
        if grounder_batch is not None:
            grounder_batch.meta_info = dict(grounder_batch.meta_info)
            grounder_batch.meta_info[DataProtoConfig.auto_padding_key] = True

        return planner_batch, grounder_batch, planner_indices, grounder_indices

    def _process_batch_common(self, batch, metrics, timing_raw, local_trigger_step=None):
        """
        Override the parent's _process_batch_common to implement dual-model training.
        
        Key difference: advantage is computed on the FULL batch first,
        then the batch is split by role for independent old_log_prob, ref_log_prob, and updates.
        """
        # ========== Step 1: Compute reward (JOINT, unchanged) ==========
        with marked_timer("reward", timing_raw, color="yellow"):
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if "token_level_scores" in batch.batch.keys():
                reward_tensor = batch.batch["token_level_scores"]
                reward_extra_infos_dict = {}
                for key_i in batch.non_tensor_batch.keys():
                    if "score" in key_i or "reward" in key_i:
                        reward_extra_infos_dict[key_i] = batch.non_tensor_batch[key_i].tolist()
            else:
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    batch.batch["token_level_scores"] = reward_tensor

        # ========== Step 2: Compute advantage on FULL batch (JOINT) ==========
        with marked_timer("adv", timing_raw, color="brown"):
            if "token_level_scores" in batch.batch.keys():
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            else:
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # Rollout correction (needs old_log_probs; use rollout_log_probs as default before per-role recomputation)
            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                if "old_log_probs" not in batch.batch:
                    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                metrics.update(is_metrics)

            # GRPO advantage on FULL batch (preserves group structure)
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

        # ========== Step 3: Split batch by role ==========
        with marked_timer("split_batch", timing_raw, color="cyan"):
            planner_batch, grounder_batch, planner_idx, grounder_idx = self._split_batch_by_role(batch)

        print(
            f"[DualModelTrainer] Batch split: "
            f"planner={len(planner_idx)}, grounder={len(grounder_idx)}, total={len(batch)}"
        )

        # Pad sub-batches to be divisible by dp_size so that
        # dispatch_nd_compute produces equal chunks across DP ranks.
        # Without this, unequal chunks cause different mini_batch counts
        # in update_policy, leading to NCCL all_reduce deadlock.
        dp_size = self._get_dp_size()
        if planner_batch is not None and len(planner_batch) % dp_size != 0:
            planner_batch, _ = pad_dataproto_to_divisor(planner_batch, dp_size)
            print(f"[DualModelTrainer] Padded planner_batch to {len(planner_batch)} (dp_size={dp_size})")
        if grounder_batch is not None and len(grounder_batch) % dp_size != 0:
            grounder_batch, _ = pad_dataproto_to_divisor(grounder_batch, dp_size)
            print(f"[DualModelTrainer] Padded grounder_batch to {len(grounder_batch)} (dp_size={dp_size})")

        # ========== Step 4: Independent old_log_prob + ref_log_prob + update ==========
        # --- Planner ---
        if planner_batch is not None and len(planner_idx) > 0:
            with marked_timer("planner_old_log_prob", timing_raw, color="blue"):
                planner_batch = self._compute_old_log_prob(
                    planner_batch, self.actor_rollout_wg, metrics, "planner", local_trigger_step
                )

            if self.use_reference_policy:
                with marked_timer("planner_ref", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(planner_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(planner_batch)
                    planner_batch = planner_batch.union(ref_log_prob)

            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_planner", timing_raw, color="red"):
                    planner_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(planner_batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update({f"planner/{k}": v for k, v in actor_output_metrics.items()})

        # --- Grounder ---
        if grounder_batch is not None and len(grounder_idx) > 0 and self.grounder_actor_wg is not None:
            with marked_timer("grounder_old_log_prob", timing_raw, color="blue"):
                grounder_batch = self._compute_old_log_prob(
                    grounder_batch, self.grounder_actor_wg, metrics, "grounder", local_trigger_step
                )

            if self.use_reference_policy:
                with marked_timer("grounder_ref", timing_raw, color="olive"):
                    if self.grounder_ref_policy_wg is not None and self.grounder_ref_policy_wg != self.grounder_actor_wg:
                        ref_log_prob = self.grounder_ref_policy_wg.compute_ref_log_prob(grounder_batch)
                    else:
                        ref_log_prob = self.grounder_actor_wg.compute_ref_log_prob(grounder_batch)
                    grounder_batch = grounder_batch.union(ref_log_prob)

            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_grounder", timing_raw, color="orange"):
                    grounder_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    grounder_output = self.grounder_actor_wg.update_actor(grounder_batch)
                grounder_output_metrics = reduce_metrics(grounder_output.meta_info["metrics"])
                metrics.update({f"grounder/{k}": v for k, v in grounder_output_metrics.items()})

        return batch, reward_extra_infos_dict

    def _compute_old_log_prob(self, sub_batch, actor_wg, metrics, role_name, local_trigger_step):
        """Compute old_log_prob for a sub-batch using the specified actor worker group."""

        def _compute(batch):
            old_log_prob = actor_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {f"{role_name}/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            # Remove stale old_log_probs (set earlier for rollout correction) before union
            if "old_log_probs" in batch.batch.keys():
                batch.batch.pop("old_log_probs")
            batch = batch.union(old_log_prob)
            if "rollout_log_probs" in batch.batch.keys():
                from verl.utils.debug.metrics import calculate_debug_metrics
                metrics.update(calculate_debug_metrics(batch))
            return batch

        async_training = self.config.get("async_training", None)
        if async_training and async_training.use_rollout_log_probs:
            if local_trigger_step == 1:
                actor_wg.save_model_to_cpu(1)
                sub_batch = _compute(sub_batch)
            elif local_trigger_step is not None:
                actor_wg.save_model_to_cpu(local_trigger_step)
                actor_wg.restore_model_from_cpu(1)
                sub_batch = _compute(sub_batch)
                actor_wg.restore_model_from_cpu(local_trigger_step)
                actor_wg.clear_cpu_model(local_trigger_step)
            else:
                sub_batch.batch["old_log_probs"] = sub_batch.batch["rollout_log_probs"]
                sub_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        else:
            sub_batch = _compute(sub_batch)

        return sub_batch

    def _get_samples_from_queue(self):
        """Use WebAgent's customized sample collection."""
        print(
            f"[DualModelTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            sample, queue_len = self.message_queue_client.get_sample_sync()
            if sample is None:
                print(
                    f"[DualModelTrainer] Detected termination signal. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break
            queue_samples.append(sample)
            if len(queue_samples) % 64 == 0:
                print(
                    f"[DualModelTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[DualModelTrainer] Not enough samples collected")
            return None, None

        total_wait_time = consumer_end - consumer_start
        print(
            f"[DualModelTrainer] Collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"wait time: {total_wait_time:.2f}s, mq_len: {queue_len}"
        )

        deserialize_start = time.time()
        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        deserialize_time = time.time() - deserialize_start

        assemble_start = time.time()
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, self._balance_batch
            )
        else:
            batch = assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, None
            )
        assemble_time = time.time() - assemble_start

        print(
            f"[DualModelTrainer] Deserialize: {deserialize_time:.2f}s, "
            f"Assemble: {assemble_time:.2f}s"
        )

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/deserialize_time"] = deserialize_time
        batch.meta_info["fully_async/assemble_time"] = assemble_time
        return 0, batch

    def _save_checkpoint(self):
        """Save checkpoints for both planner and grounder."""
        ckpt_start = time.time()
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )
        print(f"[DualModelTrainer] Saving checkpoint to {local_global_step_folder}")

        # Save planner
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_actor_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous else 1
        max_critic_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous else 1

        planner_save_start = time.time()
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_keep
        )
        print(f"[DualModelTrainer] Planner checkpoint saved in {time.time() - planner_save_start:.2f}s")

        # Save grounder
        if self.grounder_actor_wg is not None and self.grounder_actor_wg != self.actor_wg:
            grounder_local_path = os.path.join(local_global_step_folder, "grounder")
            grounder_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.current_param_version}", "grounder"
                )
            )
            grounder_save_start = time.time()
            self.grounder_actor_wg.save_checkpoint(
                grounder_local_path, grounder_remote_path,
                self.current_param_version, max_ckpt_to_keep=max_actor_keep
            )
            print(f"[DualModelTrainer] Grounder checkpoint saved in {time.time() - grounder_save_start:.2f}s")

        # Save critic
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path,
                self.current_param_version, max_ckpt_to_keep=max_critic_keep,
            )

        ray.get(self.param_synchronizer.rollouter_save_checkpoint.remote(local_global_step_folder))

        # Latest checkpointed iteration tracker
        local_latest = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest, "w") as f:
            f.write(str(self.current_param_version))

        print(f"[DualModelTrainer] Total checkpoint save time: {time.time() - ckpt_start:.2f}s")

    def load_checkpoint(self):
        """Load checkpoints for both planner and grounder."""
        result = super().load_checkpoint()

        if result > 0 and self.grounder_actor_wg is not None and self.grounder_actor_wg != self.actor_wg:
            # Try to load grounder checkpoint
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
            if global_step_folder:
                grounder_path = os.path.join(global_step_folder, "grounder")
                if os.path.exists(grounder_path):
                    self.grounder_actor_wg.load_checkpoint(
                        grounder_path,
                        del_local_after_load=self.config.trainer.del_local_ckpt_after_load
                    )
                    print(f"[DualModelTrainer] Loaded grounder checkpoint from {grounder_path}")
                else:
                    print(f"[DualModelTrainer] No grounder checkpoint found at {grounder_path}, using initialized weights")
                    self.grounder_actor_wg.load_checkpoint(None)
            else:
                self.grounder_actor_wg.load_checkpoint(None)
        elif self.grounder_actor_wg is not None and self.grounder_actor_wg != self.actor_wg:
            self.grounder_actor_wg.load_checkpoint(None)

        return result

    def _collect_metrics(self, batch, epoch, metrics, timing_raw):
        """Collect metrics including per-role (planner/grounder) breakdowns for TensorBoard."""
        metrics_start = time.time()
        steps_duration = timing_raw["step"]
        self.max_steps_duration = max(self.max_steps_duration, steps_duration)

        metrics.update({
            "training/global_step": self.global_steps,
            "training/epoch": epoch,
        })

        # ---- Full-batch metrics (unchanged from parent) ----
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        # ---- Role split info ----
        roles = batch.non_tensor_batch.get('role', None)
        if roles is not None:
            planner_mask = np.array([r == 'planner' for r in roles])
            grounder_mask = np.array([r == 'grounder' for r in roles])
            planner_count = int(planner_mask.sum())
            grounder_count = int(grounder_mask.sum())
            metrics["dual_model/planner_samples"] = planner_count
            metrics["dual_model/grounder_samples"] = grounder_count
            metrics["dual_model/planner_ratio"] = planner_count / max(len(roles), 1)

            # ---- Per-role data metrics ----
            planner_indices = np.where(planner_mask)[0]
            grounder_indices = np.where(grounder_mask)[0]

            for role_name, indices in [("planner", planner_indices), ("grounder", grounder_indices)]:
                if len(indices) == 0:
                    continue
                sub_batch = batch.select_idxs(indices)
                self._collect_role_data_metrics(sub_batch, role_name, metrics)

        # ---- WebAgent custom metrics (full batch + per-role) ----
        self._collect_webagent_metrics(batch, roles, metrics)

        # ---- Timing summary for this step ----
        metrics_overhead = time.time() - metrics_start
        metrics["timing_s/metrics_overhead"] = metrics_overhead
        # Log all timing keys for easy inspection
        for k, v in timing_raw.items():
            metrics[f"timing_s/{k}"] = v
        # Log deserialization / assemble from meta_info if available
        for extra_key in ["fully_async/deserialize_time", "fully_async/assemble_time"]:
            if extra_key in batch.meta_info:
                metrics[f"timing_s/{extra_key.replace('/', '_')}"] = batch.meta_info[extra_key]

    def _collect_role_data_metrics(self, sub_batch, role_name, metrics):
        """Compute per-role score/reward/advantage/response_length metrics."""
        try:
            # Score & reward
            sequence_score = sub_batch.batch["token_level_scores"].sum(-1)
            sequence_reward = sub_batch.batch["token_level_rewards"].sum(-1)

            max_response_length = sub_batch.batch["responses"].shape[-1]
            response_mask = sub_batch.batch["attention_mask"][:, -max_response_length:]
            response_length = response_mask.sum(-1).float()

            # Filter out aborted samples (response_length == 0)
            non_aborted = (response_length > 0)
            if non_aborted.any():
                na_score = sequence_score[non_aborted]
                na_reward = sequence_reward[non_aborted]
                na_resp_len = response_length[non_aborted]

                metrics[f"{role_name}/score/mean"] = torch.mean(na_score).item()
                metrics[f"{role_name}/score/max"] = torch.max(na_score).item()
                metrics[f"{role_name}/score/min"] = torch.min(na_score).item()

                metrics[f"{role_name}/rewards/mean"] = torch.mean(na_reward).item()
                metrics[f"{role_name}/rewards/max"] = torch.max(na_reward).item()
                metrics[f"{role_name}/rewards/min"] = torch.min(na_reward).item()

                metrics[f"{role_name}/response_length/mean"] = torch.mean(na_resp_len).item()
                metrics[f"{role_name}/response_length/max"] = torch.max(na_resp_len).item()
                metrics[f"{role_name}/response_length/min"] = torch.min(na_resp_len).item()
                metrics[f"{role_name}/response_length/clip_ratio"] = (
                    torch.mean(torch.eq(na_resp_len, max_response_length).float()).item()
                )

            metrics[f"{role_name}/aborted_ratio"] = torch.mean((~non_aborted).float()).item()

            # Advantage & returns (computed on full sub-batch, masked by response tokens)
            if "advantages" in sub_batch.batch:
                response_mask_bool = sub_batch.batch["response_mask"].bool()
                advantages = sub_batch.batch["advantages"]
                returns = sub_batch.batch["returns"]
                valid_adv = torch.masked_select(advantages, response_mask_bool)
                valid_ret = torch.masked_select(returns, response_mask_bool)

                if valid_adv.numel() > 0:
                    metrics[f"{role_name}/advantages/mean"] = torch.mean(valid_adv).item()
                    metrics[f"{role_name}/advantages/max"] = torch.max(valid_adv).item()
                    metrics[f"{role_name}/advantages/min"] = torch.min(valid_adv).item()
                    metrics[f"{role_name}/returns/mean"] = torch.mean(valid_ret).item()

            # Prompt length
            prompt_mask = sub_batch.batch["attention_mask"][:, :-max_response_length]
            prompt_length = prompt_mask.sum(-1).float()
            metrics[f"{role_name}/prompt_length/mean"] = torch.mean(prompt_length).item()
            metrics[f"{role_name}/prompt_length/max"] = torch.max(prompt_length).item()
            metrics[f"{role_name}/prompt_length/min"] = torch.min(prompt_length).item()

        except Exception as e:
            logger.warning(f"[DualModelTrainer] Failed to compute {role_name} metrics: {e}")

    def _collect_webagent_metrics(self, batch, roles, metrics):
        """Collect WebAgent custom metrics, both full-batch and per-role."""
        for key in ['final_reward', 'format_score', 'answer_score']:
            if key not in batch.non_tensor_batch:
                continue
            values = batch.non_tensor_batch[key]

            # Full-batch
            valid_values = self._extract_valid_floats(values)
            if valid_values:
                metrics[f"webagent/{key}/mean"] = np.mean(valid_values)
                metrics[f"webagent/{key}/max"] = np.max(valid_values)
                metrics[f"webagent/{key}/min"] = np.min(valid_values)

            # Per-role
            if roles is not None:
                for role_name, role_label in [("planner", "planner"), ("grounder", "grounder")]:
                    role_mask = np.array([r == role_label for r in roles])
                    if not role_mask.any():
                        continue
                    role_values = self._extract_valid_floats(values[role_mask])
                    if role_values:
                        metrics[f"webagent/{role_name}/{key}/mean"] = np.mean(role_values)
                        metrics[f"webagent/{role_name}/{key}/max"] = np.max(role_values)
                        metrics[f"webagent/{role_name}/{key}/min"] = np.min(role_values)

    @staticmethod
    def _extract_valid_floats(values):
        """Extract valid float values from a numpy array, skipping None/invalid."""
        result = []
        for v in values:
            if v is not None:
                try:
                    result.append(float(v))
                except (ValueError, TypeError):
                    pass
        return result
