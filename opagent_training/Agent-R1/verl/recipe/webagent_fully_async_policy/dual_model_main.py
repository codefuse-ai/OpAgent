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
Dual-Model WebAgent Fully Async Task Runner.

This module creates a task runner that initializes:
- Two vLLM rollout worker groups (planner + grounder)
- Two FSDP training worker groups (planner + grounder)
- A DualModelParameterSynchronizer
- A DualModelWebAgentFullyAsyncTrainer

When dual_model.enable=False, falls back to the standard single-model runner.
"""

import hydra
import ray

from recipe.webagent_fully_async_policy.fully_async_rollouter import WebAgentFullyAsyncRollouter
from recipe.webagent_fully_async_policy.fully_async_trainer import WebAgentFullyAsyncTrainer
from recipe.webagent_fully_async_policy.dual_model_trainer import DualModelWebAgentFullyAsyncTrainer
from recipe.webagent_fully_async_policy.dual_model_rollouter import DualModelWebAgentFullyAsyncRollouter
from recipe.fully_async_policy.fully_async_main import (
    FullyAsyncTaskRunnerBase,
    create_resource_pool_manager,
)
from recipe.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local

from omegaconf import OmegaConf
from pprint import pprint
import os
import socket


def create_grounder_rollout_resource_pool(config) -> ResourcePoolManager:
    """Create a separate resource pool for grounder rollout GPUs.
    
    Uses config.grounder_rollout.n_gpus_per_node and config.grounder_rollout.nnodes
    to allocate GPUs exclusively for the grounder vLLM engine.
    """
    grounder_rollout_config = config.get("grounder_rollout", None)
    if grounder_rollout_config is None:
        return None

    n_gpus = grounder_rollout_config.get("n_gpus_per_node", 0)
    nnodes = grounder_rollout_config.get("nnodes", config.rollout.get("nnodes", 1))

    if n_gpus <= 0:
        return None

    resource_pool_spec = {
        "grounder_rollout_pool": [n_gpus] * nnodes,
    }
    # Use a dummy role mapping since we manage the worker group manually
    mapping = {Role.Rollout: "grounder_rollout_pool"}

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


@ray.remote(num_cpus=1)
class DualModelWebAgentFullyAsyncTaskRunner(FullyAsyncTaskRunnerBase):
    """
    Dual-Model WebAgent Async Task Runner.
    
    Conditionally creates either:
    - Standard single-model pipeline (when dual_model.enable=False)
    - Dual-model pipeline with separate planner/grounder (when dual_model.enable=True)
    """

    # ---- WebAgent fallback overrides (used when dual_model.enable=False) ----
    # When the fallback path calls super()._initialize_components() → _create_rollouter()/_create_trainer(),
    # these overrides ensure the WebAgent-specific classes are created instead of the generic base ones.

    def _create_rollouter(self, config) -> None:
        """Create WebAgent rollouter (for fallback single-model mode)."""
        rollouter = WebAgentFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={Role.Rollout: self.components["role_worker_mapping"][Role.Rollout]},
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[DUAL_MODEL_MAIN] WebAgentFullyAsyncRollouter created (single-model fallback)")

    def _create_trainer(self, config) -> None:
        """Create WebAgent trainer (for fallback single-model mode)."""
        trainer = WebAgentFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=self.components["role_worker_mapping"],
            resource_pool_manager=create_resource_pool_manager(
                config, roles=[Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]
            ),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        ray.get(trainer.set_total_train_steps.remote(self.components["rollouter"].get_total_train_steps.remote()))

        self.components["trainer"] = trainer
        print("[DUAL_MODEL_MAIN] WebAgentFullyAsyncTrainer created (single-model fallback)")

    def _initialize_components(self, config) -> None:
        """Override to handle dual-model initialization."""
        dual_model_config = config.get("dual_model", {})
        self.dual_model_enabled = dual_model_config.get("enable", False)

        if not self.dual_model_enabled:
            # Fallback to standard WebAgent initialization
            # (uses _create_rollouter/_create_trainer overrides above)
            return super()._initialize_components(config)

        print("[DUAL_MODEL_MAIN] Starting dual-model initialization...")
        print(f"[DUAL_MODEL_MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.resolve(config)

        # ========== Tokenizer/Processor ==========
        print("[DUAL_MODEL_MAIN] Initializing tokenizer and processor...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        # ========== Worker Mapping ==========
        from recipe.fully_async_policy.fully_async_main import create_role_worker_mapping
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # ========== Rollouter (with dual vLLM engines) ==========
        print("[DUAL_MODEL_MAIN] Creating DualModelWebAgentFullyAsyncRollouter...")
        self._create_dual_model_rollouter(config)

        # ========== Trainer (with dual FSDP groups) ==========
        print("[DUAL_MODEL_MAIN] Creating DualModelWebAgentFullyAsyncTrainer...")
        self._create_dual_model_trainer(config)

        # ========== Sync train steps (deferred to after load_checkpoint) ==========
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"[DUAL_MODEL_MAIN] total_train_steps: {total_train_steps}")

        # ========== Message Queue ==========
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[DUAL_MODEL_MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(message_queue_client))
        ray.get(self.components["trainer"].set_message_queue_client.remote(message_queue_client))

        # ========== Parameter Synchronizer ==========
        print("[DUAL_MODEL_MAIN] Setting up DualModelParameterSynchronizer...")
        from recipe.webagent_fully_async_policy.dual_model_param_sync import DualModelParameterSynchronizer

        param_synchronizer = DualModelParameterSynchronizer.remote(
            config=config,
            trainer=self.components["trainer"],
            rollouter=self.components["rollouter"],
            mq=message_queue_client,
        )
        ray.get(self.components["trainer"].set_parameter_synchronizer.remote(param_synchronizer))

        # ========== Load Checkpoint & Initial Sync ==========
        val_before_train = config.trainer.get("val_before_train", True)
        param_version = ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        # Init progress bar with actual resume step to avoid confusion
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps, param_version))
        print(f"[DUAL_MODEL_MAIN] Progress bar initialized: {param_version}/{total_train_steps}")

        ray.get(param_synchronizer.sync_weights.remote(version=param_version, validate=val_before_train))
        ray.get(param_synchronizer.wait_last_valid.remote())

        self.components["param_synchronizer"] = param_synchronizer
        print("[DUAL_MODEL_MAIN] All dual-model components initialized successfully")

    def _create_dual_model_rollouter(self, config) -> None:
        """Create DualModelWebAgentFullyAsyncRollouter with separate planner/grounder vLLM engines.
        
        When dual_model is enabled, creates a separate grounder resource pool
        and passes it to the rollouter for creating the grounder vLLM engine.
        """
        grounder_rpm = create_grounder_rollout_resource_pool(config)

        rollouter = DualModelWebAgentFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={Role.Rollout: self.components["role_worker_mapping"][Role.Rollout]},
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
            grounder_resource_pool_manager=grounder_rpm,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[DUAL_MODEL_MAIN] DualModelWebAgentFullyAsyncRollouter created")

    def _create_dual_model_trainer(self, config) -> None:
        """Create DualModelWebAgentFullyAsyncTrainer with both planner and grounder."""
        trainer = DualModelWebAgentFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=self.components["role_worker_mapping"],
            resource_pool_manager=create_resource_pool_manager(
                config, roles=[Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]
            ),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())

        self.components["trainer"] = trainer
        print("[DUAL_MODEL_MAIN] DualModelWebAgentFullyAsyncTrainer created")

    def _create_trainer(self, config) -> None:
        """Standard single-model trainer (used when dual_model.enable=False)."""
        trainer = WebAgentFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=self.components["role_worker_mapping"],
            resource_pool_manager=create_resource_pool_manager(
                config, roles=[Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]
            ),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[DUAL_MODEL_MAIN] WebAgentFullyAsyncTrainer created")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    from time import time

    start_time = time()
    run_ppo(config, task_runner_class=DualModelWebAgentFullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
