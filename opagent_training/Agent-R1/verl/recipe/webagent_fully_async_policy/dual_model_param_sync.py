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
DualModelParameterSynchronizer: Synchronizes both planner and grounder parameters
between their respective training and rollout worker groups.

Extends ParameterSynchronizer with:
- A second NCCL collective group for grounder weights
- Synchronized weight sync for both models
"""

import logging
import time

import ray
from ray.util.collective import collective

from verl.utils.device import get_nccl_backend

logger = logging.getLogger(__name__)


@ray.remote
class DualModelParameterSynchronizer:
    """
    Dual-model parameter synchronizer.
    
    Manages two NCCL collective groups:
    - planner_group: planner actor_wg <-> planner rollout_wg
    - grounder_group: grounder actor_wg <-> grounder rollout_wg
    """

    def __init__(self, config, trainer, rollouter, mq):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter
        self.mq_client = mq

        # Planner worker groups
        self.planner_actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.planner_rollout_wg = ray.get(rollouter.get_rollout_wg.remote())

        # Grounder worker groups
        self.grounder_actor_wg = ray.get(trainer.get_grounder_actor_wg.remote())
        self.grounder_rollout_wg = ray.get(rollouter.get_grounder_rollout_wg.remote())

        # Basic attributes
        self.planner_weights_info = None
        self.grounder_weights_info = None
        self.sync_group_initialized = False
        self.planner_sync_group_name = "planner_actor_rollout"
        self.grounder_sync_group_name = "grounder_actor_rollout"
        self.wait_last_update = None
        self.wait_last_resume = None

        self.current_version = 0

        self._init_weights_info()
        self._init_sync_groups()

    def get_current_param_version(self) -> int:
        return self.current_version

    def get_weights_info(self):
        return {
            "planner": self.planner_weights_info,
            "grounder": self.grounder_weights_info,
        }

    def _init_weights_info(self):
        # Planner weights info
        self.planner_weights_info = self.planner_actor_wg.get_actor_weights_info()[0]
        self.planner_rollout_wg.set_actor_weights_info(self.planner_weights_info)

        # Grounder weights info
        self.grounder_weights_info = self.grounder_actor_wg.get_actor_weights_info()[0]
        self.grounder_rollout_wg.set_actor_weights_info(self.grounder_weights_info)

    def _init_sync_groups(self):
        logger.info("[DualModelParamSync] Initializing parameter synchronization groups...")
        init_start = time.time()

        # Check if planner and grounder share the same workers (fallback mode)
        self.shared_workers = (
            self.grounder_actor_wg is self.planner_actor_wg
            and self.grounder_rollout_wg is self.planner_rollout_wg
        )

        if self.shared_workers:
            # Shared workers: use a single group named "actor_rollout" (compatible with default)
            self.planner_sync_group_name = "actor_rollout"
            self.grounder_sync_group_name = "actor_rollout"
            workers = self.planner_actor_wg.workers + self.planner_rollout_wg.workers
            collective.create_collective_group(
                workers,
                len(workers),
                list(range(0, len(workers))),
                backend=get_nccl_backend(),
                group_name="actor_rollout",
            )
            logger.info(f"[DualModelParamSync] Shared mode: single sync group with {len(workers)} workers")
        else:
            # Separate workers: create two distinct groups
            # Planner sync group
            planner_workers = self.planner_actor_wg.workers + self.planner_rollout_wg.workers
            collective.create_collective_group(
                planner_workers,
                len(planner_workers),
                list(range(0, len(planner_workers))),
                backend=get_nccl_backend(),
                group_name=self.planner_sync_group_name,
            )
            logger.info(f"[DualModelParamSync] Planner sync group created with {len(planner_workers)} workers")

            # Grounder sync group
            grounder_workers = self.grounder_actor_wg.workers + self.grounder_rollout_wg.workers
            collective.create_collective_group(
                grounder_workers,
                len(grounder_workers),
                list(range(0, len(grounder_workers))),
                backend=get_nccl_backend(),
                group_name=self.grounder_sync_group_name,
            )
            logger.info(f"[DualModelParamSync] Grounder sync group created with {len(grounder_workers)} workers")

        self.sync_group_initialized = True
        logger.info(f"[DualModelParamSync] Sync groups initialized in {time.time() - init_start:.2f}s")

    def sync_weights(self, version, validate=False, global_steps=0):
        """Sync weights for both planner and grounder."""
        start_time = time.time()
        self.current_version = version
        logger.info(f"[DualModelParamSync] Starting dual weight sync (version {version})...")

        # Pause rollouter
        ray.get(self.rollouter.pause.remote())
        pause_time = time.time() - start_time
        logger.info(f"[DualModelParamSync] Rollout paused. Cost {pause_time:.2f}s")

        # Update MQ version
        mq_start = time.time()
        self.mq_client.update_param_version_sync(version)
        mq_time = time.time() - mq_start
        logger.info(f"[DualModelParamSync] MQ version updated. Cost {mq_time:.2f}s")

        if self.shared_workers:
            # Shared mode: single sync (same as single-model)
            sync_start = time.time()
            self.planner_actor_wg.sync_rollout_weights(self.planner_sync_group_name)
            ray.get(self.planner_rollout_wg.sync_rollout_weights(self.planner_sync_group_name))
            logger.info(f"[DualModelParamSync] Shared sync done. Cost {time.time() - sync_start:.2f}s")
        else:
            # Sync planner weights
            planner_start = time.time()
            self.planner_actor_wg.sync_rollout_weights(self.planner_sync_group_name)
            ray.get(self.planner_rollout_wg.sync_rollout_weights(self.planner_sync_group_name))
            planner_sync_time = time.time() - planner_start
            logger.info(f"[DualModelParamSync] Planner sync done. Cost {planner_sync_time:.2f}s")

            # Sync grounder weights
            grounder_start = time.time()
            self.grounder_actor_wg.sync_rollout_weights(self.grounder_sync_group_name)
            ray.get(self.grounder_rollout_wg.sync_rollout_weights(self.grounder_sync_group_name))
            grounder_sync_time = time.time() - grounder_start
            logger.info(f"[DualModelParamSync] Grounder sync done. Cost {grounder_sync_time:.2f}s")

        end_time = time.time()
        total_sync_time = end_time - start_time
        logger.info(
            f"[DualModelParamSync] Dual sync_weights complete. "
            f"pause={pause_time:.2f}s, mq={mq_time:.2f}s, total={total_sync_time:.2f}s"
        )

        # Async update rollout version & validation
        self.wait_last_update = self.rollouter.update_param_version.remote(version, validate, global_steps)
        self.wait_last_resume = self.rollouter.resume.remote(self.wait_last_update)

    def wait_last_valid(self):
        logger.info("[DualModelParamSync] Waiting last sync and validate...")
        start_time = time.time()
        if self.wait_last_update:
            ray.get(self.wait_last_update)
        if self.wait_last_resume:
            ray.get(self.wait_last_resume)
        logger.info(f"[DualModelParamSync] Wait done. Cost {time.time() - start_time:.2f}s")

    def rollouter_save_checkpoint(self, local_global_step_folder: str):
        logger.info(f"[DualModelParamSync] Triggering checkpoint save at {local_global_step_folder}")
        ckpt_start = time.time()
        result = ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        logger.info(f"[DualModelParamSync] Checkpoint save done. Cost {time.time() - ckpt_start:.2f}s")
        return result
