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


import hydra
import ray

from recipe.webagent_fully_async_policy.fully_async_rollouter import WebAgentFullyAsyncRollouter
from recipe.webagent_fully_async_policy.fully_async_trainer import WebAgentFullyAsyncTrainer
from recipe.fully_async_policy.fully_async_main import FullyAsyncTaskRunnerBase, create_resource_pool_manager
from verl.trainer.ppo.utils import Role


@ray.remote(num_cpus=1)
class WebAgentFullyAsyncTaskRunner(FullyAsyncTaskRunnerBase):
    """
    WebAgent specific Async Task Runner.
    Inherits from FullyAsyncTaskRunner and overrides the rollouter and trainer creation
    to use WebAgent specific implementations.
    """

    def _create_rollouter(self, config) -> None:
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
        print("[ASYNC MAIN] WebAgentFullyAsyncRollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        # Use WebAgentFullyAsyncTrainer instead of FullyAsyncTrainer
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
        print("[ASYNC MAIN] WebAgentFullyAsyncTrainer created and initialized successfully")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    from time import time

    start_time = time()
    run_ppo(config, task_runner_class=WebAgentFullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
