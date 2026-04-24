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
DualModelServerManager: Routes generate requests to planner or grounder vLLM engines
based on the engine_tag parameter.

When engine_tag is not specified, defaults to the planner engine
(backward compatible with single-model usage).
"""

import heapq
import logging
import os
import random
from collections import OrderedDict
from typing import Any, Optional
from uuid import uuid4

import ray
from omegaconf import DictConfig

from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput


class LRUCache(OrderedDict):
    """Simple LRU cache based on OrderedDict (no external dependency)."""

    def __init__(self, maxsize=1024):
        super().__init__()
        self._maxsize = maxsize

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            self.popitem(last=False)

    def __contains__(self, key):
        return OrderedDict.__contains__(self, key)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DualModelServerManager:
    """
    Manages two sets of vLLM server handles: one for the planner and one for the grounder.
    
    Routes generate() calls based on engine_tag parameter:
    - engine_tag="planner" -> planner servers
    - engine_tag="grounder" -> grounder servers
    - engine_tag=None -> planner servers (default, backward compatible)
    
    Each set has its own load balancer and sticky session cache.
    """

    def __init__(
        self,
        config: DictConfig,
        planner_server_handles: list[ray.actor.ActorHandle],
        grounder_server_handles: list[ray.actor.ActorHandle],
        max_cache_size: int = 10000,
    ):
        self.config = config

        # Planner servers
        self.planner_handles = list(planner_server_handles)
        random.shuffle(self.planner_handles)
        self.planner_weighted = [[0, (hash(s), s)] for s in self.planner_handles]
        heapq.heapify(self.planner_weighted)
        self.planner_cache = LRUCache(maxsize=max_cache_size)

        # Grounder servers
        self.grounder_handles = list(grounder_server_handles)
        random.shuffle(self.grounder_handles)
        self.grounder_weighted = [[0, (hash(s), s)] for s in self.grounder_handles]
        heapq.heapify(self.grounder_weighted)
        self.grounder_cache = LRUCache(maxsize=max_cache_size)

        # Also expose combined server_handles for compatibility
        self.server_handles = self.planner_handles + self.grounder_handles

        logger.info(
            f"[DualModelServerManager] Initialized with "
            f"{len(self.planner_handles)} planner servers, "
            f"{len(self.grounder_handles)} grounder servers"
        )

    def _choose_server(self, request_id: str, engine_tag: Optional[str] = None) -> ray.actor.ActorHandle:
        """Choose a server based on engine_tag with load balancing and sticky session."""
        if engine_tag == "grounder":
            cache = self.grounder_cache
            weighted = self.grounder_weighted
        else:
            # Default to planner
            cache = self.planner_cache
            weighted = self.planner_weighted

        if request_id in cache:
            return cache[request_id]

        server = weighted[0][1][1]
        weighted[0][0] += 1
        heapq.heapreplace(weighted, weighted[0])
        cache[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        engine_tag: Optional[str] = None,
    ) -> TokenOutput:
        """Generate tokens, routing to the appropriate engine.

        Args:
            request_id: Request id for sticky session.
            prompt_ids: List of prompt token ids.
            sampling_params: Sampling parameters.
            image_data: Optional image data for VLM.
            engine_tag: "planner", "grounder", or None (defaults to planner).

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id, engine_tag)
        output = await server.generate.remote(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output
