"""
Vectorized Unity environment manager for multiple Unity instances.

Similar to ml-agents' SubprocessEnvManager, this manages multiple Unity
subprocess wrappers and coordinates parallel step/reset operations.
"""

import multiprocessing as mp
from multiprocessing import Queue
from typing import List, Optional, Dict, Any
import logging

import numpy as np
from gymnasium import Space
from omegaconf import DictConfig

from envs.mytypes import BaseEnv, TimeStep, Action
from envs.wrappers.unity_subprocess_wrapper import (
    UnitySubprocessWrapper,
    UnitySubprocessCommand
)


logger = logging.getLogger(__name__)


class UnitySubprocessVecEnv(BaseEnv):
    """
    Manages multiple Unity instances running in parallel subprocesses.

    Each instance has its own Unity executable with num_areas training areas.
    Total parallel environments = num_instances * num_areas

    Architecture:
        Main Process (this class)
            |
            +-> UnitySubprocessWrapper 0 (worker_id=0, port=base_port+0)
            |       └─> Unity Instance 0 with num_areas
            +-> UnitySubprocessWrapper 1 (worker_id=1, port=base_port+1)
            |       └─> Unity Instance 1 with num_areas
            ...
            +-> UnitySubprocessWrapper N (worker_id=N, port=base_port+N)
                    └─> Unity Instance N with num_areas

    Communication:
        - Commands sent via individual Pipes (non-blocking)
        - Results collected via shared Queue (blocking until all ready)
    """

    def __init__(
        self,
        env_config: DictConfig,
        num_instances: int,
        num_areas: int
    ):
        """
        Initialize vectorized Unity environment manager.

        Args:
            env_config: Unity environment configuration
            num_instances: Number of Unity instances to create
            num_areas: Number of training areas per instance

        Raises:
            ValueError: If num_instances < 1
            RuntimeError: If any worker fails to initialize
        """
        if num_instances < 1:
            raise ValueError(f"num_instances must be >= 1, got {num_instances}")

        self.num_instances = num_instances
        self.num_areas = num_areas
        self._env_config = env_config
        self._closed = False

        # Create shared queue for step results
        ctx = mp.get_context('spawn')
        self._step_queue: Queue = ctx.Queue()

        # Create worker wrappers
        self._workers: List[UnitySubprocessWrapper] = []

        logger.info(
            f"Creating {num_instances} Unity instances, "
            f"each with {num_areas} areas "
            f"(total {num_instances * num_areas} parallel environments)"
        )

        for worker_id in range(num_instances):
            try:
                worker = UnitySubprocessWrapper(
                    env_config=env_config,
                    num_areas=num_areas,
                    worker_id=worker_id,
                    step_queue=self._step_queue,
                    ctx=ctx
                )
                self._workers.append(worker)
                logger.info(f"Worker {worker_id} initialized successfully")
            except Exception as e:
                # Cleanup already created workers
                for w in self._workers:
                    w.close()
                raise RuntimeError(
                    f"Failed to initialize worker {worker_id}/{num_instances}: {e}"
                ) from e

        # Verify all workers have same spaces and num_agents
        self._validate_workers()

        # Store from first worker (all should be identical)
        first_worker = self._workers[0]
        self._observation_space = first_worker.observation_space
        self._action_space = first_worker.action_space
        self._num_agents = first_worker.num_agents

        logger.info(
            f"UnitySubprocessVecEnv initialized: "
            f"{num_instances} instances × {num_areas} areas × {self._num_agents} agents = "
            f"{num_instances * num_areas * self._num_agents} total agents"
        )

    def _validate_workers(self):
        """Ensure all workers have identical spaces and num_agents."""
        if not self._workers:
            raise RuntimeError("No workers created")

        first_worker = self._workers[0]
        for i, worker in enumerate(self._workers[1:], start=1):
            if worker.num_agents != first_worker.num_agents:
                raise RuntimeError(
                    f"Worker {i} has {worker.num_agents} agents, "
                    f"but worker 0 has {first_worker.num_agents} agents"
                )
            if worker.action_space != first_worker.action_space:
                raise RuntimeError(
                    f"Worker {i} has different action space than worker 0"
                )
            if worker.observation_space != first_worker.observation_space:
                raise RuntimeError(
                    f"Worker {i} has different observation space than worker 0"
                )

    def _queue_resets(self, seed: Optional[int] = None) -> None:
        """Send reset commands to all workers (non-blocking)."""
        for worker in self._workers:
            worker_seed = None if seed is None else seed + worker._worker_id
            worker.send_command(UnitySubprocessCommand.RESET, {"seed": worker_seed})

    def _queue_steps(self, actions: Action) -> None:
        """
        Send step commands to all workers (non-blocking).

        Args:
            actions: Shape (num_instances, num_areas, num_agents, ...)
        """
        for i, worker in enumerate(self._workers):
            # Extract actions for this worker's instance
            worker_actions = actions[i]  # Shape: (num_areas, num_agents, ...)
            worker.send_command(UnitySubprocessCommand.STEP, {"actions": worker_actions})

    def _collect_results(self, num_expected: int) -> List[Dict[str, Any]]:
        """
        Collect results from the step queue.

        Args:
            num_expected: Number of results to wait for

        Returns:
            List of results sorted by worker_id

        Raises:
            RuntimeError: If any worker reports failure
        """
        results = []
        worker_ids_seen = set()

        while len(results) < num_expected:
            result = self._step_queue.get()  # Blocking

            # Check for errors
            if not result.get("success", False):
                worker_id = result.get("worker_id", "unknown")
                error_msg = result.get("error", "Unknown error")
                traceback_msg = result.get("traceback", "")
                raise RuntimeError(
                    f"Worker {worker_id} failed:\n{error_msg}\n{traceback_msg}"
                )

            # Avoid duplicate results from same worker
            worker_id = result["worker_id"]
            if worker_id in worker_ids_seen:
                logger.warning(f"Duplicate result from worker {worker_id}, ignoring")
                continue

            worker_ids_seen.add(worker_id)
            results.append(result)

        # Sort by worker_id to maintain deterministic order
        results.sort(key=lambda x: x["worker_id"])
        return results

    def _stack_timesteps(self, timesteps: List[TimeStep]) -> TimeStep:
        """
        Stack timesteps from all workers into a single vectorized timestep.

        Args:
            timesteps: List of TimeStep from each worker
                Each has shape (num_areas, num_agents, ...)

        Returns:
            TimeStep with shape (num_instances * num_areas, num_agents, ...)

        Handles both single observation (array) and multiple observations (dict of arrays).
        """
        # Stack along first dimension: (num_instances, num_areas, num_agents, ...)
        # Check if observations are dicts
        first_obs = timesteps[0].observation
        if isinstance(first_obs, dict):
            # Multiple observations - stack each key separately
            observations = {
                key: np.stack([ts.observation[key] for ts in timesteps], axis=0)
                for key in first_obs.keys()
            }
        else:
            # Single observation - stack directly
            observations = np.stack([ts.observation for ts in timesteps], axis=0)

        rewards = np.stack([ts.reward for ts in timesteps], axis=0)
        terminated = np.stack([ts.terminated for ts in timesteps], axis=0)
        truncated = np.stack([ts.truncated for ts in timesteps], axis=0)
        action_masks = np.stack([ts.action_mask for ts in timesteps], axis=0)

        # Reshape to (num_instances * num_areas, num_agents, ...)
        # Merge first two dimensions
        batch_size = self.num_instances * self.num_areas

        if isinstance(observations, dict):
            # Multiple observations - reshape each key separately
            observations = {
                key: val.reshape(batch_size, self._num_agents, *val.shape[3:])
                for key, val in observations.items()
            }
        else:
            # Single observation - reshape directly
            observations = observations.reshape(batch_size, self._num_agents, *observations.shape[3:])

        rewards = rewards.reshape(batch_size, self._num_agents)
        terminated = terminated.reshape(batch_size, self._num_agents)
        truncated = truncated.reshape(batch_size, self._num_agents)
        action_masks = action_masks.reshape(batch_size, self._num_agents, *action_masks.shape[3:])

        # Merge info dicts with worker_id prefix
        merged_info = {}
        for i, ts in enumerate(timesteps):
            for key, value in ts.info.items():
                merged_info[f"worker_{i}_{key}"] = value

        return TimeStep(
            observation=observations,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            action_mask=action_masks,
            info=merged_info,
        )

    def reset(self, seed: Optional[int] = None) -> TimeStep:
        """
        Reset all Unity instances in parallel.

        Args:
            seed: Base seed (each worker gets seed + worker_id)

        Returns:
            TimeStep with shape (num_instances * num_areas, num_agents, ...)
        """
        if self._closed:
            raise RuntimeError("Environment is closed")

        # Send reset commands to all workers
        self._queue_resets(seed)

        # Collect results
        results = self._collect_results(self.num_instances)

        # Extract timesteps and stack
        timesteps = [r["timestep"] for r in results]
        return self._stack_timesteps(timesteps)

    def step(self, actions: Action) -> TimeStep:
        """
        Step all Unity instances in parallel.

        Args:
            actions: Shape (num_instances * num_areas, num_agents, ...)
                Will be reshaped to (num_instances, num_areas, num_agents, ...)

        Returns:
            TimeStep with shape (num_instances * num_areas, num_agents, ...)
        """
        if self._closed:
            raise RuntimeError("Environment is closed")

        # Reshape actions from (batch, agents, ...) to (num_instances, num_areas, agents, ...)
        actions_reshaped = actions.reshape(
            self.num_instances,
            self.num_areas,
            self._num_agents,
            *actions.shape[2:]
        )

        # Send step commands to all workers
        self._queue_steps(actions_reshaped)

        # Collect results
        results = self._collect_results(self.num_instances)

        # Extract timesteps and stack
        timesteps = [r["timestep"] for r in results]
        return self._stack_timesteps(timesteps)

    @property
    def observation_space(self) -> Space:
        """Get observation space."""
        return self._observation_space

    @property
    def action_space(self) -> Space:
        """Get action space."""
        return self._action_space

    @property
    def num_agents(self) -> int:
        """Get number of agents per area."""
        return self._num_agents

    def render(self):
        """Render not supported in multi-instance mode."""
        logger.warning("Render not supported in UnitySubprocessVecEnv")
        return None

    def close(self) -> None:
        """Close all worker environments."""
        if self._closed:
            return

        self._closed = True

        logger.info(f"Closing {len(self._workers)} Unity workers")
        for i, worker in enumerate(self._workers):
            try:
                worker.close()
                logger.debug(f"Worker {i} closed successfully")
            except Exception as e:
                logger.error(f"Error closing worker {i}: {e}")

        # Clear workers list
        self._workers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Check if _closed exists (initialization may have failed)
        if hasattr(self, '_closed') and not self._closed:
            self.close()

    def __repr__(self) -> str:
        return (
            f"UnitySubprocessVecEnv("
            f"num_instances={self.num_instances}, "
            f"num_areas={self.num_areas}, "
            f"num_agents={self._num_agents}, "
            f"total_envs={self.num_instances * self.num_areas})"
        )
