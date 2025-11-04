from typing import Any, Dict, Callable
from functools import cached_property
import numpy as np
from gymnasium import Space

from envs.mytypes import BaseEnv, TimeStep, Action


class DummyVecEnv(BaseEnv):
    """
    Vectorized environment wrapper using synchronous (dummy) execution.
    Runs multiple environment instances sequentially.
    """

    def __init__(self, env_fns: list[Callable[[], BaseEnv]]):
        """
        Args:
            env_fns: List of functions that create environment instances
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        if self.num_envs == 0:
            raise ValueError("Must provide at least one environment")

        # Verify all envs have same spaces and num_agents
        base_env = self.envs[0]
        for env in self.envs[1:]:
            assert env.num_agents == base_env.num_agents, \
                "All environments must have the same number of agents"
            assert env.action_space == base_env.action_space, \
                "All environments must have the same action space"
            assert env.observation_space == base_env.observation_space, \
                "All environments must have the same observation space"

        super().__init__()

    def __repr__(self) -> str:
        return f"DummyVecEnv(num_envs={self.num_envs}, env={self.envs[0]!r})"

    @cached_property
    def num_agents(self) -> int:
        """Number of agents per environment"""
        return self.envs[0].num_agents

    @cached_property
    def observation_space(self) -> Space:
        return self.envs[0].observation_space

    @cached_property
    def action_space(self) -> Space:
        return self.envs[0].action_space

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> TimeStep:
        """
        Reset all environments.

        Returns:
            TimeStep with vectorized fields:
                - observation: (num_envs, num_agents, *obs_shape)
                - reward: (num_envs, num_agents)
                - terminated: (num_envs, num_agents)
                - truncated: (num_envs, num_agents)
        """
        timesteps = []
        for i, env in enumerate(self.envs):
            # Use different seed for each env if seed is provided
            env_seed = None if seed is None else seed + i
            timesteps.append(env.reset(seed=env_seed, options=options))

        return self._stack_timesteps(timesteps)

    def step(self, actions: Action) -> TimeStep:
        """
        Step all environments with the given actions.

        Args:
            actions: Batched actions with shape (num_envs, num_agents, *action_shape)

        Returns:
            TimeStep with vectorized fields
        """
        timesteps = []
        for i, env in enumerate(self.envs):
            # Extract action for this environment
            env_action = actions[i]
            timesteps.append(env.step(env_action))

        return self._stack_timesteps(timesteps)

    def _stack_timesteps(self, timesteps: list[TimeStep]) -> TimeStep:
        """Stack a list of TimeSteps into a single vectorized TimeStep."""
        # Stack observations
        observations = np.stack([ts.observation for ts in timesteps], axis=0)

        # Stack rewards
        rewards = np.stack([ts.reward for ts in timesteps], axis=0)

        # Stack terminated and truncated flags
        terminated = np.stack([ts.terminated for ts in timesteps], axis=0)
        truncated = np.stack([ts.truncated for ts in timesteps], axis=0)

        # Merge info dicts - add env index to keys to avoid collisions
        merged_info = {}
        for i, ts in enumerate(timesteps):
            for key, value in ts.info.items():
                merged_info[f"env_{i}_{key}"] = value

        return TimeStep(
            observation=observations,
            reward=rewards,
            terminated=terminated,
            truncated=truncated,
            info=merged_info,
        )

    def render(self) -> Any:
        """Render the first environment."""
        return self.envs[0].render()

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()

    @property
    def unwrapped(self) -> BaseEnv:
        """Returns the first unwrapped environment."""
        return self.envs[0].unwrapped
