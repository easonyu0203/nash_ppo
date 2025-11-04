from typing import Any, Dict
from functools import cached_property
import numpy as np
import gymnasium as gym
from gymnasium import Space
from gymnasium.spaces import Dict as DictSpace, Discrete, MultiDiscrete, Box

from envs.mytypes import BaseEnv, TimeStep, Action


class GymnasiumWrapper(BaseEnv):
    """
    Wrapper to adapt Gymnasium environments to BaseEnv interface.
    Treats single-agent Gymnasium environments as multi-agent with num_agents=1.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: A Gymnasium environment instance

        Raises:
            ValueError: If environment has unsupported observation or action space
        """
        self._env = env

        # Check for unsupported space types
        if isinstance(env.observation_space, DictSpace):
            raise ValueError(
                f"Dict observation spaces are not supported. "
                f"Got: {env.observation_space}"
            )

        if isinstance(env.action_space, DictSpace):
            raise ValueError(
                f"Dict action spaces are not supported. "
                f"Got: {env.action_space}"
            )

        if not isinstance(env.action_space, (Discrete, MultiDiscrete, Box)):
            raise ValueError(
                f"Only Discrete, MultiDiscrete, and Box action spaces are supported. "
                f"Got: {type(env.action_space).__name__}"
            )

        super().__init__()

    def __repr__(self) -> str:
        return f"GymnasiumWrapper({self._env!r})"

    @cached_property
    def num_agents(self) -> int:
        """Single agent environment"""
        return 1

    @cached_property
    def observation_space(self) -> Space:
        return self._env.observation_space

    @cached_property
    def action_space(self) -> Space:
        return self._env.action_space


    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> TimeStep:
        """Reset the environment and return initial timestep."""
        obs, info = self._env.reset(seed=seed, options=options)

        # Wrap observation in array with shape (num_agents, ...)
        obs_wrapped = np.expand_dims(obs, axis=0)

        return TimeStep(
            reward=np.zeros(self.num_agents, dtype=np.float32),
            terminated=np.array([False], dtype=bool),
            truncated=np.array([False], dtype=bool),
            observation=obs_wrapped,
            info=info,
        )

    def step(self, action: Action) -> TimeStep:
        """
        Step the environment with the given action.

        Args:
            action: Action with shape (num_agents, ...) - extracts first agent's action
        """
        # Extract single agent's action
        single_action = action[0]

        obs, reward, terminated, truncated, info = self._env.step(single_action)

        # Wrap observation in array with shape (num_agents, ...)
        obs_wrapped = np.expand_dims(obs, axis=0)

        return TimeStep(
            reward=np.array([reward], dtype=np.float32),
            terminated=np.array([terminated], dtype=bool),
            truncated=np.array([truncated], dtype=bool),
            observation=obs_wrapped,
            info=info,
        )

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    @property
    def unwrapped(self) -> BaseEnv:
        """Returns the unwrapped environment."""
        return self
