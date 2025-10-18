from typing import Any, Dict
from functools import cached_property
import numpy as np
import gymnasium as gym
from gymnasium import Space
from gymnasium.spaces import Dict as DictSpace, Discrete, MultiDiscrete

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

        if not isinstance(env.action_space, (Discrete, MultiDiscrete)):
            raise ValueError(
                f"Only Discrete and MultiDiscrete action spaces are supported. "
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

    @cached_property
    def _default_action_mask(self) -> Action:
        """
        Generate default action mask (all actions available).
        Returns array with shape (num_agents, num_actions) for Discrete spaces,
        or (num_agents, num_dims) for MultiDiscrete spaces.
        """
        if isinstance(self.action_space, Discrete):
            # All discrete actions are available
            return np.ones((self.num_agents, self.action_space.n), dtype=np.int8)
        elif isinstance(self.action_space, MultiDiscrete):
            # All actions available for each dimension
            return np.ones((self.num_agents, *self.action_space.nvec.shape), dtype=np.int8)
        else:
            # Should never reach here due to __init__ checks
            raise RuntimeError("Unsupported action space type")

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> TimeStep:
        """Reset the environment and return initial timestep."""
        obs, info = self._env.reset(seed=seed, options=options)

        # Wrap observation in array with shape (num_agents, ...)
        obs_wrapped = np.expand_dims(obs, axis=0)

        return TimeStep(
            reward=np.zeros(self.num_agents, dtype=np.float32),
            done=np.array(False),
            observation=obs_wrapped,
            action_mask=self._default_action_mask,
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
        done = terminated or truncated

        # Wrap observation in array with shape (num_agents, ...)
        obs_wrapped = np.expand_dims(obs, axis=0)

        return TimeStep(
            reward=np.array([reward], dtype=np.float32),
            done=np.array(done),
            observation=obs_wrapped,
            action_mask=self._default_action_mask,
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
